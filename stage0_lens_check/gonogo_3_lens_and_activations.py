import torch, numpy as np, json, re, time
from transformers import AutoTokenizer
import jlens

MODEL = "google/gemma-3-12b-it"
LAYERS = [24, 30, 36]
POS = ["prompt_end", "plans_end", "pre_fields"]          # single-token positions (lens + probe)
rows = [json.loads(l) for l in open(f"{TEST_DIR}/test_replies.jsonl")]

# load once; reuse if still in memory from Part B
if "hf" not in globals():
    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        from transformers import Gemma3ForConditionalGeneration as M
    except ImportError:
        from transformers import AutoModelForCausalLM as M
    hf = M.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
if "lens" not in globals():
    lens_model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained("neuronpedia/jacobian-lens",
        filename="gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt")

def char_to_tok(offsets, c):
    for i, (s, e) in enumerate(offsets):
        if s <= c < e:
            return i
    raise ValueError(f"no token covers char {c}")

def find_positions(prompt_text, reply, offsets):
    P = len(prompt_text)
    mp = re.search(r'"plans"\s*:\s*"(.*?)"\s*,\s*"fields"', reply, re.S)
    mf = re.search(r'"WHEAT"\s*:\s*', reply)
    if mp is None or mf is None or mp.end(1) == mp.start(1):
        return None
    pos = {"prompt_end": char_to_tok(offsets, P - 1),
           "plans_start": char_to_tok(offsets, P + mp.start(1)),
           "plans_end": char_to_tok(offsets, P + mp.end(1) - 1),
           "pre_fields": char_to_tok(offsets, P + mf.end() - 1)}
    return pos

acts = {f"{p}_L{L}": [] for p in POS + ["plans_mean"] for L in LAYERS}
meta, skipped, misaligned = [], 0, 0
t = time.time()
for i, r in enumerate(rows):
    prompt_text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                          tokenize=False, add_generation_prompt=True)
    full = prompt_text + r["reply"]
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    pos = find_positions(prompt_text, r["reply"], enc["offset_mapping"])
    if pos is None:
        skipped += 1; continue
    ids = torch.tensor([enc["input_ids"]], device="cuda")

    # (a) raw activations for the probe
    with torch.no_grad():
        hs = hf(input_ids=ids, output_hidden_states=True).hidden_states   # hs[L] = residual after L blocks
    for L in LAYERS:
        h = hs[L][0].float()
        for p in POS:
            acts[f"{p}_L{L}"].append(h[pos[p]].cpu().numpy())
        acts[f"plans_mean_L{L}"].append(h[pos["plans_start"]:pos["plans_end"] + 1].mean(0).cpu().numpy())
    del hs

    # (b) J-lens top-50 words at the same positions
    lens_text = full[len(tok.bos_token):] if full.startswith(tok.bos_token) else full
    out, _, ids_l = lens.apply(lens_model, lens_text, layers=LAYERS,
                               positions=[pos[p] for p in POS], max_seq_len=4096)
    aligned = list(torch.as_tensor(ids_l).flatten().tolist()) == enc["input_ids"]
    misaligned += (not aligned)
    tops = {f"{p}_L{L}": [tok.decode([j]) for j in torch.as_tensor(out[L][k]).float().topk(50).indices.tolist()]
            for L in LAYERS for k, p in enumerate(POS)}

    d = r["decision"] or {}
    meta.append({"id": r["id"], "cond": r["cond"], "wording": r["wording"], "day": r["day"],
                 "wheat": d.get("fields", {}).get("WHEAT"), "melon": d.get("fields", {}).get("MELON"),
                 "insights": d.get("insights", ""), "plans": d.get("plans", ""),
                 "aligned": aligned, "tops": tops})
    if i % 50 == 0:
        print(f"{i}/{len(rows)}  {time.time()-t:.0f}s")

np.savez_compressed(f"{TEST_DIR}/acts.npz", **{k: np.stack(v) for k, v in acts.items()})
json.dump(meta, open(f"{TEST_DIR}/meta.json", "w"))
print(f"done in {time.time()-t:.0f}s | kept {len(meta)} | skipped {skipped} | lens misaligned {misaligned}")
print("shape per key:", np.stack(acts["plans_mean_L30"]).shape)
ex = meta[0]
print("\nexample:", ex["id"], "| plans:", ex["plans"][:120])
for k in ["plans_end_L24", "plans_end_L30", "pre_fields_L30"]:
    print(k, ex["tops"][k][:15])
