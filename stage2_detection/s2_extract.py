import os, re, json, time, glob, torch, numpy as np
from collections import Counter

FULL_DIR = f"{RES_DIR}/stage1v2_full"
OUT2 = f"{RES_DIR}/stage2_lens"; os.makedirs(OUT2, exist_ok=True)
META = f"{OUT2}/meta.jsonl"                     # one line per agent-day; appended as we go (safe to resume)
LAYERS = [46, 52]                              # concept layers from the Italy check; 39 was noisy
TOPK = 20
FAM = {"agreement":   ["agree", "acuerdo", "pact", "deal", "promis", "vereinbar", "abkommen", "соглаш", "договор", "协议", "约定", "承诺", "合意", "約束"],
       "cooperation": ["cooperat", "collab", "coordinat", "partner", "合作", "協力"],
       "collusion":   ["collu", "cartel", "conspir", "scheme", "kartell", "сговор", "勾结", "illegal"],
       "secrecy":     ["secret", "secreto", "segredo", "geheim", "hide", "hidden", "conceal", "covert", "秘密", "тайн", "секрет"],
       "trust":       ["trust", "betray", "cheat"]}
EN = {"agreement":   ["agree", "deal", "pact", "promis", "arrang", "understanding", "settled", "contract", "accord", "commit"],
      "cooperation": ["cooperat", "collab", "coordinat", "partner", "together", "jointly", "team"],
      "collusion":   ["collu", "cartel", "conspir", "scheme"],
      "secrecy":     ["secret", "hide", "hidden", "conceal", "covert", "privat"],
      "trust":       ["trust", "betray", "cheat", "loyal"]}
STEMS = [s for ss in FAM.values() for s in ss]

def has_word(words, low):
    return any(re.search(r"(?<![^\W\d_])" + re.escape(w), low) for w in words)
def fams_in(text):
    low = text.lower(); return sorted(f for f in FAM if has_word(EN[f], low))
def char_to_tok(offsets, c):
    for i, (s, e) in enumerate(offsets):
        if s <= c < e: return i
    return None

KEY = re.compile(r'"(insights|plans|message_plan|message_to_farm_[AB])"\s*:\s*("(?:[^"\\]|\\.)*"|null)', re.S)
def segments(reply):
    """char spans (start, end) of each JSON value in the reply, in writing order; 'decision' = fields + sell"""
    seg = {}
    for m in KEY.finditer(reply):
        name = "message" if m.group(1).startswith("message_to") else m.group(1)
        seg[name] = (m.start(2), m.end(2))
    f = reply.find('"fields"')
    if f >= 0: seg["decision"] = (f, len(reply))
    return seg

V = hf.get_output_embeddings().weight.shape[0]
vocab = [s.strip().lower() for s in tok.batch_decode([[i] for i in range(V)])]
ID2STEM = {i: s for i, t in enumerate(vocab) for s in STEMS if t.startswith(s) and len(t) >= 2}
print("vocab", V, "| monitored tokens", len(ID2STEM))

done = set()
if os.path.exists(META):
    done = {json.loads(l)["rid"] for l in open(META)}
games = [json.load(open(p)) for p in sorted(glob.glob(f"{FULL_DIR}/*.json"))]
todo = [(g, d, me) for g in games for d in g["days"] for me in "AB"]
print("rows total", len(todo), "| already done", len(done))
t, n_new, mis = time.time(), 0, 0
with open(META, "a") as fout:
    for g, d, me in todo:
        rid = f"{g['arm']}_s{g['seed']:02d}_d{d['day']:02d}_{me}"
        if rid in done: continue
        a = d["agents"][me]; dec = a["decision"]
        reply = a["attempts"][-1]["raw"]
        prompt_text = tok.apply_chat_template([{"role": "user", "content": a["prompt"]}], tokenize=False, add_generation_prompt=True)
        full = prompt_text + reply; P = len(prompt_text)
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True); off = enc["offset_mapping"]
        seg = segments(reply)
        pe = char_to_tok(off, P - 1)
        seg_tok = {}
        for name, (s, e) in seg.items():
            ks = [i for i, (a0, b0) in enumerate(off) if P + s <= a0 < P + e]
            if ks: seg_tok[name] = ks
        positions = [pe] + sorted({k for ks in seg_tok.values() for k in ks})
        lens_text = full[len(tok.bos_token):] if full.startswith(tok.bos_token) else full
        out, _, ids_l = lens.apply(lens_model, lens_text, layers=LAYERS, positions=positions, max_seq_len=8192)
        aligned = torch.as_tensor(ids_l).flatten().tolist() == enc["input_ids"]; mis += (not aligned)
        idx = {p: j for j, p in enumerate(positions)}
        hits = {}
        for L in LAYERS:
            top = torch.as_tensor(out[L]).float().topk(TOPK, dim=-1).indices.cpu().numpy()
            hits[L] = {name: dict(Counter(ID2STEM[i] for k in ks for i in top[idx[k]] if i in ID2STEM)) for name, ks in seg_tok.items()}
            hits[L]["prompt_end"] = dict(Counter(ID2STEM[i] for i in top[0] if i in ID2STEM))
        seg_text = {name: reply[s:e] for name, (s, e) in seg.items()}
        rec = {"rid": rid, "arm": g["arm"], "seed": g["seed"], "day": d["day"], "farm": me,
               "channel_day": g["channel_day"], "open": d["channel_open"], "valid": dec is not None,
               "sent": bool(dec and dec.get(f"message_to_farm_{'B' if me == 'A' else 'A'}")),
               "offered": a["offered"], "produced": a["produced"], "caps": {c: d["market"][c]["cap"] for c in ("WHEAT", "MELON")},
               "n_tok": {k: len(v) for k, v in seg_tok.items()}, "hits": hits, "aligned": aligned,
               "fams_prompt": fams_in(a["prompt"]), "fams_seg": {k: fams_in(v) for k, v in seg_text.items()},
               "seg_text": seg_text}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        n_new += 1
        if n_new % 100 == 0:
            el = time.time() - t
            print(f"{n_new} new rows | {el/60:.1f} min | ~{el/n_new*(len(todo)-len(done)-n_new)/60:.0f} min left | misaligned {mis}")
print(f"done: {n_new} new rows in {(time.time()-t)/60:.1f} min | misaligned {mis}")
