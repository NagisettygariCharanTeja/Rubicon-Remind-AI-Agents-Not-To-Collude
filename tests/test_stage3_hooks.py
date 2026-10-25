import sys, json, torch, random
sys.path.insert(0, "/home/claude/day1"); sys.path.insert(0, "/home/claude/stage1v2"); sys.path.insert(0, "/home/claude/stage3")
import stage3 as s3
from channel_game_v2 import CHAN
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel
chars = sorted(set(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789{}[]\":,._-'!?()/\n<>*") | set("агрсвоопкнд协议合作"))
vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2}; [vocab.setdefault(c, len(vocab)) for c in chars]
for w in [" cooperate", " agree", " collude", " acuerdo"]: vocab[w] = len(vocab)
tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Split("", "isolated"); tk.decoder = decoders.Fuse()
tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", unk_token="<unk>", bos_token="<bos>")
tok.chat_template = "<bos>{% for m in messages %}U:{{ m['content'] }}\nM:{% endfor %}"
torch.manual_seed(0)
model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=8192, n_embd=64, n_layer=6, n_head=4)).eval()
layers = [3, 4, 5]; mods = {i: model.transformer.h[i] for i in range(6)}
# 1) hooks: projection removes the concept subspace at the hooked layer outputs, only on new positions
Qc = s3.random_dirs(64, layers, k=4, seed=1, device="cpu"); Qr = s3.random_dirs(64, layers, k=4, seed=2, device="cpu")
texts = ["U:hello there\nM:", "U:a much longer prompt with more text\nM:"]
o0, e0 = s3.generate_rows(model, tok, texts, 5, 12, hooks=s3.RowHooks(mods, Qc, None))
o1, e1 = s3.generate_rows(model, tok, texts, 5, 12, hooks=s3.RowHooks(mods, Qc, Qc))
o2, e2 = s3.generate_rows(model, tok, texts, 5, 12, hooks=s3.RowHooks(mods, Qc, Qr))
o3, _ = s3.generate_rows(model, tok, texts, 5, 12)
print("no-hook == measure-only (hooks must not change output when nothing is removed):", o0 == o3)
print("steer changes output:", o0 != o1, "| random changes output:", o0 != o2)
print("energy none/steer/random:", [round(x, 3) for x in e0], [round(x, 3) for x in e1], [round(x, 3) for x in e2])
# verify removal: residual after a steered layer has ~0 component in Qc
cap = {}
h = model.transformer.h[5].register_forward_hook(lambda m, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o)[:, -1:, :].clone()))
with s3.RowHooks(mods, Qc, Qc):
    model(**tok(texts, return_tensors="pt", padding=True, add_special_tokens=False))
h.remove()
print("component left in concept subspace after removal (should be ~0):", float((cap["x"].float() @ Qc[5]).norm()))
# 2) pipeline with a fake agent and fake lens
class FakeLens:
    jacobians = {L: torch.eye(64) for L in layers}
    def apply(self, lm, text, layers, positions, max_seq_len):
        return {L: torch.randn(len(positions), len(vocab)) for L in layers}, None, None
def fake_base(model_, tok_, texts, seed, max_new):
    r = random.Random(seed)
    outs = [json.dumps({"insights": "i", "plans": "p", "message_plan": r.choice(["tell B to sell less", "say nothing"]),
                        "message_to_farm_" + ("B" if "Farm A" in t else "A"): None, "fields": {"WHEAT": 2, "MELON": 2}, "sell": {"WHEAT": 4, "MELON": 2}})
            for t in texts]
    return outs, [None] * len(texts)
states = [{"sid": f"s{i}", "arm": "chan_comply", "seed": 10, "day": 5, "farm": "A", "other": "B",
           "prompt": "You manage Farm A ... Reply with ONE JSON object and nothing else", "caps": {"WHEAT": 8, "MELON": 4}} for i in range(5)]
id2stem = {vocab[" cooperate"]: "cooperat", vocab[" agree"]: "agree"}
chan = {o: CHAN[o] for o in "AB"}
s3.run(model, tok, FakeLens(), None, id2stem, states, chan, "/tmp/claude-0/s3test", Qc, Qr, mods, n_samples=2, bs=4,
       max_full=20, max_cont=15, base_gen=fake_base, log=print)
for arm in ["base", "none", "steer", "random", "remind"]:
    rows = [json.loads(l) for l in open(f"/tmp/claude-0/s3test/{arm}.jsonl")]
    print(arm, len(rows), rows[0].get("score"), rows[0].get("trigger"), rows[0]["outcome"].get("valid"), str(rows[0].get("continuation", rows[0].get("prefix")))[:60])
