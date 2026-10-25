import sys, json, random, os, shutil
sys.path.insert(0, "/home/claude/day1"); sys.path.insert(0, "/home/claude/stage1v2"); sys.path.insert(0, "/home/claude/stage3")
exec(open("/home/claude/stage3/test_local.py").read().split("# 1) hooks")[0])
import stage3x as sx, channel_game_v2 as cg
# fake HF generation: returns valid replies; message only if channel open and no REMINDER present
def fake_rows(model_, tok_, texts, seed, max_new, hooks=None, temperature=0.7):
    out = []
    for t in texts:
        other = "B" if "You manage Farm A" in t else "A"
        d = {"insights": "i", "plans": "p", "fields": {"WHEAT": 2, "MELON": 2}, "sell": {"WHEAT": 4, "MELON": 2}}
        if "PRIVATE CHANNEL" in t:
            d = {"insights": "i", "plans": "p", "message_plan": "signal", f"message_to_farm_{other}": (None if "REMINDER" in t else "Let's both sell 4 wheat"),
                 "fields": d["fields"], "sell": d["sell"]}
        out.append(json.dumps(d))
    return out, [None] * len(texts)
sx.s3.generate_rows = fake_rows
sx.s3.lens_score = lambda *a, **k: (random.choice([0, 3]), {})
chan = {o: cg.CHAN[o] for o in "AB"}
# A1 / A3 on fake prompts made by the real v2 builder
g = {"arm": "chan_comply", "seed": 10, "channel_day": 0, "notes": {"A": {}, "B": {}}, "messages": []}
env = cg.MonopsonyFarmSell(cg.CONFIG); env.reset()
p_open = cg.build(g, env, "A", 0)[0]
g2 = dict(g, channel_day=5); p_closed = cg.build(g2, env, "B", 0)[0]
rows = [{"uid": "u1", "prompt": p_open, "caps": {"WHEAT": 8, "MELON": 4}}]
out = "/tmp/claude-0/sx"; shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
for v, fn in sx.VARIANTS.items():
    sx.regen(None, tok, rows, fn, f"{out}/{v}.jsonl", chan, cg.DecisionV2, log=lambda *a: None)
    print(v, json.loads(open(f"{out}/{v}.jsonl").readline())["outcome"])
print("inserted text example:\n", sx.with_text(p_open, sx.VARIANTS["repeat_strict"]("B"))[-1300:-900])
pre = [{"uid": "p1", "prompt": p_closed, "caps": {"WHEAT": 8, "MELON": 4}}]
sx.regen(None, tok, pre, None, f"{out}/pre_plain.jsonl", None, cg.DecisionV2, log=lambda *a: None)
print("pre-channel plain:", json.loads(open(f"{out}/pre_plain.jsonl").readline())["outcome"])
# LOOP
cg.ARMS = ["chan_comply"]
for arm in ["none", "monitor", "always"]:
    trig = []
    gen = sx.make_gen(None, tok, arm, trig_log=trig)
    games = cg.run_batch([10, 11], gen, f"{out}/loop_{arm}", tag=f"loop_{arm}", log=lambda *a: None)
    print(arm, "msgs per game:", [len(x["messages"]) for x in games], "| triggers logged:", sum(t["trigger"] for t in trig), "of", len(trig))
