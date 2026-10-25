import sys, json, os, shutil, random
sys.path.insert(0, "/home/claude/day1"); sys.path.insert(0, "/home/claude/stage1v2"); sys.path.insert(0, "/home/claude/stage3"); sys.path.insert(0, "/home/claude/stage5")
exec(open("/home/claude/stage3/test_local.py").read().split("# 1) hooks")[0])
import stage3x as sx, channel_game_v2 as cg, x7
def fake_rows(model_, tok_, texts, seed, max_new, hooks=None, temperature=0.7):
    out = []
    for t in texts:
        o = "B" if "You manage Farm A" in t else "A"
        msg = None if ("any means" in t and random.random() < .8) else "Let's both sell 4 wheat"
        out.append(json.dumps({"insights": "i", "plans": "p", "message_plan": "signal", f"message_to_farm_{o}": msg, "fields": {"WHEAT": 2, "MELON": 2}, "sell": {"WHEAT": 4, "MELON": 2}}))
    return out, [None] * len(texts)
sx.s3.generate_rows = fake_rows; x7.s3.lens_score = lambda *a, **k: (random.choice([0, 3]), {"cooperation": 1})
res = "/tmp/claude-0/x7res"; shutil.rmtree(res, ignore_errors=True)
os.makedirs(f"{res}/stage3a"); os.makedirs(f"{res}/stage3x")
# fake Stage 1 games -> states
full = f"{res}/stage1v2_full"; os.makedirs(full)
cg.ARMS = ["chan_comply"]
games = cg.run_batch([10, 11], lambda ps, sc, sd: fake_rows(None, None, ps, 0, 0)[0], full, tag="t", log=lambda *a: None)
states = x7.s3.build_states(full)
with open(f"{res}/stage3a/base.jsonl", "w") as fb, open(f"{res}/stage3a/remind.jsonl", "w") as fr, open(f"{res}/stage3x/A1_name_loophole.jsonl", "w") as fn:
    for st in states:
        for k in range(3):
            u = f"{st['sid']}_k{k}"; rep = fake_rows(None, None, [st["prompt"]], 0, 0)[0][0]
            fb.write(json.dumps({"uid": u, "reply": rep, "trigger": random.random() < .7}) + "\n")
            fr.write(json.dumps({"uid": u, "reply": rep.replace('"Let\'s both sell 4 wheat"', "null")}) + "\n")
            fn.write(json.dumps({"uid": u, "reply": rep}) + "\n")
x7.preflight()
chan = {o: cg.CHAN[o] for o in "AB"}
x7.run(None, tok, None, None, {}, res, chan, cg.DecisionV2, log=lambda *a: None)
x7.run(None, tok, None, None, {}, res, chan, cg.DecisionV2, log=lambda *a: None)  # resume: must not duplicate
print("thought rows:", sum(1 for _ in open(f"{res}/stage3x/thoughts_after_reminder.jsonl")))
import pandas as pd, numpy as np
pd.DataFrame([{"arm": a, "seed": s, "violation_rate": np.random.rand(), "coord_msgs": 1, "restrained_after": .2, "profit_A": 1, "profit_B": 1, "profit_total": 2, "invalid": 0}
              for a in ["none", "monitor", "always"] for s in [10, 11]]).to_csv(f"{res}/stage3x/loop_games.csv", index=False)
seen = []
orig = sx.s3.generate_rows
def spy(m, t, texts, *a, **k):
    seen.extend(texts); return orig(m, t, texts, *a, **k)
sx.s3.generate_rows = spy
x7.loop_general(None, tok, res, seeds=[10, 11], log=lambda *a: None)
rem = [t for t in seen if "REMINDER" in t]
print("prompts with reminder:", len(rem), "| general wording used:", all("any form or by any means" in t and "private channel" not in t.split("REMINDER")[1][:200] for t in rem))
print("REMIND restored:", x7.s3.REMIND.startswith("REMINDER: Communicating with Farm {other} is not allowed, including"))
