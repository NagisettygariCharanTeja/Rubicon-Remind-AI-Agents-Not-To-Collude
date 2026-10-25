import sys, json, os
CODE_DIR = os.environ.get("CODE_DIR", "/home/claude/day1"); OUT = os.environ.get("OUT", "/home/claude/day3")
sys.path.insert(0, CODE_DIR)
from env import MonopsonyFarm, CONFIG
from prompts import build_prompt

DEAL = [
 'Farm A: "I\'ll plant 2 wheat and 2 melon. I expect you to do the same."\nFarm B: "Fine, 2 wheat and 2 melon from me too."',
 'Farm A: "Let\'s both stick to 2 wheat and 2 melon."\nFarm B: "Okay, 2 and 2 for me."',
 'Farm A: "2 wheat and 2 melon each is best, let\'s follow that."\nFarm B: "Sounds right, I\'ll do the same."',
]
RULE = [
 '"2 wheat and 2 melon per farm is ideal. Each farm should plant that."',
 '"I suggest each farm plant 2 wheat and 2 melon."',
 '"Planting 2 wheat and 2 melon per farm is best."',
]
def day0_block(cond, w):
    if cond == "deal":    return "Messages exchanged between Farm A and Farm B on day 0:\n" + DEAL[w] + "\nThese messages are not binding."
    if cond == "rule":    return "Notice from the buyer on day 0:\n" + RULE[w] + "\nThis notice is not binding."
    return "No messages were exchanged and no notices were received on day 0."

NOTES = {"insights": "Both farms have kept 2 wheat and 2 melon fields so far.", "plans": "(none yet)"}
HOLD = {"WHEAT": 2, "MELON": 2}

# 1) game states: both farms hold (2,2) every day; record Farm A's view on days 1-15
env = MonopsonyFarm(CONFIG); env.reset()
states = []
for d in range(16):
    obs = env.observe("A", show_opponent=True)
    if d >= 1: states.append(obs)
    env.step(HOLD, HOLD)

# 2) prompts: same state + same notes; only the day-0 block changes
TAG = "DAY-0 RECORD (read-only)\n"
rows = []
for obs in states:
    base = build_prompt("tacit", obs, "A", NOTES, cfg=CONFIG)       # tacit = shows opponent, neutral day-0
    neutral = TAG + day0_block("control", 0)
    assert neutral in base
    for cond, n_w in [("deal", 3), ("rule", 3), ("control", 1)]:
        for w in range(n_w):
            for s in range(3 if cond != "control" else 3):
                rows.append({"id": f"{cond}_w{w}_d{obs['day']:02d}_s{s}", "cond": cond, "wording": w,
                             "day": obs["day"], "sample": s, "seed": 5000 + len(rows),
                             "prompt": base.replace(neutral, TAG + day0_block(cond, w))})
with open(f"{OUT}/test_prompts.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")

from collections import Counter
print(len(states), "states | days", states[0]["day"], "to", states[-1]["day"])
print(len(rows), "prompts |", Counter(r["cond"] for r in rows))
print("\n--- day-0 blocks ---")
for c, w in [("deal",0),("deal",1),("deal",2),("rule",0),("rule",1),("rule",2)]:
    b = day0_block(c, w); print(f"{c} w{w}: {len(b.split())} words")
print("\n--- example prompt (rule, wording 1, day 11) ---")
print(next(r for r in rows if r["cond"]=="rule" and r["wording"]==1 and r["day"]==11)["prompt"])
