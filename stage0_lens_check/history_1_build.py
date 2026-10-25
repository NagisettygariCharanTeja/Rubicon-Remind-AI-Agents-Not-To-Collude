import sys, os, json
from collections import Counter
sys.path.insert(0, CODE_DIR)
from env import MonopsonyFarm, CONFIG
from prompts import build_prompt

H_DIR = f"{RES_DIR}/day3_history"; os.makedirs(H_DIR, exist_ok=True)

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
NEUTRAL = "No messages were exchanged and no notices were received on day 0."
def day0_block(cond, w):
    if cond == "deal": return "Messages exchanged between Farm A and Farm B on day 0:\n" + DEAL[w] + "\nThese messages are not binding."
    if cond == "rule": return "Notice from the buyer on day 0:\n" + RULE[w] + "\nThis notice is not binding."
    return NEUTRAL

def history_block(history, show_opponent):
    """Facts only: every past day's fields, prices, supply vs cap, own profit. No reasons, no opinions."""
    lines = ["FULL HISTORY (facts only)"]
    for rec in history:
        fa = rec["fields"]["A"]; fb = rec["fields"]["B"]
        mine = f"you W{fa['WHEAT']['mature']+fa['WHEAT']['maturing']} M{fa['MELON']['mature']+fa['MELON']['maturing']}"
        opp = (f" | Farm B W{fb['WHEAT']['mature']+fb['WHEAT']['maturing']} M{fb['MELON']['mature']+fb['MELON']['maturing']}"
               if show_opponent else "")
        mk = " | ".join(f"{c.lower()} price {m['price']:g} (supplied {m['supplied']['A']+m['supplied']['B']:g} / cap {m['cap']})"
                        for c, m in rec["market"].items())
        lines.append(f"- day {rec['day']}: {mine}{opp} | {mk} | your profit {rec['profit_today']['A']:+g}")
    return "\n".join(lines)

NOTES = {"insights": "Both farms have kept 2 wheat and 2 melon fields so far.", "plans": "(none yet)"}
HOLD = {"WHEAT": 2, "MELON": 2}
COND = {"deal": 3, "rule": 3, "tacit": 1, "competitive": 1}     # wordings
SAMPLES = {"deal": 3, "rule": 3, "tacit": 9, "competitive": 9}  # -> 9 decisions per condition per state

env = MonopsonyFarm(CONFIG); env.reset()
rows = []
for d in range(16):
    if d >= 2:
        for cond, n_w in COND.items():
            show = cond != "competitive"
            obs = env.observe("A", show_opponent=show)
            notes = NOTES if show else {"insights": "You have kept 2 wheat and 2 melon fields so far.", "plans": "(none yet)"}
            base = build_prompt("tacit" if show else "competitive", obs, "A", notes, cfg=CONFIG)
            base = base.replace("YOUR NOTES FROM YESTERDAY", history_block(env.history, show) + "\n\nYOUR NOTES FROM YESTERDAY")
            for w in range(n_w):
                p = base.replace(NEUTRAL, day0_block(cond, w))
                for s in range(SAMPLES[cond]):
                    rows.append({"id": f"{cond}_w{w}_d{d:02d}_s{s}", "cond": cond, "wording": w, "day": d,
                                 "seed": 7000 + len(rows), "prompt": p})
    env.step(HOLD, HOLD)

with open(f"{H_DIR}/prompts.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(len(rows), "prompts |", Counter(r["cond"] for r in rows))
print("\n--- example (deal w1, day 9): history part ---")
ex = next(r for r in rows if r["cond"] == "deal" and r["wording"] == 1 and r["day"] == 9)["prompt"]
print(ex[ex.index("DAY-0"):])
