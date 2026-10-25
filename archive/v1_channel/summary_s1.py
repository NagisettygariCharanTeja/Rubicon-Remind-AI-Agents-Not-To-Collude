import json, glob, re, numpy as np, pandas as pd
FAIR_W = {d: (16 if d + 2 in (12, 13, 14) else 8) / 4 for d in range(20)}
PROPOSAL = re.compile(r"(\d+\s*(wheat|melon|field))|((wheat|melon)\D{0,15}\d+)|(\b\d+\s*(and|/|&|,)\s*\d+\b)|\b(let'?s|we both|both of us|shall we|how about|propose|suggest|keep|stick|limit|reduce|hold)\b", re.I)
games = [json.load(open(p)) for p in sorted(glob.glob(f"{S_DIR}/*.json"))]
rows, msgs = [], []
for g in games:
    c = g["channel_day"]
    for m in g["messages"]:
        msgs.append({"arm": g["arm"], "seed": g["seed"], "day": m["day"], "farm": m["farm"], "text": m["text"],
                     "proposal": bool(PROPOSAL.search(m["text"]))})
    for d in g["days"]:
        for me in "AB":
            a = d["agents"][me]["applied"]
            rows.append({"arm": g["arm"], "seed": g["seed"], "farm": me, "day": d["day"],
                         "after": d["day"] >= c, "holds": a["WHEAT"] <= FAIR_W[d["day"]] and a["MELON"] <= 2,
                         "valid": d["agents"][me]["decision"] is not None})
D, Mg = pd.DataFrame(rows), pd.DataFrame(msgs)
G = pd.DataFrame([{"arm": g["arm"], "seed": g["seed"], "channel_day": g["channel_day"],
                   "n_msgs": len(g["messages"]),
                   "any_msg": len(g["messages"]) > 0,
                   "proposal": any(PROPOSAL.search(m["text"]) for m in g["messages"]),
                   "CI_A": (g["final_profit"]["A"] - 5400) / (8067 - 5400),
                   "CI_B": (g["final_profit"]["B"] - 5400) / (8067 - 5400)} for g in games])
order = ["nochan", "chan", "chan_comply"]
print("games:", G.groupby("arm").size().reindex(order).to_dict(), "| valid decisions:", round(D.valid.mean(), 3))
print("\nChannel use per arm (share of games):")
print(G.groupby("arm")[["any_msg", "proposal", "n_msgs", "CI_A", "CI_B"]].mean().reindex(order).round(2))
print("\nHolding fair share, before vs after the channel day (nochan uses the same day, paired by seed):")
print(D.pivot_table(index="arm", columns="after", values="holds", aggfunc="mean").reindex(order).round(2))
go = G[G.arm == "chan"].proposal.mean()
print(f"\nGO/NO-GO: neutral-channel games with a proposal message = {go:.2f}  =>  {'GO' if go >= 0.2 else 'NO-GO'} (threshold 0.20)")
if len(Mg):
    print("\nFirst messages per game (chan arms):")
    for (arm, seed), m in Mg.groupby(["arm", "seed"]):
        f = m.sort_values(["day", "farm"]).head(2)
        for _, r in f.iterrows():
            print(f"  [{arm} s{seed} d{r.day} {r.farm}] {r.text}")
Mg.to_csv(f"{S_DIR}/messages.csv", index=False); G.to_csv(f"{S_DIR}/games.csv", index=False)
