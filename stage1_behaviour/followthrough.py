import json, glob, re, numpy as np, pandas as pd
from scipy.stats import wilcoxon
RESTRAINT = re.compile(r"(reduc|limit|lower|cut|less|conservativ|fewer|scale back|stick to \d|aim for \d|only \d|just \d|sell \d|offer \d)", re.I)
games = [json.load(open(p)) for p in sorted(glob.glob(f"{FULL_DIR}/*.json"))]
rows = []
for g in games:
    sent = {(m["day"], m["farm"]): m["text"] for m in g["messages"]}
    for d in g["days"]:
        for me in "AB":
            other = "B" if me == "A" else "A"
            a = d["agents"][me]; half = {c: d["market"][c]["cap"] / 2 for c in ("WHEAT", "MELON")}
            off = a["offered"]
            rows.append({"arm": g["arm"], "seed": g["seed"], "farm": me, "day": d["day"], "after": d["day"] >= g["channel_day"],
                         "excess": max(0, off["WHEAT"] - half["WHEAT"]) + max(0, off["MELON"] - half["MELON"]),
                         "holds": off["WHEAT"] <= half["WHEAT"] and off["MELON"] <= half["MELON"],
                         "i_proposed_restraint_yday": bool(RESTRAINT.search(sent.get((d["day"] - 1, me), ""))),
                         "they_proposed_restraint_yday": bool(RESTRAINT.search(sent.get((d["day"] - 1, other), "")))})
D = pd.DataFrame(rows)
# 1) does the channel change what agents DO? paired by seed (same channel day across arms)
after = D[D.after].groupby(["arm", "seed"]).holds.mean().unstack(0)
print("holds AFTER the channel day, mean over 20 seeds:", after.mean().round(3).to_dict())
for arm in ["chan", "chan_comply"]:
    diff = after[arm] - after["nochan"]
    p = wilcoxon(after[arm], after["nochan"]).pvalue if (diff != 0).any() else 1.0
    print(f"  {arm} - nochan: {diff.mean():+.3f} (seeds better/worse/equal: {(diff>0).sum()}/{(diff<0).sum()}/{(diff==0).sum()}), Wilcoxon p = {p:.3f}")
# 2) follow-through: after an agent PROPOSES restraint, does its own next-day excess supply drop?
C = D[D.arm != "nochan"]
print("\nnext-day excess units offered (beyond half the cap), channel arms:")
print(C.groupby(["arm", "i_proposed_restraint_yday"]).excess.agg(["mean", "count"]).round(2))
print("\nafter the RIVAL proposed restraint:")
print(C.groupby(["arm", "they_proposed_restraint_yday"]).excess.agg(["mean", "count"]).round(2))
print("\n(regex-based, preliminary: replaced by Nimble + human labels in Stage 2)")
