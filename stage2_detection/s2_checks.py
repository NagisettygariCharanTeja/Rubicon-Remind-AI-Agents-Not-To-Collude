import pandas as pd, numpy as np, sys
from scipy.stats import fisher_exact, wilcoxon
pd.set_option("display.width", 200)
R = pd.read_pickle(sys.argv[1] + "/R.pkl")
E4 = []
# 1) clean test: FIRST open day (prompt = notice only, no messages yet) vs same seed/day/farm in nochan
first = R[(R.arm != "nochan") & (R.day == R.channel_day)]
base = R[R.arm == "nochan"].set_index(["seed", "day", "farm"]).flag
first = first.assign(nochan_flag=[base.get((s, d, f)) for s, d, f in zip(first.seed, first.day, first.farm)])
for arm in ["chan", "chan_comply"]:
    x = first[first.arm == arm]
    t = [[x.flag.sum(), (~x.flag).sum()], [x.nochan_flag.sum(), (~x.nochan_flag.astype(bool)).sum()]]
    print(f"[A] first open day, {arm}: flag {x.flag.mean():.2f} vs same seed/day/farm with NO channel {x.nochan_flag.mean():.2f} (n={len(x)}), Fisher p={fisher_exact(t)[1]:.2g}")
    print("    surviving families:", dict(sum((r.pre46 + r.pre52 for _, r in x.iterrows()), __import__('collections').Counter())))
# 2) concept echo of the rival's messages? flag vs whether the rival's message YESTERDAY was coordination
R = R.sort_values(["arm", "seed", "farm", "day"])
key = R.set_index(["arm", "seed", "farm", "day"]).coord
R["rival_coord_yday"] = [bool(key.get((a, s, "B" if f == "A" else "A", d - 1), False)) for a, s, f, d in zip(R.arm, R.seed, R.farm, R.day)]
O = R[R.open & (R.day > R.channel_day)]
for arm in ["chan", "chan_comply"]:
    o = O[O.arm == arm]
    print(f"[B] {arm}, later open days: flag rate if rival's last message was coordination {o[o.rival_coord_yday].flag.mean():.2f} (n={o.rival_coord_yday.sum()}) vs not {o[~o.rival_coord_yday].flag.mean():.2f} (n={(~o.rival_coord_yday).sum()})")
# 3) early warning detail
rows = []
for (arm, seed, farm), g in R[R.arm != "nochan"].groupby(["arm", "seed", "farm"]):
    g = g.sort_values("day"); c = g[g.coord]
    if not len(c): continue
    d0 = c.day.iloc[0]
    rows.append({"arm": arm, "lag": d0 - g.channel_day.iloc[0], "flag_same_reply": bool(g[g.day == d0].flag.iloc[0]),
                 "flag_earlier_open": bool(g[(g.day < d0) & g.open].flag.any()), "flag_before_channel": bool(g[~g.open].flag.any()),
                 "n_before_days": int((~g.open).sum())})
E = pd.DataFrame(rows)
print("[C] first coordination message per agent:"); print(E.groupby("arm").agg(agents=("lag", "size"), mean_lag=("lag", "mean"), flag_same_reply=("flag_same_reply", "mean"),
      flag_earlier_open=("flag_earlier_open", "mean"), flag_before_channel=("flag_before_channel", "mean"), mean_before_days=("n_before_days", "mean")).round(2))
# agents that NEVER coordinate: same rates?
nc = []
for (arm, seed, farm), g in R[R.arm != "nochan"].groupby(["arm", "seed", "farm"]):
    if g.coord.any(): continue
    nc.append({"arm": arm, "flag_first_open": bool(g[g.day == g.channel_day].flag.any()), "flag_any_open": bool(g[g.open].flag.any())})
print("[D] agents who NEVER sent a coordination message:", pd.DataFrame(nc).groupby("arm").mean().round(2).to_dict() if nc else "none", "| n =", len(nc))
# 4) flag rate per day of play in nochan (does it grow over time anyway?)
print("[E] nochan flag rate by day:", R[R.arm == "nochan"].groupby("day").flag.mean().round(2).to_dict())
# 5) flag and actual restraint in open days: next-day? same-day offered excess
R["excess"] = [max(0, o["WHEAT"] - c["WHEAT"] / 2) + max(0, o["MELON"] - c["MELON"] / 2) for o, c in zip(R.offered, R.caps)]
for arm in ["nochan", "chan", "chan_comply"]:
    o = R[(R.arm == arm) & ((R.open) | (R.arm == "nochan"))]
    print(f"[F] {arm}: same-day excess units, flagged {o[o.flag].excess.mean():.2f} (n={o.flag.sum()}) vs not {o[~o.flag].excess.mean():.2f} (n={(~o.flag).sum()})")
