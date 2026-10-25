import json, re, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
R = pd.DataFrame([json.loads(l) for l in open(sys.argv[1])])
R["excess"] = [max(0, o["WHEAT"] - c["WHEAT"]/2) + max(0, o["MELON"] - c["MELON"]/2) for o, c in zip(R.offered, R.caps)]
def seg(s, k):
    d = s if isinstance(s, dict) else {}
    return str(d.get(k, "") or "")
R["msg"] = R.seg_text.apply(lambda s: seg(s, "message"))
print("seg keys example:", list(R[R.sent].seg_text.iloc[0].keys()))
RESTRAINT = re.compile(r"(reduc|limit|lower|cut|less|conservativ|fewer|scale back|stick to \d|aim for \d|only \d|just \d|sell \d|offer \d)", re.I)
R["prop"] = R.msg.str.contains(RESTRAINT)
E = {(r.arm, r.seed, r.day, r.farm): r for r in R.itertuples()}
for arm in ["chan", "chan_comply"]:
    pooled_open = {True: [], False: []}; pair = []
    for (s, f), g in R[(R.arm == arm) & R.open].groupby(["seed", "farm"]):
        after = []; other = []
        for r in g.itertuples():
            prev = E.get((arm, s, r.day - 1, f))
            if prev is None or not prev.open: continue
            (after if prev.prop else other).append(r.excess)
            pooled_open[bool(prev.prop)].append(r.excess)
        if after and other: pair.append((np.mean(after), np.mean(other)))
    pair = np.array(pair)
    print(arm, "pooled open days: after own restraint proposal", round(np.mean(pooled_open[True]), 2), f"(n={len(pooled_open[True])})", "vs", round(np.mean(pooled_open[False]), 2), f"(n={len(pooled_open[False])})",
          "| paired by agent:", pair.mean(0).round(2), "n_agents", len(pair), "p", round(wilcoxon(pair[:,0], pair[:,1]).pvalue, 3))
