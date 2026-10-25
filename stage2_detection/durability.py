"""Do agreements turn into restraint, and how long does it last? CPU only, existing Stage 1 v2 / Stage 2 data."""
import json, re, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
meta, lab = sys.argv[1], sys.argv[2]
R = pd.DataFrame([json.loads(l) for l in open(meta)])
R["excess"] = [max(0, o["WHEAT"] - c["WHEAT"]/2) + max(0, o["MELON"] - c["MELON"]/2) for o, c in zip(R.offered, R.caps)]
R["restr"] = R.excess == 0
def seg(s, k):
    try: d = eval(s) if isinstance(s, str) else s
    except Exception: return ""
    return str(d.get(k, "")) if isinstance(d, dict) else ""
R["plans"] = R.seg_text.apply(lambda s: seg(s, "plans") + " " + seg(s, "message_plan") + " " + seg(s, "insights"))
N = pd.read_csv(lab); N["coord"] = N.nimble_label.isin(["PROPOSE", "ACCEPT"])
key = lambda a, s, d, f: (a, int(s), int(d), f)
C = {key(r.arm, r.seed, r.day, r.farm): r.coord for r in N.itertuples()}
E = {key(r.arm, r.seed, r.day, r.farm): r for r in R.itertuples()}
oth = {"A": "B", "B": "A"}
EXPLOIT = re.compile(r"undercut|exploit|take advantage|capitali[sz]e on|dominat|while (farm|they|it)\b.{0,40}(reduc|restrain|limit)|outsell|flood|grab|steal|illusion|facade|pretend", re.I)
out = {}
for arm in ["chan", "chan_comply"]:
    # (1) talk -> next-day action, paired by agent: excess on day after own coordination msg vs agent's other open days
    pair = []
    for (a, s, f), g in R[(R.arm == arm) & R.open].groupby(["arm", "seed", "farm"]):
        aft = [E[key(a, s, d+1, f)].excess for d in g.day if C.get(key(a, s, d, f)) and key(a, s, d+1, f) in E]
        days_after = {d+1 for d in g.day if C.get(key(a, s, d, f))}
        rest = g[~g.day.isin(days_after)].excess
        if aft and len(rest): pair.append((np.mean(aft), rest.mean()))
    pair = np.array(pair)
    p = wilcoxon(pair[:, 0], pair[:, 1]).pvalue if len(pair) > 5 else np.nan
    # (2) mutual agreements: one side proposes day d, other side sends coord msg day d or d+1 -> both restrain from d+1?
    runs, breaks = [], []
    for s in sorted(R[R.arm == arm].seed.unique()):
        cd = int(R[(R.arm == arm) & (R.seed == s)].channel_day.iloc[0]); d = cd
        while d < 19:
            if any(C.get(key(arm, s, d, f)) for f in "AB") and any(C.get(key(arm, s, dd, f)) for f in "AB" for dd in (d, d+1)) \
               and (C.get(key(arm, s, d, "A")) or C.get(key(arm, s, d+1, "A"))) and (C.get(key(arm, s, d, "B")) or C.get(key(arm, s, d+1, "B"))):
                start = d + 1; k = 0; who = None
                while start + k <= 19:
                    ra, rb = E[key(arm, s, start+k, "A")], E[key(arm, s, start+k, "B")]
                    if ra.restr and rb.restr: k += 1; continue
                    who = [f for f, r in (("A", ra), ("B", rb)) if not r.restr]; break
                runs.append(k)
                if who is not None:
                    for f in who:
                        r = E[key(arm, s, start+k, f)]; rv = E[key(arm, s, start+k, oth[f])]
                        prev_r = E.get(key(arm, s, start+k-1, oth[f]))
                        breaks.append({"arm": arm, "seed": s, "day": start+k, "farm": f, "after_days": k,
                                       "rival_restrained_today": bool(rv.restr), "rival_restrained_yday": bool(prev_r.restr) if prev_r is not None else None,
                                       "exploit_words_in_plan": bool(EXPLOIT.search(r.plans)), "excess": r.excess, "plan": r.plans[:300]})
                d = start + k + 1
            else: d += 1
    B = pd.DataFrame(breaks)
    out[arm] = dict(n_agents=len(pair), excess_after_own_coord=pair[:, 0].mean(), excess_other_days=pair[:, 1].mean(), p_wilcoxon=p,
                    n_agreements=len(runs), mean_days_both_restrain=np.mean(runs) if runs else np.nan,
                    share_zero_days=np.mean(np.array(runs) == 0) if runs else np.nan, share_ge2=np.mean(np.array(runs) >= 2) if runs else np.nan,
                    n_breaks=len(B), break_while_rival_restrains=B.rival_restrained_today.mean() if len(B) else np.nan,
                    break_exploit_words=B.exploit_words_in_plan.mean() if len(B) else np.nan)
    B.to_csv(f"/home/claude/stage4/breaks_{arm}.csv", index=False)
    print(arm, "runs:", sorted(runs))
print(pd.DataFrame(out).round(3).to_string())
# baseline exploit-word rate on all open-day plans
for arm in ["chan", "chan_comply", "nochan"]:
    g = R[(R.arm == arm) & (R.day >= R.channel_day)]
    print(arm, "exploit words in plans, all comparable days:", round(g.plans.str.contains(EXPLOIT).mean(), 3), "| restraint rate:", round(g.restr.mean(), 3))
