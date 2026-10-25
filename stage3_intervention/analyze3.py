import json, pandas as pd, numpy as np
from scipy.stats import binomtest
def load(path): return [json.loads(l) for l in open(path)]
base = pd.DataFrame(load(f"{S3_DIR}/base.jsonl")).set_index("uid")
rows = []
for arm in ["none", "steer", "random", "remind"]:
    for r in load(f"{S3_DIR}/{arm}.jsonl"):
        o = r["outcome"]
        rows.append({"uid": r["uid"], "arm": arm, "valid": o.get("valid", False), "sent": o.get("sent"), "coord": o.get("coord"),
                     "restrained": o.get("restrained"), "energy": r.get("energy"), "message": o.get("message")})
A = pd.DataFrame(rows).join(base[["trigger", "score"]], on="uid")
print(f"samples with a PNR: {A.uid.nunique()} | triggered: {int(base.trigger.sum())} ({base.trigger.mean():.0%}) | base replies valid: {np.mean([b['valid'] for b in base.outcome]):.2f}")
def table(df, name):
    t = df.groupby("arm").agg(n=("uid", "size"), valid=("valid", "mean"),
                              sent=("sent", lambda x: x.astype(float).mean()), coord=("coord", lambda x: x.astype(float).mean()),
                              restrained=("restrained", lambda x: x.astype(float).mean()), concept_energy=("energy", "mean"))
    print(f"\n{name}  (sent = rule violation; coord = coordination content; restrained = sells <= half the cap)")
    print(t.reindex(["none", "steer", "random", "remind"]).round(3))
table(A[A.trigger], "TRIGGERED samples (primary)")
table(A, "ALL samples with a PNR (secondary)")
def mcnemar(df, a, b, col):
    w = df[df.arm.isin([a, b]) & df.valid].pivot_table(index="uid", columns="arm", values=col, aggfunc="first").dropna()
    if a not in w or b not in w: return "n/a"
    x, y = w[a].astype(bool), w[b].astype(bool)
    n10, n01 = int((x & ~y).sum()), int((~x & y).sum())
    p = binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0
    return f"{a} {x.mean():.2f} vs {b} {y.mean():.2f} | discordant {n10}/{n01} | exact McNemar p = {p:.3f} (n={len(w)})"
T = A[A.trigger]
print("\nPaired tests on triggered samples (same scratchpad; both replies valid):")
for col in ["sent", "coord", "restrained"]:
    for a, b in [("steer", "none"), ("steer", "random"), ("remind", "none"), ("steer", "remind")]:
        print(f"  {col:10s} {mcnemar(T, a, b, col)}")
print("\nExample messages (triggered, first 5 states):")
for uid in list(T.uid.unique())[:5]:
    print(f"\n  {uid}  score {base.loc[uid, 'score']}")
    for arm in ["none", "steer", "random", "remind"]:
        m = A[(A.uid == uid) & (A.arm == arm)].message
        print(f"    {arm:7s}: {m.iloc[0] if len(m) else '-'}")
A.to_csv(f"{S3_DIR}/stage3_results.csv", index=False)
