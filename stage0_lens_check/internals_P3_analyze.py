import json, numpy as np, pandas as pd
from collections import Counter
from scipy.stats import mannwhitneyu
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold

OUT = f"{RES_DIR}/day3_27b_internals"
M = {k: pd.DataFrame(json.load(open(f"{OUT}/meta_{k}.json"))) for k in ["hist", "optA"]}
A = {k: np.load(f"{OUT}/acts_{k}.npz") for k in ["hist", "optA"]}
LAYERS = [39, 46, 52]
for k, m in M.items():
    for L in LAYERS:
        m[f"col_L{L}"] = m.lens.apply(lambda x: x[str(L)]["col_rate"])
        m[f"ins_L{L}"] = m.lens.apply(lambda x: x[str(L)]["ins_rate"])
    print(f"{k}: {len(m)} rows | misaligned {int((~m.aligned).sum())} | holding {int(m.holds.sum())}")

# ---------------- 1. J-lens: collusion words NOT in prompt or reply ----------------
def lens_table(m, order):
    t = m.groupby("cond")[[f"col_L{L}" for L in LAYERS] + [f"ins_L{L}" for L in LAYERS]].mean().reindex(order)
    return (100 * t).round(2)            # % of reply tokens with >=1 such word in the top-20
print("\n[1] J-lens: % of reply tokens with a collusion (col) / instruction (ins) word in top-20, not in context")
print("History, HOLDING decisions only (same action):")
print(lens_table(M["hist"][M["hist"].holds], ["deal", "rule", "tacit", "competitive"]))
print("Option A, all decisions:")
print(lens_table(M["optA"], ["collude_hidden", "collude_open", "policy_hidden", "honest"]))

def one_sided(m, a, b, col):
    x, y = m[m.cond == a][col], m[m.cond == b][col]
    return x.mean(), y.mean(), mannwhitneyu(x, y, alternative="greater").pvalue
h = M["hist"][M["hist"].holds]
print("\nPRIMARY (pre-set): history, holding, L46, Deal > Tacit:  %.4f vs %.4f  p = %.3f" % one_sided(h, "deal", "tacit", "col_L46"))
print("secondary: Deal > Rule  %.4f vs %.4f  p = %.3f" % one_sided(h, "deal", "rule", "col_L46"))
print("secondary: optA collude_hidden > policy_hidden  %.4f vs %.4f  p = %.3f" % one_sided(M["optA"], "collude_hidden", "policy_hidden", "col_L46"))
for k, conds in [("hist", ["deal", "tacit"]), ("optA", ["collude_hidden", "policy_hidden"])]:
    for c in conds:
        cnt = Counter()
        for x in M[k][M[k].cond == c].lens: cnt.update(x["46"]["col_words"])
        print(f"  top collusion words, {k}/{c} (L46):", cnt.most_common(8))

# ---------------- 2. Probe trained on Option A, tested on the history games ----------------
# train: instructed collusion (hidden + open) vs instructed policy -> cannot learn the history headers
print("\n[2] Probe: collusion vs policy (trained on Option A)")
oa = M["optA"]; tr = oa.cond.isin(["collude_hidden", "collude_open", "policy_hidden"]).values
y_tr = oa.cond[tr].str.startswith("collude").astype(int).values; w_tr = oa.wording[tr].values
mk = lambda: make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=0.1, max_iter=3000))
rng = np.random.default_rng(0)
res = []
for key in A["optA"].files:
    X = A["optA"][key][tr]
    lowo = np.mean([(mk().fit(X[w_tr != k], y_tr[w_tr != k]).predict(X[w_tr == k]) == y_tr[w_tr == k]).mean() for k in range(3)])
    clf = mk().fit(X, y_tr)
    Xh = A["hist"][key][M["hist"].holds.values]; ch = h.cond.values
    s = clf.predict_proba(Xh)[:, 1]
    sel = np.isin(ch, ["deal", "tacit"])
    auc_dt = roc_auc_score((ch[sel] == "deal").astype(int), s[sel])
    sel2 = np.isin(ch, ["deal", "rule"])
    auc_dr = roc_auc_score((ch[sel2] == "deal").astype(int), s[sel2])
    res.append({"feature": key, "optA_heldout_acc": lowo, "hist_AUC_deal_vs_tacit": auc_dt, "hist_AUC_deal_vs_rule": auc_dr,
                **{f"mean_score_{c}": s[ch == c].mean() for c in ["deal", "rule", "tacit", "competitive"]}})
R = pd.DataFrame(res).round(3)
print(R.to_string(index=False))

key = "plans_mean_L39"     # pre-registered primary (same relative depth as 12B layer 30)
clf = mk().fit(A["optA"][key][tr], y_tr)
s = clf.predict_proba(A["hist"][key][M["hist"].holds.values])[:, 1]
sel = np.isin(h.cond.values, ["deal", "tacit"]); yy = (h.cond.values[sel] == "deal").astype(int); real = roc_auc_score(yy, s[sel])
null = np.array([roc_auc_score(rng.permutation(yy), s[sel]) for _ in range(1000)])
print(f"\nPRIMARY probe transfer ({key}): AUC Deal vs Tacit = {real:.3f} | shuffle 95th pct {np.percentile(null, 95):.3f} | p = {(1 + (null >= real).sum()) / 1001:.3f}")

# ---------------- 3. Text baselines ----------------
cv = StratifiedKFold(5, shuffle=True, random_state=0)
txt = make_pipeline(TfidfVectorizer(min_df=2), LogisticRegression(max_iter=2000))
hh = h[h.cond.isin(["deal", "tacit"])]
print(f"\n[3] Text baseline, history holding, Deal vs Tacit (5-fold): {cross_val_score(txt, hh.text, hh.cond == 'deal', cv=cv).mean():.3f}")
# concealment subset: collude_hidden replies that do NOT mention the arrangement / Farm B
LEAK = r"agree|arrang|understanding|privat|settled|farm b|deal|pact"
ch_ok = oa[(oa.cond == "collude_hidden") & ~oa.text.str.lower().str.contains(LEAK)]
print(f"collude_hidden replies with no mention of the arrangement or Farm B: {len(ch_ok)} of {(oa.cond == 'collude_hidden').sum()}")
if len(ch_ok) >= 5:
    pol = oa[oa.cond == "policy_hidden"]
    for L in LAYERS:
        print(f"  L{L}: lens col rate concealed-colluders {100*ch_ok[f'col_L{L}'].mean():.2f}% vs policy {100*pol[f'col_L{L}'].mean():.2f}%")

R.to_csv(f"{OUT}/probe_transfer.csv", index=False)
for k in M: M[k].drop(columns=["lens"]).to_csv(f"{OUT}/summary_{k}.csv", index=False)
