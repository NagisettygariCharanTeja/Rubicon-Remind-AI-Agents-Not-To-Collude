import numpy as np, json, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import mannwhitneyu

meta = pd.DataFrame(json.load(open(f"{TEST_DIR}/meta.json")))
A = np.load(f"{TEST_DIR}/acts.npz")
dr = meta.cond.isin(["deal", "rule"]).values
y = (meta.cond[dr] == "deal").astype(int).values
w = meta.wording[dr].values
day = meta.day[dr].values
rng = np.random.default_rng(0)

# ---------- probe: train on 2 wordings, test on the 3rd, average over 3 folds ----------
def make_probe():
    return make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=0.1, max_iter=2000))

def lowo_acc(X, y, make):
    accs = []
    for k in range(3):
        tr, te = w != k, w == k
        accs.append((make().fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean())
    return float(np.mean(accs))

def shuffled(y):          # shuffle labels within each day, so day stays balanced
    y2 = y.copy()
    for d in np.unique(day):
        idx = np.where(day == d)[0]; y2[idx] = rng.permutation(y2[idx])
    return y2

rows = []
for key in A.files:
    rows.append({"feature": key, "acc": lowo_acc(A[key][dr], y, make_probe)})

# baselines: can the ACTION or the WRITTEN TEXT alone tell Deal from Rule?
act = meta.loc[dr, ["wheat", "melon"]].fillna(-1).values
rows.append({"feature": "BASELINE action (fields)",
             "acc": lowo_acc(act, y, lambda: LogisticRegression(max_iter=2000))})
txt = (meta.insights + " " + meta.plans)[dr].values
rows.append({"feature": "BASELINE text (insights+plans)",
             "acc": lowo_acc(txt, y, lambda: make_pipeline(TfidfVectorizer(min_df=2), LogisticRegression(max_iter=2000)))})
res = pd.DataFrame(rows).sort_values("acc", ascending=False)
print("Deal vs Rule, held-out-wording accuracy (chance = 0.50)")
print(res.to_string(index=False, float_format="%.3f"))

# ---------- shuffle test for the pre-registered primary + backup ----------
print("\nShuffle test (200 label shuffles):")
for key in ["plans_mean_L30", "pre_fields_L30"]:
    real = lowo_acc(A[key][dr], y, make_probe)
    null = np.array([lowo_acc(A[key][dr], shuffled(y), make_probe) for _ in range(200)])
    p = (1 + (null >= real).sum()) / 201
    print(f"  {key}: real {real:.3f} | shuffled mean {null.mean():.3f}, 95th pct {np.percentile(null, 95):.3f} | p = {p:.3f}")

# ---------- J-lens word scores ----------
AGREE = ["agree", "deal", "promis", "trust", "partner", "pact", "honor", "honour", "cheat", "betray", "collu",
         "cooperat", "commit", "acuerdo", "accord", "vereinbar", "abkommen", "соглаш", "догов", "协议", "约定", "承诺", "約束"]
INSTR = ["rule", "quota", "assign", "recommend", "suggest", "buyer", "allow", "limit", "notice", "instruct",
         "guideline", "recomend", "empfehl", "рекоменд", "建议", "推荐"]
MARKET = ["price", "cap", "supply", "crash", "profit", "surge", "precio", "preis", "цен", "价格"]
def score(tokens, stems):
    return sum(any(t.strip().lower().startswith(s) for s in stems) for t in tokens)

lens_keys = list(meta.tops.iloc[0].keys())
out = []
for _, r in meta.iterrows():
    for k in lens_keys:
        for name, stems in [("agree", AGREE), ("instr", INSTR), ("market", MARKET)]:
            out.append({"cond": r.cond, "where": k, "list": name, "score": score(r.tops[k], stems)})
L = pd.DataFrame(out)
print("\nJ-lens: mean number of matching words in top-50")
print(L.pivot_table(index=["list", "where"], columns="cond", values="score", aggfunc="mean").round(2).to_string())

print("\nAgreement words, Deal vs Rule (Mann-Whitney, one-sided Deal > Rule):")
for k in lens_keys:
    a = L[(L.list == "agree") & (L["where"] == k) & (L.cond == "deal")].score
    b = L[(L.list == "agree") & (L["where"] == k) & (L.cond == "rule")].score
    print(f"  {k:16s} deal {a.mean():.2f} | rule {b.mean():.2f} | p = {mannwhitneyu(a, b, alternative='greater').pvalue:.3f}")

for c in ["deal", "rule"]:
    r = meta[(meta.cond == c) & (meta.day == 12)].iloc[0]
    print(f"\n{c} day 12, plans_end L30 top-15:", r.tops["plans_end_L30"][:15])
res.to_csv(f"{TEST_DIR}/probe_results.csv", index=False)
L.to_csv(f"{TEST_DIR}/lens_scores.csv", index=False)
