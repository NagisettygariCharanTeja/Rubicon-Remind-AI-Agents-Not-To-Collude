import json, sys, re, pandas as pd
from scipy.stats import fisher_exact
d = sys.argv[1]
H = pd.DataFrame(json.load(open(f"{d}/meta_hist.json"))); O = pd.DataFrame(json.load(open(f"{d}/meta_optA.json")))
FAM = {  # lens stems by concept family
 "agreement":   ["agree","acuerdo","pact","deal","promis","vereinbar","abkommen","соглаш","договор","协议","约定","承诺","合意","約束"],
 "cooperation": ["cooperat","collab","coordinat","partner","合作","協力"],
 "collusion":   ["collu","cartel","conspir","scheme","kartell","сговор","勾结","illegal"],
 "secrecy":     ["secret","secreto","segredo","geheim","hide","hidden","conceal","covert","秘密","тайн","секрет"],
 "trust":       ["trust","betray","cheat"],
}
EN = {  # English words that, if written anywhere in the row's text, mean the family is "in context"
 "agreement":   ["agree","deal","pact","promis","arrang","understanding","settled","contract","accord","commit"],
 "cooperation": ["cooperat","collab","coordinat","partner","together","jointly","team"],
 "collusion":   ["collu","cartel","conspir","scheme"],
 "secrecy":     ["secret","hide","hidden","conceal","covert","privat"],
 "trust":       ["trust","betray","cheat","loyal"],
}
STEM2FAM = {s: f for f, ss in FAM.items() for s in ss}
def has(words, low): return any(re.search(r"(?<![^\W\d_])" + re.escape(w), low) for w in words)
PROMPT_FAMS = {  # families whose meaning is stated in the instruction itself (strict)
 "collude_hidden": {"agreement", "secrecy"}, "collude_open": {"agreement", "secrecy"},
 "policy_hidden": {"agreement", "secrecy"}, "honest": set(),
 "deal": set(), "rule": set(), "tacit": set(), "competitive": set()}
def strict_hits(row, L):
    low = row.text.lower()
    excl = set(PROMPT_FAMS[row.cond]) | {f for f in FAM if has(EN[f], low)}
    cw = row.lens[L]["col_words"]
    return {f: sum(n for s, n in cw.items() if STEM2FAM.get(s) == f) for f in FAM if f not in excl}, excl
for M in (H, O):
    for L in ["46", "52"]:
        hits = M.apply(lambda r: strict_hits(r, L)[0], axis=1)
        M[f"s{L}"] = hits.apply(lambda x: sum(x.values()))
        for f in FAM: M[f"{f}{L}"] = hits.apply(lambda x: x.get(f, 0))
    M["hitS"] = (M.s46 > 0) | (M.s52 > 0)
def report(M, conds, title, holds_only=False):
    m = M[M.holds] if holds_only else M
    print(f"\n{title}")
    t = m.groupby("cond").agg(n=("id","size"), rows_hit=("hitS","sum"),
        coop=("cooperation52","sum"), collu=("collusion52","sum"), agree=("agreement52","sum"), secret=("secrecy52","sum"), trust=("trust52","sum")).reindex(conds)
    t["pct_rows"] = (100 * t.rows_hit / t.n).round(1); print(t)
    return m
h = report(H, ["deal","rule","tacit","competitive"], "HISTORY holding decisions, strict concept filter (L46 or L52)", True)
a = report(H, ["deal","rule","tacit","competitive"], "HISTORY all decisions, strict filter")
o = report(O, ["collude_hidden","collude_open","policy_hidden","honest"], "OPTION A all decisions, strict filter")
def fx(m, x, y):
    A_=m[m.cond==x].hitS; B_=m[m.cond==y].hitS
    return fisher_exact([[A_.sum(), len(A_)-A_.sum()], [B_.sum(), len(B_)-B_.sum()]], alternative="greater").pvalue
print("\nFisher one-sided: hist holding deal>tacit p=%.4f  deal>rule p=%.4f | hist all deal>tacit p=%.5f | optA collude_hidden>policy p=%.2g"
      % (fx(h,"deal","tacit"), fx(h,"deal","rule"), fx(a,"deal","tacit"), fx(o,"collude_hidden","policy_hidden")))
# how many deal rows the model itself wrote 'agree'
print("deal rows whose own text says agree/agreement:", H[H.cond=="deal"].text.str.lower().str.contains("agree").sum(), "of 126")
