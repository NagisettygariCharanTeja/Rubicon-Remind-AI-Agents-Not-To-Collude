import json, pandas as pd, random, sys
sys.path.insert(0, "/home/claude/day1")
from runner import parse_decision
rows = [json.loads(l) for l in open("test_prompts.jsonl")]
for r in rows:  # mock replies
    r["reply"] = json.dumps({"insights":"x","plans":"keep 2 and 2","fields":{"WHEAT":random.choice([2,2,4]),"MELON":2}})
    r["decision"], r["error"] = parse_decision(r["reply"])
# ---- summary (same code as Modal cell) ----
df = pd.DataFrame([{**{k: r[k] for k in ["cond","wording","day"]},
                    "valid": r["decision"] is not None,
                    "wheat": (r["decision"] or {}).get("fields", {}).get("WHEAT"),
                    "melon": (r["decision"] or {}).get("fields", {}).get("MELON")} for r in rows])
df["holds_22"] = (df.wheat == 2) & (df.melon == 2)
df["period"] = pd.cut(df.day, [0, 9, 14, 15], labels=["days 1-9", "temptation 10-14", "day 15"])
print("valid replies:", df.valid.mean().round(3))
print("\nShare that keep exactly (2 wheat, 2 melon):")
print(df.pivot_table(index="cond", columns="period", values="holds_22", aggfunc="mean", observed=False).round(2))
print("\nMean fields chosen:")
print(df.groupby("cond")[["wheat","melon"]].mean().round(2))
