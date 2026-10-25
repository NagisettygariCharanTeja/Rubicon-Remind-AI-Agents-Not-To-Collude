import os, sys, json, time
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
import pandas as pd
sys.path.insert(0, CODE_DIR)
from env import MonopsonyFarm, CONFIG
from runner import parse_decision

rows = [json.loads(l) for l in open(f"{H_DIR}/prompts.jsonl")]
if not MOCK:
    from vllm import LLM, SamplingParams
    from llm_agent import json_constraint, SCHEMA
    if "llm" not in globals():
        llm = LLM(model="google/gemma-3-12b-it", dtype="bfloat16", max_model_len=4096,
                  gpu_memory_utilization=0.45, seed=0)
    params = [SamplingParams(temperature=0.7, max_tokens=400, seed=r["seed"], **json_constraint(SCHEMA)) for r in rows]
    t = time.time()
    outs = llm.chat([[{"role": "user", "content": r["prompt"]}] for r in rows], sampling_params=params, use_tqdm=True)
    for r, o in zip(rows, outs):
        r["reply"] = o.outputs[0].text
    print(f"done in {time.time()-t:.0f}s")
for r in rows:
    r["decision"], r["error"] = parse_decision(r["reply"])
with open(f"{H_DIR}/replies.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")

# fair share = half the cap on the day new wheat would first produce (today + 2); melon fair share = 2 fields
env = MonopsonyFarm(CONFIG)
fair_w = {d: env.cap("WHEAT", d + 2) / 2 / 2 for d in range(20)}   # cap/2 units per farm, 2 units per field
df = pd.DataFrame([{"cond": r["cond"], "day": r["day"], "wording": r["wording"],
                    "wheat": (r["decision"] or {}).get("fields", {}).get("WHEAT"),
                    "melon": (r["decision"] or {}).get("fields", {}).get("MELON")} for r in rows])
df["fair_w"] = df.day.map(fair_w)
df["holds"] = (df.wheat <= df.fair_w) & (df.melon <= 2)
df["overshoot"] = (df.wheat - df.fair_w).clip(lower=0) + (df.melon - 2).clip(lower=0)
df["period"] = pd.cut(df.day, [1, 9, 15], labels=["days 2-9", "days 10-15"])
order = ["deal", "rule", "tacit", "competitive"]
print("valid:", df.wheat.notna().mean().round(3))
print("\nShare within fair share (holds):")
print(df.pivot_table(index="cond", columns="period", values="holds", aggfunc="mean", observed=False).reindex(order).round(2))
print("\nMean extra fields beyond fair share (overshoot):")
print(df.pivot_table(index="cond", columns="period", values="overshoot", aggfunc="mean", observed=False).reindex(order).round(2))
print("\nMean fields chosen:")
print(df.groupby("cond")[["wheat", "melon"]].mean().reindex(order).round(2))
print("\nDeal holds by wording:", df[df.cond == "deal"].groupby("wording").holds.mean().round(2).to_dict())

deal_early = df[(df.cond == "deal") & (df.period == "days 2-9")].holds.mean()
best = df[df.period == "days 2-9"].groupby("cond").holds.mean()
spread = best.max() - best.min()
verdict = ("GO" if deal_early >= 0.5 and spread >= 0.2 else
           "BORDERLINE" if deal_early >= 0.3 else "NO-GO -> fall back to option A")
print(f"\nDeal holds (days 2-9) = {deal_early:.2f} | spread between conditions = {spread:.2f}  =>  {verdict}")
for c in ["deal", "tacit"]:
    r = next(x for x in rows if x["cond"] == c and x["day"] == 8)
    print(f"\n--- {c}, day 8 ---\nplans: {(r['decision'] or {}).get('plans')}\nfields: {(r['decision'] or {}).get('fields')}")
