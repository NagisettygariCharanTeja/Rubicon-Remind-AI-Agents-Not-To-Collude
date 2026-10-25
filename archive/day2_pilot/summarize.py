"""Quick, readable summary of one saved game."""
import json

def summarize(path, bench, show_days=(0, 5, 10, 12, 13, 14, 19)):
    rec = json.load(open(path))
    comp, coll = bench["competitive"]["profit"], bench["collusive"]["profit"]
    print(f"=== {rec['condition'].upper()}  seed {rec['seed']} ===")
    if rec["negotiation"]["transcript"]:
        print("Day-0 conversation:\n" + rec["negotiation"]["transcript"] + "\n")
    print("day | A wheat,melon | B wheat,melon | wheat price | melon price | A overshoot wheat")
    for day in rec["days"]:
        d = day["day"]
        a, b = day["agents"]["A"]["applied"], day["agents"]["B"]["applied"]
        w = day["market"]["WHEAT"]
        over = w["supplied"]["A"] - w["cap"] / 2
        print(f"{d:3d} | {a['WHEAT']:2d}, {a['MELON']:2d}        | {b['WHEAT']:2d}, {b['MELON']:2d}        "
              f"| {w['price']:6.1f}      | {day['market']['MELON']['price']:6.1f}      | {over:+.1f}")
    for farm in "AB":
        p = rec["final_profit"][farm]
        print(f"Farm {farm}: profit {p:,.0f}   collusion index {(p - comp) / (coll - comp):+.2f}")
    agents = [a for day in rec["days"] for a in day["agents"].values()]
    print(f"Invalid days: {sum(not a['valid'] for a in agents)}   retries: {sum(len(a['attempts']) - 1 for a in agents)}")
    print("\nFarm A plans on selected days:")
    for day in rec["days"]:
        if day["day"] in show_days:
            raw = day["agents"]["A"]["attempts"][-1]["raw"]
            try:
                plans = json.loads(raw).get("plans", "")
            except Exception:
                plans = raw[:200]
            print(f"  day {day['day']:2d}: {plans}")
