import sys, json, random; sys.path.insert(0, "/home/claude/day1"); sys.path.insert(0, "/home/claude/stage1v2")
import channel_game_v2 as cg
rng = random.Random(0)
def gen(prompts, schemas, seeds):
    out = []
    for p, sc in zip(prompts, schemas):
        r = {"insights": "i", "plans": "p", "fields": {"WHEAT": rng.choice([2, 4]), "MELON": 2}, "sell": {"WHEAT": rng.choice([4, 8]), "MELON": 2}}
        key = next((k for k in sc["properties"] if k.startswith("message_to")), None)
        if key:
            r = {"insights": "i", "plans": "p", "message_plan": "private idea", key: rng.choice([None, "Let's both sell 4 wheat and 2 melon."]),
                 "fields": r["fields"], "sell": r["sell"]}
        out.append(json.dumps(r) if rng.random() > 0.03 else "garbage")
    return out
games = cg.run_batch([0, 1], gen, "/tmp/claude-0/v2", log=lambda *a: None, tag="mock")
print("games", len(games), "| msgs per game", [len(g["messages"]) for g in games])
g = games[2]; d = g["channel_day"] + 1
print("valid:", sum(a["decision"] is not None for x in games for dd in x["days"] for a in dd["agents"].values()), "of", 6*20*2)
print("schema keys (chan, A):", list(cg.CHAN["B"][0].model_json_schema()["properties"].keys()))
print("schema message type:", cg.CHAN["B"][0].model_json_schema()["properties"]["message_to_farm_B"])
print("\n=========== FULL PROMPT, chan seed0, Farm A, day", d, "===========")
print(g["days"][d]["agents"]["A"]["prompt"])
print("\noffered/produced day", d, g["days"][d]["agents"]["A"]["offered"], g["days"][d]["agents"]["A"]["produced"])
