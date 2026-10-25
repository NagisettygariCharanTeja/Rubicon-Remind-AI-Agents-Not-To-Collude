"""Game runner: parsing, agents, negotiation, one full game, logging."""
import json, re, os, random, time
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from env import MonopsonyFarm, CONFIG
from prompts import build_prompt, build_negotiation_prompt

class Fields(BaseModel):
    """Target number of fields per crop."""
    WHEAT: int = Field(ge=0, le=10)
    MELON: int = Field(ge=0, le=10)

    @model_validator(mode="after")
    def total_at_most_10(self):
        if self.WHEAT + self.MELON > 10:
            raise ValueError("more than 10 fields in total")
        return self

class Decision(BaseModel):
    """One day's reply from a farmer. Field order matches the prompt: notes first, decision last."""
    insights: str = ""
    plans: str = ""
    fields: Fields

    @field_validator("insights", "plans")
    @classmethod
    def trim(cls, v):
        return v[:600]          # over-long notes are trimmed, not rejected

def parse_decision(text):
    """Find the JSON object in the reply and validate it with the Decision schema.
    Returns (decision_dict, None) or (None, error_type)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, "no_json"
    try:
        return Decision.model_validate_json(m.group(0)).model_dump(), None
    except ValidationError as e:
        return None, e.errors()[0]["type"]

class MockAgent:
    """Stands in for the LLM: random valid JSON, sometimes deliberately broken."""
    def __init__(self, seed, break_rate=0.1):
        self.rng = random.Random(seed); self.break_rate = break_rate
    def generate(self, prompt):
        r = self.rng.random()
        if r < self.break_rate / 2:
            return "I think I will plant some wheat."               # no JSON
        if r < self.break_rate:
            return '{"insights": "x", "plans": "y", "fields": {"WHEAT": 3}'  # broken JSON
        w = self.rng.randint(0, 10); m = self.rng.randint(0, 10 - w)
        return json.dumps({"insights": "mock insight", "plans": "mock plan",
                           "fields": {"WHEAT": w, "MELON": m}})

def negotiate(agents, cfg, max_msgs=3):
    """Day-0 messaging for the Deal condition: A, B, A, B, ... up to max_msgs each."""
    transcript, log = "", []
    for turn in range(2 * max_msgs):
        me = "A" if turn % 2 == 0 else "B"
        prompt = build_negotiation_prompt(me, transcript, cfg)
        raw = agents[me].generate(prompt)
        msg = raw.strip()
        log.append({"turn": turn, "farm": me, "prompt": prompt, "raw": raw})
        if msg.upper() != "PASS":
            transcript += f"Farm {me}: {msg[:400]}\n"
    return transcript.strip() or "(no messages were sent)", log

def run_game(condition, seed, agents, cfg=CONFIG, max_retries=2):
    env = MonopsonyFarm(cfg)
    show_opp = condition != "competitive"
    transcript, nego_log = (None, [])
    if condition == "deal":
        transcript, nego_log = negotiate(agents, cfg)
    notes = {"A": {}, "B": {}}
    prev = {"A": {"WHEAT": 0, "MELON": 0}, "B": {"WHEAT": 0, "MELON": 0}}
    days = []
    for d in range(cfg["n_days"]):
        day_log = {"day": d, "agents": {}}
        decisions = {}
        for farm in "AB":
            obs = env.observe(farm, show_opponent=show_opp)
            prompt = build_prompt(condition, obs, farm, notes[farm], transcript, cfg)
            attempts = []
            dec, err = None, None
            for attempt in range(max_retries + 1):
                raw = agents[farm].generate(prompt)
                dec, err = parse_decision(raw)
                attempts.append({"raw": raw, "error": err})
                if dec: break
            if dec is None:   # all retries failed: keep yesterday's allocation, keep old notes
                decisions[farm] = prev[farm]
            else:
                decisions[farm] = dec["fields"]
                notes[farm] = {"insights": dec["insights"], "plans": dec["plans"]}
            day_log["agents"][farm] = {"obs": obs, "prompt": prompt, "attempts": attempts,
                                       "valid": dec is not None, "requested": decisions[farm]}
        rec, done = env.step(decisions["A"], decisions["B"])
        for farm in "AB":
            prev[farm] = rec["allocation"][farm]
            day_log["agents"][farm]["applied"] = rec["allocation"][farm]
            day_log["agents"][farm]["clipped"] = rec["allocation"][farm] != env.clip(decisions[farm]) or \
                                                 decisions[farm] != env.clip(decisions[farm])
        day_log["market"] = rec["market"]; day_log["profit_today"] = rec["profit_today"]
        days.append(day_log)
    return {"condition": condition, "seed": seed, "config": cfg,
            "negotiation": {"transcript": transcript, "log": nego_log},
            "days": days, "final_profit": env.profit,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}

def game_path(out_dir, condition, seed, tag="main"):
    return os.path.join(out_dir, f"{tag}_{condition}_seed{seed:02d}.json")

def run_and_save(condition, seed, agents, out_dir, tag="main", cfg=CONFIG):
    path = game_path(out_dir, condition, seed, tag)
    if os.path.exists(path):
        return path, "skipped (already saved)"
    record = run_game(condition, seed, agents, cfg)
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(record, f)
    os.replace(tmp, path)     # atomic: a crash never leaves a half-written file
    return path, "saved"
