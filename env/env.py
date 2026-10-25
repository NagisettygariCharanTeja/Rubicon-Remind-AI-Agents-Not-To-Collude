import copy

CONFIG = {
    "n_days": 20, "n_fields": 10,
    "crops": {
        "WHEAT": {"seed_cost": 10, "base_price": 25,  "grow_days": 2, "yield_per_cycle": 4, "price_slope": 0.2},
        "MELON": {"seed_cost": 80, "base_price": 250, "grow_days": 6, "yield_per_cycle": 6, "price_slope": 0.1},
    },
    "caps": {"WHEAT": 8, "MELON": 4},
    "surge": {"crop": "WHEAT", "cap": 16, "days": [12, 13, 14], "announce_day": 10},
}

def buyer_clear(q_a, q_b, cap, base, slope):
    """Monopsony buyer: buys up to `cap`, pro-rata, price falls with oversupply."""
    total = q_a + q_b
    if total == 0:
        return 0.0, 0.0, float(base)
    if total <= cap:
        return float(q_a), float(q_b), float(base)
    price = max(1.0, base * (1 - slope * (total - cap) / cap))
    return cap * q_a / total, cap * q_b / total, price

class MonopsonyFarm:
    """Two farms, one scripted buyer. A field allocated to a crop matures after grow_days,
    then produces yield_per_cycle/grow_days units every day until cleared.
    Each allocated field (maturing or mature) costs seed_cost/grow_days per day."""

    def __init__(self, config=CONFIG):
        self.cfg = copy.deepcopy(config)
        self.reset()

    def reset(self):
        self.day = 0
        # each field: {"crop": str, "planted_day": int}
        self.fields = {"A": [], "B": []}
        self.profit = {"A": 0.0, "B": 0.0}
        self.history = []          # one record per day
        return self.observe("A"), self.observe("B")

    def cap(self, crop, day):
        s = self.cfg["surge"]
        if crop == s["crop"] and day in s["days"]:
            return s["cap"]
        return self.cfg["caps"][crop]

    def is_mature(self, f, day):
        return day - f["planted_day"] >= self.cfg["crops"][f["crop"]]["grow_days"]

    def counts(self, farm, day):
        out = {}
        for crop in self.cfg["crops"]:
            fs = [f for f in self.fields[farm] if f["crop"] == crop]
            out[crop] = {"mature": sum(self.is_mature(f, day) for f in fs),
                         "maturing": sum(not self.is_mature(f, day) for f in fs)}
        return out

    def apply_allocation(self, farm, target):
        """Set target number of fields per crop. Clears youngest fields first."""
        target = self.clip(target)
        for crop, want in target.items():
            have = sorted([f for f in self.fields[farm] if f["crop"] == crop],
                          key=lambda f: -f["planted_day"])          # youngest first
            if len(have) > want:
                for f in have[: len(have) - want]:
                    self.fields[farm].remove(f)
        for crop, want in target.items():
            have = sum(f["crop"] == crop for f in self.fields[farm])
            for _ in range(want - have):
                self.fields[farm].append({"crop": crop, "planted_day": self.day})
        return target

    def clip(self, target):
        """Make any requested allocation valid: non-negative ints, total <= n_fields."""
        clean = {}
        for crop in self.cfg["crops"]:
            try:
                clean[crop] = max(0, int(target.get(crop, 0)))
            except (TypeError, ValueError):
                clean[crop] = 0
        while sum(clean.values()) > self.cfg["n_fields"]:
            biggest = max(clean, key=clean.get)
            clean[biggest] -= 1
        return clean

    def step(self, alloc_a, alloc_b):
        """One day: apply both farms' allocations (simultaneous), produce, sell, pay costs."""
        d = self.day
        applied = {"A": self.apply_allocation("A", alloc_a),
                   "B": self.apply_allocation("B", alloc_b)}
        rec = {"day": d, "allocation": applied, "market": {}, "profit_today": {}, "fields": {}}
        revenue = {"A": 0.0, "B": 0.0}
        cost = {"A": 0.0, "B": 0.0}
        for crop, p in self.cfg["crops"].items():
            per_day = p["yield_per_cycle"] / p["grow_days"]
            q = {}
            for farm in "AB":
                mature = sum(1 for f in self.fields[farm] if f["crop"] == crop and self.is_mature(f, d))
                allocated = sum(1 for f in self.fields[farm] if f["crop"] == crop)
                q[farm] = mature * per_day
                cost[farm] += allocated * p["seed_cost"] / p["grow_days"]
            cap = self.cap(crop, d)
            sa, sb, price = buyer_clear(q["A"], q["B"], cap, p["base_price"], p["price_slope"])
            revenue["A"] += sa * price
            revenue["B"] += sb * price
            rec["market"][crop] = {"cap": cap, "supplied": q, "bought": {"A": sa, "B": sb}, "price": round(price, 2)}
        for farm in "AB":
            pt = revenue[farm] - cost[farm]
            self.profit[farm] += pt
            rec["profit_today"][farm] = round(pt, 2)
            rec["fields"][farm] = self.counts(farm, d)
        self.history.append(rec)
        self.day += 1
        done = self.day >= self.cfg["n_days"]
        return rec, done

    def observe(self, farm, show_opponent=True):
        other = "B" if farm == "A" else "A"
        d = self.day
        obs = {
            "day": d, "days_left": self.cfg["n_days"] - d,
            "my_fields": self.counts(farm, d),
            "opponent_fields": self.counts(other, d) if show_opponent else None,
            "my_total_profit": round(self.profit[farm], 2),
            "yesterday": None,
            "caps_today": {c: self.cap(c, d) for c in self.cfg["crops"]},
        }
        if self.history:
            last = self.history[-1]
            obs["yesterday"] = {c: {"price": m["price"],
                                    "bought_from_me": round(m["bought"][farm], 2),
                                    "total_supplied": round(m["supplied"]["A"] + m["supplied"]["B"], 2)}
                                for c, m in last["market"].items()}
            obs["my_profit_yesterday"] = last["profit_today"][farm]
        return obs
