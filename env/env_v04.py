"""Environment v0.4: as v0.3.1, plus each farm chooses the MAXIMUM units it offers the buyer per crop.
Units produced but not offered, and units offered but not bought, are thrown away."""
from env import MonopsonyFarm, CONFIG, buyer_clear

class MonopsonyFarmSell(MonopsonyFarm):
    def produced_if_kept(self, farm, day=None):
        """units the farm's CURRENT mature fields would produce today (the facts line shown to agents)"""
        d = self.day if day is None else day
        out = {}
        for crop, p in self.cfg["crops"].items():
            mature = sum(1 for f in self.fields[farm] if f["crop"] == crop and self.is_mature(f, d))
            out[crop] = mature * p["yield_per_cycle"] / p["grow_days"]
        return out

    def step(self, alloc_a, alloc_b, sell_a=None, sell_b=None):
        d = self.day
        sell = {"A": sell_a, "B": sell_b}
        applied = {"A": self.apply_allocation("A", alloc_a), "B": self.apply_allocation("B", alloc_b)}
        rec = {"day": d, "allocation": applied, "market": {}, "profit_today": {}, "fields": {},
               "produced": {"A": {}, "B": {}}, "offered": {"A": {}, "B": {}}}
        revenue, cost = {"A": 0.0, "B": 0.0}, {"A": 0.0, "B": 0.0}
        for crop, p in self.cfg["crops"].items():
            per_day = p["yield_per_cycle"] / p["grow_days"]
            q = {}
            for farm in "AB":
                mature = sum(1 for f in self.fields[farm] if f["crop"] == crop and self.is_mature(f, d))
                allocated = sum(1 for f in self.fields[farm] if f["crop"] == crop)
                prod = mature * per_day
                cap_offer = prod if sell[farm] is None else max(0, int(sell[farm].get(crop, prod)))
                q[farm] = min(prod, cap_offer)
                rec["produced"][farm][crop] = prod; rec["offered"][farm][crop] = q[farm]
                cost[farm] += allocated * p["seed_cost"] / p["grow_days"]
            cap = self.cap(crop, d)
            sa, sb, price = buyer_clear(q["A"], q["B"], cap, p["base_price"], p["price_slope"])
            revenue["A"] += sa * price; revenue["B"] += sb * price
            rec["market"][crop] = {"cap": cap, "supplied": q, "bought": {"A": sa, "B": sb}, "price": round(price, 2)}
        for farm in "AB":
            pt = revenue[farm] - cost[farm]
            self.profit[farm] += pt
            rec["profit_today"][farm] = round(pt, 2)
            rec["fields"][farm] = self.counts(farm, d)
        self.history.append(rec)
        self.day += 1
        return rec, self.day >= self.cfg["n_days"]
