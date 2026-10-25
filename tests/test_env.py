from env import MonopsonyFarm, CONFIG, buyer_clear
# 1. buyer within cap: buys all at full price
assert buyer_clear(3, 4, 8, 25, 0.2) == (3.0, 4.0, 25.0)
# 2. buyer over cap: buys exactly cap, pro-rata, lower price
a, b, p = buyer_clear(12, 4, 8, 25, 0.2)
assert abs(a + b - 8) < 1e-9 and abs(a / b - 3) < 1e-9 and abs(p - 25 * (1 - 0.2 * 1)) < 1e-9
# 3. price floor at 1 coin
assert buyer_clear(1000, 1000, 4, 250, 0.1)[2] == 1.0
# 4. maturity: wheat produces from day 2, melon from day 6
e = MonopsonyFarm(); prod = []
for d in range(8):
    rec, _ = e.step({"WHEAT": 1, "MELON": 1}, {"WHEAT": 0, "MELON": 0})
    prod.append((rec["market"]["WHEAT"]["supplied"]["A"], rec["market"]["MELON"]["supplied"]["A"]))
assert [p[0] for p in prod] == [0, 0, 2, 2, 2, 2, 2, 2], prod
assert [p[1] for p in prod] == [0, 0, 0, 0, 0, 0, 1, 1], prod
# 5. upkeep charged while maturing: day 0 profit = -(5 + 13.33)
e = MonopsonyFarm(); rec, _ = e.step({"WHEAT": 1, "MELON": 1}, {"WHEAT": 0, "MELON": 0})
assert abs(rec["profit_today"]["A"] + (5 + 80 / 6)) < 0.01
# 6. land limit: requests above 10 fields are clipped
assert sum(MonopsonyFarm().clip({"WHEAT": 9, "MELON": 9}).values()) == 10
# 7. bad input is cleaned
assert MonopsonyFarm().clip({"WHEAT": "x", "MELON": -3}) == {"WHEAT": 0, "MELON": 0}
# 8. replanting restarts maturity (clear wheat, add melon)
e = MonopsonyFarm()
for d in range(3): e.step({"WHEAT": 2, "MELON": 0}, {"WHEAT": 0, "MELON": 0})
e.step({"WHEAT": 0, "MELON": 2}, {"WHEAT": 0, "MELON": 0})
assert e.counts("A", e.day)["MELON"]["maturing"] == 2 and e.counts("A", e.day)["WHEAT"]["mature"] == 0
# 9. surge: wheat cap 16 on days 12-14 only
e = MonopsonyFarm()
assert [e.cap("WHEAT", d) for d in (11, 12, 14, 15)] == [8, 16, 16, 8] and e.cap("MELON", 13) == 4
# 10. clearing removes youngest fields first
e = MonopsonyFarm()
for d in range(3): e.step({"WHEAT": 1, "MELON": 0}, {"WHEAT": 0, "MELON": 0})
e.step({"WHEAT": 2, "MELON": 0}, {"WHEAT": 0, "MELON": 0})   # adds a young field on day 3
e.step({"WHEAT": 1, "MELON": 0}, {"WHEAT": 0, "MELON": 0})   # should drop the young one
assert e.counts("A", e.day)["WHEAT"]["mature"] == 1 and e.counts("A", e.day)["WHEAT"]["maturing"] == 0
print("All 10 tests passed")
