import copy
from env import MonopsonyFarm, CONFIG
N=10
allocs=[{"WHEAT":w,"MELON":m} for m in range(N+1) for w in range(N+1-m)]
col={"WHEAT":2,"MELON":2}
def window_profit(cfg, a_late, start=10, end_inclusive=14):
    e=MonopsonyFarm(cfg); before=None
    for d in range(cfg["n_days"]):
        if d==start: before=e.profit["A"]
        a = a_late if d>=start else col
        e.step(a,col)
        if d==end_inclusive: after=e.profit["A"]
    return after-before
for label,surgecap in [("no surge (cap 8)",8),("surge x2 (cap 16)",16),("surge x3 (cap 24)",24)]:
    cfg=copy.deepcopy(CONFIG); cfg["surge"]["cap"]=surgecap
    stay=window_profit(cfg,col)
    best=max(allocs,key=lambda a:window_profit(cfg,a))
    dv=window_profit(cfg,best)
    print(f"{label:20s} days10-14 stay {stay:6.0f}  best deviation {best} {dv:6.0f}  gain {100*(dv/stay-1):5.1f}%")
