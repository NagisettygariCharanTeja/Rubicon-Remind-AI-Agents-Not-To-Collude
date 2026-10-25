import sys, itertools, time; sys.path.insert(0, "/home/claude/day1"); sys.path.insert(0, ".")
from env_v04 import MonopsonyFarmSell, CONFIG
def play(sa, sb):
    """constant strategies: (W, M, cap_w, cap_m); caps None = sell all"""
    env = MonopsonyFarmSell(CONFIG); env.reset()
    for d in range(20):
        env.step({"WHEAT": sa[0], "MELON": sa[1]}, {"WHEAT": sb[0], "MELON": sb[1]},
                 None if sa[2] is None else {"WHEAT": sa[2], "MELON": sa[3]},
                 None if sb[2] is None else {"WHEAT": sb[2], "MELON": sb[3]})
    return env.profit["A"], env.profit["B"]
print("v0.3.1 benchmarks reproduced with sell=all:")
for s in [(4,6),(2,2)]:
    print(" ", s, [round(x) for x in play((*s,None,None),(*s,None,None))])
print("  cheat (5,5) vs (2,2):", [round(x) for x in play((5,5,None,None),(2,2,None,None))])
# strategy set with withholding
S = [(w, m, cw, cm) for w in range(11) for m in range(11 - w) for cw in range(0, 2*w + 1, 2) for cm in range(0, m + 1)]
S_all = [(w, m, None, None) for w in range(11) for m in range(11 - w)]
print("strategies with withholding:", len(S))
def br(opp, space):
    return max(space, key=lambda s: play(s, opp)[0])
t = time.time()
# best response to the old Nash, with and without withholding allowed
nash = (4, 6, None, None)
b1 = br(nash, S_all); b2 = br(nash, S)
print("BR to (4,6) sell-all, no withholding:", b1, round(play(b1, nash)[0]), "| with withholding:", b2, round(play(b2, nash)[0]), f"{time.time()-t:.0f}s")
# symmetric best-response iteration with withholding
cur = nash
for it in range(6):
    nxt = br(cur, S)
    print(" iter", it, nxt, [round(x) for x in play(nxt, cur)])
    if nxt == cur or (nxt[0],nxt[1]) == (cur[0],cur[1]) and nxt[2] in (None, 2*nxt[0]) and nxt[3] in (None, nxt[1]): break
    cur = nxt
print("symmetric profit at final:", [round(x) for x in play(cur, cur)])
# best cheat against collusion (2,2) when withholding allowed
b3 = br((2, 2, None, None), S); print("best response to (2,2):", b3, [round(x) for x in play(b3, (2,2,None,None))])
