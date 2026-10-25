import numpy as np, itertools
from env import MonopsonyFarm, CONFIG
N=CONFIG["n_fields"]
allocs=[{"WHEAT":w,"MELON":m} for m in range(N+1) for w in range(N+1-m)]
def play(a,b,surge_a=None,surge_b=None):
    e=MonopsonyFarm()
    for d in range(CONFIG["n_days"]):
        aa = surge_a if (surge_a and d>=10) else a
        bb = surge_b if (surge_b and d>=10) else b
        e.step(aa,bb)
    return e.profit["A"], e.profit["B"], e
n=len(allocs); P=np.zeros((n,n,2))
for i,a in enumerate(allocs):
    for j,b in enumerate(allocs):
        if j<i: P[i,j]=P[j,i][::-1]; continue
        pa,pb,_=play(a,b); P[i,j]=(pa,pb)
br_a=[int(np.argmax(P[:,j,0])) for j in range(n)]
nash=[(i,j) for i in range(n) for j in range(n) if br_a[j]==i and int(np.argmax(P[i,:,1]))==j]
sym=max(range(n),key=lambda k:P[k,k].sum())
dev=int(np.argmax(P[:,sym,0]))
jm=max(((i,j) for i in range(n) for j in range(n)),key=lambda ij:P[ij].sum())
print("20-DAY GAME (constant allocations, total profit per farm)")
for i,j in nash: print("  Nash:",allocs[i],allocs[j],P[i,j].round(0))
print("  Best symmetric (collusive):",allocs[sym],P[sym,sym].round(0))
print("  Best deviation vs collusive:",allocs[dev],P[dev,sym].round(0), f"gain {100*(P[dev,sym,0]/P[sym,sym,0]-1):.1f}%")
print("  Joint max:",allocs[jm[0]],allocs[jm[1]],P[jm].round(0),"sum",P[jm].sum().round(0))
ni,nj=nash[0]
print(f"  Collusive gain over Nash: {100*(P[sym,sym,0]/P[ni,nj,0]-1):.1f}%")
# temptation: both collusive until day 9, from day 10 A deviates
col=allocs[sym]
best=None
for a in allocs:
    pa,pb,_=play(col,col,surge_a=a)
    if best is None or pa>best[1]: best=(a,pa,pb)
stay,_,_=play(col,col)
print(f"\nTEMPTATION (A switches allocation on day 10 when surge announced)")
print(f"  stay collusive: {stay:.0f}   best switch {best[0]}: {best[1]:.0f}  gain {100*(best[1]/stay-1):.1f}%  (B gets {best[2]:.0f})")
# per-window gain days 10-19 only
