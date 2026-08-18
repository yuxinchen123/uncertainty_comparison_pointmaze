import numpy as np, torch, torch.nn as nn
torch.set_default_dtype(torch.float64)

def slopes(B, lo, hi):
    n=np.arange(1,B.shape[0]+1); m=(n>=lo)&(n<=hi); ln=np.log(n[m])
    return np.array([np.polyfit(ln, np.log(np.maximum(B[m,i],1e-300)),1)[0] for i in range(B.shape[1])])

# 0. pure decoupled recursion b_{n+1} = b_n (1 - eta b_n^2), several starting values
print("decoupled quartic-loss recursion  b_{n+1} = b_n (1 - eta b_n^2), eta=0.5")
N=20000
for b0 in [0.1, 1.0, 5.0, 50.0]:
    b=b0; hist=[]
    for t in range(N):
        hist.append(b); b = b*(1 - 0.5*min(b*b, 1.0))     # cap for stability at huge b0
    hist=np.array(hist)
    s=(np.log(hist[-1])-np.log(hist[N//2]))/(np.log(N)-np.log(N//2))
    print(f"   b0={b0:5.1f}: b(1e4)={hist[9999]:.6e}  b(2e4)={hist[-1]:.6e}  slope={s:+.4f}  theory (2*eta*n)^-1/2={1/np.sqrt(2*0.5*N):.6e}")

# 1. full-size RND toy, last-layer exact preconditioned solve, quartic loss, NO schedule
M,DIN,H,DOUT=100,4,256,128
torch.manual_seed(0)
X=torch.zeros(M,DIN); X[:,:2]=torch.rand(M,2)*2-1
def mk(s):
    torch.manual_seed(s); return nn.Sequential(nn.Linear(DIN,H),nn.ReLU(),nn.Linear(H,DOUT))
tgt=mk(123)
for p in tgt.parameters(): p.requires_grad_(False)
Y=tgt(X).detach()
pred=mk(7)
with torch.no_grad():
    Hf=torch.relu(pred[0](X)); Hb=torch.cat([Hf,torch.ones(M,1)],1)
    W=torch.cat([pred[2].weight.t(), pred[2].bias[None,:]],0)
    Hpinv=torch.linalg.pinv(Hb)
    B=[]; eta=0.5
    STEPS=20000
    for t in range(1,STEPS+1):
        R=Hb@W-Y; nrm=R.norm(dim=1,keepdim=True)
        B.append(nrm.squeeze(1).numpy())
        step = eta*torch.clamp(nrm**2, max=1.0)*R          # quartic-loss functional gradient, clipped
        W = W + Hpinv@(-step)
B=np.array(B); s=slopes(B,2000,STEPS)
print(f"\nlast-layer exact + quartic loss, NO schedule, no initial normalisation:")
print(f"   initial bonus range [{B[0].min():.3f},{B[0].max():.3f}]")
print(f"   per-position slope (n in [2e3,2e4]) mean {s.mean():+.4f} sd {s.std():.5f} range [{s.min():+.4f},{s.max():+.4f}]")
print(f"   bonus at n=2e4: min {B[-1].min():.6e} max {B[-1].max():.6e}  spread {B[-1].max()/B[-1].min():.4f}x ; theory {1/np.sqrt(2*eta*STEPS):.6e}")

# 2. same, but non-uniform visitation: only visited positions get updated
rng=np.random.default_rng(0); pv=rng.dirichlet(np.ones(M)*0.3)
pred=mk(7)
with torch.no_grad():
    Hf=torch.relu(pred[0](X)); Hb=torch.cat([Hf,torch.ones(M,1)],1)
    W=torch.cat([pred[2].weight.t(), pred[2].bias[None,:]],0); Hpinv=torch.linalg.pinv(Hb)
    counts=np.zeros(M); STEPS2=20000
    Bd=np.zeros((STEPS2,M)); Cd=np.zeros((STEPS2,M))
    for t in range(1,STEPS2+1):
        R=Hb@W-Y; nrm=R.norm(dim=1,keepdim=True)
        Bd[t-1]=nrm.squeeze(1).numpy(); Cd[t-1]=counts
        idx=rng.choice(M,size=8,replace=True,p=pv)
        mask=torch.zeros(M,1)
        for i in idx:
            counts[i]+=1; mask[i,0]+=1.0
        step = 0.5*mask*torch.clamp(nrm**2,max=1.0)*R
        W = W + Hpinv@(-step)
sl=[]
for i in range(M):
    m=Cd[:,i]>=50
    if m.sum()>100: sl.append(np.polyfit(np.log(Cd[m,i]), np.log(np.maximum(Bd[m,i],1e-300)),1)[0])
sl=np.array(sl)
fin = Bd[-1][Cd[-1]>=50]
print(f"\nnon-uniform visitation, quartic loss, NO per-position counters used in the rule:")
print(f"   counts: min {counts.min():.0f} median {np.median(counts):.0f} max {counts.max():.0f}")
print(f"   slope vs OWN count: n={len(sl)} mean {sl.mean():+.4f} sd {sl.std():.4f} range [{sl.min():+.4f},{sl.max():+.4f}]")
# check the level: b_i should equal (2*eta*n_i)^{-1/2}
ni = Cd[-1][Cd[-1]>=50]
pred_lvl = 1/np.sqrt(2*0.5*ni)
print(f"   level check b_i * sqrt(n_i): mean {np.mean(fin*np.sqrt(ni)):.4f} sd {np.std(fin*np.sqrt(ni)):.4f} (theory {1/np.sqrt(2*0.5):.4f})")
