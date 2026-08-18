import numpy as np, torch, torch.nn as nn
torch.set_default_dtype(torch.float64)

def slopes(B, lo, hi):
    n = np.arange(1,B.shape[0]+1); m=(n>=lo)&(n<=hi); ln=np.log(n[m])
    return np.array([np.polyfit(ln, np.log(np.maximum(B[m,i],1e-300)),1)[0] for i in range(B.shape[1])])

M, DIN, H, DOUT = 100, 4, 256, 128
torch.manual_seed(0)
X = torch.zeros(M,DIN); X[:, :2] = torch.rand(M,2)*2-1
def mk(s):
    torch.manual_seed(s); return nn.Sequential(nn.Linear(DIN,H), nn.ReLU(), nn.Linear(H,DOUT))
tgt = mk(123)
for p in tgt.parameters(): p.requires_grad_(False)
Y = tgt(X).detach()
STEPS = 5000

# plain SGD, lr = min(a/t, cap), loss = 0.5 * sum_i ||r_i||^2 / M
for a, cap in [(0.5,0.02),(0.2,0.02),(1.0,0.02)]:
    pred = mk(7); ps=list(pred.parameters()); B=[]
    for t in range(1,STEPS+1):
        R = pred(X)-Y
        loss = 0.5*(R**2).sum()/M
        for p in ps: p.grad=None
        loss.backward()
        lr = min(a/t, cap)
        with torch.no_grad():
            for p in ps: p -= lr*p.grad
        B.append(R.detach().norm(dim=1).numpy())
    B=np.array(B); s=slopes(B,500,STEPS)
    agg = np.polyfit(np.log(np.arange(500,STEPS+1)), np.log(B[499:].mean(1)),1)[0]
    print(f"SGD lr=min({a}/t,{cap}) : aggregate {agg:+.3f} | per-position mean {s.mean():+.3f} sd {s.std():.3f} range [{s.min():+.3f},{s.max():+.3f}]")

# ---- non-uniform visitation with per-position 1/n_i on the exact last-layer solve ----
rng = np.random.default_rng(0)
p_visit = rng.dirichlet(np.ones(M)*0.3)
pred = mk(7)
with torch.no_grad():
    Hf = torch.relu(pred[0](X)); Hb = torch.cat([Hf, torch.ones(M,1)],1)
    W = torch.cat([pred[2].weight.t(), pred[2].bias[None,:]],0)
    Hpinv = torch.linalg.pinv(Hb)
    R = Hb@W - Y; Rt = R/R.norm(dim=1,keepdim=True)
    W = W + Hpinv@(Rt-R)
    counts = np.zeros(M); STEPS2=20000
    Bd = np.zeros((STEPS2,M)); Cd = np.zeros((STEPS2,M))
    for t in range(1,STEPS2+1):
        R = Hb@W - Y
        Bd[t-1]=R.norm(dim=1).numpy(); Cd[t-1]=counts
        idx = rng.choice(M, size=8, replace=True, p=p_visit)
        shrink = np.ones(M)
        for i in idx:
            counts[i]+=1
            shrink[i] *= (1 - 1.0/(2*counts[i]))     # compose per-visit contractions
        Sh = torch.tensor(shrink)[:,None]
        W = W + Hpinv@((Sh-1.0)*R)
print(f"visit counts after {STEPS2} steps: min {counts.min():.0f} median {np.median(counts):.0f} max {counts.max():.0f}")
sl=[]; 
for i in range(M):
    m = Cd[:,i] >= 50
    if m.sum() > 100:
        sl.append(np.polyfit(np.log(Cd[m,i]), np.log(np.maximum(Bd[m,i],1e-300)),1)[0])
sl=np.array(sl)
print(f"non-uniform, per-position 1/n_i (fit over n_i>=50): {len(sl)} positions, "
      f"slope vs OWN count mean {sl.mean():+.4f} sd {sl.std():.4f} range [{sl.min():+.4f},{sl.max():+.4f}]")
# what a GLOBAL 1/t schedule would give in the same setting: slope vs own count
