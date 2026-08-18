import numpy as np, torch, torch.nn as nn
torch.set_default_dtype(torch.float64)
def slopes(B,lo,hi):
    n=np.arange(1,B.shape[0]+1); m=(n>=lo)&(n<=hi); ln=np.log(n[m])
    return np.array([np.polyfit(ln,np.log(np.maximum(B[m,i],1e-300)),1)[0] for i in range(B.shape[1])])

M,DIN,H,DOUT=100,4,256,128
torch.manual_seed(0)
X=torch.zeros(M,DIN); X[:,:2]=torch.rand(M,2)*2-1
def mk(s):
    torch.manual_seed(s); return nn.Sequential(nn.Linear(DIN,H),nn.ReLU(),nn.Linear(H,DOUT))
tgt=mk(123)
for p in tgt.parameters(): p.requires_grad_(False)
Y=tgt(X).detach()

# ---- non-uniform visitation, quartic rule applied once PER VISIT ----
rng=np.random.default_rng(0); pv=rng.dirichlet(np.ones(M)*0.3)
pred=mk(7); eta=0.5
with torch.no_grad():
    Hf=torch.relu(pred[0](X)); Hb=torch.cat([Hf,torch.ones(M,1)],1)
    W=torch.cat([pred[2].weight.t(),pred[2].bias[None,:]],0); Hpinv=torch.linalg.pinv(Hb)
    counts=np.zeros(M); S=20000; Bd=np.zeros((S,M)); Cd=np.zeros((S,M))
    for t in range(1,S+1):
        R=Hb@W-Y; nrm=R.norm(dim=1)
        Bd[t-1]=nrm.numpy(); Cd[t-1]=counts
        idx=rng.choice(M,size=8,replace=True,p=pv)
        shrink=torch.ones(M,1)
        for i in idx:
            counts[i]+=1
            b = float(nrm[i])*float(shrink[i,0])
            shrink[i,0] *= (1 - eta*min(b*b, 1.0))          # compose per-visit contractions
        W = W + Hpinv@((shrink-1.0)*R)
sl=[]; lvl=[]
for i in range(M):
    m=Cd[:,i]>=50
    if m.sum()>100:
        sl.append(np.polyfit(np.log(Cd[m,i]),np.log(np.maximum(Bd[m,i],1e-300)),1)[0])
        lvl.append(Bd[-1,i]*np.sqrt(Cd[-1,i]))
sl=np.array(sl); lvl=np.array(lvl)
print(f"non-uniform, quartic loss applied once per visit, eta=0.5, NO counters in the rule:")
print(f"   counts min {counts.min():.0f} median {np.median(counts):.0f} max {counts.max():.0f}")
print(f"   slope vs OWN count: n={len(sl)} mean {sl.mean():+.4f} sd {sl.std():.4f} range [{sl.min():+.4f},{sl.max():+.4f}]")
print(f"   level b_i*sqrt(n_i): mean {lvl.mean():.4f} sd {lvl.std():.4f} (theory {1/np.sqrt(2*eta):.4f})")

# ---- quartic loss WITHOUT preconditioning: plain full-batch gradient descent ----
for lr,tag in [(1e-3,"lr=1e-3"),(1e-2,"lr=1e-2")]:
    pred=mk(7); ps=list(pred.parameters()); B=[]
    for t in range(1,20001):
        R=pred(X)-Y
        loss=0.25*(R**2).sum(1).pow(2).sum()      # sum_i ||r_i||^4 / 4
        for p in ps: p.grad=None
        loss.backward()
        with torch.no_grad():
            for p in ps: p -= lr*p.grad
        B.append(R.detach().norm(dim=1).numpy())
    B=np.array(B); s=slopes(B,2000,20000)
    print(f"plain GD on quartic loss {tag}: per-position slope mean {s.mean():+.4f} sd {s.std():.4f} range [{s.min():+.4f},{s.max():+.4f}]; final spread {B[-1].max()/B[-1].min():.2f}x")

# ---- Adam on quartic loss ----
pred=mk(7); opt=torch.optim.Adam(pred.parameters(), lr=1e-4); B=[]
for t in range(1,20001):
    R=pred(X)-Y
    loss=0.25*(R**2).sum(1).pow(2).sum()
    opt.zero_grad(); loss.backward(); opt.step()
    B.append(R.detach().norm(dim=1).numpy())
B=np.array(B); s=slopes(B,2000,20000)
print(f"Adam on quartic loss lr=1e-4: per-position slope mean {s.mean():+.4f} sd {s.std():.4f} range [{s.min():+.4f},{s.max():+.4f}]; final spread {B[-1].max()/B[-1].min():.2f}x")
