import torch, numpy as np
torch.manual_seed(0); torch.set_num_threads(8)
def net(): return torch.nn.Sequential(torch.nn.Linear(4,256), torch.nn.ReLU(), torch.nn.Linear(256,128))
gx,gy=torch.meshgrid(torch.linspace(-1,1,10),torch.linspace(-1,1,10),indexing='ij')
X=torch.stack([gx.reshape(-1),gy.reshape(-1),torch.zeros(100),torch.zeros(100)],1)
f=net(); [p.requires_grad_(False) for p in f.parameters()]
with torch.no_grad(): F=f(X)

# ---- how ill-conditioned is the tangent-kernel Gram matrix on these 100 points? ----
g0=net()
J=[]
for i in range(100):
    out=g0(X[i:i+1]).sum()
    gr=torch.autograd.grad(out, list(g0.parameters()), retain_graph=True)
    J.append(torch.cat([a.reshape(-1) for a in gr]))
J=torch.stack(J)                      # (100, P) for the summed-output direction
K=(J@J.T).double().numpy()
w=np.linalg.eigvalsh(K)
print("tangent-kernel Gram matrix on the 100 grid points (summed-output direction):")
print("  eigenvalues: max %.3e  median %.3e  min %.3e  condition number %.3e" % (w[-1], np.median(w), w[0], w[-1]/max(w[0],1e-300)))

def bonus(g):
    with torch.no_grad(): return (g(X)-F).norm(dim=1)

# ---- METHOD D: residual-shrinkage targets, per-point rate c / n_i ----
# outer step n: target_i = F_i + (1 - c/n_i) * (g(x_i) - F_i);  inner fit with Adam
def method_d(c=0.5, outer=400, inner=150, lr=1e-3, p=None, seed=0):
    torch.manual_seed(100+seed)
    g=net(); opt=torch.optim.Adam(g.parameters(), lr=lr)
    counts=np.zeros(100); ns=[]; hist=[]; cnts=[]
    rng=np.random.default_rng(seed)
    b0=bonus(g).numpy().copy()
    for n in range(1, outer+1):
        if p is None: S=np.arange(100)
        else:         S=np.unique(rng.choice(100, size=100, p=p))
        counts[S]+=1
        with torch.no_grad():
            cur=g(X)
            shrink=torch.ones(100)
            shrink[S]=torch.tensor(1.0-c/counts[S], dtype=torch.float32)
            T=F + shrink[:,None]*(cur-F)
        for _ in range(inner):
            opt.zero_grad(); ((g(X)-T)**2).sum(1).mean().backward(); opt.step()
        if n in {1,2,5,10,20,50,100,200,400}:
            ns.append(n); hist.append(bonus(g).numpy().copy()); cnts.append(counts.copy())
    return np.array(ns), np.array(hist), np.array(cnts), b0

pnu=np.linspace(1,6,100); pnu=pnu/pnu.sum()
for label, p in [("uniform visitation", None), ("non-uniform visitation, 6x spread", pnu)]:
    ns,H,C,b0 = method_d(p=p)
    B=H/b0
    lo=len(ns)//2
    sl=np.array([np.polyfit(np.log(ns[lo:]), np.log(np.maximum(B[lo:,i],1e-30)),1)[0] for i in range(100)])
    print("\nMETHOD D (residual-shrinkage targets, c=0.5), %s" % label)
    print("  per-position slope vs n: mean %+.4f min %+.4f max %+.4f spread %.4f" % (sl.mean(), sl.min(), sl.max(), sl.ptp()))
    print("  bonus at the last outer step: min %.4f max %.4f ratio %.3f   (ideal 1/sqrt(400)=%.4f)"
          % (B[-1].min(), B[-1].max(), B[-1].max()/B[-1].min(), 400**-0.5))
    ni=np.maximum(C[-1],1.0)
    r=B[-1]*np.sqrt(ni)
    print("  b_i * sqrt(n_i) at the end (should be constant if b_i = n_i^-1/2): min %.3f max %.3f ratio %.2f" % (r.min(), r.max(), r.max()/r.min()))
    slc=np.array([np.polyfit(np.log(np.maximum(C[lo:,i],1.0)), np.log(np.maximum(B[lo:,i],1e-30)),1)[0] for i in range(100)])
    print("  per-position slope vs its OWN visit count n_i: mean %+.4f min %+.4f max %+.4f" % (slc.mean(), slc.min(), slc.max()))
