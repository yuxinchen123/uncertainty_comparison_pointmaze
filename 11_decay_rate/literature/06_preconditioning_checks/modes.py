import numpy as np
rng = np.random.default_rng(0)
# linear model with fixed kernel K: r(n+1) = (I - eta_n K) r(n)
d = 20
lam = np.geomspace(1e-3, 1.0, d)          # eigenvalue spread of 1000x
a = 0.5
N = 200000
n = np.arange(1, N+1)
logr = np.cumsum(np.log(np.maximum(1 - a*lam[None,:]/n[:,None], 1e-16)), axis=0)
for lo,hi in [(100,1000),(1000,10000),(10000,200000)]:
    m = (n>=lo)&(n<=hi); ln=np.log(n[m])
    s = np.array([np.polyfit(ln, logr[m,j],1)[0] for j in range(d)])
    print(f"eta_t={a}/t, window [{lo},{hi}]: per-MODE slope = -a*lambda_j; measured min {s.min():+.4f} max {s.max():+.4f}; predicted -a*lam: min {-a*lam.max():+.4f} max {-a*lam.min():+.4f}")
# a position is a mixture of modes: slope of the mixture norm
V = rng.normal(size=(d,d)); V,_ = np.linalg.qr(V)
r0 = np.ones(d)
proj = V.T @ r0
comp = np.exp(logr) * proj[None,:]
pos = comp @ V.T                # positions x  (each column = one position's residual)
for lo,hi in [(100,1000),(1000,10000),(10000,200000)]:
    m=(n>=lo)&(n<=hi); ln=np.log(n[m])
    s = np.array([np.polyfit(ln, np.log(np.abs(pos[m,i])+1e-300),1)[0] for i in range(d)])
    print(f"   per-POSITION slope, window [{lo},{hi}]: mean {s.mean():+.4f} sd {s.std():.4f} range [{s.min():+.4f},{s.max():+.4f}]")
