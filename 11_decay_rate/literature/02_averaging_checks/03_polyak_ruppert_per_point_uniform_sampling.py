import numpy as np
rng = np.random.default_rng(0)
N, d = 40, 40
U = np.linalg.qr(rng.normal(size=(N,N)))[0]
Vt = np.linalg.qr(rng.normal(size=(d,d)))[0][:N]
Phi = U @ np.diag(np.logspace(0,-1,N)) @ Vt
H  = Phi.T @ Phi / N; Hi = np.linalg.pinv(H); sigma=1.0
M  = Hi @ (sigma**2*H) @ Hi
pred = np.einsum('ij,jk,ik->i', Phi, M, Phi)
print("predicted per-point residual variance x n: min %.4f max %.4f ratio %.2f (mean %.4f, sigma^2 d/N=%.4f)"
      % (pred.min(), pred.max(), pred.max()/pred.min(), pred.mean(), sigma**2*d/N))

R = 200                      # parallel replicas
gamma = 0.2/np.max((Phi**2).sum(1))
rg = np.random.default_rng(7)
W = np.zeros((R,d)); Wbar = np.zeros((R,d))
keep = [10**3, 10**4, 10**5]; kset=set(keep); res={}
for t in range(1, keep[-1]+1):
    idx = rg.integers(N, size=R)
    P = Phi[idx]                                  # (R,d)
    xi = sigma*rg.normal(size=R)
    W = W - gamma*P*((np.einsum('rd,rd->r', P, W) - xi)[:,None])
    Wbar += (W - Wbar)/t
    if t in kset: res[t] = Wbar @ Phi.T           # (R,N) residuals
prev=None
for k in keep:
    ms = (res[k]**2).mean(0); rms = np.sqrt(ms)
    line = "n=%7d rms/point min %.5f max %.5f ratio %.2f | (n*var)/pred min %.2f max %.2f" % (
        k, rms.min(), rms.max(), rms.max()/rms.min(), (k*ms/pred).min(), (k*ms/pred).max())
    if prev is not None:
        sl = np.log(rms/prev)/np.log(10)
        line += " | per-point slope this decade min %+.3f max %+.3f" % (sl.min(), sl.max())
    prev = rms; print(line)
