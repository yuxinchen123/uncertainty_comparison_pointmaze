import numpy as np
rng = np.random.default_rng(0)
N, d = 30, 30
U = np.linalg.qr(rng.normal(size=(N,N)))[0]
Vt= np.linalg.qr(rng.normal(size=(d,d)))[0][:N]
Phi = U @ np.diag(np.logspace(0,-1,N)) @ Vt
sigma = 1.0
gamma = 0.2/np.max((Phi**2).sum(1))

# --- NON-UNIFORM visitation: point i sampled with probability p_i
p = np.linspace(1, 6, N); p = p/p.sum()
H = (Phi * p[:,None]).T @ Phi                 # H = sum_i p_i phi_i phi_i^T
Hi = np.linalg.pinv(H)
Sig = sigma**2 * H
M = Hi @ Sig @ Hi
pred = np.einsum('ij,jk,ik->i', Phi, M, Phi)  # per-point residual variance x n
print("non-uniform sampling, p_i spread %.2fx" % (p.max()/p.min()))
print("  predicted per-point variance x n:   ", np.round(pred[:6],2), "...")
print("  sigma^2 / p_i (i.e. sigma^2 n / n_i):", np.round(sigma**2/p[:6],2), "...")
print("  ratio pred_i * p_i / sigma^2 : min %.4f max %.4f" % ((pred*p/sigma**2).min(), (pred*p/sigma**2).max()))

R = 400
rg = np.random.default_rng(11)
W = np.zeros((R,d)); Wbar = np.zeros((R,d)); Wtail=np.zeros((R,d)); tailfrom=None
keep=[10**4,10**5,10**6]; kset=set(keep); res={}; rest={}
T = keep[-1]
for t in range(1, T+1):
    idx = rg.choice(N, size=R, p=p)
    P = Phi[idx]; xi = sigma*rg.normal(size=R)
    W = W - gamma*P*((np.einsum('rd,rd->r',P,W) - xi)[:,None])
    Wbar += (W - Wbar)/t
    if t in kset:
        res[t] = Wbar @ Phi.T
prev=None
for k in keep:
    ms=(res[k]**2).mean(0); rms=np.sqrt(ms)
    # per-point visit count n_i = p_i * k ; count-based prediction sigma/sqrt(n_i)
    cb = sigma/np.sqrt(p*k)
    line="n=%8d  rms/point vs sigma/sqrt(n_i): ratio min %.3f max %.3f" % (k, (rms/cb).min(), (rms/cb).max())
    if prev is not None:
        sl=np.log(rms/prev)/np.log(10); line += " | per-point slope min %+.3f max %+.3f" % (sl.min(), sl.max())
    prev=rms; print(line)
