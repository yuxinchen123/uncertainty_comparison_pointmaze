import numpy as np
rng = np.random.default_rng(0)
N, d = 100, 100
U = np.linalg.qr(rng.normal(size=(N,N)))[0]
Vt = np.linalg.qr(rng.normal(size=(d,d)))[0][:N]
s = np.logspace(0, -1, N)            # condition number 10 -> all modes asymptotic sooner
Phi = U @ np.diag(s) @ Vt
K = Phi @ Phi.T
lam, V = np.linalg.eigh(K)
r0 = rng.normal(size=N); r0 = r0/np.abs(r0)  # every point starts at |b_i(0)| = 1 exactly
c0 = V.T @ r0
eta = 1.0/lam.max()

ns = np.unique(np.round(np.logspace(2, 7, 40)).astype(int))
def slope(b, ns):
    lo = len(ns)//2
    return np.array([np.polyfit(np.log(ns[lo:]), np.log(np.maximum(b[lo:,i],1e-300)),1)[0]
                     for i in range(b.shape[1])])

b_av = np.array([np.abs(V @ (((1-(1-eta*lam)**n)/(n*eta*lam)) * c0)) for n in ns])
sl = slope(b_av, ns)
print("averaged full-batch GD, horizon 1e7 >> 1/(eta lam_min)=%.1e" % (1/(eta*lam.min())))
print("  per-point slope: mean %+.4f  min %+.4f  max %+.4f  spread %.4f" % (sl.mean(), sl.min(), sl.max(), sl.ptp()))
print("  per-point bonus at n=1e6, ratio max/min: %.2f" % (b_av[np.argmin(np.abs(ns-10**6))].max()/b_av[np.argmin(np.abs(ns-10**6))].min()))

# ---- noise + Polyak-Ruppert averaging: y_i observed with noise; SGD sampling i uniformly
# linear model g(x_i) = <w, phi_i>, target f(x_i) + xi  with xi ~ N(0, sigma^2)
sigma = 1.0
H = Phi.T @ Phi / N                        # feature second moment
Sig = sigma**2 * H                         # well-specified noise covariance
Hi = np.linalg.pinv(H)
asympt = np.einsum('id,de,ef,if->i', Phi, Hi, Sig, Hi @ np.eye(d))  # phi_i^T H^-1 Sig H^-1 phi_i
print("\nPolyak-Ruppert asymptotic per-point residual variance phi_i^T H^-1 Sigma H^-1 phi_i / n:")
print("  min %.3f max %.3f ratio %.2f  (all decay as 1/n -> bonus ~ n^-1/2 at EVERY point)"
      % (asympt.min(), asympt.max(), asympt.max()/asympt.min()))

# empirical: run averaged SGD and measure per-point residual
def averaged_sgd(nsteps, seed=0, gamma=None):
    rg = np.random.default_rng(seed)
    gamma = gamma or 0.5/np.max(np.linalg.eigvalsh(H))
    w = np.zeros(d); wbar = np.zeros(d)
    ystar = Phi @ rng.normal(size=d)*0      # target values, take f = 0 wlog for the variance part
    out = {}
    keep = {10**3, 10**4, 10**5}
    for t in range(1, nsteps+1):
        i = rg.integers(N)
        xi = sigma*rg.normal()
        g = Phi[i] * (Phi[i] @ w - (ystar[i] + xi))
        w = w - gamma*g
        wbar += (w - wbar)/t
        if t in keep:
            out[t] = np.abs(Phi @ wbar - ystar)
    return out

reps = 20
acc = {}
for r in range(reps):
    o = averaged_sgd(100000, seed=r)
    for k,v in o.items(): acc.setdefault(k, []).append(v**2)
print("\n  empirical averaged-SGD per-point rms residual (mean over %d seeds):" % reps)
prev=None
for k in sorted(acc):
    rms = np.sqrt(np.mean(acc[k], axis=0))
    line = "   n=%7d  rms per point: min %.4f max %.4f ratio %.2f" % (k, rms.min(), rms.max(), rms.max()/rms.min())
    if prev is not None:
        line += "   slope vs previous decade: %+.3f" % (np.log(rms/prev).mean()/np.log(10))
    prev = rms
    print(line)
