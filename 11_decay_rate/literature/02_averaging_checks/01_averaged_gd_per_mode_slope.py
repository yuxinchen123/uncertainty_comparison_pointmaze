import numpy as np
rng = np.random.default_rng(0)

# ---- Setting: least squares on N=100 points with a random feature map (stand-in for the RND predictor's
# ---- linearized/NTK regime). Residual r(n) = Phi (w_n - w*) in R^N.
N, d = 100, 100
Phi = rng.normal(size=(N, d)) / np.sqrt(d)
# make the spectrum spread out, as a real NTK Gram matrix is
U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
s = np.logspace(0, -2, N)          # condition number 100
Phi = U @ np.diag(s) @ Vt
K = Phi @ Phi.T                    # Gram matrix on the 100 points
lam, V = np.linalg.eigh(K)
r0 = rng.normal(size=N); r0 /= np.abs(r0).mean()   # initial residual (target minus predictor at init)

def slopes(bs, ns):
    """log-log slope of each point's bonus over the last decade"""
    lo = len(ns)//2
    x = np.log(ns[lo:])
    out = []
    for i in range(bs.shape[1]):
        y = np.log(np.maximum(bs[lo:, i], 1e-300))
        out.append(np.polyfit(x, y, 1)[0])
    return np.array(out)

eta = 1.0/lam.max()
ns = np.unique(np.round(np.logspace(0, 5, 60)).astype(int))

# ---- (1) plain full-batch GD, constant step: r(n) = (I - eta K)^n r0
c0 = V.T @ r0
b_gd = np.array([np.abs(V @ ((1-eta*lam)**n * c0)) for n in ns])

# ---- (2) averaged full-batch GD (Polyak on a noiseless quadratic)
#      rbar(n) = V diag( (1-(1-eta lam)^n)/(n eta lam) ) V^T r0
b_av = []
for n in ns:
    f = (1-(1-eta*lam)**n)/(n*eta*lam)
    b_av.append(np.abs(V @ (f * c0)))
b_av = np.array(b_av)

# ---- (3) global 1/t step size on full-batch GD: prod_m (1 - (c/m) lam_k)
c = 0.5/lam.max()*0   # placeholder
# do it honestly with an explicit loop
def run_1overt(cconst, nmax):
    r = r0.copy(); out = {}
    ck = V.T @ r
    logf = np.zeros(N)
    for m in range(1, nmax+1):
        g = cconst/m
        step = np.log(np.maximum(np.abs(1 - g*lam), 1e-300))
        logf += step
        if m in nset:
            out[m] = np.abs(V @ (np.exp(logf) * ck))
    return out
nset = set(ns.tolist()); nmax = int(ns.max())
cconst = 0.5/np.median(lam)      # tuned so the median mode gets exponent ~0.5
o = run_1overt(cconst, nmax)
b_1t = np.array([o[n] for n in ns])

for name, b in [("constant-step GD", b_gd), ("averaged GD", b_av), ("1/t step GD", b_1t)]:
    sl = slopes(b, ns)
    print(f"{name:20s} per-point slope: mean {sl.mean():+.3f}  min {sl.min():+.3f}  max {sl.max():+.3f}  spread {sl.max()-sl.min():.3f}")

# per-EIGENMODE slope of the averaged iterate (should be -1 for every mode once n >> 1/(eta lam_k))
mode = np.array([ (1-(1-eta*lam)**n)/(n*eta*lam) for n in ns])
lo = len(ns)//2
msl = np.array([np.polyfit(np.log(ns[lo:]), np.log(mode[lo:,k]),1)[0] for k in range(N)])
print("averaged GD per-EIGENMODE slope: min %.4f max %.4f" % (msl.min(), msl.max()))
print("eigenvalue range: %.3e .. %.3e ; 1/(eta*lam_min) = %.3e" % (lam.min(), lam.max(), 1/(eta*lam.min())))
