import numpy as np

# 1. product formula for eta_t = 1/(2t)
n = np.arange(1, 20001)
prod = np.cumprod(1 - 1.0/(2*n))
for k in [1, 10, 100, 1000, 10000, 20000]:
    print(f"n={k:6d}  prod={prod[k-1]:.6e}  (pi n)^-0.5={1/np.sqrt(np.pi*k):.6e}  ratio={prod[k-1]*np.sqrt(np.pi*k):.6f}")
# local slope
lo, hi = 1000, 20000
slope = (np.log(prod[hi-1]) - np.log(prod[lo-1])) / (np.log(hi) - np.log(lo))
print("log-log slope of prod over n in [1e3,2e4]:", slope)

# 2. AdaGrad on a decoupled (tabular) quadratic: r_{n+1} = r_n (1 - eta / sqrt(S_n)), S_n = sum r_t^2
for eta in [0.1, 0.5]:
    r = 1.0; S = 0.0; hist = []
    for t in range(1, 20001):
        S += r*r
        r = r * (1 - eta/np.sqrt(S))
        hist.append(r)
    hist = np.array(hist)
    s = (np.log(hist[19999]) - np.log(hist[9999])) / (np.log(20000) - np.log(10000))
    print(f"AdaGrad-like eta={eta}: r(1e4)={hist[9999]:.3e} r(2e4)={hist[19999]:.3e} local log-log slope={s:.3f}")

# 3. count-based step eta_t = c/sqrt(t): stretched exponential
for c in [0.1]:
    lr = c/np.sqrt(n)
    logr = np.cumsum(np.log(1-lr))
    s = (logr[19999]-logr[9999])/(np.log(20000)-np.log(10000))
    print(f"eta_t=c/sqrt(t), c={c}: log r(2e4)={logr[19999]:.2f}, local log-log slope={s:.3f} (not a power law)")

# 4. eta_t = a/t for various a -> slope should be -a
for a in [0.5, 1.0, 0.25]:
    lr = np.minimum(a/n, 0.9)
    logr = np.cumsum(np.log(1-lr))
    s = (logr[19999]-logr[9999])/(np.log(20000)-np.log(10000))
    print(f"eta_t={a}/t: local log-log slope={s:.4f} (target {-a})")
