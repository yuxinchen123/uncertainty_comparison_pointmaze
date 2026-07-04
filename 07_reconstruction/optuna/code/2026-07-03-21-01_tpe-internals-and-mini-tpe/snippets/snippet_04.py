# Mini-TPE in numpy (no optuna): 10 random startup, top-25% split, KDE l/g, 24 candidates, argmax
import hashlib
import numpy as np

LOW, HIGH, PEAK = np.array([-6., -3.]), np.array([-2., -1.]), np.array([-6., -2.])
RANGE = HIGH - LOW

def substream(base, *parts):                       # one RNG per named quantity, hashed key
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)

def evaluate(p, base, method, i):                  # bump (max 50, width 1 decade) + noise SD 10
    val = 50.0 * np.exp(-np.sum((p - PEAK) ** 2) / 2.0)
    return val + substream(base, method, "noise", i).normal(0, 10)

def kde_logpdf(x, data, bw):                        # log Gaussian-KDE density at points x
    z = (x[:, None] - data[None, :]) / bw
    lk = -0.5 * z**2 - np.log(bw) - 0.5 * np.log(2 * np.pi)
    m = lk.max(1)
    return m + np.log(np.mean(np.exp(lk - m[:, None]), 1))

def mini_tpe(base, n_trials=60, n_cand=24):         # returns evaluated points
    pts, vals = [], []
    su = substream(base, "mtpe", "startup")
    for i in range(10):                              # 10 random startup points
        p = LOW + su.uniform(0, 1, 2) * RANGE
        pts.append(p); vals.append(evaluate(p, base, "mtpe", i))
    for i in range(10, n_trials):
        P, V = np.array(pts), np.array(vals)
        n_good = max(1, int(np.ceil(0.25 * len(V))))     # top-25% good / rest split
        order = np.argsort(-V)
        good, rest = P[order[:n_good]], P[order[n_good:]]
        cr = substream(base, "mtpe", "cand", i)
        cand, score = np.empty((n_cand, 2)), np.zeros(n_cand)
        for d in range(2):
            bwg, bwr = RANGE[d] / np.sqrt(len(good)), RANGE[d] / np.sqrt(max(len(rest), 1))
            c = np.clip(good[cr.integers(0, len(good), n_cand), d] + cr.normal(0, bwg, n_cand),
                        LOW[d], HIGH[d])              # draw candidates from the good density l
            cand[:, d] = c
            score += kde_logpdf(c, good[:, d], bwg) - kde_logpdf(c, rest[:, d], bwr)  # log l - log g
        best = cand[int(np.argmax(score))]
        pts.append(best); vals.append(evaluate(best, base, "mtpe", i))
    return np.array(pts)

def random_pts(base, n=60):
    r = substream(base, "rand")
    return np.array([LOW + r.uniform(0, 1, 2) * RANGE for _ in range(n)])

def hits(pts):                                       # trials within half a decade of the peak
    return int(sum(np.sqrt(np.sum((p - PEAK) ** 2)) < 0.5 for p in pts))

print("seed  mini_tpe  random  (hits within 0.5 of peak, out of 60)")
for s in (0, 1, 2):
    print(f"{s:>4} {hits(mini_tpe(s)):>9} {hits(random_pts(s)):>7}")
