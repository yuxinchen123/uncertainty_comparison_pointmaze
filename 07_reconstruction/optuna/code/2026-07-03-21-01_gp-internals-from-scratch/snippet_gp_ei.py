"""Gaussian process posterior + closed-form Expected Improvement, numpy only. Self-contained."""
from math import erf
import numpy as np

# Toy 1-D objective: reward vs x = log10(ridge) on [-6,-2], a Gaussian bump peaking at x=-4.
true = lambda x: 50.0 * np.exp(-(((x + 4.0) / 1.2) ** 2))
x_obs = np.array([-6.0, -5.5, -5.0, -3.0, -2.5, -2.0])          # six flank observations
y_obs = true(x_obs) + np.array([0.3, -0.2, 0.1, -0.1, 0.2, -0.3])  # fixed inline noise -> deterministic
S2, L, NOISE = 400.0, 0.8, 1.0                                  # fixed hyperparameters (signal var, lengthscale, noise var)

def matern52(xa, xb):
    """Matern 5/2 covariance k(r)=s2(1+sqrt5 r/L+5r^2/3L^2)exp(-sqrt5 r/L) for 1-D inputs."""
    r = np.abs(xa[:, None] - xb[None, :])          # pairwise distances
    s = np.sqrt(5.0) * r / L                        # scaled distance sqrt(5) r / L
    return S2 * (1.0 + s + s * s / 3.0) * np.exp(-s)

def gp_posterior(xg):
    """Posterior mean mu and std sigma on grid xg via mu=k*(K+nI)^-1 y, var=k(x,x)-k*(K+nI)^-1 k*."""
    K = matern52(x_obs, x_obs) + NOISE * np.eye(len(x_obs))   # training covariance + noise
    Lc = np.linalg.cholesky(K)                                # Cholesky for stable solves
    alpha = np.linalg.solve(Lc.T, np.linalg.solve(Lc, y_obs)) # (K+nI)^-1 y
    Ks = matern52(xg, x_obs)                                   # cross-covariance grid vs data
    mu = Ks @ alpha                                           # posterior mean
    v = np.linalg.solve(Lc, Ks.T)                             # L^-1 k*^T
    var = np.clip(S2 - np.sum(v * v, axis=0), 1e-12, None)    # posterior variance (k(x,x)=S2)
    return mu, np.sqrt(var)

def ei(mu, sigma, f_best):
    """Closed-form EI(x)=(mu-f_best)Phi(z)+sigma phi(z), z=(mu-f_best)/sigma; higher is better."""
    z = (mu - f_best) / sigma                                 # standardized improvement
    phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)           # standard normal pdf
    Phi = 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))   # standard normal cdf
    return (mu - f_best) * Phi + sigma * phi                  # exploit term + explore term

# Evaluate posterior and EI on a grid; incumbent f_best is the best observed reward.
xg = np.round(np.arange(-6.0, -2.0 + 1e-9, 0.25), 2)
f_best = y_obs.max()
mu, sigma = gp_posterior(xg)
z = (mu - f_best) / sigma
ei_vals = ei(mu, sigma, f_best)

print("f_best=%.2f at x=%+.2f  (fixed: s2=%.0f L=%.2f noise=%.1f)" % (f_best, x_obs[y_obs.argmax()], S2, L, NOISE))
print(f"{'x':>7}{'mu':>9}{'sigma':>9}{'z':>8}{'EI':>9}")
for xi, mi, si, zi, ev in zip(xg, mu, sigma, z, ei_vals):
    print(f"{xi:7.2f}{mi:9.2f}{si:9.2f}{zi:8.2f}{ev:9.3f}")
i = int(ei_vals.argmax())
print("argmax EI at x=%+.2f (mu=%.2f sigma=%.2f) -- in the unsampled gap between x=-5 and x=-3"
      % (xg[i], mu[i], sigma[i]))

# Explore vs exploit: x=-4.00 (far from data, big sigma) vs x=-4.75 (next to best obs, mu above f_best).
for label, xq in [("EXPLORE (sigma large)", -4.00), ("EXPLOIT (mu high)", -4.75)]:
    j = int(np.argmin(np.abs(xg - xq)))
    exploit_term = (mu[j] - f_best) * 0.5 * (1 + erf((z[j]) / np.sqrt(2)))
    explore_term = sigma[j] * np.exp(-0.5 * z[j] ** 2) / np.sqrt(2 * np.pi)
    print("%-22s x=%+.2f mu=%.2f sigma=%.2f z=%.2f EI=%.3f [exploit=%.2f explore=%.2f]"
          % (label, xg[j], mu[j], sigma[j], z[j], ei_vals[j], exploit_term, explore_term))
