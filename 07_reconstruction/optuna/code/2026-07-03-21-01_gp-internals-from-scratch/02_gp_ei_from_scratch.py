"""Gaussian process posterior + closed-form Expected Improvement, numpy only (no optuna, no torch).

Toy problem: final training reward as a function of x = log10(ridge) on [-6, -2].
The true curve peaks in the interior at x = -4 (a Gaussian bump). Six observations are placed on
the two flanks, leaving the peak region between x=-5 and x=-3 unsampled, so the interesting region
is a gap between observations. Kernel hyperparameters are FIXED (stated below); the point is the
mechanism, not the fitting.
"""

from math import erf

import numpy as np

# ----- fixed toy setup (all inline so the demo is deterministic) -----
# True objective: a Gaussian reward bump centered at x=-4 (interior of the [-6,-2] domain).
TRUE = lambda x: 50.0 * np.exp(-(((x + 4.0) / 1.2) ** 2))
# Six observation locations on the flanks; the peak region (-5,-3) is deliberately left unsampled.
X_OBS = np.array([-6.0, -5.5, -5.0, -3.0, -2.5, -2.0])
# Small fixed measurement offsets written inline (deterministic; break exact left/right symmetry).
NOISE_OFFSETS = np.array([+0.3, -0.2, +0.1, -0.1, +0.2, -0.3])
Y_OBS = TRUE(X_OBS) + NOISE_OFFSETS
# Fixed kernel hyperparameters: signal variance s2, lengthscale L (x-units), observation noise var.
S2 = 400.0        # signal variance; prior std sqrt(S2)=20 reward units
L = 0.8           # lengthscale in x = log10(ridge) units
NOISE_VAR = 1.0   # assumed observation noise variance used in (K + noise*I)


def matern52(xa, xb, s2, ell):
    """Matern 5/2 covariance k(r)=s2*(1+sqrt5 r/L+5 r^2/(3 L^2))*exp(-sqrt5 r/L) for 1-D inputs."""
    # Pairwise absolute distances r between every xa and every xb.
    r = np.abs(xa[:, None] - xb[None, :])
    # Scaled distance sqrt(5)*r/L that appears in every term of the Matern 5/2 form.
    s = np.sqrt(5.0) * r / ell
    # Assemble the closed-form Matern 5/2 kernel matrix.
    return s2 * (1.0 + s + (s * s) / 3.0) * np.exp(-s)


def normal_pdf(z):
    """Standard normal density phi(z) = exp(-z^2/2)/sqrt(2 pi)."""
    # Evaluate the closed-form standard normal density.
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def normal_cdf(z):
    """Standard normal CDF Phi(z) = 0.5(1+erf(z/sqrt2)), computed elementwise via math.erf."""
    # Apply the scalar error function to each entry to get the standard normal CDF.
    return 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))


def gp_posterior(x_grid, x_obs, y_obs, s2, ell, noise_var):
    """Return GP posterior mean mu(x) and std sigma(x) on x_grid using the standard formulas."""
    # Training covariance K = k(X,X) + noise*I, then its Cholesky factor for stable solves.
    K = matern52(x_obs, x_obs, s2, ell) + noise_var * np.eye(len(x_obs))
    Lc = np.linalg.cholesky(K)
    # alpha = (K+noise I)^-1 y via two triangular solves (no explicit inverse).
    alpha = np.linalg.solve(Lc.T, np.linalg.solve(Lc, y_obs))
    # Cross-covariance k*(grid, X) between the query grid and the observations.
    Ks = matern52(x_grid, x_obs, s2, ell)
    # Posterior mean mu = k*^T (K+noise I)^-1 y.
    mu = Ks @ alpha
    # v = L^-1 k*^T so that k*^T (K+noise I)^-1 k* = sum(v^2) columnwise.
    v = np.linalg.solve(Lc, Ks.T)
    # Posterior variance sigma^2 = k(x,x) - k*^T (K+noise I)^-1 k*; k(x,x)=s2 for Matern 5/2.
    var = s2 - np.sum(v * v, axis=0)
    # Clamp tiny negatives from round-off, then take the square root for the std.
    var = np.clip(var, 1e-12, None)
    return mu, np.sqrt(var)


def expected_improvement(mu, sigma, f_best):
    """Closed-form EI(x) = (mu - f_best) Phi(z) + sigma phi(z), z = (mu - f_best)/sigma (maximize)."""
    # Standardized improvement z; sigma is strictly positive here so no divide-by-zero.
    z = (mu - f_best) / sigma
    # Expected improvement combines an exploitation term and an exploration term.
    return (mu - f_best) * normal_cdf(z) + sigma * normal_pdf(z)


# Query grid over the whole domain and the incumbent (best observed reward so far).
x_grid = np.round(np.arange(-6.0, -2.0 + 1e-9, 0.25), 4)
f_best = Y_OBS.max()

# Compute posterior and EI on the grid.
mu, sigma = gp_posterior(x_grid, X_OBS, Y_OBS, S2, L, NOISE_VAR)
z = (mu - f_best) / sigma
ei = expected_improvement(mu, sigma, f_best)

# Report the setup so the demo is fully self-contained.
print("Fixed hyperparameters: s2(signal var)=%.1f  L(lengthscale)=%.2f  noise_var=%.2f"
      % (S2, L, NOISE_VAR))
print("Observations (x=log10 ridge, y=reward):")
for xi, yi in zip(X_OBS, Y_OBS):
    print("  x=%+.2f  y=%6.2f" % (xi, yi))
print("f_best (incumbent) = %.3f at x=%+.2f" % (f_best, X_OBS[np.argmax(Y_OBS)]))
print()

# Print the grid table of x, mu, sigma, z, EI.
print(f"{'x':>7} {'mu':>9} {'sigma':>9} {'z':>8} {'EI':>9}")
for xi, mi, si, zi, ei_i in zip(x_grid, mu, sigma, z, ei):
    print(f"{xi:7.2f} {mi:9.3f} {si:9.3f} {zi:8.3f} {ei_i:9.4f}")

# State where the acquisition maximum falls.
i_star = int(np.argmax(ei))
print()
print("argmax EI at x=%+.2f  (mu=%.2f, sigma=%.2f, EI=%.4f)"
      % (x_grid[i_star], mu[i_star], sigma[i_star], ei[i_star]))
print("Nearest observations bracketing it: this x lies in the unsampled gap between x=-5 and x=-3.")

# Explore/exploit split: pick the highest-EI grid point in the far-from-data interior (large sigma)
# versus the highest-EI grid point adjacent to the best observation (mu near f_best, small sigma).
# "Far from data" = interior gap points; "near good data" = within 0.6 of the incumbent location.
interior = (x_grid > -4.9) & (x_grid < -3.1)          # inside the unsampled gap
near_best = np.abs(x_grid - X_OBS[np.argmax(Y_OBS)]) <= 0.6   # next to the incumbent observation

# Among interior points, the one with the largest sigma is the exploration-driven pick.
explore_i = np.flatnonzero(interior)[np.argmax(sigma[interior])]
# Among near-incumbent points (excluding the observation itself), the largest-mu is exploit-driven.
cand_exploit = np.flatnonzero(near_best & (sigma > sigma.min() + 1e-9))
exploit_i = cand_exploit[np.argmax(mu[cand_exploit])]

def ei_terms(mu_i, sigma_i, f_best):
    """Split EI into its exploitation term (mu-f_best)Phi(z) and exploration term sigma*phi(z)."""
    # Recompute z, then return the two additive pieces of the EI formula separately.
    zi = (mu_i - f_best) / sigma_i
    return (mu_i - f_best) * float(normal_cdf(np.array([zi]))[0]), sigma_i * float(normal_pdf(zi))


# Decompose EI at the two archetype points to show which term dominates.
ex_exploit_term, ex_explore_term = ei_terms(mu[explore_i], sigma[explore_i], f_best)
xp_exploit_term, xp_explore_term = ei_terms(mu[exploit_i], sigma[exploit_i], f_best)

print()
print("EXPLORE-driven point (EI high mainly because sigma is large / far from data):")
print("  x=%+.2f  mu=%.2f  sigma=%.2f  z=%.3f  EI=%.4f  [exploit term=%.3f, explore term=%.3f]"
      % (x_grid[explore_i], mu[explore_i], sigma[explore_i], z[explore_i], ei[explore_i],
         ex_exploit_term, ex_explore_term))
print("  -> mu is BELOW f_best (z<0), so the exploitation term is negative; all EI comes from sigma.")
print("EXPLOIT-driven point (EI high mainly because mu is high / near the best observation):")
print("  x=%+.2f  mu=%.2f  sigma=%.2f  z=%.3f  EI=%.4f  [exploit term=%.3f, explore term=%.3f]"
      % (x_grid[exploit_i], mu[exploit_i], sigma[exploit_i], z[exploit_i], ei[exploit_i],
         xp_exploit_term, xp_explore_term))
print("  -> mu is ABOVE f_best (z>0): the predicted mean itself beats the incumbent, with small sigma.")
