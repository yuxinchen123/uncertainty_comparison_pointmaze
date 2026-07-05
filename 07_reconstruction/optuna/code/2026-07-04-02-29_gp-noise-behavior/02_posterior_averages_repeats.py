"""Point 2: repeated noisy observations at one point are averaged by the GP posterior.

Setup: put k = 1, 3, 10 noisy reward draws at ONE location x_target (true mean 30, project bimodal
model so each draw is either ~0 or ~80), plus 3 anchor observations at other locations so the GP is
spatially well posed. Fit optuna's own GP (optuna._gp.gp.fit_kernel_params) exactly as GPSampler
does (same standardization, same prior, same noise floor), then read the posterior mean at x_target
back in raw reward units.

Reported per k: posterior mean at x_target vs the running sample mean of the k draws vs the LAST
draw. With the default fitted noise (deterministic_objective=False) the posterior mean should track
the sample mean, not the last draw. The same fit with deterministic_objective=True (noise pinned to
the 1e-6 floor) is shown for contrast: it tries to honor every conflicting draw at one location.
"""

import hashlib

import numpy as np
import torch

import optuna._gp.gp as gp
import optuna._gp.prior as prior


def substream(base, *parts):
    # one generator per named quantity, keyed by a stable string
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def bimodal_rewards(rng, mean, n):
    # project noise model: one reward per seed, success -> Normal(80,10), else -> Normal(0,3)
    p = mean / 80.0
    success = rng.random(n) < p
    return np.where(success, rng.normal(80.0, 10.0, n), rng.normal(0.0, 3.0, n))


def posterior_mean_raw(X, y_raw, x_query, deterministic):
    # standardize y as GPSampler does, fit optuna's GP, return posterior mean at x_query in raw units
    means = np.mean(y_raw)
    stds = np.std(y_raw)
    y = (y_raw - means) / max(1e-10, stds)
    is_categorical = np.zeros(X.shape[1], dtype=bool)
    gpr = gp.fit_kernel_params(
        X=X, Y=y, is_categorical=is_categorical,
        log_prior=prior.default_log_prior, minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=deterministic, gpr_cache=None,
    )
    post_mean_std, post_var_std = gpr.posterior(torch.from_numpy(np.asarray(x_query, dtype=np.float64)))
    # de-standardize the posterior mean and std back to raw reward units
    mean_raw = float(post_mean_std.item()) * stds + means
    std_raw = float(np.sqrt(post_var_std.item())) * stds
    return mean_raw, std_raw, float(gpr.noise_var.item())


# Three anchor observations elsewhere in the unit interval, each a single bimodal draw, so the GP has
# spatial structure to fit a lengthscale against. x_target is where we stack the repeated draws.
X_TARGET = 0.5
ANCHOR_X = np.array([0.05, 0.30, 0.95])
ANCHOR_MEANS = np.array([10.0, 60.0, 5.0])
anchor_rng = substream(2026, "point2", "anchors")
ANCHOR_Y = np.array([bimodal_rewards(anchor_rng, m, 1)[0] for m in ANCHOR_MEANS])

print("anchor observations (fixed across all k):")
for xi, yi in zip(ANCHOR_X, ANCHOR_Y):
    print(f"  x={xi:4.2f}  draw={yi:7.2f}")
print(f"x_target = {X_TARGET}  (true mean 30, so each draw is ~0 or ~80)")
print()

# Draw 10 rewards once at x_target; the k=1,3,10 cases use the first 1,3,10 of these same draws so
# the running sample mean is a genuine prefix average.
target_rng = substream(2026, "point2", "target")
all_target_y = bimodal_rewards(target_rng, 30.0, 10)
print("the 10 draws at x_target (in order):", np.array2string(all_target_y, precision=2))
print()

header = f"{'k':>3} {'draws so far':>34} {'sample_mean':>12} {'last_draw':>10} {'post_mean(fit)':>15} {'post_mean(det)':>15} {'noise_var(fit)':>15}"
print(header)
print("-" * len(header))
for k in (1, 3, 10):
    # build the training set: k stacked draws at x_target plus the 3 anchors
    ky = all_target_y[:k]
    X = np.vstack([np.full((k, 1), X_TARGET), ANCHOR_X.reshape(-1, 1)])
    y_raw = np.concatenate([ky, ANCHOR_Y])
    xq = np.array([X_TARGET])
    pm_fit, ps_fit, nv_fit = posterior_mean_raw(X, y_raw, xq, deterministic=False)
    pm_det, ps_det, nv_det = posterior_mean_raw(X, y_raw, xq, deterministic=True)
    draws_str = np.array2string(ky, precision=1, max_line_width=200)
    print(f"{k:>3} {draws_str:>34} {np.mean(ky):>12.2f} {ky[-1]:>10.2f} {pm_fit:>15.2f} {pm_det:>15.2f} {nv_fit:>15.4f}")

print()
print("Reading: post_mean(fit) is the DEFAULT GPSampler behavior (noise fitted); it tracks the")
print("sample mean of the draws at x_target, not the last draw. post_mean(det) pins noise to the")
print("1e-6 floor; with several conflicting draws stacked at ONE x it cannot interpolate them all,")
print("so the least-squares solution still lands near the mean but the posterior std collapses.")
print()

# Extra: distinct-but-nearby cluster so deterministic_objective=True can actually INTERPOLATE each
# point (posterior mean AT a training x equals that x's draw). This is the honest 'interpolation'
# contrast: with the noise floor the GP passes through every noisy draw instead of averaging.
print("Interpolation contrast on DISTINCT nearby points (cluster of 5 around x=0.5):")
cluster_x = np.array([0.46, 0.48, 0.50, 0.52, 0.54])
cluster_rng = substream(2026, "point2", "cluster")
cluster_y = bimodal_rewards(cluster_rng, 30.0, 5)
Xc = np.vstack([cluster_x.reshape(-1, 1), ANCHOR_X.reshape(-1, 1)])
yc = np.concatenate([cluster_y, ANCHOR_Y])
print("  cluster draws:", np.array2string(cluster_y, precision=2), " sample mean = %.2f" % np.mean(cluster_y))
print(f"  {'x':>6} {'draw':>8} {'post_mean(fit)':>15} {'post_mean(det)':>15}")
for xi, yi in zip(cluster_x, cluster_y):
    pm_fit, _, _ = posterior_mean_raw(Xc, yc, np.array([xi]), deterministic=False)
    pm_det, _, _ = posterior_mean_raw(Xc, yc, np.array([xi]), deterministic=True)
    print(f"  {xi:>6.2f} {yi:>8.2f} {pm_fit:>15.2f} {pm_det:>15.2f}")
print("  -> deterministic post_mean sits on each raw draw (interpolates the noise); the fitted-noise")
print("     post_mean is pulled toward the common cluster mean %.2f (averages the noise)." % np.mean(cluster_y))
