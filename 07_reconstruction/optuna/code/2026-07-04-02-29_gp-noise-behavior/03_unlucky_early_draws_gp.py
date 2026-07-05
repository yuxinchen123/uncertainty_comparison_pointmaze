"""Point 3: unlucky early draws sitting exactly at the true optimum -- does GPSampler come back?

Landscape (reward to MAXIMIZE): m(x) = 50 * exp(-(x-8)^2 / 2) on [0, 10]. The true optimum is x=8.
We pre-load a study with:
  - 20 informative observations at random x, valued at the true mean m(x) (they reveal the bump), and
  - 3 forced ~0 draws all placed at x=8 (three unlucky seeds at the true optimum).
Then we run 60 honest single-seed GPSampler trials: each new trial's value is ONE draw from the
project bimodal model at the proposed x (success prob m(x)/80 -> Normal(80,10), else Normal(0,3)).

We report, per 20-trial window, how many of the 60 proposals land near the optimum (|x-8| < 1),
averaged over 3 base seeds. As supporting evidence for the mechanism we also read the GP posterior
mean at x=8 right after the pre-load, with the noise fitted (default) versus pinned to the floor
(deterministic_objective=True): the fitted noise treats the 3 zeros as noise and keeps the posterior
mean at x=8 high; the pinned-noise fit is dragged down toward those zeros.
"""

import hashlib

import numpy as np
import torch

import optuna
import optuna._gp.gp as gp
import optuna._gp.prior as prior

optuna.logging.set_verbosity(optuna.logging.WARNING)

X_LOW, X_HIGH = 0.0, 10.0
DIST = optuna.distributions.FloatDistribution(X_LOW, X_HIGH)


def substream(base, *parts):
    # one generator per named quantity, keyed by a stable string
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def true_mean(x):
    # landscape: reward mean as a function of x, a Gaussian bump peaking at x=8 with height 50
    return 50.0 * np.exp(-((x - 8.0) ** 2) / 2.0)


def bimodal_reward(rng, x):
    # one honest single-seed reward draw at x: success -> Normal(80,10), else -> Normal(0,3)
    p = true_mean(x) / 80.0
    if rng.random() < p:
        return float(rng.normal(80.0, 10.0))
    return float(rng.normal(0.0, 3.0))


def build_preloaded_study(base_seed):
    # create a study pre-seeded with 20 informative (noise-free-mean) points + 3 forced zeros at x=8
    sampler = optuna.samplers.GPSampler(seed=base_seed, n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # 20 informative observations at random x, valued at the true mean m(x)
    xr = substream(base_seed, "point3", "preload_x").uniform(X_LOW, X_HIGH, 20)
    for xv in xr:
        study.add_trial(optuna.trial.create_trial(
            params={"x": float(xv)}, distributions={"x": DIST}, value=float(true_mean(xv))))
    # 3 forced ~0 draws all at the true optimum x=8 (three unlucky seeds)
    zrng = substream(base_seed, "point3", "forced_zeros")
    forced = np.abs(zrng.normal(0.0, 1.0, 3))  # small positive values near 0
    for zv in forced:
        study.add_trial(optuna.trial.create_trial(
            params={"x": 8.0}, distributions={"x": DIST}, value=float(zv)))
    return study, forced


def posterior_mean_at(study, x_query, deterministic):
    # fit optuna's GP on the study's current completed trials, return posterior mean at x_query (raw)
    ts = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    x = np.array([[t.params["x"] / X_HIGH] for t in ts])  # normalize params to [0,1] like the sampler
    y_raw = np.array([t.value for t in ts])
    means, stds = np.mean(y_raw), np.std(y_raw)
    y = (y_raw - means) / max(1e-10, stds)
    gpr = gp.fit_kernel_params(
        X=x, Y=y, is_categorical=np.zeros(1, dtype=bool),
        log_prior=prior.default_log_prior, minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=deterministic, gpr_cache=None)
    pm, _ = gpr.posterior(torch.from_numpy(np.array([x_query / X_HIGH])))
    return float(pm.item()) * stds + means, float(gpr.noise_var.item())


# ---- run 60 honest trials from the pre-loaded study, 3 base seeds ----
N_TRIALS = 60
WINDOWS = [(0, 20), (20, 40), (40, 60)]
BASE_SEEDS = [0, 1, 2]

print("Landscape m(x) = 50*exp(-(x-8)^2/2), optimum at x=8. Pre-load = 20 informative + 3 zeros@x=8.")
print("Metric: number of the 60 online GPSampler proposals with |x-8| < 1, per 20-trial window.\n")

per_seed_counts = []
for base_seed in BASE_SEEDS:
    study, forced = build_preloaded_study(base_seed)
    draw_rng = substream(base_seed, "point3", "online_draws")
    proposed_x = []

    def objective(trial):
        # honest single-seed evaluation: draw one bimodal reward at the proposed x
        x = trial.suggest_float("x", X_LOW, X_HIGH)
        proposed_x.append(x)
        return bimodal_reward(draw_rng, x)

    study.optimize(objective, n_trials=N_TRIALS)
    proposed_x = np.array(proposed_x)
    counts = [int(np.sum(np.abs(proposed_x[a:b] - 8.0) < 1.0)) for a, b in WINDOWS]
    per_seed_counts.append(counts)
    best = study.best_trial
    print(f"base_seed={base_seed}: forced zeros = {np.array2string(forced, precision=2)}  "
          f"near-optimum proposals per window {WINDOWS} = {counts}  "
          f"(best online value {best.value:6.2f} at x={best.params['x']:.2f})")

per_seed_counts = np.array(per_seed_counts)
mean_counts = per_seed_counts.mean(axis=0)
print(f"\nmean over 3 seeds, near-optimum (|x-8|<1) proposals per window: "
      f"[{mean_counts[0]:.1f}, {mean_counts[1]:.1f}, {mean_counts[2]:.1f}]  (out of 20 each)")
print()

# ---- mechanism check: posterior mean at x=8 right after the pre-load (seed 0) ----
study0, forced0 = build_preloaded_study(0)
pm_fit, nv_fit = posterior_mean_at(study0, 8.0, deterministic=False)
pm_det, nv_det = posterior_mean_at(study0, 8.0, deterministic=True)
print("Right after the pre-load (20 informative + 3 zeros@x=8), GP posterior mean at x=8:")
print(f"  true mean m(8)                         = {true_mean(8.0):.2f}")
print(f"  the 3 forced draws at x=8              = {np.array2string(forced0, precision=2)}")
print(f"  posterior mean, noise FITTED (default) = {pm_fit:6.2f}   (fitted noise_var={nv_fit:.4f})")
print(f"  posterior mean, noise PINNED (det=True)= {pm_det:6.2f}   (noise_var={nv_det:.6f})")
print("  -> fitted noise treats the 3 zeros as noise and keeps the mean near the bump height;")
print("     pinned noise is pulled down toward the zeros, so the GP would avoid the optimum longer.")
