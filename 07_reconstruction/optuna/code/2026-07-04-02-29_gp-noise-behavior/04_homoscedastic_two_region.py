"""Point 4: the GP fits ONE noise variance for the whole space (homoscedastic limit).

Two-region landscape on x in [0,10], reward to MAXIMIZE, each trial is one honest single-seed draw:
  - region A (x < 5): Normal(30, 2)                 -> true mean 30, small per-seed spread
  - region B (x >= 5): 80 with prob 0.15, else Normal(15, 3)  -> true mean 0.15*80+0.85*15 = 24.75

Region A has the higher true mean (30 > 24.75). Region B occasionally returns a large 80. A sampler
that reacts to individual large draws is pulled into B; a sampler that averages each region's draws
prefers A. GPSampler fits a single observation-noise variance shared across both regions, so it
averages B's draws toward ~25 and prefers A -- but that one shared variance is inflated by region B's
large spread, widening the posterior uncertainty everywhere (including region A).

We run 150 single-seed GPSampler trials for 3 base seeds and report, per seed: how the last 50
proposals split between A and B, the best single trial, the location of the best GP posterior mean,
the GP posterior mean at the center of each region, and the single fitted noise variance.
"""

import hashlib

import numpy as np
import torch

import optuna
import optuna._gp.gp as gp
import optuna._gp.prior as prior

optuna.logging.set_verbosity(optuna.logging.WARNING)

X_LOW, X_HIGH, BOUNDARY = 0.0, 10.0, 5.0
N_TRIALS, LAST = 150, 50
BASE_SEEDS = [0, 1, 2]


def substream(base, *parts):
    # one generator per named quantity, keyed by a stable string
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def region_draw(rng, x):
    # one honest single-seed reward draw from whichever region x falls in
    if x < BOUNDARY:
        return float(rng.normal(30.0, 2.0))          # region A: true mean 30
    if rng.random() < 0.15:
        return 80.0                                  # region B: the 15%-chance large draw
    return float(rng.normal(15.0, 3.0))              # region B: the common low draw (true mean 24.75)


def fit_gpr_on_study(study):
    # fit optuna's GP on all completed trials; return the fitted regressor + standardization stats
    ts = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    x = np.array([[t.params["x"] / X_HIGH] for t in ts])  # normalize params to [0,1] like the sampler
    y_raw = np.array([t.value for t in ts])
    means, stds = np.mean(y_raw), np.std(y_raw)
    y = (y_raw - means) / max(1e-10, stds)
    gpr = gp.fit_kernel_params(
        X=x, Y=y, is_categorical=np.zeros(1, dtype=bool),
        log_prior=prior.default_log_prior, minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=False, gpr_cache=None)
    return gpr, means, stds


def posterior_mean_grid(gpr, means, stds, grid):
    # posterior mean in raw reward units over a grid of raw x values
    xn = torch.from_numpy((grid / X_HIGH).reshape(-1, 1))
    pm, _ = gpr.posterior(xn)
    return pm.detach().numpy() * stds + means


print("Two-region: A(x<5)=Normal(30,2) mean 30; B(x>=5)=80 w.p. 0.15 else Normal(15,3) mean 24.75.")
print(f"Region A has the higher true mean. Running {N_TRIALS} single-seed GPSampler trials x 3 seeds.\n")

grid = np.linspace(X_LOW, X_HIGH, 201)
header = f"{'seed':>4} {'A|B last50':>11} {'best trial':>22} {'best post-mean x':>17} {'postmean A/B ctr':>18} {'fitted noise_var':>16}"
print(header)
print("-" * len(header))
frac_A_last = []
for base_seed in BASE_SEEDS:
    sampler = optuna.samplers.GPSampler(seed=base_seed)  # default n_startup_trials=10, noise fitted
    study = optuna.create_study(direction="maximize", sampler=sampler)
    draw_rng = substream(base_seed, "point4", "draws")
    proposed_x = []

    def objective(trial):
        # honest single-seed evaluation in the region of the proposed x
        x = trial.suggest_float("x", X_LOW, X_HIGH)
        proposed_x.append(x)
        return region_draw(draw_rng, x)

    study.optimize(objective, n_trials=N_TRIALS)
    proposed_x = np.array(proposed_x)
    last = proposed_x[-LAST:]
    nA = int(np.sum(last < BOUNDARY))
    nB = LAST - nA
    frac_A_last.append(nA / LAST)

    gpr, m_, s_ = fit_gpr_on_study(study)
    pm_grid = posterior_mean_grid(gpr, m_, s_, grid)
    x_best_post = grid[int(np.argmax(pm_grid))]
    pm_A = posterior_mean_grid(gpr, m_, s_, np.array([2.5]))[0]
    pm_B = posterior_mean_grid(gpr, m_, s_, np.array([7.5]))[0]
    noise_var_raw = float(gpr.noise_var.item()) * s_ ** 2  # de-standardize to raw reward^2 units
    best = study.best_trial
    print(f"{base_seed:>4} {nA:>4}|{nB:<6} {best.value:>8.2f} at x={best.params['x']:<7.2f} "
          f"{x_best_post:>17.2f} {pm_A:>8.2f}/{pm_B:<8.2f} "
          f"{gpr.noise_var.item():>8.4f}(std)")

print()
print(f"mean fraction of last-{LAST} proposals in region A (higher true mean): "
      f"{np.mean(frac_A_last):.2f}  (>0.5 means the GP prefers A)")
print()
print("Reading:")
print("  - 'A|B last50' : how the final 50 proposals split between region A (x<5) and B (x>=5).")
print("  - 'best post-mean x' : argmax of the GP posterior mean over x; lands in region A (x<5).")
print("  - 'postmean A/B ctr' : posterior mean at x=2.5 (A) vs x=7.5 (B). B is averaged toward ~25,")
print("     below A's ~30, so the GP prefers A -- it does NOT chase region B's occasional 80s.")
print("  - 'fitted noise_var' : ONE shared value (standardized units). It is inflated by region B's")
print("     large spread and applies to region A as well, widening uncertainty across the whole space.")
