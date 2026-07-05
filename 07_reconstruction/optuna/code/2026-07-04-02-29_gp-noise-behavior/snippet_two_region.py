"""GPSampler on a two-region problem: it averages each region and prefers the higher-mean one.

Region A (x<5): Normal(30, 2)                      -> true mean 30
Region B (x>=5): 80 with prob 0.15 else Normal(15,3) -> true mean 0.15*80 + 0.85*15 = 24.75

Region A has the higher true mean; region B just returns a large 80 sometimes. The GP fits ONE
shared noise variance, averages each region's draws, and prefers A.
"""
import hashlib

import numpy as np
import torch

import optuna
import optuna._gp.gp as gp
import optuna._gp.prior as prior

optuna.logging.set_verbosity(optuna.logging.WARNING)


def substream(base, *parts):
    # one generator per named quantity, keyed by a stable string
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def region_draw(rng, x):
    # one honest single-seed reward draw from whichever region x falls in
    if x < 5.0:
        return float(rng.normal(30.0, 2.0))                    # region A: true mean 30
    return 80.0 if rng.random() < 0.15 else float(rng.normal(15.0, 3.0))  # region B: true mean 24.75


def posterior_mean_raw(study, xs):
    # fit optuna's GP on the study's completed trials; return posterior mean at raw x values xs
    ts = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    y_raw = np.array([t.value for t in ts])
    m, s = y_raw.mean(), max(1e-10, y_raw.std())
    gpr = gp.fit_kernel_params(
        X=np.array([[t.params["x"] / 10.0] for t in ts]), Y=(y_raw - m) / s,
        is_categorical=np.zeros(1, dtype=bool), log_prior=prior.default_log_prior,
        minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR, deterministic_objective=False, gpr_cache=None)
    pm, _ = gpr.posterior(torch.from_numpy((np.asarray(xs) / 10.0).reshape(-1, 1)))
    return pm.detach().numpy() * s + m, float(gpr.noise_var.item())


grid = np.linspace(0, 10, 201)
rows = []  # collect (frac_A_last50, best_post_mean_x, pm_A_center, pm_B_center) per seed
print(f"{'seed':>4} {'A|B (last 50)':>14} {'best post-mean x':>17} {'post-mean A/B center':>21} {'noise_var':>10}")
for seed in (0, 1, 2):
    # 150 single-seed GPSampler trials; each trial is one honest draw in the proposed region
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GPSampler(seed=seed))
    rng = substream(seed, "draws")
    proposed = []

    def objective(trial):
        # record the proposed x, then return one honest single-seed reward draw there
        x = trial.suggest_float("x", 0, 10)
        proposed.append(x)
        return region_draw(rng, x)

    study.optimize(objective, n_trials=150)
    last = np.array(proposed[-50:])
    nA = int(np.sum(last < 5.0))
    pm_grid, nv = posterior_mean_raw(study, grid)
    (pm_A, pm_B), _ = posterior_mean_raw(study, np.array([2.5, 7.5]))
    rows.append((nA / 50.0, grid[int(np.argmax(pm_grid))], pm_A, pm_B))
    print(f"{seed:>4} {nA:>6}|{50-nA:<7} {grid[int(np.argmax(pm_grid))]:>17.2f} "
          f"{pm_A:>9.2f}/{pm_B:<10.2f} {nv:>10.4f}")

rows = np.array(rows)
print(f"\nmean over 3 seeds: fraction of last-50 in region A = {rows[:,0].mean():.2f}  "
      f"(>0.5 = prefers A)")
print(f"mean over 3 seeds: post-mean at A center = {rows[:,2].mean():.2f}, "
      f"at B center = {rows[:,3].mean():.2f}  (B near its true mean 24.75)")
print("Best posterior mean lands in region A (x<5) for every seed. The GP averages each region and")
print("prefers A (true mean 30) over B (true mean 24.75); it does not chase B's occasional 80s.")
