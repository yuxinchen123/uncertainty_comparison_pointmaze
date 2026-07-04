"""Point 5: insert already-finished (params, value) pairs into a fresh study WITHOUT running them,
using optuna.trial.create_trial(...) + study.add_trial(...). Part A mirrors the project's grid
(categorical + log-float distributions) and shows a TPESampler study continues from the imports.
Part B verifies the imported COMPLETE trials count toward TPESampler.n_startup_trials: once the
completed count reaches the threshold, the next ask samples from the TPE model, not uniformly."""

import hashlib
import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution


def substream(base_seed, *parts):
    """Return a numpy generator seeded by hashing a stable name, one stream per named quantity."""
    # stable key -> 32-bit seed so each named quantity is independent and reproducible
    key = "::".join(str(p) for p in (base_seed, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


# the project-grid search space expressed as Optuna distribution objects
GRID_DISTRIBUTIONS = {
    "normalization": CategoricalDistribution(["none", "unit"]),
    "ridge_lambda": FloatDistribution(1e-6, 1e-2, log=True),
    "bonus_clip": CategoricalDistribution(["inf", "5"]),
    "beta": FloatDistribution(1e-3, 1e-1, log=True),
}


def import_project_grid():
    """Build a fresh TPE study and add 20 finished grid runs (no training), then ask one new trial."""
    # enumerate the real grid: 2 norms x 3 ridge x 2 clips x 3 betas = 36 cells, take the first 20
    combos = []
    for norm in ["none", "unit"]:
        for ridge in [1e-6, 1e-4, 1e-2]:
            for clip in ["inf", "5"]:
                for beta in [1e-3, 1e-2, 1e-1]:
                    combos.append({"normalization": norm, "ridge_lambda": ridge,
                                   "bonus_clip": clip, "beta": beta})
    combos = combos[:20]

    # a fresh study with a seeded TPE sampler; maximize because the objective is final eval reward
    study = optuna.create_study(
        study_name="point5_grid_import",
        sampler=optuna.samplers.TPESampler(seed=0),
        direction="maximize",
    )

    # turn each finished grid cell into a COMPLETE trial and add it, WITHOUT running anything
    for combo in combos:
        # a plausible final reward: best near beta=1e-2 + unit norm, plus reproducible per-cell noise
        base = 40.0 if combo["beta"] == 1e-2 else 15.0
        base += 8.0 if combo["normalization"] == "unit" else 0.0
        reward = base + substream("reward", combo["normalization"], combo["ridge_lambda"],
                                  combo["bonus_clip"], combo["beta"]).normal(0.0, 5.0)
        # create_trial with params+distributions+value defaults its state to COMPLETE
        trial = optuna.trial.create_trial(
            params=combo, distributions=GRID_DISTRIBUTIONS, value=reward
        )
        study.add_trial(trial)

    # verify the imports landed as finished trials before any ask happens
    states = {t.state.name for t in study.trials}
    print(f"[A] len(study.trials) before any ask: {len(study.trials)}")
    print(f"[A] states of imported trials: {states}")
    print(f"[A] best imported value: {study.best_value:.3f}  params: {study.best_params}")

    # the study continues: the next ask is trial number 20 and suggests all four grid params
    trial = study.ask()
    suggested = {
        "normalization": trial.suggest_categorical("normalization", ["none", "unit"]),
        "ridge_lambda": trial.suggest_float("ridge_lambda", 1e-6, 1e-2, log=True),
        "bonus_clip": trial.suggest_categorical("bonus_clip", ["inf", "5"]),
        "beta": trial.suggest_float("beta", 1e-3, 1e-1, log=True),
    }
    print(f"[A] ask() after import succeeded: trial.number={trial.number}")
    print(f"[A] suggested params for the new trial: {suggested}")


def build_single_param_study_with_imports(n_startup_trials, seed):
    """Build a maximize TPE study, import 20 finished single-param trials peaked at x=5, return it."""
    # single continuous parameter so the sampler's concentration is easy to measure
    dist = {"x": FloatDistribution(-10.0, 10.0)}
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=seed),
        direction="maximize",
    )
    # 20 finished trials whose value peaks at x=5, spread across the range on a fixed grid
    xs = np.linspace(-10.0, 10.0, 20)
    for x in xs:
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": float(x)}, distributions=dist, value=-((x - 5.0) ** 2)
            )
        )
    return study


def n_startup_behavior():
    """Show that imported COMPLETE trials count toward n_startup_trials by comparing two thresholds."""
    # ask 20 fresh trials without telling; each conditions only on the 20 imported COMPLETE trials
    def sample_20_xs(study):
        # collect the x each ask suggests; RUNNING asked trials do not inform the default TPE sampler
        xs = []
        for _ in range(20):
            t = study.ask()
            xs.append(t.suggest_float("x", -10.0, 10.0))
        return np.array(xs)

    # threshold BELOW the 20 imports: 20 >= 10, so the very first ask already uses the TPE model
    study_post = build_single_param_study_with_imports(n_startup_trials=10, seed=1)
    completed_post = len([t for t in study_post.trials if t.state == optuna.trial.TrialState.COMPLETE])
    xs_post = sample_20_xs(study_post)

    # threshold ABOVE the 20 imports: 20 < 100, so asks are still in the random startup phase
    study_startup = build_single_param_study_with_imports(n_startup_trials=100, seed=1)
    xs_startup = sample_20_xs(study_startup)

    # report the sampler settings and the concentration of the asked x around the known optimum x=5
    print(f"[B] completed trials the sampler sees: {completed_post}")
    print(f"[B] n_startup_trials=10  (20>=10 -> TPE model): mean|x-5| of 20 asks = {np.mean(np.abs(xs_post - 5.0)):.3f}")
    print(f"[B] n_startup_trials=100 (20<100 -> random)   : mean|x-5| of 20 asks = {np.mean(np.abs(xs_startup - 5.0)):.3f}")
    print(f"[B] uniform baseline E|x-5| over [-10,10] is about 5.0")
    print(f"[B] first 5 asked x, n_startup=10 : {np.round(xs_post[:5], 2)}")
    print(f"[B] first 5 asked x, n_startup=100: {np.round(xs_startup[:5], 2)}")


# run both parts
print("=== Part A: import the project grid into a fresh TPE study ===")
import_project_grid()
print()
print("=== Part B: imported COMPLETE trials count toward n_startup_trials ===")
n_startup_behavior()
