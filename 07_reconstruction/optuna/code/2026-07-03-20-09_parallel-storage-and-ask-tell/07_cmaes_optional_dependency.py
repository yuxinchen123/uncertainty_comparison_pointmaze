"""Optional-dependency fact: the cmaes package is not installed in this env, and it was NOT
installed for this check. Record how a missing optional backend surfaces: CmaEsSampler constructs
(its cmaes import is deferred), but the first CMA-ES sample raises ModuleNotFoundError. The TPE
sampler used everywhere else in this tutorial needs no optional package."""

import importlib.util
import optuna


def report_cmaes_presence():
    """Print whether the optional cmaes package can be found, without importing or installing it."""
    # find_spec does not import the module; it only checks whether it is installed
    spec = importlib.util.find_spec("cmaes")
    print(f"cmaes package installed: {spec is not None}")


def show_cmaes_failure_point():
    """Construct CmaEsSampler (deferred import) and show the first sample raises for missing cmaes."""
    # quiet optuna's per-trial failure banner so the caught error is the visible line
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # a trivial two-parameter objective
    def objective(trial):
        x = trial.suggest_float("x", -5.0, 5.0)
        y = trial.suggest_float("y", -5.0, 5.0)
        return x * x + y * y

    # construction succeeds because CmaEsSampler defers importing cmaes until it samples
    study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(seed=0))
    print("CmaEsSampler constructed OK (cmaes import is deferred)")

    # the CMA-ES sample (after the single startup trial) triggers the missing import
    try:
        study.optimize(objective, n_trials=3)
        print("optimize succeeded -> cmaes was actually importable")
    except ModuleNotFoundError as exc:
        print(f"first CMA-ES sample raised {type(exc).__name__}: {exc}")


# run the two checks
report_cmaes_presence()
show_cmaes_failure_point()
