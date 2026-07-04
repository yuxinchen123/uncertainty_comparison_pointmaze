"""CmaEsSampler construction: report whether it builds in this env or what error it raises.

CmaEsSampler needs the optional `cmaes` package. The point of this test is to observe what
Optuna does when that dependency's presence is checked, so we catch the specific error type
Optuna raises (ImportError) and print its message verbatim. Any other error is left to raise.
"""

import optuna


# Try to construct CmaEsSampler and run one trial; report success or the exact ImportError message.
def main():
    # announce version
    print("optuna", optuna.__version__)
    # the only failure we expect and want to observe is a missing-dependency ImportError
    try:
        # construct the sampler with an explicit seed
        sampler = optuna.samplers.CmaEsSampler(seed=0)
        # a 2-D objective (CMA-ES needs at least 2 float params) run for a few trials
        study = optuna.create_study(direction="maximize", sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            lambda t: -(t.suggest_float("x", -5, 5) ** 2 + t.suggest_float("y", -5, 5) ** 2),
            n_trials=8)
        # if we got here the package is present
        print("CmaEsSampler constructed and ran; best_value =", round(study.best_value, 4))
        print("=> the 'cmaes' package IS installed in this env")
    except ImportError as e:
        # observe and print the exact message Optuna raises when 'cmaes' is missing
        print("CmaEsSampler raised ImportError:")
        print(repr(str(e)))


if __name__ == "__main__":
    main()
