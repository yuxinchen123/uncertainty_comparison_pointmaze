"""GridSampler per method with method-specific axes, three studies in one storage.

adam and adagrad have a 2 x 8 grid (readout x beta) = 16 cells. sgd1t has a
2 x 8 x 5 x 2 grid (readout x beta x eta0 x t0) = 160 cells. Each method gets its own
GridSampler study (its own search-space dict) in the same journal file. GridSampler
stops the study automatically once every cell is evaluated, so the final trial count
equals the grid size exactly.
"""

import os

import numpy as np
import optuna
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock, JournalStorage

from toy import reward

optuna.logging.set_verbosity(optuna.logging.WARNING)

# the eight log-spaced beta values as log10(beta): exponents -3..4 (beta = 1e-3 .. 1e4)
LOG10_BETA_GRID = list(np.linspace(-3.0, 4.0, 8))          # [-3, -2, -1, 0, 1, 2, 3, 4]
ETA0_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]                 # 5 values
T0_GRID = [1e3, 1e4]                                       # 2 values
READOUT_GRID = ["mse", "l2"]                               # 2 values

# per-method grid search spaces: adam/adagrad have 2 axes, sgd1t has 4
GRID_SPACES = {
    "adam": {"readout": READOUT_GRID, "log10_beta": LOG10_BETA_GRID},
    "adagrad": {"readout": READOUT_GRID, "log10_beta": LOG10_BETA_GRID},
    "sgd1t": {"readout": READOUT_GRID, "log10_beta": LOG10_BETA_GRID,
              "eta0": ETA0_GRID, "t0": T0_GRID},
}

JOURNAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "journal_demo", "grid_3_2_1.log")


def per_method_objective(method):
    """Build a fixed-method grid objective that suggests exactly the method's grid axes."""
    def _obj(trial):
        # every method suggests readout and beta from the grid
        readout = trial.suggest_categorical("readout", READOUT_GRID)
        log10_beta = trial.suggest_categorical("log10_beta", LOG10_BETA_GRID)
        # only sgd1t suggests eta0 and t0 (its grid has the extra two axes)
        if method == "sgd1t":
            eta0 = trial.suggest_categorical("eta0", ETA0_GRID)
            t0 = trial.suggest_categorical("t0", T0_GRID)
            return reward(method, readout, log10_beta, eta0=eta0, t0=t0, base_seed=0)
        return reward(method, readout, log10_beta, base_seed=0)
    return _obj


def open_storage(path):
    """Open a JournalStorage over one file using the open-file lock (current import path)."""
    return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


def main():
    """Run one GridSampler study per method in a shared journal file; report exact cell counts."""
    # start from a clean journal file so reruns are reproducible
    os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
    for p in (JOURNAL_PATH, JOURNAL_PATH + ".lock"):
        if os.path.exists(p):
            os.remove(p)
    storage = open_storage(JOURNAL_PATH)

    print(f"{'method':>8} {'grid_size':>10} {'n_trials_run':>13} {'unique_cells':>13} {'best_value':>11}")
    print("-" * 60)
    for method, space in GRID_SPACES.items():
        # the grid size is the product of the axis lengths
        grid_size = int(np.prod([len(v) for v in space.values()]))
        # one GridSampler study per method, its own search-space dict, seeded for reproducibility
        study = optuna.create_study(direction="maximize",
                                    study_name=f"grid_{method}",
                                    sampler=optuna.samplers.GridSampler(space, seed=0),
                                    storage=storage)
        # pass a budget >= grid size; GridSampler calls study.stop() once every cell is visited
        study.optimize(per_method_objective(method), n_trials=grid_size + 50)

        # count trials and count DISTINCT parameter cells actually evaluated
        n_run = len(study.trials)
        unique_cells = len({tuple(sorted(t.params.items())) for t in study.trials})
        print(f"{method:>8} {grid_size:>10} {n_run:>13} {unique_cells:>13} {study.best_value:>11.3f}")

    # confirm the three grids are the expected sizes and live independently in one storage
    names = optuna.get_all_study_names(storage)
    print(f"\nstudies in the one storage: {sorted(names)}")
    sizes = {m: int(np.prod([len(v) for v in s.values()])) for m, s in GRID_SPACES.items()}
    print(f"expected grid sizes: adam={sizes['adam']}, adagrad={sizes['adagrad']}, sgd1t={sizes['sgd1t']} "
          f"(16 / 16 / 160)")


if __name__ == "__main__":
    main()
