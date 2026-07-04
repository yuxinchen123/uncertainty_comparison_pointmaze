import os
import tempfile
import numpy as np
import optuna
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock, JournalStorage

optuna.logging.set_verbosity(optuna.logging.WARNING)

# the shared axes, plus sgd1t-only axes
LOG10_BETA = list(np.linspace(-3.0, 4.0, 8))        # 8 beta values: 1e-3 .. 1e4
READOUT = ["mse", "l2"]                             # 2
ETA0 = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]              # 5 (sgd1t only)
T0 = [1e3, 1e4]                                     # 2 (sgd1t only)

# per-method grid search spaces: adam/adagrad have 2 axes (16 cells), sgd1t has 4 (160 cells)
SPACES = {
    "adam":    {"readout": READOUT, "log10_beta": LOG10_BETA},
    "adagrad": {"readout": READOUT, "log10_beta": LOG10_BETA},
    "sgd1t":   {"readout": READOUT, "log10_beta": LOG10_BETA, "eta0": ETA0, "t0": T0},
}


def make_objective(method):
    # build a trivial objective that suggests exactly this method's grid axes
    def _obj(trial):
        trial.suggest_categorical("readout", READOUT)
        trial.suggest_categorical("log10_beta", LOG10_BETA)
        if method == "sgd1t":                      # sgd1t grid carries the extra eta0/t0 axes
            trial.suggest_categorical("eta0", ETA0)
            trial.suggest_categorical("t0", T0)
        return 0.0
    return _obj


path = os.path.join(tempfile.mkdtemp(), "grid_3_2_1.log")
storage = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))

for method, space in SPACES.items():
    grid_size = int(np.prod([len(v) for v in space.values()]))
    study = optuna.create_study(direction="maximize", study_name=f"grid_{method}",
                                sampler=optuna.samplers.GridSampler(space, seed=0), storage=storage)
    # GridSampler calls study.stop() once every cell is visited, so a large budget still stops exactly
    study.optimize(make_objective(method), n_trials=grid_size + 50)
    unique = len({tuple(sorted(t.params.items())) for t in study.trials})
    print(f"{method:>8}: grid_size={grid_size:3d}  trials_run={len(study.trials):3d}  unique_cells={unique:3d}")
