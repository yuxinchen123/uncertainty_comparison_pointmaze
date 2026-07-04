"""enqueue_trial works per study: seed a reference configuration into one method's study.

For train run 3.2.1 you often want each method's study to run a known reference cell first
(for example the canonical Adam mse cell at beta=1e2). study.enqueue_trial adds that exact
configuration to the front of one study's queue; the next call to study.optimize runs it as
trial 0. We enqueue it into the adam study only and confirm it lands there and nowhere else.
"""

import os

import optuna
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock, JournalStorage

from toy import reward

optuna.logging.set_verbosity(optuna.logging.WARNING)

# the canonical reference cell for adam: mse readout at beta = 1e2 (log10_beta = 2.0)
ADAM_REFERENCE = {"readout": "mse", "log10_beta": 2.0}

JOURNAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "journal_demo", "enqueue_3_2_1.log")


def adam_objective(trial):
    """Fixed-method adam objective: suggest readout and beta, return the noisy reward."""
    # adam has no eta0/t0; only readout and beta are searched
    readout = trial.suggest_categorical("readout", ["mse", "l2"])
    log10_beta = trial.suggest_float("log10_beta", -3.0, 4.0)
    return reward("adam", readout, log10_beta, base_seed=0)


def sgd1t_objective(trial):
    """Fixed-method sgd1t objective (kept separate to show enqueue is per study)."""
    # sgd1t adds eta0 and t0 on top of readout and beta
    readout = trial.suggest_categorical("readout", ["mse", "l2"])
    log10_beta = trial.suggest_float("log10_beta", -3.0, 4.0)
    eta0 = trial.suggest_float("eta0", 1e-3, 1e-1, log=True)
    t0 = trial.suggest_categorical("t0", [1e3, 1e4])
    return reward("sgd1t", readout, log10_beta, eta0=eta0, t0=t0, base_seed=0)


def open_storage(path):
    """Open a JournalStorage over one file using the open-file lock (current import path)."""
    return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))


def main():
    """Enqueue the adam reference cell into the adam study only, then optimize both studies."""
    # start from a clean journal file so reruns are reproducible
    os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
    for p in (JOURNAL_PATH, JOURNAL_PATH + ".lock"):
        if os.path.exists(p):
            os.remove(p)
    storage = open_storage(JOURNAL_PATH)

    # create the two method studies in one storage
    adam_study = optuna.create_study(direction="maximize", study_name="3_2_1_adam",
                                     sampler=optuna.samplers.TPESampler(seed=0), storage=storage)
    sgd1t_study = optuna.create_study(direction="maximize", study_name="3_2_1_sgd1t",
                                      sampler=optuna.samplers.TPESampler(seed=0), storage=storage)

    # enqueue the reference cell into the ADAM study only (runs before any sampler suggestion)
    adam_study.enqueue_trial(ADAM_REFERENCE)

    # run each study; the adam study runs the enqueued cell as trial 0, then samples the rest
    adam_study.optimize(adam_objective, n_trials=8)
    sgd1t_study.optimize(sgd1t_objective, n_trials=8)

    # confirm trial 0 of the adam study is exactly the enqueued reference cell
    t0 = adam_study.trials[0]
    print("adam study, trial 0 (the enqueued reference cell):")
    print(f"  params        : {t0.params}")
    print(f"  matches ref   : {t0.params == ADAM_REFERENCE}")
    print(f"  value (noisy) : {t0.value:.3f}\n")

    # confirm the sgd1t study never received the adam reference cell (enqueue is per study)
    sgd1t_has_ref = any(t.params.get("readout") == ADAM_REFERENCE["readout"]
                        and t.params.get("log10_beta") == ADAM_REFERENCE["log10_beta"]
                        and "eta0" not in t.params
                        for t in sgd1t_study.trials)
    print("sgd1t study did NOT get the adam reference cell (enqueue is per study): "
          f"{not sgd1t_has_ref}")
    print(f"sgd1t study trial 0 params: {sgd1t_study.trials[0].params}")

    # show the enqueued trial is a normal COMPLETE trial afterwards (usable in the best search)
    print(f"\nadam study trial count: {len(adam_study.trials)}; "
          f"trial 0 state: {t0.state.name}; adam best value: {adam_study.best_value:.3f}")


if __name__ == "__main__":
    main()
