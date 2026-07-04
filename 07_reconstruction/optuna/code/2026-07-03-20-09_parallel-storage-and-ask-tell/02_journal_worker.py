"""Worker process for point 2: attach to an existing journal-file study on NFS and run 5 trials.
Launched as a separate OS process (subprocess.Popen), NOT a thread, so this is real
process-level parallelism sharing one study through the journal file."""

import sys
import hashlib
import numpy as np
import optuna


def substream(base_seed, *parts):
    """Return a numpy generator seeded by hashing a stable name, one stream per named quantity."""
    # build a stable key string then hash it to a 32-bit seed
    key = "::".join(str(p) for p in (base_seed, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def objective(trial):
    """A small noisy quadratic: minimize (x-2)^2 + (y+1)^2 plus per-trial noise."""
    # sample two continuous parameters
    x = trial.suggest_float("x", -10.0, 10.0)
    y = trial.suggest_float("y", -10.0, 10.0)
    # add reproducible per-trial noise keyed by the trial number (not a shared stream)
    noise = substream("obj", trial.number).normal(0.0, 0.5)
    return (x - 2.0) ** 2 + (y + 1.0) ** 2 + noise


def main():
    """Attach to the named study on the given journal file and optimize 5 trials."""
    # command-line args: study name and journal file path
    study_name = sys.argv[1]
    journal_path = sys.argv[2]
    # rebuild the exact same storage object every worker uses (JournalFileOpenLock for NFSv3+)
    lock = optuna.storages.journal.JournalFileOpenLock(journal_path)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(journal_path, lock_obj=lock)
    )
    # load (do not create) the shared study and run 5 trials into it
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=5)
    # report how many trials this process personally completed (for the launcher log)
    print(f"worker pid done; study now has {len(study.trials)} trials", flush=True)


main()
