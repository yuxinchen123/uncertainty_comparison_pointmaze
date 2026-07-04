# Optuna 4.9.0 verification: parallel storage on NFS + ask/tell over a file work queue

Date: 2026-07-03. Environment: `conda run -n exploration python` (Python 3.11, optuna 4.9.0,
numpy 1.26.4, scipy 1.16.0). Everything below was run locally on the login node; no Slurm, no wandb.

The folder this lives in is on real NFS: `corezfs02:/p/rlprojects`, mount option `vers=3`
(so `JournalFileOpenLock`, the "NFSv3 or later" lock, is the right lock here).

## What each script verifies

- `01_storage_backends.py` — lists the classes under `optuna.storages` and
  `optuna.storages.journal`; quotes the exact NFS / SQLite guidance from the `JournalFileBackend`
  and `RDBStorage` docstrings and the lock-object docstrings; shows the current import path
  (`optuna.storages.journal.JournalFileBackend`, no warning) versus the deprecated ones
  (`optuna.storages.JournalFileStorage` and top-level `optuna.storages.JournalFileOpenLock`, which
  emit a `FutureWarning`); quotes the `Study.optimize` `n_jobs` docstring (threads).
- `02_journal_nfs_processes.py` (+ `02_journal_worker.py`) — creates a `JournalStorage` study on a
  journal file in this NFS folder, launches 8 concurrent worker **processes**
  (`subprocess.Popen`, not threads); each does `optuna.load_study(...)` and
  `study.optimize(objective, n_trials=5)`. Asserts the shared study ends with exactly 40 trials,
  all COMPLETE, trial numbers 0..39 unique (no collision).
- `03_load_if_exists.py` — `create_study(..., load_if_exists=True)` attaches to an existing study;
  a second `create_study` on the same name WITHOUT `load_if_exists` raises
  `optuna.exceptions.DuplicatedStudyError`.
- `04_ask_tell_controller.py` (+ `04_ask_tell_worker.py`) — the main integration demo. The
  controller owns a `TPESampler(seed=0)` + `JournalStorage` study; it `ask()`s trials, fixes each
  search space with `suggest_*`, writes `{trial_number, params}` as a JSON into `queue/pending/`.
  Four separate worker **processes** claim a pending JSON via atomic `os.rename` into `running/`,
  compute a small noisy objective, write `{trial_number, value}` into `results/`, move the marker to
  `done/`. The controller polls `results/`, calls `study.tell(trial_number, value)` with the plain
  int number, and keeps ~8 trials in flight until 40 are told. Also tells one deliberately failed
  trial with `TrialState.FAIL` and shows the study keeps going.
- `05_import_finished_results.py` — inserts already-finished `(params, value)` pairs with
  `optuna.trial.create_trial(...)` + `study.add_trial(...)` (categorical + log-float distributions
  matching the project grid), no training. Part A: 20 imported trials are COMPLETE before any ask,
  and the next `ask()` succeeds. Part B: imported COMPLETE trials count toward
  `TPESampler.n_startup_trials`.
- `06_parallelism_facts.py` — quotes the docstrings for the three one-sentence facts (n_jobs uses
  threads; SQLite not for parallel + MySQL server note; journal needs only a file path).
- `07_cmaes_optional_dependency.py` — records that the optional `cmaes` package is NOT installed
  (and was NOT installed for this check) and how that surfaces.

## Rerun commands

```
cd /p/rlprojects/RND/07_reconstruction/optuna/code/2026-07-03-20-09_parallel-storage-and-ask-tell/
conda run -n exploration python -u 01_storage_backends.py           2>&1 | tee 01_storage_backends_output.txt
conda run -n exploration python -u 02_journal_nfs_processes.py       2>&1 | tee 02_journal_nfs_processes_output.txt
conda run -n exploration python -u 03_load_if_exists.py              2>&1 | tee 03_load_if_exists_output.txt
conda run -n exploration python -u 04_ask_tell_controller.py         2>&1 | tee 04_ask_tell_controller_output.txt
conda run -n exploration python -u 05_import_finished_results.py     2>&1 | tee 05_import_finished_results_output.txt
conda run -n exploration python -u 06_parallelism_facts.py           2>&1 | tee 06_parallelism_facts_output.txt
conda run -n exploration python -u 07_cmaes_optional_dependency.py   2>&1 | tee 07_cmaes_optional_dependency_output.txt
```

`02_journal_worker.py` and `04_ask_tell_worker.py` are launched as subprocesses by scripts 02 and 04;
you do not run them directly. Scripts 02 and 04 delete their journal file (and script 04 recreates
its `04_queue/` directories) at the start of each run, so reruns start clean.

## Generated artifacts (safe to delete; recreated on rerun)

`02_study_journal.log`, `03_study_journal.log`, `04_study_journal.log`, `04_queue/` (with
`pending/ running/ done/ results/`, the `STOP` sentinel, and `worker_*.log`).

## One sharp edge found while writing script 04

A file-queue worker must list only finalized `*.json` markers in `pending/`. The controller writes
each config as `<n>.json.tmp` then atomically renames it to `<n>.json`. A first version of the
worker listed every name and took the first one; it grabbed a `.json.tmp` mid-write, which both
corrupted the worker's `json.load` and stole the temp file out from under the controller's rename.
All four workers died within the first second, and the controller then waited forever for results
that never came (it looked like a slow run; it was actually a hang — the `done/` markers all carried
timestamps inside one ~1-second window, then nothing). Fix: the worker lists
`[n for n in os.listdir(pending) if n.endswith(".json")]` (a `.json.tmp` name does not end in
`.json`). Script 04 also gives each worker its own `worker_*.log` and the controller a liveness
check that raises if all workers exit early, so a future crash is visible instead of a silent hang.
