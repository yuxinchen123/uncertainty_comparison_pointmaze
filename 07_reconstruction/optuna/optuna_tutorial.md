# Optuna tutorial — adaptive hyperparameter search for the 07_reconstruction sweeps

Written 2026-07-03, expanded the same day with the implementation-level sections (7–10). Everything
in this document was verified by actually running it in this project's environment: **Optuna
4.9.0**, installed 2026-07-03 into the `exploration` conda env (`conda run -n exploration python
...`, Python 3.11, numpy 1.26.4, scipy 1.16.0, torch 2.10.0). Every code block that shows an
"Output (observed)" was executed; the full scripts, with their raw outputs saved next to them, live
in nine folders under `07_reconstruction/optuna/code/` (listed in §12). Where a statement comes
from Optuna's own documentation or source rather than from a run, it is quoted and marked as such.

## Table of contents

1. [The problem Optuna solves here](#1-the-problem-optuna-solves-here)
2. [Optuna's objects, defined](#2-optunas-objects-defined)
3. [Basic functions and usage](#3-basic-functions-and-usage)
   - [3.1 Declaring parameters: the `suggest_*` calls](#31-declaring-parameters-the-suggest_-calls)
   - [3.2 Running: `study.optimize`](#32-running-studyoptimize)
   - [3.3 Reading results](#33-reading-results)
   - [3.4 Forcing specific combinations: `enqueue_trial`](#34-forcing-specific-combinations-enqueue_trial)
   - [3.5 Driving trials without `study.optimize`: ask and tell](#35-driving-trials-without-studyoptimize-ask-and-tell)
   - [3.6 Sharing a study across processes: storage](#36-sharing-a-study-across-processes-storage)
4. [Fitting Optuna to this project's infrastructure](#4-fitting-optuna-to-this-projects-infrastructure)
   - [4.1 What maps to what](#41-what-maps-to-what)
   - [4.2 Where each piece runs: the controller pattern](#42-where-each-piece-runs-the-controller-pattern)
   - [4.3 Storage on this cluster: journal file, not SQLite](#43-storage-on-this-cluster-journal-file-not-sqlite)
   - [4.4 Importing the finished run-3.1.2 results](#44-importing-the-finished-run-312-results)
5. [Question 1 — grid search, and the "10 seeds, then threshold" plan](#5-question-1--grid-search-and-the-10-seeds-then-threshold-plan)
   - [5.1 Yes: `GridSampler` is exhaustive grid search](#51-yes-gridsampler-is-exhaustive-grid-search)
   - [5.2 Your allocation rule, written down](#52-your-allocation-rule-written-down)
   - [5.3 Can Optuna do this? Yes — as roughly ten lines of your own pruning rule](#53-can-optuna-do-this-yes--as-roughly-ten-lines-of-your-own-pruning-rule)
   - [5.4 What the built-in pruners would do instead, on the same simulation](#54-what-the-built-in-pruners-would-do-instead-on-the-same-simulation)
   - [5.5 Running the rule at cluster scale: who loops over seeds?](#55-running-the-rule-at-cluster-scale-who-loops-over-seeds)
   - [5.6 Where the rule is unreliable: decisions near the bar at 10 seeds](#56-where-the-rule-is-unreliable-decisions-near-the-bar-at-10-seeds)
6. [Question 2 — searching intervals instead of grids](#6-question-2--searching-intervals-instead-of-grids)
   - [6.1 Set or interval? You choose, per parameter](#61-set-or-interval-you-choose-per-parameter)
   - [6.2 The samplers compared](#62-the-samplers-compared)
   - [6.3 The technique behind `TPESampler`: Tree-structured Parzen Estimator](#63-the-technique-behind-tpesampler-tree-structured-parzen-estimator)
   - [6.4 The technique behind `GPSampler`: Bayesian optimization with a Gaussian process](#64-the-technique-behind-gpsampler-bayesian-optimization-with-a-gaussian-process)
   - [6.5 `CmaEsSampler`, for completeness](#65-cmaessampler-for-completeness)
   - [6.6 Handling a noisy objective](#66-handling-a-noisy-objective)
7. [TPE as Optuna implements it](#7-tpe-as-optuna-implements-it)
   - [7.1 The exact defaults and what "independent mode" fits](#71-the-exact-defaults-and-what-independent-mode-fits)
   - [7.2 The Parzen mixture: one kernel per observation, a prior, bandwidths, magic clip](#72-the-parzen-mixture-one-kernel-per-observation-a-prior-bandwidths-magic-clip)
   - [7.3 One ask, step by step: 24 candidates from the good density](#73-one-ask-step-by-step-24-candidates-from-the-good-density)
   - [7.4 Why the density ratio is expected improvement: the derivation, checked numerically](#74-why-the-density-ratio-is-expected-improvement-the-derivation-checked-numerically)
   - [7.5 A mini-TPE from scratch in numpy](#75-a-mini-tpe-from-scratch-in-numpy)
8. [The Gaussian-process sampler, in detail](#8-the-gaussian-process-sampler-in-detail)
   - [8.1 The surrogate Optuna fits](#81-the-surrogate-optuna-fits)
   - [8.2 The posterior and the acquisition, from scratch in numpy](#82-the-posterior-and-the-acquisition-from-scratch-in-numpy)
   - [8.3 Optuna's asks agree with the from-scratch acquisition landscape](#83-optunas-asks-agree-with-the-from-scratch-acquisition-landscape)
9. [Multi-algorithm sweeps: the best configuration per method (run 3.2.1)](#9-multi-algorithm-sweeps-the-best-configuration-per-method-run-321)
   - [9.1 Run 3.2.1's search problem](#91-run-321s-search-problem)
   - [9.2 Method-specific parameters in one study: conditional suggests work](#92-method-specific-parameters-in-one-study-conditional-suggests-work)
   - [9.3 Why one shared study fails the best-per-method question](#93-why-one-shared-study-fails-the-best-per-method-question)
   - [9.4 Reading a mixed study anyway, and the max-of-noise trap](#94-reading-a-mixed-study-anyway-and-the-max-of-noise-trap)
   - [9.5 The recommended layout: one study per method, one journal file](#95-the-recommended-layout-one-study-per-method-one-journal-file)
10. [Parallel Optuna at this cluster's scale](#10-parallel-optuna-at-this-clusters-scale)
    - [10.1 The compute you actually have](#101-the-compute-you-actually-have)
    - [10.2 The two parallel patterns, and where the sampler runs](#102-the-two-parallel-patterns-and-where-the-sampler-runs)
    - [10.3 At launch, the first wave is random search](#103-at-launch-the-first-wave-is-random-search)
    - [10.4 Parallel asks cluster; `constant_liar` spreads them](#104-parallel-asks-cluster-constant_liar-spreads-them)
    - [10.5 Pruning works across processes — but pass the pruner everywhere](#105-pruning-works-across-processes--but-pass-the-pruner-everywhere)
    - [10.6 Dead workers: the journal file has no heartbeat](#106-dead-workers-the-journal-file-has-no-heartbeat)
    - [10.7 What the journal file costs at scale](#107-what-the-journal-file-costs-at-scale)
11. [A concrete design for train run 3.2.1](#11-a-concrete-design-for-train-run-321)
12. [Version notes and the runnable example folders](#12-version-notes-and-the-runnable-example-folders)

---

## 1. The problem Optuna solves here

Train run 3.1.2 was a grid fixed in advance: 2 feature normalizations, 3 ridge values, 2 clip
settings, and 3 values of the intrinsic coefficient — 36 configurations — each run for 50 seeds, so
1800 training runs of 1M steps. At the measured arm-B pace (67,352 steps per CPU-hour), one run is
about 15 hours on one CPU, so the whole grid costs roughly 27,000 CPU-hours. Every configuration gets
the same 50 seeds no matter what the first seeds show — yet the finished analysis says 7 of the 12
cells (at their best coefficient) scored below 20, and several cells collapse to about 0. Most of the
computation confirmed, at full cost, results that were already clear after a handful of seeds.

"Hard to grid search the hyperparameters and allocate computation to high-value combinations" splits
into two separate problems, and Optuna has one mechanism for each:

1. **Stop spending on combinations the data has already shown to be bad.** In Optuna this is a **pruner**:
   a rule that ends a partially-evaluated trial early. Your "run 10 seeds, then continue only if the
   mean plus the 95% confidence interval still reaches 35" plan is exactly a pruning rule (§5).
2. **Choose the *next* combination using the results so far.** In Optuna this is a **sampler**: grid
   and random samplers ignore history; the adaptive samplers (TPE, Gaussian process) concentrate new
   trials where past results were good, and they can search continuous intervals such as every ridge
   value in [1e-6, 1e-2] instead of the three points a grid pre-commits to (§6).

Optuna itself never trains anything. It is a bookkeeping-plus-decision library: it stores which
combinations were tried and what they scored, and it answers two questions — "what should I try
next?" (sampler) and "should I stop this one early?" (pruner). The training stays in `train.py`.

---

## 2. Optuna's objects, defined

Optuna has five core objects. The names below are the library's literal class and argument names.

1. **Objective** — a Python function you write, `objective(trial) -> float`. It receives a `Trial`
   object, asks it for parameter values, runs whatever evaluation you want (here: a training run, or
   a batch of training runs), and returns one number (here: final evaluation reward). Optuna
   maximizes or minimizes that number according to the study's `direction`.
2. **Trial** (`optuna.trial.Trial`) — one evaluation of the objective at one parameter combination.
   The trial provides parameter values through `trial.suggest_*` calls and records everything: the
   parameters (`trial.params`), the returned value, intermediate values you report, and the final
   state. A trial ends in one of three states: `COMPLETE` (returned a value), `PRUNED` (stopped
   early), or `FAIL` (raised an error).
3. **Study** (`optuna.create_study`) — the collection of all trials for one search, plus the goal
   (`direction="maximize"` here, since the metric is a reward). `study.optimize(objective, n_trials=...)`
   runs the loop; `study.best_trial` holds the best completed trial.
4. **Sampler** (`study`'s `sampler=` argument) — the rule that picks the next combination:
   `GridSampler`, `RandomSampler`, `TPESampler` (the default), `GPSampler`, `CmaEsSampler`. Detailed
   in §5–§6.
5. **Pruner** (`study`'s `pruner=` argument) — the rule that decides whether a partially-run trial
   should stop. The objective cooperates by calling `trial.report(value, step)` as it goes and
   checking `trial.should_prune()`. Detailed in §5.

One more piece matters on a cluster: **storage** (`create_study(storage=...)`). By default a study
lives in the memory of one Python process and disappears with it. With a storage backend (a file or a
database), many processes on many nodes can attach to the same study at once — that is how a sweep
with hundreds of Slurm workers shares one search (§4.3).

A note on vocabulary: in this project's language, one Optuna "trial" is *not* necessarily one
training run. A trial is one evaluation of the objective — and the objective can run one training run
or aggregate ten seeds' training runs, whichever you define (§5.5 discusses which to choose).

### The smallest complete example

Everything above in one short script (runnable as-is; observed output follows):

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # silence per-trial log lines

def objective(trial):
    # toy stand-in for a training run: reward peaks at beta = 1e-2, normalization "unit" adds 5
    beta = trial.suggest_float("beta", 1e-3, 1e-1, log=True)
    normalization = trial.suggest_categorical("normalization", ["none", "unit"])
    return -abs(beta - 1e-2) * 100.0 + (5.0 if normalization == "unit" else 0.0)

sampler = optuna.samplers.TPESampler(seed=12345)          # always pass a seed
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=15)

print("len(study.trials):", len(study.trials))
print("best_trial.value:", study.best_trial.value)
print("best_params:", study.best_params)
```

Output (observed, Optuna 4.9.0):

```
len(study.trials): 15
best_trial.value: 4.765721447163506
best_params: {'beta': 0.007657214471635064, 'normalization': 'unit'}
```

---

## 3. Basic functions and usage

### 3.1 Declaring parameters: the `suggest_*` calls

The search space is declared *by the calls the objective makes* (Optuna calls this "define-by-run"):
there is no separate search-space file for the adaptive samplers — whatever the objective asks for is
the space. The four forms, runnable as-is:

```python
import optuna

def objective(trial):
    # declare the search space by the calls made here; the returned value is unused in this demo
    norm  = trial.suggest_categorical("normalization", ["none", "unit"])   # a finite SET
    beta  = trial.suggest_float("beta_linear", 0.001, 0.1)          # linear interval
    ridge = trial.suggest_float("ridge_log", 1e-6, 1e-2, log=True)  # log interval
    n_layers = trial.suggest_int("n_layers", 1, 4)
    print(f"trial {trial.number}: norm={norm!r} beta={beta:.5f} ridge={ridge:.3e} n_layers={n_layers}")
    return 0.0

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective, n_trials=6)
```

Output (observed, Optuna 4.9.0; first four trials):

```
trial 0: norm='unit' beta=0.06067 ridge=1.512e-04 n_layers=2
trial 1: norm='none' beta=0.08929 ridge=7.156e-03 n_layers=2
trial 2: norm='none' beta=0.05724 ridge=5.039e-03 n_layers=1
trial 3: norm='none' beta=0.08343 ridge=1.296e-03 n_layers=4
```

Two things to read off the output. The `log=True` float comes back spread across decades
(`1.5e-04`, `7.2e-03`, `5.0e-03`, `1.3e-03`), while the linear float clusters in the upper part of
its range (`0.061`, `0.089`, `0.057`, `0.083` — on a linear scale over [0.001, 0.1], about 90% of
uniform draws sit above 0.01). Scale parameters like ridge and beta, whose interesting values span
decades, must use `log=True`, or the sampler will almost never visit the small decades.

Which form to use is the "set or interval" question of Question 2, answered in §6.1: `GridSampler`
needs finite sets; the adaptive samplers work with sets *and* intervals and are more effective with
intervals.

### 3.2 Running: `study.optimize`

```python
study.optimize(objective, n_trials=50)              # run 50 trials, then stop
study.optimize(objective, timeout=3600)             # or: stop after one hour
study.optimize(objective, n_trials=50, catch=(ValueError,))  # a raised ValueError marks that
                                                    # trial FAIL and the loop continues
```

Three verified behaviors worth knowing:

- Without `catch=`, an exception raised inside the objective stops the whole `optimize` call (the
  trial is still recorded as FAIL).
- `n_trials=None` runs forever for most samplers — but under `GridSampler` the study stops by itself
  once the grid is exhausted (§5.1).
- `study.optimize(..., n_jobs=8)` is threads inside one process, not processes. Optuna's own
  docstring: "`n_jobs` allows parallelization using `threading` and may suffer from Python's GIL. It
  is recommended to use process-based parallelization if `func` is CPU bound." For CPU-bound SAC
  training, parallelism must come from separate processes sharing a storage (§4.3), never `n_jobs`.

### 3.3 Reading results

```python
study.best_trial          # FrozenTrial: .number, .value, .params
study.best_params         # == study.best_trial.params
study.trials              # every trial, any state
study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))   # filter by state
study.trials_dataframe()  # pandas frame: number, value, datetime_start,
                          # datetime_complete, duration, params_<name>..., state
```

Verified: `PRUNED` and `FAIL` trials carry `value == None` and are never `best_trial`;
`trials_dataframe()` creates one `params_<name>` column per suggested parameter.

### 3.4 Forcing specific combinations: `enqueue_trial`

`study.enqueue_trial({"beta": 1e-3, "normalization": "unit"})` puts an exact combination at the front
of the queue; the sampler takes over after the queued trials are used. Verified: the first trials use
exactly the enqueued parameter values, in enqueue order. This is how you guarantee reference
configurations (for example the run-2 replica cell) are evaluated inside an otherwise adaptive study.

### 3.5 Driving trials without `study.optimize`: ask and tell

`study.optimize` assumes the evaluation happens inside the objective function, inside the calling
process. On this cluster the evaluation is a 15-hour `train.py` run on some other node, so the
function-call shape does not fit. The ask-and-tell interface separates the two halves:

```python
trial = study.ask()                       # sampler picks a combination NOW
beta = trial.suggest_float("beta", 1e-3, 1e-1, log=True)   # fix its parameters
# ... hand {trial.number, params} to whatever actually runs the training ...
study.tell(trial.number, 43.7)            # report the result LATER (int number works)
study.tell(other_number, state=optuna.trial.TrialState.FAIL)  # or report a crash
```

Verified end-to-end with a file work queue and 4 separate worker processes (§4.2): `study.tell`
accepts the plain integer trial number, a FAIL tell does not stop the study, and the sampler keeps
proposing new combinations afterward.

### 3.6 Sharing a study across processes: storage

```python
lock = optuna.storages.journal.JournalFileOpenLock("study.log")
storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend("study.log", lock_obj=lock))
study = optuna.create_study(study_name="sweep", storage=storage,
                            direction="maximize", load_if_exists=True)
# any other process, any node:
study = optuna.load_study(study_name="sweep", storage=storage)
```

The study now lives in the file `study.log`; every process that attaches sees the same trials.
`load_if_exists=True` makes `create_study` attach to an existing study instead of raising
`DuplicatedStudyError` (both behaviors verified). Why this exact backend on this cluster: §4.3.

---

## 4. Fitting Optuna to this project's infrastructure

### 4.1 What maps to what

| Optuna object | In this project |
|---|---|
| objective value | final evaluation extrinsic reward (100-episode standalone eval at 1M steps) |
| trial parameters | the swept knobs: normalization, ridge, clip, beta |
| one trial | one configuration's evaluation (one seed, or an aggregate over seeds — §5.5) |
| sampler | replaces the fixed config list that `build_queue.py` writes |
| pruner / early stop | replaces "every configuration always gets all 50 seeds" |
| storage (journal file) | lives in the run folder, next to `queue/` and `data/` |
| `a_seed` | NOT an Optuna parameter — a repetition index the controller assigns (§5.5) |

The training seed deserves its own warning: never let an adaptive sampler "optimize" `a_seed`. The
sampler would learn which seeds happened to score well, which is meaningless. Seeds are repetitions
of the same combination; the controller (or the objective's internal loop) assigns them `0, 1, 2, ...`
per combination. (A *grid* over seeds is different and fine — §5.1 point 3 — because the grid does
not steer toward lucky seeds; it just enumerates them.)

### 4.2 Where each piece runs: the controller pattern

Today's sweep machinery (run-3.1.2 form): `build_queue.py` writes every config JSON into
`queue/<sweep_id>/pending/` up front; each of ~100–500 Slurm workers claims one by an atomic
`os.rename` into `running/`, runs `train.py` as a subprocess, and moves the marker to `done/` or
`failed/`. The training results are written to per-run JSONs under `data/<sweep_id>/local/`.

Optuna is added as a **controller process** that replaces the build-everything-up-front step with an
ask-as-you-go loop. The workers need exactly one change — the claim loop must list only finalized
files — because a controller writes into `pending/` while workers are running, which the current
`worker.py` never has to deal with (today `build_queue.py` finishes writing the whole queue before
any worker starts):

1. The controller owns the study (sampler + storage). It calls `study.ask()`, fixes the parameters,
   and writes the config JSON into `pending/` — same file format the workers already read, plus a
   `trial_number` field.
2. Workers claim, train, and write results as today, except the claim loop filters to finalized
   names: `sorted(n for n in os.listdir(pending) if n.endswith(".json"))`.
3. The controller polls the finished-results location, calls `study.tell(trial_number, reward)` (or
   `state=FAIL` for a crashed run), and asks new trials to keep a target number of configs in the
   queue.

This was verified end-to-end in miniature (controller + 4 worker processes + 40 trials over a
pending/running/done queue on the real filesystem — folder in §12). Two details from that
verification that were only found because a first version failed:

- **Workers must claim only finalized files.** The controller writes `<n>.json.tmp` then atomically
  renames to `<n>.json`; a worker that lists `pending/` and grabs a `.json.tmp` mid-write reads
  partial JSON and steals the temp file out from under the rename. In the first version of the test
  this crashed all 4 workers within a second, and the controller then waited forever for results
  that could never arrive. Filter on `name.endswith(".json")` — and publish results the same way
  (write temp file, then `os.rename`).
- The controller should watch for "all workers exited" and raise, so a queue that can no longer
  make progress becomes a visible error, not an endless quiet wait.

The controller is light (it sleeps between polls); it can run on the login node or as a 1-CPU Slurm
job, same as the eval-job convention.

### 4.3 Storage on this cluster: journal file, not SQLite

`/p/rlprojects` is NFS (verified: `findmnt` shows `corezfs02:/p/rlprojects`, type `nfs`, mount option
`vers=3`). That rules out the storage most tutorials show first:

- SQLite (`storage="sqlite:///study.db"`) depends on `fcntl()` file locking. Optuna's
  `JournalFileBackend` docstring, quoted: "SQLite3 might not work on NFS (Network File System) since
  `fcntl()` file locking is broken on many NFS implementations." And the `RDBStorage` docstring:
  "We would never recommend SQLite3 for parallel optimization."
- A real database server (MySQL/PostgreSQL through `RDBStorage`) is Optuna's documented answer for
  very large parallel sweeps — but this cluster does not run one, and the sweep does not need one.
- The **journal file backend** was built for exactly this situation: one append-only log file plus a
  lock protocol that works on NFS. `JournalFileOpenLock` is the lock documented for "NFSv3 or later
  on kernel 2.6 or later" — this mount is NFSv3, so it is the right lock here
  (`JournalFileSymlinkLock` exists for older NFS).

Verified on the real filesystem: 8 concurrent worker *processes* attached to one journal-file study
and ran 40 trials total — all 40 recorded, trial numbers exactly 0..39, no duplicates, no errors.

```python
# The import paths that are current in 4.9.0 (the old top-level names
# optuna.storages.JournalFileStorage / optuna.storages.JournalFileOpenLock still work
# but print FutureWarning "deprecated in v4.0.0 ... removed in v6.0.0"):
lock = optuna.storages.journal.JournalFileOpenLock(path)
backend = optuna.storages.journal.JournalFileBackend(path, lock_obj=lock)
storage = optuna.storages.JournalStorage(backend)
```

One documented limit to respect (JournalFileBackend docstring): the journal backend "doesn't support a
high level of write concurrency" — fine here, because each write happens once per training run
(hours apart per worker), not per step. With the controller pattern of §4.2 only the controller
writes at all.

### 4.4 Importing the finished run-3.1.2 results

The ~900+ finished run-3.1.2 runs are (parameters, reward) pairs, and Optuna can ingest finished
results without re-running anything. This means an adaptive study does not have to start from
zero — it starts knowing everything the grid already measured:

```python
from optuna.distributions import CategoricalDistribution, FloatDistribution

dists = {
    "normalization": CategoricalDistribution(["none", "unit"]),
    "ridge":         FloatDistribution(1e-6, 1e-2, log=True),
    "clip":          CategoricalDistribution(["inf", "5"]),
    "beta":          FloatDistribution(1e-3, 1e-1, log=True),
}
for params, reward in finished_combinations:      # ONE entry per combination (see below)
    study.add_trial(optuna.trial.create_trial(
        params=params, distributions=dists, value=reward))   # state defaults to COMPLETE
```

One decision before importing: the per-run JSONs under `data/<sweep_id>/local/` are **per seed**
(one JSON per training run), but the trial values a study compares must all live on one scale. If
new trials will carry 10-seed batch means (§6.6), do not import one trial per seed — single-seed
values are bimodal (near 0 or near 80) and would distort the sampler's good/bad split against the
smoother batch means. Aggregate first: one imported trial per *combination*, with `value` = that
combination's mean final reward over its finished seeds (record the seed count with
`trial.set_user_attr` so the analysis can weight it later).

Verified: 20 imported results show up as COMPLETE trials before any ask; the next `study.ask()`
continues from trial number 20; and — the part that matters — **imported trials count toward
`TPESampler`'s random startup phase** (`n_startup_trials`, default 10). With 20 informative imports
and `n_startup_trials=10`, the very first asked trial already used the TPE model and fell near the
optimum of the one-parameter test problem (mean distance 0.306 from the peak, versus 7.9 when the
startup threshold was set above the import count). With the finished grid imported as 36
combination means (built from the 900+ finished seed runs), a new study is past the 10-trial
startup phase from its first ask.

---

## 5. Question 1 — grid search, and the "10 seeds, then threshold" plan

### 5.1 Yes: `GridSampler` is exhaustive grid search

`optuna.samplers.GridSampler(search_space)` takes a dict of finite value lists and evaluates every
combination exactly once. Run-3.1.2's exact grid:

```python
SEARCH_SPACE = {
    "normalization": ["none", "unit"],
    "ridge": [1e-6, 1e-4, 1e-2],
    "clip": ["inf", "5"],
    "beta": [1e-3, 1e-2, 1e-1],
}  # 2 x 3 x 2 x 3 = 36 combinations

def objective(trial):
    for name, choices in SEARCH_SPACE.items():
        trial.suggest_categorical(name, choices)
    return train_and_eval(trial.params)     # placeholder for the real evaluation

sampler = optuna.samplers.GridSampler(SEARCH_SPACE, seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=None)    # stops by itself after all 36
```

Seven verified behaviors (each one is a way to get a wrong sweep if unknown):

1. **`n_trials` is only an upper bound.** With the 36-point grid, `n_trials=None` ran exactly 36
   trials and stopped by itself; `n_trials=50` also ran exactly 36 (no re-runs, no error). A 12-cell
   grid asked for 60 trials runs 12. GridSampler never pads a budget by repeating combinations.
2. **Every combination visited exactly once** — verified by collecting the 36 parameter tuples into
   a set (36 unique, 36 calls).
3. **The seed can be one more grid axis.** Adding `"a_seed": list(range(10))` gives a 360-point grid
   and all 360 (combination, seed) pairs ran exactly once. This reproduces today's
   config-times-seed queue semantics inside Optuna. The objective must actually call
   `trial.suggest_categorical("a_seed", ...)` and pass the value to training — see point 6.
4. **Visit order is a seeded shuffle, not nested-loop order.** Two studies with `seed=42` visit the
   36 cells in the same order; `seed=99` gives a different order; neither matches the Cartesian
   nested-loop order. Note for our conventions: the run-id rule (early seeds finish first) is an
   ordering promise — with GridSampler you would recover a deterministic order by fixing the
   sampler seed, but not the seed-outermost order; the controller pattern (§4.2) can enforce any
   order it wants instead.
5. **Re-running `optimize` on an exhausted grid is not a clean no-op**: it runs one extra trial on an
   already-visited combination and prints a warning ("re-evaluating a configuration because the grid
   has been exhausted"). Do not loop `optimize` expecting it to idle.
6. **Grid keys and `suggest_*` calls must match exactly, both directions.** Suggesting a name absent
   from the grid raises `ValueError: The parameter name, ridge, is not found in the given grid.` —
   an explicit error, easy to notice. The reverse produces no error and is dangerous: a grid key the
   objective never suggests silently
   multiplies the trial count (each unused value becomes its own trial) and never appears in
   `trial.params`. With an `a_seed` grid axis that the objective forgets to suggest, you get 10
   identical runs per cell and no record of which seed ran.
7. **Under GridSampler, the low/high arguments of `suggest_float` have no effect.** A grid value
   outside the stated range is used anyway, with only a `UserWarning`. Keep the grid list and the
   range arguments consistent by hand; nothing validates them for you.

So: the full factorial you already run is one `GridSampler` study. On its own that changes the
bookkeeping, not the cost. The savings come from the second half of your plan.

### 5.2 Your allocation rule, written down

After a combination has run $n$ seeds with final evaluation rewards $R_1, \dots, R_n$, define the
mean, the sample variance $s^2$ (its square root $s$ is the sample standard deviation), and the
upper limit of the 95% confidence interval:

$$\bar{R} = \dfrac{1}{n}\sum_{i=1}^{n} R_i, \qquad s^2 = \dfrac{1}{n-1}\sum_{i=1}^{n} (R_i - \bar{R})^2, \qquad U = \bar{R} + 1.96 \cdot \dfrac{s}{\sqrt{n}}.$$

Your rule: give every combination 10 seeds; afterwards, keep spending seeds only on combinations with
$U \ge 35$ — the combinations whose data cannot yet rule out a true mean of 35. ($U$ is the upper
limit of the two-sided 95% interval; 1.96 is the normal-distribution value. "Mean + 95% confidence
interval $\ge$ 35" in your phrasing = $U \ge 35$ here.) Stopping when $U < 35$ means stopping only
when the whole interval has moved clearly below the bar, which is the optimistic (safe-to-keep)
direction.

### 5.3 Can Optuna do this? Yes — as roughly ten lines of your own pruning rule

**What Optuna offers natively** is the pruning *interface*, and it is exactly shaped for this:

- the objective reports progress: `trial.report(running_mean, step=n_seeds_so_far)`;
- something decides to stop: either a built-in pruner consulted through `trial.should_prune()`, or
  your own condition;
- the objective ends the trial by raising `optuna.TrialPruned()` — pruning is **cooperative**.
  Verified explicitly: a pruner whose decision is always "stop" changes nothing unless the objective
  raises; `should_prune()` returning `True` does nothing by itself.

**No built-in pruner implements an absolute confidence-bound rule.** All built-ins present in 4.9.0
(Median, Percentile, SuccessiveHalving, Hyperband, Wilcoxon, Threshold, Patient, Nop) either compare
a trial against *other trials* (median/percentile/top-fraction/best) or compare a single reported
value against a fixed bound (Threshold — no interval, no running mean). Your rule compares a
*confidence bound on the running mean* against an *absolute bar*, so you write it yourself — and the
simplest correct form needs no pruner object at all:

```python
import optuna, numpy as np

def objective(trial):
    params = suggest_configuration(trial)        # normalization / ridge / clip / beta
    rewards = []
    for seed_index in range(50):                 # cap: 50 seeds per combination
        rewards.append(run_one_seed(params, seed_index))   # one training run's final eval reward
        n = len(rewards)
        mean = float(np.mean(rewards))
        sd = float(np.std(rewards, ddof=1)) if n > 1 else float("inf")
        upper = mean + 1.96 * sd / np.sqrt(n)
        trial.report(mean, step=n)               # keep the partial curve readable afterwards
        if n >= 10 and upper < 35.0:             # the rule
            raise optuna.TrialPruned()
    return float(np.mean(rewards))
```

Verified on a 12-cell simulation calibrated to run 3.1.2 (true cell means 52.8, 35.2, 29, 26.1, 26,
17.9, 17.3, 9.6, 4.9, 1, 0.5, 0.1; per-seed outcomes bimodal — success draws near 80, failure draws
near 0 — matching the real per-seed standard deviation of roughly 35):

- At simulation base seed 0: **252 seed-runs instead of 600** (the 12-cell, 50-seed full grid), and
  the surviving set was exactly the two cells whose true mean is at or above 35.
- The identical rule as a custom `BasePruner` subclass produced the identical outcome (the subclass
  needs exactly one method, `prune(study, trial) -> bool`; it reads `trial.intermediate_values`, and
  anything extra it needs — the running standard deviation — must be passed through
  `trial.set_user_attr`, which makes the subclass strictly more plumbing than the inline `if`).
  Prefer the inline form.
- Two interface facts that protect you: pruned trials keep their reported curve
  (`trial.intermediate_values` survives with state `PRUNED`), so partial results stay analyzable;
  and if you never pass a pruner but do call `trial.should_prune()`, you are silently consulting the
  **default pruner, which is `MedianPruner`, not a no-op** — for this rule, do not call
  `should_prune()` at all (or pass `optuna.pruners.NopPruner()` explicitly).

### 5.4 What the built-in pruners would do instead, on the same simulation

For calibration, the same 12-cell simulation under the three built-ins closest to your intent. Seeds
per cell capped at 50; "seed-runs" counts every training run the scheme performs (full grid = 600).

| scheme | decision rule | seed-runs | cells run to 50 seeds |
|---|---|---|---|
| your rule (inline) | absolute: 95% upper limit below 35 | 252 | the two cells at or above 35 |
| WilcoxonPruner | relative: signed-rank test vs the best trial | 214 | best cell (second stopped at 48) |
| SuccessiveHalvingPruner | relative: top fraction per rung | 160 | best cell only |
| HyperbandPruner | halving over several start budgets | 400 | four cells |

(One simulation, base seed 0 — the numbers move with the noise; the *shape* of the comparison is the
point.) The built-ins answer "which single combination is best" and stop everything that is merely
*worse than the leader* — SuccessiveHalving pruned a cell whose 10-seed mean was 41.6, and Wilcoxon
stopped the true-mean-35.2 cell, both simply because the 52.8 cell existed. Your rule answers a
different question — "which combinations clear an absolute bar" — and keeps every such combination.
Use the built-ins when you want the single winner cheaply; use your rule when the deliverable is the
set of combinations above a standard (as in this project, where the analysis compares all surviving
cells). Two notes on the built-ins, verified:

- `WilcoxonPruner` reports differently: you report each seed's *own* reward at `step=seed_index`
  (not the running mean), and its docstring tells you to *return* the current estimate on
  `should_prune()` instead of raising — such trials end COMPLETE, not PRUNED, so counting "pruned"
  trials undercounts what it stopped. It is experimental in 4.9.0.
- `SuccessiveHalvingPruner(min_resource=10, reduction_factor=2)` with resource = seed count promotes
  the top half at rungs 10, 20, 40. `HyperbandPruner` runs several such schedules with different
  starting rungs so one bad `min_resource` choice cannot ruin the search — at the price of more
  total seed-runs (400 versus 160 here).

The schedule behind those last two, written down. Successive halving with starting resource $r_0$
(here $r_0 = 10$ seeds) and reduction factor $\eta$ evaluates the surviving configurations at rung
$k$ with resource $r_k = r_0 \eta^k$, then promotes only the best $1/\eta$ fraction to the next
rung — so the rungs here are 10, 20, 40 seeds with half the survivors dropped at each. Each rung
costs about the same total ($\eta$ times fewer configurations at $\eta$ times the resource), which
is how it fits many configurations into a small budget. Its weakness is the choice of $r_0$: a
configuration that looks bad at 10 seeds but would look good at 40 is eliminated at the first rung
and never gets there. Hyperband's answer is to run several successive-halving schedules (called
brackets) whose starting resources step up from $r_0$ toward the maximum $R$ — about
$\log_{\eta}(R/r_0) + 1$ of them (verified: 3 brackets for $r_0 = 10$, $R = 50$, $\eta = 2$) — so
no single $r_0$ choice is fatal; the price is the extra budget (400 versus 160 seed-runs here).

### 5.5 Running the rule at cluster scale: who loops over seeds?

The code in §5.3 loops over seeds *inside one objective call*, which means one combination's seeds
run one after another in one process. Fine for the simulation; wrong for the cluster, where the 10
first seeds of a combination should run as 10 parallel Slurm workers.

The resolution is the controller pattern of §4.2, with the seed loop lifted out of the objective and
into the controller. The controller keeps, per combination, the list of seed results so far, and:

1. Stage 1 — for every combination the sampler proposes, submit seeds 0..9 to the queue (10 parallel
   training runs; the controller assigns `a_seed`, per §4.1).
2. As each result arrives, update that combination's $\bar{R}$, $s$, and $U = \bar{R} + 1.96 \cdot s/\sqrt{n}$.
3. If $U < 35$ and $n \ge 10$: stop that combination — `study.tell(trial_number, mean_so_far)` (or
   tell `state=PRUNED`), and submit no more of its seeds.
4. If $U \ge 35$: keep submitting seeds for it (in batches, so the queue stays full), up to the
   50-seed cap; then tell the final mean.

Here one Optuna trial = one combination (its value = the mean over however many seeds it ran), and
the threshold logic is ordinary controller code operating on the same file queue the workers already
serve. Every ingredient of this loop is individually verified (ask/tell with integer trial numbers,
FAIL/PRUNED tells, the queue protocol, journal storage under concurrent access); the assembled
controller is a sketch to write when the next sweep is designed, not a finished script in this repo.

### 5.6 Where the rule is unreliable: decisions near the bar at 10 seeds

The per-seed outcome in this project is strongly bimodal (a seed either learns to reach the goal or
scores near 0), so a 10-seed mean is itself noisy, and the rule's decisions near the bar are
unstable. Across three repetitions of the §5.3 simulation (different noise draws only):

- total seed-runs: 238–308 of 600 (a 49–60% saving each time);
- the surviving sets were {52.8, 35.2}, {52.8, 29, 26.1, 26}, and {52.8, 29} (by true mean) — the
  cell whose true mean sits exactly on the bar (35.2) was wrongly stopped in two of three
  repetitions, and cells at 26–29 were wrongly kept in one repetition.

None of this is an Optuna defect; it is what a 95% interval at $n = 10$ with per-seed standard
deviation near 35 can and cannot resolve: the interval half-width at 10 seeds is about
$1.96 \cdot 35/\sqrt{10} \approx 21.7$ reward points. If a distinction near the bar matters
(35.2 versus 35), either raise the minimum seeds before pruning, or lower the bar by a margin (keep
while $U \ge 35 - m$), or accept the misclassification rate. The saving is real either way; just do
not read the surviving set as an exact ranking.

---

## 6. Question 2 — searching intervals instead of grids

### 6.1 Set or interval? You choose, per parameter

Both, and per-parameter — this is the direct answer to "do I need to specify the combination by a
set, or do I specify an interval":

- `suggest_categorical("clip", ["inf", "5"])` — a finite **set**. Required by `GridSampler`; the
  natural form for genuinely discrete switches (normalization on/off, clip on/off).
- `suggest_float("ridge", 1e-6, 1e-2, log=True)` — a continuous **interval**. Unavailable to a grid
  (a grid must pre-commit to points); natural for scale parameters like ridge and beta, where
  run 3.1.2's own conclusion ("reward declines as the ridge ratio grows, coarsely") lives *between*
  the three grid points. `log=True` makes the sampler treat equal decade-ratios as equal distances —
  without it, a [1e-6, 1e-2] range would place almost all random draws above 1e-3.

The adaptive samplers accept any mix (verified: TPE with one categorical + one log float in the same
objective works and concentrates on the better category). A practical split for this project:
categorical for normalization and clip, log intervals for ridge and beta.

### 6.2 The samplers compared

| sampler | space form | how it picks the next combination |
|---|---|---|
| GridSampler | finite sets | every combination once, seeded shuffled order |
| RandomSampler | sets and intervals | independent draws, ignores all results |
| TPESampler (default) | sets and intervals | density ratio of good trials to the rest (§6.3) |
| GPSampler | sets and intervals | Gaussian-process fit + expected improvement (§6.4) |
| CmaEsSampler | intervals | evolution strategy (§6.5); extra package, not installed |

Here is the comparison, runnable as-is: all four samplers at an equal 60-trial budget on a noisy
2-parameter landscape shaped like this project's problem (single peak at log10 ridge = −6,
log10 beta = −2; smooth bump of height 50, width one decade; additive noise of standard
deviation 10):

```python
# Four samplers, equal 60-trial budget, on a noisy landscape shaped like the ridge-beta sweep.
import hashlib
import warnings

import numpy as np
import optuna
from optuna.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)  # GPSampler is experimental
optuna.logging.set_verbosity(optuna.logging.WARNING)
PEAK = np.array([-6.0, -2.0])  # (log10 ridge, log10 beta) of the peak


def substream(base, *parts):
    # one generator per named quantity, keyed by a stable string
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def make_objective(name):
    # noisy bump: height 50, width one decade, additive noise SD 10, keyed per (sampler, trial)
    def objective(trial):
        r = trial.suggest_float("log10_ridge", -6.0, -2.0)
        b = trial.suggest_float("log10_beta", -3.0, -1.0)
        clean = 50.0 * np.exp(-((r - PEAK[0]) ** 2 + (b - PEAK[1]) ** 2) / 2.0)
        return clean + substream(0, name, "noise", trial.number).normal(0.0, 10.0)
    return objective


GRID = {"log10_ridge": list(np.linspace(-6, -2, 4)), "log10_beta": [-3.0, -2.0, -1.0],
        "rep": [0, 1, 2, 3, 4]}  # rep axis makes the 12-cell grid spend all 60 trials


def grid_objective(trial):
    # the grid objective must suggest every grid key, including the rep axis
    trial.suggest_categorical("rep", GRID["rep"])
    return make_objective("GridSampler")(trial)


SAMPLERS = {
    "RandomSampler": optuna.samplers.RandomSampler(seed=0),
    "TPESampler": optuna.samplers.TPESampler(seed=0, n_startup_trials=10),
    "GPSampler": optuna.samplers.GPSampler(seed=0),
    "GridSampler": optuna.samplers.GridSampler(GRID, seed=0),
}
print(f"{'sampler':>14} {'best':>6} {'near-peak trials of 60':>24}")
for name, sampler in SAMPLERS.items():
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(grid_objective if name == "GridSampler" else make_objective(name), n_trials=60)
    pts = np.array([[t.params["log10_ridge"], t.params["log10_beta"]] for t in study.trials])
    near = int(np.sum(np.max(np.abs(pts - PEAK), axis=1) < 0.5))  # within half a decade in BOTH axes
    print(f"{name:>14} {study.best_value:6.1f} {near:>24}")
```

Output (observed, Optuna 4.9.0):

```
       sampler   best   near-peak trials of 60
 RandomSampler   53.1                        3
    TPESampler   63.5                       18
     GPSampler   67.3                       32
   GridSampler   54.2                        5
```

The pattern is stable across noise draws — an independent run of the same comparison at three other
noise seeds gave near-peak counts of 3–3 for random, 5–5 for the grid (fixed by construction: one
of its 12 cells is near the peak, visited 5 times), 9–23 for TPE, and 8–28 for the Gaussian
process. The adaptive samplers both allocate most of the budget near the peak *and* find better
points, because they can move between grid points.

This is the mechanism you asked for in "allocate computation to high-value combinations": an
adaptive sampler reallocates *future trials* toward the region that looks good so far, which is
complementary to §5's pruning (which cuts off *started* evaluations early). Use both.

### 6.3 The technique behind `TPESampler`: Tree-structured Parzen Estimator

TPE turns "where were the good results?" into a probability model. After some startup trials
(random; `n_startup_trials`, default 10 — verified: with the startup set to the whole budget, TPE's
draws are byte-identical to `RandomSampler` with the same seed):

1. Sort the completed trials by objective value and split them into a **good** group and the
   **rest**. Optuna's default puts the best $\min(\lceil 0.1 n \rceil, 25)$ of the $n$ finished
   trials in the good group — about the best tenth, capped at 25 trials. (That fraction is the
   $\gamma$ in the formulas below. The corresponding `gamma` constructor argument is deprecated as
   of 4.9.0, so treat the split as a fixed default, not a knob to set.)
2. Fit one probability density to the good group's parameter values and another to the rest's —
   $\ell(x)$ over the good, $g(x)$ over the rest. These are Parzen estimators, i.e. smoothed
   histograms: a kernel per observed point for floats, reweighted category frequencies for
   categoricals. In Optuna's default mode each parameter gets its own independent one-dimensional
   pair of densities with a shared good/rest split — correlations between parameters are not
   modeled unless you opt into the experimental `multivariate=True` (§7.1). ("Tree-structured"
   refers to handling search spaces where one parameter only exists conditional on another, e.g. a
   learning-rate schedule knob that exists only for one optimizer — §9.2 uses exactly this.)
3. Propose candidates from $\ell(x)$ and pick the candidate that maximizes the ratio

$$x_{\text{next}} = \arg\max_x \dfrac{\ell(x)}{g(x)}.$$

In words: prefer parameter values that look like the good trials and unlike the bad ones. The ratio
is not an arbitrary choice — Bergstra et al. (2011, "Algorithms for Hyper-Parameter Optimization"),
the paper Optuna's TPE implements, show that under this two-density model the expected improvement
over the split point $y^{\star}$ is a monotone function of the ratio:

$$\mathrm{EI}(x) \propto \left(\gamma + \dfrac{g(x)}{\ell(x)} (1 - \gamma)\right)^{-1},$$

so maximizing $\ell(x)/g(x)$ maximizes expected improvement. Exploration is kept alive because
$\ell$ is a smoothed density (it has tails) and the good/rest split is re-drawn every trial —
verified: with the better of two categories worth +25 reward, the last 15 of 60 trials still
included 2 draws of the worse category (13 of the better).

Practical properties, verified: deterministic given a seed (two studies with `TPESampler(seed=0)`
produced identical suggestion sequences); near-zero overhead (0.004 s per suggestion — irrelevant
next to a 15-hour training run); mixed categorical + log-float spaces are handled natively.

Section 7 opens the implementation: the exact defaults, the mixture Optuna actually fits (with its
prior component, bandwidth rule, and variance clipping), the full derivation of the
expected-improvement identity with a numerical check, and a from-scratch reimplementation in a page
of numpy.

### 6.4 The technique behind `GPSampler`: Bayesian optimization with a Gaussian process

The Gaussian-process sampler models the objective itself, not just the split into good and bad. A
Gaussian process is a probability distribution over functions; fitted (conditioned) on the observed
pairs of parameters and objective values, it yields at every candidate $x$ a predicted mean
$\mu(x)$ and a predicted standard deviation $\sigma(x)$ — the model's value guess and its own
uncertainty about that guess. The sampler then scores candidates by **expected improvement** over
the best observed value $f^{\star}$:

$$\mathrm{EI}(x) = \mathbb{E}\left[\max(0, f(x) - f^{\star})\right] = (\mu(x) - f^{\star}) \Phi(z) + \sigma(x) \phi(z), \qquad z = \dfrac{\mu(x) - f^{\star}}{\sigma(x)},$$

where $\Phi$ and $\phi$ are the standard normal cumulative distribution and density. (Optuna's
implementation optimizes the logarithm of this quantity for numerical stability.) The formula makes
the exploration trade-off explicit: a candidate scores high either because its predicted mean is
high (exploit) or because its uncertainty is high (explore) — a large $\sigma(x)$ increases the
chance of a large improvement.

Verified in this env: `GPSampler(seed=0)` runs (it needs torch, which is installed), matched or beat
TPE on the toy landscape (8–28 of 60 trials near the peak; best value 65.0 versus TPE's 60.9 on one
noise seed), and costs roughly one to two orders of magnitude more per suggestion (0.06 s versus
0.004 s here; the gap grows with trial count) — still irrelevant against 15-hour trainings. It is
marked experimental in 4.9.0, so pin the Optuna version if you adopt it. Section 8 opens the
implementation: the exact kernel and its fitting, the posterior and acquisition formulas rebuilt
from scratch in numpy, and a check that Optuna's asks fall where the from-scratch acquisition says
they should.

### 6.5 `CmaEsSampler`, for completeness

CMA-ES (covariance matrix adaptation evolution strategy) maintains a Gaussian *search distribution*
over the parameter space and iteratively shifts and reshapes it toward better-scoring samples. It
is strong for purely continuous spaces of moderate dimension and does not model categoricals well.
Not usable here without a new package: the class constructs, but the first sampled trial raises
`ModuleNotFoundError: No module named 'cmaes'` (verified; the `cmaes` package is not installed in
the `exploration` env, and the error appears only at sampling time, not construction).

### 6.6 Handling a noisy objective

One caution before adopting an adaptive sampler here: TPE and GP treat each trial's value as *the*
value of that combination. With per-seed noise this large (bimodal, standard deviation near 35), a
single-seed trial value is mostly noise, and the sampler would favor combinations whose sampled
seeds happened to score well. Two clean options:

1. **Trial value = mean over a fixed seed batch.** Each trial runs (say) 10 seeds and returns
   $\bar{R}$; the standard error the sampler sees drops by $\sqrt{10}$. Combine with §5's rule for
   extending good combinations beyond the initial batch. This is the recommended shape (§11).
2. **Repeat trials at the same point.** Adaptive samplers happily re-propose near-identical points;
   the model averages over them implicitly. Cheaper per trial but slower to concentrate; and note that
   for a *grid* study repeats must instead be made explicit (§5.1 point 1 — GridSampler never
   repeats on its own).

Related, for the controller pattern: when many asks are outstanding at once (a batch of
configurations training in parallel before any of them reports back), `TPESampler` has a
`constant_liar=True` option that makes it treat still-running trials as if they had returned a poor
value, so a batch of simultaneous asks spreads out instead of clustering on one promising point.
Verified with measured numbers in §10.4: a batch of 8 unanswered asks had mean pairwise distance
0.118 in the two log axes without the option and 0.519 with it.

---

## 7. TPE as Optuna implements it

Section 6.3 gave the idea; this section gives the implementation — what Optuna 4.9.0 literally
computes on every ask. Every statement here was either read from the installed source (module and
function named in place) or verified by running it; the scripts are in the
`tpe-internals-and-mini-tpe` folder (§12).

### 7.1 The exact defaults and what "independent mode" fits

```python
# TPESampler defaults, the good/rest split size, and the trial-age weighting (optuna 4.9.0)
import math
import numpy as np
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import default_gamma, default_weights

s = TPESampler()                                   # all defaults
print("n_startup_trials:", s._n_startup_trials)    # random search until this many finish
print("n_ei_candidates :", s._n_ei_candidates)     # candidates drawn per ask
print("multivariate    :", s._multivariate)        # per-parameter (independent) by default
print("constant_liar   :", s._constant_liar)

# default_gamma(n) = min(ceil(0.1 n), 25): the number of "good" trials, capped at 25.
print("gamma at n=100,600,9600:", default_gamma(100), default_gamma(600), default_gamma(9600))

# default_weights(n): below 25 finished trials every trial weighs 1; at 25+ the oldest
# n-25 trials ramp from 1/n up to 1 while the most recent 25 keep weight 1.
w = default_weights(30)
print("weights(30) oldest-first, first4:", np.round(w[:4], 3), "last4:", np.round(w[-4:], 3))
```

Output (observed, Optuna 4.9.0):

```
n_startup_trials: 10
n_ei_candidates : 24
multivariate    : False
constant_liar   : False
gamma at n=100,600,9600: 10 25 25
weights(30) oldest-first, first4: [0.033 0.275 0.517 0.758] last4: [1. 1. 1. 1.]
```

Four facts behind those numbers, from the installed source:

1. **The good-group size.** `default_gamma` in `optuna/samplers/_tpe/sampler.py` is exactly
   `return min(math.ceil(0.1 * x), 25)`: the top 10% of finished trials, capped at 25. The cap
   matters at this project's scale — at 9600 finished trials the good density $\ell(x)$ is still
   built from only 25 observations, so the good model stops sharpening once a study passes 250
   trials. (The `gamma` constructor argument that overrides this is deprecated in 4.9.0.)
2. **Trial-age weights.** With fewer than 25 finished trials, all weigh 1. From 25 on, the oldest
   $n - 25$ observations get weights ramping linearly from $1/n$ up to 1, and the newest 25 keep
   weight 1 — old evidence fades, recent evidence stays at full strength.
3. **Independent mode.** With the default `multivariate=False`, `infer_relative_search_space`
   returns an empty dict and every parameter goes through `sample_independent`, which builds a
   one-dimensional Parzen pair $\ell, g$ for *that parameter alone*. The good/rest split is shared
   (it depends only on objective values), but correlations between parameters are not modeled.
   `multivariate=True` switches to one joint kernel per observation (experimental, warns on
   construction); `group=True` additionally partitions a conditional space into independent joint
   blocks and requires `multivariate=True` (a `ValueError` otherwise).
4. **Deprecated tuning knobs, hard-wired.** `consider_prior`, `prior_weight`,
   `consider_magic_clip`, `consider_endpoints`, `gamma`, and `weights` are all deprecated in 4.9.0
   (removal scheduled for v6.0.0) and internally fixed to: prior on, prior weight 1.0, magic clip
   on, endpoints off. What §7.2 describes is therefore not adjustable behavior — it is *the*
   behavior.

### 7.2 The Parzen mixture: one kernel per observation, a prior, bandwidths, magic clip

For a numeric parameter on the interval $[a, b]$, given the good group's $n$ observed values
$\mu_1, \dots, \mu_n$, Optuna builds the good density as a mixture of $n + 1$ truncated normal
components:

$$\ell(x) = \sum_{k=1}^{n+1} w_k N(x; \mu_k, \sigma_k),$$

with every component truncated to $[a, b]$ and renormalized, where:

1. **One kernel per observation**: component $k \le n$ is centered at the observed value $\mu_k$.
2. **One prior component**: component $n + 1$ has mean $(a+b)/2$ and standard deviation $b - a$ —
   a nearly flat component over the whole interval, entering the weight normalization with prior
   weight 1.0. It guarantees $\ell(x) > 0$ everywhere, so no region is ever unreachable.
3. **Bandwidths from neighbor gaps**: sort the observations and give each interior kernel
   $\sigma_k = \max(\text{gap to left neighbor}, \text{gap to right neighbor})$, with the interval
   endpoints $a$ and $b$ serving as the outermost neighbors — dense clusters of good values get
   narrow kernels, sparse regions get wide ones. The two boundary kernels (the smallest and largest
   observation) are then overwritten (when there are at least two observations) to use only the gap
   to their single interior neighbor; the endpoint gap is discarded. That overwrite is the
   hard-wired `consider_endpoints=False` behavior — keeping the endpoint gap at the boundary was
   the now-deprecated `consider_endpoints=True` path. (In multivariate mode all of this is replaced
   by one Scott-style bandwidth $0.2 n^{-1/(d+4)} (b-a)$ shared by all kernels, $d$ = number of
   parameters.)
4. **Magic clip**: every $\sigma_k$ is clipped into $[\sigma_{\min}, b-a]$ with
   $\sigma_{\min} = \dfrac{b-a}{\min(100, n+2)}$ — duplicate or nearly-duplicate good values
   cannot produce a near-zero bandwidth that would make TPE re-propose one exact point forever.

The rest density $g(x)$ is built the same way from the remaining observations. For a
**categorical** parameter the mixture is over category-probability rows instead: each of the
$n + 1$ kernels starts every choice at prior-weight$/(n+1)$, each observation adds $+1$ to its own
choice in its own row, and rows are normalized — so a never-observed choice keeps nonzero
probability. All of this is observable directly:

```python
# The numeric Parzen mixture: one truncated-normal per observation + one prior; magic clip
import numpy as np
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator, _ParzenEstimatorParameters
from optuna.distributions import FloatDistribution

# defaults folded from the (now-deprecated) knobs: prior_weight=1, magic_clip=True, endpoints=False
params = _ParzenEstimatorParameters(
    prior_weight=1.0, consider_magic_clip=True, consider_endpoints=False,
    weights=lambda n: np.ones(n), multivariate=False, categorical_distance_func={},
)
space = {"x": FloatDistribution(low=0.0, high=10.0)}

# three observations -> 3 kernels + 1 prior kernel (mean at range center 5, sigma = full range 10)
mpe = _ParzenEstimator({"x": np.array([2.0, 3.0, 8.0])}, space, params)
comp = mpe._mixture_distribution.distributions[0]
print("mixture weights:", np.round(mpe._mixture_distribution.weights, 3), "(last = prior)")
print("component mus  :", np.round(comp.mu, 3), "(last = prior center 5.0)")
print("component sigma:", np.round(comp.sigma, 3), "(last = prior sigma = range 10)")

# magic clip floors the smallest sigma at range/min(100, 1+n_kernels); here 10/5 = 2.0
obs = {"x": np.array([5.000, 5.001, 1.0])}   # a near-duplicate pair would give sigma ~ 0.001
sig = _ParzenEstimator(obs, space, params)._mixture_distribution.distributions[0].sigma
print("sigmas with magic clip:", np.round(sig, 3), "-> smallest floored to 2.0, not 0.001")
```

Output (observed, Optuna 4.9.0):

```
mixture weights: [0.25 0.25 0.25 0.25] (last = prior)
component mus  : [2. 3. 8. 5.] (last = prior center 5.0)
component sigma: [ 2.  5.  5. 10.] (last = prior sigma = range 10)
sigmas with magic clip: [ 4.  2.  4. 10.] -> smallest floored to 2.0, not 0.001
```

(These internals — `_ParzenEstimator` and friends — are private API, shown here to make the
mechanism concrete; do not build production code on them.)

### 7.3 One ask, step by step: 24 candidates from the good density

Each `study.ask()` under TPE, in order:

1. If fewer than `n_startup_trials` (10) trials have *finished*, return a random draw.
2. Split the finished trials into the good group (size $k(n) = \min(\lceil 0.1 n \rceil, 25)$ — the
   source's `default_gamma`; the corresponding fraction $\gamma = k(n)/n$ is the $\gamma$ of §7.4's
   formulas) and the rest. In the source the good group is called "below" — loss-oriented naming; for a
   maximize study it is the *highest*-value trials.
3. Fit $\ell$ from the good group and $g$ from the rest (per parameter, in the default mode).
4. Draw `n_ei_candidates` (24) samples *from* $\ell$ — candidates come from where good trials
   cluster, not uniformly.
5. Return the candidate maximizing $\log \ell(x) - \log g(x)$.

Steps 4–5, reproduced by driving the real sampler's internals on a 60-trial study:

```python
# Each ask draws n_ei_candidates from l(x) (good) and returns argmax of log l - log g
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import _split_trials
from optuna.trial import TrialState
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 60 completed trials on a bump peaked at x=8 in [0,10]
space = {"x": optuna.distributions.FloatDistribution(0.0, 10.0)}
sampler = TPESampler(seed=0)
study = optuna.create_study(direction="maximize", sampler=sampler)
rng = np.random.default_rng(0)
for _ in range(60):
    x = rng.uniform(0, 10)
    study.add_trial(optuna.trial.create_trial(
        params={"x": x}, distributions=space,
        value=float(np.exp(-((x - 8.0) ** 2) / 2))))

# reproduce one ask through the sampler internals
trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=False)
n_below = sampler._gamma(len(trials))                              # good count = gamma(n)
below, above = _split_trials(study, trials, n_below, False)
mpe_below = sampler._build_parzen_estimator(study, space, below, handle_below=True)
mpe_above = sampler._build_parzen_estimator(study, space, above, handle_below=False)

cands = mpe_below.sample(sampler._rng.rng, sampler._n_ei_candidates)   # drawn from l(x)
acq = sampler._compute_acquisition_func(cands, mpe_below, mpe_above)   # log l - log g
ret = TPESampler._compare(cands, acq)                                  # argmax
print("candidates drawn from l(x):", cands["x"].size)
print("acq equals log l - log g   :",
      np.allclose(acq, mpe_below.log_pdf(cands) - mpe_above.log_pdf(cands)))
print("returned x == argmax cand  :", np.isclose(ret["x"], cands["x"][np.argmax(acq)]))
print("returned x (near peak 8)   :", round(ret["x"], 3))
```

Output (observed, Optuna 4.9.0):

```
candidates drawn from l(x): 24
acq equals log l - log g   : True
returned x == argmax cand  : True
returned x (near peak 8)   : 8.302
```

(As in §7.2: `study._get_trials`, `_split_trials`, `sampler._build_parzen_estimator`, and
`_compute_acquisition_func` are private API, driven here only to expose the mechanism — do not call
them in production code.)

### 7.4 Why the density ratio is expected improvement: the derivation, checked numerically

This is the Bergstra et al. (2011) argument, in full. Use the minimization convention (matching
Optuna's internal naming; a maximize study just flips which trials are "good"). Fix the threshold
$y^{\star}$ at the $\gamma$-quantile of the observed objective values, and model the two
conditional densities and the split probability:

$$p(x \mid y < y^{\star}) = \ell(x), \qquad p(x \mid y \ge y^{\star}) = g(x), \qquad P(y < y^{\star}) = \gamma.$$

Expected improvement over $y^{\star}$ at a candidate $x$ is the expected amount by which the
outcome beats the threshold:

$$\mathrm{EI}(x) = \int_{-\infty}^{y^{\star}} (y^{\star} - y) p(y \mid x) dy.$$

Apply Bayes' rule, $p(y \mid x) = p(x \mid y) p(y) / p(x)$, with the mixture marginal
$p(x) = \gamma \ell(x) + (1 - \gamma) g(x)$. On the integration region $y < y^{\star}$ the factor
$p(x \mid y)$ equals $\ell(x)$, which does not depend on $y$ and factors out of the integral:

$$\mathrm{EI}(x) = \dfrac{\ell(x) \int_{-\infty}^{y^{\star}} (y^{\star} - y) p(y) dy}{\gamma \ell(x) + (1 - \gamma) g(x)} = \dfrac{c}{\gamma + (1 - \gamma) \dfrac{g(x)}{\ell(x)}},$$

where $c$ is the integral — a constant that does not depend on $x$. The right-hand side is a
strictly increasing function of $\ell(x) / g(x)$, so the candidate maximizing the density ratio is
exactly the candidate maximizing expected improvement, whatever the modeled $p(y)$. Two things the
derivation makes plain:

1. TPE never models *how much better* a candidate will be — the shape of $p(y)$ is absorbed into
   the constant $c$. It only models *where* good and bad outcomes occur.
2. The exploration pressure lives entirely in the densities: the prior component keeps $\ell$
   positive everywhere, and the finite bandwidths keep it smooth, so the ratio never collapses onto
   a single point.

The identity, checked by numerical integration (no citation needed — you can run it):

```python
# Numerical check: maximizing l(x)/g(x) maximizes expected improvement (Bergstra et al. 2011)
import numpy as np
from scipy import stats

GAMMA = 0.25                                    # P(y < y_star): the "good" fraction

def kde(x, data, bw):                            # normalized 1-D Gaussian KDE density
    z = (x[:, None] - data[None, :]) / bw
    return (np.exp(-0.5 * z**2) / (bw * np.sqrt(2 * np.pi))).mean(1)

good = np.array([1.6, 2.0, 2.1, 2.4, 3.0])       # fixed good set (low y) and rest set
rest = np.array([3.5, 4.5, 5.0, 6.0, 6.5, 7.5, 8.0])
xs = np.linspace(-1, 10, 400)
l, g = kde(xs, good, 0.6), kde(xs, rest, 1.0)
ratio = l / g

# minimization convention: model p(y)=N(0,1); y_star is its gamma-quantile so P(y<y*)=gamma
y_star = stats.norm.ppf(GAMMA)
ys = np.linspace(-6, 6, 2000); dy = ys[1] - ys[0]
p_y = stats.norm.pdf(ys); below = ys < y_star

ei = np.empty_like(xs)                            # EI(x) integrated numerically over y
for i in range(len(xs)):
    p_x_given_y = np.where(below, l[i], g[i])     # p(x|y): l below threshold, g above
    p_x = GAMMA * l[i] + (1 - GAMMA) * g[i]
    p_y_given_x = p_x_given_y * p_y / p_x
    ei[i] = np.sum((y_star - ys[below]) * p_y_given_x[below]) * dy

rho, _ = stats.spearmanr(ei, ratio)
print(f"Spearman rank correlation  EI(x) vs l(x)/g(x): {rho:.10f}")
print(f"argmax x by l/g : {xs[np.argmax(ratio)]:.4f}   argmax x by EI: {xs[np.argmax(ei)]:.4f}")
```

Output (observed):

```
Spearman rank correlation  EI(x) vs l(x)/g(x): 1.0000000000
argmax x by l/g : 0.7920   argmax x by EI: 0.7920
```

The correlation is exactly 1 by construction — the derivation shows EI is a monotone transform of
the ratio — and the numerical integral reproduces it; the check would catch any error in the
derivation, and it also would break if the implementation ranked candidates any other way.

### 7.5 A mini-TPE from scratch in numpy

The whole mechanism — startup, split, densities, candidates, ratio — in about 50 lines of numpy
with no Optuna import, run on the same kind of noisy two-parameter landscape as §6.2:

```python
# Mini-TPE in numpy (no optuna): 10 random startup, top-25% split, KDE l/g, 24 candidates, argmax
import hashlib
import numpy as np

LOW, HIGH, PEAK = np.array([-6., -3.]), np.array([-2., -1.]), np.array([-6., -2.])
RANGE = HIGH - LOW

def substream(base, *parts):                       # one RNG per named quantity, hashed key
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)

def evaluate(p, base, method, i):                  # bump (max 50, width 1 decade) + noise SD 10
    val = 50.0 * np.exp(-np.sum((p - PEAK) ** 2) / 2.0)
    return val + substream(base, method, "noise", i).normal(0, 10)

def kde_logpdf(x, data, bw):                        # log Gaussian-KDE density at points x
    z = (x[:, None] - data[None, :]) / bw
    lk = -0.5 * z**2 - np.log(bw) - 0.5 * np.log(2 * np.pi)
    m = lk.max(1)
    return m + np.log(np.mean(np.exp(lk - m[:, None]), 1))

def mini_tpe(base, n_trials=60, n_cand=24):         # returns evaluated points
    pts, vals = [], []
    su = substream(base, "mtpe", "startup")
    for i in range(10):                              # 10 random startup points
        p = LOW + su.uniform(0, 1, 2) * RANGE
        pts.append(p); vals.append(evaluate(p, base, "mtpe", i))
    for i in range(10, n_trials):
        P, V = np.array(pts), np.array(vals)
        n_good = max(1, int(np.ceil(0.25 * len(V))))     # top-25% good / rest split
        order = np.argsort(-V)
        good, rest = P[order[:n_good]], P[order[n_good:]]
        cr = substream(base, "mtpe", "cand", i)
        cand, score = np.empty((n_cand, 2)), np.zeros(n_cand)
        for d in range(2):
            bwg, bwr = RANGE[d] / np.sqrt(len(good)), RANGE[d] / np.sqrt(max(len(rest), 1))
            c = np.clip(good[cr.integers(0, len(good), n_cand), d] + cr.normal(0, bwg, n_cand),
                        LOW[d], HIGH[d])              # draw candidates from the good density l
            cand[:, d] = c
            score += kde_logpdf(c, good[:, d], bwg) - kde_logpdf(c, rest[:, d], bwr)  # log l - log g
        best = cand[int(np.argmax(score))]
        pts.append(best); vals.append(evaluate(best, base, "mtpe", i))
    return np.array(pts)

def random_pts(base, n=60):
    r = substream(base, "rand")
    return np.array([LOW + r.uniform(0, 1, 2) * RANGE for _ in range(n)])

def hits(pts):                                       # trials within half a decade of the peak
    return int(sum(np.sqrt(np.sum((p - PEAK) ** 2)) < 0.5 for p in pts))

print("seed  mini_tpe  random  (hits within 0.5 of peak, out of 60)")
for s in (0, 1, 2):
    print(f"{s:>4} {hits(mini_tpe(s)):>9} {hits(random_pts(s)):>7}")
```

Output (observed):

```
seed  mini_tpe  random  (hits within 0.5 of peak, out of 60)
   0        23       1
   1        37       2
   2        23       2
```

The fuller script in the verification folder runs Optuna's real `TPESampler` on the same landscape
alongside (with a different noise realization): mini-TPE placed 19/28/22 of 60 trials near the peak
across three seeds, the real TPE 15/10/13, pure random 2/3/1. The from-scratch version differs from
Optuna's in the details that §7.1–7.2 pin down (bandwidth rule, prior component, magic clip,
trial-age weights, gamma schedule) — but the behavior class is the same, which is the point: TPE's
full mechanism fits in a page of numpy, with nothing hidden.

---

## 8. The Gaussian-process sampler, in detail

### 8.1 The surrogate Optuna fits

From the installed 4.9.0 source (`optuna/samplers/_gp/sampler.py` and `optuna/_gp/`):

1. **Kernel.** A Matérn 5/2 kernel with one lengthscale per parameter (the docstring: "The current
   implementation uses Matern kernel with nu=2.5 (twice differentiable) with automatic relevance
   determination (ARD) for the length scale of each parameter."). With per-dimension lengthscales
   $L_j$, signal variance $\sigma_f^2$, and scaled distance $u$:

$$k(x, x') = \sigma_f^2 \left(1 + \sqrt{5} u + \dfrac{5 u^2}{3}\right) \exp(-\sqrt{5} u), \qquad u^2 = \sum_{j=1}^{d} \dfrac{(x_j - x_j')^2}{L_j^2}.$$

2. **Input and output normalization.** Each numeric parameter is mapped to $[0, 1]$; a
   `log=True` parameter is mapped through the natural log first, so the GP models it on the log
   axis. A categorical parameter is compared by "same or different" (its squared distance is 0 or
   1). Objective values are standardized to zero mean and unit standard deviation, and a
   minimization study is sign-flipped — so the surrogate's units are standard deviations of the
   observed rewards, and its zero prior mean is the *average* observed value, not zero reward.
3. **Hyperparameter fitting, on every ask.** The lengthscales, signal variance, and noise variance
   are fitted by maximizing the marginal log-likelihood of the observations plus a log prior (a
   maximum-a-posteriori point estimate), with analytic gradients and L-BFGS-B:

$$\log p(y \mid X, \theta) = -\dfrac{1}{2} y^{\top} K_n^{-1} y - \dfrac{1}{2} \log \det K_n - \dfrac{n}{2} \log 2\pi, \qquad K_n = K + \sigma_n^2 I.$$

   The priors are Gamma distributions on the signal and noise variances plus a hand-written prior
   on the inverse squared lengthscales; the noise variance is floored at $10^{-6}$.
4. **Acquisition.** Log expected improvement (`optuna._gp.acqf.LogEI`; the docstring names "log
   expected improvement (logEI) for single-objective optimization"). The log is for numerical
   stability only — its argmax equals EI's argmax. The improvement threshold is simply the best
   observed (standardized) value; there is no extra exploration margin parameter.
5. **Acquisition optimization.** Neither pure random search nor pure gradient descent, but both:
   2048 quasi-random (Sobol) candidates are scored, the best one plus up to 9 further starting
   points (chosen with probability proportional to the exponentiated acquisition) seed local
   L-BFGS-B ascents (up to 200 iterations) for the continuous axes, with exhaustive or line search
   on discrete axes; the best point found wins.

That the installed kernel is exactly the textbook formula is a one-screen check:

```python
"""Confirm optuna's installed Matern 5/2 kernel equals the textbook closed form. Self-contained."""
import numpy as np
import torch
from optuna._gp.gp import Matern52Kernel

# Optuna's kernel takes the SQUARED scaled distance t = (r/L)^2 (dims summed after ARD scaling).
t = np.array([0.0, 0.01, 0.25, 1.0, 2.25, 4.0, 9.0], dtype=np.float64)

# Installed autograd kernel: exp(-sqrt5d)*(1 + sqrt5d + sqrt5d^2/3), sqrt5d = sqrt(5 t).
installed = Matern52Kernel.apply(torch.from_numpy(t)).detach().numpy()

# Textbook closed form k(r)/s2 = (1 + sqrt5 r/L + 5 r^2/3L^2) exp(-sqrt5 r/L) with s=sqrt5 r/L=sqrt(5t).
s = np.sqrt(5.0 * t)
closed = (1.0 + s + s * s / 3.0) * np.exp(-s)

print(f"{'t=(r/L)^2':>12}{'installed':>14}{'closed_form':>14}")
for ti, a, c in zip(t, installed, closed):
    print(f"{ti:12.4f}{a:14.10f}{c:14.10f}")
print("max abs diff = %.2e ; matches:" % np.max(np.abs(installed - closed)),
      bool(np.allclose(installed, closed, atol=1e-12)))
```

Output (observed, Optuna 4.9.0; trimmed):

```
   t=(r/L)^2     installed   closed_form
      0.0000  1.0000000000  1.0000000000
      1.0000  0.5239941088  0.5239941088
      9.0000  0.0277234219  0.0277234219
max abs diff = 2.22e-16 ; matches: True
```

### 8.2 The posterior and the acquisition, from scratch in numpy

Conditioned on observations $(X, y)$, the GP posterior at a candidate $x$ is the standard pair

$$\mu(x) = k_*^{\top} (K + \sigma_n^2 I)^{-1} y, \qquad \sigma^2(x) = k(x, x) - k_*^{\top} (K + \sigma_n^2 I)^{-1} k_*,$$

where $K$ is the covariance matrix over the training inputs and $k_*$ the covariance vector between
$x$ and the training inputs (Optuna's `GPRegressor.posterior` docstring states exactly these
formulas). Expected improvement then has the closed form of §6.4. Here is the whole thing in numpy
on a one-dimensional slice of this project's problem — reward against log10 ridge — with fixed
hyperparameters so every number is reproducible:

```python
"""Gaussian process posterior + closed-form Expected Improvement, numpy only. Self-contained."""
from math import erf
import numpy as np

# Toy 1-D objective: reward vs x = log10(ridge) on [-6,-2], a Gaussian bump peaking at x=-4.
true = lambda x: 50.0 * np.exp(-(((x + 4.0) / 1.2) ** 2))
x_obs = np.array([-6.0, -5.5, -5.0, -3.0, -2.5, -2.0])          # six flank observations
y_obs = true(x_obs) + np.array([0.3, -0.2, 0.1, -0.1, 0.2, -0.3])  # fixed inline noise -> deterministic
S2, L, NOISE = 400.0, 0.8, 1.0                                  # fixed hyperparameters (signal var, lengthscale, noise var)

def matern52(xa, xb):
    """Matern 5/2 covariance k(r)=s2(1+sqrt5 r/L+5r^2/3L^2)exp(-sqrt5 r/L) for 1-D inputs."""
    r = np.abs(xa[:, None] - xb[None, :])          # pairwise distances
    s = np.sqrt(5.0) * r / L                        # scaled distance sqrt(5) r / L
    return S2 * (1.0 + s + s * s / 3.0) * np.exp(-s)

def gp_posterior(xg):
    """Posterior mean mu and std sigma on grid xg via mu=k*(K+nI)^-1 y, var=k(x,x)-k*(K+nI)^-1 k*."""
    K = matern52(x_obs, x_obs) + NOISE * np.eye(len(x_obs))   # training covariance + noise
    Lc = np.linalg.cholesky(K)                                # Cholesky for stable solves
    alpha = np.linalg.solve(Lc.T, np.linalg.solve(Lc, y_obs)) # (K+nI)^-1 y
    Ks = matern52(xg, x_obs)                                   # cross-covariance grid vs data
    mu = Ks @ alpha                                           # posterior mean
    v = np.linalg.solve(Lc, Ks.T)                             # L^-1 k*^T
    var = np.clip(S2 - np.sum(v * v, axis=0), 1e-12, None)    # posterior variance (k(x,x)=S2)
    return mu, np.sqrt(var)

def ei(mu, sigma, f_best):
    """Closed-form EI(x)=(mu-f_best)Phi(z)+sigma phi(z), z=(mu-f_best)/sigma; higher is better."""
    z = (mu - f_best) / sigma                                 # standardized improvement
    phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)           # standard normal pdf
    Phi = 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))   # standard normal cdf
    return (mu - f_best) * Phi + sigma * phi                  # exploit term + explore term

# Evaluate posterior and EI on a grid; incumbent f_best is the best observed reward.
xg = np.round(np.arange(-6.0, -2.0 + 1e-9, 0.25), 2)
f_best = y_obs.max()
mu, sigma = gp_posterior(xg)
z = (mu - f_best) / sigma
ei_vals = ei(mu, sigma, f_best)

print("f_best=%.2f at x=%+.2f  (fixed: s2=%.0f L=%.2f noise=%.1f)" % (f_best, x_obs[y_obs.argmax()], S2, L, NOISE))
print(f"{'x':>7}{'mu':>9}{'sigma':>9}{'z':>8}{'EI':>9}")
for xi, mi, si, zi, ev in zip(xg, mu, sigma, z, ei_vals):
    print(f"{xi:7.2f}{mi:9.2f}{si:9.2f}{zi:8.2f}{ev:9.3f}")
i = int(ei_vals.argmax())
print("argmax EI at x=%+.2f (mu=%.2f sigma=%.2f) -- in the unsampled gap between x=-5 and x=-3"
      % (xg[i], mu[i], sigma[i]))

# Explore vs exploit: x=-4.00 (far from data, big sigma) vs x=-4.75 (next to best obs, mu above f_best).
for label, xq in [("EXPLORE (sigma large)", -4.00), ("EXPLOIT (mu high)", -4.75)]:
    j = int(np.argmin(np.abs(xg - xq)))
    exploit_term = (mu[j] - f_best) * 0.5 * (1 + erf((z[j]) / np.sqrt(2)))
    explore_term = sigma[j] * np.exp(-0.5 * z[j] ** 2) / np.sqrt(2 * np.pi)
    print("%-22s x=%+.2f mu=%.2f sigma=%.2f z=%.2f EI=%.3f [exploit=%.2f explore=%.2f]"
          % (label, xg[j], mu[j], sigma[j], z[j], ei_vals[j], exploit_term, explore_term))
```

Output (observed; middle rows trimmed):

```
f_best=25.07 at x=-5.00  (fixed: s2=400 L=0.80 noise=1.0)
      x       mu    sigma       z        EI
  -6.00     3.39     1.00  -21.75    0.000
  -5.00    24.97     1.00   -0.10    0.350
  -4.75    27.18     6.23    0.34    3.681
  -4.25    23.77    14.94   -0.09    5.337
  -4.00    22.79    16.02   -0.14    5.320
  -3.25    26.74     6.23    0.27    3.410
  -3.00    24.78     1.00   -0.29    0.269
  -2.00     2.80     1.00  -22.34    0.000
argmax EI at x=-4.25 (mu=23.77 sigma=14.94) -- in the unsampled gap between x=-5 and x=-3
EXPLORE (sigma large)  x=-4.00 mu=22.79 sigma=16.02 z=-0.14 EI=5.320 [exploit=-1.01 explore=6.33]
EXPLOIT (mu high)      x=-4.75 mu=27.18 sigma=6.23 z=0.34 EI=3.681 [exploit=1.33 explore=2.35]
```

Read the two labeled rows: at $x = -4.00$, far from all data, the posterior mean is *below* the
incumbent (the exploit term is negative, $-1.01$) and the entire EI comes from the large
$\sigma$ — a purely exploration-driven candidate. At $x = -4.75$, next to the best observation, the
posterior mean itself beats the incumbent (exploit term $+1.33$) with a much smaller $\sigma$. EI
adds the two pressures in one number, which is the whole idea of the acquisition function. Note
this demo runs in raw reward units with a zero prior mean and fixed hyperparameters so the
mechanism is visible; Optuna standardizes the objective and refits the hyperparameters on every ask
(§8.1), so its numbers differ in scale.

### 8.3 Optuna's asks agree with the from-scratch acquisition landscape

Import the same six observations into a real `GPSampler` study and ask where it wants to sample:

```python
"""Optuna GPSampler: import 6 observations, then ask where it wants to sample. Self-contained."""
import warnings
import numpy as np
import optuna
from optuna.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Same toy problem as the from-scratch GP: reward bump at x=-4, six flank observations.
true = lambda x: 50.0 * np.exp(-(((x + 4.0) / 1.2) ** 2))
x_obs = np.array([-6.0, -5.5, -5.0, -3.0, -2.5, -2.0])
y_obs = true(x_obs) + np.array([0.3, -0.2, 0.1, -0.1, 0.2, -0.3])
dist = optuna.distributions.FloatDistribution(-6.0, -2.0)

# Ten independent asks (fresh study per seed); each study sees exactly the six observations.
# n_startup_trials=5 < 6 completed, so the Gaussian process (not random sampling) drives the ask.
asks = []
for seed in range(10):
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.GPSampler(seed=seed, n_startup_trials=5))
    for xi, yi in zip(x_obs, y_obs):                    # import observations as finished trials
        study.add_trial(optuna.trial.create_trial(
            params={"x": float(xi)}, distributions={"x": dist}, value=float(yi)))
    trial = study.ask()                                 # ask where to sample next
    asks.append(trial.suggest_float("x", -6.0, -2.0))

asks = np.array(asks)
print("GPSampler asks (10 seeds):", np.array2string(asks, precision=3, floatmode="fixed"))
print("mean ask x = %+.3f" % asks.mean())
```

Output (observed, Optuna 4.9.0):

```
GPSampler asks (10 seeds): [-4.033 -4.033 -4.033 -4.033 -4.033 -4.033 -4.033 -4.033 -4.033 -4.033]
mean ask x = -4.033
```

All ten asks fall at $x = -4.03$ — inside the from-scratch EI table's high band (EI within 10% of
its maximum spans roughly $[-4.6, -3.5]$) and within 0.15 of the from-scratch argmax found on a
fine grid. The agreement is in region, not to the decimal, because Optuna standardizes the
objective and fits its own hyperparameters. Two more measured facts:

1. **Per-ask cost grows with the trial count**: about 0.09 s per ask at 20 completed trials and
   0.32 s at 200 (login-node CPU). At these sizes the growth is dominated by evaluating 2048
   acquisition candidates (roughly linear in the trial count); the cubic-cost Cholesky
   factorization takes over at larger studies. Still negligible against 15-hour trainings.
2. **Pending trials are automatically spread**: if asks are issued without telling results, the
   RUNNING trials get a built-in constant-liar treatment (they are treated as if they had returned
   the best observed value), so consecutive unanswered asks spread across the gap instead of
   repeating $-4.03$ (verified: ten unanswered asks ranged over $[-4.81, -3.21]$). This matters for
   parallel batches — §10.4.

---

## 9. Multi-algorithm sweeps: the best configuration per method (run 3.2.1)

### 9.1 Run 3.2.1's search problem

Train run 3.2.1 (writeup §5.3.6) sweeps three optimizer methods for the RND predictor, each with its
own grid:

| method | swept axes | configurations |
|---|---|---|
| O1 Adam | readout (2) x beta (8) | 16 |
| O2 AdaGrad | readout (2) x beta (8) | 16 |
| O3 SGD-1/t | readout (2) x beta (8) x eta0 (5) x t0 (2) | 160 |

192 configurations x 50 seeds = 9600 training runs, scored on the final training-episode reward
(standalone eval off). At the measured ~15 hours per run, the full grid costs about 143,000
CPU-hours — about 4.4 days at this cluster's full ~1344-CPU ceiling, and about 12 days at the rate
run 3.1.2 actually sustained (§10.1). Two features distinguish this from the run-3.1.2 sweep, and
each maps to one Optuna capability:

1. **Method-specific parameters.** The schedule knobs eta0 and t0 exist only for O3. In Optuna this
   is a *conditional search space* — the "tree" in Tree-structured Parzen Estimator (§9.2).
2. **The deliverable is the best configuration PER method** — three winners, because the run's
   research questions (readout scale, count memory, decreasing step) each compare methods at their
   own best. This changes the right study layout entirely (§9.3–§9.5).

### 9.2 Method-specific parameters in one study: conditional suggests work

The objective may branch — suggest a parameter only when the method needs it. This runs under the
default TPE with no error and no warning:

```python
import hashlib
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def reward(method, readout, log10_beta, eta0=None, t0=None):
    # noise-free bump per method (adam peak 20 @ beta=1e2, adagrad 27 @ 1e1, sgd1t 45 @ 1e2)
    center, amp = {"adam": (2.0, 20.0), "adagrad": (1.0, 27.0), "sgd1t": (2.0, 45.0)}[method]
    mean = amp * np.exp(-0.5 * ((log10_beta - center) / 0.65) ** 2)
    if method == "adam" and readout == "l2":
        mean += 3.0
    if method == "sgd1t":
        mean += -4.0 * (np.log10(eta0) + 2.0) ** 2 - 2.0 * abs(np.log10(t0) - 4.0)
    # reproducible additive noise SD 6: one generator per configuration, keyed by a stable string
    key = "::".join(str(p) for p in (method, readout, round(log10_beta, 6), eta0, t0))
    gen = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)
    return mean + gen.normal(0.0, 6.0)


def objective(trial):
    # method is a top-level categorical; eta0/t0 are suggested ONLY in the sgd1t branch
    method = trial.suggest_categorical("method", ["adam", "adagrad", "sgd1t"])
    readout = trial.suggest_categorical("readout", ["mse", "l2"])
    log10_beta = trial.suggest_float("log10_beta", -3.0, 4.0)
    if method == "sgd1t":
        eta0 = trial.suggest_float("eta0", 1e-3, 1e-1, log=True)
        t0 = trial.suggest_categorical("t0", [1e3, 1e4])
        return reward(method, readout, log10_beta, eta0, t0)
    return reward(method, readout, log10_beta)


# default TPESampler is multivariate=False -> independent per-parameter estimators
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=60)

df = study.trials_dataframe()
cols = ["number", "params_method", "params_log10_beta", "params_eta0", "params_t0"]
print(df[cols].head(6).to_string(index=False))
n_sgd = sum(t.params.get("method") == "sgd1t" for t in study.trials)
n_eta0 = sum("eta0" in t.params for t in study.trials)
print(f"\ntrials total={len(study.trials)}  sgd1t={n_sgd}  with eta0={n_eta0}  "
      f"(eta0 present only in sgd1t trials: {n_eta0 == n_sgd})")
non_sgd = df[df["params_method"] != "sgd1t"]
print(f"non-sgd1t rows all have eta0/t0 = NaN: "
      f"{non_sgd['params_eta0'].isna().all() and non_sgd['params_t0'].isna().all()}")
```

Output (observed, Optuna 4.9.0):

```
 number params_method  params_log10_beta  params_eta0  params_t0
      0       adagrad           1.521259          NaN        NaN
      1         sgd1t           0.702264     0.013680     1000.0
      2         sgd1t           3.850328     0.039657    10000.0
      3       adagrad          -0.097366          NaN        NaN
      4       adagrad           1.323448          NaN        NaN
      5         sgd1t           0.059224     0.024846    10000.0

trials total=60  sgd1t=8  with eta0=8  (eta0 present only in sgd1t trials: True)
non-sgd1t rows all have eta0/t0 = NaN: True
```

Why this works: the independent-mode TPE fits each parameter's density pair from *the trials where
that parameter exists* (the source guards the accumulation with a subset check on the trial's
parameter names), so eta0's estimator is built from sgd1t trials only, and the other methods'
trials simply have no eta0 entry (`NaN` in the dataframe). One warning for later: do not switch a
conditional space to `multivariate=True` without also `group=True` — the joint estimator alone
does not support a dynamic space and falls back to independent sampling with a warning.

### 9.3 Why one shared study fails the best-per-method question

The tempting layout — one study, `method` as a categorical — actively works against you, because
the sampler's whole purpose is to concentrate its budget where the objective looks best. Measured
on a toy landscape with per-method true peaks adam 23 / adagrad 27 / sgd1t 45 and noise of standard
deviation 6: a single 90-trial TPE study allocated (adam / adagrad / sgd1t) 9/20/61, 10/10/70, and
10/10/70 across three seeds — the leading method got 49–78% of the budget in every one of eight
probe seeds, leaving the other methods roughly their startup-phase 10 trials. Two consequences,
both observed:

1. **The methods given few trials return poor winners.** Comparing the noise-free reward of the best
   cell each scheme actually found (so max-of-noise cannot inflate any method's score), the shared study versus three
   30-trial per-method studies at the same 90-trial total: adam 17.5 versus 23.0 (of a true peak
   23), adagrad 27.0 versus 26.9 (peak 27), sgd1t 25.3 versus 44.6 (peak 45), averaged over seeds.
   For the best-per-method question, the split studies are simply a different — and correct —
   experiment.
2. **The concentration target is chosen by noisy early trials, and can be the wrong method.** In
   one realization the shared study put 71 of 90 trials on adagrad (true peak 27) and only 10 on
   the actually-best sgd1t (true peak 45), whose best-found cell then reached a noise-free 25.3.
   With this project's strongly bimodal per-seed rewards, this risk is worse, not better.

Here is that realization, runnable as-is:

```python
import hashlib
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
METHODS = ["adam", "adagrad", "sgd1t"]
TRUE_PEAK = {"adam": 23.0, "adagrad": 27.0, "sgd1t": 45.0}


def reward_mean(method, readout, log10_beta, eta0=None, t0=None):
    # noise-free mean: Gaussian bump per method, +3 for adam l2, mild eta0/t0 dependence for sgd1t
    peak = {"adam": (2.0, 20.0), "adagrad": (1.0, 27.0), "sgd1t": (2.0, 45.0)}[method]
    mean = peak[1] * np.exp(-0.5 * ((log10_beta - peak[0]) / 0.65) ** 2)
    if method == "adam" and readout == "l2":
        mean += 3.0
    if method == "sgd1t":
        mean += -4.0 * (np.log10(eta0) + 2.0) ** 2 - 2.0 * abs(np.log10(t0) - 4.0)
    return mean


def reward(method, readout, log10_beta, eta0=None, t0=None):
    # reproducible additive noise SD 6, keyed by the configuration
    key = "::".join(str(p) for p in (method, readout, round(log10_beta, 6), eta0, t0))
    gen = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)
    return reward_mean(method, readout, log10_beta, eta0, t0) + gen.normal(0.0, 6.0)


def suggest(trial, method):
    # suggest one method's axes; sgd1t alone carries eta0/t0
    readout = trial.suggest_categorical("readout", ["mse", "l2"])
    log10_beta = trial.suggest_float("log10_beta", -3.0, 4.0)
    if method == "sgd1t":
        return reward(method, readout, log10_beta,
                      trial.suggest_float("eta0", 1e-3, 1e-1, log=True),
                      trial.suggest_categorical("t0", [1e3, 1e4]))
    return reward(method, readout, log10_beta)


def true_mean_of(method, p):
    # noise-free mean at a trial's chosen params (search quality, no max-of-noise bias)
    if method == "sgd1t":
        return reward_mean(method, p["readout"], p["log10_beta"], p["eta0"], p["t0"])
    return reward_mean(method, p["readout"], p["log10_beta"])


seed = 0
# shared study: 90 trials, method is a suggested categorical
shared = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
shared.optimize(lambda t: suggest(t, t.suggest_categorical("method", METHODS)), n_trials=90)

# split: three studies, 30 trials each (same 90-trial total), method fixed per study
split = {}
for m in METHODS:
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(lambda t, m=m: suggest(t, m), n_trials=30)
    split[m] = st

print(f"{'method':>8} {'shared_n':>9} {'shared_true':>12} {'split_n':>8} {'split_true':>11} {'true_peak':>10}")
for m in METHODS:
    sh = [t for t in shared.trials if t.params.get("method") == m]
    sh_true = max(true_mean_of(m, t.params) for t in sh)
    sp_true = max(true_mean_of(m, t.params) for t in split[m].trials)
    print(f"{m:>8} {len(sh):>9} {sh_true:>12.2f} {30:>8} {sp_true:>11.2f} {TRUE_PEAK[m]:>10.1f}")
```

Output (observed, Optuna 4.9.0):

```
  method  shared_n  shared_true  split_n  split_true  true_peak
    adam         9        17.47       30       23.00       23.0
 adagrad        71        27.00       30       26.89       27.0
   sgd1t        10        25.29       30       44.63       45.0
```

Read the sgd1t row: the shared study gave the true-best method 10 trials and its best-found cell
reached a noise-free 25.29 of the 45.0 peak; the dedicated 30-trial study reached 44.63. The shared
study concentrated on adagrad because early noise made it look best.

### 9.4 Reading a mixed study anyway, and the max-of-noise trap

If a mixed study already exists, the per-method winners are recoverable — filter completed trials
on `params["method"]` and take the argmax per group:

```python
# recover the best trial PER method from the mixed study of section 9.3
winners = {}
for t in shared.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
    m = t.params.get("method")
    if m is not None and (m not in winners or t.value > winners[m].value):
        winners[m] = t

for m in METHODS:
    t = winners[m]
    n_m = sum(x.params.get("method") == m for x in shared.trials)
    print(f"{m:>8}: best value={t.value:6.2f}  from {n_m:2d} trials  params={t.params}")
```

Output (observed, Optuna 4.9.0):

```
    adam: best value= 26.01  from  9 trials  params={'method': 'adam', 'readout': 'mse', 'log10_beta': 1.6616268613509622}
 adagrad: best value= 36.18  from 71 trials  params={'method': 'adagrad', 'readout': 'mse', 'log10_beta': 0.8023088191639868}
   sgd1t: best value= 24.26  from 10 trials  params={'method': 'sgd1t', 'readout': 'mse', 'log10_beta': 1.3501992387220194, 'eta0': 0.01134725777275251, 't0': 1000.0}
```

Two traps are visible in that output, and the second one applies to every noisy study, not just
mixed ones:

1. Each winner is only as informed as the trials its method received — the sgd1t "winner" comes
   from 10 trials of a 160-configuration method.
2. **The best single value is inflated by max-of-noise.** Adagrad's winner shows value 36.18
   against a true peak of 27.0 — the maximum over 71 noisy draws (noise SD 6) sits far above any
   true mean, and it even exceeds the sgd1t winner's value although sgd1t's true peak is 45.
   Ranking methods by `study.best_value` with unequal trial counts selects the wrong method. The
   bias grows with the number of trials, which is exactly why run 3.2.1's design scores every
   configuration by its **mean over 50 seeds** — report winners by the best per-configuration seed
   mean (with its standard error, §5.2), never by the best single run.

### 9.5 The recommended layout: one study per method, one journal file

One journal file in the run folder holds any number of named studies, so "one study per method"
costs nothing in infrastructure. Verified: three studies with different budgets in one file, listed
by `optuna.get_all_study_names`, each reloading by name with exactly its own trials:

```python
import os
import tempfile
import optuna
# current (v4.x) import paths -- the journal classes live under optuna.storages.journal
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock, JournalStorage

optuna.logging.set_verbosity(optuna.logging.WARNING)
METHOD_TRIALS = {"adam": 12, "adagrad": 8, "sgd1t": 20}   # different budgets to show independence


def objective(trial):
    # trivial method-agnostic objective; the point is the storage layout, not the landscape
    x = trial.suggest_float("x", -5.0, 5.0)
    return -(x - 1.0) ** 2


# one journal file (one file in the run folder) holds one study per method
path = os.path.join(tempfile.mkdtemp(), "runs_3_2_1.log")
storage = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))

# create + optimize one study per method in the SAME storage
for method, n in METHOD_TRIALS.items():
    study = optuna.create_study(direction="maximize", study_name=f"3_2_1_{method}",
                                sampler=optuna.samplers.TPESampler(seed=0), storage=storage)
    study.optimize(objective, n_trials=n)

# list every study name recorded in the storage
print("study names:", sorted(optuna.get_all_study_names(storage)))

# reload each study by name from a FRESH storage handle; each holds exactly its own budget
reloaded = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
for method, expected in METHOD_TRIALS.items():
    study = optuna.load_study(study_name=f"3_2_1_{method}", storage=reloaded)
    print(f"  3_2_1_{method:<8} n_trials={len(study.trials):2d}  expected={expected:2d}  "
          f"isolated={len(study.trials) == expected}")
```

Output (observed, Optuna 4.9.0):

```
study names: ['3_2_1_adagrad', '3_2_1_adam', '3_2_1_sgd1t']
  3_2_1_adam     n_trials=12  expected=12  isolated=True
  3_2_1_adagrad  n_trials= 8  expected= 8  isolated=True
  3_2_1_sgd1t    n_trials=20  expected=20  isolated=True
```

The same layout carries the run-3.2.1 grids directly — each method's study gets a `GridSampler`
over exactly its own axes, and each stops by itself at its own grid size (verified: 16 / 16 / 160
trials, every cell visited exactly once, in one shared storage file):

```python
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
```

Output (observed, Optuna 4.9.0):

```
    adam: grid_size= 16  trials_run= 16  unique_cells= 16
 adagrad: grid_size= 16  trials_run= 16  unique_cells= 16
   sgd1t: grid_size=160  trials_run=160  unique_cells=160
```

`enqueue_trial` is also per-study (verified): enqueueing the canonical Adam-mse reference cell in
the adam study makes it that study's trial 0 and leaves the other studies untouched. What this
layout deliberately gives up is cross-method transfer — the three studies share no information —
which for run 3.2.1 is the point: three independent, equally-budgeted searches, one honest winner
each.

---

## 10. Parallel Optuna at this cluster's scale

### 10.1 The compute you actually have

Measured on train run 3.1.2 (the record is in that run's folder and analysis):

1. One training run = 1M steps ≈ 15 hours on one CPU at the 16-tasks-x-1-CPU shape's measured pace
   (67,352 steps per CPU-hour, the shape now recommended in `slurm-submission.md`).
2. The fleet: 70 Slurm job submissions; at the 2026-07-03 snapshot, 349 workers were running
   concurrently (176 arm A + 173 arm B; the arm-A workers use 2 CPUs each, so those 349 workers
   held about 525 CPUs) and had finished 1333 of 1800 runs in about 41 hours — about 780 training
   runs per day at that fleet size.
3. The documented per-user ceiling across all pools (cpu 400 + gpu 400 + nolim 80 + gnolim 80 +
   reserved jaguar03 224 + puma01 160) is about 1344 CPUs — roughly 2150 runs per day if fully
   held.
4. Run 3.2.1's full grid (9600 runs) therefore costs about 143,000 CPU-hours — about 4.4 days at
   the ceiling, and about 12 days at the observed rate of 780 runs per day. The 10-seeds-first
   stage (192 x 10 = 1920 runs) costs about 28,500 CPU-hours: about 21 hours at the ceiling, about
   2.5 days at the observed rate.

What this means for Optuna: hundreds of trials are always running at once, each result arrives
hours after its ask, and the sampler/pruner computations cost milliseconds to sub-seconds (§6.3,
§8.3) — the decision layer is never the constraint. The two real design questions are coordination
(how hundreds of processes share one study — storage, §10.2/§10.7) and allocation (which runs to
start and stop — everything else in this document).

### 10.2 The two parallel patterns, and where the sampler runs

1. **Worker-owned `study.optimize`.** Every worker process does `optuna.load_study(...)` on the
   shared journal file and runs `study.optimize(objective, n_trials=...)`; the objective launches
   `train.py` as a subprocess and waits. The sampler executes inside *every worker* at its own ask
   time. Verified with 8 worker processes: all trials recorded, and an observer process watching
   the shared study saw all 8 RUNNING trials at once — running trials from other processes are
   visible through the storage, which is what makes the next two subsections work.
2. **Controller ask/tell.** One controller owns the study and the sampler; workers only evaluate,
   through the existing file queue (§4.2). The sampler executes *only in the controller*.

Both work on this cluster. The controller pattern is the recommended one here because it keeps
every decision (asks, tells, seed allocation, stopping rules, dead-worker cleanup) in one process
whose log is the sweep's decision record — and the workers stay the plain claim-train-report loop
they already are. The controller side, in miniature (runnable as-is):

```python
import optuna, tempfile, os
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)

# One journal-file study on the shared folder.
path = os.path.join(tempfile.mkdtemp(), "journal.log")
def open_storage():
    # a fresh handle to the one shared journal-file storage
    return JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))

# Controller pattern: the controller keeps K trials asked-but-untold.
K = 8
controller = optuna.create_study(study_name="demo", storage=open_storage())
outstanding = [controller.ask() for _ in range(K)]          # ask() runs the sampler here
for t in outstanding:
    t.suggest_float("x", -5, 5)                             # controller fixes the params

# Any other process (here: a fresh open of the same journal) sees those K trials RUNNING.
observer = optuna.load_study(study_name="demo", storage=open_storage())
running = [t for t in observer.get_trials(deepcopy=False) if t.state == TrialState.RUNNING]
print(f"asked-but-untold trials visible as RUNNING from another view: {len(running)}")

# When a worker returns a result, the controller tells that trial number.
controller.tell(outstanding[0].number, 0.123)
observer2 = optuna.load_study(study_name="demo", storage=open_storage())
n_running = sum(t.state == TrialState.RUNNING for t in observer2.get_trials(deepcopy=False))
n_done = sum(t.state == TrialState.COMPLETE for t in observer2.get_trials(deepcopy=False))
print(f"after telling one: RUNNING={n_running} COMPLETE={n_done}")
```

Output (observed, Optuna 4.9.0):

```
asked-but-untold trials visible as RUNNING from another view: 8
after telling one: RUNNING=7 COMPLETE=1
```

### 10.3 At launch, the first wave is random search

`n_startup_trials` counts **finished** trials, not asked ones. At sweep launch, when hundreds of
workers ask before anything completes, every ask of that first wave is a random draw — whatever the
sampler and whatever `n_startup_trials`:

```python
import optuna, numpy as np
from optuna.samplers import TPESampler, RandomSampler
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Ask n trials from a study but never tell a result, and collect the suggested points.
def ask_without_telling(sampler, n=40):
    study = optuna.create_study(sampler=sampler, direction="minimize")
    pts = [(t.suggest_float("x", -5, 5), t.suggest_float("y", -5, 5))
           for t in (study.ask() for _ in range(n))]
    return np.array(pts), study

# TPESampler with a startup threshold of 10, but with zero completed trials.
tpe_pts, tpe_study = ask_without_telling(TPESampler(seed=0, n_startup_trials=10))
# A plain RandomSampler with the same seed.
rnd_pts, _ = ask_without_telling(RandomSampler(seed=0))

# n_startup_trials counts COMPLETED trials (0 here), so all 40 TPE asks are random.
n_match = int(np.all(np.isclose(tpe_pts, rnd_pts), axis=1).sum())
print(f"TPE asks identical to RandomSampler(seed=0): {n_match} / 40")
states = {}
for t in tpe_study.get_trials(deepcopy=False):
    states[t.state.name] = states.get(t.state.name, 0) + 1
print("TPE-study trial states after 40 asks, 0 tells:", states)
```

Output (observed, Optuna 4.9.0):

```
TPE asks identical to RandomSampler(seed=0): 40 / 40
TPE-study trial states after 40 asks, 0 tells: {'RUNNING': 40}
```

For a per-method grid stage this is irrelevant (a grid is enumeration anyway). It matters for any
adaptive stage: with 15-hour trials and ~350 workers, the first ~350 trials of a fresh adaptive
study are random search, and steering begins only as tells arrive. Importing earlier results
(§4.4) removes the random wave entirely — imported COMPLETE trials count as finished.

### 10.4 Parallel asks cluster; `constant_liar` spreads them

Once a TPE study *is* informed, a batch of simultaneous asks has the opposite problem: the model
does not change between asks (no new results), so every ask falls near the same model optimum. The
`constant_liar=True` option makes the sampler count each RUNNING trial as if it had returned a poor
value (it is placed in the rest group $g$), pushing later asks away from points already being
evaluated:

```python
# constant_liar spreads a parallel batch: 8 asks with no results reported, with vs without it
import hashlib, itertools
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.distributions import FloatDistribution
optuna.logging.set_verbosity(optuna.logging.WARNING)

PEAK = np.array([-6., -2.])
DIST = {"r": FloatDistribution(-6, -2), "b": FloatDistribution(-3, -1)}

def substream(base, *parts):                       # one RNG per named quantity, hashed key
    key = "::".join(str(p) for p in (base, *parts))
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)

def bump(r, b):                                    # noiseless signal peaked at PEAK, width 1 decade
    return 50.0 * np.exp(-((r + 6) ** 2 + (b + 2) ** 2) / 2.0)

def batch(constant_liar):                          # seed 30 completed trials, then ask 8 (RUNNING)
    sampler = TPESampler(seed=0, constant_liar=constant_liar)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for i in range(30):
        r = substream(0, "r", i).uniform(-6, -2)   # per-quantity seeding, not one shared stream
        b = substream(0, "b", i).uniform(-3, -1)
        noise = substream(0, "noise", i).normal(0, 2)
        study.add_trial(optuna.trial.create_trial(
            params={"r": r, "b": b}, distributions=DIST, value=float(bump(r, b) + noise)))
    pts = []
    for _ in range(8):                             # ask() leaves each trial RUNNING (no result told)
        t = study.ask(DIST)
        pts.append([t.params["r"], t.params["b"]])
    return np.array(pts)

def spread(pts):                                   # mean pairwise distance across the 8 points
    return float(np.mean([np.hypot(*(a - b)) for a, b in itertools.combinations(pts, 2)]))

off, on = batch(False), batch(True)
print(f"constant_liar=False  mean pairwise distance: {spread(off):.3f}  (clustered)")
print(f"constant_liar=True   mean pairwise distance: {spread(on):.3f}  (spread out)")
```

Output (observed, Optuna 4.9.0; the option prints a one-line ExperimentalWarning on construction):

```
constant_liar=False  mean pairwise distance: 0.118  (clustered)
constant_liar=True   mean pairwise distance: 0.519  (spread out)
```

Notes: the first ask of a batch is identical either way (no running trials exist yet to penalize);
the option matters in *both* parallel patterns of §10.2, because in both a batch of RUNNING trials
exists whenever many workers evaluate at once; and `GPSampler` needs no switch — it applies its own
best-value penalty to RUNNING trials automatically (§8.3, verified).

### 10.5 Pruning works across processes — but pass the pruner everywhere

Verified across real OS processes: process A completed a trial with per-step reports through the
shared journal file; process B then started a bad trial, and B's `trial.should_prune()` used A's
history — intermediate values and completed-trial history flow through the storage. The trap is
elsewhere: **the sampler and pruner are attributes of the in-memory `Study` object, not stored in
the storage.** A process that calls `optuna.load_study` without `pruner=` silently gets the default
`MedianPruner(n_startup_trials=5)` — not your pruner:

```python
import optuna, tempfile, os
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.pruners import MedianPruner
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Build a shared journal-file study (what every worker opens on the NFS folder).
path = os.path.join(tempfile.mkdtemp(), "journal.log")
storage = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))

# One process completes a good trial: high intermediate value at every step.
study = optuna.create_study(study_name="demo", storage=storage, direction="maximize",
                            pruner=MedianPruner(n_startup_trials=1, n_warmup_steps=0))
good = study.ask()
for step in range(10):
    good.report(1.0, step)
study.tell(good, 1.0)

# A second view WITHOUT passing a pruner: it silently gets the DEFAULT
# MedianPruner(n_startup_trials=5), so with one completed trial it never prunes.
view_default = optuna.load_study(study_name="demo", storage=storage)
bad = view_default.ask()
bad.report(0.0, 0)
print("no pruner passed  -> should_prune():", bad.should_prune())
view_default.tell(bad, state=TrialState.FAIL)

# A third view that passes the SAME pruner: now the low trial is pruned using the
# good trial's history, which reached this process through the shared journal.
view_pruner = optuna.load_study(study_name="demo", storage=storage,
                                pruner=MedianPruner(n_startup_trials=1, n_warmup_steps=0))
bad2 = view_pruner.ask()
bad2.report(0.0, 0)
print("same pruner passed -> should_prune():", bad2.should_prune())
```

Output (observed, Optuna 4.9.0):

```
no pruner passed  -> should_prune(): False
same pruner passed -> should_prune(): True
```

So in the worker-owned pattern, every worker must construct the identical pruner (and sampler) when
it loads the study. In the controller pattern this problem does not exist — only the controller
ever consults them. (For your own threshold rule none of this applies anyway: §5.3's inline
`raise optuna.TrialPruned()` consults no pruner object at all.)

### 10.6 Dead workers: the journal file has no heartbeat

On this cluster workers die routinely (job walltime, the 24-hour per-run stopwatch, node failures).
Optuna's built-in remedy — heartbeats plus `fail_stale_trials` — is **RDB-only**: `RDBStorage`
subclasses `BaseHeartbeat`, `JournalStorage` does not. Verified: `fail_stale_trials` on a journal
study returns silently having changed nothing, `JournalStorage(heartbeat_interval=60)` raises
`TypeError`, and a SIGKILLed worker's trial stays RUNNING forever. The recovery is a controller
responsibility, and it works from any process:

```python
import optuna, tempfile, os, warnings
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.storages._heartbeat import BaseHeartbeat, is_heartbeat_enabled
from optuna.trial import TrialState
from optuna.exceptions import ExperimentalWarning

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ExperimentalWarning)  # fail_stale_trials is experimental

# Journal-file study on the shared folder.
path = os.path.join(tempfile.mkdtemp(), "journal.log")
storage = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
study = optuna.create_study(study_name="demo", storage=storage)

# Journal storage does NOT implement the heartbeat mechanism (that is RDB-only).
print("is BaseHeartbeat:", isinstance(storage, BaseHeartbeat))
print("heartbeat enabled:", is_heartbeat_enabled(storage))

# A worker that dies mid-trial leaves the trial RUNNING. Simulate the leftover trial.
orphan = study.ask()
orphan.suggest_float("x", 0.0, 1.0)
print("orphan trial state:", study.trials[orphan.number].state.name)

# fail_stale_trials is a silent no-op on journal storage: nothing changes, no error.
optuna.storages.fail_stale_trials(study)
print("after fail_stale_trials:", study.trials[orphan.number].state.name)

# The cleanup a controller must do itself: tell the trial number FAIL.
# tell accepts an int trial number, so a process that never created the trial can fail it.
study.tell(orphan.number, state=TrialState.FAIL)
print("after tell(number, FAIL):", study.trials[orphan.number].state.name)
```

Output (observed, Optuna 4.9.0):

```
is BaseHeartbeat: False
heartbeat enabled: False
orphan trial state: RUNNING
after fail_stale_trials: RUNNING
after tell(number, FAIL): FAIL
```

The full verification also SIGKILLed a real subprocess mid-trial and confirmed a *different*
process could fail the leftover trial by number. Two related recovery facts: the file lock a dead
worker was holding is forcibly released after a grace period (default 30 seconds) — a killed
lock-holder does not block the study for others — and a controller should treat "a config's Slurm
job disappeared without a result" the way `worker.py` treats rc != 0: tell FAIL, optionally requeue
(archiving the partial JSON first, per `run-id-and-logging.md`).

### 10.7 What the journal file costs at scale

Measured on the real `/p` filesystem, Optuna 4.9.0:

1. **Write concurrency**: 32 concurrent worker processes x 10 fast trials each = 320 trials, all
   recorded, trial numbers exactly 0..319, in 12.9 s total. Under that write pressure the log
   printed "It is taking longer than 10.0 seconds to acquire the lock file ... Retrying" — an
   informational wait notice, not an error. Real sweeps write once per training run (hours apart),
   nowhere near this pressure.
2. **File size**: about 614 bytes per simple one-value trial; a 9600-trial study measured 5.89 MB.
   Every `trial.report(...)` intermediate value is one more record, so per-seed reporting
   multiplies this (measured: 1000 trials with 20 reports each = 3.5 MB; records grow linearly, so
   a 9600-trial study with 20 reports each is about 34 MB — still small).
3. **Load cost**: the journal backend keeps no snapshot — every `load_study` (each worker start,
   each controller restart) replays the whole file. Measured: 0.10 s at 1000 trials, 0.77 s at
   9600. Budget it, but at one load per worker start it is nothing.
4. **Sampler cost at scale**: TPE refits its densities over all completed trials on every ask, so
   asks slow down as a study grows — a single-process fill of a 9600-trial TPE study (with per-step
   reports) had not finished after 16 minutes, while the same fill under `RandomSampler` took
   149 s. Do not drive one giant many-thousand-trial study with TPE. The §9.5 layout avoids this
   by construction: per-method studies, grids/enumeration for the exhaustive stages (constant-time
   asks), TPE only for a refinement stage of tens-to-hundreds of trials.

---

## 11. A concrete design for train run 3.2.1

Putting §9 and §10 together into the sweep I would actually run. The metric everywhere is the
final training-episode reward (the run's design, §9.1); every training run still writes its per-run
JSON checkpointed at the eval cadence — Optuna decides *what to run*, the durable record stays as
it is.

1. **Layout.** One journal file `optuna_journal.log` in the run folder (next to `queue/` and
   `data/`), with `JournalFileOpenLock` (§4.3). Three studies: `3_2_1_adam`, `3_2_1_adagrad`,
   `3_2_1_sgd1t`, all `direction="maximize"` (§9.5). One controller process (login node or a 1-CPU
   job) owns all three studies and drives the existing file queue; workers are unchanged except the
   one-line claim filter of §4.2. The controller writes a manifest of every decision (trial asked,
   seeds submitted, config stopped and why) — the adaptive-sweep equivalent of `SWEEPS.md`, so the
   allocation is reconstructable afterwards.
2. **Stage 1 — every configuration gets 10 seeds.** Per-method `GridSampler` studies over exactly
   the design grids (16 / 16 / 160; §9.5). One Optuna trial = one configuration; the controller
   fans each asked configuration out as 10 queue entries (seeds 0–9, controller-assigned, §4.1),
   collects the 10 results, and tells the trial their mean. Cost: 1920 training runs ≈ 28,500
   CPU-hours ≈ 2.5 days at run 3.1.2's observed rate (§10.1). No adaptive sampler is involved
   yet, so the §10.3 random-wave issue does not exist; the launch order within each grid is the
   seeded shuffle of §5.1.
3. **Stage 2 — race each method's survivors to 50 seeds.** The deliverable is each method's best
   configuration, so within a method the right stopping rule is *relative* (unlike run 3.1.2's
   absolute bar, §5.4): after each new result for configuration $c$ with $n_c$ seeds, mean
   $\bar{R}_c$, and standard deviation $s_c$, compute the interval ends
   $U_c = \bar{R}_c + 1.96 \cdot s_c/\sqrt{n_c}$ and $L_c = \bar{R}_c - 1.96 \cdot s_c/\sqrt{n_c}$,
   and stop $c$ once $U_c < \max_b L_b$ over that method's configurations $b$ — some other
   configuration's interval sits entirely above $c$'s. Keep submitting seeds (in batches, keeping
   the queue full) for every configuration still in the race, to the 50-seed cap. This is the
   controller-code analogue of what `WilcoxonPruner` does statistically (§5.4), placed where the
   seeds actually parallelize (§5.5). With the beta grid spanning 8 orders of magnitude, most
   configurations should stop at 10 seeds, so the expected total is far below the 9600-run grid —
   how far depends on how many near-best configurations each method has, which is exactly what the
   intervals measure. The §5.6 caution applies unchanged: bimodal per-seed rewards make 10-seed
   intervals wide, so near-best configurations will run deep, which is the correct behavior.
4. **Stage 3 (optional) — refine between the grid points.** Per method, a `TPESampler(seed=...,
   constant_liar=True)` (§10.4) or `GPSampler` study over continuous log-scale beta (and log eta0
   for O3), initialized from the stage-1/2 results imported one-trial-per-configuration with
   mean values (§4.4, §9.4); trial value = mean over a 10-seed batch (§6.6). This probes the runs'
   open question of where each method's beta optimum actually sits, rather than which of 8
   pre-chosen decades is least wrong.
5. **Reporting.** Each method's winner is the configuration with the best per-configuration seed
   MEAN, reported with its standard error and seed count — never `study.best_value` (§9.4). The
   winners' table follows the analysis conventions (best bold, second underlined); the winner
   rerun with distance logging stays as the design document specifies.

---

## 12. Version notes and the runnable example folders

Version facts to keep in mind when rereading this document later:

- Pinned context: Optuna **4.9.0**, installed 2026-07-03 into `exploration` with
  `conda run -n exploration pip install optuna` (brought in sqlalchemy, alembic, colorlog, Mako).
- Experimental in 4.9.0 (each prints an `ExperimentalWarning`; interfaces may change):
  `GPSampler`, `WilcoxonPruner`, `TPESampler(multivariate=True)`, `TPESampler(group=True)`,
  `TPESampler(constant_liar=True)`, `fail_stale_trials`.
- Deprecated in 4.9.0 with removal scheduled for v6.0.0: the TPE tuning knobs (`gamma`, `weights`,
  `consider_prior`, `prior_weight`, `consider_magic_clip`, `consider_endpoints`) — hard-wired to
  the §7 defaults; and the old storage import paths (`optuna.storages.JournalFileStorage`,
  top-level `optuna.storages.JournalFileOpenLock`). Use `optuna.storages.journal.*`.
- Heartbeats (`heartbeat_interval`, `fail_stale_trials` doing real work,
  `RetryHeartbeatStaleTrialCallback`) exist only on `RDBStorage` — not on the journal storage this
  cluster uses (§10.6).
- The `cmaes` package is not installed (so `CmaEsSampler` is unavailable); everything else in this
  document runs with the env as it is.

The nine verification folders under `07_reconstruction/optuna/code/` (each contains a `README.md`
with the exact rerun command for every script, and each script's raw output saved as
`<script>_output.txt`):

1. `2026-07-03-20-09_grid-sampler-behavior/` — the minimal workflow; all seven GridSampler
   behaviors of §5.1; `enqueue_trial`; trial states and `catch=`.
2. `2026-07-03-20-09_samplers-tpe-random-gp-comparison/` — the `suggest_*` forms (§3.1); the
   original four-sampler comparison behind §6.2's stability ranges; TPE determinism, mixed spaces,
   and startup behavior; GP runtime; the missing-`cmaes` behavior.
3. `2026-07-03-20-09_seed-allocation-pruning/` — the §5.3 threshold rule inline and as a custom
   `BasePruner`, with the three-repetition stability check (§5.6); WilcoxonPruner,
   SuccessiveHalving, and Hyperband on the same simulation (§5.4).
4. `2026-07-03-20-09_parallel-storage-and-ask-tell/` — storage classes and NFS guidance quotes
   (§4.3); 8 processes sharing one journal-file study; ask/tell through a pending/running/done
   file queue with 4 worker processes (§4.2); importing finished results (§4.4).
5. `2026-07-03-21-01_samplers-inline-comparison/` — the exact §6.2 comparison block and its
   output.
6. `2026-07-03-21-01_tpe-internals-and-mini-tpe/` — everything in §7: defaults, Parzen internals,
   candidate mechanics, the EI numerical check, the mini-TPE, and the constant-liar batch
   measurement (§10.4).
7. `2026-07-03-21-01_gp-internals-from-scratch/` — everything in §8: kernel check, from-scratch
   GP + EI, GPSampler agreement, per-ask timing.
8. `2026-07-03-21-01_multi-algorithm-per-method-best/` — everything in §9: conditional spaces,
   the shared-study allocation problem (including the 8-seed probe), per-method extraction,
   many studies in one journal file, per-method grids, per-study `enqueue_trial`.
9. `2026-07-03-21-01_parallel-scale-and-stale-trials/` — everything in §10: cross-process pruning,
   the per-process pruner trap, heartbeat facts and dead-worker cleanup, the startup wave, journal
   size/load/concurrency measurements, and the TPE-at-9600-trials cost observation.
