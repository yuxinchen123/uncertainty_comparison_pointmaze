# Optuna parallel-scale and stale-trial verification

Date: 2026-07-03. Verified on optuna 4.9.0 (Python 3.11, numpy 1.26.4, scipy 1.16.0, torch 2.10.0).

These scripts verify how Optuna behaves when a sweep runs as hundreds of 1-CPU worker
processes that share one journal-file study on an NFS folder, and when workers die at
random (Slurm walltime, per-run stopwatch, node failure). Every script runs locally,
uses genuine separate OS processes (it re-invokes itself with a role argument through
`subprocess`), and finishes in a few minutes.

Storage used everywhere:
`optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(path, lock_obj=optuna.storages.journal.JournalFileOpenLock(path)))`.

## Scripts and how to rerun

Run each with the project env; the second form also saves the printed output.

- `s1_pruning_across_processes.py` — MedianPruner state flows across processes through
  the shared journal. Process A completes one good trial; process B is pruned using A's
  history. Also shows the pruner is NOT stored in storage: each process must pass its own.
  - `conda run -n exploration python -u s1_pruning_across_processes.py 2>&1 | tee s1_pruning_across_processes_output.txt`

- `s2_stale_trials_heartbeat.py` — (a) journal storage has no heartbeat; `fail_stale_trials`
  is a silent no-op. (b) a subprocess asks a trial then is SIGKILLed; the trial stays
  RUNNING. (c) recovery: a different process calls `study.tell(number, state=FAIL)`.
  - `conda run -n exploration python -u s2_stale_trials_heartbeat.py 2>&1 | tee s2_stale_trials_heartbeat_output.txt`

- `s3_startup_wave.py` — TPESampler(seed=0, n_startup_trials=10): 40 asks with zero tells
  are bit-identical to RandomSampler(seed=0), because n_startup_trials counts COMPLETED
  trials (0 here). All 40 trials stay RUNNING.
  - `conda run -n exploration python -u s3_startup_wave.py 2>&1 | tee s3_startup_wave_output.txt`

- `s4_journal_at_scale.py` — (a) 32 worker processes x 10 trials = 320 into one journal;
  verify 320 COMPLETE with unique numbers and measure wall time. (b) grow a study to 1000
  trials; report journal file size and a fresh process's load time.
  - `conda run -n exploration python -u s4_journal_at_scale.py 2>&1 | tee s4_journal_at_scale_output.txt`

- `s4b_sweepscale_probe.py` — same as s4(b) but at the real sweep size (9600 trials), using
  RandomSampler so ask() is constant-time (the default TPESampler refits over all completed
  trials on every ask, which dominates and confounds a storage measurement). Reports fill
  time, journal size, and cold-open time at 9600 trials.
  - `conda run -n exploration python -u s4b_sweepscale_probe.py 2>&1 | tee s4b_sweepscale_probe_output.txt`

- `s5_two_parallel_patterns.py` — (a) worker-owned optimize: 8 workers each run
  study.optimize; an observer process sees RUNNING trials from the other processes.
  (b) controller ask/tell: one controller keeps 8 asked-but-untold trials, workers only
  evaluate via a file queue. Shows where the sampler runs in each pattern.
  - `conda run -n exploration python -u s5_two_parallel_patterns.py 2>&1 | tee s5_two_parallel_patterns_output.txt`

Each `<script>_output.txt` next to the script holds that script's real printed output.

The `snippet_1.py` .. `snippet_5.py` files are the trimmed, self-contained one-file
versions used in the tutorial; each has its own `snippet_N_output.txt`.

## Generated data files (safe to delete)

`s1_journal.log`, `s2_journal.log`, `s2_probe.log`, `s4a_journal.log`, `s4b_journal.log`,
`s4d_journal.log`, `s5a_journal.log`, `s5b_journal.log`, their `.lock` files, and the
`s5b_ask/`, `s5b_done/` queue dirs are regenerated on each run.
