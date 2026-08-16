# Where the copies-per-worker curve stops rising on jaguar03

## Purpose

The report's processor sweep runs independent single-thread worker processes on jaguar03 and stops
at 224 workers holding 16 copies each, that is 3,584 copies. Total throughput is still rising
there, so the measurement does not show the machine's ceiling. The knob left unpushed is copies
per worker: 224 workers is already the machine's logical-processor count, but each worker can hold
more copies.

This run answers three questions:

1. Where does total throughput stop rising as copies per worker grow, for each of the two update
   conventions (sixteen updates per batch, and one update per batch)?
2. What runs out at that point — the memory system's rate, the last-level cache, the sharing of a
   core by two hardware threads, or the memory the node has?
3. Is there any reason to run more copies per worker than that point, given that total throughput
   stops improving there while the rate one copy gets keeps falling?

## Key hyperparameters

| Parameter | Value |
|---|---|
| node | jaguar03, AMD EPYC 7663, 224 logical processors (112 cores, two threads each), 1 TB memory |
| node held | exclusively (`--exclusive`), matching how the earlier processor measurements were taken |
| parallelisation | independent processes, one thread each (`--mode processes`, `torch.set_num_threads(1)`) |
| workers | 224 for the ladder; 1 and 112 for the contention separation |
| copies per worker | 1, 4, 16, 32, 64, 128, 256, 512 — stopped by the memory guard or by the plateau |
| update conventions | `full_batch` (one update per batch) and `epoch_minibatch` (sixteen updates per batch) |
| environment steps per copy per iteration | 512 (`num_steps` 128 x `n_envs` 4) |
| iterations timed per point | chosen so each point times about 60 seconds of continuous work, between 3 and 150 |
| warm-up iterations | 1 |
| trainer configuration | `bench_train_cpu.cpu_config`: no recorded operation sequence, no reduced-precision matrix mode, no fused optimiser, no compiler |
| memory guard | a point is refused if its predicted memory exceeds 70% of the node's memory or would leave less than 150 GB free |

## Code and config changes

- `code/bench_copies_per_worker.py` (new). Runs the copies-per-worker ladder at a fixed worker
  count. It reuses `benchmarks/bench_train_cpu.py`'s `cpu_config`, so the trainer configuration
  cannot drift from the measurements already in the report, and adds four things that benchmark
  does not have:
  - peak resident memory per worker from `getrusage`, the sum across workers, and the node's own
    memory reading sampled every half second while the point runs;
  - a guard that predicts a point's memory from the points below it and refuses the point rather
    than let the node run out, which would lose every later point of the job;
  - an iteration count chosen so every point times about a minute of continuous work, because a
    three-iteration measurement reads the opening seconds of a load, when a server processor is
    still running above its sustained clock;
  - both readings from one run — the median over the first three iterations (the opening burst,
    comparable with the earlier short measurements) and the median over all of them.
- `code/memory_pressure.py` (new). Two measurements of whether the memory system is the limit: the
  rate the node delivers to a growing number of stream processes run alone, and the rate those same
  stream processes get while the training load runs beside them at a small and a large
  copies-per-worker setting.
- `code/test_bench_copies_per_worker.py` (new). Unit tests for the iteration-count choice, the two
  memory-prediction lines, the rate arithmetic and the opening-burst split.
- Nothing in `benchmarks/`, `ppo/` or the report generators was changed to take the measurements.

Points at or below 16 copies per worker repeat settings the report already has a file for. They are
measured anyway — a plateau read off rungs timed for different lengths is not a plateau — but they
are written to this run folder only, so the report's existing rows are not replaced by numbers
taken under a different timing window. Points at 32 copies per worker and above are new settings
and are written to `benchmarks/results/`, where the report's loader reads them.

## Git state

Commit: `b535f39` (branch `Use-RLexplore-RND`).

The working tree carries other sessions' in-flight work in other folders; only this run folder's
code, its Slurm script and this file were committed for this run.
