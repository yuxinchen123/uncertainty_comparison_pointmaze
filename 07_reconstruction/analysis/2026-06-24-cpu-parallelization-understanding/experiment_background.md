## Purpose

Understand the CPU parallelization of `07_reconstruction/04_many_exploration_method.py`:
which parts of a training run can use multiple CPU cores and which are stuck on a
single core, with **measured evidence** (not assumptions), and what the wall-clock
performance difference is for **1, 4, 8, 16** CPU cores. Also: discover and
empirically verify this cluster's per-user resource limits.

The run profiles the real training loop (SAC on PointMaze_Large-v3 with switchable
intrinsic reward) at the actual tensor sizes used, plus per-primitive
microbenchmarks that isolate each numerical library (torch/MKL, numpy/OpenBLAS,
MuJoCo, pure-Python visit-count loop).

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| script profiled | `04_many_exploration_method.py` (faithful snapshot in `code/profile_e2e.py`) |
| env | PointMaze_Large-v3, max_episode_steps=400, fixed start + bottom_left goal |
| algorithm | SAC (SB3 MlpPolicy, net_arch [256,256]), batch_size 256 |
| algorithms swept | no_exploration, rnd_linear_next_state, gt_position (visit count), rnd_elliptical |
| beta | 0.01 (0 for no_exploration) |
| a_seed | 1 |
| thread caps | 1, 4, 8, 16 (OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_NUM_THREADS + torch.set_num_threads) |
| device | cpu (primary), plus one cuda comparison job |
| e2e total_timesteps / warmup | 4000 / 1200 (steady-state window excludes warmup) |
| microbench window | 2.5 s per primitive after 0.8 s warmup |
| replication | 17 distinct compute nodes (diverse CPU families) run the full sweep independently |

## Code and config changes

- New profiling module `code/cpu_profiler.py` — opt-in via env var `RND_PROFILE=1`
  (no-op when unset, so it can stay in a training script). Provides: `CpuSampler`
  (background CPU% / thread sampler), `StepRateProfiler` (SB3 callback measuring
  steady-state steps/sec and effective cores), `apply_thread_limit`, and
  `per_thread_cpu_active` (counts threads that actually accumulated CPU time).
- `code/microbench.py` — per-primitive thread-scaling benchmark at the real sizes.
- `code/profile_e2e.py` — snapshot of the env+model construction from
  `04_many_exploration_method.py`; reuses the project's real classes (RND /
  VisitCount / EllipticalBonus, PointMaze wrappers, VectorIntrinsicReplayBuffer),
  attaches the profiler, measures steady-state training throughput with
  intermediate eval disabled.
- `code/run_node_sweep.sh` + `code/node_sweep.slurm` — per-node sweep, fanned out
  one sbatch job per node (pinned `--nodelist`, `--cpus-per-task=20`,
  `--gpus-per-node=0`, gpu+cpu partitions). The UNMODIFIED `04_many_exploration_method.py`
  is also run under `/usr/bin/time -v` as a cross-check.
- No edits to `04_many_exploration_method.py` or any other project file.

## Cluster limits (discovered + empirically verified this session)

- Per-user, per-partition cap (QOS `cspartcpu` / `cspartgpu`, DenyOnLimit):
  **cpu = 400**, **gres/gpu = 40**, **mem = 4 TB**. Global `MaxJobCount = 100000`,
  `MaxArraySize = 2048`. No binding per-user job-count cap (association QOS = normal).
- Verified: a single job over a cap is rejected at submit with the exact QOS reason
  (`QOSMaxCpuPerUserLimit`, `QOSMaxGRESPerUser`, `QOSMaxMemoryPerUser`). The running
  aggregate is also capped: 22×20=440 cpus submitted to the cpu partition → exactly
  20 jobs (400 cpus) ran, 2 pended with `QOSMaxCpuPerUserLimit`.

## Follow-up: --ntasks / job-array / concurrency / memory (§5, added later)

A second set of experiments (forked session "cpu array") answers how `--ntasks`
vs `--cpus-per-task` combine, the other submission mechanisms, and memory:

- `code/probe_task_layout.py` — per-`srun`-task affinity / default torch threads / mem.
- `code/run_array_taskset.sh` + `code/array2_node.slurm` — per-node: co-located
  (4 workers pinned via `taskset` to cpus-per-task ∈ {1,2,4}) vs isolated (1 worker),
  plus a memory-bandwidth triad probe (1 vs 4 concurrent streams). Run on 5 nodes.
- `code/aggregate_array.py` → `results/array_concurrency*.csv`, `results/array_bandwidth.csv`,
  `plots/array_concurrency.png`.
- Submission-mechanism + queue-view demo (sleep jobs): `logs/array/queue_view_mechanisms.txt`.
- An earlier `srun`-substep driver (`run_array_experiments.sh` / `node_sweep`-style)
  hung at `--cpus-per-task=1`; replaced by the `taskset` driver. total_timesteps
  reduced to 1500 (warmup 500) for these runs.

Key results: ntasks=4 cpus-per-task=4 → 16 cpus, each task its own disjoint 4 cpus,
torch defaults to 4 threads/task (no oversubscription), `--mem` is per-node total
shared. Co-located 4-on-a-node costs 0.66× per run at cpus-per-task=4; aggregate
throughput peaks at cpus-per-task=2. Memory bandwidth saturates (7.8→6.2 GB/s per
copy at 4 concurrent).

## Git state

Commit: `3b212d7cc42901715199d71872701c9c0d6677b0` (branch `Use-RLexplore-RND`)

Working tree dirty (pre-existing local edits to the 07_reconstruction scripts;
this analysis adds only the new folder `analysis/2026-06-24-cpu-parallelization-understanding/`).
