## Purpose

Measure how fast the JAX PPO+RND trainer (`09_parallelization/ppo/jax_ppo/jax_ppo_rnd.py`, the
single-update style, PointMaze Large with no reset noise on the start or the goal) runs on every
distinct graphics-card configuration this cluster offers, at 512, 1,024, 2,048 and 4,096 training
copies, with 8, 16 and 32 processors.

The question the numbers answer: for a given number of copies, which card should a run be sent
to, how long will it take, and which cards cannot hold the run at all.

Times are displayed in Pacific Time with a `PT` marker; the cluster's machines run Eastern.

## Key hyperparameters

| Parameter | Value |
|---|---|
| trainer | `JaxPPORND` (`ppo/jax_ppo/jax_ppo_rnd.py`) |
| update style | `full_batch` — one gradient step per rollout, the single-update style |
| environment | PointMaze Large, start cell (7, 1), goal cell (1, 10), `position_noise` 0.0 |
| copies swept | 512, 1,024, 2,048, 4,096 |
| processors swept | 8, 16, 32 — and each node class's own maximum where 32 is unallocatable |
| environments per copy | 4 |
| rollout length | 128 steps |
| environment steps per iteration | 512 per copy |
| learning rate | 3e-4, annealed |
| discount (extrinsic / intrinsic) | 0.999 / 0.99 |
| intrinsic / extrinsic advantage weight | 1.0 / 2.0 |
| random-network feature dimension | 128, hidden 256 |
| seed | `base_seed` 0 |
| node classes covered | 27 — every class of the `gpu` and `gnolim` partitions |
| measurement cells | up to 4 copy counts x 80 (node class, processor count) jobs |

## How a number is measured

One Slurm job = one node class at one processor count, pinned to a named node with
`--nodelist` so the measurement knows what hardware it measured. Inside the job each copy count
runs as its own process (`code/bench_cell.py` under `code/run_job.py`), so a card that runs out
of memory at 4,096 copies still reports 512, 1,024 and 2,048.

Within a cell:

1. The trainer is built and the whole iteration compiled; that first iteration is discarded.
2. Five warm-up iterations are discarded — allocator growth, autotuning, and the card reaching
   its sustained clock.
3. Iterations are then timed in rounds sized to about two seconds each. Rounds continue until
   the middle half of the per-iteration times sits within 2% of their median, with a floor of
   six rounds and a ceiling of twenty-four.
4. The reported figure is the median across rounds; the minimum, the maximum, the spread and
   every round's own value are kept in the result file, so a number that never settled is
   visible as such rather than quoted as if it had.

Each job is held under a 27-minute deadline, inside a 45-minute Slurm walltime: the remaining
time is divided equally among the copy counts still to run, and a copy count that cannot start
before the deadline is recorded as not attempted instead of being started and killed. The
45-minute walltime is deliberately bounded, as the user asked for runs under 30 minutes; it is
the profiling-job exception to the usual "request the maximum walltime" rule.

## Code and config changes

- New shared environment `/p/rlprojects/RND/.venvs/jax_gpu` (Python 3.11, JAX 0.10.2 with the
  CUDA 12 plugin). The project's existing `jax_bench` environment is a symlink into a private
  home directory and reaches no graphics card; `exploration` has no JAX at all. (`jax_bench` was
  deleted on 2026-08-16 at the owner's request; the platform's canonical JAX environment is the one
  registered in `/p/rlprojects/RND/.venvs/ENVS.md`, currently `/p/rlprojects/RND/.venvs/platform_jax`.)
- The CUDA libraries had to be installed into the environment explicitly
  (`jax-cuda12-plugin[with-cuda]`). The first install resolved them as already satisfied from
  the user-site directory `~/.local`, which every job hides with `PYTHONNOUSERSITE=1`, so the
  first probe wave failed on all 23 nodes it reached with "Unable to load CUDA" while
  `nvidia-smi` on the same node showed a healthy card.
- No change to the trainer or the environment: the survey imports
  `ppo/jax_ppo/jax_ppo_rnd.py` as committed.
- `XLA_PYTHON_CLIENT_PREALLOCATE=true` with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`: the card is
  the job's alone, and a preallocated pool keeps the timing free of allocator growth between
  iterations. The memory figure the report quotes is peak bytes in use, not the pool size.

## Git state

Commit: `432a67e42bed03b9c479a8acf928c3abdda61921`

The trainer and environment code (`ppo/jax_ppo/jax_ppo_rnd.py`, `pointmaze/common/pm_common.py`)
were already committed at `199f857` ("PointMaze: no reset noise on the start or the goal") and
are unchanged by this survey.
