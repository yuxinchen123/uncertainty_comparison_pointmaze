## Purpose

Prove that the extracted platform trains before anything in it is refactored, and exercise the run
folder's structure for the first time.

This run answers two questions:

1. **Does the copied code train end to end from the new folder, with the shared environment, on the
   graphics card?** The golden-parity gate already showed that the copied trainer computes exactly
   what the frozen `09_parallelization` baseline computes for three iterations. This run goes
   further: two hundred iterations, one hundred and twenty-eight copies, through
   `scripts/run_training.py`, with the shared environment `/p/rlprojects/RND/.venvs/platform_jax`
   on serval05's H100 NVL under the graphics-card lock.
2. **Does the run folder's structure work as written down?** Launch-time files, one shard per unit
   written line by line as the run goes, run-level files computed from the shards by the aggregator
   rather than by the training process, and no checkpoints. This run is one unit, so the
   aggregation is trivial by design — the point is that the run-level files come from the same code
   path that a many-job sweep will use.

It is a proof run, not a science run: two hundred iterations is a hundred thousand environment steps
per copy, far too short for the sparse goal to be found, so the extrinsic reward is expected to stay
at zero and only the maze coverage is expected to move.

## Key hyperparameters

| Parameter | Value |
|---|---|
| environment specification | `pointmaze_large_cont400_nonoise@1` (large map, start cell (7, 1), goal cell (1, 10), position noise 0, 400-step cap, continuing task, goal radius 0.45) |
| agent | PPO, one update per rollout batch (`full_batch`) |
| intrinsic bonus | Random Network Distillation on the next state, feature dimension 128, hidden width 256 |
| copies | 128 |
| environments per copy | 4 |
| rollout steps per iteration | 128 |
| iterations | 200 |
| environment steps per copy | 102,400 |
| environment steps in total | 13,107,200 |
| learning rate | 3e-4, annealed to zero over the 200 iterations |
| intrinsic-reward weight (`int_coef`) | 1.0 |
| extrinsic-reward weight (`ext_coef`) | 2.0 |
| discount, extrinsic / intrinsic | 0.999 / 0.99 |
| generalized-advantage parameter | 0.95 |
| clip, value coefficient, entropy coefficient, gradient-norm cap | 0.2 / 0.5 / 0.0 / 0.5 |
| observation-statistics priming iterations | 10 |
| base seed / run seed | 0 / 0 |
| coverage tracking | on (a per-copy visited-cell map kept on the device) |
| record cadence | every 10 iterations, and at the last iteration |
| precision | float32 throughout, float64 for the running statistics |
| device | one NVIDIA H100 NVL (95,830 MiB) on serval05, driver 580.159.04, under the exclusive lock |
| interpreter | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` — python 3.12.13, jax 0.11.0 |

The full resolved configuration is in `config_resolved.yaml`, and the exact command in
`command.txt`.

## Code and config changes

- The trainer is the copy at `10_jax_exploration_platform/src/exploration_platform/agents/ppo/jax_ppo_rnd.py`,
  which differs from `09_parallelization/ppo/jax_ppo/jax_ppo_rnd.py` in two import-path lines and
  nothing else (`../../ORIGIN.md`). The golden-parity gate holds it to that.
- The environment is the copy at `src/exploration_platform/envs/pointmaze/`, likewise unchanged
  apart from one import-path line.
- New for this run: `scripts/run_training.py` (runs one unit, writes the launch-time files and the
  shard) and `scripts/aggregate_run.py` (rebuilds `metrics.jsonl` and `summary.json` from the
  shards). Neither touches the trainer's arithmetic; the runner drives the same
  `prime_obs_rms` and per-iteration call the trainer's own `train` method uses, with the metric
  records flushed to the shard as they are produced instead of collected at the end.
- The environment `/p/rlprojects/RND/.venvs/platform_jax` was built for this stage and pinned to
  python 3.12 with `jax[cuda12]` 0.11.0 and numpy 2.5.2 — exactly the stack every
  `09_parallelization` number was measured on.
- `PYTHONNOUSERSITE=1` is set, so no package in `~/.local` can shadow the environment's own.
- No slurm. The run is one command on serval05 through
  `09_parallelization/locks/gpu_run.sh`, which takes the machine's exclusive graphics-card lock, so
  there is no `slurm/` folder and no job-id file for this run.

## Git state

Commit at launch: `d5c04ff8ea3e2cbe0e57a1e0e8f6b60de1f7bd52`
(tag at that point: `jax-rnd-baseline-v0.1.0` plus the commits of this stage; the platform is tagged
`jax-platform-v0.1.0` immediately after this run).

Working tree at launch:

```
 M RLeXplore
```

The one uncommitted item is the `RLeXplore` submodule pointer, which belongs to another effort in
this repository and is not part of this run's code. Everything this run executes — the trainer, the
environment, the runner, the aggregator — was committed before launch.
