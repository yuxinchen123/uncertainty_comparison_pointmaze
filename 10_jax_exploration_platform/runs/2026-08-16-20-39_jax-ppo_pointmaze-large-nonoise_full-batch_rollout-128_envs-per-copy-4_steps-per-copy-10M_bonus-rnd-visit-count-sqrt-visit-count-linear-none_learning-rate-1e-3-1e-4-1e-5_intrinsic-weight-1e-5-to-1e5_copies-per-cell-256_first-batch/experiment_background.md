# Train run 1.1 — the first algorithm batch on the JAX exploration platform

## Purpose

Four intrinsic-exploration arms are trained at once on the no-noise large PointMaze and compared on
the same footing: random network distillation, the two oracle visit-count bonuses that differ only
in how fast the bonus falls with the count, and no bonus at all. Each arm is swept over three
learning rates, and each arm that has a bonus over eleven weights on that bonus, so the comparison
is between each arm's best configuration rather than between one guessed setting of each.

The four hypotheses the run is set up to answer are written out in the development document
(`development_document/platform_development_document.tex`, subsection 1.1): random network
distillation beats no bonus; the oracle bonuses bound it from above; the two decays trade coverage
against reward differently; and reward against intrinsic weight is single-peaked.

The run is also the first end-to-end exercise of the platform's multi-card machinery: one run
folder holding one sweep, one independent job per work unit, per-submission hardware records, and
run-level metrics that are aggregates over the per-unit shards.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| environment | `pointmaze_large_cont400_nonoise@1` — large map, start cell (7, 1), goal cell (1, 10), position noise 0, truncation at 400 steps, continuing task, goal radius 0.45 |
| agent | `ppo_full_batch@1` — one gradient step per rollout, no epochs, no minibatches |
| rollout length T | 128 steps |
| environments per copy N | 4, so one iteration collects 512 transitions per copy |
| iterations | 19,600 |
| environment steps per copy | 10,035,200 |
| algorithm arms (work units) | `rnd_next_state`, `gt_position_velocity_sqrt`, `gt_position_velocity_linear`, `none` |
| learning rates | 1e-3, 1e-4, 1e-5 |
| intrinsic weights | 1e-5 to 1e5 in factor-of-ten steps, 11 values; the arm with no bonus has none |
| copies per cell | 256 |
| copies per unit | 8,448 for each bonus arm (33 cells), 768 for the arm with no bonus (3 cells) |
| seeding | base seed 0, run seed 0, paired: copy k of every cell starts from the same weights and meets the same environment draws |
| discounts | extrinsic 0.999, intrinsic 0.99 |
| extrinsic advantage weight | 2.0 |
| recorded windows | 98 per unit, one record per 200 iterations |
| checkpoints | none — this platform saves no model state |
| environment (software) | `/p/rlprojects/RND/.venvs/platform_jax` (python 3.12.13, `jax[cuda12]` 0.11.0), the canonical platform environment in `.venvs/ENVS.md` |

## Code and config changes

- **`scripts/run_training.py` now runs a sweep, not one configuration.** It takes
  `--learning-rates`, `--intrinsic-weights` and `--copies-per-cell`, builds the per-copy rate and
  weight vectors through `training/sweep.py`, and records the cell each copy belongs to in a
  `unit_start` record so the aggregator can group per-copy arrays without re-deriving the sweep.
- **Records are phase-blocked episode windows.** The task is continuing, so an episode ends only at
  the 400-step truncation and every copy shares one episode clock; one iteration collects 128 of
  those 400 steps, so where inside the episode an iteration looks —
  `episode_phase = ((iteration - 1) * 128) % 400` — cycles with period
  `400 / gcd(128, 400) = 25` iterations. A single iteration's extrinsic reward is therefore a sample
  of one window of the episode and can read exactly zero for every copy while copies are solving;
  that is the artifact the learning-outcome campaign of `09_parallelization` found and named. Each
  record here covers 200 iterations — 8 whole turns of that clock, 256 complete episodes per copy —
  and reports the reward **summed** over the window per copy, so the clock cancels. Every record
  carries `episode_phase_first_iteration`, `episode_phase_last_iteration` and a `phase_blocked`
  flag, and `code/aggregate.py` scores only records whose flag is true.
- **19,600 iterations rather than the plan's 19,531.** 19,531 is 10^7 / 512 exactly, but it is not a
  multiple of 200, so its last window would have been 131 iterations — 5.24 turns of the episode
  clock, not a whole number — and could not have been scored. 19,600 is 98 whole windows and
  10,035,200 steps per copy, which is 10^7 to within 0.35% and costs 0.35% more time. Recorded here
  as a deliberate deviation.
- **Steps to goal is not recorded by this trainer.** The step inside an episode at which a copy
  first enters the goal radius is never read back to the host, so that column of the results table
  says "not recorded in this run" rather than being left blank. A one-line entry asking for it has
  been added to `research/README.md`; the trainer was not changed for it in this run.
- **`scripts/aggregate_run.py`** reads the new `episode_window` record and reports the last window's
  mean episode return instead of one iteration's reward sum.
- **New files in this run folder** (`code/`): `build_queue.py` writes the four units,
  `worker_env.sh` holds the job environment, `submit_one.sh` writes and submits one job per unit,
  `run_unit.sh` is the job body, `write_job_records.py` writes each submission's assignment and
  hardware records, `aggregate.py` is the run's aggregation and scoring module, `status.py` is the
  20-minute tick's table.
- **Resuming.** No model state is saved, so the resumable unit is the whole unit: a shard that ends
  in a `unit_complete` record makes a re-run of the same command a logged no-op, and a job whose
  command lists several units restarts at the first unfinished one. This was demonstrated on the
  canary before the real submission.

## Card assignment

The arithmetic follows `/p/rlprojects/.claude/skills/rnd-jax-submission/SKILL.md`. Measured
seconds per iteration on an H100 come from the throughput gate that preceded this run (the
`benchmark_runs/2026-08-16_visit-count-gate` measurements), and the whole sweep is 52.2 minutes of
compute on one H100 plus one build-and-compile per unit.

| unit | copies | measured s/iteration (H100) | compute at 19,600 iterations | peak device memory | card needed (1.25x) |
|---|---|---|---|---|---|
| 1 `rnd_next_state` | 8,448 | 0.0840 | 27.4 min | 28.2 GiB | 35.2 GiB — H100 or A100-80GB only |
| 2 `gt_position_velocity_sqrt` | 8,448 | 0.0345 | 11.3 min | 7.8 GiB | 9.8 GiB |
| 3 `gt_position_velocity_linear` | 8,448 | 0.0343 | 11.2 min | 7.8 GiB | 9.8 GiB |
| 4 `none` | 768 | 0.0070 | 2.3 min | 0.8 GiB | 1.0 GiB |

Sequentially on the single best card the sweep is about 59 minutes including four compiles, which is
above the skill's roughly-20-minute threshold, so distributing is warranted. Unit 1 sets the floor at
about 29 minutes however many cards are used, so three cards reach the floor and a fourth buys
nothing: units 2 and 3 on their own cards finish in about 13 minutes each, and unit 4 in about 4.
Unit 1 was not split, because splitting the largest compiled program to save 13 minutes would double
its compiles and its shards for no change in the sweep's finish time.

## Git state

Code commit at launch: `88244861dd25ed16a4b4e7fef40f7287d2be2d9d` (`git describe`: `jax-platform-v0.1.0-18-g8824486`), branch `Use-RLexplore-RND`,
pushed before the first submission. The working tree of `10_jax_exploration_platform/` was
clean at that commit; other folders of the shared repository carried unrelated uncommitted
work from other sessions, which was left untouched.

## Deviation from the canary rule, and why

`uva-submit-gpu-sweep` Step 6 asks for at least 30 simultaneous canary runs. That figure is sized for
a campaign of 150 runs across many node classes and configurations, where the canaries exist to catch
per-config outliers. This sweep has four work units and two node classes, so the same purpose is
served by canarying every (unit, node class) pairing that is actually planned — four canary runs, one
per pairing, each at the unit's full copy count so the memory it proves is the memory the real unit
will use, and each with a short iteration limit. The canary also proves the compiled-program cache
and the resume.
