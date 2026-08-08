# Experiment background — train run 6: the four train-run-8.1.2 algorithms on the train-run-5 PointMaze task

## Purpose

Train run 8.1.2 built four algorithm arms (1, 2.1, 2.2, 2.3) and raced them on two AntMaze
environments, where nothing converges (see the writeup's "Why the AntMaze runs do not converge").
Train run 6 asks the same comparison question on the task where the original RND demonstrably
works: train run 5's PointMaze Large top-right task. One question:

> Do any of the four algorithm variations beat the run-5 original RND (Adam 10⁻⁴, best bonus
> weight 10³, final reward 38.6412 over 300 seeds) on its own task, under its own score rule?

The original RND is NOT re-run — its 300-seed final data from train run 5 is the frozen reference.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| environment | `initial_single_large_pointmaze_max_400` — train run 5's task verbatim: PointMaze Large, top-right goal, episode cap 400, discount 0.999, raw sparse reward |
| base RL algorithm | SB3 SAC `MlpPolicy`, defaults (Adam 3e-4, batch 256, tau 0.005, buffer 1e6, auto entropy) |
| the four arms | train run 8.1.2's, parameter dictionaries copied VERBATIM and pinned by unit test: alg1 = plain constant-rate SGD predictor, l2 readout, bias normal 0.5, reward norm OFF; alg2.1 = alg1 + frozen-init predictor copy, ratio bonus; alg2.2 = 2.1 + init-normalized loss; alg2.3 = 2.1 + LayerNorm |
| predictor learning rate | swept {10⁻³, 10⁻²} (plain SGD, constant) |
| bonus weight | swept over the 15 values 10⁻³, 3×10⁻³, 10⁻², 3×10⁻², 10⁻¹, 3×10⁻¹, 1, 3, 10, 30, 10², 3×10², 10³, 3×10³, 10⁴ |
| configurations | 4 arms × 2 learning rates × 15 bonus weights = **120** |
| steps per run | 1,000,000 |
| seeds | `a_seed` 1200–1499 (300 fresh; disjoint from run 5's 600–899 and the addendum's 900–1199) |
| score rule | train run 5's: final reward = last `train_history` row's `train/mean_extrinsic_reward` |
| frozen bar | 38.6412 (run-5 original RND, Adam 10⁻⁴, β=10³, n=300) — `slurm/FROZEN_BARS.json`, sha256 recorded on every decision |
| truncation phase 1 | from 20 completed seeds: truncate when mean + 2.576·s/√n < 38.6412; survivors race to 100 |
| truncation phase 2 | at 100 seeds, once every sibling of an arm is resolved: the arm's best-mean configuration CONTINUES to 300 seeds; the other candidates stop (verdict `stopped_at_100`) |
| logging | one JSON per run under `data/<sweep_id>/local/`, checkpointed at each 50k-step eval with a `completed` flag; wandb off |

## Code and config changes

- New run folder; worker/queue/monitor machinery copied from the proven learning-rate addendum
  run (walltime guard, plan_jobs with the pool-sizing and socket-pinning fixes, requeue_orphans).
- New two-phase `slurm/truncation_controller.py` + rewritten
  `20_mins_monitoring/truncation_check.py` (7 invariants incl. per-arm winner uniqueness and
  winner-mean dominance) + per-arm `monitoring_report.py`.
- Launch gates: 24 unit tests (queue convention, arm byte-equality against train run 8.1.2's
  build_queue.py, two-phase controller, walltime guard) and `slurm/simulate_truncation.py --full`
  at full scale — 120 synthetic configurations: 108 truncated at the 20-seed floor, 8 stopped at
  100, 4 winners complete at 300, 0 invariant violations at every wave.
- Submission: cpu → nolim → reservation (the user's order for this run; no gpu/gnolim), per the
  shared submit-cpu-sweep skill. Collaborator packet in `for_collaborator/`.

## Git state

Commit at folder creation: `2cde90d7813969af1241a61117b6880315cf0924`; the launch commit (with
this folder's code) is recorded below when the queue is built.

Launch commit: `5df4463` (queue built 2026-08-08 02:46, sweep_id 2026-08-08-02-46_run6)
