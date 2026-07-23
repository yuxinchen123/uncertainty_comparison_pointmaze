# Point maze + ant maze train run 1 (writeup subsection 8.1) — experiment background

Launched 2026-07-23 by sl5nw (session: the section-8 plan of
`~/.claude/plans/2026-07-22-17-40_antmaze-section-4-and-hyperparameter-table-skill.md`).
Writeup: `development_document/main.tex` section 8 ("point maze + ant maze"), subsections 8.0
(environment specification) and 8.1 (this run's frozen env table + algorithm table).

## Purpose

First run on the 8 new env configs (the `start_bottom_left` set): compare SAC alone, SAC + RND
(the train-run-5 original-small stack), and SAC + ground-truth visit-count bonuses on 4 PointMaze
and 4 AntMaze environments, sweeping the bonus weight over 15 values with seed racing. Hypothesis
1 (writeup 8.1): the ground-truth bonus performs better than RND on every PointMaze config in
this run.

## Environments (8 configs, one per env id; the named EnvSetup dataclasses are the source of truth)

`rnd_exploration/envs/env_setups.py` — `PointMaze_{UMaze,Open,Medium,Large}-v3_start_bottom_left`
and `AntMaze_{UMaze,Open,Medium,Large}-v5_start_bottom_left`. Shared setup: start bottom-left
corner cell, goal top-left (UMaze) / top-right (others), EXACT cell centers
(`position_noise_range=0`, set on the built env), `continuing_task=False` (terminated=True at
distance <= 0.45 m), `reset_target=False`, reward shift −1 (−1 per step, 0 on the goal-reaching
step; ExPLORe convention), registered default episode limits (PointMaze 300/300/600/800; AntMaze
700/700/1000/1000), SAC discount 0.99. AntMaze state = 29-d (achieved_goal x,y re-attached +
27-d body state, contact forces off); PointMaze state = 4-d. Goal keys never observed.

## Algorithms (308 configs total; per-cell counts in `slurm/build_queue.py`)

1. SAC alone = `no_exploration` (beta forced 0) — 1 config per env, 8 total.
2. SAC + RND = `rnd_next_state` with the train-run-5 original-small stack verbatim (adam 1e-4,
   mse_mean readout, leaky_relu 0.2, predictor +1 layer, out 128, env-steps warmup 6400, reward
   norm gamma 0.99) — 15 bonus weights per env, 120 total.
3. SAC + gt_position_velocity (PointMaze only; 1 m position cell x 10x10 velocity bins) — 60.
4. SAC + gt_position_maze_cell (AntMaze only; 4 m maze-cell counts) — 60.
5. SAC + gt_position_1m (AntMaze only; 1 m sub-grid counts) — 60.
Ground-truth bonus = min(1, 1/sqrt(n)) (`visit_count_decay=-0.5`).
Bonus weights (all bonus algorithms): 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 1e2,
3e2, 1e3, 3e3, 1e4.

## Seed racing (the shared `sweep_prune` skill; THIS run's overrides)

Seeds 0–299 (up to 300 per config), all enqueued up front, seed-outermost ids (seed s owns ids
[308s .. 308s+307]). Score = whole-run mean per-episode extrinsic return (mean over every
`train_episode_history` row; completed records only). Cell = (env_setup, algorithm). Prune floor
**20** (overrides the skill default 30): config c is pruned when
mean_c + 2.576*s_c/sqrt(n_c) < the cell's best mean (both n >= 20). Winner-only continuation at
**100**: once every surviving config of a cell has 100 finished seeds, only the cell's best keeps
its pending seeds, to the **300** maximum. Controller: `slurm/prune_controller.py`
(`--n_floor 20 --n_winner 100`); every decision logged to
`slurm/prune_decisions_<sweep_id>.jsonl` and re-verified each tick by
`20_mins_monitoring/prune_check.py`.

## Logging

Per-run JSON (run-id-and-logging convention) under `data/<sweep_id>/local/`, one file per run,
checkpointed each 50k-step eval cadence, `completed` flag. New fields this run: `env_setup`,
`env_name`, cells, `reward_shift`, `continuing_task`, **`device`** ("cpu"/"cuda" — ONE shared
queue runs on both device types; each record says where it computed), per-episode
`train/success` + `train/steps_to_goal`, per-eval `visit_counts_1m/*` (1 m coverage; equals the
cell coverage for PointMaze). `eval_standalone=False` (scored on training episodes),
`log_distance=False`, wandb off.

## Code / environment

- Python: `/p/rlprojects/RND/.venvs/exploration/bin/python` (shared canonical env; gymnasium
  1.2.3, gymnasium-robotics 1.3.1, mujoco 3.1.6, SB3 2.7.1).
- Git state: branch `Use-RLexplore-RND`, commit `6c0dd0b` (commit-before-submit; recorded
  here at first sbatch).
- Entry point: `07_reconstruction/train.py --env_setup <name> ...` (the worker builds argv from
  the queue JSON; `--device` overridden per node type via `WORKER_DEVICE`).

## Launch / monitoring

Submission per `submit-cpu-sweep` + `submit-gpu-sweep`: resource probe first (CPU runtime/RSS per
family; GPU packing count W per the canary rule), then cpu -> nolim -> gpu (device=cuda only) ->
gnolim -> reservation on puma01 ONLY (jaguar03 skipped, user instruction 2026-07-23). CPU shapes
14x1 / 12x1 / 30x1 / 28x1 per node class. Job ids only in `slurm/submitted_jobids_<sweep_id>.txt`.
Monitoring: 20-minute loop per the shared `sweep-monitoring` skill — `slurm/monitor.sh` each tick
runs requeue, the prune controller, `prune_check.py`, and prints the running-status table +
the 8 per-env interim metrics tables (`20_mins_monitoring/`), snapshotting to
`20_mins_monitoring/outputs/`. Collaborator packet: `for_collaborator/` generated after the probe.

## Analysis plan

After the sweep: per-env result tables (one row per algorithm's best config, all knobs spelled
out; reward, success rate, steps-to-goal on successes, maze-cell coverage, 1 m coverage; each
mean ± standard error + completed-seeds N; ranked by reward, bold best / underline second best)
assembled 4x2 in the writeup, plus the 4x2 reward-curve plot of each algorithm's best config.
Only `completed=true` records enter aggregates; the per-run `device` mix is reported as a
caveat line.
