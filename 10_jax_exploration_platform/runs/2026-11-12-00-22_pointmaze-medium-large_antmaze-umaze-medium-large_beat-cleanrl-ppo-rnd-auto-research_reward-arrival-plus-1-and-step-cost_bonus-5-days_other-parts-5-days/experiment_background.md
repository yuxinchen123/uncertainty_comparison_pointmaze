## Purpose

Find a change to the intrinsic bonus (days 1 to 5), then to other parts (days 6 to 10), that is ahead of CleanRL
PPO-RND on PointMaze Medium and Large and AntMaze U-maze, Medium and Large. The fixed protocol (environments, the two
reward settings, the learner, the measure, the budgets, what may change) is `program.md`; the log is `STATE.md`; each
batch of ideas has a folder under `batches/` with `ideas.md` (written before its runs) and `summary.md`.

The RND baseline that every idea is compared with is in the run folder
`runs/2026-11-12-02-24_cleanrl-ppo-rnd-baseline_pointmaze-medium-large_antmaze-umaze-medium-large_goal-termination_reward-arrival-plus-1-and-step-cost_first-pass-60-configurations_seeds-0-15_confirmation-seeds-from-30000`.

## Key hyperparameters

They are those of `program.md`. Per batch, the screen's study description in the batch folder lists the swept
values:

| batch | case | steps per policy | seeds | swept |
|---|---|---|---|---|
| 1: the start, the scale and the decay of the bonus | PointMaze Large, reward 1 on arrival | 8,388,608 | 0 to 15 | intrinsic weight (12 values), and each idea's own setting (3 values where it has one) |
| 2: where the bonus points at the start, and features that keep the fit local | PointMaze Large, reward 1 on arrival | 8,388,608 | 0 to 15 | intrinsic weight (12 values), and each idea's own setting |
| validation block (ideas near the control, and the control) | PointMaze Large, reward 1 on arrival; PointMaze Medium, step cost | 8,388,608; 1,048,576 | 10000 to 10255 | the idea's best weights |

Fixed in batch 1: learning rate 2e-3 annealed to zero over the run, entropy 0.01, clip 0.1, 4 environments per
policy, 128-step rollouts, 256 logging windows.

## Code and config changes

Batch 1 adds options of the RND bonus, all off by default (`--bonus-options` of `scripts/run_training.py`): even
start, the reward divisor (running, frozen after the random-action rollouts, none), a separate predictor step size,
the error length as the bonus, a fast last layer inside the rollout, a scheduled last layer on fixed features, and a
record of the bonus on the maze grid.

## Git state

| batch | commit of the frozen code copy |
|---|---|
| 1 | `2b84cab185ef9030ec5225e413c32863e17ca4b0` (frozen 2026-11-12 13:32 PT; screen submitted 13:35 PT) |
| 2 | `d7054945b292c2fb2b0d8a6d4d7f8407659f59f5` (frozen 2026-11-12 15:26 PT; screen submitted 15:27 PT). The validation units of batch 1 still waiting at that time run on this copy too; with no option set, or with batch 1's options, it gives the same arrays as the earlier copy (job 6889910, 12 of 12 cases). |
