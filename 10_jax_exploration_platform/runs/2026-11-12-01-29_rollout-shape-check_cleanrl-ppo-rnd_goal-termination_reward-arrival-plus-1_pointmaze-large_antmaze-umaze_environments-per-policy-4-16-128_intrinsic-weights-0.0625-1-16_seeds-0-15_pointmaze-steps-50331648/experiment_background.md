## Purpose

Choose how many parallel environments one policy uses (4, 16 or 128, each with 128-step rollouts and 4 epochs x 4
minibatches) for the PointMaze family and for the AntMaze family, before the RND baseline is swept (plan
`~/.claude/plans/2026-11-11-22-40_pointmaze-antmaze-speed-and-beat-rnd.md`, section 1.3; the user's decision of
2026-11-11: "chosen by a baseline check"). The count with the best whole-curve mean of the training extrinsic return
for the same number of environment steps is kept for the family and then fixed for RND and for every method.

Times in this folder are Pacific (the machines run on Eastern).

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| learner | CleanRL PPO-RND values: clipped PPO, 4 epochs x 4 minibatches, 128-step rollouts, clip 0.1, learning rate 1e-4 annealed to zero over the run, entropy 0.001, extrinsic coefficient 2, discounts 0.999 and 0.99, GAE 0.95, 50 priming rollouts; the predictor trains on the whole batch |
| environments per policy | 4, 16, 128 (one unit each) |
| intrinsic weight | 0.0625, 1, 16 (three cells per unit) |
| seeds | 0 to 15 (16 policies per cell, 48 per unit) |
| PointMaze | Large map, start (7, 1), goal (1, 10), 400-step limit, episode ends at the goal, reward 1 on arrival |
| PointMaze budget | 50,331,648 steps per policy (192 logging windows of 262,144 steps) in every unit |
| AntMaze | U-maze, 700-step limit, same reward; started after the ant's physics settings are decided by the speed study, with its budget written here then |
| measure | whole-curve mean of the training extrinsic return (the share of training episodes that reach the goal), and the final level over the last 10% of steps |

## Code and config changes

Configuration file `configs/experiments/pointmaze_large_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-1024000.toml`
with these options changed per unit: `--envs-per-copy`, `--intrinsic-weights 0.0625,1,16 --copies-per-cell 16`,
`--iterations`, `--window-iterations`, `--lr-anneal-iterations` (equal to the iterations), and
`--gather-minibatches-per-epoch` for 128 environments. Job script: `infrastructure/unit_job.sh`.

## Git state

Commit `528081b5` (pushed): the trainer changes of the study.
