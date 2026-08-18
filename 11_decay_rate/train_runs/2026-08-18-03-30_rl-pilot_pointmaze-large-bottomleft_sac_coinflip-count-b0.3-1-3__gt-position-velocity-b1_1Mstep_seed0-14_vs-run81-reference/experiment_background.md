# Experiment background — RL pilot: coin-flip pseudo-count bonus in the SAC loop

## Purpose

Test whether the 11_decay_rate campaign's count-exact bonus construction matters end to end:
SAC on PointMaze_Large-v3 (start bottom-left, the run-8.1 setting) with the new
`coinflip_count` intrinsic bonus, against (a) run-8.1's recorded winners (RND at bonus
weight 3000: whole-run reward -331.6 +- 1.7, success 0.997; count oracle
gt_position_velocity at bonus weight 1: -337.8 +- 1.3, success 1.000; no_exploration:
-630.2, success 0.463) and (b) an in-pilot gt_position_velocity control arm that guards
against configuration drift relative to those recorded numbers.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| env_setup | `PointMaze_Large-v3_start_bottom_left` (reward shift -1, gamma 0.99, default episode cap — all carried by the setup) |
| algorithm arms | `coinflip_count` (beta 0.3, 1, 3), `gt_position_velocity` (beta 1, control) |
| coinflip knobs | defaults: d 128, insertion radius 0.5 (cell scale), bandwidth 0.1-0.5, 512 max centers, solve cache 256 visits |
| total_timesteps | 1,000,000 (run-8.1's budget, for comparability) |
| eval_freq / n_eval_episodes | 50,000 / 100 (run-8.1's) |
| seeds | 0-14 (15 per config; 60 runs total) |
| device | cpu |
| logging | per-run JSON via `--z_logging_mode=local` (the work-queue convention) |

## Code and config changes

- New `rnd_exploration.methods.coinflip_count` module + registry row (additive; committed with
  unit tests before this run). No changes to train.py, the buffer, or any existing method.
- Queue/worker scripts copied from run-8.1's `slurm/` and trimmed to this run's four configs.

## Git state

Commit recorded in `command.txt` at launch (the tree is committed before submission).
