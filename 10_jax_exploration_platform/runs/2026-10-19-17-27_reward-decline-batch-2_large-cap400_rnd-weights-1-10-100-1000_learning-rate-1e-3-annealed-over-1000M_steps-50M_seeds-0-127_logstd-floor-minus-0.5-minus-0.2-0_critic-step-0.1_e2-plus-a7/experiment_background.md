# Reward-decline study, batch 2: the policy's spread, the critic's step, and two holders combined

## 1. Purpose

Batch 1 (at 75% of its run) and the map screening point at one mechanism: in runs that fall, the
policy's log standard deviation drops (control, weight 10: -0.15 at the peak, -0.47 later) and the
ball slows (mean speed 4.5 -> 2.7), because much of its speed came from action noise clipped to
[-1, 1] while holding position inside the goal radius rewards less noise. Runs that hold keep
log std near -0.1 to -0.3 (episodes ending at the goal, E1/E2; large intrinsic weights). An entropy
bonus (A4) raised the level threefold but let log std grow without limit (+4.4).

| id | unit | the one change against batch 1's control |
|---|---|---|
| B1 | `b1_logstd-floor-minus-0.5` | log std kept at or above -0.5 (std at least 0.61) |
| B2 | `b2_logstd-floor-minus-0.2` | log std at or above -0.2 (std at least 0.82) |
| B3 | `b3_logstd-floor-0` | log std at or above 0 (std at least 1, its starting value) |
| B4 | `b4_critic-step-0.1` | the critic moves 0.1 times the step Adam proposes |
| B5 | `b5_terminate-at-goal_reward-shift-minus-1_gradient-clip-per-network` | E2 and A7 together |

The control is batch 1's: same copy count (512), seeds 0-127, environments and policy noise.

## 2. Key hyperparameters

As batch 1: large PointMaze, cap 400; RND with intrinsic weights 1, 10, 100, 1000 packed in one
512-copy program; learning rate 1e-3 annealed over 1000M steps; full-batch updates; 50M steps per
copy; records every 250 iterations with diagnostics and coverage.

## 3. Code and config changes

`--logstd-floor` (the floor is applied after every update), `--critic-lr-scale`, and batch 1's
`--terminate-at-goal --reward-shift -1 --clip-per-network`. Workers from two accounts claim from one
queue (sl5nw on jaguar03, yuxinchen on open RTX 2080 Ti cards).

## 4. Git state

Recorded at submission below.
