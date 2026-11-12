# Program: beat CleanRL PPO-RND on PointMaze and AntMaze

Fixed before any run (2026-11-12 PT). Source of every item: the approved plan
`~/.claude/plans/2026-11-11-22-40_pointmaze-antmaze-speed-and-beat-rnd.md` and the user's decisions of 2026-11-11.
Nothing in this file changes without the user, except the lines marked "set during the baseline".

## The goal

A new method whose training curve of extrinsic return beats swept CleanRL PPO-RND in all five environments, with
statistical significance, in reward setting A or in reward setting B.

## Environments and rewards

| family | map | start (row, column) | goal | step limit |
|---|---|---|---|---|
| PointMaze | Medium | (6, 1) | (1, 6) | 400 |
| PointMaze | Large | (7, 1) | (1, 10) | 400 |
| AntMaze | U-maze | (3, 1) | (1, 1) | 700 |
| AntMaze | Medium | (6, 1) | (1, 6) | 1,000 |
| AntMaze | Large | (7, 1) | (1, 10) | 1,000 |

- Fixed start and goal, no reset noise. The goal is not observed.
- Reaching the goal (within 0.45 m) ends the episode as a termination: no value is added after it. The step limit is
  a truncation: the extrinsic return continues with the value of the last state. The intrinsic return runs across both.
- Define $H$ as the step limit and $t$ as the step at which the goal is reached.
  - **Setting A:** 0 per step, +1 on arrival.
  - **Setting B:** $-1/H$ per step, 0 on arrival.
  - Extrinsic discount 0.999 in both.

## The learner (RND baseline and every method)

CleanRL PPO-RND: clipped PPO, 4 epochs × 4 minibatches, 128-step rollouts, clip 0.1, learning rate annealed linearly
to zero over the run's own budget, advantages normalized per minibatch, extrinsic value loss clipped, extrinsic and
intrinsic coefficients 2 and (swept), discounts 0.999 and 0.99, GAE 0.95, value coefficient 0.5, joint gradient clip
0.5, RND input whitened and clipped to ±5 after 50 random-action rollouts, intrinsic reward divided by the running
standard deviation of its discounted sum. **One change from CleanRL, in every run: the predictor trains on the whole
batch (no 25% subsampling).** Networks as in the platform: policy and value 64-64 tanh; RND target 256-128,
predictor 256-128-128.

- Environments per policy: set during the baseline (rollout-shape check), one value per family.
  - **PointMaze: 4** (set 2026-11-12 02:05 PT). On PointMaze Large at 50.3 million steps per policy, intrinsic
    weight 1, the whole-curve mean was 0.89 with 4 environments per policy, 0.70 with 16 and 0.24 with 128; at
    20.7 million steps it was 0.79 with 4 and 0.54 with 2; at 11.8 million steps 0.67 with 4 and 0.03 with 1.
  - AntMaze: not set yet (after the ant's physics settings are decided).
- First-pass sweep: intrinsic weight $4^{-6}, 4^{-5}, \dots, 4^{3}$ (10 values) × learning rate {1e-4, 3e-4, 1e-3} ×
  entropy {0.001, 0.01} = 60 configurations × 16 development seeds (0 to 15).
- Edge rule: a chosen intrinsic weight at an end of the grid extends the grid by three values on that side before
  anything is confirmed.
- Second pass, around the chosen cell only: intrinsic weight at factor 2; then one at a time the intrinsic discount
  {0.99, 0.999} and the clip {0.1, 0.2}.
- A new method: the same first pass, plus its own one or two most important settings at three values each.

## The measure

Define $K$ as the number of seeds, $B$ as the budget in environment steps per policy, and $R_i(s)$ as the mean
extrinsic return of the training episodes of policy $i$ that end in the logging window containing step $s$ (a window
with no finished episode takes the last value).

- Whole-curve mean $W(B)$: the mean of $R_i(s)$ over all steps up to $B$ and over the $K$ policies.
- Final level $L(B)$: the same over the last 10% of the budget.
- Configurations are chosen by the highest $W(B)$ on the development seeds (ties: higher $L(B)$).

**Beat (one environment, one reward setting, same fresh seeds):** the 95% interval of the difference in $W$ (method
minus RND) is entirely above zero, and the 95% interval of the difference in $L$ is not entirely below zero.
Intervals: 10,000 resamples of seeds on the paired differences.

**Success:** beat in all five environments in setting A, or in all five in setting B.

**Fresh seeds:** from 30000; 1,024 on PointMaze, 256 on AntMaze (1,024 if the speed work makes it affordable).

**Reported beside, never used to choose:** mean capped first-goal time, share of policies that ever reach the goal,
maze-cell coverage, mean steps to the goal, share of training episodes reaching the goal in the last 10%; the final
level at 25%, 50%, 75% and 100% of the budget and the curve to twice the budget for one map per family.

## Budgets (set during the baseline, then fixed)

$B$ per environment and reward setting = twice the step count at which the best three RND configurations reach 95%
of their final level, rounded up to a power of two. Written here before any method is compared.

| environment | setting A | setting B |
|---|---|---|
| PointMaze Medium | 1,048,576 (set 2026-11-12) | 1,048,576 (set 2026-11-12) |
| PointMaze Large | 8,388,608 (set 2026-11-12) | 33,554,432 (set 2026-11-12) |
| AntMaze U-maze | (not set yet) | (not set yet) |
| AntMaze Medium | (not set yet) | (not set yet) |
| AntMaze Large | (not set yet) | (not set yet) |

How the PointMaze budgets were read (second pass of the development sweep, 16 seeds, 256 logging windows, curve
smoothed over 5% of the run): the best three configurations reach 95% of their final level at 262, 328 and 344
thousand steps on Medium with reward 1 on arrival, at 279, 295 and 328 thousand on Medium with the step cost, and at
2.75, 3.47 and 4.19 million on Large with reward 1 on arrival.

RND's chosen configuration (development runs trained for exactly the budget, 16 seeds):

| case | learning rate | entropy | intrinsic weight | clip | whole-curve mean | final level |
|---|---|---|---|---|---|---|
| PointMaze Large, reward 1 on arrival | 2e-3 | 0.01 | 2 | 0.1 | 0.888 ± 0.008 | 0.994 |
| PointMaze Medium, reward 1 on arrival | 6e-3 | 0.01 | 2 | 0.1 | 0.942 ± 0.007 | 0.999 |
| PointMaze Medium, step cost | 6e-3 | 0.01 | 0.001953125 (replaced 0.0078125 on 2026-11-12 after lower weights were tried) | 0.1 | −0.3405 ± 0.003 | −0.304 |
| PointMaze Large, step cost | 2e-3 | 0.01 | 4 | 0.1 | −0.7247 ± 0.0009 | −0.712 |

## What may change, and when

- Days 1 to 5 (to 2026-11-16 PT): the bonus only (how the intrinsic reward is initialized, computed, decayed,
  refreshed and normalized). PPO, the policy and value networks, and the shared sweep stay as in the baseline.
- Days 6 to 10 (to 2026-11-21 PT): the other parts, each tied in `STATE.md` to something observed in days 1 to 5.
- Never the result: a re-implementation of a published RND variant (NGU, NovelD, DRND, coin-flip networks, E3B,
  random distribution distillation, self-supervised network distillation). `literature/novelty.md` names the closest
  published work and the exact difference before finalists are chosen.
- Every idea is compared with the same confirmed RND baseline on the same seeds. The final method is also confirmed
  with each of its parts removed (256 fresh seeds per case). Every case is reported as it comes out.

## The same treatment for RND and for every method

Episodes end at the goal; the learning rate is annealed within the run; the entropy coefficient is in both sweeps. A
chosen configuration whose final level is below 90% of its curve's own peak is reported and diagnosed, and any fix
is applied to RND and to the method alike before they are compared.
