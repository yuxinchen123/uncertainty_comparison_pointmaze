# Batch 13: the fast last layer until the first arrival, then the divisor of the predictor's own scores

Generated 2026-11-13 11:35 PT by the command below. This file is generated: regenerate it, do not edit it. Times are Pacific, converted from machines running Eastern. Run folder: `2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days`; batch folder: `batches/13_fast-last-layer-until-the-first-arrival_then-the-predictors-own-divisor_four-pointmaze-cases`.

```bash
cd /p/rlprojects/RND/10_jax_exploration_platform
bash "benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh" -s scripts/maze_campaign/analysis/batch_summary_job.py \
    --run-dir=runs/2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days \
    --batch=13_fast-last-layer-until-the-first-arrival_then-the-predictors-own-divisor_four-pointmaze-cases
```

## Contents

| section | first written | last changed | status |
|---|---|---|---|
| [Measures](#measures) | 2026-11-13 11:35 PT | 2026-11-13 11:35 PT | unread |
| [Validation seeds, case by case](#validation-seeds-case-by-case) | 2026-11-13 11:35 PT | 2026-11-13 11:35 PT | unread |
| [Arms at the same configuration, on the same seeds](#arms-at-the-same-configuration-on-the-same-seeds) | 2026-11-13 11:35 PT | 2026-11-13 11:35 PT | unread |
| [Predictions](#predictions) | 2026-11-13 11:35 PT | 2026-11-13 11:35 PT | unread |

Times are Pacific, converted from machines running Eastern, and come from this report's own history of regenerations. [The same report with unread sections in blue and changed text in brown](summary.html). No section marks itself read.

## Measures

Two metrics are reported, each in its own table with its own best configuration per method: the whole-curve mean $W$ measures discovery and optimization together, and the mean capped first-goal time $M$ measures discovery alone. The two are never mixed in one choice.

Define $K$ as the number of policies (seeds) of one configuration, $B$ as the budget in environment steps per policy (named with every table), $R_i(s)$ as the mean return of the training episodes of policy $i$ that end in the logging window containing step $s$, and $\Delta s$ as the number of that window's steps inside the range being averaged. Define $H$ as the episode step limit and $t$ as the step of an episode at which the goal is reached. With reward 1 on arrival an episode returns 1 if it reaches the goal and 0 otherwise; with the step cost it returns $-(t-1)/H$ if it reaches the goal and $-1$ otherwise. A window in which no episode ends takes the value of the window before it; before a policy's first finished episode the value is the return of an episode that never reaches the goal.

1. **Whole-curve mean:** $W(B) = \dfrac{1}{K}\sum_{i=1}^{K} \dfrac{1}{B}\sum_{s \le B} R_i(s) \thinspace \Delta s$, the area under the training curve divided by the budget.
2. **Final level:** $L(B) = \dfrac{1}{K}\sum_{i=1}^{K} \dfrac{1}{0.1 B}\sum_{0.9 B \lt s \le B} R_i(s) \thinspace \Delta s$, the same mean over the last 10% of the budget.
3. **Mean capped first-goal time:** define $T_i$ as the training step of policy $i$'s first arrival at the goal ($T_i = \infty$ if it has none by step $B$). $M(B) = \dfrac{1}{K}\sum_{i=1}^{K} \min(T_i, B)$. Smaller is better. The tables give $M$ in thousands of steps.
4. **Best configuration:** per method and case, for $W$ the configuration with the highest $W$ (ties: the higher $L$), for $M$ the one with the smallest $M$ (ties: more policies reach the goal). The two can differ.
5. **Difference from the baseline:** define $d_i$ as the value of seed $i$ under the method at its best configuration minus its value under the baseline at the baseline's own best configuration for the same metric. The tables give $\bar d = \dfrac{1}{K}\sum_{i=1}^{K} d_i$ over the seeds both have, with a 95% interval: the 2.5% and 97.5% points of the means of 10,000 resamples of those seeds, drawn with replacement. For $W$ and $L$ a method is ahead when the interval is entirely above zero; for $M$ it is ahead when the interval is entirely below zero.

Standard errors are over the $K$ policies.

## Validation seeds, case by case

A case is one environment with one reward setting. The validation seeds are neither the development seeds nor the confirmation seeds. They come in two blocks. On the first block each side stands at its best configuration among the few run there, so that block is a second selection step. Only complete units are read; nothing is estimated for the others.

### PointMaze Medium, reward 1 on arrival

#### First block: each side at its best configuration there

The units of the phases `validation-block` and `discovery-validation` in PointMaze Medium, reward 1 on arrival. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9349 ± 0.0031 | 0.9950 ± 0.0010 | 25 ± 1.46 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.01, entropy coefficient 0.01 | 0.9275 ± 0.0027 | 0.9892 ± 0.0015 | 23.5 ± 1.28 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9194 ± 0.0023 | 0.9794 ± 0.0023 | 19.9 ± 1.16 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 0.8718 ± 0.0034 | 0.9537 ± 0.0028 | 17.6 ± 0.96 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.01 | 0.6440 ± 0.0073 | 0.7891 ± 0.0078 | 18.3 ± 0.963 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 | 0.0073 ± 0.0003 | 0.0000 ± 0.0000 | 17.5 ± 1.41 | 256 of 256 |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | **0.9445** ± 0.0028 | **1.0000** ± 0.0000 | 16 ± 1.08 | 256 of 256 |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9416 ± 0.0026 | <u>0.9999</u> ± 0.0000 | **14.7** ± 0.807 | 256 of 256 |
| fast last layer until the first arrival, then the predictor's own divisor, left running | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9385 ± 0.0028 | 0.9959 ± 0.0006 | 15.8 ± 0.993 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | <u>0.9435</u> ± 0.0027 | <u>0.9999</u> ± 0.0000 | 18.5 ± 1.13 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.01, entropy coefficient 0.01 | 0.9412 ± 0.0024 | **1.0000** ± 0.0000 | 23.1 ± 1.34 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9334 ± 0.0029 | **1.0000** ± 0.0000 | 29.4 ± 1.85 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.8424 ± 0.0063 | 0.9747 ± 0.0037 | <u>15.7</u> ± 0.84 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 0.7759 ± 0.0077 | 0.9490 ± 0.0060 | **14.7** ± 0.957 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 6) | 0.9349 ± 0.0031 | control | 0.9950 ± 0.0010 | control | 256 |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 2) | **0.9445** ± 0.0028 | **+0.0095** [+0.0023, +0.0167] | **1.0000** ± 0.0000 | **+0.0050** [+0.0033, +0.0071] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | <u>0.9435</u> ± 0.0027 | <u>+0.0085</u> [+0.0009, +0.0161] | <u>0.9999</u> ± 0.0000 | <u>+0.0049</u> [+0.0033, +0.0070] | 256 |
| fast last layer until the first arrival, then the predictor's own divisor, left running | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (the only configuration read) | 0.9385 ± 0.0028 | +0.0036 [-0.0040, +0.0112] | 0.9959 ± 0.0006 | +0.0009 [-0.0011, +0.0032] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 2) | 0.8424 ± 0.0063 | -0.0925 [-0.1068, -0.0788] | 0.9747 ± 0.0037 | -0.0203 [-0.0285, -0.0134] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (best of 6) | 17.5 ± 1.41 | control | 256 of 256 | 0.0073 | 0.0000 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (best of 2) | **14.7** ± 0.957 | **-2.85** [-6.45, +0.268] | 256 of 256 | 0.7759 | 0.9490 | 256 |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 2) | **14.7** ± 0.807 | <u>-2.77</u> [-6.06, +0.0909] | 256 of 256 | <u>0.9416</u> | **0.9999** | 256 |
| fast last layer until the first arrival, then the predictor's own divisor, left running | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (the only configuration read) | <u>15.8</u> ± 0.993 | -1.75 [-5.12, +1.24] | 256 of 256 | 0.9385 | <u>0.9959</u> | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | 18.5 ± 1.13 | +0.993 [-2.74, +4.35] | 256 of 256 | **0.9435** | **0.9999** | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:14 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:15 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:14 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 5 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:53 PT | none |
| 6 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 7 | fast last layer until the first arrival, then the predictor's own divisor, held | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:52 PT | none |
| 8 | fast last layer until the first arrival, then the predictor's own divisor, held | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:48 PT | none |
| 9 | fast last layer until the first arrival, then the predictor's own divisor, left running | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:55 PT | none |
| 10 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:32 PT | none |
| 11 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:32 PT | none |
| 12 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:34 PT | none |
| 13 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 03:06 PT | none |
| 14 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 03:08 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.1_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-then-predictor-divisor-held_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-then-predictor-divisor-held_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-then-predictor-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
14. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

### PointMaze Medium, step cost

#### First block: each side at its best configuration there

The units of the phases `validation-block` and `discovery-validation` in PointMaze Medium, step cost. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01 | -0.3505 ± 0.0016 | -0.3078 ± 0.0007 | 20 ± 1.17 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.0078125, learning rate 0.006, entropy coefficient 0.01 | -0.3514 ± 0.0018 | -0.3074 ± 0.0011 | 20.9 ± 1.27 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.01, entropy coefficient 0.01 | -0.3549 ± 0.0018 | -0.3078 ± 0.0011 | 20.1 ± 1 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | -0.3680 ± 0.0015 | -0.3244 ± 0.0010 | 14.5 ± 0.63 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.01 | -0.4220 ± 0.0028 | -0.3519 ± 0.0025 | 13.9 ± 0.762 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 | -0.5526 ± 0.0109 | -0.6884 ± 0.0206 | **12.4** ± 0.471 | 256 of 256 |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | **-0.3476** ± 0.0013 | -0.3068 ± 0.0006 | <u>12.7</u> ± 0.612 | 256 of 256 |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 | -0.3489 ± 0.0022 | -0.3070 ± 0.0006 | 13.1 ± 0.66 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 | **-0.3476** ± 0.0014 | -0.3073 ± 0.0007 | 15.4 ± 0.693 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | <u>-0.3484</u> ± 0.0014 | **-0.3058** ± 0.0004 | 14.2 ± 0.617 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | -0.3504 ± 0.0015 | <u>-0.3066</u> ± 0.0005 | 13.4 ± 0.562 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3692 ± 0.0030 | -0.3132 ± 0.0008 | 12.9 ± 0.654 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | -0.4057 ± 0.0034 | -0.3299 ± 0.0015 | 15 ± 1.74 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01 (best of 6) | <u>-0.3505</u> ± 0.0016 | control | -0.3078 ± 0.0007 | control | 256 |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 2) | **-0.3476** ± 0.0013 | **+0.0029** [-0.0013, +0.0072] | **-0.3068** ± 0.0006 | **+0.0010** [-0.0009, +0.0028] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 (best of 3) | **-0.3476** ± 0.0014 | **+0.0029** [-0.0015, +0.0072] | <u>-0.3073</u> ± 0.0007 | <u>+0.0005</u> [-0.0014, +0.0025] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 2) | -0.3692 ± 0.0030 | <u>-0.0187</u> [-0.0258, -0.0120] | -0.3132 ± 0.0008 | -0.0054 [-0.0076, -0.0034] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (best of 6) | **12.4** ± 0.471 | control | 256 of 256 | -0.5526 | -0.6884 | 256 |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 2) | <u>12.7</u> ± 0.612 | **+0.265** [-1.13, +1.73] | 256 of 256 | **-0.3476** | <u>-0.3068</u> | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 2) | 12.9 ± 0.654 | <u>+0.523</u> [-0.938, +2.07] | 256 of 256 | -0.3692 | -0.3132 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | 13.4 ± 0.562 | +0.958 [-0.349, +2.26] | 256 of 256 | <u>-0.3504</u> | **-0.3066** | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:16 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:19 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:18 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:58 PT | none |
| 5 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:53 PT | none |
| 6 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:54 PT | none |
| 7 | fast last layer until the first arrival, then the predictor's own divisor, held | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:52 PT | none |
| 8 | fast last layer until the first arrival, then the predictor's own divisor, held | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:52 PT | none |
| 9 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:35 PT | none |
| 10 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:35 PT | none |
| 11 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:34 PT | none |
| 12 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:11 PT | none |
| 13 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:13 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.0078125_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-medium_reward-step-cost_fast-then-predictor-divisor-held_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-step-cost_fast-then-predictor-divisor-held_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-medium_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-medium_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

### PointMaze Large, reward 1 on arrival

#### First block: each side at its best configuration there

The units of the phases `validation-block` and `discovery-validation` in PointMaze Large, reward 1 on arrival. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 8,388,608 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8781 ± 0.0021 | 0.9920 ± 0.0005 | 269 ± 10.8 | **256 of 256** |
| control (CleanRL PPO-RND) | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 0.8759 ± 0.0022 | 0.9955 ± 0.0004 | 385 ± 13.1 | **256 of 256** |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.003, entropy coefficient 0.01 | 0.8701 ± 0.0023 | 0.9906 ± 0.0006 | 203 ± 8.06 | **256 of 256** |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.001, entropy coefficient 0.01 | 0.8695 ± 0.0029 | 0.9911 ± 0.0007 | 344 ± 17.8 | **256 of 256** |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.8626 ± 0.0025 | 0.9765 ± 0.0019 | 205 ± 8.2 | **256 of 256** |
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.01 | 0.0063 ± 0.0002 | 0.0015 ± 0.0003 | 135 ± 6.06 | **256 of 256** |
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 | 0.0054 ± 0.0002 | 0.0000 ± 0.0000 | 119 ± 5.46 | **256 of 256** |
| control (CleanRL PPO-RND) | intrinsic weight 256, learning rate 0.001, entropy coefficient 0.03 | 0.0006 ± 0.0000 | 0.0000 ± 0.0000 | 245 ± 45.9 | <u>254 of 256</u> |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.8806 ± 0.0029 | **0.9979** ± 0.0002 | 96.2 ± 6.95 | **256 of 256** |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8728 ± 0.0055 | 0.9945 ± 0.0028 | **85.2** ± 5.18 | **256 of 256** |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 0.8284 ± 0.0118 | 0.9591 ± 0.0121 | 136 ± 14.9 | **256 of 256** |
| fast last layer until the first arrival, then the predictor's own divisor, left running | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | **0.8894** ± 0.0022 | 0.9931 ± 0.0005 | 91.7 ± 5.75 | **256 of 256** |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | <u>0.8874</u> ± 0.0023 | <u>0.9977</u> ± 0.0002 | 259 ± 10.6 | **256 of 256** |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.8852 ± 0.0023 | 0.9976 ± 0.0003 | 201 ± 9.11 | **256 of 256** |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 0.8680 ± 0.0033 | **0.9979** ± 0.0003 | 375 ± 12.4 | **256 of 256** |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8707 ± 0.0058 | 0.9872 ± 0.0055 | 90.7 ± 5.08 | **256 of 256** |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 | 0.5725 ± 0.0106 | 0.8495 ± 0.0123 | <u>86.1</u> ± 6.37 | **256 of 256** |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 8) | 0.8781 ± 0.0021 | control | 0.9920 ± 0.0005 | control | 256 |
| fast last layer until the first arrival, then the predictor's own divisor, left running | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (the only configuration read) | **0.8894** ± 0.0022 | **+0.0114** [+0.0054, +0.0173] | 0.9931 ± 0.0005 | +0.0011 [-0.0003, +0.0024] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 3) | <u>0.8874</u> ± 0.0023 | <u>+0.0094</u> [+0.0032, +0.0156] | <u>0.9977</u> ± 0.0002 | <u>+0.0057</u> [+0.0046, +0.0068] | 256 |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (best of 3) | 0.8806 ± 0.0029 | +0.0025 [-0.0042, +0.0091] | **0.9979** ± 0.0002 | **+0.0058** [+0.0047, +0.0070] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 2) | 0.8707 ± 0.0058 | -0.0074 [-0.0203, +0.0037] | 0.9872 ± 0.0055 | -0.0048 [-0.0169, +0.0034] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (best of 8) | 119 ± 5.46 | control | 256 of 256 | 0.0054 | 0.0000 | 256 |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 3) | **85.2** ± 5.18 | **-34** [-48.8, -19] | 256 of 256 | 0.8728 | <u>0.9945</u> | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 (best of 2) | <u>86.1</u> ± 6.37 | <u>-33.2</u> [-49, -16.5] | 256 of 256 | 0.5725 | 0.8495 | 256 |
| fast last layer until the first arrival, then the predictor's own divisor, left running | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (the only configuration read) | 91.7 ± 5.75 | -27.5 [-43.1, -11.9] | 256 of 256 | **0.8894** | 0.9931 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (best of 3) | 201 ± 9.11 | +81.6 [+61.3, +102] | 256 of 256 | <u>0.8852</u> | **0.9976** | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:11 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:11 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:08 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 20:19 PT | none |
| 5 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 15:26 PT | none |
| 6 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 15:26 PT | none |
| 7 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 15:27 PT | none |
| 8 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 20:18 PT | none |
| 9 | fast last layer until the first arrival, then the predictor's own divisor, held | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:39 PT | none |
| 10 | fast last layer until the first arrival, then the predictor's own divisor, held | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:12 PT | none |
| 11 | fast last layer until the first arrival, then the predictor's own divisor, held | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:01 PT | none |
| 12 | fast last layer until the first arrival, then the predictor's own divisor, left running | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:31 PT | none |
| 13 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:55 PT | none |
| 14 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:55 PT | none |
| 15 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:07 PT | none |
| 16 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 03:44 PT | none |
| 17 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:46 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.001_entropy-coefficient-0.03_intrinsic-weight-256.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.001_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-then-predictor-divisor-held_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-then-predictor-divisor-held_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-then-predictor-divisor-held_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-then-predictor-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
14. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
15. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
16. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
17. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`

### PointMaze Large, step cost

#### First block: each side at its best configuration there

The units of the phases `validation-block` and `discovery-validation` in PointMaze Large, step cost. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 33,554,432 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 | -0.7310 ± 0.0018 | -0.7163 ± 0.0017 | 191 ± 10.1 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 | -0.7411 ± 0.0036 | -0.7229 ± 0.0033 | 241 ± 13.4 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.0045 | -0.8831 ± 0.0022 | -0.8849 ± 0.0026 | 131 ± 6.45 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006 | -0.9669 ± 0.0017 | -0.9803 ± 0.0022 | 120 ± 5.6 | 256 of 256 |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 8, learning rate 0.003 | <u>-0.7243</u> ± 0.0023 | -0.7088 ± 0.0022 | 96.6 ± 22.3 | 256 of 256 |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 8, learning rate 0.002 | -0.7292 ± 0.0030 | -0.7148 ± 0.0030 | 105 ± 33.4 | 256 of 256 |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 4, learning rate 0.002 | -0.7409 ± 0.0040 | -0.7277 ± 0.0041 | **69.4** ± 5.2 | 256 of 256 |
| fast last layer until the first arrival, then the predictor's own divisor, left running | intrinsic weight 8, learning rate 0.002 | -0.7386 ± 0.0011 | -0.7329 ± 0.0019 | <u>71.5</u> ± 4.37 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 | **-0.7236** ± 0.0021 | <u>-0.7079</u> ± 0.0021 | 163 ± 8.72 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.002 | -0.7268 ± 0.0025 | -0.7115 ± 0.0025 | 189 ± 9.38 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002 | -0.7335 ± 0.0034 | -0.7198 ± 0.0035 | 206 ± 11.5 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002 | -0.7264 ± 0.0016 | **-0.7031** ± 0.0018 | <u>71.5</u> ± 4.37 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.0045 | -0.7638 ± 0.0030 | -0.7281 ± 0.0032 | 78.6 ± 5.91 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 (best of 4) | -0.7310 ± 0.0018 | control | -0.7163 ± 0.0017 | control | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 (best of 3) | **-0.7236** ± 0.0021 | **+0.0074** [+0.0018, +0.0129] | <u>-0.7079</u> ± 0.0021 | <u>+0.0084</u> [+0.0029, +0.0137] | 256 |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 8, learning rate 0.003 (best of 3) | <u>-0.7243</u> ± 0.0023 | <u>+0.0067</u> [+0.0007, +0.0126] | -0.7088 ± 0.0022 | +0.0075 [+0.0018, +0.0130] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (best of 2) | -0.7264 ± 0.0016 | +0.0046 [-0.0003, +0.0094] | **-0.7031** ± 0.0018 | **+0.0132** [+0.0081, +0.0182] | 256 |
| fast last layer until the first arrival, then the predictor's own divisor, left running | intrinsic weight 8, learning rate 0.002 (the only configuration read) | -0.7386 ± 0.0011 | -0.0076 [-0.0117, -0.0032] | -0.7329 ± 0.0019 | -0.0165 [-0.0213, -0.0115] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006 (best of 4) | 120 ± 5.6 | control | 256 of 256 | -0.9669 | -0.9803 | 256 |
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 4, learning rate 0.002 (best of 3) | **69.4** ± 5.2 | **-50.8** [-65.6, -35.1] | 256 of 256 | -0.7409 | -0.7277 | 256 |
| fast last layer until the first arrival, then the predictor's own divisor, left running | intrinsic weight 8, learning rate 0.002 (the only configuration read) | <u>71.5</u> ± 4.37 | <u>-48.8</u> [-63, -34.6] | 256 of 256 | -0.7386 | -0.7329 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (best of 2) | <u>71.5</u> ± 4.37 | <u>-48.8</u> [-62.6, -34.3] | 256 of 256 | <u>-0.7264</u> | **-0.7031** | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 (best of 3) | 163 ± 8.72 | +42.9 [+24.2, +62.7] | 256 of 256 | **-0.7236** | <u>-0.7079</u> | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:48 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:47 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:25 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:03 PT | none |
| 5 | fast last layer until the first arrival, then the predictor's own divisor, held | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 06:42 PT | none |
| 6 | fast last layer until the first arrival, then the predictor's own divisor, held | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 07:26 PT | none |
| 7 | fast last layer until the first arrival, then the predictor's own divisor, held | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 08:06 PT | 1 continued from a checkpoint |
| 8 | fast last layer until the first arrival, then the predictor's own divisor, left running | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 08:21 PT | 1 continued from a checkpoint |
| 9 | divisor held from each policy's first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:32 PT | 1 continued from a checkpoint |
| 10 | divisor held from each policy's first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:04 PT | none |
| 11 | divisor held from each policy's first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:05 PT | none |
| 12 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:16 PT | 1 continued from a checkpoint |
| 13 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:30 PT | 1 continued from a checkpoint |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.0045_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `validation-block_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-large_reward-step-cost_fast-then-predictor-divisor-held_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-large_reward-step-cost_fast-then-predictor-divisor-held_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-large_reward-step-cost_fast-then-predictor-divisor-held_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-large_reward-step-cost_fast-then-predictor-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-large_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-large_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-large_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-large_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-large_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.0045_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`

## Arms at the same configuration, on the same seeds

Each group compares one arm with other sides at the configurations both trained, matched on the intrinsic weight, learning rate, entropy coefficient and clip coefficient, and paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Define $W$, $L$ and $M$ as in the section Measures. No configuration is chosen here.

### Fast last layer until the first arrival, then the predictor's own divisor, held

#### PointMaze Medium, reward 1 on arrival

The arm: fast last layer until the first arrival, then the predictor's own divisor, held, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9445 ± 0.0028 | 0.9334 ± 0.0029 | +0.0110 [+0.0038, +0.0182] | +0.0000 [-0.0000, +0.0000] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9416 ± 0.0026 | 0.9435 ± 0.0027 | -0.0019 [-0.0093, +0.0054] | -0.0000 [-0.0001, +0.0001] | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9416 ± 0.0026 | 0.7001 ± 0.0055 | +0.2414 [+0.2299, +0.2530] | +0.0585 [+0.0521, +0.0651] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9416 ± 0.0026 | 0.8424 ± 0.0063 | +0.0991 [+0.0871, +0.1122] | +0.0252 [+0.0186, +0.0333] | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9445 ± 0.0028 | 0.9349 ± 0.0031 | +0.0095 [+0.0024, +0.0166] | +0.0050 [+0.0034, +0.0071] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9416 ± 0.0026 | 0.9194 ± 0.0023 | +0.0221 [+0.0155, +0.0286] | +0.0205 [+0.0163, +0.0254] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 16 ± 1.08 | 29.4 ± 1.85 | -13.4 [-17.6, -9.39] | 256 of 256; 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 14.7 ± 0.807 | 18.5 ± 1.13 | -3.76 [-6.48, -1.15] | 256 of 256; 256 of 256 | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 14.7 ± 0.807 | 14.6 ± 0.737 | +0.13 [-1.62, +1.91] | 256 of 256; 256 of 256 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 14.7 ± 0.807 | 15.7 ± 0.84 | -0.937 [-2.95, +1.07] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 16 ± 1.08 | 25 ± 1.46 | -9.02 [-12.5, -5.57] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 14.7 ± 0.807 | 19.9 ± 1.16 | -5.16 [-7.72, -2.68] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 1, level at 1, behind at 0 of 2; $M$ sooner at 2, level at 0, later at 0 of 2.
- against fast last layer until the first arrival, then the first-arrival hold: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1; not trained by it: intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01.
- against fast last layer (step 0.1) with the first-arrival hold: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1; not trained by it: intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01.
- against CleanRL PPO-RND: $W$ ahead at 2, level at 0, behind at 0 of 2; $M$ sooner at 2, level at 0, later at 0 of 2.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer until the first arrival, then the predictor's own divisor, held: best $W$ 0.9445 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 14.7 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 2 configurations).
- divisor held from each policy's first arrival: best $W$ 0.9435 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 18.5 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 3 configurations).
- fast last layer until the first arrival, then the first-arrival hold: best $W$ 0.7001 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 14 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (of 2 configurations).
- fast last layer (step 0.1) with the first-arrival hold: best $W$ 0.8424 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 14.7 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ 0.9349 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 17.5 thousand steps at intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (of 6 configurations).

#### PointMaze Medium, step cost

The arm: fast last layer until the first arrival, then the predictor's own divisor, held, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 | -0.3489 ± 0.0022 | -0.3476 ± 0.0014 | -0.0013 [-0.0068, +0.0036] | +0.0004 [-0.0015, +0.0022] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3476 ± 0.0013 | -0.3484 ± 0.0014 | +0.0008 [-0.0030, +0.0045] | -0.0010 [-0.0025, +0.0003] | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3476 ± 0.0013 | -0.3821 ± 0.0018 | +0.0345 [+0.0303, +0.0386] | +0.0084 [+0.0068, +0.0100] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3476 ± 0.0013 | -0.3692 ± 0.0030 | +0.0216 [+0.0158, +0.0278] | +0.0064 [+0.0046, +0.0084] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 | 13.1 ± 0.66 | 15.4 ± 0.693 | -2.29 [-3.92, -0.636] | 256 of 256; 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 12.7 ± 0.612 | 14.2 ± 0.617 | -1.58 [-3.14, -0.0148] | 256 of 256; 256 of 256 | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 12.7 ± 0.612 | 12.7 ± 0.593 | -0.00111 [-1.5, +1.47] | 256 of 256; 256 of 256 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 12.7 ± 0.612 | 12.9 ± 0.654 | -0.258 [-1.77, +1.21] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 2, behind at 0 of 2; $M$ sooner at 2, level at 0, later at 0 of 2.
- against fast last layer until the first arrival, then the first-arrival hold: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1; not trained by it: intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01.
- against fast last layer (step 0.1) with the first-arrival hold: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1; not trained by it: intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 0 of 0; $M$ sooner at 0, level at 0, later at 0 of 0; not trained by it: intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01; intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer until the first arrival, then the predictor's own divisor, held: best $W$ -0.3476 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 12.7 thousand steps at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (of 2 configurations).
- divisor held from each policy's first arrival: best $W$ -0.3476 at intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01; best $M$ 13.4 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 3 configurations).
- fast last layer until the first arrival, then the first-arrival hold: best $W$ -0.3821 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 12.7 thousand steps at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (of 2 configurations).
- fast last layer (step 0.1) with the first-arrival hold: best $W$ -0.3692 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 12.9 thousand steps at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ -0.3505 at intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01; best $M$ 12.4 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (of 6 configurations).

#### PointMaze Large, reward 1 on arrival

The arm: fast last layer until the first arrival, then the predictor's own divisor, held, on 256 seeds from 10000 to 10255, read at 8,388,608 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 0.8284 ± 0.0118 | 0.8680 ± 0.0033 | -0.0396 [-0.0645, -0.0165] | -0.0388 [-0.0628, -0.0157] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8728 ± 0.0055 | 0.8874 ± 0.0023 | -0.0147 [-0.0272, -0.0036] | -0.0031 [-0.0094, +0.0005] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.8806 ± 0.0029 | 0.8852 ± 0.0023 | -0.0046 [-0.0119, +0.0028] | +0.0002 [-0.0005, +0.0010] | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8728 ± 0.0055 | 0.8496 ± 0.0036 | +0.0231 [+0.0104, +0.0350] | +0.0027 [-0.0039, +0.0073] | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.8806 ± 0.0029 | 0.7871 ± 0.0045 | +0.0935 [+0.0829, +0.1044] | +0.0208 [+0.0169, +0.0252] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8728 ± 0.0055 | 0.8707 ± 0.0058 | +0.0021 [-0.0127, +0.0169] | +0.0074 [-0.0033, +0.0210] | 256 |
| CleanRL PPO-RND | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 0.8284 ± 0.0118 | 0.8759 ± 0.0022 | -0.0475 [-0.0722, -0.0254] | -0.0364 [-0.0606, -0.0134] | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8728 ± 0.0055 | 0.8781 ± 0.0021 | -0.0053 [-0.0176, +0.0055] | +0.0025 [-0.0039, +0.0064] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.8806 ± 0.0029 | 0.8626 ± 0.0025 | +0.0180 [+0.0109, +0.0252] | +0.0213 [+0.0177, +0.0254] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 136 ± 14.9 | 375 ± 12.4 | -239 [-275, -199] | 256 of 256; 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 85.2 ± 5.18 | 259 ± 10.6 | -174 [-197, -151] | 256 of 256; 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 96.2 ± 6.95 | 201 ± 9.11 | -105 [-126, -83.3] | 256 of 256; 256 of 256 | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 85.2 ± 5.18 | 106 ± 11.2 | -20.8 [-46.5, +0.63] | 256 of 256; 256 of 256 | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 96.2 ± 6.95 | 82.9 ± 4.46 | +13.3 [-2.48, +29.8] | 256 of 256; 256 of 256 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 85.2 ± 5.18 | 90.7 ± 5.08 | -5.53 [-18.7, +7.81] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 136 ± 14.9 | 385 ± 13.1 | -249 [-283, -210] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 85.2 ± 5.18 | 269 ± 10.8 | -184 [-208, -161] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 96.2 ± 6.95 | 205 ± 8.2 | -109 [-130, -87.1] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 1, behind at 2 of 3; $M$ sooner at 3, level at 0, later at 0 of 3.
- against fast last layer until the first arrival, then the first-arrival hold: $W$ ahead at 2, level at 0, behind at 0 of 2; $M$ sooner at 0, level at 2, later at 0 of 2; not trained by it: intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01.
- against fast last layer (step 0.1) with the first-arrival hold: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1; not trained by it: intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01; intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01.
- against CleanRL PPO-RND: $W$ ahead at 1, level at 1, behind at 1 of 3; $M$ sooner at 3, level at 0, later at 0 of 3.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer until the first arrival, then the predictor's own divisor, held: best $W$ 0.8806 at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01; best $M$ 85.2 thousand steps at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).
- divisor held from each policy's first arrival: best $W$ 0.8874 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 201 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).
- fast last layer until the first arrival, then the first-arrival hold: best $W$ 0.8496 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 82.9 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).
- fast last layer (step 0.1) with the first-arrival hold: best $W$ 0.8707 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 86.1 thousand steps at intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ 0.8781 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 119 thousand steps at intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (of 8 configurations).

#### PointMaze Large, step cost

The arm: fast last layer until the first arrival, then the predictor's own divisor, held, on 256 seeds from 10000 to 10255, read at 33,554,432 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | -0.7409 ± 0.0040 | -0.7335 ± 0.0034 | -0.0074 [-0.0170, +0.0023] | -0.0079 [-0.0180, +0.0020] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | -0.7292 ± 0.0030 | -0.7268 ± 0.0025 | -0.0024 [-0.0106, +0.0055] | -0.0032 [-0.0111, +0.0050] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01 | -0.7243 ± 0.0023 | -0.7236 ± 0.0021 | -0.0007 [-0.0069, +0.0055] | -0.0009 [-0.0072, +0.0052] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | -0.7292 ± 0.0030 | -0.7264 ± 0.0016 | -0.0028 [-0.0098, +0.0035] | -0.0116 [-0.0189, -0.0048] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | -0.7409 ± 0.0040 | -0.7310 ± 0.0018 | -0.0099 [-0.0185, -0.0015] | -0.0114 [-0.0200, -0.0031] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 69.4 ± 5.2 | 206 ± 11.5 | -137 [-161, -114] | 256 of 256; 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | 105 ± 33.4 | 189 ± 9.38 | -83.5 [-133, -5.44] | 256 of 256; 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01 | 96.6 ± 22.3 | 163 ± 8.72 | -66.5 [-102, -16.1] | 256 of 256; 256 of 256 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | 105 ± 33.4 | 71.5 ± 4.37 | +33.8 [-8.8, +107] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 69.4 ± 5.2 | 191 ± 10.1 | -122 [-144, -100] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 3, behind at 0 of 3; $M$ sooner at 3, level at 0, later at 0 of 3.
- against fast last layer (step 0.1) with the first-arrival hold: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1; not trained by it: intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01; intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 1 of 1; $M$ sooner at 1, level at 0, later at 0 of 1; not trained by it: intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01; intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer until the first arrival, then the predictor's own divisor, held: best $W$ -0.7243 at intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01; best $M$ 69.4 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).
- divisor held from each policy's first arrival: best $W$ -0.7236 at intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01; best $M$ 163 thousand steps at intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01 (of 3 configurations).
- fast last layer (step 0.1) with the first-arrival hold: best $W$ -0.7264 at intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01; best $M$ 71.5 thousand steps at intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ -0.7310 at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01; best $M$ 120 thousand steps at intrinsic weight 8, learning rate 0.006, entropy coefficient 0.01 (of 4 configurations).

### The predictor's own divisor left running

#### PointMaze Medium, reward 1 on arrival

The arm: fast last layer until the first arrival, then the predictor's own divisor, left running, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9385 ± 0.0028 | 0.9445 ± 0.0028 | -0.0059 [-0.0129, +0.0009] | -0.0041 [-0.0054, -0.0030] | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9385 ± 0.0028 | 0.9349 ± 0.0031 | +0.0036 [-0.0041, +0.0113] | +0.0009 [-0.0012, +0.0032] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 15.8 ± 0.993 | 16 ± 1.08 | -0.24 [-2.98, +2.35] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 15.8 ± 0.993 | 25 ± 1.46 | -9.26 [-12.6, -5.94] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against fast last layer until the first arrival, then the predictor's own divisor, held: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 1, level at 0, later at 0 of 1.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer until the first arrival, then the predictor's own divisor, left running: best $W$ 0.9385 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 15.8 thousand steps at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (of 1 configuration).
- fast last layer until the first arrival, then the predictor's own divisor, held: best $W$ 0.9445 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 14.7 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ 0.9349 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 17.5 thousand steps at intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (of 6 configurations).

#### PointMaze Large, reward 1 on arrival

The arm: fast last layer until the first arrival, then the predictor's own divisor, left running, on 256 seeds from 10000 to 10255, read at 8,388,608 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8894 ± 0.0022 | 0.8728 ± 0.0055 | +0.0167 [+0.0057, +0.0291] | -0.0015 [-0.0053, +0.0051] | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8894 ± 0.0022 | 0.8781 ± 0.0021 | +0.0114 [+0.0054, +0.0175] | +0.0011 [-0.0004, +0.0024] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 91.7 ± 5.75 | 85.2 ± 5.18 | +6.51 [-8.75, +21.7] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 91.7 ± 5.75 | 269 ± 10.8 | -178 [-200, -155] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against fast last layer until the first arrival, then the predictor's own divisor, held: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against CleanRL PPO-RND: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 1, level at 0, later at 0 of 1.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer until the first arrival, then the predictor's own divisor, left running: best $W$ 0.8894 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 91.7 thousand steps at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (of 1 configuration).
- fast last layer until the first arrival, then the predictor's own divisor, held: best $W$ 0.8806 at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01; best $M$ 85.2 thousand steps at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).
- CleanRL PPO-RND: best $W$ 0.8781 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 119 thousand steps at intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (of 8 configurations).

#### PointMaze Large, step cost

The arm: fast last layer until the first arrival, then the predictor's own divisor, left running, on 256 seeds from 10000 to 10255, read at 33,554,432 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | -0.7386 ± 0.0011 | -0.7292 ± 0.0030 | -0.0094 [-0.0157, -0.0029] | -0.0181 [-0.0252, -0.0106] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| fast last layer until the first arrival, then the predictor's own divisor, held | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | 71.5 ± 4.37 | 105 ± 33.4 | -33.8 [-108, +9.35] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against fast last layer until the first arrival, then the predictor's own divisor, held: $W$ ahead at 0, level at 0, behind at 1 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 0 of 0; $M$ sooner at 0, level at 0, later at 0 of 0; not trained by it: intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer until the first arrival, then the predictor's own divisor, left running: best $W$ -0.7386 at intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01; best $M$ 71.5 thousand steps at intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 (of 1 configuration).
- fast last layer until the first arrival, then the predictor's own divisor, held: best $W$ -0.7243 at intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01; best $M$ 69.4 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).
- CleanRL PPO-RND: best $W$ -0.7310 at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01; best $M$ 120 thousand steps at intrinsic weight 8, learning rate 0.006, entropy coefficient 0.01 (of 4 configurations).

## Predictions

The predictions were written in `ideas.md` before the runs. Each is quoted, then read part by part: a part held or did not hold on the numbers given, or cannot be read yet because a unit it needs is not complete. A prediction held when every part held, and did not hold when at least one part did not. Where a prediction leaves a choice, it is read as follows. Every reading is on the first validation block (seeds 10000 to 10255), paired by seed. "The first-goal time as the fast last layer's at the same configuration" is read against an arm that runs the same program until the first arrival, at the same configuration: batch 11's arm (the fast last layer until the first arrival, then the hold) where it was trained there, else batch 10's arm (the fast last layer throughout, with the hold); it holds when the 95% interval of the paired difference in $M$ holds zero. "Within 0.01" is read on the mean difference, at every configuration the arm and the hold alone both trained (final level, prediction 2) or at each side's best (whole-curve mean, prediction 5). "Ahead with the interval above zero" is read on the interval; "ahead of or level with" and "not behind" as: the interval is not entirely below zero; "the higher final level" on the mean difference. "RND's best" is RND at its best configuration for $W$ among all those run on the same seeds.

### Prediction 1: the first-goal time as the fast last layer's at the same configuration

> **First-goal time as the fast last layer's at the same configuration** (the same program until the arrival): about 15 to 17 thousand steps on Medium with reward 1 on arrival, about 100 on Large with reward 1 on arrival at weight 2 and about 85 at weight 4, about 70 on Large with the step cost at (2e-3, 0.01, 8).

Observed: **held**.

- Medium with reward 1 on arrival (6e-3, weight 4), against batch 11's arm: **held**. At (learning rate 0.006, weight 4): $M$ 14.7 thousand steps against 14.6 thousand steps; difference +0.13 [-1.62, +1.91] thousand steps.
- Large with reward 1 on arrival (2e-3, weight 2), against batch 11's arm: **held**. At (learning rate 0.002, weight 2): $M$ 85.2 thousand steps against 106 thousand steps; difference -20.8 [-46.5, +0.63] thousand steps.
- Large with reward 1 on arrival (2e-3, weight 4), against batch 11's arm: **held**. At (learning rate 0.002, weight 4): $M$ 96.2 thousand steps against 82.9 thousand steps; difference +13.3 [-2.48, +29.8] thousand steps.
- Large with the step cost (2e-3, weight 8), against batch 10's arm: **held**. At (learning rate 0.002, weight 8): $M$ 105 thousand steps against 71.5 thousand steps; difference +33.8 [-8.8, +107] thousand steps.

### Prediction 2: the failure of batch 11 is gone at full length

> **The failure of batch 11 is gone at full length:** in every case the held arm's final level is within 0.01 of the hold alone's at the same configuration.

Observed: **did not hold**.

- Medium with reward 1 on arrival: the final level within 0.01 of the hold alone's at every common configuration: **held**. At (learning rate 0.006, weight 2): $L$ 1.0000 against 1.0000; difference +0.0000 [-0.0000, +0.0000]. At (learning rate 0.006, weight 4): $L$ 0.9999 against 0.9999; difference -0.0000 [-0.0001, +0.0001].
- Medium with the step cost: the final level within 0.01 of the hold alone's at every common configuration: **held**. At (learning rate 0.006, weight 1): $L$ -0.3070 against -0.3073; difference +0.0004 [-0.0015, +0.0022]. At (learning rate 0.006, weight 2): $L$ -0.3068 against -0.3058; difference -0.0010 [-0.0025, +0.0003].
- Large with reward 1 on arrival: the final level within 0.01 of the hold alone's at every common configuration: **did not hold**. At (learning rate 0.002, weight 1): $L$ 0.9591 against 0.9979; difference -0.0388 [-0.0628, -0.0157]. At (learning rate 0.002, weight 2): $L$ 0.9945 against 0.9977; difference -0.0031 [-0.0094, +0.0005]. At (learning rate 0.002, weight 4): $L$ 0.9979 against 0.9976; difference +0.0002 [-0.0005, +0.0010].
- Large with the step cost: the final level within 0.01 of the hold alone's at every common configuration: **held**. At (learning rate 0.002, weight 4): $L$ -0.7277 against -0.7198; difference -0.0079 [-0.0180, +0.0020]. At (learning rate 0.002, weight 8): $L$ -0.7148 against -0.7115; difference -0.0032 [-0.0111, +0.0050]. At (learning rate 0.003, weight 8): $L$ -0.7088 against -0.7079; difference -0.0009 [-0.0072, +0.0052].

### Prediction 3: PointMaze Large with reward 1 on arrival

> **PointMaze Large with reward 1 on arrival:** the held arm is ahead of RND's best (0.8781) with the interval above zero at weight 1 or 2, by about 0.01, because the first arrival comes about 2% of the budget sooner. I am unsure: after the arrival the arm at weight w settles like RND at a weight between w and 2w.

Observed: **did not hold**.

- Ahead of RND's best with the interval above zero at weight 1 or 2: **did not hold**. At (learning rate 0.002, weight 1): $W$ 0.8284, difference from RND's best -0.0496 [-0.0735, -0.0276]. At (learning rate 0.002, weight 2): $W$ 0.8728, difference from RND's best -0.0053 [-0.0171, +0.0057].

### Prediction 4: PointMaze Large with the step cost

> **PointMaze Large with the step cost:** the held arm is ahead of RND's best (−0.7310) with the interval above zero at one of its three configurations, and not behind batch 10's arm (−0.7264).

Observed: **held**.

- Ahead of RND's best with the interval above zero at one of its three configurations: **held**. At (learning rate 0.002, weight 4): $W$ -0.7409, difference from RND's best -0.0099 [-0.0184, -0.0017]. At (learning rate 0.002, weight 8): $W$ -0.7292, difference from RND's best +0.0018 [-0.0053, +0.0087]. At (learning rate 0.003, weight 8): $W$ -0.7243, difference from RND's best +0.0067 [+0.0010, +0.0124].
- Not behind batch 10's arm (each at its best): **held**. $W$ -0.7243 against -0.7264; difference +0.0021 [-0.0035, +0.0075].

### Prediction 5: PointMaze Medium

> **PointMaze Medium:** within 0.01 of the hold alone on the whole-curve mean in both cases, so ahead of or level with RND's best; the first arrival is only a few thousand steps sooner there.

Observed: **held**.

- Medium with reward 1 on arrival: within 0.01 of the hold alone on $W$ (each at its best): **held**. $W$ 0.9445 against 0.9435; difference +0.0010 [-0.0059, +0.0078].
- Medium with reward 1 on arrival: ahead of or level with RND's best: **held**. $W$ 0.9445; difference from RND's best +0.0095 [+0.0023, +0.0167].
- Medium with the step cost: within 0.01 of the hold alone on $W$ (each at its best): **held**. $W$ -0.3476 against -0.3476; difference +0.0001 [-0.0038, +0.0039].
- Medium with the step cost: ahead of or level with RND's best: **held**. $W$ -0.3476; difference from RND's best +0.0029 [-0.0013, +0.0072].

### Prediction 6: running against held

> **Running against held:** the held arm has the higher final level in all three cases where both run.

Observed: **held**.

- Medium with reward 1 on arrival (6e-3, weight 2): the held arm has the higher final level: **held**. At (learning rate 0.006, weight 2): $L$ 0.9959 against 1.0000; difference -0.0041 [-0.0054, -0.0030].
- Large with reward 1 on arrival (2e-3, weight 2): the held arm has the higher final level: **held**. At (learning rate 0.002, weight 2): $L$ 0.9931 against 0.9945; difference -0.0015 [-0.0053, +0.0051].
- Large with the step cost (2e-3, weight 8): the held arm has the higher final level: **held**. At (learning rate 0.002, weight 8): $L$ -0.7329 against -0.7148; difference -0.0181 [-0.0252, -0.0106].

