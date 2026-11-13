# Batch 14: the smaller steps of the fast last layer with the hold on the ant U-maze, and the second block on PointMaze Medium

Generated 2026-11-13 11:34 PT by the command below. This file is generated: regenerate it, do not edit it. Times are Pacific, converted from machines running Eastern. Run folder: `2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days`; batch folder: `batches/14_smaller-steps-of-the-fast-last-layer-with-the-hold_ant-umaze-screen_and-medium-second-block`.

```bash
cd /p/rlprojects/RND/10_jax_exploration_platform
bash "benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh" -s scripts/maze_campaign/analysis/batch_summary_job.py \
    --run-dir=runs/2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days \
    --batch=14_smaller-steps-of-the-fast-last-layer-with-the-hold_ant-umaze-screen_and-medium-second-block
```

## Contents

| section | first written | last changed | status |
|---|---|---|---|
| [Measures](#measures) | 2026-11-13 11:34 PT | 2026-11-13 11:34 PT | unread |
| [Validation seeds, case by case](#validation-seeds-case-by-case) | 2026-11-13 11:34 PT | 2026-11-13 11:34 PT | unread |
| [Arms at the same configuration, on the same seeds](#arms-at-the-same-configuration-on-the-same-seeds) | 2026-11-13 11:34 PT | 2026-11-13 11:34 PT | unread |
| [Predictions](#predictions) | 2026-11-13 11:34 PT | 2026-11-13 11:34 PT | unread |

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

A case is one environment with one reward setting. The validation seeds are neither the development seeds nor the confirmation seeds. They come in two blocks. On the first block each side stands at its best configuration among the few run there, so that block is a second selection step. On the second block nothing was chosen: a method stands there at the configuration fixed on the first block, and is compared with the control under equal rules and with the control favoured. Only complete units are read; nothing is estimated for the others.

### PointMaze Medium, reward 1 on arrival

#### First block: each side at its best configuration there

The units of the phases `validation-block` and `discovery-validation` in PointMaze Medium, reward 1 on arrival. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9349 ± 0.0031 | 0.9950 ± 0.0010 | 25 ± 1.46 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.01, entropy coefficient 0.01 | 0.9275 ± 0.0027 | 0.9892 ± 0.0015 | 23.5 ± 1.28 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9194 ± 0.0023 | 0.9794 ± 0.0023 | 19.9 ± 1.16 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 0.8718 ± 0.0034 | 0.9537 ± 0.0028 | <u>17.6</u> ± 0.96 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.01 | 0.6440 ± 0.0073 | 0.7891 ± 0.0078 | 18.3 ± 0.963 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 | 0.0073 ± 0.0003 | 0.0000 ± 0.0000 | **17.5** ± 1.41 | 256 of 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9400 ± 0.0031 | 0.9993 ± 0.0002 | 18.5 ± 1 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | **0.9435** ± 0.0027 | <u>0.9999</u> ± 0.0000 | 18.5 ± 1.13 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.01, entropy coefficient 0.01 | <u>0.9412</u> ± 0.0024 | **1.0000** ± 0.0000 | 23.1 ± 1.34 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9334 ± 0.0029 | **1.0000** ± 0.0000 | 29.4 ± 1.85 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 6) | 0.9349 ± 0.0031 | control | 0.9950 ± 0.0010 | control | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | **0.9435** ± 0.0027 | **+0.0085** [+0.0009, +0.0161] | **0.9999** ± 0.0000 | **+0.0049** [+0.0033, +0.0070] | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (the only configuration read) | <u>0.9400</u> ± 0.0031 | <u>+0.0051</u> [-0.0028, +0.0128] | <u>0.9993</u> ± 0.0002 | <u>+0.0043</u> [+0.0026, +0.0065] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (best of 6) | **17.5** ± 1.41 | control | 256 of 256 | 0.0073 | 0.0000 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | <u>18.5</u> ± 1.13 | **+0.993** [-2.74, +4.35] | 256 of 256 | **0.9435** | **0.9999** | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (the only configuration read) | <u>18.5</u> ± 1 | <u>+0.995</u> [-2.46, +4.1] | 256 of 256 | <u>0.9400</u> | <u>0.9993</u> | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:14 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:15 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:14 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 5 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:53 PT | none |
| 6 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 7 | fast last layer at step 0.01 with the first-arrival hold | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:15 PT | none |
| 8 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:32 PT | none |
| 9 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:32 PT | none |
| 10 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:34 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.1_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-step-0-point-01-with-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phases `validation-block-two` and `discovery-block-two` in PointMaze Medium, reward 1 on arrival. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9308 ± 0.0030 | 0.9907 ± 0.0039 | 26.3 ± 1.88 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.01, entropy coefficient 0.01 | 0.9276 ± 0.0026 | 0.9798 ± 0.0059 | 23.6 ± 1.4 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9141 ± 0.0030 | 0.9821 ± 0.0018 | 19.6 ± 1.14 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 | 0.0075 ± 0.0003 | 0.0000 ± 0.0000 | **16.8** ± 1.43 | 256 of 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | <u>0.9403</u> ± 0.0025 | <u>0.9995</u> ± 0.0001 | <u>17.8</u> ± 0.934 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | **0.9404** ± 0.0030 | **0.9999** ± 0.0000 | 20.6 ± 1.12 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (its best on the first block and on this block) | 0.9308 ± 0.0030 | control | control | 0.9907 ± 0.0039 | control | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | **0.9404** ± 0.0030 | +0.0095 [+0.0013, +0.0177] | +0.0095 [+0.0013, +0.0177] | **0.9999** ± 0.0000 | **+0.0092** [+0.0042, +0.0180] | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | <u>0.9403</u> ± 0.0025 | +0.0095 [+0.0020, +0.0171] | +0.0095 [+0.0020, +0.0171] | <u>0.9995</u> ± 0.0001 | <u>+0.0088</u> [+0.0039, +0.0174] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (its best on the first block and on this block) | **16.8** ± 1.43 | control | control | 256 of 256 | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | <u>17.8</u> ± 0.934 | **+1.01** [-2.49, +4.05] | **+1.01** [-2.49, +4.05] | 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | 20.6 ± 1.12 | <u>+3.83</u> [+0.451, +7.02] | <u>+3.83</u> [+0.451, +7.02] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 10:58 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:16 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:19 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:16 PT | none |
| 5 | fast last layer at step 0.01 with the first-arrival hold | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:33 PT | none |
| 6 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:31 PT | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.1_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
5. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_fast-step-0-point-01-with-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
6. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`

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
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | **-0.3470** ± 0.0013 | <u>-0.3066</u> ± 0.0006 | 13.8 ± 0.588 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 | <u>-0.3476</u> ± 0.0014 | -0.3073 ± 0.0007 | 15.4 ± 0.693 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3484 ± 0.0014 | **-0.3058** ± 0.0004 | 14.2 ± 0.617 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | -0.3504 ± 0.0015 | <u>-0.3066</u> ± 0.0005 | <u>13.4</u> ± 0.562 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01 (best of 6) | -0.3505 ± 0.0016 | control | -0.3078 ± 0.0007 | control | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (the only configuration read) | **-0.3470** ± 0.0013 | **+0.0035** [-0.0006, +0.0077] | **-0.3066** ± 0.0006 | **+0.0012** [-0.0005, +0.0029] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 (best of 3) | <u>-0.3476</u> ± 0.0014 | <u>+0.0029</u> [-0.0015, +0.0072] | <u>-0.3073</u> ± 0.0007 | <u>+0.0005</u> [-0.0014, +0.0025] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (best of 6) | **12.4** ± 0.471 | control | 256 of 256 | -0.5526 | <u>-0.6884</u> | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | <u>13.4</u> ± 0.562 | **+0.958** [-0.349, +2.26] | 256 of 256 | <u>-0.3504</u> | **-0.3066** | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (the only configuration read) | 13.8 ± 0.588 | <u>+1.4</u> [+0.0507, +2.77] | 256 of 256 | **-0.3470** | **-0.3066** | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:16 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:19 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:18 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:58 PT | none |
| 5 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:53 PT | none |
| 6 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:54 PT | none |
| 7 | fast last layer at step 0.01 with the first-arrival hold | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:19 PT | none |
| 8 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:35 PT | none |
| 9 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:35 PT | none |
| 10 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:34 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.0078125_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-medium_reward-step-cost_fast-step-0-point-01-with-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phases `validation-block-two` and `discovery-block-two` in PointMaze Medium, step cost. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.0078125, learning rate 0.006, entropy coefficient 0.01 | -0.3510 ± 0.0016 | -0.3078 ± 0.0007 | 20.4 ± 0.931 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01 | -0.3535 ± 0.0024 | -0.3082 ± 0.0012 | 19.9 ± 1.01 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.01, entropy coefficient 0.01 | -0.3541 ± 0.0014 | -0.3073 ± 0.0006 | 20.6 ± 0.841 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 | -0.5581 ± 0.0108 | -0.7035 ± 0.0206 | **13.2** ± 0.497 | 256 of 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3494 ± 0.0018 | <u>-0.3068</u> ± 0.0005 | <u>13.5</u> ± 0.628 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 | **-0.3470** ± 0.0015 | -0.3069 ± 0.0006 | 16.9 ± 0.742 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | <u>-0.3472</u> ± 0.0015 | **-0.3059** ± 0.0005 | 14.7 ± 0.632 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND), at its first-block best | intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | -0.3535 ± 0.0024 | control | -0.0025 [-0.0081, +0.0025] | -0.3082 ± 0.0012 | control | 256 |
| control (CleanRL PPO-RND), at its best on this block | intrinsic weight 0.0078125, learning rate 0.006, entropy coefficient 0.01 (best of 4 on this block) | -0.3510 ± 0.0016 | +0.0025 [-0.0025, +0.0082] | control | -0.3078 ± 0.0007 | +0.0004 [-0.0023, +0.0035] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | **-0.3470** ± 0.0015 | **+0.0065** [+0.0012, +0.0121] | **+0.0040** [-0.0001, +0.0080] | <u>-0.3069</u> ± 0.0006 | <u>+0.0013</u> [-0.0010, +0.0041] | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | <u>-0.3494</u> ± 0.0018 | <u>+0.0041</u> [-0.0018, +0.0103] | <u>+0.0016</u> [-0.0032, +0.0061] | **-0.3068** ± 0.0005 | **+0.0014** [-0.0009, +0.0043] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (its best on the first block and on this block) | **13.2** ± 0.497 | control | control | 256 of 256 | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | <u>13.5</u> ± 0.628 | **+0.291** [-1.07, +1.69] | **+0.291** [-1.07, +1.69] | 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (its best on this block: its first-block best, intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01, was not run here) | 14.7 ± 0.632 | <u>+1.46</u> [+0.109, +2.91] | <u>+1.46</u> [+0.109, +2.91] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 11:00 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:36 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:38 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:38 PT | none |
| 5 | fast last layer at step 0.01 with the first-arrival hold | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 06:34 PT | none |
| 6 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 02:31 PT | none |
| 7 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 02:31 PT | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.0078125_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-20000-to-20255`
5. `validation-block-two_pointmaze-medium_reward-step-cost_fast-step-0-point-01-with-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
6. `validation-block-two_pointmaze-medium_reward-step-cost_held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-20000-to-20255`
7. `validation-block-two_pointmaze-medium_reward-step-cost_held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`

## Arms at the same configuration, on the same seeds

Each group compares one arm with other sides at the configurations both trained, matched on the intrinsic weight, learning rate, entropy coefficient and clip coefficient, and paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Define $W$, $L$ and $M$ as in the section Measures. No configuration is chosen here.

### AntMaze U-maze, 128 environments per policy: fast last layer at step 0.03 with the first-arrival hold against RND

#### AntMaze U-maze, reward 1 on arrival

The arm: fast last layer at step 0.03 with the first-arrival hold, on 16 seeds from 0 to 15, read at 16,777,216 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.003, entropy coefficient 0.001 | 0.026 ± 0.001 | 0.035 ± 0.001 | -0.010 [-0.013, -0.007] | -0.010 [-0.014, -0.007] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 0.036 ± 0.002 | 0.043 ± 0.001 | -0.008 [-0.012, -0.004] | -0.007 [-0.012, -0.003] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.003, entropy coefficient 0.001 | 0.030 ± 0.002 | 0.038 ± 0.001 | -0.008 [-0.012, -0.004] | -0.009 [-0.014, -0.003] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 0.041 ± 0.002 | 0.046 ± 0.001 | -0.005 [-0.008, -0.002] | -0.007 [-0.011, -0.004] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.003, entropy coefficient 0.001 | 0.033 ± 0.001 | 0.037 ± 0.001 | -0.004 [-0.007, -0.001] | -0.006 [-0.009, -0.003] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 0.042 ± 0.001 | 0.049 ± 0.001 | -0.007 [-0.010, -0.004] | -0.008 [-0.012, -0.004] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.003, entropy coefficient 0.001 | 0.031 ± 0.001 | 0.038 ± 0.001 | -0.008 [-0.010, -0.005] | -0.007 [-0.010, -0.004] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001 | 0.043 ± 0.002 | 0.047 ± 0.001 | -0.003 [-0.007, -0.000] | -0.004 [-0.008, +0.001] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.003, entropy coefficient 0.001 | 0.026 ± 0.001 | 0.018 ± 0.001 | +0.008 [+0.005, +0.011] | +0.009 [+0.005, +0.013] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 0.036 ± 0.002 | 0.026 ± 0.002 | +0.010 [+0.007, +0.013] | +0.012 [+0.008, +0.016] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.003, entropy coefficient 0.001 | 0.030 ± 0.002 | 0.022 ± 0.001 | +0.008 [+0.005, +0.011] | +0.011 [+0.006, +0.016] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 0.041 ± 0.002 | 0.033 ± 0.002 | +0.009 [+0.005, +0.013] | +0.009 [+0.005, +0.014] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.003, entropy coefficient 0.001 | 0.033 ± 0.001 | 0.024 ± 0.001 | +0.010 [+0.006, +0.012] | +0.010 [+0.007, +0.013] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 0.042 ± 0.001 | 0.038 ± 0.001 | +0.004 [+0.002, +0.006] | +0.005 [+0.002, +0.008] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 64, learning rate 0.003, entropy coefficient 0.001 | 0.031 ± 0.001 | 0.025 ± 0.001 | +0.006 [+0.003, +0.009] | +0.005 [+0.001, +0.008] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001 | 0.043 ± 0.002 | 0.038 ± 0.001 | +0.005 [+0.002, +0.008] | +0.006 [+0.002, +0.010] | 16 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.003, entropy coefficient 0.001 | 3,082 ± 84.5 | 2,753 ± 95.5 | +329 [+63.6, +599] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 2,195 ± 112 | 1,865 ± 77 | +329 [+118, +564] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.003, entropy coefficient 0.001 | 3,245 ± 147 | 2,631 ± 99.1 | +614 [+254, +983] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 2,277 ± 120 | 1,981 ± 75.8 | +296 [+59.9, +518] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.003, entropy coefficient 0.001 | 3,160 ± 131 | 2,605 ± 109 | +555 [+198, +924] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 2,053 ± 101 | 2,112 ± 72.3 | -59 [-287, +177] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.003, entropy coefficient 0.001 | 3,228 ± 136 | 2,547 ± 128 | +681 [+344, +1,008] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001 | 2,119 ± 84.6 | 1,926 ± 86.8 | +194 [-46.6, +439] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.003, entropy coefficient 0.001 | 3,082 ± 84.5 | 4,103 ± 190 | -1,021 [-1,330, -702] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 2,195 ± 112 | 2,601 ± 206 | -406 [-830, -17.9] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.003, entropy coefficient 0.001 | 3,245 ± 147 | 3,885 ± 165 | -640 [-997, -301] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 2,277 ± 120 | 2,076 ± 126 | +201 [-157, +574] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.003, entropy coefficient 0.001 | 3,160 ± 131 | 3,842 ± 180 | -682 [-1,088, -251] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 2,053 ± 101 | 2,375 ± 94 | -322 [-618, -55.1] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 64, learning rate 0.003, entropy coefficient 0.001 | 3,228 ± 136 | 3,839 ± 207 | -610 [-1,122, -149] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001 | 2,119 ± 84.6 | 2,323 ± 137 | -203 [-539, +136] | 16 of 16; 16 of 16 | 16 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against CleanRL PPO-RND (environments-per-policy check): $W$ ahead at 0, level at 0, behind at 8 of 8; $M$ sooner at 0, level at 2, later at 6 of 8.
- against fast last layer (step 0.1) with the first-arrival hold (batch 10): $W$ ahead at 8, level at 0, behind at 0 of 8; $M$ sooner at 6, level at 2, later at 0 of 8.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer at step 0.03 with the first-arrival hold: best $W$ 0.0434 at intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001; best $M$ 2,053 thousand steps at intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 (of 8 configurations).
- CleanRL PPO-RND (environments-per-policy check): best $W$ 0.0487 at intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001; best $M$ 1,841 thousand steps at intrinsic weight 64, learning rate 0.01, entropy coefficient 0.01 (of 48 configurations).
- fast last layer (step 0.1) with the first-arrival hold (batch 10): best $W$ 0.0385 at intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001; best $M$ 2,076 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 (of 8 configurations).

### AntMaze U-maze, 128 environments per policy: fast last layer at step 0.01 with the first-arrival hold against RND

#### AntMaze U-maze, reward 1 on arrival

The arm: fast last layer at step 0.01 with the first-arrival hold, on 16 seeds from 0 to 15, read at 16,777,216 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.003, entropy coefficient 0.001 | 0.033 ± 0.001 | 0.035 ± 0.001 | -0.003 [-0.006, +0.000] | -0.006 [-0.010, -0.002] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 0.042 ± 0.002 | 0.043 ± 0.001 | -0.001 [-0.006, +0.004] | +0.000 [-0.005, +0.005] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.003, entropy coefficient 0.001 | 0.036 ± 0.001 | 0.038 ± 0.001 | -0.002 [-0.005, +0.001] | -0.002 [-0.006, +0.002] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 0.046 ± 0.002 | 0.046 ± 0.001 | -0.001 [-0.005, +0.003] | -0.002 [-0.007, +0.002] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.003, entropy coefficient 0.001 | 0.038 ± 0.001 | 0.037 ± 0.001 | +0.001 [-0.002, +0.004] | +0.000 [-0.003, +0.004] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 0.048 ± 0.001 | 0.049 ± 0.001 | -0.001 [-0.004, +0.002] | -0.001 [-0.005, +0.003] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.003, entropy coefficient 0.001 | 0.036 ± 0.001 | 0.038 ± 0.001 | -0.002 [-0.004, -0.000] | -0.001 [-0.004, +0.002] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001 | 0.046 ± 0.001 | 0.047 ± 0.001 | -0.001 [-0.003, +0.001] | -0.000 [-0.004, +0.003] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.003, entropy coefficient 0.001 | 0.033 ± 0.001 | 0.018 ± 0.001 | +0.015 [+0.013, +0.017] | +0.013 [+0.009, +0.018] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 0.042 ± 0.002 | 0.026 ± 0.002 | +0.016 [+0.013, +0.020] | +0.020 [+0.015, +0.025] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.003, entropy coefficient 0.001 | 0.036 ± 0.001 | 0.022 ± 0.001 | +0.014 [+0.013, +0.016] | +0.018 [+0.014, +0.022] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 0.046 ± 0.002 | 0.033 ± 0.002 | +0.013 [+0.010, +0.016] | +0.014 [+0.010, +0.018] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.003, entropy coefficient 0.001 | 0.038 ± 0.001 | 0.024 ± 0.001 | +0.014 [+0.012, +0.016] | +0.016 [+0.012, +0.021] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 0.048 ± 0.001 | 0.038 ± 0.001 | +0.009 [+0.007, +0.012] | +0.012 [+0.009, +0.015] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 64, learning rate 0.003, entropy coefficient 0.001 | 0.036 ± 0.001 | 0.025 ± 0.001 | +0.011 [+0.010, +0.013] | +0.011 [+0.008, +0.014] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001 | 0.046 ± 0.001 | 0.038 ± 0.001 | +0.007 [+0.005, +0.010] | +0.009 [+0.005, +0.014] | 16 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.003, entropy coefficient 0.001 | 2,696 ± 116 | 2,753 ± 95.5 | -56.9 [-317, +199] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 1,834 ± 66.2 | 1,865 ± 77 | -31.1 [-243, +174] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.003, entropy coefficient 0.001 | 2,750 ± 108 | 2,631 ± 99.1 | +119 [-173, +400] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 1,930 ± 94.3 | 1,981 ± 75.8 | -50.9 [-232, +156] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.003, entropy coefficient 0.001 | 2,549 ± 78.7 | 2,605 ± 109 | -55.5 [-318, +206] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 1,973 ± 101 | 2,112 ± 72.3 | -139 [-397, +121] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.003, entropy coefficient 0.001 | 2,698 ± 169 | 2,547 ± 128 | +151 [-252, +571] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001 | 2,045 ± 140 | 1,926 ± 86.8 | +120 [-205, +444] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.003, entropy coefficient 0.001 | 2,696 ± 116 | 4,103 ± 190 | -1,407 [-1,850, -961] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 1,834 ± 66.2 | 2,601 ± 206 | -767 [-1,297, -330] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.003, entropy coefficient 0.001 | 2,750 ± 108 | 3,885 ± 165 | -1,136 [-1,444, -811] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 1,930 ± 94.3 | 2,076 ± 126 | -146 [-426, +126] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.003, entropy coefficient 0.001 | 2,549 ± 78.7 | 3,842 ± 180 | -1,293 [-1,655, -905] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 1,973 ± 101 | 2,375 ± 94 | -402 [-658, -146] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 64, learning rate 0.003, entropy coefficient 0.001 | 2,698 ± 169 | 3,839 ± 207 | -1,141 [-1,539, -764] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001 | 2,045 ± 140 | 2,323 ± 137 | -277 [-634, +85.7] | 16 of 16; 16 of 16 | 16 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against CleanRL PPO-RND (environments-per-policy check): $W$ ahead at 0, level at 7, behind at 1 of 8; $M$ sooner at 0, level at 8, later at 0 of 8.
- against fast last layer (step 0.1) with the first-arrival hold (batch 10): $W$ ahead at 8, level at 0, behind at 0 of 8; $M$ sooner at 6, level at 2, later at 0 of 8.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer at step 0.01 with the first-arrival hold: best $W$ 0.0476 at intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001; best $M$ 1,834 thousand steps at intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 (of 8 configurations).
- CleanRL PPO-RND (environments-per-policy check): best $W$ 0.0487 at intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001; best $M$ 1,841 thousand steps at intrinsic weight 64, learning rate 0.01, entropy coefficient 0.01 (of 48 configurations).
- fast last layer (step 0.1) with the first-arrival hold (batch 10): best $W$ 0.0385 at intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001; best $M$ 2,076 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 (of 8 configurations).

### AntMaze U-maze, 16 environments per policy: fast last layer at step 0.03 with the first-arrival hold against RND

#### AntMaze U-maze, reward 1 on arrival

The arm: fast last layer at step 0.03 with the first-arrival hold, on 16 seeds from 0 to 15, read at 16,777,216 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 0.053 ± 0.002 | 0.049 ± 0.003 | +0.004 [-0.001, +0.010] | +0.008 [-0.002, +0.017] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.01, entropy coefficient 0.001 | 0.028 ± 0.001 | 0.025 ± 0.001 | +0.003 [-0.001, +0.007] | +0.004 [-0.002, +0.012] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 0.039 ± 0.001 | 0.034 ± 0.001 | +0.005 [+0.003, +0.007] | +0.004 [+0.000, +0.009] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 0.041 ± 0.001 | 0.041 ± 0.002 | +0.000 [-0.003, +0.004] | -0.000 [-0.005, +0.005] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 0.041 ± 0.001 | 0.037 ± 0.001 | +0.003 [+0.001, +0.005] | +0.001 [-0.002, +0.006] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 0.040 ± 0.001 | 0.038 ± 0.001 | +0.002 [-0.000, +0.005] | +0.001 [-0.003, +0.005] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 0.042 ± 0.001 | 0.039 ± 0.001 | +0.003 [+0.001, +0.004] | +0.002 [-0.003, +0.007] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 0.042 ± 0.001 | 0.039 ± 0.002 | +0.002 [+0.000, +0.005] | +0.002 [-0.003, +0.007] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 0.053 ± 0.002 | 0.054 ± 0.004 | -0.001 [-0.008, +0.006] | +0.002 [-0.009, +0.012] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 0.0625, learning rate 0.01, entropy coefficient 0.001 | 0.028 ± 0.001 | 0.032 ± 0.002 | -0.004 [-0.009, +0.001] | -0.003 [-0.012, +0.006] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 0.039 ± 0.001 | 0.042 ± 0.001 | -0.003 [-0.005, -0.000] | -0.002 [-0.008, +0.004] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 0.041 ± 0.001 | 0.042 ± 0.001 | -0.001 [-0.003, +0.001] | -0.002 [-0.006, +0.003] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 0.041 ± 0.001 | 0.045 ± 0.001 | -0.004 [-0.006, -0.001] | -0.003 [-0.007, +0.000] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 0.040 ± 0.001 | 0.042 ± 0.001 | -0.002 [-0.004, -0.001] | -0.005 [-0.010, -0.001] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 0.042 ± 0.001 | 0.045 ± 0.001 | -0.003 [-0.005, -0.002] | -0.000 [-0.004, +0.004] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 0.042 ± 0.001 | 0.045 ± 0.002 | -0.003 [-0.006, -0.000] | -0.001 [-0.007, +0.005] | 16 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 1,195 ± 66.1 | 1,258 ± 61.2 | -62.9 [-205, +70.4] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.01, entropy coefficient 0.001 | 430 ± 20.3 | 489 ± 46.1 | -58.8 [-169, +43.5] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 621 ± 30.3 | 827 ± 37.4 | -207 [-292, -123] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 263 ± 23 | 338 ± 31.7 | -75.3 [-137, -15.2] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 556 ± 22.4 | 720 ± 41.3 | -164 [-258, -71.9] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 317 ± 26.9 | 319 ± 19.8 | -1.73 [-69.4, +67] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 579 ± 33.2 | 680 ± 43.2 | -101 [-217, -6.84] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 276 ± 18 | 349 ± 27 | -73 [-138, -7.75] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 1,195 ± 66.1 | 1,165 ± 62.6 | +30.2 [-66.2, +129] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 0.0625, learning rate 0.01, entropy coefficient 0.001 | 430 ± 20.3 | 361 ± 39.3 | +68.6 [-21.3, +150] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 621 ± 30.3 | 570 ± 28.9 | +50.4 [-38.7, +140] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 263 ± 23 | 298 ± 31.9 | -34.8 [-118, +43.5] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 556 ± 22.4 | 489 ± 25.8 | +66.7 [+12, +114] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 317 ± 26.9 | 276 ± 27.1 | +40.8 [-32.8, +119] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 579 ± 33.2 | 473 ± 32.1 | +105 [+25.3, +180] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 276 ± 18 | 251 ± 25.7 | +25.4 [-36, +88.3] | 16 of 16; 16 of 16 | 16 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against CleanRL PPO-RND (environments-per-policy check): $W$ ahead at 4, level at 4, behind at 0 of 8; $M$ sooner at 5, level at 3, later at 0 of 8.
- against fast last layer (step 0.1) with the first-arrival hold (batch 10): $W$ ahead at 0, level at 3, behind at 5 of 8; $M$ sooner at 0, level at 6, later at 2 of 8.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer at step 0.03 with the first-arrival hold: best $W$ 0.0530 at intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001; best $M$ 263 thousand steps at intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 (of 8 configurations).
- CleanRL PPO-RND (environments-per-policy check): best $W$ 0.0487 at intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001; best $M$ 314 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (of 48 configurations).
- fast last layer (step 0.1) with the first-arrival hold (batch 10): best $W$ 0.0538 at intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001; best $M$ 251 thousand steps at intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 (of 8 configurations).

### AntMaze U-maze, 16 environments per policy: fast last layer at step 0.01 with the first-arrival hold against RND

#### AntMaze U-maze, reward 1 on arrival

The arm: fast last layer at step 0.01 with the first-arrival hold, on 16 seeds from 0 to 15, read at 16,777,216 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 0.051 ± 0.003 | 0.049 ± 0.003 | +0.002 [-0.004, +0.009] | +0.003 [-0.007, +0.012] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.01, entropy coefficient 0.001 | 0.027 ± 0.002 | 0.025 ± 0.001 | +0.002 [-0.002, +0.006] | +0.002 [-0.005, +0.009] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 0.035 ± 0.001 | 0.034 ± 0.001 | +0.001 [-0.001, +0.003] | -0.002 [-0.006, +0.003] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 0.038 ± 0.002 | 0.041 ± 0.002 | -0.003 [-0.006, +0.001] | -0.003 [-0.009, +0.003] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 0.037 ± 0.001 | 0.037 ± 0.001 | +0.000 [-0.002, +0.002] | -0.000 [-0.005, +0.004] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 0.037 ± 0.002 | 0.038 ± 0.001 | -0.001 [-0.005, +0.003] | -0.001 [-0.007, +0.005] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 0.040 ± 0.001 | 0.039 ± 0.001 | +0.001 [-0.001, +0.002] | -0.000 [-0.005, +0.005] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 0.038 ± 0.001 | 0.039 ± 0.002 | -0.002 [-0.005, +0.001] | -0.004 [-0.010, +0.002] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 0.051 ± 0.003 | 0.054 ± 0.004 | -0.003 [-0.011, +0.005] | -0.003 [-0.014, +0.007] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 0.0625, learning rate 0.01, entropy coefficient 0.001 | 0.027 ± 0.002 | 0.032 ± 0.002 | -0.006 [-0.010, -0.002] | -0.006 [-0.012, +0.001] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 0.035 ± 0.001 | 0.042 ± 0.001 | -0.007 [-0.010, -0.004] | -0.008 [-0.013, -0.003] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 0.038 ± 0.002 | 0.042 ± 0.001 | -0.004 [-0.007, -0.001] | -0.004 [-0.010, +0.002] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 0.037 ± 0.001 | 0.045 ± 0.001 | -0.007 [-0.009, -0.005] | -0.005 [-0.010, +0.000] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 0.037 ± 0.002 | 0.042 ± 0.001 | -0.005 [-0.008, -0.003] | -0.007 [-0.011, -0.003] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 0.040 ± 0.001 | 0.045 ± 0.001 | -0.006 [-0.007, -0.004] | -0.003 [-0.007, +0.002] | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 0.038 ± 0.001 | 0.045 ± 0.002 | -0.007 [-0.010, -0.004] | -0.006 [-0.011, -0.002] | 16 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 1,225 ± 57.7 | 1,258 ± 61.2 | -33.3 [-154, +95.5] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.01, entropy coefficient 0.001 | 423 ± 34.6 | 489 ± 46.1 | -65.9 [-193, +60.4] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 686 ± 31.7 | 827 ± 37.4 | -141 [-215, -66.1] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 334 ± 18.4 | 338 ± 31.7 | -4.33 [-76.2, +65.9] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 611 ± 43 | 720 ± 41.3 | -110 [-199, -22.6] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 329 ± 26 | 319 ± 19.8 | +10.8 [-43.4, +68.5] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 632 ± 32.9 | 680 ± 43.2 | -47.8 [-113, +16.9] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 309 ± 14.9 | 349 ± 27 | -39.6 [-101, +18.1] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 1,225 ± 57.7 | 1,165 ± 62.6 | +59.7 [-63.7, +172] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 0.0625, learning rate 0.01, entropy coefficient 0.001 | 423 ± 34.6 | 361 ± 39.3 | +61.4 [-42.3, +156] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 686 ± 31.7 | 570 ± 28.9 | +116 [+63.3, +172] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 334 ± 18.4 | 298 ± 31.9 | +36.1 [-29.5, +101] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 611 ± 43 | 489 ± 25.8 | +121 [+33.9, +218] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 329 ± 26 | 276 ± 27.1 | +53.3 [-24.2, +123] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 632 ± 32.9 | 473 ± 32.1 | +159 [+77.7, +242] | 16 of 16; 16 of 16 | 16 |
| fast last layer (step 0.1) with the first-arrival hold (batch 10) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 309 ± 14.9 | 251 ± 25.7 | +58.8 [+5.73, +107] | 16 of 16; 16 of 16 | 16 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against CleanRL PPO-RND (environments-per-policy check): $W$ ahead at 0, level at 8, behind at 0 of 8; $M$ sooner at 2, level at 6, later at 0 of 8.
- against fast last layer (step 0.1) with the first-arrival hold (batch 10): $W$ ahead at 0, level at 1, behind at 7 of 8; $M$ sooner at 0, level at 4, later at 4 of 8.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer at step 0.01 with the first-arrival hold: best $W$ 0.0510 at intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001; best $M$ 309 thousand steps at intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 (of 8 configurations).
- CleanRL PPO-RND (environments-per-policy check): best $W$ 0.0487 at intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001; best $M$ 314 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (of 48 configurations).
- fast last layer (step 0.1) with the first-arrival hold (batch 10): best $W$ 0.0538 at intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001; best $M$ 251 thousand steps at intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 (of 8 configurations).

## Predictions

The predictions were written in `ideas.md` before the runs. Each is quoted, then read part by part: a part held or did not hold on the numbers given, or cannot be read yet because a unit it needs is not complete. A prediction held when every part held, and did not hold when at least one part did not. Where a prediction leaves a choice, it is read as follows. On the ant (16 development seeds, 16,777,216 steps per policy) every comparison is with RND of the check at the same configuration and seeds, and the configurations are counted by the sign of the mean paired difference in $W$: "ahead" is above RND in at least 6 of the 8, "behind in most" below RND in at least 5 of the 8, "level" neither above nor below in 6 or more, and "within 0.003" is read on the mean difference. "By less than step 0.1" is read as: the mean over the 8 configurations of the paired difference in $W$ between step 0.03 and step 0.1 is below zero. On PointMaze Medium the second block is read with each side at the configuration fixed on the first block; "within 0.005 of the hold alone" is read on the mean paired difference, and "ahead of RND" as: the 95% interval of the difference under equal rules is entirely above zero. "A few per unit" is read as at most 10.

### Prediction 1: ant, 128 environments per policy

> **Ant, 128 environments per policy:** the smaller the step, the closer to RND: at step 0.01 the whole-curve mean is within 0.003 of RND's at the same configuration in at least 6 of 8 configurations; at step 0.03 it is still behind in most. No step is ahead of RND at this count.

Observed: **held**.

- Step 0.01: within 0.003 of RND in at least 6 of 8 configurations: **held**. $W$ difference from RND: above zero in 1 and below zero in 7 of 8 configurations (from -0.0026 to +0.0008).
- Step 0.03: behind RND in most configurations: **held**. $W$ difference from RND: above zero in 0 and below zero in 8 of 8 configurations (from -0.0095 to -0.0034).
- No step is ahead of RND at this count: **held**. Step 0.03: above zero in 0 and below zero in 8 of 8 configurations (from -0.0095 to -0.0034). Step 0.01: above zero in 1 and below zero in 7 of 8 configurations (from -0.0026 to +0.0008).

### Prediction 2: ant, 16 environments per policy

> **Ant, 16 environments per policy:** step 0.03 is ahead of RND at the same configuration in at least 6 of 8 configurations, by less than step 0.1; step 0.01 is level with RND.

Observed: **held**.

- Step 0.03: ahead of RND in at least 6 of 8 configurations: **held**. $W$ difference from RND: above zero in 8 and below zero in 0 of 8 configurations (from +0.0005 to +0.0049).
- Step 0.03: by less than step 0.1: **held**. $W$ difference from step 0.1: above zero in 0 and below zero in 8 of 8 configurations (from -0.0042 to -0.0009).
- Step 0.01: level with RND: **held**. $W$ difference from RND: above zero in 5 and below zero in 3 of 8 configurations (from -0.0026 to +0.0023).

### Prediction 3: PointMaze Medium, second block

> **PointMaze Medium, second block:** step 0.01 with the hold is within 0.005 of the hold alone on the whole-curve mean in both cases, and ahead of RND with reward 1 on arrival.

Observed: **held**.

- Reward 1 on arrival: step 0.01 with the hold within 0.005 of the hold alone: **held**. $W$ 0.9403 against the hold alone's 0.9404; paired difference -0.0000 [-0.0074, +0.0075].
- Step cost: step 0.01 with the hold within 0.005 of the hold alone: **held**. $W$ -0.3494 against the hold alone's -0.3470; paired difference -0.0024 [-0.0070, +0.0021].
- Reward 1 on arrival: ahead of RND: **held**. $W$ 0.9403; difference from RND under equal rules +0.0095 [+0.0020, +0.0171].

### Prediction 4: fault-guard respawns

> **Fault-guard respawns** stay at a few per unit.

Observed: **did not hold**.

- At most a few per unit: **did not hold**. Respawns: step 0.03 with 128 environments per policy 13; step 0.03 with 16 environments per policy 23; step 0.01 with 128 environments per policy 8; step 0.01 with 16 environments per policy 38.

