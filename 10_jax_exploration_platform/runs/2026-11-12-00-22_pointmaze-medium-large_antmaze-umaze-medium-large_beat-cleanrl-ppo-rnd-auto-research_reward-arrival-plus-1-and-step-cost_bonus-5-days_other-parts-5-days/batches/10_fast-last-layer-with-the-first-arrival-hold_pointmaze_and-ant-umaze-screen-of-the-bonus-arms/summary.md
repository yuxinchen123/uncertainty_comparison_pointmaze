# Batch 10: the fast last layer with the first-arrival hold on PointMaze, and the first AntMaze U-maze screen of the bonus arms

Generated 2026-11-13 11:35 PT by the command below. This file is generated: regenerate it, do not edit it. Times are Pacific, converted from machines running Eastern. Run folder: `2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days`; batch folder: `batches/10_fast-last-layer-with-the-first-arrival-hold_pointmaze_and-ant-umaze-screen-of-the-bonus-arms`.

```bash
cd /p/rlprojects/RND/10_jax_exploration_platform
bash "benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh" -s scripts/maze_campaign/analysis/batch_summary_job.py \
    --run-dir=runs/2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days \
    --batch=10_fast-last-layer-with-the-first-arrival-hold_pointmaze_and-ant-umaze-screen-of-the-bonus-arms
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

A case is one environment with one reward setting. The validation seeds are neither the development seeds nor the confirmation seeds. They come in two blocks. On the first block each side stands at its best configuration among the few run there, so that block is a second selection step. On the second block nothing was chosen: a method stands there at the configuration fixed on the first block, and is compared with the control under equal rules and with the control favoured. Only complete units are read; nothing is estimated for the others.

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
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.8424 ± 0.0063 | 0.9747 ± 0.0037 | <u>15.7</u> ± 0.84 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 0.7759 ± 0.0077 | 0.9490 ± 0.0060 | **14.7** ± 0.957 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | **0.9435** ± 0.0027 | <u>0.9999</u> ± 0.0000 | 18.5 ± 1.13 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.01, entropy coefficient 0.01 | <u>0.9412</u> ± 0.0024 | **1.0000** ± 0.0000 | 23.1 ± 1.34 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9334 ± 0.0029 | **1.0000** ± 0.0000 | 29.4 ± 1.85 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 6) | <u>0.9349</u> ± 0.0031 | control | <u>0.9950</u> ± 0.0010 | control | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | **0.9435** ± 0.0027 | **+0.0085** [+0.0009, +0.0161] | **0.9999** ± 0.0000 | **+0.0049** [+0.0033, +0.0070] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 2) | 0.8424 ± 0.0063 | <u>-0.0925</u> [-0.1068, -0.0788] | 0.9747 ± 0.0037 | <u>-0.0203</u> [-0.0285, -0.0134] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (best of 6) | <u>17.5</u> ± 1.41 | control | 256 of 256 | 0.0073 | 0.0000 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (best of 2) | **14.7** ± 0.957 | **-2.85** [-6.45, +0.268] | 256 of 256 | <u>0.7759</u> | <u>0.9490</u> | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | 18.5 ± 1.13 | <u>+0.993</u> [-2.74, +4.35] | 256 of 256 | **0.9435** | **0.9999** | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:14 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:15 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:14 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 5 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:53 PT | none |
| 6 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 7 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 03:06 PT | none |
| 8 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 03:08 PT | none |
| 9 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:32 PT | none |
| 10 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:32 PT | none |
| 11 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:34 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.1_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phases `validation-block-two` and `discovery-block-two` in PointMaze Medium, reward 1 on arrival. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | <u>0.9308</u> ± 0.0030 | <u>0.9907</u> ± 0.0039 | 26.3 ± 1.88 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.01, entropy coefficient 0.01 | 0.9276 ± 0.0026 | 0.9798 ± 0.0059 | 23.6 ± 1.4 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9141 ± 0.0030 | 0.9821 ± 0.0018 | <u>19.6</u> ± 1.14 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 | 0.0075 ± 0.0003 | 0.0000 ± 0.0000 | **16.8** ± 1.43 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | **0.9404** ± 0.0030 | **0.9999** ± 0.0000 | 20.6 ± 1.12 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (its best on the first block and on this block) | <u>0.9308</u> ± 0.0030 | control | control | <u>0.9907</u> ± 0.0039 | control | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | **0.9404** ± 0.0030 | +0.0095 [+0.0013, +0.0177] | +0.0095 [+0.0013, +0.0177] | **0.9999** ± 0.0000 | +0.0092 [+0.0042, +0.0180] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (its best on the first block and on this block) | **16.8** ± 1.43 | control | control | 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | <u>20.6</u> ± 1.12 | +3.83 [+0.451, +7.02] | +3.83 [+0.451, +7.02] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 10:58 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:16 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:19 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:16 PT | none |
| 5 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:31 PT | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.1_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
5. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`

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
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3692 ± 0.0030 | -0.3132 ± 0.0008 | <u>12.9</u> ± 0.654 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | -0.4057 ± 0.0034 | -0.3299 ± 0.0015 | 15 ± 1.74 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 | **-0.3476** ± 0.0014 | -0.3073 ± 0.0007 | 15.4 ± 0.693 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | <u>-0.3484</u> ± 0.0014 | **-0.3058** ± 0.0004 | 14.2 ± 0.617 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | -0.3504 ± 0.0015 | <u>-0.3066</u> ± 0.0005 | 13.4 ± 0.562 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01 (best of 6) | <u>-0.3505</u> ± 0.0016 | control | <u>-0.3078</u> ± 0.0007 | control | 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 (best of 3) | **-0.3476** ± 0.0014 | **+0.0029** [-0.0015, +0.0072] | **-0.3073** ± 0.0007 | **+0.0005** [-0.0014, +0.0025] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 2) | -0.3692 ± 0.0030 | <u>-0.0187</u> [-0.0258, -0.0120] | -0.3132 ± 0.0008 | <u>-0.0054</u> [-0.0076, -0.0034] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (best of 6) | **12.4** ± 0.471 | control | 256 of 256 | -0.5526 | -0.6884 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 2) | <u>12.9</u> ± 0.654 | **+0.523** [-0.938, +2.07] | 256 of 256 | <u>-0.3692</u> | <u>-0.3132</u> | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | 13.4 ± 0.562 | <u>+0.958</u> [-0.349, +2.26] | 256 of 256 | **-0.3504** | **-0.3066** | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:16 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:19 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:18 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:58 PT | none |
| 5 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:53 PT | none |
| 6 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:54 PT | none |
| 7 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:11 PT | none |
| 8 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:13 PT | none |
| 9 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:35 PT | none |
| 10 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:35 PT | none |
| 11 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:34 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.0078125_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-medium_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phases `validation-block-two` and `discovery-block-two` in PointMaze Medium, step cost. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.0078125, learning rate 0.006, entropy coefficient 0.01 | -0.3510 ± 0.0016 | -0.3078 ± 0.0007 | 20.4 ± 0.931 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01 | -0.3535 ± 0.0024 | -0.3082 ± 0.0012 | 19.9 ± 1.01 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.01, entropy coefficient 0.01 | -0.3541 ± 0.0014 | -0.3073 ± 0.0006 | 20.6 ± 0.841 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 | -0.5581 ± 0.0108 | -0.7035 ± 0.0206 | **13.2** ± 0.497 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 | **-0.3470** ± 0.0015 | <u>-0.3069</u> ± 0.0006 | 16.9 ± 0.742 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | <u>-0.3472</u> ± 0.0015 | **-0.3059** ± 0.0005 | <u>14.7</u> ± 0.632 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND), at its first-block best | intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | -0.3535 ± 0.0024 | control | -0.0025 [-0.0081, +0.0025] | -0.3082 ± 0.0012 | control | 256 |
| control (CleanRL PPO-RND), at its best on this block | intrinsic weight 0.0078125, learning rate 0.006, entropy coefficient 0.01 (best of 4 on this block) | <u>-0.3510</u> ± 0.0016 | +0.0025 [-0.0025, +0.0082] | control | <u>-0.3078</u> ± 0.0007 | +0.0004 [-0.0023, +0.0035] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | **-0.3470** ± 0.0015 | +0.0065 [+0.0012, +0.0121] | +0.0040 [-0.0001, +0.0080] | **-0.3069** ± 0.0006 | +0.0013 [-0.0010, +0.0041] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (its best on the first block and on this block) | **13.2** ± 0.497 | control | control | 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (its best on this block: its first-block best, intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01, was not run here) | <u>14.7</u> ± 0.632 | +1.46 [+0.109, +2.91] | +1.46 [+0.109, +2.91] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 11:00 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:36 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:38 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:38 PT | none |
| 5 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 02:31 PT | none |
| 6 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 02:31 PT | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.0078125_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-20000-to-20255`
5. `validation-block-two_pointmaze-medium_reward-step-cost_held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-20000-to-20255`
6. `validation-block-two_pointmaze-medium_reward-step-cost_held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`

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
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8707 ± 0.0058 | 0.9872 ± 0.0055 | <u>90.7</u> ± 5.08 | **256 of 256** |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 | 0.5725 ± 0.0106 | 0.8495 ± 0.0123 | **86.1** ± 6.37 | **256 of 256** |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | **0.8874** ± 0.0023 | <u>0.9977</u> ± 0.0002 | 259 ± 10.6 | **256 of 256** |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | <u>0.8852</u> ± 0.0023 | 0.9976 ± 0.0003 | 201 ± 9.11 | **256 of 256** |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 0.8680 ± 0.0033 | **0.9979** ± 0.0003 | 375 ± 12.4 | **256 of 256** |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 8) | <u>0.8781</u> ± 0.0021 | control | <u>0.9920</u> ± 0.0005 | control | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 3) | **0.8874** ± 0.0023 | **+0.0094** [+0.0032, +0.0156] | **0.9977** ± 0.0002 | **+0.0057** [+0.0046, +0.0068] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 2) | 0.8707 ± 0.0058 | <u>-0.0074</u> [-0.0203, +0.0037] | 0.9872 ± 0.0055 | <u>-0.0048</u> [-0.0169, +0.0034] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (best of 8) | <u>119</u> ± 5.46 | control | 256 of 256 | 0.0054 | 0.0000 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 (best of 2) | **86.1** ± 6.37 | **-33.2** [-49, -16.5] | 256 of 256 | <u>0.5725</u> | <u>0.8495</u> | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (best of 3) | 201 ± 9.11 | <u>+81.6</u> [+61.3, +102] | 256 of 256 | **0.8852** | **0.9976** | 256 |

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
| 9 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 03:44 PT | none |
| 10 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:46 PT | none |
| 11 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:55 PT | none |
| 12 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:55 PT | none |
| 13 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:07 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.001_entropy-coefficient-0.03_intrinsic-weight-256.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.001_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phases `validation-block-two` and `discovery-block-two` in PointMaze Large, reward 1 on arrival. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 8,388,608 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 | **0.8842** ± 0.0019 | <u>0.9927</u> ± 0.0005 | <u>266</u> ± 10.6 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006 | 0.0057 ± 0.0002 | 0.0003 ± 0.0002 | **130** ± 5.17 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002 | <u>0.8839</u> ± 0.0025 | **0.9976** ± 0.0003 | <u>266</u> ± 11.6 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 (its best on the first block and on this block) | **0.8842** ± 0.0019 | control | control | <u>0.9927</u> ± 0.0005 | control | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002 (its best on the first block) | <u>0.8839</u> ± 0.0025 | -0.0003 [-0.0066, +0.0061] | -0.0003 [-0.0066, +0.0061] | **0.9976** ± 0.0003 | +0.0049 [+0.0038, +0.0059] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006 (its best on the first block and on this block) | **130** ± 5.17 | control | control | 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002 (its best on this block: its first-block best, intrinsic weight 4, learning rate 0.002, was not run here) | <u>266</u> ± 11.6 | +136 [+112, +161] | +136 [+112, +161] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 11:13 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:45 PT | none |
| 3 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:53 PT | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-large_reward-arrival-plus-one_held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`

### PointMaze Large, step cost

#### First block: each side at its best configuration there

The units of the phases `validation-block` and `discovery-validation` in PointMaze Large, step cost. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 33,554,432 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 | -0.7310 ± 0.0018 | -0.7163 ± 0.0017 | 191 ± 10.1 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 | -0.7411 ± 0.0036 | -0.7229 ± 0.0033 | 241 ± 13.4 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.0045 | -0.8831 ± 0.0022 | -0.8849 ± 0.0026 | 131 ± 6.45 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006 | -0.9669 ± 0.0017 | -0.9803 ± 0.0022 | 120 ± 5.6 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002 | <u>-0.7264</u> ± 0.0016 | **-0.7031** ± 0.0018 | **71.5** ± 4.37 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.0045 | -0.7638 ± 0.0030 | -0.7281 ± 0.0032 | <u>78.6</u> ± 5.91 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 | **-0.7236** ± 0.0021 | <u>-0.7079</u> ± 0.0021 | 163 ± 8.72 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.002 | -0.7268 ± 0.0025 | -0.7115 ± 0.0025 | 189 ± 9.38 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002 | -0.7335 ± 0.0034 | -0.7198 ± 0.0035 | 206 ± 11.5 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 (best of 4) | -0.7310 ± 0.0018 | control | -0.7163 ± 0.0017 | control | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 (best of 3) | **-0.7236** ± 0.0021 | **+0.0074** [+0.0018, +0.0129] | <u>-0.7079</u> ± 0.0021 | <u>+0.0084</u> [+0.0029, +0.0137] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (best of 2) | <u>-0.7264</u> ± 0.0016 | <u>+0.0046</u> [-0.0003, +0.0094] | **-0.7031** ± 0.0018 | **+0.0132** [+0.0081, +0.0182] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006 (best of 4) | <u>120</u> ± 5.6 | control | 256 of 256 | -0.9669 | -0.9803 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (best of 2) | **71.5** ± 4.37 | **-48.8** [-62.6, -34.3] | 256 of 256 | <u>-0.7264</u> | **-0.7031** | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 (best of 3) | 163 ± 8.72 | <u>+42.9</u> [+24.2, +62.7] | 256 of 256 | **-0.7236** | <u>-0.7079</u> | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:48 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:47 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:25 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:03 PT | none |
| 5 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:16 PT | 1 continued from a checkpoint |
| 6 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:30 PT | 1 continued from a checkpoint |
| 7 | divisor held from each policy's first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:32 PT | 1 continued from a checkpoint |
| 8 | divisor held from each policy's first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:04 PT | none |
| 9 | divisor held from each policy's first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:05 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.0045_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `validation-block_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-large_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-large_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.0045_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-large_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-large_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-large_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phases `validation-block-two` and `discovery-block-two` in PointMaze Large, step cost. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 33,554,432 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 | -0.7328 ± 0.0020 | -0.7175 ± 0.0021 | 203 ± 10.8 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006 | -0.9636 ± 0.0020 | -0.9753 ± 0.0025 | <u>121</u> ± 5.71 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002 | **-0.7245** ± 0.0007 | **-0.7021** ± 0.0009 | **76.1** ± 7.8 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 | <u>-0.7255</u> ± 0.0023 | <u>-0.7101</u> ± 0.0023 | 152 ± 7.71 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 (its best on the first block and on this block) | -0.7328 ± 0.0020 | control | control | -0.7175 ± 0.0021 | control | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (its best on the first block) | **-0.7245** ± 0.0007 | **+0.0083** [+0.0044, +0.0128] | **+0.0083** [+0.0044, +0.0128] | **-0.7021** ± 0.0009 | **+0.0154** [+0.0110, +0.0202] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 (its best on the first block) | <u>-0.7255</u> ± 0.0023 | <u>+0.0072</u> [+0.0013, +0.0131] | <u>+0.0072</u> [+0.0013, +0.0131] | <u>-0.7101</u> ± 0.0023 | <u>+0.0074</u> [+0.0013, +0.0135] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006 (its best on the first block and on this block) | <u>121</u> ± 5.71 | control | control | 256 of 256 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (its best on the first block) | **76.1** ± 7.8 | **-44.8** [-62.5, -24.7] | **-44.8** [-62.5, -24.7] | 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 (its best on the first block) | 152 ± 7.71 | <u>+31.5</u> [+11.9, +52.1] | <u>+31.5</u> [+11.9, +52.1] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 11:32 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 04:00 PT | none |
| 3 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 08:43 PT | none |
| 4 | divisor held from each policy's first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 04:25 PT | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-large_reward-step-cost_fast-step-0-point-1-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-large_reward-step-cost_held-from-first-arrival_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`

## Arms at the same configuration, on the same seeds

Each group compares one arm with other sides at the configurations both trained, matched on the intrinsic weight, learning rate, entropy coefficient and clip coefficient, and paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Define $W$, $L$ and $M$ as in the section Measures. No configuration is chosen here.

### Fast last layer (step 0.1) with the first-arrival hold, first validation block

#### PointMaze Medium, reward 1 on arrival

The arm: fast last layer (step 0.1) with the first-arrival hold, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.8424 ± 0.0063 | 0.9435 ± 0.0027 | -0.1010 [-0.1148, -0.0883] | -0.0252 [-0.0332, -0.0188] | 256 |
| fast last layer alone (step 0.1) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 0.7759 ± 0.0077 | 0.3785 ± 0.0072 | +0.3974 [+0.3775, +0.4169] | +0.4379 [+0.4122, +0.4642] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.8424 ± 0.0063 | 0.9194 ± 0.0023 | -0.0770 [-0.0899, -0.0652] | -0.0047 [-0.0139, +0.0036] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 0.7759 ± 0.0077 | 0.8718 ± 0.0034 | -0.0959 [-0.1114, -0.0811] | -0.0048 [-0.0185, +0.0071] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 15.7 ± 0.84 | 18.5 ± 1.13 | -2.83 [-5.42, -0.385] | 256 of 256; 256 of 256 | 256 |
| fast last layer alone (step 0.1) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 14.7 ± 0.957 | 13.8 ± 0.763 | +0.887 [-1.07, +2.93] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 15.7 ± 0.84 | 19.9 ± 1.16 | -4.23 [-6.81, -1.62] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 14.7 ± 0.957 | 17.6 ± 0.96 | -2.9 [-5.39, -0.365] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 0, behind at 1 of 1; $M$ sooner at 1, level at 0, later at 0 of 1; not trained by it: intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01.
- against fast last layer alone (step 0.1): $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1; not trained by it: intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 2 of 2; $M$ sooner at 2, level at 0, later at 0 of 2.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer (step 0.1) with the first-arrival hold: best $W$ 0.8424 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 14.7 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (of 2 configurations).
- divisor held from each policy's first arrival: best $W$ 0.9435 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 18.5 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 3 configurations).
- fast last layer alone (step 0.1): best $W$ 0.3785 at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01; best $M$ 12.5 thousand steps at intrinsic weight 16, learning rate 0.01, entropy coefficient 0.03 (of 8 configurations).
- CleanRL PPO-RND: best $W$ 0.9349 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 17.5 thousand steps at intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (of 6 configurations).

#### PointMaze Medium, step cost

The arm: fast last layer (step 0.1) with the first-arrival hold, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3692 ± 0.0030 | -0.3484 ± 0.0014 | -0.0208 [-0.0273, -0.0147] | -0.0074 [-0.0094, -0.0059] | 256 |
| fast last layer alone (step 0.1) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | -0.4057 ± 0.0034 | -0.5857 ± 0.0044 | +0.1800 [+0.1696, +0.1909] | +0.1647 [+0.1531, +0.1770] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | -0.4057 ± 0.0034 | -0.3680 ± 0.0015 | -0.0377 [-0.0451, -0.0309] | -0.0055 [-0.0092, -0.0023] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 12.9 ± 0.654 | 14.2 ± 0.617 | -1.32 [-2.84, +0.256] | 256 of 256; 256 of 256 | 256 |
| fast last layer alone (step 0.1) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 15 ± 1.74 | 11.7 ± 0.564 | +3.34 [+0.353, +7.34] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 15 ± 1.74 | 14.5 ± 0.63 | +0.513 [-2.49, +4.43] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 0, behind at 1 of 1; $M$ sooner at 0, level at 1, later at 0 of 1; not trained by it: intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01.
- against fast last layer alone (step 0.1): $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 0, later at 1 of 1; not trained by it: intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 1 of 1; $M$ sooner at 0, level at 1, later at 0 of 1; not trained by it: intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer (step 0.1) with the first-arrival hold: best $W$ -0.3692 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 12.9 thousand steps at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (of 2 configurations).
- divisor held from each policy's first arrival: best $W$ -0.3476 at intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01; best $M$ 13.4 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 3 configurations).
- fast last layer alone (step 0.1): best $W$ -0.5857 at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01; best $M$ 11.4 thousand steps at intrinsic weight 16, learning rate 0.01, entropy coefficient 0.03 (of 8 configurations).
- CleanRL PPO-RND: best $W$ -0.3505 at intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01; best $M$ 12.4 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (of 6 configurations).

#### PointMaze Large, reward 1 on arrival

The arm: fast last layer (step 0.1) with the first-arrival hold, on 256 seeds from 10000 to 10255, read at 8,388,608 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8707 ± 0.0058 | 0.8874 ± 0.0023 | -0.0168 [-0.0305, -0.0051] | -0.0105 [-0.0224, -0.0024] | 256 |
| fast last layer alone (step 0.1) | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 | 0.5725 ± 0.0106 | 0.0360 ± 0.0012 | +0.5365 [+0.5155, +0.5563] | +0.8078 [+0.7826, +0.8310] | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8707 ± 0.0058 | 0.8781 ± 0.0021 | -0.0074 [-0.0203, +0.0034] | -0.0048 [-0.0169, +0.0034] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 90.7 ± 5.08 | 259 ± 10.6 | -169 [-192, -146] | 256 of 256; 256 of 256 | 256 |
| fast last layer alone (step 0.1) | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 | 86.1 ± 6.37 | 77.7 ± 4.54 | +8.39 [-5.8, +22.8] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 90.7 ± 5.08 | 269 ± 10.8 | -179 [-201, -156] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 0, behind at 1 of 1; $M$ sooner at 1, level at 0, later at 0 of 1; not trained by it: intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01.
- against fast last layer alone (step 0.1): $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1; not trained by it: intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 1, level at 0, later at 0 of 1; not trained by it: intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer (step 0.1) with the first-arrival hold: best $W$ 0.8707 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 86.1 thousand steps at intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 (of 2 configurations).
- divisor held from each policy's first arrival: best $W$ 0.8874 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 201 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).
- fast last layer alone (step 0.1): best $W$ 0.0360 at intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01; best $M$ 55.8 thousand steps at intrinsic weight 16, learning rate 0.006, entropy coefficient 0.03 (of 3 configurations).
- CleanRL PPO-RND: best $W$ 0.8781 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 119 thousand steps at intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (of 8 configurations).

#### PointMaze Large, step cost

The arm: fast last layer (step 0.1) with the first-arrival hold, on 256 seeds from 10000 to 10255, read at 33,554,432 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | -0.7264 ± 0.0016 | -0.7268 ± 0.0025 | +0.0004 [-0.0052, +0.0064] | +0.0084 [+0.0023, +0.0145] | 256 |
| fast last layer alone (step 0.1) | intrinsic weight 8, learning rate 0.0045, entropy coefficient 0.01 | -0.7638 ± 0.0030 | -0.9950 ± 0.0002 | +0.2311 [+0.2251, +0.2365] | +0.2689 [+0.2621, +0.2748] | 256 |
| CleanRL PPO-RND | intrinsic weight 8, learning rate 0.0045, entropy coefficient 0.01 | -0.7638 ± 0.0030 | -0.8831 ± 0.0022 | +0.1192 [+0.1119, +0.1264] | +0.1568 [+0.1484, +0.1646] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | 71.5 ± 4.37 | 189 ± 9.38 | -117 [-138, -96.8] | 256 of 256; 256 of 256 | 256 |
| fast last layer alone (step 0.1) | intrinsic weight 8, learning rate 0.0045, entropy coefficient 0.01 | 78.6 ± 5.91 | 76.4 ± 4.83 | +2.14 [-12, +16.9] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 8, learning rate 0.0045, entropy coefficient 0.01 | 78.6 ± 5.91 | 131 ± 6.45 | -52.1 [-68.6, -34.8] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 1, level at 0, later at 0 of 1; not trained by it: intrinsic weight 8, learning rate 0.0045, entropy coefficient 0.01.
- against fast last layer alone (step 0.1): $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1; not trained by it: intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01.
- against CleanRL PPO-RND: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 1, level at 0, later at 0 of 1; not trained by it: intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer (step 0.1) with the first-arrival hold: best $W$ -0.7264 at intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01; best $M$ 71.5 thousand steps at intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 (of 2 configurations).
- divisor held from each policy's first arrival: best $W$ -0.7236 at intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01; best $M$ 163 thousand steps at intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01 (of 3 configurations).
- fast last layer alone (step 0.1): best $W$ -0.9950 at intrinsic weight 8, learning rate 0.0045, entropy coefficient 0.01; best $M$ 58.9 thousand steps at intrinsic weight 16, learning rate 0.006, entropy coefficient 0.03 (of 3 configurations).
- CleanRL PPO-RND: best $W$ -0.7310 at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01; best $M$ 120 thousand steps at intrinsic weight 8, learning rate 0.006, entropy coefficient 0.01 (of 4 configurations).

### AntMaze U-maze, 128 environments per policy: divisor held from each policy's first arrival against RND

#### AntMaze U-maze, reward 1 on arrival

The arm: divisor held from each policy's first arrival, on 16 seeds from 0 to 15, read at 16,777,216 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.003, entropy coefficient 0.001 | 0.032 ± 0.001 | 0.035 ± 0.001 | -0.003 [-0.005, -0.001] | -0.006 [-0.009, -0.004] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 0.044 ± 0.001 | 0.043 ± 0.001 | +0.001 [-0.002, +0.004] | -0.003 [-0.005, -0.000] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.003, entropy coefficient 0.001 | 0.036 ± 0.001 | 0.038 ± 0.001 | -0.002 [-0.004, +0.001] | -0.003 [-0.006, +0.000] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 0.046 ± 0.001 | 0.046 ± 0.001 | -0.000 [-0.003, +0.003] | -0.001 [-0.005, +0.003] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.003, entropy coefficient 0.001 | 0.038 ± 0.001 | 0.037 ± 0.001 | +0.001 [-0.001, +0.004] | +0.001 [-0.002, +0.004] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 0.048 ± 0.001 | 0.049 ± 0.001 | -0.001 [-0.004, +0.002] | -0.001 [-0.006, +0.004] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.003, entropy coefficient 0.001 | 0.038 ± 0.001 | 0.038 ± 0.001 | -0.000 [-0.003, +0.002] | -0.001 [-0.004, +0.001] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001 | 0.046 ± 0.001 | 0.047 ± 0.001 | -0.001 [-0.003, +0.002] | -0.003 [-0.007, +0.002] | 16 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.003, entropy coefficient 0.001 | 2,795 ± 83.5 | 2,753 ± 95.5 | +42.6 [-129, +207] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 1,840 ± 72.3 | 1,865 ± 77 | -25.4 [-251, +188] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.003, entropy coefficient 0.001 | 2,568 ± 116 | 2,631 ± 99.1 | -63 [-251, +149] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 1,959 ± 108 | 1,981 ± 75.8 | -22.2 [-252, +208] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.003, entropy coefficient 0.001 | 2,597 ± 113 | 2,605 ± 109 | -7.39 [-335, +309] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 1,784 ± 79.6 | 2,112 ± 72.3 | -328 [-549, -101] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.003, entropy coefficient 0.001 | 2,633 ± 84.8 | 2,547 ± 128 | +86 [-202, +376] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001 | 1,999 ± 95.4 | 1,926 ± 86.8 | +73.7 [-105, +258] | 16 of 16; 16 of 16 | 16 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against CleanRL PPO-RND (environments-per-policy check): $W$ ahead at 0, level at 7, behind at 1 of 8; $M$ sooner at 1, level at 7, later at 0 of 8.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- divisor held from each policy's first arrival: best $W$ 0.0475 at intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001; best $M$ 1,784 thousand steps at intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 (of 8 configurations).
- CleanRL PPO-RND (environments-per-policy check): best $W$ 0.0487 at intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001; best $M$ 1,841 thousand steps at intrinsic weight 64, learning rate 0.01, entropy coefficient 0.01 (of 48 configurations).

### AntMaze U-maze, 128 environments per policy: fast last layer (step 0.1) against RND

#### AntMaze U-maze, reward 1 on arrival

The arm: fast last layer (step 0.1), on 16 seeds from 0 to 15, read at 16,777,216 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.003, entropy coefficient 0.001 | 0.019 ± 0.001 | 0.035 ± 0.001 | -0.016 [-0.020, -0.012] | -0.020 [-0.027, -0.013] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 0.029 ± 0.003 | 0.043 ± 0.001 | -0.014 [-0.019, -0.009] | -0.016 [-0.023, -0.010] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.003, entropy coefficient 0.001 | 0.022 ± 0.001 | 0.038 ± 0.001 | -0.016 [-0.019, -0.013] | -0.021 [-0.024, -0.017] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 0.038 ± 0.002 | 0.046 ± 0.001 | -0.008 [-0.012, -0.004] | -0.009 [-0.016, -0.002] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.003, entropy coefficient 0.001 | 0.025 ± 0.001 | 0.037 ± 0.001 | -0.011 [-0.014, -0.009] | -0.014 [-0.017, -0.011] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 0.040 ± 0.002 | 0.049 ± 0.001 | -0.009 [-0.012, -0.006] | -0.007 [-0.012, -0.002] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.003, entropy coefficient 0.001 | 0.026 ± 0.001 | 0.038 ± 0.001 | -0.013 [-0.015, -0.011] | -0.010 [-0.013, -0.007] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001 | 0.042 ± 0.002 | 0.047 ± 0.001 | -0.005 [-0.007, -0.003] | -0.004 [-0.009, +0.002] | 16 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.003, entropy coefficient 0.001 | 4,125 ± 201 | 2,753 ± 95.5 | +1,372 [+898, +1,863] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 3,003 ± 221 | 1,865 ± 77 | +1,138 [+717, +1,602] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.003, entropy coefficient 0.001 | 3,944 ± 201 | 2,631 ± 99.1 | +1,312 [+1,013, +1,603] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 2,147 ± 94.7 | 1,981 ± 75.8 | +166 [-22.4, +371] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.003, entropy coefficient 0.001 | 3,889 ± 151 | 2,605 ± 109 | +1,285 [+929, +1,634] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 2,521 ± 101 | 2,112 ± 72.3 | +409 [+148, +671] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.003, entropy coefficient 0.001 | 3,919 ± 166 | 2,547 ± 128 | +1,372 [+1,000, +1,763] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001 | 2,188 ± 146 | 1,926 ± 86.8 | +262 [-18, +544] | 16 of 16; 16 of 16 | 16 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against CleanRL PPO-RND (environments-per-policy check): $W$ ahead at 0, level at 0, behind at 8 of 8; $M$ sooner at 0, level at 2, later at 6 of 8.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer (step 0.1): best $W$ 0.0416 at intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001; best $M$ 2,147 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 (of 8 configurations).
- CleanRL PPO-RND (environments-per-policy check): best $W$ 0.0487 at intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001; best $M$ 1,841 thousand steps at intrinsic weight 64, learning rate 0.01, entropy coefficient 0.01 (of 48 configurations).

### AntMaze U-maze, 128 environments per policy: fast last layer (step 0.1) with the first-arrival hold against RND

#### AntMaze U-maze, reward 1 on arrival

The arm: fast last layer (step 0.1) with the first-arrival hold, on 16 seeds from 0 to 15, read at 16,777,216 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.003, entropy coefficient 0.001 | 0.018 ± 0.001 | 0.035 ± 0.001 | -0.017 [-0.020, -0.014] | -0.019 [-0.024, -0.015] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 0.026 ± 0.002 | 0.043 ± 0.001 | -0.018 [-0.021, -0.014] | -0.020 [-0.025, -0.015] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.003, entropy coefficient 0.001 | 0.022 ± 0.001 | 0.038 ± 0.001 | -0.017 [-0.020, -0.013] | -0.020 [-0.024, -0.015] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 0.033 ± 0.002 | 0.046 ± 0.001 | -0.014 [-0.020, -0.008] | -0.016 [-0.022, -0.011] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.003, entropy coefficient 0.001 | 0.024 ± 0.001 | 0.037 ± 0.001 | -0.013 [-0.016, -0.011] | -0.016 [-0.020, -0.013] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 0.038 ± 0.001 | 0.049 ± 0.001 | -0.011 [-0.013, -0.008] | -0.013 [-0.017, -0.008] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.003, entropy coefficient 0.001 | 0.025 ± 0.001 | 0.038 ± 0.001 | -0.014 [-0.016, -0.012] | -0.012 [-0.016, -0.008] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001 | 0.038 ± 0.001 | 0.047 ± 0.001 | -0.008 [-0.011, -0.005] | -0.010 [-0.015, -0.005] | 16 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.003, entropy coefficient 0.001 | 4,103 ± 190 | 2,753 ± 95.5 | +1,350 [+901, +1,831] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 2,601 ± 206 | 1,865 ± 77 | +736 [+346, +1,244] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.003, entropy coefficient 0.001 | 3,885 ± 165 | 2,631 ± 99.1 | +1,254 [+855, +1,646] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 2,076 ± 126 | 1,981 ± 75.8 | +95.1 [-183, +397] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.003, entropy coefficient 0.001 | 3,842 ± 180 | 2,605 ± 109 | +1,237 [+840, +1,654] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 2,375 ± 94 | 2,112 ± 72.3 | +263 [+84, +450] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.003, entropy coefficient 0.001 | 3,839 ± 207 | 2,547 ± 128 | +1,292 [+864, +1,717] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001 | 2,323 ± 137 | 1,926 ± 86.8 | +397 [+80.9, +719] | 16 of 16; 16 of 16 | 16 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against CleanRL PPO-RND (environments-per-policy check): $W$ ahead at 0, level at 0, behind at 8 of 8; $M$ sooner at 0, level at 1, later at 7 of 8.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer (step 0.1) with the first-arrival hold: best $W$ 0.0385 at intrinsic weight 64, learning rate 0.01, entropy coefficient 0.001; best $M$ 2,076 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 (of 8 configurations).
- CleanRL PPO-RND (environments-per-policy check): best $W$ 0.0487 at intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001; best $M$ 1,841 thousand steps at intrinsic weight 64, learning rate 0.01, entropy coefficient 0.01 (of 48 configurations).

### AntMaze U-maze, 16 environments per policy: divisor held from each policy's first arrival against RND

#### AntMaze U-maze, reward 1 on arrival

The arm: divisor held from each policy's first arrival, on 16 seeds from 0 to 15, read at 16,777,216 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 0.056 ± 0.003 | 0.049 ± 0.003 | +0.007 [+0.000, +0.014] | +0.017 [+0.007, +0.028] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.01, entropy coefficient 0.001 | 0.023 ± 0.002 | 0.025 ± 0.001 | -0.003 [-0.006, +0.001] | -0.002 [-0.009, +0.004] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 0.033 ± 0.001 | 0.034 ± 0.001 | -0.001 [-0.004, +0.001] | -0.004 [-0.008, -0.000] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 0.033 ± 0.001 | 0.041 ± 0.002 | -0.008 [-0.013, -0.003] | -0.007 [-0.014, +0.001] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 0.036 ± 0.001 | 0.037 ± 0.001 | -0.001 [-0.004, +0.001] | -0.001 [-0.005, +0.003] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 0.036 ± 0.002 | 0.038 ± 0.001 | -0.002 [-0.006, +0.002] | -0.003 [-0.009, +0.004] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 0.037 ± 0.001 | 0.039 ± 0.001 | -0.002 [-0.004, -0.000] | -0.003 [-0.007, +0.001] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 0.038 ± 0.001 | 0.039 ± 0.002 | -0.001 [-0.004, +0.002] | -0.004 [-0.008, +0.001] | 16 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 1,312 ± 62.3 | 1,258 ± 61.2 | +54.3 [-66.7, +163] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.01, entropy coefficient 0.001 | 392 ± 38.1 | 489 ± 46.1 | -96.3 [-188, -10.5] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 799 ± 44.9 | 827 ± 37.4 | -28.3 [-135, +70.6] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 375 ± 41.4 | 338 ± 31.7 | +36.7 [-50.8, +135] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 730 ± 47.7 | 720 ± 41.3 | +9.57 [-90.5, +119] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 298 ± 20.3 | 319 ± 19.8 | -20.9 [-69.4, +31.3] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 765 ± 43.1 | 680 ± 43.2 | +85.3 [+12.1, +161] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 346 ± 23.9 | 349 ± 27 | -2.66 [-62, +51.4] | 16 of 16; 16 of 16 | 16 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against CleanRL PPO-RND (environments-per-policy check): $W$ ahead at 1, level at 5, behind at 2 of 8; $M$ sooner at 1, level at 6, later at 1 of 8.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- divisor held from each policy's first arrival: best $W$ 0.0558 at intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001; best $M$ 298 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 (of 8 configurations).
- CleanRL PPO-RND (environments-per-policy check): best $W$ 0.0487 at intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001; best $M$ 314 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (of 48 configurations).

### AntMaze U-maze, 16 environments per policy: fast last layer (step 0.1) against RND

#### AntMaze U-maze, reward 1 on arrival

The arm: fast last layer (step 0.1), on 16 seeds from 0 to 15, read at 16,777,216 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 0.052 ± 0.002 | 0.049 ± 0.003 | +0.003 [-0.004, +0.010] | +0.001 [-0.012, +0.014] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.01, entropy coefficient 0.001 | 0.035 ± 0.002 | 0.025 ± 0.001 | +0.010 [+0.007, +0.014] | +0.014 [+0.008, +0.021] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 0.043 ± 0.001 | 0.034 ± 0.001 | +0.009 [+0.007, +0.011] | +0.006 [+0.001, +0.011] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 0.044 ± 0.002 | 0.041 ± 0.002 | +0.003 [+0.000, +0.006] | -0.001 [-0.005, +0.003] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 0.045 ± 0.001 | 0.037 ± 0.001 | +0.008 [+0.006, +0.009] | +0.005 [+0.002, +0.009] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 0.041 ± 0.001 | 0.038 ± 0.001 | +0.004 [+0.001, +0.006] | +0.005 [-0.002, +0.013] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 0.046 ± 0.001 | 0.039 ± 0.001 | +0.007 [+0.005, +0.009] | +0.005 [+0.002, +0.009] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 0.041 ± 0.001 | 0.039 ± 0.002 | +0.002 [-0.001, +0.005] | -0.000 [-0.005, +0.004] | 16 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 1,184 ± 56.4 | 1,258 ± 61.2 | -74.1 [-210, +61.5] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.01, entropy coefficient 0.001 | 315 ± 27.4 | 489 ± 46.1 | -174 [-259, -84.2] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 554 ± 23.8 | 827 ± 37.4 | -274 [-368, -179] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 250 ± 17.1 | 338 ± 31.7 | -88.4 [-157, -23.1] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 516 ± 33.3 | 720 ± 41.3 | -205 [-298, -109] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 249 ± 19 | 319 ± 19.8 | -70.1 [-131, -2.59] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 493 ± 20.4 | 680 ± 43.2 | -187 [-279, -107] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 282 ± 20.1 | 349 ± 27 | -67.1 [-124, -13] | 16 of 16; 16 of 16 | 16 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against CleanRL PPO-RND (environments-per-policy check): $W$ ahead at 6, level at 2, behind at 0 of 8; $M$ sooner at 7, level at 1, later at 0 of 8.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer (step 0.1): best $W$ 0.0516 at intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001; best $M$ 249 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 (of 8 configurations).
- CleanRL PPO-RND (environments-per-policy check): best $W$ 0.0487 at intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001; best $M$ 314 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (of 48 configurations).

### AntMaze U-maze, 16 environments per policy: fast last layer (step 0.1) with the first-arrival hold against RND

#### AntMaze U-maze, reward 1 on arrival

The arm: fast last layer (step 0.1) with the first-arrival hold, on 16 seeds from 0 to 15, read at 16,777,216 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 0.054 ± 0.004 | 0.049 ± 0.003 | +0.005 [-0.004, +0.014] | +0.006 [-0.007, +0.020] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.01, entropy coefficient 0.001 | 0.032 ± 0.002 | 0.025 ± 0.001 | +0.007 [+0.003, +0.012] | +0.008 [-0.000, +0.016] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 0.042 ± 0.001 | 0.034 ± 0.001 | +0.008 [+0.005, +0.010] | +0.007 [+0.003, +0.010] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 0.042 ± 0.001 | 0.041 ± 0.002 | +0.002 [-0.002, +0.005] | +0.001 [-0.005, +0.007] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 0.045 ± 0.001 | 0.037 ± 0.001 | +0.007 [+0.006, +0.009] | +0.005 [+0.000, +0.009] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 0.042 ± 0.001 | 0.038 ± 0.001 | +0.004 [+0.002, +0.007] | +0.006 [+0.002, +0.011] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 0.045 ± 0.001 | 0.039 ± 0.001 | +0.006 [+0.004, +0.008] | +0.002 [-0.002, +0.007] | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 0.045 ± 0.002 | 0.039 ± 0.002 | +0.005 [+0.003, +0.008] | +0.003 [-0.003, +0.007] | 16 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 1,165 ± 62.6 | 1,258 ± 61.2 | -93.1 [-222, +31.7] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 0.0625, learning rate 0.01, entropy coefficient 0.001 | 361 ± 39.3 | 489 ± 46.1 | -127 [-246, -11.9] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 570 ± 28.9 | 827 ± 37.4 | -257 [-352, -166] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 1, learning rate 0.01, entropy coefficient 0.001 | 298 ± 31.9 | 338 ± 31.7 | -40.5 [-119, +34.3] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 489 ± 25.8 | 720 ± 41.3 | -231 [-325, -140] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.001 | 276 ± 27.1 | 319 ± 19.8 | -42.5 [-113, +36.1] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 473 ± 32.1 | 680 ± 43.2 | -206 [-322, -99.3] | 16 of 16; 16 of 16 | 16 |
| CleanRL PPO-RND (environments-per-policy check) | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 | 251 ± 25.7 | 349 ± 27 | -98.3 [-167, -32.4] | 16 of 16; 16 of 16 | 16 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against CleanRL PPO-RND (environments-per-policy check): $W$ ahead at 6, level at 2, behind at 0 of 8; $M$ sooner at 5, level at 3, later at 0 of 8.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer (step 0.1) with the first-arrival hold: best $W$ 0.0538 at intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001; best $M$ 251 thousand steps at intrinsic weight 16, learning rate 0.01, entropy coefficient 0.001 (of 8 configurations).
- CleanRL PPO-RND (environments-per-policy check): best $W$ 0.0487 at intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001; best $M$ 314 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (of 48 configurations).

## Predictions

The predictions were written in `ideas.md` before the runs. Each is quoted, then read part by part: a part held or did not hold on the numbers given, or cannot be read yet because a unit it needs is not complete. A prediction held when every part held, and did not hold when at least one part did not. Where a prediction leaves a choice, it is read as follows. On PointMaze every comparison is paired on the first validation block (seeds 10000 to 10255) at one configuration; "RND's best" is RND at its best configuration for that metric among all those run on the same seeds. "Equals within its interval" and "level" are read as: the 95% interval of the paired difference holds zero; "above with the interval above zero", "behind" and "lowers the first-goal time" as: the interval is entirely on that side; "within 0.005" and "behind by more than 0.01" are read on the mean difference. "Ahead of RND's best on both metrics" is read as: ahead on $W$ against RND's best for $W$ and ahead on $M$ against RND's best for $M$, each with the interval on its side of zero. On the ant (16 development seeds) "above" or "below RND at the same configuration" counts the configurations by the sign of the mean paired difference; "the whole-curve mean is not ahead" is read as: above RND in fewer than 6 of the 8 configurations; "between the two single arms" as: at the same configuration, between the fast last layer's and the hold's $W$; "as rare as in the check" as: at most twice the check's respawns per environment step at the same number of environments per policy.

### Prediction 1: PointMaze, discovery unchanged by the hold

> **PointMaze, discovery unchanged by the hold.** At the same configuration and seeds the combined arm's first-goal time equals the fast last layer's alone within its interval, because the two are the same program until a policy first arrives (13.8, 11.7, 77.7 and 76.4 thousand steps at the four second configurations).

Observed: **did not hold**.

- Medium with reward 1 on arrival, second configuration: $M$ equals the fast last layer's alone within the interval: **held**. At (learning rate 0.01, weight 4): $M$ 14.7 thousand steps against 13.8 thousand steps; difference +0.887 [-1.07, +2.93] thousand steps.
- Medium with the step cost, second configuration: $M$ equals the fast last layer's alone within the interval: **did not hold**. At (learning rate 0.01, weight 4): $M$ 15 thousand steps against 11.7 thousand steps; difference +3.34 [+0.353, +7.34] thousand steps.
- Large with reward 1 on arrival, second configuration: $M$ equals the fast last layer's alone within the interval: **held**. At (learning rate 0.002, weight 16): $M$ 86.1 thousand steps against 77.7 thousand steps; difference +8.39 [-5.8, +22.8] thousand steps.
- Large with the step cost, second configuration: $M$ equals the fast last layer's alone within the interval: **held**. At (learning rate 0.0045, weight 8): $M$ 78.6 thousand steps against 76.4 thousand steps; difference +2.14 [-12, +16.9] thousand steps.

### Prediction 2: PointMaze, the hold gives back part of what the fast last layer costs

> **PointMaze, the hold gives back part of what the fast last layer costs.** At the four second configurations the whole-curve mean is above the fast last layer's alone with the interval above zero (0.379 on Medium with reward 1 on arrival, −0.586 on Medium with the step cost, 0.036 on Large with reward 1 on arrival), and still behind RND's best in the two Medium cases.

Observed: **held**.

- Medium with reward 1 on arrival, second configuration: $W$ above the fast last layer's alone, interval above zero: **held**. At (learning rate 0.01, weight 4): $W$ 0.7759 against 0.3785; difference +0.3974 [+0.3775, +0.4169].
- Medium with the step cost, second configuration: $W$ above the fast last layer's alone, interval above zero: **held**. At (learning rate 0.01, weight 4): $W$ -0.4057 against -0.5857; difference +0.1800 [+0.1696, +0.1909].
- Large with reward 1 on arrival, second configuration: $W$ above the fast last layer's alone, interval above zero: **held**. At (learning rate 0.002, weight 16): $W$ 0.5725 against 0.0360; difference +0.5365 [+0.5155, +0.5563].
- Large with the step cost, second configuration: $W$ above the fast last layer's alone, interval above zero: **held**. At (learning rate 0.0045, weight 8): $W$ -0.7638 against -0.9950; difference +0.2311 [+0.2251, +0.2365].
- Medium with reward 1 on arrival, second configuration: still behind RND's best: **held**. At (learning rate 0.01, weight 4): $W$ 0.7759 against RND's best 0.9349; difference -0.1590 [-0.1758, -0.1430].
- Medium with the step cost, second configuration: still behind RND's best: **held**. At (learning rate 0.01, weight 4): $W$ -0.4057 against RND's best -0.3505; difference -0.0552 [-0.0629, -0.0479].

### Prediction 3: PointMaze, at the hold's own best configurations

> **PointMaze, at the hold's own best configurations.** Adding the fast last layer lowers the first-goal time against the hold alone on both Large cases (interval below zero) and does not raise the whole-curve mean: level on Large (within 0.005), behind by more than 0.01 on Medium with reward 1 on arrival, as the three-part arm was (−0.029). So no configuration of this batch is ahead of RND's best on both metrics in a Medium case.

Observed: **did not hold**.

- Large with reward 1 on arrival: the first-goal time is lower than the hold alone's, interval below zero: **held**. At (learning rate 0.002, weight 2): $M$ 90.7 thousand steps against 259 thousand steps; difference -169 [-192, -146] thousand steps.
- Large with the step cost: the first-goal time is lower than the hold alone's, interval below zero: **held**. At (learning rate 0.002, weight 8): $M$ 71.5 thousand steps against 189 thousand steps; difference -117 [-138, -96.8] thousand steps.
- Large with reward 1 on arrival: $W$ level with the hold alone (within 0.005): **did not hold**. At (learning rate 0.002, weight 2): $W$ 0.8707 against 0.8874; difference -0.0168 [-0.0305, -0.0051].
- Large with the step cost: $W$ level with the hold alone (within 0.005): **held**. At (learning rate 0.002, weight 8): $W$ -0.7264 against -0.7268; difference +0.0004 [-0.0052, +0.0064].
- Medium with reward 1 on arrival: $W$ behind the hold alone by more than 0.01: **held**. At (learning rate 0.006, weight 4): $W$ 0.8424 against 0.9435; difference -0.1010 [-0.1148, -0.0883].
- Medium with reward 1 on arrival: no configuration ahead of RND's best on both metrics: **held**. Against RND's best for each metric: (learning rate 0.006, weight 4): $W$ -0.0925 [-0.1069, -0.0787], $M$ -1.83 [-5.24, +1.1] thousand steps; (learning rate 0.01, weight 4): $W$ -0.1590 [-0.1758, -0.1430], $M$ -2.85 [-6.33, +0.255] thousand steps.
- Medium with the step cost: no configuration ahead of RND's best on both metrics: **held**. Against RND's best for each metric: (learning rate 0.006, weight 2): $W$ -0.0187 [-0.0257, -0.0121], $M$ +0.523 [-0.948, +2.04] thousand steps; (learning rate 0.01, weight 4): $W$ -0.0552 [-0.0629, -0.0479], $M$ +2.63 [-0.215, +6.52] thousand steps.

### Prediction 4: ant, the first-arrival hold

> **Ant, the first-arrival hold.** Its whole-curve mean is above RND's at the same configuration in at least 6 of the 8 configurations at each environment count, its best is at least 0.06 (RND's best: 0.0487), and its final level rises more than its whole-curve mean (at least 0.09 against 0.065). Reason: RND's divisor keeps the intrinsic reward at the same scale for the whole run, which pays the ant for throwing itself about; held, the bonus falls as the predictor learns. I am least sure of this one: the low return may come from the learner needing more steps, whatever the bonus does.

Observed: **did not hold**.

- 128 environments per policy: $W$ above RND at the same configuration in at least 6 of 8: **did not hold**. $W$ difference from RND: above zero in 2 and below zero in 6 of 8 configurations (from -0.0030 to +0.0013).
- 128 environments per policy: its best $W$ is at least 0.06: **did not hold**. At its best configuration for $W$ (learning rate 0.01, weight 16): $W$ 0.0475, $L$ 0.0637; RND's best $W$ over its 48 configurations of the check 0.0487.
- 128 environments per policy: its final level at that configuration is at least 0.09: **did not hold**. At its best configuration for $W$ (learning rate 0.01, weight 16): $W$ 0.0475, $L$ 0.0637; RND's best $W$ over its 48 configurations of the check 0.0487.
- 16 environments per policy: $W$ above RND at the same configuration in at least 6 of 8: **did not hold**. $W$ difference from RND: above zero in 1 and below zero in 7 of 8 configurations (from -0.0079 to +0.0071).
- 16 environments per policy: its best $W$ is at least 0.06: **did not hold**. At its best configuration for $W$ (learning rate 0.0003, weight 0.0625): $W$ 0.0558, $L$ 0.0807; RND's best $W$ over its 48 configurations of the check 0.0487.
- 16 environments per policy: its final level at that configuration is at least 0.09: **did not hold**. At its best configuration for $W$ (learning rate 0.0003, weight 0.0625): $W$ 0.0558, $L$ 0.0807; RND's best $W$ over its 48 configurations of the check 0.0487.

### Prediction 5: ant, the fast last layer

> **Ant, the fast last layer.** Its first-goal time is below RND's at the same configuration in at least 6 of 8 configurations at each count, and its whole-curve mean is not ahead.

Observed: **did not hold**.

- 128 environments per policy: $M$ below RND at the same configuration in at least 6 of 8: **did not hold**. $M$ difference from RND: above zero in 8 and below zero in 0 of 8 configurations (from +166 to +1,372 thousand steps).
- 128 environments per policy: $W$ not ahead (above RND in fewer than 6 of 8): **held**. $W$ difference from RND: above zero in 0 and below zero in 8 of 8 configurations (from -0.0161 to -0.0052).
- 16 environments per policy: $M$ below RND at the same configuration in at least 6 of 8: **held**. $M$ difference from RND: above zero in 0 and below zero in 8 of 8 configurations (from -274 to -67 thousand steps).
- 16 environments per policy: $W$ not ahead (above RND in fewer than 6 of 8): **did not hold**. $W$ difference from RND: above zero in 8 and below zero in 0 of 8 configurations (from +0.0019 to +0.0103).

### Prediction 6: ant, both together

> **Ant, both together.** First-goal time as the fast last layer's; whole-curve mean between the two single arms.

Observed: **did not hold**.

- 128 environments per policy: $M$ as the fast last layer's (in at least 6 of 8 configurations): **held**. the interval of the paired difference in $M$ from the fast last layer alone holds zero in 8 of 8 configurations.
- 128 environments per policy: $W$ between the two single arms (in at least 6 of 8 configurations): **did not hold**. $W$ lies between the fast last layer's and the hold's in 0 of 8 configurations.
- 16 environments per policy: $M$ as the fast last layer's (in at least 6 of 8 configurations): **held**. the interval of the paired difference in $M$ from the fast last layer alone holds zero in 8 of 8 configurations.
- 16 environments per policy: $W$ between the two single arms (in at least 6 of 8 configurations): **held**. $W$ lies between the fast last layer's and the hold's in 6 of 8 configurations.

### Prediction 7: ant, the physics guard

> **Ant, the physics guard.** Respawns by the fault guard stay as rare as in the check (32 and 43 in about 13 billion steps per environment count).

Observed: **did not hold**.

- 128 environments per policy: respawns as rare as in the check: **did not hold**. The check: 32 in 12.9 billion steps. The screen: divisor held from each policy's first arrival 12 in 2.1 billion steps; fast last layer (step 0.1) 5 in 2.1 billion steps; fast last layer (step 0.1) with the first-arrival hold 7 in 2.1 billion steps.
- 16 environments per policy: respawns as rare as in the check: **did not hold**. The check: 43 in 12.9 billion steps. The screen: divisor held from each policy's first arrival 30 in 2.1 billion steps; fast last layer (step 0.1) 32 in 2.1 billion steps; fast last layer (step 0.1) with the first-arrival hold 52 in 2.1 billion steps.

