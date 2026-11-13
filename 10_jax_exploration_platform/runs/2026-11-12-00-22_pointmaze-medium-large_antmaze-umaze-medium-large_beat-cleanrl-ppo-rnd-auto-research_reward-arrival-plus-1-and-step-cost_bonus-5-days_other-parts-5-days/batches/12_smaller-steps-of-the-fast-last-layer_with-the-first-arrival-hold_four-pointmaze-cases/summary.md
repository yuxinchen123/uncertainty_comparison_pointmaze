# Batch 12: smaller steps of the fast last layer, with the first-arrival hold

Generated 2026-11-13 11:35 PT by the command below. This file is generated: regenerate it, do not edit it. Times are Pacific, converted from machines running Eastern. Run folder: `2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days`; batch folder: `batches/12_smaller-steps-of-the-fast-last-layer_with-the-first-arrival-hold_four-pointmaze-cases`.

```bash
cd /p/rlprojects/RND/10_jax_exploration_platform
bash "benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh" -s scripts/maze_campaign/analysis/batch_summary_job.py \
    --run-dir=runs/2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days \
    --batch=12_smaller-steps-of-the-fast-last-layer_with-the-first-arrival-hold_four-pointmaze-cases
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
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9194 ± 0.0034 | 0.9965 ± 0.0005 | 17.3 ± 0.903 | 256 of 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9400 ± 0.0031 | 0.9993 ± 0.0002 | 18.5 ± 1 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.8424 ± 0.0063 | 0.9747 ± 0.0037 | <u>15.7</u> ± 0.84 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 0.7759 ± 0.0077 | 0.9490 ± 0.0060 | **14.7** ± 0.957 | 256 of 256 |
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
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (the only configuration read) | 0.9194 ± 0.0034 | -0.0155 [-0.0242, -0.0069] | 0.9965 ± 0.0005 | +0.0015 [-0.0004, +0.0037] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 2) | 0.8424 ± 0.0063 | -0.0925 [-0.1068, -0.0788] | 0.9747 ± 0.0037 | -0.0203 [-0.0285, -0.0134] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (best of 6) | 17.5 ± 1.41 | control | 256 of 256 | 0.0073 | 0.0000 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (best of 2) | **14.7** ± 0.957 | **-2.85** [-6.45, +0.268] | 256 of 256 | 0.7759 | 0.9490 | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (the only configuration read) | <u>17.3</u> ± 0.903 | <u>-0.214</u> [-3.71, +2.76] | 256 of 256 | 0.9194 | 0.9965 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | 18.5 ± 1.13 | +0.993 [-2.74, +4.35] | 256 of 256 | **0.9435** | **0.9999** | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (the only configuration read) | 18.5 ± 1 | +0.995 [-2.46, +4.1] | 256 of 256 | <u>0.9400</u> | <u>0.9993</u> | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:14 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:15 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:14 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 5 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:53 PT | none |
| 6 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 7 | fast last layer at step 0.03 with the first-arrival hold | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:15 PT | none |
| 8 | fast last layer at step 0.01 with the first-arrival hold | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:15 PT | none |
| 9 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 03:06 PT | none |
| 10 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 03:08 PT | none |
| 11 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:32 PT | none |
| 12 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:32 PT | none |
| 13 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:34 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.1_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-step-0-point-03-with-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-step-0-point-01-with-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`

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
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3491 ± 0.0014 | -0.3067 ± 0.0003 | 14 ± 0.628 | 256 of 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | **-0.3470** ± 0.0013 | <u>-0.3066</u> ± 0.0006 | 13.8 ± 0.588 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3692 ± 0.0030 | -0.3132 ± 0.0008 | <u>12.9</u> ± 0.654 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | -0.4057 ± 0.0034 | -0.3299 ± 0.0015 | 15 ± 1.74 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 | <u>-0.3476</u> ± 0.0014 | -0.3073 ± 0.0007 | 15.4 ± 0.693 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3484 ± 0.0014 | **-0.3058** ± 0.0004 | 14.2 ± 0.617 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | -0.3504 ± 0.0015 | <u>-0.3066</u> ± 0.0005 | 13.4 ± 0.562 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01 (best of 6) | -0.3505 ± 0.0016 | control | -0.3078 ± 0.0007 | control | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (the only configuration read) | **-0.3470** ± 0.0013 | **+0.0035** [-0.0006, +0.0077] | **-0.3066** ± 0.0006 | **+0.0012** [-0.0005, +0.0029] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 (best of 3) | <u>-0.3476</u> ± 0.0014 | <u>+0.0029</u> [-0.0015, +0.0072] | -0.3073 ± 0.0007 | +0.0005 [-0.0014, +0.0025] | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (the only configuration read) | -0.3491 ± 0.0014 | +0.0014 [-0.0027, +0.0055] | <u>-0.3067</u> ± 0.0003 | <u>+0.0011</u> [-0.0003, +0.0026] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 2) | -0.3692 ± 0.0030 | -0.0187 [-0.0258, -0.0120] | -0.3132 ± 0.0008 | -0.0054 [-0.0076, -0.0034] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (best of 6) | **12.4** ± 0.471 | control | 256 of 256 | -0.5526 | -0.6884 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 2) | <u>12.9</u> ± 0.654 | **+0.523** [-0.938, +2.07] | 256 of 256 | -0.3692 | -0.3132 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | 13.4 ± 0.562 | <u>+0.958</u> [-0.349, +2.26] | 256 of 256 | -0.3504 | **-0.3066** | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (the only configuration read) | 13.8 ± 0.588 | +1.4 [+0.0507, +2.77] | 256 of 256 | **-0.3470** | **-0.3066** | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (the only configuration read) | 14 ± 0.628 | +1.64 [+0.331, +2.98] | 256 of 256 | <u>-0.3491</u> | <u>-0.3067</u> | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:16 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:19 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:18 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:58 PT | none |
| 5 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:53 PT | none |
| 6 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:54 PT | none |
| 7 | fast last layer at step 0.03 with the first-arrival hold | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:15 PT | none |
| 8 | fast last layer at step 0.01 with the first-arrival hold | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:19 PT | none |
| 9 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:11 PT | none |
| 10 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:13 PT | none |
| 11 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:35 PT | none |
| 12 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:35 PT | none |
| 13 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:34 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.0078125_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-medium_reward-step-cost_fast-step-0-point-03-with-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-step-cost_fast-step-0-point-01-with-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-medium_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-medium_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

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
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | **0.8976** ± 0.0021 | 0.9967 ± 0.0003 | 128 ± 6.96 | **256 of 256** |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | <u>0.8928</u> ± 0.0025 | 0.9971 ± 0.0003 | 180 ± 8.38 | **256 of 256** |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8707 ± 0.0058 | 0.9872 ± 0.0055 | <u>90.7</u> ± 5.08 | **256 of 256** |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 | 0.5725 ± 0.0106 | 0.8495 ± 0.0123 | **86.1** ± 6.37 | **256 of 256** |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8874 ± 0.0023 | <u>0.9977</u> ± 0.0002 | 259 ± 10.6 | **256 of 256** |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.8852 ± 0.0023 | 0.9976 ± 0.0003 | 201 ± 9.11 | **256 of 256** |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 0.8680 ± 0.0033 | **0.9979** ± 0.0003 | 375 ± 12.4 | **256 of 256** |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 8) | 0.8781 ± 0.0021 | control | 0.9920 ± 0.0005 | control | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (the only configuration read) | **0.8976** ± 0.0021 | **+0.0195** [+0.0136, +0.0256] | 0.9967 ± 0.0003 | +0.0047 [+0.0035, +0.0060] | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (the only configuration read) | <u>0.8928</u> ± 0.0025 | <u>+0.0148</u> [+0.0084, +0.0213] | <u>0.9971</u> ± 0.0003 | <u>+0.0051</u> [+0.0040, +0.0062] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 3) | 0.8874 ± 0.0023 | +0.0094 [+0.0032, +0.0156] | **0.9977** ± 0.0002 | **+0.0057** [+0.0046, +0.0068] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 2) | 0.8707 ± 0.0058 | -0.0074 [-0.0203, +0.0037] | 0.9872 ± 0.0055 | -0.0048 [-0.0169, +0.0034] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (best of 8) | <u>119</u> ± 5.46 | control | 256 of 256 | 0.0054 | 0.0000 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 (best of 2) | **86.1** ± 6.37 | **-33.2** [-49, -16.5] | 256 of 256 | 0.5725 | 0.8495 | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (the only configuration read) | 128 ± 6.96 | <u>+8.91</u> [-8.23, +25.5] | 256 of 256 | **0.8976** | 0.9967 | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (the only configuration read) | 180 ± 8.38 | +60.7 [+41.2, +80.8] | 256 of 256 | <u>0.8928</u> | <u>0.9971</u> | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (best of 3) | 201 ± 9.11 | +81.6 [+61.3, +102] | 256 of 256 | 0.8852 | **0.9976** | 256 |

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
| 9 | fast last layer at step 0.03 with the first-arrival hold | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:35 PT | none |
| 10 | fast last layer at step 0.01 with the first-arrival hold | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:08 PT | 1 continued from a checkpoint |
| 11 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 03:44 PT | none |
| 12 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:46 PT | none |
| 13 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:55 PT | none |
| 14 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:55 PT | none |
| 15 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:07 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.001_entropy-coefficient-0.03_intrinsic-weight-256.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.001_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-step-0-point-03-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-step-0-point-01-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
14. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
15. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phases `validation-block-two` and `discovery-block-two` in PointMaze Large, reward 1 on arrival. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 8,388,608 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 | 0.8842 ± 0.0019 | 0.9927 ± 0.0005 | 266 ± 10.6 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006 | 0.0057 ± 0.0002 | 0.0003 ± 0.0002 | <u>130</u> ± 5.17 | 256 of 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 2, learning rate 0.002 | **0.8995** ± 0.0022 | 0.9968 ± 0.0004 | **118** ± 6.21 | 256 of 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 2, learning rate 0.002 | <u>0.8945</u> ± 0.0023 | <u>0.9975</u> ± 0.0002 | 180 ± 8.79 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002 | 0.8839 ± 0.0025 | **0.9976** ± 0.0003 | 266 ± 11.6 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 (its best on the first block and on this block) | 0.8842 ± 0.0019 | control | control | 0.9927 ± 0.0005 | control | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 2, learning rate 0.002 (its best on the first block) | **0.8995** ± 0.0022 | **+0.0153** [+0.0096, +0.0209] | **+0.0153** [+0.0096, +0.0209] | 0.9968 ± 0.0004 | +0.0041 [+0.0029, +0.0053] | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 2, learning rate 0.002 (its best on the first block) | <u>0.8945</u> ± 0.0023 | <u>+0.0103</u> [+0.0050, +0.0157] | <u>+0.0103</u> [+0.0050, +0.0157] | <u>0.9975</u> ± 0.0002 | <u>+0.0048</u> [+0.0037, +0.0058] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002 (its best on the first block) | 0.8839 ± 0.0025 | -0.0003 [-0.0066, +0.0061] | -0.0003 [-0.0066, +0.0061] | **0.9976** ± 0.0003 | **+0.0049** [+0.0038, +0.0059] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006 (its best on the first block and on this block) | <u>130</u> ± 5.17 | control | control | 256 of 256 | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 2, learning rate 0.002 (its best on the first block) | **118** ± 6.21 | **-12.1** [-27.7, +4.01] | **-12.1** [-27.7, +4.01] | 256 of 256 | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 2, learning rate 0.002 (its best on the first block) | 180 ± 8.79 | <u>+50</u> [+31.1, +69.2] | <u>+50</u> [+31.1, +69.2] | 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002 (its best on this block: its first-block best, intrinsic weight 4, learning rate 0.002, was not run here) | 266 ± 11.6 | +136 [+112, +161] | +136 [+112, +161] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 11:13 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:45 PT | none |
| 3 | fast last layer at step 0.03 with the first-arrival hold | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:19 PT | none |
| 4 | fast last layer at step 0.01 with the first-arrival hold | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:17 PT | none |
| 5 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:53 PT | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-large_reward-arrival-plus-one_fast-step-0-point-03-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-large_reward-arrival-plus-one_fast-step-0-point-01-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
5. `validation-block-two_pointmaze-large_reward-arrival-plus-one_held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`

### PointMaze Large, step cost

#### First block: each side at its best configuration there

The units of the phases `validation-block` and `discovery-validation` in PointMaze Large, step cost. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 33,554,432 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 | -0.7310 ± 0.0018 | -0.7163 ± 0.0017 | 191 ± 10.1 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 | -0.7411 ± 0.0036 | -0.7229 ± 0.0033 | 241 ± 13.4 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.0045 | -0.8831 ± 0.0022 | -0.8849 ± 0.0026 | 131 ± 6.45 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006 | -0.9669 ± 0.0017 | -0.9803 ± 0.0022 | 120 ± 5.6 | 256 of 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 8, learning rate 0.002 | **-0.7223** ± 0.0018 | <u>-0.7064</u> ± 0.0018 | 83.3 ± 4.26 | 256 of 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 8, learning rate 0.002 | -0.7250 ± 0.0023 | -0.7094 ± 0.0023 | 123 ± 7.1 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002 | -0.7264 ± 0.0016 | **-0.7031** ± 0.0018 | **71.5** ± 4.37 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.0045 | -0.7638 ± 0.0030 | -0.7281 ± 0.0032 | <u>78.6</u> ± 5.91 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 | <u>-0.7236</u> ± 0.0021 | -0.7079 ± 0.0021 | 163 ± 8.72 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.002 | -0.7268 ± 0.0025 | -0.7115 ± 0.0025 | 189 ± 9.38 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002 | -0.7335 ± 0.0034 | -0.7198 ± 0.0035 | 206 ± 11.5 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 (best of 4) | -0.7310 ± 0.0018 | control | -0.7163 ± 0.0017 | control | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (the only configuration read) | **-0.7223** ± 0.0018 | **+0.0087** [+0.0039, +0.0135] | <u>-0.7064</u> ± 0.0018 | <u>+0.0100</u> [+0.0054, +0.0145] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 (best of 3) | <u>-0.7236</u> ± 0.0021 | <u>+0.0074</u> [+0.0018, +0.0129] | -0.7079 ± 0.0021 | +0.0084 [+0.0029, +0.0137] | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (the only configuration read) | -0.7250 ± 0.0023 | +0.0060 [+0.0003, +0.0114] | -0.7094 ± 0.0023 | +0.0069 [+0.0015, +0.0124] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (best of 2) | -0.7264 ± 0.0016 | +0.0046 [-0.0003, +0.0094] | **-0.7031** ± 0.0018 | **+0.0132** [+0.0081, +0.0182] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006 (best of 4) | 120 ± 5.6 | control | 256 of 256 | -0.9669 | -0.9803 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (best of 2) | **71.5** ± 4.37 | **-48.8** [-62.6, -34.3] | 256 of 256 | -0.7264 | **-0.7031** | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (the only configuration read) | <u>83.3</u> ± 4.26 | <u>-36.9</u> [-50.9, -23.1] | 256 of 256 | **-0.7223** | <u>-0.7064</u> | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (the only configuration read) | 123 ± 7.1 | +2.35 [-14.5, +20] | 256 of 256 | -0.7250 | -0.7094 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 (best of 3) | 163 ± 8.72 | +42.9 [+24.2, +62.7] | 256 of 256 | <u>-0.7236</u> | -0.7079 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:48 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:47 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:25 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:03 PT | none |
| 5 | fast last layer at step 0.03 with the first-arrival hold | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 07:16 PT | 1 continued from a checkpoint |
| 6 | fast last layer at step 0.01 with the first-arrival hold | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 06:49 PT | 1 continued from a checkpoint |
| 7 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:16 PT | 1 continued from a checkpoint |
| 8 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:30 PT | 1 continued from a checkpoint |
| 9 | divisor held from each policy's first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:32 PT | 1 continued from a checkpoint |
| 10 | divisor held from each policy's first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:04 PT | none |
| 11 | divisor held from each policy's first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:05 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.0045_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `validation-block_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-large_reward-step-cost_fast-step-0-point-03-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-large_reward-step-cost_fast-step-0-point-01-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-large_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-large_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.0045_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-large_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-large_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-large_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phases `validation-block-two` and `discovery-block-two` in PointMaze Large, step cost. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 33,554,432 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 | -0.7328 ± 0.0020 | -0.7175 ± 0.0021 | 203 ± 10.8 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006 | -0.9636 ± 0.0020 | -0.9753 ± 0.0025 | 121 ± 5.71 | 256 of 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 8, learning rate 0.002 | **-0.7216** ± 0.0016 | <u>-0.7056</u> ± 0.0016 | <u>89.3</u> ± 4.71 | 256 of 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 8, learning rate 0.002 | <u>-0.7233</u> ± 0.0021 | -0.7079 ± 0.0021 | 129 ± 6.56 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002 | -0.7245 ± 0.0007 | **-0.7021** ± 0.0009 | **76.1** ± 7.8 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 | -0.7255 ± 0.0023 | -0.7101 ± 0.0023 | 152 ± 7.71 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 (its best on the first block and on this block) | -0.7328 ± 0.0020 | control | control | -0.7175 ± 0.0021 | control | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (its best on the first block) | **-0.7216** ± 0.0016 | **+0.0112** [+0.0066, +0.0160] | **+0.0112** [+0.0066, +0.0160] | <u>-0.7056</u> ± 0.0016 | <u>+0.0119</u> [+0.0069, +0.0171] | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (its best on the first block) | <u>-0.7233</u> ± 0.0021 | <u>+0.0095</u> [+0.0039, +0.0149] | <u>+0.0095</u> [+0.0039, +0.0149] | -0.7079 ± 0.0021 | +0.0096 [+0.0042, +0.0153] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (its best on the first block) | -0.7245 ± 0.0007 | +0.0083 [+0.0044, +0.0128] | +0.0083 [+0.0044, +0.0128] | **-0.7021** ± 0.0009 | **+0.0154** [+0.0110, +0.0202] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 (its best on the first block) | -0.7255 ± 0.0023 | +0.0072 [+0.0013, +0.0131] | +0.0072 [+0.0013, +0.0131] | -0.7101 ± 0.0023 | +0.0074 [+0.0013, +0.0135] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006 (its best on the first block and on this block) | 121 ± 5.71 | control | control | 256 of 256 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (its best on the first block) | **76.1** ± 7.8 | **-44.8** [-62.5, -24.7] | **-44.8** [-62.5, -24.7] | 256 of 256 | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (its best on the first block) | <u>89.3</u> ± 4.71 | <u>-31.6</u> [-45.9, -17.4] | <u>-31.6</u> [-45.9, -17.4] | 256 of 256 | 256 |
| fast last layer at step 0.01 with the first-arrival hold | intrinsic weight 8, learning rate 0.002 (its best on the first block) | 129 ± 6.56 | +7.72 [-9.14, +24.5] | +7.72 [-9.14, +24.5] | 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 (its best on the first block) | 152 ± 7.71 | +31.5 [+11.9, +52.1] | +31.5 [+11.9, +52.1] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 11:32 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 04:00 PT | none |
| 3 | fast last layer at step 0.03 with the first-arrival hold | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 08:43 PT | none |
| 4 | fast last layer at step 0.01 with the first-arrival hold | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 08:44 PT | none |
| 5 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 08:43 PT | none |
| 6 | divisor held from each policy's first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 04:25 PT | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-large_reward-step-cost_fast-step-0-point-03-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-large_reward-step-cost_fast-step-0-point-01-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`
5. `validation-block-two_pointmaze-large_reward-step-cost_fast-step-0-point-1-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`
6. `validation-block-two_pointmaze-large_reward-step-cost_held-from-first-arrival_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`

## Arms at the same configuration, on the same seeds

Each group compares one arm with other sides at the configurations both trained, matched on the intrinsic weight, learning rate, entropy coefficient and clip coefficient, and paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Define $W$, $L$ and $M$ as in the section Measures. No configuration is chosen here.

### Fast last layer at step 0.03 with the first-arrival hold, first validation block

#### PointMaze Medium, reward 1 on arrival

The arm: fast last layer at step 0.03 with the first-arrival hold, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9194 ± 0.0034 | 0.9435 ± 0.0027 | -0.0241 [-0.0319, -0.0159] | -0.0035 [-0.0045, -0.0025] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9194 ± 0.0034 | 0.8424 ± 0.0063 | +0.0770 [+0.0633, +0.0911] | +0.0217 [+0.0152, +0.0297] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9194 ± 0.0034 | 0.9194 ± 0.0023 | -0.0000 [-0.0073, +0.0072] | +0.0170 [+0.0127, +0.0222] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 17.3 ± 0.903 | 18.5 ± 1.13 | -1.21 [-3.91, +1.58] | 256 of 256; 256 of 256 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 17.3 ± 0.903 | 15.7 ± 0.84 | +1.62 [-0.787, +3.98] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 17.3 ± 0.903 | 19.9 ± 1.16 | -2.61 [-5.2, -0.0568] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 0, behind at 1 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against fast last layer (step 0.1) with the first-arrival hold: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 1, level at 0, later at 0 of 1.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer at step 0.03 with the first-arrival hold: best $W$ 0.9194 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 17.3 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 1 configuration).
- divisor held from each policy's first arrival: best $W$ 0.9435 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 18.5 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 3 configurations).
- fast last layer (step 0.1) with the first-arrival hold: best $W$ 0.8424 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 14.7 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ 0.9349 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 17.5 thousand steps at intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (of 6 configurations).

#### PointMaze Medium, step cost

The arm: fast last layer at step 0.03 with the first-arrival hold, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3491 ± 0.0014 | -0.3484 ± 0.0014 | -0.0008 [-0.0047, +0.0032] | -0.0009 [-0.0018, +0.0000] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3491 ± 0.0014 | -0.3692 ± 0.0030 | +0.0200 [+0.0142, +0.0265] | +0.0065 [+0.0051, +0.0084] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 14 ± 0.628 | 14.2 ± 0.617 | -0.197 [-1.68, +1.26] | 256 of 256; 256 of 256 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 14 ± 0.628 | 12.9 ± 0.654 | +1.12 [-0.458, +2.63] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against fast last layer (step 0.1) with the first-arrival hold: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 0 of 0; $M$ sooner at 0, level at 0, later at 0 of 0; not trained by it: intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer at step 0.03 with the first-arrival hold: best $W$ -0.3491 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 14 thousand steps at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (of 1 configuration).
- divisor held from each policy's first arrival: best $W$ -0.3476 at intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01; best $M$ 13.4 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 3 configurations).
- fast last layer (step 0.1) with the first-arrival hold: best $W$ -0.3692 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 12.9 thousand steps at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ -0.3505 at intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01; best $M$ 12.4 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (of 6 configurations).

#### PointMaze Large, reward 1 on arrival

The arm: fast last layer at step 0.03 with the first-arrival hold, on 256 seeds from 10000 to 10255, read at 8,388,608 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8976 ± 0.0021 | 0.8874 ± 0.0023 | +0.0101 [+0.0040, +0.0162] | -0.0009 [-0.0018, -0.0002] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8976 ± 0.0021 | 0.8707 ± 0.0058 | +0.0269 [+0.0159, +0.0403] | +0.0096 [+0.0013, +0.0215] | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8976 ± 0.0021 | 0.8781 ± 0.0021 | +0.0195 [+0.0136, +0.0255] | +0.0047 [+0.0035, +0.0059] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 128 ± 6.96 | 259 ± 10.6 | -131 [-155, -108] | 256 of 256; 256 of 256 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 128 ± 6.96 | 90.7 ± 5.08 | +37.4 [+20.9, +54] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 128 ± 6.96 | 269 ± 10.8 | -141 [-166, -117] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 1, level at 0, later at 0 of 1.
- against fast last layer (step 0.1) with the first-arrival hold: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 0, later at 1 of 1.
- against CleanRL PPO-RND: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 1, level at 0, later at 0 of 1.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer at step 0.03 with the first-arrival hold: best $W$ 0.8976 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 128 thousand steps at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (of 1 configuration).
- divisor held from each policy's first arrival: best $W$ 0.8874 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 201 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).
- fast last layer (step 0.1) with the first-arrival hold: best $W$ 0.8707 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 86.1 thousand steps at intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ 0.8781 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 119 thousand steps at intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (of 8 configurations).

#### PointMaze Large, step cost

The arm: fast last layer at step 0.03 with the first-arrival hold, on 256 seeds from 10000 to 10255, read at 33,554,432 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | -0.7223 ± 0.0018 | -0.7268 ± 0.0025 | +0.0045 [-0.0016, +0.0108] | +0.0052 [-0.0009, +0.0114] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | -0.7223 ± 0.0018 | -0.7264 ± 0.0016 | +0.0041 [-0.0008, +0.0088] | -0.0032 [-0.0086, +0.0019] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | 83.3 ± 4.26 | 189 ± 9.38 | -105 [-126, -85.2] | 256 of 256; 256 of 256 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | 83.3 ± 4.26 | 71.5 ± 4.37 | +11.9 [-0.241, +23.9] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 1, level at 0, later at 0 of 1.
- against fast last layer (step 0.1) with the first-arrival hold: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 0 of 0; $M$ sooner at 0, level at 0, later at 0 of 0; not trained by it: intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer at step 0.03 with the first-arrival hold: best $W$ -0.7223 at intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01; best $M$ 83.3 thousand steps at intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 (of 1 configuration).
- divisor held from each policy's first arrival: best $W$ -0.7236 at intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01; best $M$ 163 thousand steps at intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01 (of 3 configurations).
- fast last layer (step 0.1) with the first-arrival hold: best $W$ -0.7264 at intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01; best $M$ 71.5 thousand steps at intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ -0.7310 at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01; best $M$ 120 thousand steps at intrinsic weight 8, learning rate 0.006, entropy coefficient 0.01 (of 4 configurations).

### Fast last layer at step 0.01 with the first-arrival hold, first validation block

#### PointMaze Medium, reward 1 on arrival

The arm: fast last layer at step 0.01 with the first-arrival hold, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9400 ± 0.0031 | 0.9435 ± 0.0027 | -0.0035 [-0.0115, +0.0042] | -0.0006 [-0.0011, -0.0003] | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9400 ± 0.0031 | 0.9194 ± 0.0034 | +0.0206 [+0.0126, +0.0288] | +0.0028 [+0.0018, +0.0039] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9400 ± 0.0031 | 0.9194 ± 0.0023 | +0.0206 [+0.0132, +0.0277] | +0.0199 [+0.0157, +0.0250] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 18.5 ± 1 | 18.5 ± 1.13 | +0.00195 [-2.7, +2.65] | 256 of 256; 256 of 256 | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 18.5 ± 1 | 17.3 ± 0.903 | +1.21 [-1.36, +3.65] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 18.5 ± 1 | 19.9 ± 1.16 | -1.4 [-4.09, +1.29] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against fast last layer at step 0.03 with the first-arrival hold: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against CleanRL PPO-RND: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer at step 0.01 with the first-arrival hold: best $W$ 0.9400 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 18.5 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 1 configuration).
- divisor held from each policy's first arrival: best $W$ 0.9435 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 18.5 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 3 configurations).
- fast last layer at step 0.03 with the first-arrival hold: best $W$ 0.9194 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 17.3 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 1 configuration).
- CleanRL PPO-RND: best $W$ 0.9349 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 17.5 thousand steps at intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (of 6 configurations).

#### PointMaze Medium, step cost

The arm: fast last layer at step 0.01 with the first-arrival hold, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3470 ± 0.0013 | -0.3484 ± 0.0014 | +0.0013 [-0.0024, +0.0050] | -0.0008 [-0.0023, +0.0005] | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3470 ± 0.0013 | -0.3491 ± 0.0014 | +0.0021 [-0.0017, +0.0059] | +0.0001 [-0.0012, +0.0012] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 13.8 ± 0.588 | 14.2 ± 0.617 | -0.442 [-1.84, +1] | 256 of 256; 256 of 256 | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 13.8 ± 0.588 | 14 ± 0.628 | -0.244 [-1.73, +1.23] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against fast last layer at step 0.03 with the first-arrival hold: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 0 of 0; $M$ sooner at 0, level at 0, later at 0 of 0; not trained by it: intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer at step 0.01 with the first-arrival hold: best $W$ -0.3470 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 13.8 thousand steps at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (of 1 configuration).
- divisor held from each policy's first arrival: best $W$ -0.3476 at intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01; best $M$ 13.4 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 3 configurations).
- fast last layer at step 0.03 with the first-arrival hold: best $W$ -0.3491 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 14 thousand steps at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (of 1 configuration).
- CleanRL PPO-RND: best $W$ -0.3505 at intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01; best $M$ 12.4 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (of 6 configurations).

#### PointMaze Large, reward 1 on arrival

The arm: fast last layer at step 0.01 with the first-arrival hold, on 256 seeds from 10000 to 10255, read at 8,388,608 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8928 ± 0.0025 | 0.8874 ± 0.0023 | +0.0054 [-0.0012, +0.0119] | -0.0006 [-0.0013, +0.0001] | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8928 ± 0.0025 | 0.8976 ± 0.0021 | -0.0047 [-0.0112, +0.0019] | +0.0003 [-0.0005, +0.0012] | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8928 ± 0.0025 | 0.8781 ± 0.0021 | +0.0148 [+0.0082, +0.0212] | +0.0051 [+0.0040, +0.0062] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 180 ± 8.38 | 259 ± 10.6 | -79.4 [-107, -52.4] | 256 of 256; 256 of 256 | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 180 ± 8.38 | 128 ± 6.96 | +51.8 [+31.2, +73] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 180 ± 8.38 | 269 ± 10.8 | -89.3 [-115, -64.3] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 1, level at 0, later at 0 of 1.
- against fast last layer at step 0.03 with the first-arrival hold: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 0, level at 0, later at 1 of 1.
- against CleanRL PPO-RND: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 1, level at 0, later at 0 of 1.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer at step 0.01 with the first-arrival hold: best $W$ 0.8928 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 180 thousand steps at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (of 1 configuration).
- divisor held from each policy's first arrival: best $W$ 0.8874 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 201 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).
- fast last layer at step 0.03 with the first-arrival hold: best $W$ 0.8976 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 128 thousand steps at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (of 1 configuration).
- CleanRL PPO-RND: best $W$ 0.8781 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 119 thousand steps at intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (of 8 configurations).

#### PointMaze Large, step cost

The arm: fast last layer at step 0.01 with the first-arrival hold, on 256 seeds from 10000 to 10255, read at 33,554,432 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | -0.7250 ± 0.0023 | -0.7268 ± 0.0025 | +0.0018 [-0.0050, +0.0086] | +0.0021 [-0.0046, +0.0089] | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | -0.7250 ± 0.0023 | -0.7223 ± 0.0018 | -0.0027 [-0.0087, +0.0030] | -0.0031 [-0.0090, +0.0027] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | 123 ± 7.1 | 189 ± 9.38 | -66.2 [-88.2, -43.9] | 256 of 256; 256 of 256 | 256 |
| fast last layer at step 0.03 with the first-arrival hold | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | 123 ± 7.1 | 83.3 ± 4.26 | +39.3 [+23.5, +56] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 1, level at 0, later at 0 of 1.
- against fast last layer at step 0.03 with the first-arrival hold: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 0, level at 0, later at 1 of 1.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 0 of 0; $M$ sooner at 0, level at 0, later at 0 of 0; not trained by it: intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer at step 0.01 with the first-arrival hold: best $W$ -0.7250 at intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01; best $M$ 123 thousand steps at intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 (of 1 configuration).
- divisor held from each policy's first arrival: best $W$ -0.7236 at intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01; best $M$ 163 thousand steps at intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01 (of 3 configurations).
- fast last layer at step 0.03 with the first-arrival hold: best $W$ -0.7223 at intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01; best $M$ 83.3 thousand steps at intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 (of 1 configuration).
- CleanRL PPO-RND: best $W$ -0.7310 at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01; best $M$ 120 thousand steps at intrinsic weight 8, learning rate 0.006, entropy coefficient 0.01 (of 4 configurations).

## Predictions

The predictions were written in `ideas.md` before the runs. Each is quoted, then read part by part: a part held or did not hold on the numbers given, or cannot be read yet because a unit it needs is not complete. A prediction held when every part held, and did not hold when at least one part did not. Where a prediction leaves a choice, it is read as follows. Every reading is on the first validation block (seeds 10000 to 10255) at the batch's one configuration per case. "Both rise as the step falls" is read on the means: $W$ and $M$ at step 0.1 below those at step 0.03, and those below the ones at step 0.01; the hold alone (step 0) is given beside. "Below with the interval below zero" and "above with the interval above zero" are read on the 95% interval of the paired difference; "at least 60 thousand steps", "within 0.005", "behind by 0.02 to 0.05" and "within 0.01" on the mean difference. "RND's best" is RND at its best configuration for $W$ among all those run on the same seeds.

### Prediction 1: ordered by the step

> **Ordered by the step.** In every case the first-goal time and the whole-curve mean both rise as the step falls from 0.1 to 0.03 to 0.01, toward the hold alone.

Observed: **did not hold**.

- Large with reward 1 on arrival: $W$ and $M$ both rise from step 0.1 to 0.03 to 0.01: **did not hold**. At (learning rate 0.002, weight 2): step 0.1: $W$ 0.8707, $M$ 90.7 thousand steps; step 0.03: $W$ 0.8976, $M$ 128 thousand steps; step 0.01: $W$ 0.8928, $M$ 180 thousand steps; step 0 (the hold alone): $W$ 0.8874, $M$ 259 thousand steps.
- Large with the step cost: $W$ and $M$ both rise from step 0.1 to 0.03 to 0.01: **did not hold**. At (learning rate 0.002, weight 8): step 0.1: $W$ -0.7264, $M$ 71.5 thousand steps; step 0.03: $W$ -0.7223, $M$ 83.3 thousand steps; step 0.01: $W$ -0.7250, $M$ 123 thousand steps; step 0 (the hold alone): $W$ -0.7268, $M$ 189 thousand steps.
- Medium with reward 1 on arrival: $W$ and $M$ both rise from step 0.1 to 0.03 to 0.01: **held**. At (learning rate 0.006, weight 4): step 0.1: $W$ 0.8424, $M$ 15.7 thousand steps; step 0.03: $W$ 0.9194, $M$ 17.3 thousand steps; step 0.01: $W$ 0.9400, $M$ 18.5 thousand steps; step 0 (the hold alone): $W$ 0.9435, $M$ 18.5 thousand steps.
- Medium with the step cost: $W$ and $M$ both rise from step 0.1 to 0.03 to 0.01: **did not hold**. At (learning rate 0.006, weight 2): step 0.1: $W$ -0.3692, $M$ 12.9 thousand steps; step 0.03: $W$ -0.3491, $M$ 14 thousand steps; step 0.01: $W$ -0.3470, $M$ 13.8 thousand steps; step 0 (the hold alone): $W$ -0.3484, $M$ 14.2 thousand steps.

### Prediction 2: PointMaze Large with reward 1 on arrival

> **Large with reward 1 on arrival.** At step 0.03 the first-goal time is at least 60 thousand steps below the hold alone's (interval below zero) and the whole-curve mean is within 0.005 of the hold alone's. At step 0.03 or 0.01 the whole-curve mean is above RND's best (0.8781) with the interval above zero. I am unsure of the second sentence: the hold alone is itself only level with RND on the second block.

Observed: **did not hold**.

- Step 0.03: $M$ at least 60 thousand steps below the hold alone's, interval below zero: **held**. At (learning rate 0.002, weight 2): $M$ 128 thousand steps against 259 thousand steps; difference -131 [-155, -108] thousand steps.
- Step 0.03: $W$ within 0.005 of the hold alone's: **did not hold**. At (learning rate 0.002, weight 2): $W$ 0.8976 against 0.8874; difference +0.0101 [+0.0040, +0.0162].
- Step 0.03 or 0.01: $W$ above RND's best with the interval above zero: **held**. Step 0.03: $W$ 0.8976, difference from RND's best (0.8781) +0.0195 [+0.0136, +0.0253]. Step 0.01: $W$ 0.8928, difference from RND's best (0.8781) +0.0148 [+0.0082, +0.0212].

### Prediction 3: PointMaze Medium with reward 1 on arrival

> **Medium with reward 1 on arrival.** Step 0.03 is behind the hold alone by 0.02 to 0.05; step 0.01 is within 0.01 of it.

Observed: **held**.

- Step 0.03 is behind the hold alone by 0.02 to 0.05: **held**. At (learning rate 0.006, weight 4): $W$ 0.9194 against 0.9435; difference -0.0241 [-0.0319, -0.0159].
- Step 0.01 is within 0.01 of the hold alone: **held**. At (learning rate 0.006, weight 4): $W$ 0.9400 against 0.9435; difference -0.0035 [-0.0115, +0.0042].

