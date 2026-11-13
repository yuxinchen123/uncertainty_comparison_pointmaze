# Batch 9: the fast last layer for discovery in the four PointMaze cases, on two validation blocks

Generated 2026-11-13 11:34 PT by the command below. This file is generated: regenerate it, do not edit it. Times are Pacific, converted from machines running Eastern. Run folder: `2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days`; batch folder: `batches/09_fast-last-layer_discovery-in-the-other-three-pointmaze-cases`.

```bash
cd /p/rlprojects/RND/10_jax_exploration_platform
bash "benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh" -s scripts/maze_campaign/analysis/batch_summary_job.py \
    --run-dir=runs/2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days \
    --batch=09_fast-last-layer_discovery-in-the-other-three-pointmaze-cases
```

## Contents

| section | first written | last changed | status |
|---|---|---|---|
| [Measures](#measures) | 2026-11-13 11:13 PT | 2026-11-13 11:13 PT | unread |
| [Validation seeds, case by case](#validation-seeds-case-by-case) | 2026-11-13 11:13 PT | 2026-11-13 11:34 PT | unread |
| [Predictions](#predictions) | 2026-11-13 11:13 PT | 2026-11-13 11:34 PT | unread |

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

The units of the phase `discovery-validation` in PointMaze Medium, reward 1 on arrival. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | **0.8718** ± 0.0034 | **0.9537** ± 0.0028 | 17.6 ± 0.96 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.01 | <u>0.6440</u> ± 0.0073 | <u>0.7891</u> ± 0.0078 | 18.3 ± 0.963 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 | 0.0073 ± 0.0003 | 0.0000 ± 0.0000 | 17.5 ± 1.41 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 0.3785 ± 0.0072 | 0.5111 ± 0.0121 | 13.8 ± 0.763 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.01 | 0.2536 ± 0.0067 | 0.2941 ± 0.0104 | 14.8 ± 1.11 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 | 0.2504 ± 0.0046 | 0.0739 ± 0.0078 | 13.8 ± 0.655 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.01 | 0.1550 ± 0.0030 | 0.1846 ± 0.0068 | 13.6 ± 0.825 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.02, entropy coefficient 0.01 | 0.1440 ± 0.0029 | 0.1639 ± 0.0056 | <u>13.1</u> ± 0.859 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.03 | 0.1141 ± 0.0025 | 0.0071 ± 0.0008 | 14.2 ± 0.758 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.03 | 0.1132 ± 0.0017 | 0.0368 ± 0.0029 | **12.5** ± 0.565 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.02, entropy coefficient 0.03 | 0.0699 ± 0.0012 | 0.0070 ± 0.0006 | 13.5 ± 0.778 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (best of 3) | **0.8718** ± 0.0034 | control | **0.9537** ± 0.0028 | control | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (best of 8) | <u>0.3785</u> ± 0.0072 | -0.4933 [-0.5080, -0.4789] | <u>0.5111</u> ± 0.0121 | -0.4427 [-0.4663, -0.4190] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (best of 3) | <u>17.5</u> ± 1.41 | control | 256 of 256 | <u>0.0073</u> | <u>0.0000</u> | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.03 (best of 8) | **12.5** ± 0.565 | -4.99 [-8.32, -2.37] | 256 of 256 | **0.1132** | **0.0368** | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:14 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:15 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:14 PT | none |
| 4 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:15 PT | none |
| 5 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:15 PT | none |
| 6 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:14 PT | none |
| 7 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:18 PT | none |
| 8 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:13 PT | none |
| 9 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:18 PT | none |
| 10 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:15 PT | none |
| 11 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:16 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.1_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_fast-last-layer-step-0-point-1_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_fast-last-layer-step-0-point-1_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_fast-last-layer-step-0-point-1_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_fast-last-layer-step-0-point-1_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_fast-last-layer-step-0-point-1_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_fast-last-layer-step-0-point-1_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_fast-last-layer-step-0-point-1_learning-rate-0.02_entropy-coefficient-0.03_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_fast-last-layer-step-0-point-1_learning-rate-0.02_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phase `discovery-block-two` in PointMaze Medium, reward 1 on arrival. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 | <u>0.0075</u> ± 0.0003 | <u>0.0000</u> ± 0.0000 | <u>16.8</u> ± 1.43 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.03 | **0.1154** ± 0.0017 | **0.0403** ± 0.0026 | **13** ± 0.627 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND), at its best on this block | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (the only configuration read on this block) | <u>0.0075</u> ± 0.0003 | the control's first-block best (intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01) has no complete unit on this block | control | <u>0.0000</u> ± 0.0000 | the control's first-block best (intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01) has no complete unit on this block | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.03 (its best on this block: its first-block best, intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01, was not run here) | **0.1154** ± 0.0017 | the control's first-block best (intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01) has no complete unit on this block | +0.1079 [+0.1045, +0.1113] | **0.0403** ± 0.0026 | the control's first-block best (intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01) has no complete unit on this block | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (its best on the first block and on this block) | <u>16.8</u> ± 1.43 | control | control | 256 of 256 | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.03 (its best on the first block) | **13** ± 0.627 | -3.78 [-7.19, -0.924] | -3.78 [-7.19, -0.924] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 10:58 PT | none |
| 2 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 10:58 PT | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.1_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `discovery-block-two_pointmaze-medium_reward-arrival-plus-one_fast-layer-step-0-point-1_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-20000-to-20255`

### PointMaze Medium, step cost

#### First block: each side at its best configuration there

The units of the phase `discovery-validation` in PointMaze Medium, step cost. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | **-0.3680** ± 0.0015 | **-0.3244** ± 0.0010 | 14.5 ± 0.63 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.01 | <u>-0.4220</u> ± 0.0028 | <u>-0.3519</u> ± 0.0025 | 13.9 ± 0.762 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 | -0.5526 ± 0.0109 | -0.6884 ± 0.0206 | 12.4 ± 0.471 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | -0.5857 ± 0.0044 | -0.4946 ± 0.0060 | <u>11.7</u> ± 0.564 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.01 | -0.6902 ± 0.0064 | -0.5906 ± 0.0091 | 12.5 ± 0.682 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 | -0.8423 ± 0.0023 | -0.9989 ± 0.0001 | 11.9 ± 0.532 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.01 | -0.8694 ± 0.0032 | -0.8254 ± 0.0062 | 13.3 ± 0.883 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.03 | -0.9095 ± 0.0021 | -0.9992 ± 0.0001 | 13.1 ± 0.659 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.02, entropy coefficient 0.01 | -0.9183 ± 0.0043 | -0.9059 ± 0.0064 | 14 ± 1.05 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.03 | -0.9430 ± 0.0011 | -0.9971 ± 0.0002 | **11.4** ± 0.506 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.02, entropy coefficient 0.03 | -0.9723 ± 0.0008 | -0.9991 ± 0.0001 | 12.4 ± 0.701 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (best of 3) | **-0.3680** ± 0.0015 | control | **-0.3244** ± 0.0010 | control | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (best of 8) | <u>-0.5857</u> ± 0.0044 | -0.2177 [-0.2270, -0.2087] | <u>-0.4946</u> ± 0.0060 | -0.1702 [-0.1821, -0.1589] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (best of 3) | <u>12.4</u> ± 0.471 | control | 256 of 256 | **-0.5526** | **-0.6884** | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.03 (best of 8) | **11.4** ± 0.506 | -1.04 [-2.35, +0.241] | 256 of 256 | <u>-0.9430</u> | <u>-0.9971</u> | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:16 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:19 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:18 PT | none |
| 4 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:18 PT | none |
| 5 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:20 PT | none |
| 6 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:21 PT | none |
| 7 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:21 PT | none |
| 8 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:17 PT | none |
| 9 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:17 PT | none |
| 10 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:17 PT | none |
| 11 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:17 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `discovery-validation_pointmaze-medium_reward-step-cost_fast-last-layer-step-0-point-1_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `discovery-validation_pointmaze-medium_reward-step-cost_fast-last-layer-step-0-point-1_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `discovery-validation_pointmaze-medium_reward-step-cost_fast-last-layer-step-0-point-1_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `discovery-validation_pointmaze-medium_reward-step-cost_fast-last-layer-step-0-point-1_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `discovery-validation_pointmaze-medium_reward-step-cost_fast-last-layer-step-0-point-1_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `discovery-validation_pointmaze-medium_reward-step-cost_fast-last-layer-step-0-point-1_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `discovery-validation_pointmaze-medium_reward-step-cost_fast-last-layer-step-0-point-1_learning-rate-0.02_entropy-coefficient-0.03_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `discovery-validation_pointmaze-medium_reward-step-cost_fast-last-layer-step-0-point-1_learning-rate-0.02_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phase `discovery-block-two` in PointMaze Medium, step cost. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, entropy coefficient 0.03 | **-0.5581** ± 0.0108 | **-0.7035** ± 0.0206 | <u>13.2</u> ± 0.497 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, entropy coefficient 0.03 | <u>-0.9403</u> ± 0.0012 | <u>-0.9961</u> ± 0.0005 | **11.4** ± 0.459 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND), at its best on this block | intrinsic weight 4, entropy coefficient 0.03 (the only configuration read on this block) | **-0.5581** ± 0.0108 | the control's first-block best (intrinsic weight 4, entropy coefficient 0.01) has no complete unit on this block | control | **-0.7035** ± 0.0206 | the control's first-block best (intrinsic weight 4, entropy coefficient 0.01) has no complete unit on this block | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, entropy coefficient 0.03 (its best on this block: its first-block best, intrinsic weight 4, entropy coefficient 0.01, was not run here) | <u>-0.9403</u> ± 0.0012 | the control's first-block best (intrinsic weight 4, entropy coefficient 0.01) has no complete unit on this block | -0.3823 [-0.4034, -0.3607] | <u>-0.9961</u> ± 0.0005 | the control's first-block best (intrinsic weight 4, entropy coefficient 0.01) has no complete unit on this block | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, entropy coefficient 0.03 (its best on the first block and on this block) | <u>13.2</u> ± 0.497 | control | control | 256 of 256 | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, entropy coefficient 0.03 (its best on the first block) | **11.4** ± 0.459 | -1.77 [-3.12, -0.402] | -1.77 [-3.12, -0.402] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 11:00 PT | none |
| 2 | fast last layer inside the rollout, step 0.1 | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 11:02 PT | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `discovery-block-two_pointmaze-medium_reward-step-cost_fast-layer-step-0-point-1_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-20000-to-20255`

### PointMaze Large, reward 1 on arrival

#### First block: each side at its best configuration there

The units of the phase `discovery-validation` in PointMaze Large, reward 1 on arrival. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 8,388,608 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.01 | <u>0.0063</u> ± 0.0002 | <u>0.0015</u> ± 0.0003 | 135 ± 6.06 | **256 of 256** |
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 | 0.0054 ± 0.0002 | 0.0000 ± 0.0000 | 119 ± 5.46 | **256 of 256** |
| control (CleanRL PPO-RND) | intrinsic weight 256, learning rate 0.001, entropy coefficient 0.03 | 0.0006 ± 0.0000 | 0.0000 ± 0.0000 | 245 ± 45.9 | <u>254 of 256</u> |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 | **0.0360** ± 0.0012 | **0.0417** ± 0.0021 | 77.7 ± 4.54 | **256 of 256** |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.03 | 0.0025 ± 0.0001 | 0.0000 ± 0.0000 | <u>66.3</u> ± 3.17 | **256 of 256** |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.006, entropy coefficient 0.03 | 0.0010 ± 0.0000 | 0.0000 ± 0.0000 | **55.8** ± 2.19 | **256 of 256** |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.01 (best of 3) | <u>0.0063</u> ± 0.0002 | control | <u>0.0015</u> ± 0.0003 | control | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 (best of 3) | **0.0360** ± 0.0012 | +0.0297 [+0.0274, +0.0321] | **0.0417** ± 0.0021 | +0.0402 [+0.0362, +0.0445] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (best of 3) | <u>119</u> ± 5.46 | control | 256 of 256 | **0.0054** | 0.0000 | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.006, entropy coefficient 0.03 (best of 3) | **55.8** ± 2.19 | -63.4 [-75.2, -52] | 256 of 256 | <u>0.0010</u> | 0.0000 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:11 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:11 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:08 PT | none |
| 4 | fast last layer inside the rollout, step 0.1 | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:45 PT | none |
| 5 | fast last layer inside the rollout, step 0.1 | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:46 PT | none |
| 6 | fast last layer inside the rollout, step 0.1 | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:46 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.001_entropy-coefficient-0.03_intrinsic-weight-256.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `discovery-validation_pointmaze-large_reward-arrival-plus-one_fast-last-layer-step-0-point-1_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `discovery-validation_pointmaze-large_reward-arrival-plus-one_fast-last-layer-step-0-point-1_learning-rate-0.002_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `discovery-validation_pointmaze-large_reward-arrival-plus-one_fast-last-layer-step-0-point-1_learning-rate-0.006_entropy-coefficient-0.03_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phase `discovery-block-two` in PointMaze Large, reward 1 on arrival. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 8,388,608 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | learning rate 0.006 | 0.0057 ± 0.0002 | 0.0003 ± 0.0002 | 130 ± 5.17 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND), at its best on this block | learning rate 0.006 (the only configuration read on this block) | 0.0057 ± 0.0002 | the control's first-block best (learning rate 0.01) has no complete unit on this block | control | 0.0003 ± 0.0002 | the control's first-block best (learning rate 0.01) has no complete unit on this block | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | learning rate 0.006 (its best on the first block and on this block) | 130 ± 5.17 | control | control | 256 of 256 | 256 |

Not complete, left out: fast last layer inside the rollout, step 0.1: 0 of 1 configurations read (0 of 1 units complete).

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 11:13 PT | none |
| 2 | fast last layer inside the rollout, step 0.1 | PointMaze Large, reward 1 on arrival | 1 | unfinished | 222 of 256 | not finished | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `discovery-block-two_pointmaze-large_reward-arrival-plus-one_fast-layer-step-0-point-1_learning-rate-0.006_entropy-coefficient-0.03_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-20000-to-20255`

### PointMaze Large, step cost

#### First block: each side at its best configuration there

The units of the phase `discovery-validation` in PointMaze Large, step cost. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 33,554,432 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.0045, entropy coefficient 0.01 | **-0.8831** ± 0.0022 | **-0.8849** ± 0.0026 | 131 ± 6.45 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006, entropy coefficient 0.01 | <u>-0.9669</u> ± 0.0017 | <u>-0.9803</u> ± 0.0022 | 120 ± 5.6 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 8, learning rate 0.0045, entropy coefficient 0.01 | -0.9950 ± 0.0002 | -0.9970 ± 0.0003 | 76.4 ± 4.83 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.03 | -0.9999 ± 0.0000 | -1.0000 ± 0.0000 | <u>62.1</u> ± 2.52 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.006, entropy coefficient 0.03 | -1.0000 ± 0.0000 | -1.0000 ± 0.0000 | **58.9** ± 2.49 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.0045, entropy coefficient 0.01 (best of 2) | **-0.8831** ± 0.0022 | control | **-0.8849** ± 0.0026 | control | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 8, learning rate 0.0045, entropy coefficient 0.01 (best of 3) | <u>-0.9950</u> ± 0.0002 | -0.1119 [-0.1161, -0.1076] | <u>-0.9970</u> ± 0.0003 | -0.1121 [-0.1175, -0.1070] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006, entropy coefficient 0.01 (best of 2) | <u>120</u> ± 5.6 | control | 256 of 256 | **-0.9669** | **-0.9803** | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.006, entropy coefficient 0.03 (best of 3) | **58.9** ± 2.49 | -61.3 [-73.5, -49.6] | 256 of 256 | <u>-1.0000</u> | <u>-1.0000</u> | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:48 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:47 PT | none |
| 3 | fast last layer inside the rollout, step 0.1 | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 02:50 PT | none |
| 4 | fast last layer inside the rollout, step 0.1 | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 04:40 PT | none |
| 5 | fast last layer inside the rollout, step 0.1 | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 02:48 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.0045_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-large_reward-step-cost_fast-last-layer-step-0-point-1_learning-rate-0.002_entropy-coefficient-0.03_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `discovery-validation_pointmaze-large_reward-step-cost_fast-last-layer-step-0-point-1_learning-rate-0.0045_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `discovery-validation_pointmaze-large_reward-step-cost_fast-last-layer-step-0-point-1_learning-rate-0.006_entropy-coefficient-0.03_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phase `discovery-block-two` in PointMaze Large, step cost. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 33,554,432 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | learning rate 0.006 | -0.9636 ± 0.0020 | -0.9753 ± 0.0025 | 121 ± 5.71 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND), at its best on this block | learning rate 0.006 (the only configuration read on this block) | -0.9636 ± 0.0020 | the control's first-block best (learning rate 0.0045) has no complete unit on this block | control | -0.9753 ± 0.0025 | the control's first-block best (learning rate 0.0045) has no complete unit on this block | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | learning rate 0.006 (its best on the first block and on this block) | 121 ± 5.71 | control | control | 256 of 256 | 256 |

Not complete, left out: fast last layer inside the rollout, step 0.1: 0 of 1 configurations read (0 of 1 units complete).

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 11:32 PT | none |
| 2 | fast last layer inside the rollout, step 0.1 | PointMaze Large, step cost | 1 | unfinished | 90 of 256 | not finished | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `discovery-block-two_pointmaze-large_reward-step-cost_fast-layer-step-0-point-1_learning-rate-0.006_entropy-coefficient-0.03_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-20000-to-20255`

## Predictions

The predictions were written in `ideas.md` before the runs. Each is quoted, then read part by part: a part held or did not hold on the numbers given, or cannot be read yet because a unit it needs is not complete. A prediction held when every part held, and did not hold when at least one part did not. Where a prediction leaves a choice, it is read as follows. Every reading is of $M$. On the first block each side stands at its best configuration for $M$ among those run there; on the second block each side stands at the configuration that was its best for $M$ on the first block (equal rules). "Ahead with the interval below zero" is read as: the 95% interval of the paired difference (fast last layer minus RND) is entirely below zero. "By at least 40 thousand steps" and "by more than 40 thousand" are read on the mean difference. "A few thousand steps at most" is read as: the mean difference is below zero and at most 6 thousand steps in size. "Level" is read as: the interval holds zero.

### Prediction 1: PointMaze Large, step cost (first block)

> **PointMaze Large, step cost:** the fast last layer is ahead of RND by at least 40 thousand steps with its interval below zero, as with reward 1 on arrival, because the reward setting does not matter before the first arrival except through the value network's start.

Observed: **held**.

- Ahead of RND by at least 40 thousand steps, interval below zero: **held**. $M$ 58.9 thousand steps; difference from RND's best -61.3 [-73.5, -49.6] thousand steps.

### Prediction 2: the two PointMaze Medium cases (first block)

> **The two PointMaze Medium cases:** it is ahead of RND by a few thousand steps at most; the interval is below zero in at least one of the two.

Observed: **held**.

- Reward 1 on arrival: ahead by a few thousand steps at most: **held**. $M$ 12.5 thousand steps; difference from RND's best -4.99 [-8.32, -2.37] thousand steps.
- Step cost: ahead by a few thousand steps at most: **held**. $M$ 11.4 thousand steps; difference from RND's best -1.04 [-2.35, +0.241] thousand steps.
- The interval is below zero in at least one of the two: **held**. Reward 1 on arrival: $M$ 12.5 thousand steps; difference from RND's best -4.99 [-8.32, -2.37] thousand steps. Step cost: $M$ 11.4 thousand steps; difference from RND's best -1.04 [-2.35, +0.241] thousand steps.

### Prediction 3: the second block for discovery (addendum, written 2026-11-13 10:53 PT)

> the fast last layer is ahead with the interval below zero in the same three cases (both Large cases by more than 40 thousand steps, Medium with reward 1 on arrival by 2 to 6 thousand) and level on Medium with the step cost.

Observed: **did not hold**.

- PointMaze Large with reward 1 on arrival: ahead by more than 40 thousand steps, interval below zero: **cannot be read yet**. Waiting for complete units in PointMaze Large, reward 1 on arrival, second block: fast last layer inside the rollout, step 0.1: 0 of 1 configurations read (0 of 1 units complete).
- PointMaze Large with the step cost: ahead by more than 40 thousand steps, interval below zero: **cannot be read yet**. Waiting for complete units in PointMaze Large, step cost, second block: fast last layer inside the rollout, step 0.1: 0 of 1 configurations read (0 of 1 units complete).
- PointMaze Medium with reward 1 on arrival: ahead by 2 to 6 thousand steps, interval below zero: **held**. $M$ 13 thousand steps; difference from RND under equal rules -3.78 [-7.19, -0.924] thousand steps.
- PointMaze Medium with the step cost: level: **did not hold**. $M$ 11.4 thousand steps; difference from RND under equal rules -1.77 [-3.12, -0.402] thousand steps.

