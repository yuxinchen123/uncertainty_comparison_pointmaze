# Batch 11: the fast last layer until each policy's first arrival, with the divisor held from then on

Generated 2026-11-13 11:35 PT by the command below. This file is generated: regenerate it, do not edit it. Times are Pacific, converted from machines running Eastern. Run folder: `2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days`; batch folder: `batches/11_fast-last-layer-until-the-first-arrival_with-the-first-arrival-hold_four-pointmaze-cases`.

```bash
cd /p/rlprojects/RND/10_jax_exploration_platform
bash "benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh" -s scripts/maze_campaign/analysis/batch_summary_job.py \
    --run-dir=runs/2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days \
    --batch=11_fast-last-layer-until-the-first-arrival_with-the-first-arrival-hold_four-pointmaze-cases
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
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.7001 ± 0.0055 | 0.9414 ± 0.0033 | <u>14.6</u> ± 0.737 | 256 of 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 0.5548 ± 0.0067 | 0.8353 ± 0.0069 | **14** ± 0.777 | 256 of 256 |
| fast last layer until the first arrival, then RND's running divisor | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.6403 ± 0.0077 | 0.8320 ± 0.0089 | 17.7 ± 1.08 | 256 of 256 |
| fast last layer until the first arrival, then RND's running divisor | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.4487 ± 0.0065 | 0.6323 ± 0.0099 | 17.4 ± 1.07 | 256 of 256 |
| fast last layer until the first arrival, divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.6610 ± 0.0071 | 0.9205 ± 0.0045 | 16.8 ± 1.41 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | **0.9435** ± 0.0027 | <u>0.9999</u> ± 0.0000 | 18.5 ± 1.13 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.01, entropy coefficient 0.01 | <u>0.9412</u> ± 0.0024 | **1.0000** ± 0.0000 | 23.1 ± 1.34 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9334 ± 0.0029 | **1.0000** ± 0.0000 | 29.4 ± 1.85 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.8424 ± 0.0063 | 0.9747 ± 0.0037 | 15.7 ± 0.84 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 0.7759 ± 0.0077 | 0.9490 ± 0.0060 | 14.7 ± 0.957 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 6) | <u>0.9349</u> ± 0.0031 | control | <u>0.9950</u> ± 0.0010 | control | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | **0.9435** ± 0.0027 | **+0.0085** [+0.0009, +0.0161] | **0.9999** ± 0.0000 | **+0.0049** [+0.0033, +0.0070] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 2) | 0.8424 ± 0.0063 | <u>-0.0925</u> [-0.1068, -0.0788] | 0.9747 ± 0.0037 | <u>-0.0203</u> [-0.0285, -0.0134] | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 2) | 0.7001 ± 0.0055 | -0.2348 [-0.2464, -0.2230] | 0.9414 ± 0.0033 | -0.0536 [-0.0605, -0.0469] | 256 |
| fast last layer until the first arrival, divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (the only configuration read) | 0.6610 ± 0.0071 | -0.2740 [-0.2886, -0.2595] | 0.9205 ± 0.0045 | -0.0745 [-0.0836, -0.0658] | 256 |
| fast last layer until the first arrival, then RND's running divisor | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 2) | 0.6403 ± 0.0077 | -0.2946 [-0.3108, -0.2783] | 0.8320 ± 0.0089 | -0.1630 [-0.1809, -0.1459] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (best of 6) | 17.5 ± 1.41 | control | 256 of 256 | 0.0073 | 0.0000 | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (best of 2) | **14** ± 0.777 | **-3.52** [-6.93, -0.627] | 256 of 256 | 0.5548 | 0.8353 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (best of 2) | <u>14.7</u> ± 0.957 | <u>-2.85</u> [-6.45, +0.268] | 256 of 256 | <u>0.7759</u> | <u>0.9490</u> | 256 |
| fast last layer until the first arrival, divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (the only configuration read) | 16.8 ± 1.41 | -0.683 [-4.45, +3.07] | 256 of 256 | 0.6610 | 0.9205 | 256 |
| fast last layer until the first arrival, then RND's running divisor | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 2) | 17.4 ± 1.07 | -0.149 [-3.58, +2.9] | 256 of 256 | 0.4487 | 0.6323 | 256 |
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
| 7 | fast last layer until the first arrival, then the first-arrival hold | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:05 PT | none |
| 8 | fast last layer until the first arrival, then the first-arrival hold | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:04 PT | none |
| 9 | fast last layer until the first arrival, then RND's running divisor | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:18 PT | none |
| 10 | fast last layer until the first arrival, then RND's running divisor | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:18 PT | none |
| 11 | fast last layer until the first arrival, divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:21 PT | none |
| 12 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:32 PT | none |
| 13 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:32 PT | none |
| 14 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:34 PT | none |
| 15 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 03:06 PT | none |
| 16 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 03:08 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.1_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-until-arrival-then-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-until-arrival-then-hold_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-until-arrival-no-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-until-arrival-no-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-until-arrival-fixed-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
14. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
15. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
16. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

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
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3821 ± 0.0018 | -0.3153 ± 0.0006 | <u>12.7</u> ± 0.593 | 256 of 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | -0.4307 ± 0.0026 | -0.3296 ± 0.0013 | 12.9 ± 0.684 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 | **-0.3476** ± 0.0014 | -0.3073 ± 0.0007 | 15.4 ± 0.693 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | <u>-0.3484</u> ± 0.0014 | **-0.3058** ± 0.0004 | 14.2 ± 0.617 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | -0.3504 ± 0.0015 | <u>-0.3066</u> ± 0.0005 | 13.4 ± 0.562 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3692 ± 0.0030 | -0.3132 ± 0.0008 | 12.9 ± 0.654 | 256 of 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | -0.4057 ± 0.0034 | -0.3299 ± 0.0015 | 15 ± 1.74 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01 (best of 6) | <u>-0.3505</u> ± 0.0016 | control | <u>-0.3078</u> ± 0.0007 | control | 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 (best of 3) | **-0.3476** ± 0.0014 | **+0.0029** [-0.0015, +0.0072] | **-0.3073** ± 0.0007 | **+0.0005** [-0.0014, +0.0025] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 2) | -0.3692 ± 0.0030 | <u>-0.0187</u> [-0.0258, -0.0120] | -0.3132 ± 0.0008 | <u>-0.0054</u> [-0.0076, -0.0034] | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 2) | -0.3821 ± 0.0018 | -0.0316 [-0.0361, -0.0270] | -0.3153 ± 0.0006 | -0.0074 [-0.0092, -0.0056] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (best of 6) | **12.4** ± 0.471 | control | 256 of 256 | -0.5526 | -0.6884 | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 2) | <u>12.7</u> ± 0.593 | **+0.266** [-1.12, +1.64] | 256 of 256 | -0.3821 | -0.3153 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 2) | 12.9 ± 0.654 | <u>+0.523</u> [-0.938, +2.07] | 256 of 256 | <u>-0.3692</u> | <u>-0.3132</u> | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | 13.4 ± 0.562 | +0.958 [-0.349, +2.26] | 256 of 256 | **-0.3504** | **-0.3066** | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:16 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:19 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:18 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:58 PT | none |
| 5 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:53 PT | none |
| 6 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:54 PT | none |
| 7 | fast last layer until the first arrival, then the first-arrival hold | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 04:02 PT | none |
| 8 | fast last layer until the first arrival, then the first-arrival hold | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 04:02 PT | none |
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
7. `validation-block_pointmaze-medium_reward-step-cost_fast-until-arrival-then-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-step-cost_fast-until-arrival-then-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
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
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8496 ± 0.0036 | 0.9919 ± 0.0011 | 106 ± 11.2 | **256 of 256** |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.7871 ± 0.0045 | 0.9770 ± 0.0021 | **82.9** ± 4.46 | **256 of 256** |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 | 0.3370 ± 0.0092 | 0.6110 ± 0.0137 | 89.1 ± 8.37 | **256 of 256** |
| fast last layer until the first arrival, then RND's running divisor | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.4258 ± 0.0056 | 0.6341 ± 0.0071 | 96.2 ± 8.72 | **256 of 256** |
| fast last layer until the first arrival, then RND's running divisor | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.1356 ± 0.0039 | 0.2376 ± 0.0071 | 91.2 ± 6.69 | **256 of 256** |
| fast last layer until the first arrival, then RND's running divisor | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 | 0.0230 ± 0.0004 | 0.0305 ± 0.0016 | <u>83</u> ± 6.26 | **256 of 256** |
| fast last layer until the first arrival, divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.5718 ± 0.0045 | 0.8776 ± 0.0045 | 83.3 ± 5.16 | **256 of 256** |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | **0.8874** ± 0.0023 | <u>0.9977</u> ± 0.0002 | 259 ± 10.6 | **256 of 256** |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | <u>0.8852</u> ± 0.0023 | 0.9976 ± 0.0003 | 201 ± 9.11 | **256 of 256** |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 0.8680 ± 0.0033 | **0.9979** ± 0.0003 | 375 ± 12.4 | **256 of 256** |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8707 ± 0.0058 | 0.9872 ± 0.0055 | 90.7 ± 5.08 | **256 of 256** |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 | 0.5725 ± 0.0106 | 0.8495 ± 0.0123 | 86.1 ± 6.37 | **256 of 256** |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 8) | <u>0.8781</u> ± 0.0021 | control | <u>0.9920</u> ± 0.0005 | control | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 3) | **0.8874** ± 0.0023 | **+0.0094** [+0.0032, +0.0156] | **0.9977** ± 0.0002 | **+0.0057** [+0.0046, +0.0068] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 2) | 0.8707 ± 0.0058 | <u>-0.0074</u> [-0.0203, +0.0037] | 0.9872 ± 0.0055 | -0.0048 [-0.0169, +0.0034] | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 3) | 0.8496 ± 0.0036 | -0.0284 [-0.0372, -0.0208] | 0.9919 ± 0.0011 | <u>-0.0001</u> [-0.0027, +0.0019] | 256 |
| fast last layer until the first arrival, divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (the only configuration read) | 0.5718 ± 0.0045 | -0.3063 [-0.3161, -0.2966] | 0.8776 ± 0.0045 | -0.1144 [-0.1234, -0.1058] | 256 |
| fast last layer until the first arrival, then RND's running divisor | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 3) | 0.4258 ± 0.0056 | -0.4522 [-0.4641, -0.4403] | 0.6341 ± 0.0071 | -0.3579 [-0.3723, -0.3443] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (best of 8) | 119 ± 5.46 | control | 256 of 256 | 0.0054 | 0.0000 | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (best of 3) | **82.9** ± 4.46 | **-36.4** [-50.6, -22.5] | 256 of 256 | <u>0.7871</u> | <u>0.9770</u> | 256 |
| fast last layer until the first arrival, then RND's running divisor | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 (best of 3) | <u>83</u> ± 6.26 | <u>-36.2</u> [-52.4, -18.8] | 256 of 256 | 0.0230 | 0.0305 | 256 |
| fast last layer until the first arrival, divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (the only configuration read) | 83.3 ± 5.16 | -36 [-50.7, -21.5] | 256 of 256 | 0.5718 | 0.8776 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 (best of 2) | 86.1 ± 6.37 | -33.2 [-49, -16.5] | 256 of 256 | 0.5725 | 0.8495 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (best of 3) | 201 ± 9.11 | +81.6 [+61.3, +102] | 256 of 256 | **0.8852** | **0.9976** | 256 |

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
| 9 | fast last layer until the first arrival, then the first-arrival hold | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:22 PT | none |
| 10 | fast last layer until the first arrival, then the first-arrival hold | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:11 PT | none |
| 11 | fast last layer until the first arrival, then the first-arrival hold | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:23 PT | none |
| 12 | fast last layer until the first arrival, then RND's running divisor | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:09 PT | none |
| 13 | fast last layer until the first arrival, then RND's running divisor | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:10 PT | none |
| 14 | fast last layer until the first arrival, then RND's running divisor | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:09 PT | none |
| 15 | fast last layer until the first arrival, divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:09 PT | none |
| 16 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:55 PT | none |
| 17 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:55 PT | none |
| 18 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:07 PT | none |
| 19 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 03:44 PT | none |
| 20 | fast last layer (step 0.1) with the first-arrival hold | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:46 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.001_entropy-coefficient-0.03_intrinsic-weight-256.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.001_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-until-arrival-then-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-until-arrival-then-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-until-arrival-then-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-until-arrival-no-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-until-arrival-no-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
14. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-until-arrival-no-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
15. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-until-arrival-fixed-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
16. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
17. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
18. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
19. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
20. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`

## Arms at the same configuration, on the same seeds

Each group compares one arm with other sides at the configurations both trained, matched on the intrinsic weight, learning rate, entropy coefficient and clip coefficient, and paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Define $W$, $L$ and $M$ as in the section Measures. No configuration is chosen here.

### Fast last layer until the first arrival, then the first-arrival hold

#### PointMaze Medium, reward 1 on arrival

The arm: fast last layer until the first arrival, then the first-arrival hold, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.7001 ± 0.0055 | 0.9435 ± 0.0027 | -0.2433 [-0.2551, -0.2311] | -0.0585 [-0.0651, -0.0521] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.7001 ± 0.0055 | 0.8424 ± 0.0063 | -0.1423 [-0.1558, -0.1281] | -0.0333 [-0.0425, -0.0233] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 0.5548 ± 0.0067 | 0.7759 ± 0.0077 | -0.2211 [-0.2372, -0.2049] | -0.1136 [-0.1316, -0.0956] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.7001 ± 0.0055 | 0.9194 ± 0.0023 | -0.2193 [-0.2303, -0.2088] | -0.0380 [-0.0463, -0.0296] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 0.5548 ± 0.0067 | 0.8718 ± 0.0034 | -0.3170 [-0.3304, -0.3035] | -0.1184 [-0.1327, -0.1048] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 14.6 ± 0.737 | 18.5 ± 1.13 | -3.89 [-6.42, -1.41] | 256 of 256; 256 of 256 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 14.6 ± 0.737 | 15.7 ± 0.84 | -1.07 [-2.95, +0.783] | 256 of 256; 256 of 256 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 14 ± 0.777 | 14.7 ± 0.957 | -0.666 [-2.89, +1.48] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 14.6 ± 0.737 | 19.9 ± 1.16 | -5.29 [-7.94, -2.77] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 14 ± 0.777 | 17.6 ± 0.96 | -3.57 [-5.82, -1.37] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 0, behind at 1 of 1; $M$ sooner at 1, level at 0, later at 0 of 1; not trained by it: intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01.
- against fast last layer (step 0.1) with the first-arrival hold: $W$ ahead at 0, level at 0, behind at 2 of 2; $M$ sooner at 0, level at 2, later at 0 of 2.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 2 of 2; $M$ sooner at 2, level at 0, later at 0 of 2.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer until the first arrival, then the first-arrival hold: best $W$ 0.7001 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 14 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (of 2 configurations).
- divisor held from each policy's first arrival: best $W$ 0.9435 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 18.5 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 3 configurations).
- fast last layer (step 0.1) with the first-arrival hold: best $W$ 0.8424 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 14.7 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ 0.9349 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 17.5 thousand steps at intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (of 6 configurations).

#### PointMaze Medium, step cost

The arm: fast last layer until the first arrival, then the first-arrival hold, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3821 ± 0.0018 | -0.3484 ± 0.0014 | -0.0337 [-0.0381, -0.0295] | -0.0095 [-0.0108, -0.0082] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | -0.4307 ± 0.0026 | -0.3504 ± 0.0015 | -0.0803 [-0.0862, -0.0745] | -0.0229 [-0.0259, -0.0201] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3821 ± 0.0018 | -0.3692 ± 0.0030 | -0.0129 [-0.0190, -0.0062] | -0.0021 [-0.0038, +0.0001] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 12.7 ± 0.593 | 14.2 ± 0.617 | -1.57 [-2.94, -0.195] | 256 of 256; 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 12.9 ± 0.684 | 13.4 ± 0.562 | -0.441 [-2.01, +1.15] | 256 of 256; 256 of 256 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 12.7 ± 0.593 | 12.9 ± 0.654 | -0.257 [-1.76, +1.22] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 0, behind at 2 of 2; $M$ sooner at 1, level at 1, later at 0 of 2.
- against fast last layer (step 0.1) with the first-arrival hold: $W$ ahead at 0, level at 0, behind at 1 of 1; $M$ sooner at 0, level at 1, later at 0 of 1; not trained by it: intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 0 of 0; $M$ sooner at 0, level at 0, later at 0 of 0; not trained by it: intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer until the first arrival, then the first-arrival hold: best $W$ -0.3821 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 12.7 thousand steps at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (of 2 configurations).
- divisor held from each policy's first arrival: best $W$ -0.3476 at intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01; best $M$ 13.4 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 3 configurations).
- fast last layer (step 0.1) with the first-arrival hold: best $W$ -0.3692 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 12.9 thousand steps at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ -0.3505 at intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01; best $M$ 12.4 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (of 6 configurations).

#### PointMaze Large, reward 1 on arrival

The arm: fast last layer until the first arrival, then the first-arrival hold, on 256 seeds from 10000 to 10255, read at 8,388,608 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8496 ± 0.0036 | 0.8874 ± 0.0023 | -0.0378 [-0.0457, -0.0304] | -0.0058 [-0.0082, -0.0040] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.7871 ± 0.0045 | 0.8852 ± 0.0023 | -0.0980 [-0.1081, -0.0881] | -0.0206 [-0.0248, -0.0169] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8496 ± 0.0036 | 0.8707 ± 0.0058 | -0.0211 [-0.0334, -0.0068] | +0.0047 [-0.0041, +0.0175] | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 | 0.3370 ± 0.0092 | 0.5725 ± 0.0106 | -0.2355 [-0.2607, -0.2097] | -0.2385 [-0.2713, -0.2056] | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8496 ± 0.0036 | 0.8781 ± 0.0021 | -0.0284 [-0.0368, -0.0205] | -0.0001 [-0.0027, +0.0019] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.7871 ± 0.0045 | 0.8626 ± 0.0025 | -0.0755 [-0.0854, -0.0661] | +0.0005 [-0.0047, +0.0056] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 106 ± 11.2 | 259 ± 10.6 | -153 [-182, -121] | 256 of 256; 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 82.9 ± 4.46 | 201 ± 9.11 | -118 [-138, -98.9] | 256 of 256; 256 of 256 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 106 ± 11.2 | 90.7 ± 5.08 | +15.3 [-6.62, +41.9] | 256 of 256; 256 of 256 | 256 |
| fast last layer (step 0.1) with the first-arrival hold | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 | 89.1 ± 8.37 | 86.1 ± 6.37 | +3.02 [-15.1, +23.1] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 106 ± 11.2 | 269 ± 10.8 | -163 [-193, -131] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 82.9 ± 4.46 | 205 ± 8.2 | -122 [-141, -103] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held from each policy's first arrival: $W$ ahead at 0, level at 0, behind at 2 of 2; $M$ sooner at 2, level at 0, later at 0 of 2; not trained by it: intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01.
- against fast last layer (step 0.1) with the first-arrival hold: $W$ ahead at 0, level at 0, behind at 2 of 2; $M$ sooner at 0, level at 2, later at 0 of 2; not trained by it: intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 2 of 2; $M$ sooner at 2, level at 0, later at 0 of 2; not trained by it: intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer until the first arrival, then the first-arrival hold: best $W$ 0.8496 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 82.9 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).
- divisor held from each policy's first arrival: best $W$ 0.8874 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 201 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).
- fast last layer (step 0.1) with the first-arrival hold: best $W$ 0.8707 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 86.1 thousand steps at intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ 0.8781 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 119 thousand steps at intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (of 8 configurations).

### Fast last layer until the first arrival, then RND's running divisor (addendum 1)

#### PointMaze Medium, reward 1 on arrival

The arm: fast last layer until the first arrival, then RND's running divisor, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.6403 ± 0.0077 | 0.9349 ± 0.0031 | -0.2946 [-0.3110, -0.2780] | -0.1630 [-0.1807, -0.1462] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.4487 ± 0.0065 | 0.9194 ± 0.0023 | -0.4707 [-0.4833, -0.4580] | -0.3471 [-0.3662, -0.3283] | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.4487 ± 0.0065 | 0.7001 ± 0.0055 | -0.2514 [-0.2658, -0.2371] | -0.3091 [-0.3290, -0.2897] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 17.7 ± 1.08 | 25 ± 1.46 | -7.33 [-10.8, -3.91] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 17.4 ± 1.07 | 19.9 ± 1.16 | -2.54 [-5.34, +0.232] | 256 of 256; 256 of 256 | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 17.4 ± 1.07 | 14.6 ± 0.737 | +2.75 [+0.543, +5.04] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 2 of 2; $M$ sooner at 1, level at 1, later at 0 of 2.
- against fast last layer until the first arrival, then the first-arrival hold: $W$ ahead at 0, level at 0, behind at 1 of 1; $M$ sooner at 0, level at 0, later at 1 of 1; not trained by it: intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer until the first arrival, then RND's running divisor: best $W$ 0.6403 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 17.4 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ 0.9349 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 17.5 thousand steps at intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (of 6 configurations).
- fast last layer until the first arrival, then the first-arrival hold: best $W$ 0.7001 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 14 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (of 2 configurations).

#### PointMaze Large, reward 1 on arrival

The arm: fast last layer until the first arrival, then RND's running divisor, on 256 seeds from 10000 to 10255, read at 8,388,608 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.4258 ± 0.0056 | 0.8781 ± 0.0021 | -0.4522 [-0.4639, -0.4406] | -0.3579 [-0.3719, -0.3445] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.1356 ± 0.0039 | 0.8626 ± 0.0025 | -0.7270 [-0.7357, -0.7181] | -0.7389 [-0.7527, -0.7243] | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.4258 ± 0.0056 | 0.8496 ± 0.0036 | -0.4238 [-0.4363, -0.4110] | -0.3578 [-0.3720, -0.3440] | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.1356 ± 0.0039 | 0.7871 ± 0.0045 | -0.6515 [-0.6615, -0.6409] | -0.7394 [-0.7532, -0.7253] | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 | 0.0230 ± 0.0004 | 0.3370 ± 0.0092 | -0.3140 [-0.3322, -0.2958] | -0.5805 [-0.6074, -0.5537] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 96.2 ± 8.72 | 269 ± 10.8 | -173 [-199, -147] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 91.2 ± 6.69 | 205 ± 8.2 | -114 [-133, -93.2] | 256 of 256; 256 of 256 | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 96.2 ± 8.72 | 106 ± 11.2 | -9.81 [-38.9, +16.7] | 256 of 256; 256 of 256 | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 91.2 ± 6.69 | 82.9 ± 4.46 | +8.26 [-5.68, +23.5] | 256 of 256; 256 of 256 | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 | 83 ± 6.26 | 89.1 ± 8.37 | -6.07 [-27.4, +13.3] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 2 of 2; $M$ sooner at 2, level at 0, later at 0 of 2; not trained by it: intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01.
- against fast last layer until the first arrival, then the first-arrival hold: $W$ ahead at 0, level at 0, behind at 3 of 3; $M$ sooner at 0, level at 3, later at 0 of 3.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer until the first arrival, then RND's running divisor: best $W$ 0.4258 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 83 thousand steps at intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).
- CleanRL PPO-RND: best $W$ 0.8781 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 119 thousand steps at intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (of 8 configurations).
- fast last layer until the first arrival, then the first-arrival hold: best $W$ 0.8496 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 82.9 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).

### Fast last layer until the first arrival, divisor held after one sixteenth of the run (addendum 1)

#### PointMaze Medium, reward 1 on arrival

The arm: fast last layer until the first arrival, divisor held after one sixteenth of the run, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| fast last layer until the first arrival, then RND's running divisor | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.6610 ± 0.0071 | 0.4487 ± 0.0065 | +0.2123 [+0.1963, +0.2281] | +0.2883 [+0.2679, +0.3090] | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.6610 ± 0.0071 | 0.7001 ± 0.0055 | -0.0392 [-0.0525, -0.0254] | -0.0209 [-0.0308, -0.0110] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.6610 ± 0.0071 | 0.9194 ± 0.0023 | -0.2585 [-0.2726, -0.2446] | -0.0589 [-0.0689, -0.0487] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| fast last layer until the first arrival, then RND's running divisor | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 16.8 ± 1.41 | 17.4 ± 1.07 | -0.534 [-3.72, +2.79] | 256 of 256; 256 of 256 | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 16.8 ± 1.41 | 14.6 ± 0.737 | +2.22 [-0.232, +5.01] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 16.8 ± 1.41 | 19.9 ± 1.16 | -3.08 [-6.35, +0.339] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against fast last layer until the first arrival, then RND's running divisor: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against fast last layer until the first arrival, then the first-arrival hold: $W$ ahead at 0, level at 0, behind at 1 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 1 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer until the first arrival, divisor held after one sixteenth of the run: best $W$ 0.6610 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 16.8 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 1 configuration).
- fast last layer until the first arrival, then RND's running divisor: best $W$ 0.6403 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 17.4 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 2 configurations).
- fast last layer until the first arrival, then the first-arrival hold: best $W$ 0.7001 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 14 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ 0.9349 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 17.5 thousand steps at intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (of 6 configurations).

#### PointMaze Large, reward 1 on arrival

The arm: fast last layer until the first arrival, divisor held after one sixteenth of the run, on 256 seeds from 10000 to 10255, read at 8,388,608 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| fast last layer until the first arrival, then RND's running divisor | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.5718 ± 0.0045 | 0.1356 ± 0.0039 | +0.4362 [+0.4264, +0.4458] | +0.6400 [+0.6252, +0.6547] | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.5718 ± 0.0045 | 0.7871 ± 0.0045 | -0.2153 [-0.2253, -0.2052] | -0.0994 [-0.1085, -0.0907] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.5718 ± 0.0045 | 0.8626 ± 0.0025 | -0.2908 [-0.3006, -0.2810] | -0.0989 [-0.1079, -0.0903] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| fast last layer until the first arrival, then RND's running divisor | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 83.3 ± 5.16 | 91.2 ± 6.69 | -7.87 [-23.7, +7.36] | 256 of 256; 256 of 256 | 256 |
| fast last layer until the first arrival, then the first-arrival hold | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 83.3 ± 5.16 | 82.9 ± 4.46 | +0.389 [-11.8, +13.2] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 83.3 ± 5.16 | 205 ± 8.2 | -121 [-139, -103] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against fast last layer until the first arrival, then RND's running divisor: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against fast last layer until the first arrival, then the first-arrival hold: $W$ ahead at 0, level at 0, behind at 1 of 1; $M$ sooner at 0, level at 1, later at 0 of 1.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 1 of 1; $M$ sooner at 1, level at 0, later at 0 of 1.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- fast last layer until the first arrival, divisor held after one sixteenth of the run: best $W$ 0.5718 at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01; best $M$ 83.3 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 1 configuration).
- fast last layer until the first arrival, then RND's running divisor: best $W$ 0.4258 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 83 thousand steps at intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).
- fast last layer until the first arrival, then the first-arrival hold: best $W$ 0.8496 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 82.9 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).
- CleanRL PPO-RND: best $W$ 0.8781 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 119 thousand steps at intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (of 8 configurations).

## Predictions

The predictions were written in `ideas.md` before the runs. Each is quoted, then read part by part: a part held or did not hold on the numbers given, or cannot be read yet because a unit it needs is not complete. A prediction held when every part held, and did not hold when at least one part did not. Where a prediction leaves a choice, it is read as follows. Every reading is on the first validation block (seeds 10000 to 10255), paired by seed. "The hold's own best configuration" in a case is, among the configurations both this arm and the hold alone trained, the one at which the hold alone has its highest $W$. "Within the interval of the difference" and "level" are read as: the 95% interval of the paired difference holds zero; "ahead" and "behind" as: the interval is entirely above or below zero; "at least as good" as: not behind. "Not behind by more than 0.003", "by 0.01 to 0.02" and "within 0.03" are read on the mean difference. "RND's best" is RND at its best configuration for that metric among all those run on the same seeds.

### Prediction 1: the first-goal time is that of batch 10's arm at the same configuration

> **The first-goal time is that of batch 10's arm at the same configuration** (the same program until the first arrival): 15.7 and 14.7 thousand steps at the two configurations of Medium with reward 1 on arrival, 12.9 on Medium with the step cost at (6e-3, 0.01, 2), 86 on Large with reward 1 on arrival at weight 16; each within the interval of the difference.

Observed: **held**.

- Medium with reward 1 on arrival (6e-3, weight 4): **held**. At (learning rate 0.006, weight 4): $M$ 14.6 thousand steps against 15.7 thousand steps; difference -1.07 [-2.95, +0.783] thousand steps.
- Medium with reward 1 on arrival (1e-2, weight 4): **held**. At (learning rate 0.01, weight 4): $M$ 14 thousand steps against 14.7 thousand steps; difference -0.666 [-2.89, +1.48] thousand steps.
- Medium with the step cost (6e-3, weight 2): **held**. At (learning rate 0.006, weight 2): $M$ 12.7 thousand steps against 12.9 thousand steps; difference -0.257 [-1.76, +1.22] thousand steps.
- Large with reward 1 on arrival (2e-3, weight 16): **held**. At (learning rate 0.002, weight 16): $M$ 89.1 thousand steps against 86.1 thousand steps; difference +3.02 [-15.1, +23.1] thousand steps.

### Prediction 2: the cost after the arrival is gone

> **The cost after the arrival is gone.** At the hold's own best configurations the whole-curve mean is not behind the hold alone by more than 0.003 in any case (batch 10's arm was behind by 0.101 and 0.021).

Observed: **did not hold**.

- Medium with reward 1 on arrival: $W$ not behind the hold alone by more than 0.003 at the hold's best configuration: **did not hold**. At (learning rate 0.006, weight 4): $W$ 0.7001 against 0.9435; difference -0.2433 [-0.2551, -0.2311].
- Medium with the step cost: $W$ not behind the hold alone by more than 0.003 at the hold's best configuration: **did not hold**. At (learning rate 0.006, weight 2): $W$ -0.3821 against -0.3484; difference -0.0337 [-0.0381, -0.0295].
- Large with reward 1 on arrival: $W$ not behind the hold alone by more than 0.003 at the hold's best configuration: **did not hold**. At (learning rate 0.002, weight 2): $W$ 0.8496 against 0.8874; difference -0.0378 [-0.0457, -0.0304].

### Prediction 3: PointMaze Medium

> **PointMaze Medium:** level with the hold alone on the whole-curve mean (the goal is found only about 3 thousand steps sooner, 0.3% of the budget) and ahead of RND's best as the hold is.

Observed: **did not hold**.

- reward 1 on arrival: level with the hold alone on $W$: **did not hold**. At (learning rate 0.006, weight 4): $W$ 0.7001 against 0.9435; difference -0.2433 [-0.2551, -0.2311].
- reward 1 on arrival: ahead of RND's best, as the hold is: **did not hold**. At (learning rate 0.006, weight 4): $W$ 0.7001; difference from RND's best -0.2348 [-0.2466, -0.2230].
- step cost: level with the hold alone on $W$: **did not hold**. At (learning rate 0.006, weight 2): $W$ -0.3821 against -0.3484; difference -0.0337 [-0.0381, -0.0295].
- step cost: ahead of RND's best, as the hold is: **did not hold**. At (learning rate 0.006, weight 2): $W$ -0.3821; difference from RND's best -0.0316 [-0.0361, -0.0271].

### Prediction 4: PointMaze Large with reward 1 on arrival

> **PointMaze Large with reward 1 on arrival:** ahead of RND's best (0.8781) with the interval above zero at weight 4 or 16, by 0.01 to 0.02, because the first arrival comes 1 to 2% of the budget sooner. I am least sure of this one: after the switch a weight of 16 may still be too large for the policy to settle, as the hold alone already loses 0.002 from weight 2 to weight 4.

Observed: **did not hold**.

- Ahead of RND's best with the interval above zero at weight 4 or 16, by 0.01 to 0.02: **did not hold**. At (learning rate 0.002, weight 4): $W$ 0.7871; difference from RND's best -0.0909 [-0.1005, -0.0816]. At (learning rate 0.002, weight 16): $W$ 0.3370; difference from RND's best -0.5411 [-0.5593, -0.5228].

### Prediction 5: without the hold

> **Without the hold** the arm is behind the arm with the hold on the final level in both cases and level with it on the first-goal time.

Observed: **did not hold**.

- Medium with reward 1 on arrival (6e-3, weight 4): behind the arm with the hold on $L$: **held**. At (learning rate 0.006, weight 4): $L$ 0.6323 against 0.9414; difference -0.3091 [-0.3290, -0.2897].
- Medium with reward 1 on arrival (6e-3, weight 4): level with it on $M$: **did not hold**. At (learning rate 0.006, weight 4): $M$ 17.4 thousand steps against 14.6 thousand steps; difference +2.75 [+0.543, +5.04] thousand steps.
- Large with reward 1 on arrival (2e-3, weight 4): behind the arm with the hold on $L$: **held**. At (learning rate 0.002, weight 4): $L$ 0.2376 against 0.9770; difference -0.7394 [-0.7532, -0.7253].
- Large with reward 1 on arrival (2e-3, weight 4): level with it on $M$: **held**. At (learning rate 0.002, weight 4): $M$ 91.2 thousand steps against 82.9 thousand steps; difference +8.26 [-5.68, +23.5] thousand steps.

### Prediction 6: addendum 1: the arm with RND's running divisor against RND at the same configuration

> Under reading (a), the arm with RND's running divisor is within 0.03 of RND at the same configuration on Medium with reward 1 on arrival (RND: 0.9349 at weight 2), where the arm with the first-arrival hold is 0.24 behind. Under reading (b) it stays more than 0.05 behind. I expect (a), with a small loss left from the rollouts in which the running divisor is still catching up.

Observed: **did not hold**.

- Medium with reward 1 on arrival (6e-3, weight 2): within 0.03 of RND on $W$: **did not hold**. At (learning rate 0.006, weight 2): $W$ 0.6403 against 0.9349; difference -0.2946 [-0.3110, -0.2780].
- Beside: At weight 4: At (learning rate 0.006, weight 4): $W$ 0.4487 against 0.9194; difference -0.4707 [-0.4833, -0.4580].

### Prediction 7: addendum 1: the divisor held after one sixteenth against the running divisor

> The arm with the divisor held after 1/16 is at least as good as the arm with the running divisor at the same configuration, as the fixed hold is for RND.

Observed: **held**.

- Medium with reward 1 on arrival (6e-3, weight 4): at least as good on $W$: **held**. At (learning rate 0.006, weight 4): $W$ 0.6610 against 0.4487; difference +0.2123 [+0.1963, +0.2281].
- Large with reward 1 on arrival (2e-3, weight 4): at least as good on $W$: **held**. At (learning rate 0.002, weight 4): $W$ 0.5718 against 0.1356; difference +0.4362 [+0.4264, +0.4458].

### Prediction 8: addendum 1: the running divisor on PointMaze Large with reward 1 on arrival

> On Large with reward 1 on arrival the arm with the running divisor keeps the early first arrival (below 150 thousand steps at weights 4 and 16, against 269 for RND's best) and its whole-curve mean is ahead of RND's best (0.8781) at one of the three weights. I am unsure: at weight 16 RND itself may settle too slowly.

Observed: **did not hold**.

- First arrival within 150 thousand steps at weight 4: **held**. At (learning rate 0.002, weight 4): $M$ 91.2 thousand steps.
- First arrival within 150 thousand steps at weight 16: **held**. At (learning rate 0.002, weight 16): $M$ 83 thousand steps.
- $W$ ahead of RND's best at one of the three weights: **did not hold**. At (learning rate 0.002, weight 2): $W$ 0.4258; difference from RND's best -0.4522 [-0.4640, -0.4406]. At (learning rate 0.002, weight 4): $W$ 0.1356; difference from RND's best -0.7425 [-0.7508, -0.7337]. At (learning rate 0.002, weight 16): $W$ 0.0230; difference from RND's best -0.8551 [-0.8590, -0.8510].

