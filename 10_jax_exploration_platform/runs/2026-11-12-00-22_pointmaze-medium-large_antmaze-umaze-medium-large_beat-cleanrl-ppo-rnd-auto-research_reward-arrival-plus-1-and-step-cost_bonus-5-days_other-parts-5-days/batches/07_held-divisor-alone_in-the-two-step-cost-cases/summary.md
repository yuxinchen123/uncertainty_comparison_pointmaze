# Batch 7: the held divisor alone in the two step-cost cases

Generated 2026-11-13 00:22 PT by the command below. This file is generated: regenerate it, do not edit it. Times are Pacific, converted from machines running Eastern. Run folder: `2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days`; batch folder: `batches/07_held-divisor-alone_in-the-two-step-cost-cases`.

```bash
cd /p/rlprojects/RND/10_jax_exploration_platform
bash "benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh" -s scripts/maze_campaign/analysis/batch_summary_job.py \
    --run-dir=runs/2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days \
    --batch=07_held-divisor-alone_in-the-two-step-cost-cases
```

## Contents

| section | first written | last changed | status |
|---|---|---|---|
| [Measures](#measures) | 2026-11-13 00:16 PT | 2026-11-13 00:16 PT | unread |
| [Validation seeds, case by case](#validation-seeds-case-by-case) | 2026-11-13 00:16 PT | 2026-11-13 00:22 PT | unread |
| [Predictions](#predictions) | 2026-11-13 00:16 PT | 2026-11-13 00:22 PT | unread |

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

### PointMaze Medium, step cost

#### First block: each side at its best configuration there

The units of the phase `validation-block` in PointMaze Medium, step cost. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006 | -0.3505 ± 0.0016 | -0.3078 ± 0.0007 | 20 ± 1.17 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.0078125, learning rate 0.006 | -0.3514 ± 0.0018 | -0.3074 ± 0.0011 | 20.9 ± 1.27 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.01 | -0.3549 ± 0.0018 | -0.3078 ± 0.0011 | 20.1 ± 1 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 | **-0.3463** ± 0.0012 | <u>-0.3066</u> ± 0.0005 | 13.9 ± 0.528 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.006 | -0.3495 ± 0.0018 | -0.3076 ± 0.0011 | 16.5 ± 0.757 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 0.03125, learning rate 0.006 | -0.3499 ± 0.0016 | **-0.3061** ± 0.0005 | 20.1 ± 0.96 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 0.125, learning rate 0.006 | -0.3518 ± 0.0020 | -0.3074 ± 0.0012 | 21.3 ± 1.14 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 0.00195312, learning rate 0.006 | -0.3518 ± 0.0021 | -0.3089 ± 0.0015 | 21.5 ± 1.01 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 | -0.3520 ± 0.0014 | -0.3070 ± 0.0005 | 14.6 ± 0.653 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 0.03125, learning rate 0.01 | -0.3522 ± 0.0014 | <u>-0.3066</u> ± 0.0004 | 19.7 ± 0.991 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 0.0078125, learning rate 0.006 | -0.3534 ± 0.0020 | -0.3086 ± 0.0009 | 20.9 ± 1.02 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 0.125, learning rate 0.01 | -0.3537 ± 0.0015 | -0.3072 ± 0.0005 | 19.1 ± 0.864 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.006 | -0.3667 ± 0.0018 | -0.3095 ± 0.0010 | 15.7 ± 0.771 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 16, learning rate 0.006 | -0.3935 ± 0.0025 | -0.3142 ± 0.0006 | 17.6 ± 1.05 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.006 | <u>-0.3491</u> ± 0.0018 | -0.3076 ± 0.0005 | 13.7 ± 0.573 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 0.0078125, learning rate 0.006 | -0.3539 ± 0.0020 | -0.3079 ± 0.0011 | 22 ± 1.17 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 0.00195312, learning rate 0.006 | -0.3544 ± 0.0020 | -0.3071 ± 0.0007 | 24.3 ± 1.84 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 2, learning rate 0.006 | -0.3547 ± 0.0021 | -0.3089 ± 0.0012 | <u>13.4</u> ± 0.627 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 4, learning rate 0.006 | -0.3637 ± 0.0025 | -0.3096 ± 0.0010 | **13** ± 0.585 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006 (best of 3) | -0.3505 ± 0.0016 | control | -0.3078 ± 0.0007 | control | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 (best of 11) | **-0.3463** ± 0.0012 | **+0.0042** [+0.0005, +0.0080] | **-0.3066** ± 0.0005 | **+0.0013** [-0.0004, +0.0030] | 256 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.006 (best of 5) | <u>-0.3491</u> ± 0.0018 | <u>+0.0014</u> [-0.0033, +0.0059] | <u>-0.3076</u> ± 0.0005 | <u>+0.0003</u> [-0.0015, +0.0020] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006 (best of 3) | 20 ± 1.17 | control | 256 of 256 | <u>-0.3505</u> | <u>-0.3078</u> | 256 |
| predictor at half speed with held divisor | intrinsic weight 4, learning rate 0.006 (best of 5) | **13** ± 0.585 | **-6.97** [-9.67, -4.65] | 256 of 256 | -0.3637 | -0.3096 | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 (best of 11) | <u>13.9</u> ± 0.528 | <u>-6.13</u> [-8.63, -4.1] | 256 of 256 | **-0.3463** | **-0.3066** | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:58 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:53 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:54 PT | none |
| 4 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:54 PT | none |
| 5 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:58 PT | none |
| 6 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:58 PT | none |
| 7 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:12 PT | none |
| 8 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:14 PT | none |
| 9 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:34 PT | none |
| 10 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:16 PT | none |
| 11 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:34 PT | none |
| 12 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:35 PT | none |
| 13 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:18 PT | none |
| 14 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:19 PT | none |
| 15 | predictor at half speed with held divisor | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:02 PT | none |
| 16 | predictor at half speed with held divisor | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:02 PT | none |
| 17 | predictor at half speed with held divisor | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 00:13 PT | none |
| 18 | predictor at half speed with held divisor | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 00:12 PT | none |
| 19 | predictor at half speed with held divisor | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 00:12 PT | none |

Unit names, by the number in the first column:

1. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
2. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.0078125_clip-coefficient-0.1_seeds-10000-to-10255`
3. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.0078125_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.03125_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.125_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.03125_clip-coefficient-0.1_seeds-10000-to-10255`
14. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.125_clip-coefficient-0.1_seeds-10000-to-10255`
15. `validation-block_pointmaze-medium_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
16. `validation-block_pointmaze-medium_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.0078125_clip-coefficient-0.1_seeds-10000-to-10255`
17. `validation-block_pointmaze-medium_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
18. `validation-block_pointmaze-medium_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
19. `validation-block_pointmaze-medium_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phase `validation-block-two` in PointMaze Medium, step cost. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.0078125, learning rate 0.006 | <u>-0.3510</u> ± 0.0016 | -0.3078 ± 0.0007 | 20.4 ± 0.931 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006 | -0.3535 ± 0.0024 | -0.3082 ± 0.0012 | 19.9 ± 1.01 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.01 | -0.3541 ± 0.0014 | -0.3073 ± 0.0006 | 20.6 ± 0.841 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 | **-0.3477** ± 0.0012 | **-0.3062** ± 0.0004 | <u>15</u> ± 0.68 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 | -0.3514 ± 0.0014 | <u>-0.3069</u> ± 0.0004 | **13.8** ± 0.593 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.006 | -0.3680 ± 0.0021 | -0.3089 ± 0.0003 | 15.9 ± 0.715 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND), at its first-block best | intrinsic weight 0.00195312, learning rate 0.006 (its best on the first block) | -0.3535 ± 0.0024 | control | -0.0025 [-0.0080, +0.0025] | -0.3082 ± 0.0012 | control | 256 |
| control (CleanRL PPO-RND), at its best on this block | intrinsic weight 0.0078125, learning rate 0.006 (best of 3 on this block) | <u>-0.3510</u> ± 0.0016 | +0.0025 [-0.0026, +0.0080] | control | <u>-0.3078</u> ± 0.0007 | +0.0004 [-0.0023, +0.0035] | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 (its best on the first block) | **-0.3477** ± 0.0012 | +0.0058 [+0.0007, +0.0115] | +0.0033 [-0.0004, +0.0069] | **-0.3062** ± 0.0004 | +0.0020 [-0.0002, +0.0047] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006 (its best on the first block and on this block) | <u>19.9</u> ± 1.01 | control | control | 256 of 256 | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 (its best on the first block) | **15** ± 0.68 | -4.9 [-7.17, -2.73] | -4.9 [-7.17, -2.73] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:36 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:38 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:38 PT | none |
| 4 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:39 PT | none |
| 5 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:40 PT | none |
| 6 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:40 PT | none |

Unit names, by the number in the first column:

1. `validation-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.0078125_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-medium_reward-step-cost_divisor-held-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
5. `validation-block-two_pointmaze-medium_reward-step-cost_divisor-held-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
6. `validation-block-two_pointmaze-medium_reward-step-cost_divisor-held-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`

### PointMaze Large, step cost

#### First block: each side at its best configuration there

The units of the phase `validation-block` in PointMaze Large, step cost. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 33,554,432 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 | -0.7310 ± 0.0018 | -0.7163 ± 0.0017 | 191 ± 10.1 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 | -0.7411 ± 0.0036 | -0.7229 ± 0.0033 | 241 ± 13.4 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002 | **-0.7215** ± 0.0008 | **-0.7025** ± 0.0009 | 182 ± 9.19 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002 | -0.7255 ± 0.0023 | -0.7098 ± 0.0022 | 204 ± 10.3 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.003 | -0.7270 ± 0.0003 | -0.7055 ± 0.0002 | 147 ± 6.83 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 8, learning rate 0.003 | <u>-0.7216</u> ± 0.0014 | <u>-0.7053</u> ± 0.0014 | **132** ± 9.08 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 4, learning rate 0.003 | -0.7370 ± 0.0036 | -0.7208 ± 0.0035 | <u>142</u> ± 9.75 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 (best of 2) | -0.7310 ± 0.0018 | control | -0.7163 ± 0.0017 | control | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002 (best of 3) | **-0.7215** ± 0.0008 | **+0.0095** [+0.0057, +0.0137] | **-0.7025** ± 0.0009 | **+0.0139** [+0.0102, +0.0178] | 256 |
| predictor at half speed with held divisor | intrinsic weight 8, learning rate 0.003 (best of 2) | <u>-0.7216</u> ± 0.0014 | <u>+0.0094</u> [+0.0048, +0.0140] | <u>-0.7053</u> ± 0.0014 | <u>+0.0111</u> [+0.0067, +0.0155] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 (best of 2) | 191 ± 10.1 | control | 256 of 256 | -0.7310 | -0.7163 | 256 |
| predictor at half speed with held divisor | intrinsic weight 8, learning rate 0.003 (best of 2) | **132** ± 9.08 | **-58.9** [-85.7, -32] | 256 of 256 | **-0.7216** | **-0.7053** | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.003 (best of 3) | <u>147</u> ± 6.83 | <u>-43.8</u> [-68, -20.2] | 256 of 256 | <u>-0.7270</u> | <u>-0.7055</u> | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:25 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:03 PT | none |
| 3 | held divisor alone, held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:50 PT | none |
| 4 | held divisor alone, held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 00:19 PT | none |
| 5 | held divisor alone, held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:54 PT | none |
| 6 | predictor at half speed with held divisor | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:51 PT | none |
| 7 | predictor at half speed with held divisor | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:27 PT | none |

Unit names, by the number in the first column:

1. `validation-block_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `validation-block_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `validation-block_pointmaze-large_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-large_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-large_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-large_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-large_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`

## Predictions

The predictions were written in `ideas.md` before the runs. Each is quoted, then read part by part: a part held or did not hold on the numbers given, or cannot be read yet because a unit it needs is not complete. A prediction held when every part held, and did not hold when at least one part did not. Where a prediction leaves a choice, it is read as follows. The held divisor alone stands at its best configuration for the whole-curve mean $W$ among those run on the first validation block, and the control at its own; a part waits until every configuration planned there for both is read. "Ahead with its interval above zero" is read as: the 95% interval of the difference in $W$ is entirely above zero. "Level with RND (within 0.005)" is read as: the mean difference in $W$ is at most 0.005 in size. "Behind RND's best" at the weight 0.03125 is read as: the configuration at learning rate 0.006 and that weight has a mean difference in $W$ below zero from the control at its best configuration.

### Prediction 1: PointMaze Large, step cost

> **PointMaze Large, step cost:** the held divisor alone is ahead of RND on the whole-curve mean with its interval above zero, because RND's seeds there split into two groups and the worse group is the one that keeps exploring late.

Observed: **held**.

- The held divisor alone is ahead of the control with its interval above zero: **held**. $W$ -0.7215, difference from the control +0.0095 [+0.0057, +0.0137].

### Prediction 2: PointMaze Medium, step cost

> **PointMaze Medium, step cost:** level with RND (within 0.005), because RND's best weight there is nearly zero and a held divisor changes little when the bonus hardly counts; with the larger weight 0.03125 the held divisor is behind RND's best.

Observed: **did not hold**.

- The held divisor alone is level with the control (within 0.005): **held**. $W$ -0.3463, difference from the control +0.0042 [+0.0005, +0.0080]; the interval stands entirely above zero.
- At the larger weight 0.03125 it is behind the control's best: **did not hold**. Difference from the control at its best configuration +0.0006 [-0.0037, +0.0050].
- Beside: The held divisor's best intrinsic weight on the first block is 2; the control's is 0.00195312.

