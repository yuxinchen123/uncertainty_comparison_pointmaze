# Batch 8: the divisor held from each policy's first arrival, and the error length with a held divisor

Generated 2026-11-13 11:35 PT by the command below. This file is generated: regenerate it, do not edit it. Times are Pacific, converted from machines running Eastern. Run folder: `2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days`; batch folder: `batches/08_divisor-held-from-the-first-arrival_and-error-length-with-held-divisor`.

```bash
cd /p/rlprojects/RND/10_jax_exploration_platform
bash "benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh" -s scripts/maze_campaign/analysis/batch_summary_job.py \
    --run-dir=runs/2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days \
    --batch=08_divisor-held-from-the-first-arrival_and-error-length-with-held-divisor
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
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 0.8718 ± 0.0034 | 0.9537 ± 0.0028 | <u>17.6</u> ± 0.96 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.01 | 0.6440 ± 0.0073 | 0.7891 ± 0.0078 | 18.3 ± 0.963 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 | 0.0073 ± 0.0003 | 0.0000 ± 0.0000 | **17.5** ± 1.41 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | **0.9435** ± 0.0027 | <u>0.9999</u> ± 0.0000 | 18.5 ± 1.13 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.01, entropy coefficient 0.01 | <u>0.9412</u> ± 0.0024 | **1.0000** ± 0.0000 | 23.1 ± 1.34 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9334 ± 0.0029 | **1.0000** ± 0.0000 | 29.4 ± 1.85 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9382 ± 0.0027 | 0.9995 ± 0.0003 | 21 ± 1.22 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9374 ± 0.0027 | <u>0.9999</u> ± 0.0000 | 28.6 ± 1.69 | 256 of 256 |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.006, entropy coefficient 0.01 | 0.9256 ± 0.0029 | 0.9962 ± 0.0003 | 31.9 ± 2.06 | 256 of 256 |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 0.25, learning rate 0.006, entropy coefficient 0.01 | 0.9177 ± 0.0032 | 0.9994 ± 0.0001 | 47 ± 2.3 | 256 of 256 |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 0.125, learning rate 0.006, entropy coefficient 0.01 | 0.8794 ± 0.0053 | 0.9998 ± 0.0001 | 72.6 ± 3.84 | 256 of 256 |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 | 0.8766 ± 0.0031 | 0.9801 ± 0.0009 | 23.8 ± 1.2 | 256 of 256 |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.7321 ± 0.0034 | 0.9197 ± 0.0022 | 19.8 ± 0.945 | 256 of 256 |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.5416 ± 0.0030 | 0.7647 ± 0.0050 | 18.2 ± 0.836 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 6) | 0.9349 ± 0.0031 | control | 0.9950 ± 0.0010 | control | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | **0.9435** ± 0.0027 | **+0.0085** [+0.0009, +0.0161] | **0.9999** ± 0.0000 | **+0.0049** [+0.0033, +0.0070] | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 2) | <u>0.9382</u> ± 0.0027 | <u>+0.0033</u> [-0.0044, +0.0109] | <u>0.9995</u> ± 0.0003 | <u>+0.0045</u> [+0.0027, +0.0066] | 256 |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.006, entropy coefficient 0.01 (best of 6) | 0.9256 ± 0.0029 | -0.0093 [-0.0169, -0.0018] | 0.9962 ± 0.0003 | +0.0012 [-0.0006, +0.0033] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (best of 6) | **17.5** ± 1.41 | control | 256 of 256 | 0.0073 | 0.0000 | 256 |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 6) | <u>18.2</u> ± 0.836 | **+0.672** [-2.86, +3.61] | 256 of 256 | 0.5416 | 0.7647 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | 18.5 ± 1.13 | <u>+0.993</u> [-2.74, +4.35] | 256 of 256 | **0.9435** | **0.9999** | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 2) | 21 ± 1.22 | +3.47 [-0.336, +6.84] | 256 of 256 | <u>0.9382</u> | <u>0.9995</u> | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:14 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:15 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:14 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 5 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:53 PT | none |
| 6 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 7 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:32 PT | none |
| 8 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:32 PT | none |
| 9 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:34 PT | none |
| 10 | divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:44 PT | none |
| 11 | divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:45 PT | none |
| 12 | error length as the bonus, divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:55 PT | none |
| 13 | error length as the bonus, divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:55 PT | none |
| 14 | error length as the bonus, divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:56 PT | none |
| 15 | error length as the bonus, divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:35 PT | none |
| 16 | error length as the bonus, divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:34 PT | none |
| 17 | error length as the bonus, divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:35 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.1_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-medium_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.125_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-medium_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.25_clip-coefficient-0.1_seeds-10000-to-10255`
14. `validation-block_pointmaze-medium_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`
15. `validation-block_pointmaze-medium_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
16. `validation-block_pointmaze-medium_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
17. `validation-block_pointmaze-medium_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phases `validation-block-two` and `discovery-block-two` in PointMaze Medium, reward 1 on arrival. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9308 ± 0.0030 | 0.9907 ± 0.0039 | 26.3 ± 1.88 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.01, entropy coefficient 0.01 | 0.9276 ± 0.0026 | 0.9798 ± 0.0059 | 23.6 ± 1.4 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9141 ± 0.0030 | 0.9821 ± 0.0018 | <u>19.6</u> ± 1.14 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 | 0.0075 ± 0.0003 | 0.0000 ± 0.0000 | **16.8** ± 1.43 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | **0.9404** ± 0.0030 | **0.9999** ± 0.0000 | 20.6 ± 1.12 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | <u>0.9398</u> ± 0.0025 | <u>0.9995</u> ± 0.0002 | 19.7 ± 1.13 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (its best on the first block and on this block) | 0.9308 ± 0.0030 | control | control | 0.9907 ± 0.0039 | control | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | **0.9404** ± 0.0030 | **+0.0095** [+0.0013, +0.0177] | **+0.0095** [+0.0013, +0.0177] | **0.9999** ± 0.0000 | **+0.0092** [+0.0042, +0.0180] | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | <u>0.9398</u> ± 0.0025 | <u>+0.0090</u> [+0.0023, +0.0159] | <u>+0.0090</u> [+0.0023, +0.0159] | <u>0.9995</u> ± 0.0002 | <u>+0.0088</u> [+0.0038, +0.0174] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (its best on the first block and on this block) | **16.8** ± 1.43 | control | control | 256 of 256 | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | <u>19.7</u> ± 1.13 | **+2.89** [-0.835, +6.21] | **+2.89** [-0.835, +6.21] | 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | 20.6 ± 1.12 | <u>+3.83</u> [+0.451, +7.02] | <u>+3.83</u> [+0.451, +7.02] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 10:58 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:16 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:19 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:16 PT | none |
| 5 | divisor held from each policy's first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:31 PT | none |
| 6 | divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:22 PT | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.1_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
5. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
6. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_divisor-held-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`

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
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | <u>0.8874</u> ± 0.0023 | <u>0.9977</u> ± 0.0002 | 259 ± 10.6 | **256 of 256** |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.8852 ± 0.0023 | 0.9976 ± 0.0003 | 201 ± 9.11 | **256 of 256** |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 0.8680 ± 0.0033 | **0.9979** ± 0.0003 | 375 ± 12.4 | **256 of 256** |
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | **0.8876** ± 0.0021 | 0.9976 ± 0.0003 | 270 ± 10.5 | **256 of 256** |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.8821 ± 0.0020 | 0.9970 ± 0.0002 | 213 ± 8.15 | **256 of 256** |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 0.25, learning rate 0.002, entropy coefficient 0.01 | 0.8714 ± 0.0016 | 0.9943 ± 0.0003 | 259 ± 8.89 | **256 of 256** |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 0.125, learning rate 0.002, entropy coefficient 0.01 | 0.8695 ± 0.0019 | 0.9965 ± 0.0003 | 399 ± 12 | **256 of 256** |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.002, entropy coefficient 0.01 | 0.8278 ± 0.0015 | 0.9883 ± 0.0005 | 176 ± 6.38 | **256 of 256** |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 0.6557 ± 0.0026 | 0.9162 ± 0.0020 | 125 ± 5 | **256 of 256** |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.3521 ± 0.0030 | 0.6369 ± 0.0049 | <u>103</u> ± 4.22 | **256 of 256** |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.1189 ± 0.0018 | 0.2435 ± 0.0045 | **93** ± 4.07 | **256 of 256** |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 8) | 0.8781 ± 0.0021 | control | 0.9920 ± 0.0005 | control | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 2) | **0.8876** ± 0.0021 | **+0.0095** [+0.0041, +0.0151] | <u>0.9976</u> ± 0.0003 | <u>+0.0056</u> [+0.0045, +0.0068] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 (best of 3) | <u>0.8874</u> ± 0.0023 | <u>+0.0094</u> [+0.0032, +0.0156] | **0.9977** ± 0.0002 | **+0.0057** [+0.0046, +0.0068] | 256 |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 0.25, learning rate 0.002, entropy coefficient 0.01 (best of 6) | 0.8714 ± 0.0016 | -0.0066 [-0.0116, -0.0018] | 0.9943 ± 0.0003 | +0.0023 [+0.0012, +0.0034] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (best of 8) | <u>119</u> ± 5.46 | control | 256 of 256 | 0.0054 | 0.0000 | 256 |
| error length as the bonus, divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (best of 6) | **93** ± 4.07 | **-26.3** [-39.2, -13.5] | 256 of 256 | 0.1189 | 0.2435 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (best of 3) | 201 ± 9.11 | <u>+81.6</u> [+61.3, +102] | 256 of 256 | **0.8852** | **0.9976** | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (best of 2) | 213 ± 8.15 | +93.9 [+74.5, +113] | 256 of 256 | <u>0.8821</u> | <u>0.9970</u> | 256 |

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
| 9 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:55 PT | none |
| 10 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:55 PT | none |
| 11 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:07 PT | none |
| 12 | divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 16:41 PT | none |
| 13 | divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 17:02 PT | none |
| 14 | error length as the bonus, divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:04 PT | none |
| 15 | error length as the bonus, divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:04 PT | none |
| 16 | error length as the bonus, divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:11 PT | none |
| 17 | error length as the bonus, divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:40 PT | none |
| 18 | error length as the bonus, divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:45 PT | none |
| 19 | error length as the bonus, divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:45 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.001_entropy-coefficient-0.03_intrinsic-weight-256.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.001_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-after-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-after-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
14. `validation-block_pointmaze-large_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-0.125_clip-coefficient-0.1_seeds-10000-to-10255`
15. `validation-block_pointmaze-large_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-0.25_clip-coefficient-0.1_seeds-10000-to-10255`
16. `validation-block_pointmaze-large_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`
17. `validation-block_pointmaze-large_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
18. `validation-block_pointmaze-large_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
19. `validation-block_pointmaze-large_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phases `validation-block-two` and `discovery-block-two` in PointMaze Large, reward 1 on arrival. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 8,388,608 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 | **0.8842** ± 0.0019 | 0.9927 ± 0.0005 | 266 ± 10.6 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006 | 0.0057 ± 0.0002 | 0.0003 ± 0.0002 | **130** ± 5.17 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002 | <u>0.8839</u> ± 0.0025 | **0.9976** ± 0.0003 | 266 ± 11.6 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002 | 0.8819 ± 0.0030 | <u>0.9969</u> ± 0.0005 | <u>259</u> ± 10 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 (its best on the first block and on this block) | **0.8842** ± 0.0019 | control | control | 0.9927 ± 0.0005 | control | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002 (its best on the first block) | <u>0.8839</u> ± 0.0025 | **-0.0003** [-0.0066, +0.0061] | **-0.0003** [-0.0066, +0.0061] | **0.9976** ± 0.0003 | **+0.0049** [+0.0038, +0.0059] | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002 (its best on the first block) | 0.8819 ± 0.0030 | <u>-0.0023</u> [-0.0097, +0.0044] | <u>-0.0023</u> [-0.0097, +0.0044] | <u>0.9969</u> ± 0.0005 | <u>+0.0042</u> [+0.0028, +0.0055] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006 (its best on the first block and on this block) | **130** ± 5.17 | control | control | 256 of 256 | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002 (its best on this block: its first-block best, intrinsic weight 4, learning rate 0.002, was not run here) | <u>259</u> ± 10 | **+129** [+108, +151] | **+129** [+108, +151] | 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.002 (its best on this block: its first-block best, intrinsic weight 4, learning rate 0.002, was not run here) | 266 ± 11.6 | <u>+136</u> [+112, +161] | <u>+136</u> [+112, +161] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 11:13 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:45 PT | none |
| 3 | divisor held from each policy's first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:53 PT | none |
| 4 | divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:53 PT | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-large_reward-arrival-plus-one_held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-large_reward-arrival-plus-one_divisor-held-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`

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
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 | <u>-0.3476</u> ± 0.0014 | -0.3073 ± 0.0007 | 15.4 ± 0.693 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3484 ± 0.0014 | **-0.3058** ± 0.0004 | 14.2 ± 0.617 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | -0.3504 ± 0.0015 | -0.3066 ± 0.0005 | <u>13.4</u> ± 0.562 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | **-0.3463** ± 0.0012 | -0.3066 ± 0.0005 | 13.9 ± 0.528 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.006, entropy coefficient 0.01 | -0.3495 ± 0.0018 | -0.3076 ± 0.0011 | 16.5 ± 0.757 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 0.03125, learning rate 0.006, entropy coefficient 0.01 | -0.3499 ± 0.0016 | <u>-0.3061</u> ± 0.0005 | 20.1 ± 0.96 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 0.125, learning rate 0.006, entropy coefficient 0.01 | -0.3518 ± 0.0020 | -0.3074 ± 0.0012 | 21.3 ± 1.14 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01 | -0.3518 ± 0.0021 | -0.3089 ± 0.0015 | 21.5 ± 1.01 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | -0.3520 ± 0.0014 | -0.3070 ± 0.0005 | 14.6 ± 0.653 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 0.03125, learning rate 0.01, entropy coefficient 0.01 | -0.3522 ± 0.0014 | -0.3066 ± 0.0004 | 19.7 ± 0.991 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 0.0078125, learning rate 0.006, entropy coefficient 0.01 | -0.3534 ± 0.0020 | -0.3086 ± 0.0009 | 20.9 ± 1.02 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 0.125, learning rate 0.01, entropy coefficient 0.01 | -0.3537 ± 0.0015 | -0.3072 ± 0.0005 | 19.1 ± 0.864 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 8, learning rate 0.006, entropy coefficient 0.01 | -0.3667 ± 0.0018 | -0.3095 ± 0.0010 | 15.7 ± 0.771 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 16, learning rate 0.006, entropy coefficient 0.01 | -0.3935 ± 0.0025 | -0.3142 ± 0.0006 | 17.6 ± 1.05 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01 (best of 6) | -0.3505 ± 0.0016 | control | -0.3078 ± 0.0007 | control | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 11) | **-0.3463** ± 0.0012 | **+0.0042** [+0.0006, +0.0081] | **-0.3066** ± 0.0005 | **+0.0013** [-0.0004, +0.0030] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 (best of 3) | <u>-0.3476</u> ± 0.0014 | <u>+0.0029</u> [-0.0015, +0.0072] | <u>-0.3073</u> ± 0.0007 | <u>+0.0005</u> [-0.0014, +0.0025] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (best of 6) | **12.4** ± 0.471 | control | 256 of 256 | -0.5526 | <u>-0.6884</u> | 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (best of 3) | <u>13.4</u> ± 0.562 | **+0.958** [-0.349, +2.26] | 256 of 256 | <u>-0.3504</u> | **-0.3066** | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (best of 11) | 13.9 ± 0.528 | <u>+1.46</u> [+0.281, +2.64] | 256 of 256 | **-0.3463** | **-0.3066** | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:16 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:19 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:18 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:58 PT | none |
| 5 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:53 PT | none |
| 6 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:54 PT | none |
| 7 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:35 PT | none |
| 8 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:35 PT | none |
| 9 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:34 PT | none |
| 10 | divisor held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:54 PT | none |
| 11 | divisor held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:58 PT | none |
| 12 | divisor held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 22:58 PT | none |
| 13 | divisor held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:12 PT | none |
| 14 | divisor held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:14 PT | none |
| 15 | divisor held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:34 PT | none |
| 16 | divisor held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:16 PT | none |
| 17 | divisor held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:34 PT | none |
| 18 | divisor held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:35 PT | none |
| 19 | divisor held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:18 PT | none |
| 20 | divisor held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:19 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.02_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.0078125_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.0078125_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.03125_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.125_clip-coefficient-0.1_seeds-10000-to-10255`
14. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`
15. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
16. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
17. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
18. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
19. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.03125_clip-coefficient-0.1_seeds-10000-to-10255`
20. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.125_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phases `validation-block-two` and `discovery-block-two` in PointMaze Medium, step cost. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.0078125, learning rate 0.006, entropy coefficient 0.01 | -0.3510 ± 0.0016 | -0.3078 ± 0.0007 | 20.4 ± 0.931 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01 | -0.3535 ± 0.0024 | -0.3082 ± 0.0012 | 19.9 ± 1.01 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.01, entropy coefficient 0.01 | -0.3541 ± 0.0014 | -0.3073 ± 0.0006 | 20.6 ± 0.841 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 | -0.5581 ± 0.0108 | -0.7035 ± 0.0206 | **13.2** ± 0.497 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01 | **-0.3470** ± 0.0015 | -0.3069 ± 0.0006 | 16.9 ± 0.742 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | <u>-0.3472</u> ± 0.0015 | **-0.3059** ± 0.0005 | 14.7 ± 0.632 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3477 ± 0.0012 | <u>-0.3062</u> ± 0.0004 | 15 ± 0.68 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | -0.3514 ± 0.0014 | -0.3069 ± 0.0004 | <u>13.8</u> ± 0.593 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 8, learning rate 0.006, entropy coefficient 0.01 | -0.3680 ± 0.0021 | -0.3089 ± 0.0003 | 15.9 ± 0.715 | 256 of 256 |

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
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | <u>-0.3477</u> ± 0.0012 | <u>+0.0058</u> [+0.0007, +0.0113] | <u>+0.0033</u> [-0.0004, +0.0069] | **-0.3062** ± 0.0004 | **+0.0020** [-0.0002, +0.0048] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (its best on the first block and on this block) | **13.2** ± 0.497 | control | control | 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (its best on this block: its first-block best, intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01, was not run here) | <u>14.7</u> ± 0.632 | **+1.46** [+0.109, +2.91] | **+1.46** [+0.109, +2.91] | 256 of 256 | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (its best on the first block) | 15 ± 0.68 | <u>+1.82</u> [+0.471, +3.18] | <u>+1.82</u> [+0.471, +3.18] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 11:00 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:36 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:38 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:38 PT | none |
| 5 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 02:31 PT | none |
| 6 | divisor held from each policy's first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 02:31 PT | none |
| 7 | divisor held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:39 PT | none |
| 8 | divisor held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:40 PT | none |
| 9 | divisor held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:40 PT | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.0078125_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-20000-to-20255`
5. `validation-block-two_pointmaze-medium_reward-step-cost_held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-20000-to-20255`
6. `validation-block-two_pointmaze-medium_reward-step-cost_held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
7. `validation-block-two_pointmaze-medium_reward-step-cost_divisor-held-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
8. `validation-block-two_pointmaze-medium_reward-step-cost_divisor-held-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
9. `validation-block-two_pointmaze-medium_reward-step-cost_divisor-held-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`

### PointMaze Large, step cost

#### First block: each side at its best configuration there

The units of the phases `validation-block` and `discovery-validation` in PointMaze Large, step cost. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 33,554,432 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 | -0.7310 ± 0.0018 | -0.7163 ± 0.0017 | 191 ± 10.1 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 | -0.7411 ± 0.0036 | -0.7229 ± 0.0033 | 241 ± 13.4 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.0045 | -0.8831 ± 0.0022 | -0.8849 ± 0.0026 | <u>131</u> ± 6.45 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006 | -0.9669 ± 0.0017 | -0.9803 ± 0.0022 | **120** ± 5.6 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 | <u>-0.7236</u> ± 0.0021 | -0.7079 ± 0.0021 | 163 ± 8.72 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.002 | -0.7268 ± 0.0025 | -0.7115 ± 0.0025 | 189 ± 9.38 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 4, learning rate 0.002 | -0.7335 ± 0.0034 | -0.7198 ± 0.0035 | 206 ± 11.5 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002 | **-0.7215** ± 0.0008 | **-0.7025** ± 0.0009 | 182 ± 9.19 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002 | -0.7255 ± 0.0023 | -0.7098 ± 0.0022 | 204 ± 10.3 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 8, learning rate 0.003 | -0.7270 ± 0.0003 | <u>-0.7055</u> ± 0.0002 | 147 ± 6.83 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 (best of 4) | -0.7310 ± 0.0018 | control | -0.7163 ± 0.0017 | control | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002 (best of 3) | **-0.7215** ± 0.0008 | **+0.0095** [+0.0058, +0.0137] | **-0.7025** ± 0.0009 | **+0.0139** [+0.0102, +0.0179] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 (best of 3) | <u>-0.7236</u> ± 0.0021 | <u>+0.0074</u> [+0.0018, +0.0129] | <u>-0.7079</u> ± 0.0021 | <u>+0.0084</u> [+0.0029, +0.0137] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006 (best of 4) | **120** ± 5.6 | control | 256 of 256 | -0.9669 | -0.9803 | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 8, learning rate 0.003 (best of 3) | <u>147</u> ± 6.83 | **+27.2** [+9.57, +44.7] | 256 of 256 | <u>-0.7270</u> | **-0.7055** | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 (best of 3) | 163 ± 8.72 | <u>+42.9</u> [+24.2, +62.7] | 256 of 256 | **-0.7236** | <u>-0.7079</u> | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:48 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:47 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:25 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:03 PT | none |
| 5 | divisor held from each policy's first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:32 PT | 1 continued from a checkpoint |
| 6 | divisor held from each policy's first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:04 PT | none |
| 7 | divisor held from each policy's first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:05 PT | none |
| 8 | divisor held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:50 PT | none |
| 9 | divisor held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 00:19 PT | none |
| 10 | divisor held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:54 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.0045_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `validation-block_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-large_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-large_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-large_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-large_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-large_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-large_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phases `validation-block-two` and `discovery-block-two` in PointMaze Large, step cost. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 33,554,432 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 | -0.7328 ± 0.0020 | -0.7175 ± 0.0021 | 203 ± 10.8 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006 | -0.9636 ± 0.0020 | -0.9753 ± 0.0025 | **121** ± 5.71 | 256 of 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 | <u>-0.7255</u> ± 0.0023 | <u>-0.7101</u> ± 0.0023 | <u>152</u> ± 7.71 | 256 of 256 |
| divisor held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002 | **-0.7232** ± 0.0011 | **-0.7045** ± 0.0011 | 184 ± 9.22 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 (its best on the first block and on this block) | -0.7328 ± 0.0020 | control | control | -0.7175 ± 0.0021 | control | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002 (its best on the first block) | **-0.7232** ± 0.0011 | **+0.0095** [+0.0052, +0.0141] | **+0.0095** [+0.0052, +0.0141] | **-0.7045** ± 0.0011 | **+0.0130** [+0.0084, +0.0180] | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 (its best on the first block) | <u>-0.7255</u> ± 0.0023 | <u>+0.0072</u> [+0.0013, +0.0131] | <u>+0.0072</u> [+0.0013, +0.0131] | <u>-0.7101</u> ± 0.0023 | <u>+0.0074</u> [+0.0013, +0.0135] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006 (its best on the first block and on this block) | **121** ± 5.71 | control | control | 256 of 256 | 256 |
| divisor held from each policy's first arrival | intrinsic weight 8, learning rate 0.003 (its best on the first block) | <u>152</u> ± 7.71 | **+31.5** [+11.9, +52.1] | **+31.5** [+11.9, +52.1] | 256 of 256 | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002 (its best on this block: its first-block best, intrinsic weight 8, learning rate 0.003, was not run here) | 184 ± 9.22 | <u>+63.3</u> [+43.9, +83.7] | <u>+63.3</u> [+43.9, +83.7] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 11:32 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 04:00 PT | none |
| 3 | divisor held from each policy's first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 04:25 PT | none |
| 4 | divisor held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 04:25 PT | none |

Unit names, by the number in the first column:

1. `discovery-block-two_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-large_reward-step-cost_held-from-first-arrival_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-large_reward-step-cost_divisor-held-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`

## Arms at the same configuration, on the same seeds

Each group compares one arm with other sides at the configurations both trained, matched on the intrinsic weight, learning rate, entropy coefficient and clip coefficient, and paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Define $W$, $L$ and $M$ as in the section Measures. No configuration is chosen here.

### Divisor held from each policy's first arrival, first validation block

#### PointMaze Medium, reward 1 on arrival

The arm: divisor held from each policy's first arrival, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9334 ± 0.0029 | 0.9374 ± 0.0027 | -0.0040 [-0.0109, +0.0029] | +0.0001 [-0.0000, +0.0002] | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9435 ± 0.0027 | 0.9382 ± 0.0027 | +0.0053 [-0.0010, +0.0114] | +0.0005 [+0.0001, +0.0011] | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.9334 ± 0.0029 | 0.9349 ± 0.0031 | -0.0015 [-0.0092, +0.0066] | +0.0050 [+0.0034, +0.0071] | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.01, entropy coefficient 0.01 | 0.9412 ± 0.0024 | 0.9275 ± 0.0027 | +0.0137 [+0.0074, +0.0202] | +0.0108 [+0.0081, +0.0140] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.9435 ± 0.0027 | 0.9194 ± 0.0023 | +0.0240 [+0.0180, +0.0300] | +0.0205 [+0.0163, +0.0252] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 29.4 ± 1.85 | 28.6 ± 1.69 | +0.798 [-3.41, +5.23] | 256 of 256; 256 of 256 | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 18.5 ± 1.13 | 21 ± 1.22 | -2.48 [-5.36, +0.433] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 29.4 ± 1.85 | 25 ± 1.46 | +4.4 [+0.32, +8.43] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.01, entropy coefficient 0.01 | 23.1 ± 1.34 | 23.5 ± 1.28 | -0.424 [-3.16, +2.37] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 18.5 ± 1.13 | 19.9 ± 1.16 | -1.4 [-4.26, +1.44] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held after one sixteenth of the run: $W$ ahead at 0, level at 2, behind at 0 of 2; $M$ sooner at 0, level at 2, later at 0 of 2; not trained by it: intrinsic weight 2, learning rate 0.01, entropy coefficient 0.01.
- against CleanRL PPO-RND: $W$ ahead at 2, level at 1, behind at 0 of 3; $M$ sooner at 0, level at 2, later at 1 of 3.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- divisor held from each policy's first arrival: best $W$ 0.9435 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 18.5 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 3 configurations).
- divisor held after one sixteenth of the run: best $W$ 0.9382 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 21 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ 0.9349 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 17.5 thousand steps at intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (of 6 configurations).

#### PointMaze Large, reward 1 on arrival

The arm: divisor held from each policy's first arrival, on 256 seeds from 10000 to 10255, read at 8,388,608 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8874 ± 0.0023 | 0.8876 ± 0.0021 | -0.0001 [-0.0061, +0.0059] | +0.0001 [-0.0006, +0.0008] | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.8852 ± 0.0023 | 0.8821 ± 0.0020 | +0.0031 [-0.0028, +0.0088] | +0.0006 [-0.0001, +0.0013] | 256 |
| CleanRL PPO-RND | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 0.8680 ± 0.0033 | 0.8759 ± 0.0022 | -0.0080 [-0.0154, -0.0007] | +0.0024 [+0.0013, +0.0034] | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.8874 ± 0.0023 | 0.8781 ± 0.0021 | +0.0094 [+0.0030, +0.0156] | +0.0057 [+0.0046, +0.0068] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.8852 ± 0.0023 | 0.8626 ± 0.0025 | +0.0226 [+0.0159, +0.0293] | +0.0211 [+0.0175, +0.0251] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 259 ± 10.6 | 270 ± 10.5 | -10.5 [-39.3, +18.9] | 256 of 256; 256 of 256 | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 201 ± 9.11 | 213 ± 8.15 | -12.2 [-34, +9.22] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 375 ± 12.4 | 385 ± 13.1 | -10 [-42.3, +23.2] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 259 ± 10.6 | 269 ± 10.8 | -9.92 [-38, +19] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 201 ± 9.11 | 205 ± 8.2 | -3.86 [-27.8, +19.3] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held after one sixteenth of the run: $W$ ahead at 0, level at 2, behind at 0 of 2; $M$ sooner at 0, level at 2, later at 0 of 2; not trained by it: intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01.
- against CleanRL PPO-RND: $W$ ahead at 2, level at 0, behind at 1 of 3; $M$ sooner at 0, level at 3, later at 0 of 3.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- divisor held from each policy's first arrival: best $W$ 0.8874 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 201 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 3 configurations).
- divisor held after one sixteenth of the run: best $W$ 0.8876 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 213 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ 0.8781 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 119 thousand steps at intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (of 8 configurations).

#### PointMaze Medium, step cost

The arm: divisor held from each policy's first arrival, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | -0.3484 ± 0.0014 | -0.3463 ± 0.0012 | -0.0020 [-0.0056, +0.0015] | +0.0008 [-0.0005, +0.0021] | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | -0.3504 ± 0.0015 | -0.3520 ± 0.0014 | +0.0016 [-0.0023, +0.0056] | +0.0004 [-0.0010, +0.0019] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 14.2 ± 0.617 | 13.9 ± 0.528 | +0.382 [-0.911, +1.68] | 256 of 256; 256 of 256 | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 13.4 ± 0.562 | 14.6 ± 0.653 | -1.2 [-2.71, +0.3] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held after one sixteenth of the run: $W$ ahead at 0, level at 2, behind at 0 of 2; $M$ sooner at 0, level at 2, later at 0 of 2; not trained by it: intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 0 of 0; $M$ sooner at 0, level at 0, later at 0 of 0; not trained by it: intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01; intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- divisor held from each policy's first arrival: best $W$ -0.3476 at intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01; best $M$ 13.4 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 3 configurations).
- divisor held after one sixteenth of the run: best $W$ -0.3463 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 13.9 thousand steps at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 (of 11 configurations).
- CleanRL PPO-RND: best $W$ -0.3505 at intrinsic weight 0.00195312, learning rate 0.006, entropy coefficient 0.01; best $M$ 12.4 thousand steps at intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (of 6 configurations).

#### PointMaze Large, step cost

The arm: divisor held from each policy's first arrival, on 256 seeds from 10000 to 10255, read at 33,554,432 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | -0.7335 ± 0.0034 | -0.7255 ± 0.0023 | -0.0080 [-0.0165, +0.0001] | -0.0101 [-0.0183, -0.0018] | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | -0.7268 ± 0.0025 | -0.7215 ± 0.0008 | -0.0053 [-0.0110, -0.0003] | -0.0091 [-0.0146, -0.0041] | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01 | -0.7236 ± 0.0021 | -0.7270 ± 0.0003 | +0.0034 [-0.0010, +0.0071] | -0.0024 [-0.0069, +0.0014] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | -0.7335 ± 0.0034 | -0.7310 ± 0.0018 | -0.0025 [-0.0102, +0.0047] | -0.0035 [-0.0110, +0.0038] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 206 ± 11.5 | 204 ± 10.3 | +1.64 [-25.2, +28.9] | 256 of 256; 256 of 256 | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01 | 189 ± 9.38 | 182 ± 9.19 | +6.34 [-18.1, +30.3] | 256 of 256; 256 of 256 | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01 | 163 ± 8.72 | 147 ± 6.83 | +15.7 [-4.8, +36.3] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 206 ± 11.5 | 191 ± 10.1 | +14.8 [-14.1, +44.1] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held after one sixteenth of the run: $W$ ahead at 0, level at 2, behind at 1 of 3; $M$ sooner at 0, level at 3, later at 0 of 3.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 1, behind at 0 of 1; $M$ sooner at 0, level at 1, later at 0 of 1; not trained by it: intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01; intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- divisor held from each policy's first arrival: best $W$ -0.7236 at intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01; best $M$ 163 thousand steps at intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01 (of 3 configurations).
- divisor held after one sixteenth of the run: best $W$ -0.7215 at intrinsic weight 8, learning rate 0.002, entropy coefficient 0.01; best $M$ 147 thousand steps at intrinsic weight 8, learning rate 0.003, entropy coefficient 0.01 (of 3 configurations).
- CleanRL PPO-RND: best $W$ -0.7310 at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01; best $M$ 120 thousand steps at intrinsic weight 8, learning rate 0.006, entropy coefficient 0.01 (of 4 configurations).

### Error length with a held divisor, first validation block

#### PointMaze Medium, reward 1 on arrival

The arm: error length as the bonus, divisor held after one sixteenth of the run, on 256 seeds from 10000 to 10255, read at 1,048,576 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.7321 ± 0.0034 | 0.9374 ± 0.0027 | -0.2053 [-0.2142, -0.1964] | -0.0802 [-0.0845, -0.0758] | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.5416 ± 0.0030 | 0.9382 ± 0.0027 | -0.3966 [-0.4043, -0.3889] | -0.2347 [-0.2448, -0.2253] | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 0.7321 ± 0.0034 | 0.9349 ± 0.0031 | -0.2028 [-0.2117, -0.1939] | -0.0753 [-0.0797, -0.0706] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 0.5416 ± 0.0030 | 0.9194 ± 0.0023 | -0.3779 [-0.3843, -0.3714] | -0.2147 [-0.2257, -0.2038] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 19.8 ± 0.945 | 28.6 ± 1.69 | -8.85 [-12.6, -5.23] | 256 of 256; 256 of 256 | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 18.2 ± 0.836 | 21 ± 1.22 | -2.8 [-5.53, -0.0653] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01 | 19.8 ± 0.945 | 25 ± 1.46 | -5.26 [-8.48, -2.07] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 | 18.2 ± 0.836 | 19.9 ± 1.16 | -1.72 [-4.49, +0.973] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held after one sixteenth of the run: $W$ ahead at 0, level at 0, behind at 2 of 2; $M$ sooner at 2, level at 0, later at 0 of 2; not trained by it: intrinsic weight 0.125, learning rate 0.006, entropy coefficient 0.01; intrinsic weight 0.25, learning rate 0.006, entropy coefficient 0.01; intrinsic weight 0.5, learning rate 0.006, entropy coefficient 0.01; intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 2 of 2; $M$ sooner at 1, level at 1, later at 0 of 2; not trained by it: intrinsic weight 0.125, learning rate 0.006, entropy coefficient 0.01; intrinsic weight 0.25, learning rate 0.006, entropy coefficient 0.01; intrinsic weight 0.5, learning rate 0.006, entropy coefficient 0.01; intrinsic weight 1, learning rate 0.006, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- error length as the bonus, divisor held after one sixteenth of the run: best $W$ 0.9256 at intrinsic weight 0.5, learning rate 0.006, entropy coefficient 0.01; best $M$ 18.2 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 6 configurations).
- divisor held after one sixteenth of the run: best $W$ 0.9382 at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01; best $M$ 21 thousand steps at intrinsic weight 4, learning rate 0.006, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ 0.9349 at intrinsic weight 2, learning rate 0.006, entropy coefficient 0.01; best $M$ 17.5 thousand steps at intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (of 6 configurations).

#### PointMaze Large, reward 1 on arrival

The arm: error length as the bonus, divisor held after one sixteenth of the run, on 256 seeds from 10000 to 10255, read at 8,388,608 environment steps per policy. Each row is one configuration of the arm and the same configuration of the side it is compared with; differences are the arm minus that side.

Whole-curve mean $W$ and final level $L$:

| compared with | configuration | $W$ of the arm | $W$ of the other side | difference in $W$ [95% interval] | difference in $L$ [95% interval] | seeds in common |
|---|---|---|---|---|---|---|
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.3521 ± 0.0030 | 0.8876 ± 0.0021 | -0.5354 [-0.5426, -0.5281] | -0.3607 [-0.3704, -0.3512] | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.1189 ± 0.0018 | 0.8821 ± 0.0020 | -0.7632 [-0.7681, -0.7579] | -0.7535 [-0.7621, -0.7446] | 256 |
| CleanRL PPO-RND | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 0.6557 ± 0.0026 | 0.8759 ± 0.0022 | -0.2202 [-0.2270, -0.2131] | -0.0793 [-0.0832, -0.0755] | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 0.3521 ± 0.0030 | 0.8781 ± 0.0021 | -0.5259 [-0.5334, -0.5184] | -0.3551 [-0.3648, -0.3456] | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 0.1189 ± 0.0018 | 0.8626 ± 0.0025 | -0.7437 [-0.7495, -0.7378] | -0.7330 [-0.7416, -0.7242] | 256 |

Mean capped first-goal time $M$:

| compared with | configuration | $M$ of the arm, thousand steps | $M$ of the other side, thousand steps | difference in $M$ [95% interval], thousand steps | policies that reach the goal: the arm; the other side | seeds in common |
|---|---|---|---|---|---|---|
| divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 103 ± 4.22 | 270 ± 10.5 | -167 [-189, -144] | 256 of 256; 256 of 256 | 256 |
| divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 93 ± 4.07 | 213 ± 8.15 | -120 [-138, -102] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01 | 125 ± 5 | 385 ± 13.1 | -260 [-287, -234] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01 | 103 ± 4.22 | 269 ± 10.8 | -166 [-187, -145] | 256 of 256; 256 of 256 | 256 |
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 | 93 ± 4.07 | 205 ± 8.2 | -112 [-130, -93.6] | 256 of 256; 256 of 256 | 256 |

Rows on each side of zero (interval entirely above zero: ahead in $W$; entirely below zero: sooner in $M$):

- against divisor held after one sixteenth of the run: $W$ ahead at 0, level at 0, behind at 2 of 2; $M$ sooner at 2, level at 0, later at 0 of 2; not trained by it: intrinsic weight 0.125, learning rate 0.002, entropy coefficient 0.01; intrinsic weight 0.25, learning rate 0.002, entropy coefficient 0.01; intrinsic weight 0.5, learning rate 0.002, entropy coefficient 0.01; intrinsic weight 1, learning rate 0.002, entropy coefficient 0.01.
- against CleanRL PPO-RND: $W$ ahead at 0, level at 0, behind at 3 of 3; $M$ sooner at 3, level at 0, later at 0 of 3; not trained by it: intrinsic weight 0.125, learning rate 0.002, entropy coefficient 0.01; intrinsic weight 0.25, learning rate 0.002, entropy coefficient 0.01; intrinsic weight 0.5, learning rate 0.002, entropy coefficient 0.01.

Each side at its best among all its configurations read in this case (for $W$ the highest, for $M$ the smallest):

- error length as the bonus, divisor held after one sixteenth of the run: best $W$ 0.8714 at intrinsic weight 0.25, learning rate 0.002, entropy coefficient 0.01; best $M$ 93 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 6 configurations).
- divisor held after one sixteenth of the run: best $W$ 0.8876 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 213 thousand steps at intrinsic weight 4, learning rate 0.002, entropy coefficient 0.01 (of 2 configurations).
- CleanRL PPO-RND: best $W$ 0.8781 at intrinsic weight 2, learning rate 0.002, entropy coefficient 0.01; best $M$ 119 thousand steps at intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (of 8 configurations).

## Predictions

The predictions were written in `ideas.md` before the runs. Each is quoted, then read part by part: a part held or did not hold on the numbers given, or cannot be read yet because a unit it needs is not complete. A prediction held when every part held, and did not hold when at least one part did not. Where a prediction leaves a choice, it is read as follows. Both are read on the first validation block (seeds 10000 to 10255), as the batch's ideas file says, with every side at its own best configuration there: for the whole-curve mean $W$ the highest $W$, for the mean capped first-goal time $M$ the smallest $M$. "Level with the held divisor alone (within 0.005)" is read as: the mean paired difference in $W$ is at most 0.005 in size. "Ahead" and "sooner" are read as: the 95% interval of the paired difference is entirely above zero ($W$) or entirely below zero ($M$). "Level" in prediction 2 is read as: the interval of the difference in $W$ holds zero.

### Prediction 1: the divisor held from each policy's first arrival

> **Divisor held from the first arrival:** on PointMaze Large with reward 1 on arrival it is level with the held divisor alone (within 0.005 on the whole-curve mean) and ahead of RND with its interval above zero. On PointMaze Medium with reward 1 on arrival it is ahead of the held divisor after 1/16, because its hold comes at about 15 thousand steps where the fixed one comes at 66 thousand.

Observed: **did not hold**.

- PointMaze Large with reward 1 on arrival: level with the held divisor alone (within 0.005): **held**. $W$ 0.8874 against 0.8876; difference -0.0001 [-0.0061, +0.0060].
- PointMaze Large with reward 1 on arrival: ahead of RND with its interval above zero: **held**. $W$ 0.8874; difference from RND's best +0.0094 [+0.0032, +0.0156].
- PointMaze Medium with reward 1 on arrival: ahead of the held divisor after one sixteenth: **did not hold**. $W$ 0.9435 against 0.9382; difference +0.0053 [-0.0010, +0.0114].

### Prediction 2: the error length with a held divisor

> **Error length with held divisor:** it reaches the goal sooner than the held divisor alone in both cases (first-goal time lower with its interval below zero) and is level with it on the whole-curve mean.

Observed: **did not hold**.

- PointMaze Large with reward 1 on arrival: reaches the goal sooner than the held divisor alone: **held**. $M$ 93 thousand steps against 213 thousand steps; difference -120 [-138, -102] thousand steps.
- PointMaze Medium with reward 1 on arrival: reaches the goal sooner than the held divisor alone: **held**. $M$ 18.2 thousand steps against 21 thousand steps; difference -2.8 [-5.57, -0.059] thousand steps.
- PointMaze Large with reward 1 on arrival: level with the held divisor alone on $W$: **did not hold**. $W$ 0.8714 against 0.8876; difference -0.0161 [-0.0211, -0.0112].
- PointMaze Medium with reward 1 on arrival: level with the held divisor alone on $W$: **did not hold**. $W$ 0.9256 against 0.9382; difference -0.0126 [-0.0200, -0.0051].

