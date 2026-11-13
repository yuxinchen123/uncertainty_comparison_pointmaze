# Batch 6: the predictor at half speed with held divisor, a second validation block, and the PointMaze Large cases

Generated 2026-11-13 00:23 PT by the command below. This file is generated: regenerate it, do not edit it. Times are Pacific, converted from machines running Eastern. Run folder: `2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days`; batch folder: `batches/06_predictor-at-half-speed-with-held-divisor_second-validation-block_and-the-large-cases`.

```bash
cd /p/rlprojects/RND/10_jax_exploration_platform
bash "benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh" -s scripts/maze_campaign/analysis/batch_summary_job.py \
    --run-dir=runs/2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days \
    --batch=06_predictor-at-half-speed-with-held-divisor_second-validation-block_and-the-large-cases
```

## Contents

| section | first written | last changed | status |
|---|---|---|---|
| [Measures](#measures) | 2026-11-13 00:06 PT | 2026-11-13 00:06 PT | unread |
| [Validation seeds, case by case](#validation-seeds-case-by-case) | 2026-11-13 00:06 PT | 2026-11-13 00:23 PT | unread |
| [Predictions](#predictions) | 2026-11-13 00:06 PT | 2026-11-13 00:15 PT | unread |

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

The units of the phase `validation-block` in PointMaze Medium, reward 1 on arrival. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006 | 0.9349 ± 0.0031 | 0.9950 ± 0.0010 | 25 ± 1.46 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.01 | 0.9275 ± 0.0027 | 0.9892 ± 0.0015 | 23.5 ± 1.28 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.006 | 0.9194 ± 0.0023 | 0.9794 ± 0.0023 | 19.9 ± 1.16 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.006 | **0.9438** ± 0.0030 | **1.0000** ± 0.0000 | 22.6 ± 1.75 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 2, learning rate 0.006 | <u>0.9399</u> ± 0.0028 | **1.0000** ± 0.0000 | **17.4** ± 1.07 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 | 0.9382 ± 0.0027 | 0.9995 ± 0.0003 | 21 ± 1.22 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 | 0.9374 ± 0.0027 | <u>0.9999</u> ± 0.0000 | 28.6 ± 1.69 | 256 of 256 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.01 | 0.9388 ± 0.0026 | <u>0.9999</u> ± 0.0001 | 22.4 ± 1.27 | 256 of 256 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.006 | 0.9340 ± 0.0035 | <u>0.9999</u> ± 0.0001 | 30.6 ± 2.09 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 2, learning rate 0.003 | 0.9348 ± 0.0030 | <u>0.9999</u> ± 0.0000 | <u>17.9</u> ± 1 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 0.5, learning rate 0.006 | 0.9284 ± 0.0040 | **1.0000** ± 0.0000 | 33.9 ± 2.64 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 0.25, learning rate 0.01 | 0.9195 ± 0.0042 | **1.0000** ± 0.0000 | 36.2 ± 2.53 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.01 | 0.9061 ± 0.0051 | <u>0.9999</u> ± 0.0000 | 25.4 ± 1.49 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 0.25, learning rate 0.01 | 0.9061 ± 0.0054 | <u>0.9999</u> ± 0.0000 | 36.3 ± 2.46 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.003 | 0.8928 ± 0.0059 | 0.9997 ± 0.0001 | 24 ± 1.78 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006 (best of 3) | 0.9349 ± 0.0031 | control | 0.9950 ± 0.0010 | control | 256 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.006 (best of 2) | **0.9438** ± 0.0030 | **+0.0089** [+0.0013, +0.0164] | **1.0000** ± 0.0000 | **+0.0050** [+0.0034, +0.0071] | 256 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.01 (best of 2) | <u>0.9388</u> ± 0.0026 | <u>+0.0039</u> [-0.0034, +0.0113] | <u>0.9999</u> ± 0.0001 | <u>+0.0049</u> [+0.0032, +0.0070] | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 (best of 2) | 0.9382 ± 0.0027 | +0.0033 [-0.0042, +0.0109] | 0.9995 ± 0.0003 | +0.0045 [+0.0027, +0.0066] | 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 2, learning rate 0.003 (best of 3) | 0.9348 ± 0.0030 | -0.0001 [-0.0084, +0.0083] | <u>0.9999</u> ± 0.0000 | <u>+0.0049</u> [+0.0033, +0.0070] | 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.01 (best of 3) | 0.9061 ± 0.0051 | -0.0288 [-0.0401, -0.0180] | <u>0.9999</u> ± 0.0000 | <u>+0.0049</u> [+0.0033, +0.0070] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.006 (best of 3) | 19.9 ± 1.16 | control | 256 of 256 | 0.9194 | 0.9794 | 256 |
| predictor at half speed with held divisor | intrinsic weight 2, learning rate 0.006 (best of 2) | **17.4** ± 1.07 | **-2.54** [-5.53, +0.434] | 256 of 256 | **0.9399** | **1.0000** | 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 2, learning rate 0.003 (best of 3) | <u>17.9</u> ± 1 | <u>-1.98</u> [-5.03, +0.955] | 256 of 256 | 0.9348 | <u>0.9999</u> | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 (best of 2) | 21 ± 1.22 | +1.08 [-1.61, +3.82] | 256 of 256 | 0.9382 | 0.9995 | 256 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.01 (best of 2) | 22.4 ± 1.27 | +2.46 [-0.627, +5.49] | 256 of 256 | <u>0.9388</u> | <u>0.9999</u> | 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.003 (best of 3) | 24 ± 1.78 | +4.09 [+0.448, +7.95] | 256 of 256 | 0.8928 | 0.9997 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:53 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 4 | predictor at half speed with held divisor | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:52 PT | none |
| 5 | predictor at half speed with held divisor | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:51 PT | none |
| 6 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:44 PT | none |
| 7 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:45 PT | none |
| 8 | held divisor alone, held after one sixty-fourth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:46 PT | none |
| 9 | held divisor alone, held after one sixty-fourth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:48 PT | none |
| 10 | slower predictor with held divisor (divisor held after one sixteenth of the run) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:56 PT | none |
| 11 | slower predictor with held divisor (divisor held after one sixteenth of the run) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 12 | slower predictor with held divisor (divisor held after one sixteenth of the run) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:56 PT | none |
| 13 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 20:03 PT | none |
| 14 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 20:02 PT | none |
| 15 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 20:02 PT | none |

Unit names, by the number in the first column:

1. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-medium_reward-arrival-plus-one_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-medium_reward-arrival-plus-one_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-one-sixty-fourth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-one-sixty-fourth_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-medium_reward-arrival-plus-one_slower-predictor-with-held-divisor_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-medium_reward-arrival-plus-one_slower-predictor-with-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-medium_reward-arrival-plus-one_slower-predictor-with-held-divisor_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.25_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-medium_reward-arrival-plus-one_three-parts-held-one-sixteenth_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
14. `validation-block_pointmaze-medium_reward-arrival-plus-one_three-parts-held-one-sixteenth_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.25_clip-coefficient-0.1_seeds-10000-to-10255`
15. `validation-block_pointmaze-medium_reward-arrival-plus-one_three-parts-held-one-sixteenth_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`

#### Second block: seeds that chose nothing

The units of the phase `validation-block-two` in PointMaze Medium, reward 1 on arrival. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006 | 0.9308 ± 0.0030 | 0.9907 ± 0.0039 | 26.3 ± 1.88 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.01 | 0.9276 ± 0.0026 | 0.9798 ± 0.0059 | 23.6 ± 1.4 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.006 | 0.9141 ± 0.0030 | 0.9821 ± 0.0018 | <u>19.6</u> ± 1.14 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 2, learning rate 0.006 | <u>0.9416</u> ± 0.0027 | **1.0000** ± 0.0000 | **17.2** ± 0.962 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.006 | 0.9410 ± 0.0035 | <u>0.9999</u> ± 0.0001 | 22.8 ± 1.68 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 | 0.9398 ± 0.0025 | 0.9995 ± 0.0002 | 19.7 ± 1.13 | 256 of 256 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.01 | **0.9417** ± 0.0026 | **1.0000** ± 0.0000 | 22.9 ± 1.46 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006 (its best on the first block and on this block) | 0.9308 ± 0.0030 | control | control | 0.9907 ± 0.0039 | control | 256 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.01 (its best on the first block) | **0.9417** ± 0.0026 | **+0.0109** [+0.0037, +0.0183] | **+0.0109** [+0.0037, +0.0183] | **1.0000** ± 0.0000 | **+0.0093** [+0.0043, +0.0180] | 256 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.006 (its best on the first block) | <u>0.9410</u> ± 0.0035 | <u>+0.0102</u> [+0.0012, +0.0186] | <u>+0.0102</u> [+0.0012, +0.0186] | <u>0.9999</u> ± 0.0001 | <u>+0.0092</u> [+0.0042, +0.0178] | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 (its best on the first block) | 0.9398 ± 0.0025 | +0.0090 [+0.0023, +0.0159] | +0.0090 [+0.0023, +0.0159] | 0.9995 ± 0.0002 | +0.0088 [+0.0038, +0.0173] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.006 (its best on the first block and on this block) | <u>19.6</u> ± 1.14 | control | control | 256 of 256 | 256 |
| predictor at half speed with held divisor | intrinsic weight 2, learning rate 0.006 (its best on the first block) | **17.2** ± 0.962 | **-2.45** [-5.23, +0.244] | **-2.45** [-5.23, +0.244] | 256 of 256 | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 (its best on the first block) | 19.7 ± 1.13 | <u>+0.0469</u> [-2.73, +2.71] | <u>+0.0469</u> [-2.73, +2.71] | 256 of 256 | 256 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.01 (its best on the first block) | 22.9 ± 1.46 | +3.28 [+0.108, +6.56] | +3.28 [+0.108, +6.56] | 256 of 256 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:16 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:19 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:16 PT | none |
| 4 | predictor at half speed with held divisor | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:19 PT | none |
| 5 | predictor at half speed with held divisor | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:21 PT | none |
| 6 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:22 PT | none |
| 7 | held divisor alone, held after one sixty-fourth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:23 PT | none |

Unit names, by the number in the first column:

1. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_half-speed-predictor-held_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-20000-to-20255`
5. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_half-speed-predictor-held_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
6. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_divisor-held-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
7. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_divisor-held-one-sixty-fourth_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`

### PointMaze Large, reward 1 on arrival

#### First block: each side at its best configuration there

The units of the phase `validation-block` in PointMaze Large, reward 1 on arrival. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 8,388,608 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 | 0.8781 ± 0.0021 | 0.9920 ± 0.0005 | 269 ± 10.8 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 1, learning rate 0.002 | 0.8759 ± 0.0022 | 0.9955 ± 0.0004 | 385 ± 13.1 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.003 | 0.8701 ± 0.0023 | 0.9906 ± 0.0006 | 203 ± 8.06 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.001 | 0.8695 ± 0.0029 | 0.9911 ± 0.0007 | 344 ± 17.8 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 | 0.8626 ± 0.0025 | 0.9765 ± 0.0019 | 205 ± 8.2 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 4, learning rate 0.002 | 0.8849 ± 0.0023 | 0.9978 ± 0.0003 | 161 ± 10.6 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 2, learning rate 0.002 | 0.8821 ± 0.0027 | <u>0.9981</u> ± 0.0002 | 217 ± 13.5 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.002 | 0.8597 ± 0.0042 | 0.9978 ± 0.0004 | 360 ± 22.8 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002 | 0.8876 ± 0.0021 | 0.9976 ± 0.0003 | 270 ± 10.5 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002 | 0.8821 ± 0.0020 | 0.9970 ± 0.0002 | 213 ± 8.15 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 4, learning rate 0.002 | <u>0.8883</u> ± 0.0021 | **0.9982** ± 0.0002 | <u>153</u> ± 9.69 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 2, learning rate 0.002 | 0.8851 ± 0.0029 | <u>0.9981</u> ± 0.0003 | 182 ± 13.9 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.002 | **0.8917** ± 0.0026 | 0.9969 ± 0.0004 | 160 ± 11.2 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.002 | 0.8779 ± 0.0021 | 0.9948 ± 0.0004 | **124** ± 9.81 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 0.25, learning rate 0.002 | 0.8778 ± 0.0052 | 0.9974 ± 0.0004 | 284 ± 20.3 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 (best of 5) | 0.8781 ± 0.0021 | control | 0.9920 ± 0.0005 | control | 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.002 (best of 3) | **0.8917** ± 0.0026 | **+0.0137** [+0.0071, +0.0202] | 0.9969 ± 0.0004 | +0.0049 [+0.0036, +0.0063] | 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 4, learning rate 0.002 (best of 2) | <u>0.8883</u> ± 0.0021 | <u>+0.0102</u> [+0.0048, +0.0159] | **0.9982** ± 0.0002 | **+0.0062** [+0.0052, +0.0073] | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002 (best of 2) | 0.8876 ± 0.0021 | +0.0095 [+0.0040, +0.0151] | 0.9976 ± 0.0003 | +0.0056 [+0.0045, +0.0067] | 256 |
| predictor at half speed with held divisor | intrinsic weight 4, learning rate 0.002 (best of 3) | 0.8849 ± 0.0023 | +0.0068 [+0.0007, +0.0129] | <u>0.9978</u> ± 0.0003 | <u>+0.0058</u> [+0.0047, +0.0070] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.003 (best of 5) | 203 ± 8.06 | control | 256 of 256 | 0.8701 | 0.9906 | 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.002 (best of 3) | **124** ± 9.81 | **-79** [-102, -54.7] | 256 of 256 | 0.8779 | 0.9948 | 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 4, learning rate 0.002 (best of 2) | <u>153</u> ± 9.69 | <u>-50.3</u> [-75.1, -25.4] | 256 of 256 | **0.8883** | **0.9982** | 256 |
| predictor at half speed with held divisor | intrinsic weight 4, learning rate 0.002 (best of 3) | 161 ± 10.6 | -42.3 [-66.9, -18.1] | 256 of 256 | <u>0.8849</u> | <u>0.9978</u> | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002 (best of 2) | 213 ± 8.15 | +10.2 [-11, +30.4] | 256 of 256 | 0.8821 | 0.9970 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 20:19 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 15:26 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 15:26 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 15:27 PT | none |
| 5 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 20:18 PT | none |
| 6 | predictor at half speed with held divisor | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:34 PT | none |
| 7 | predictor at half speed with held divisor | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:32 PT | none |
| 8 | predictor at half speed with held divisor | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:37 PT | none |
| 9 | held divisor alone, held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 16:41 PT | none |
| 10 | held divisor alone, held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 17:02 PT | none |
| 11 | slower predictor with held divisor (divisor held after one sixteenth of the run) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 16:29 PT | none |
| 12 | slower predictor with held divisor (divisor held after one sixteenth of the run) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 16:37 PT | none |
| 13 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 18:49 PT | none |
| 14 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:03 PT | none |
| 15 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:02 PT | none |

Unit names, by the number in the first column:

1. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.001_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-large_reward-arrival-plus-one_predictor-half-speed-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-large_reward-arrival-plus-one_predictor-half-speed-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-large_reward-arrival-plus-one_predictor-half-speed-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-after-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-after-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-large_reward-arrival-plus-one_slower-predictor-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-large_reward-arrival-plus-one_slower-predictor-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-large_reward-arrival-plus-one_three-parts-held-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-0.25_clip-coefficient-0.1_seeds-10000-to-10255`
14. `validation-block_pointmaze-large_reward-arrival-plus-one_three-parts-held-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`
15. `validation-block_pointmaze-large_reward-arrival-plus-one_three-parts-held-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`

### PointMaze Large, step cost

#### First block: each side at its best configuration there

The units of the phase `validation-block` in PointMaze Large, step cost. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 33,554,432 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 | -0.7310 ± 0.0018 | -0.7163 ± 0.0017 | 191 ± 10.1 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 | -0.7411 ± 0.0036 | -0.7229 ± 0.0033 | 241 ± 13.4 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 8, learning rate 0.003 | <u>-0.7216</u> ± 0.0014 | -0.7053 ± 0.0014 | 132 ± 9.08 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 4, learning rate 0.003 | -0.7370 ± 0.0036 | -0.7208 ± 0.0035 | 142 ± 9.75 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002 | **-0.7215** ± 0.0008 | **-0.7025** ± 0.0009 | 182 ± 9.19 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002 | -0.7255 ± 0.0023 | -0.7098 ± 0.0022 | 204 ± 10.3 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.003 | -0.7270 ± 0.0003 | -0.7055 ± 0.0002 | 147 ± 6.83 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 8, learning rate 0.003 | -0.7230 ± 0.0016 | -0.7058 ± 0.0015 | 127 ± 7.73 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 16, learning rate 0.003 | -0.7243 ± 0.0006 | <u>-0.7027</u> ± 0.0003 | <u>125</u> ± 7.29 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.002 | -0.7291 ± 0.0025 | -0.7116 ± 0.0025 | **122** ± 8.85 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.0045 | -0.7320 ± 0.0028 | -0.7142 ± 0.0028 | 139 ± 10 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 (best of 2) | -0.7310 ± 0.0018 | control | -0.7163 ± 0.0017 | control | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002 (best of 3) | **-0.7215** ± 0.0008 | **+0.0095** [+0.0057, +0.0137] | **-0.7025** ± 0.0009 | **+0.0139** [+0.0102, +0.0178] | 256 |
| predictor at half speed with held divisor | intrinsic weight 8, learning rate 0.003 (best of 2) | <u>-0.7216</u> ± 0.0014 | <u>+0.0094</u> [+0.0048, +0.0140] | <u>-0.7053</u> ± 0.0014 | <u>+0.0111</u> [+0.0067, +0.0155] | 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 8, learning rate 0.003 (best of 2) | -0.7230 ± 0.0016 | +0.0081 [+0.0032, +0.0129] | -0.7058 ± 0.0015 | +0.0105 [+0.0061, +0.0149] | 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.002 (best of 2) | -0.7291 ± 0.0025 | +0.0019 [-0.0041, +0.0077] | -0.7116 ± 0.0025 | +0.0047 [-0.0013, +0.0102] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 (best of 2) | 191 ± 10.1 | control | 256 of 256 | -0.7310 | -0.7163 | 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.002 (best of 2) | **122** ± 8.85 | **-69** [-95, -42.2] | 256 of 256 | -0.7291 | -0.7116 | 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 16, learning rate 0.003 (best of 2) | <u>125</u> ± 7.29 | <u>-66.3</u> [-90.8, -41.5] | 256 of 256 | <u>-0.7243</u> | **-0.7027** | 256 |
| predictor at half speed with held divisor | intrinsic weight 8, learning rate 0.003 (best of 2) | 132 ± 9.08 | -58.9 [-85.7, -32] | 256 of 256 | **-0.7216** | <u>-0.7053</u> | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.003 (best of 3) | 147 ± 6.83 | -43.8 [-68, -20.2] | 256 of 256 | -0.7270 | -0.7055 | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:25 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:03 PT | none |
| 3 | predictor at half speed with held divisor | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:51 PT | none |
| 4 | predictor at half speed with held divisor | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:27 PT | none |
| 5 | held divisor alone, held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:50 PT | none |
| 6 | held divisor alone, held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 00:19 PT | none |
| 7 | held divisor alone, held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:54 PT | none |
| 8 | slower predictor with held divisor (divisor held after one sixteenth of the run) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:24 PT | none |
| 9 | slower predictor with held divisor (divisor held after one sixteenth of the run) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:25 PT | none |
| 10 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 00:07 PT | none |
| 11 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:50 PT | none |

Unit names, by the number in the first column:

1. `validation-block_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `validation-block_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `validation-block_pointmaze-large_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-large_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-large_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-large_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-large_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-large_reward-step-cost_slower-predictor-with-held-divisor_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-large_reward-step-cost_slower-predictor-with-held-divisor_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-large_reward-step-cost_three-parts-held-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-large_reward-step-cost_three-parts-held-one-sixteenth_learning-rate-0.0045_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`

## Predictions

The predictions were written in `ideas.md` before the runs. Each is quoted, then read part by part: a part held or did not hold on the numbers given, or cannot be read yet because a unit it needs is not complete. A prediction held when every part held, and did not hold when at least one part did not. Where a prediction leaves a choice, it is read as follows. On the first block every arm stands at its best configuration for the whole-curve mean $W$ among those run there, and the control at its own; a part waits until every configuration planned there for its arms and for the control is read. On the second block the arm stands at the configuration that was its best on the first block and is compared under equal rules, that is with the control at the configuration that was its best on the first block. "Above RND" is read as a mean difference in $W$ above zero. "Ahead with its interval above zero" is read as: the 95% interval of the difference is entirely above zero. "Level" is read as: that interval holds zero. Prediction 1 expects the interval on the second block to reach zero; that part holds when it does.

### Prediction 1: second block, PointMaze Medium with reward 1 on arrival

> **Second block, PointMaze Medium:** the predictor at half speed with held divisor is again above RND, by less than on the first block (between 0 and +0.009); its interval is above zero only if the true difference is at least about 0.008, which I do not expect.

Observed: **did not hold**.

- The predictor at half speed with held divisor is again above the control: **held**. $W$ 0.9410 on the second block, difference under equal rules +0.0102 [+0.0012, +0.0186].
- By less than on the first block: **did not hold**. First block +0.0089; second block +0.0102.
- Its interval reaches zero, as the prediction expected: **did not hold**. Difference under equal rules +0.0102 [+0.0012, +0.0186].

### Prediction 2: PointMaze Large, reward 1 on arrival

> **PointMaze Large, reward 1 on arrival:** the predictor at half speed with held divisor is within 0.005 of the slower predictor with held divisor (0.8883 at weight 4) at its best weight, and ahead of RND (0.8781) with its interval above zero.

Observed: **held**.

- The predictor at half speed with held divisor is within 0.005 of the slower predictor with held divisor: **held**. Best $W$ on the first block: the arm 0.8849, the slower predictor with held divisor 0.8883 (apart by 0.0034).
- It is ahead of the control with its interval above zero: **held**. $W$ 0.8849, difference from the control +0.0068 [+0.0007, +0.0129].

### Prediction 3: PointMaze Large, step cost

> **PointMaze Large, step cost:** all three parts stays ahead of RND with its interval above zero on 256 seeds; the slower predictor with held divisor is level.

Observed: **did not hold**.

- All three parts stays ahead of the control with its interval above zero: **did not hold**. $W$ -0.7291, difference from the control +0.0019 [-0.0041, +0.0077].
- The slower predictor with held divisor is level with the control: **did not hold**. $W$ -0.7230, difference from the control +0.0081 [+0.0032, +0.0129].

