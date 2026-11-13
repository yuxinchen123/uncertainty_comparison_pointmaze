# Batch 5: a bonus that falls away sooner after the first arrivals (PointMaze Medium, reward 1 on arrival)

Generated 2026-11-13 00:15 PT by the command below. This file is generated: regenerate it, do not edit it. Times are Pacific, converted from machines running Eastern. Run folder: `2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days`; batch folder: `batches/05_bonus-that-falls-away-sooner-after-the-first-arrivals_pointmaze-medium`.

```bash
cd /p/rlprojects/RND/10_jax_exploration_platform
bash "benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh" -s scripts/maze_campaign/analysis/batch_summary_job.py \
    --run-dir=runs/2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days \
    --batch=05_bonus-that-falls-away-sooner-after-the-first-arrivals_pointmaze-medium
```

## Contents

| section | first written | last changed | status |
|---|---|---|---|
| [Measures](#measures) | 2026-11-13 00:05 PT | 2026-11-13 00:05 PT | unread |
| [Development sweeps against the baseline, case by case](#development-sweeps-against-the-baseline-case-by-case) | 2026-11-13 00:05 PT | 2026-11-13 00:05 PT | unread |
| [Validation seeds, case by case](#validation-seeds-case-by-case) | 2026-11-13 00:05 PT | 2026-11-13 00:15 PT | unread |
| [Predictions](#predictions) | 2026-11-13 00:05 PT | 2026-11-13 00:15 PT | unread |

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

## Development sweeps against the baseline, case by case

A case is one environment with one reward setting. In each case every method of this batch was trained on a grid of configurations on the development seeds, and so was the baseline, CleanRL PPO-RND (run folder `2026-11-12-02-24_cleanrl-ppo-rnd-baseline_pointmaze-medium-large_antmaze-umaze-medium-large_goal-termination_reward-arrival-plus-1-and-step-cost_first-pass-60-configurations_seeds-0-15_confirmation-seeds-from-30000`, phases `development-at-the-budget*`, the units that train the same number of steps). Define $W$, $L$, $M$, the best configuration and the difference from the baseline as in the section Measures. Per case, two tables, one per metric: the baseline at its own best configuration in the first row, then each method at its own, from the best value on. A best configuration is named by the settings that vary in the grids, with the number of configurations it was the best of. Per column the best value is bold and the second best underlined. Every best configuration was chosen on the seeds the table is read on: these tables are a screen and not a verdict. Only complete units are read; nothing is estimated for the others.

### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 9 | complete | 256 of 256 | 2026-11-12 21:14 PT | none |
| 2 | held divisor alone, held after one sixty-fourth of the run | PointMaze Medium, reward 1 on arrival | 9 | complete | 256 of 256 | 2026-11-12 21:18 PT | none |
| 3 | slower predictor with divisor held after one sixty-fourth of the run | PointMaze Medium, reward 1 on arrival | 9 | complete | 256 of 256 | 2026-11-12 21:21 PT | none |
| 4 | predictor at half speed with held divisor | PointMaze Medium, reward 1 on arrival | 9 | complete | 256 of 256 | 2026-11-12 21:24 PT | none |

Unit names, by the number in the first column:

1. `batch-05-development_pointmaze-medium_reward-arrival-plus-one_divisor-held-after-one-sixteenth_cells-1-to-9-of-9_seeds-0-to-15`
2. `batch-05-development_pointmaze-medium_reward-arrival-plus-one_divisor-held-after-one-sixty-fourth_cells-1-to-9-of-9_seeds-0-to-15`
3. `batch-05-development_pointmaze-medium_reward-arrival-plus-one_slower-predictor-divisor-held-after-one-sixty-fourth_cells-1-to-9-of-9_seeds-0-to-15`
4. `batch-05-development_pointmaze-medium_reward-arrival-plus-one_predictor-at-half-speed-with-held-divisor_cells-1-to-9-of-9_seeds-0-to-15`

### PointMaze Medium, reward 1 on arrival

Read at 1,048,576 environment steps per policy. CleanRL PPO-RND: 16 of 16 configurations read (2 of 2 units complete). held divisor alone, held after one sixteenth of the run: 9 of 9 configurations read (1 of 1 units complete). held divisor alone, held after one sixty-fourth of the run: 9 of 9 configurations read (1 of 1 units complete). slower predictor with divisor held after one sixty-fourth of the run: 9 of 9 configurations read (1 of 1 units complete). predictor at half speed with held divisor: 9 of 9 configurations read (1 of 1 units complete).

#### Discovery and optimization: whole-curve mean

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the baseline [95% interval] | $L$ ± standard error | difference in $L$ from the baseline [95% interval] | best configuration at an end of its grid | seeds in common with the baseline |
|---|---|---|---|---|---|---|---|
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.01 (best of 16) | 0.944 ± 0.006 | baseline | <u>0.993</u> ± 0.004 | baseline | no | 16 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.006 (best of 9) | **0.964** ± 0.005 | **+0.020** [+0.003, +0.037] | **1.000** ± 0.000 | +0.007 [+0.002, +0.016] | no | 16 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 (best of 9) | <u>0.962</u> ± 0.006 | <u>+0.018</u> [+0.002, +0.034] | **1.000** ± 0.000 | +0.007 [+0.002, +0.016] | no | 16 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.006 (best of 9) | 0.959 ± 0.007 | +0.016 [-0.004, +0.036] | **1.000** ± 0.000 | +0.007 [+0.002, +0.016] | intrinsic weight at the smallest of its grid | 16 |
| slower predictor with divisor held after one sixty-fourth of the run | intrinsic weight 1, learning rate 0.01 (best of 9) | 0.956 ± 0.010 | +0.012 [-0.010, +0.033] | **1.000** ± 0.000 | +0.007 [+0.002, +0.016] | intrinsic weight at the smallest of its grid; learning rate at the largest | 16 |

CleanRL PPO-RND's best configuration here has no complete unit on the confirmation seeds (phases `confirmation`): the confirmation units of this case hold intrinsic weight 2, learning rate 0.006.

#### Discovery: mean capped first-goal time

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the baseline [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | best configuration at an end of its grid | seeds in common with the baseline |
|---|---|---|---|---|---|---|---|---|
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.02 (best of 16) | **8.73** ± 1.31 | baseline | 16 of 16 | 0.655 | <u>0.812</u> | intrinsic weight at the largest of its grid | 16 |
| slower predictor with divisor held after one sixty-fourth of the run | intrinsic weight 4, learning rate 0.01 (best of 9) | <u>9.29</u> ± 1.25 | **+0.561** [-3.15, +4.13] | 16 of 16 | 0.904 | **1.000** | intrinsic weight at the largest of its grid; learning rate at the largest | 16 |
| predictor at half speed with held divisor | intrinsic weight 4, learning rate 0.01 (best of 9) | 10.5 ± 1.83 | <u>+1.78</u> [-1.97, +5.4] | 16 of 16 | 0.924 | **1.000** | intrinsic weight at the largest of its grid; learning rate at the largest | 16 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 4, learning rate 0.006 (best of 9) | 12.9 ± 1.84 | +4.18 [+0.333, +7.82] | 16 of 16 | **0.948** | **1.000** | intrinsic weight at the largest of its grid | 16 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 (best of 9) | 16.2 ± 3.2 | +7.49 [+1.77, +14.1] | 16 of 16 | <u>0.945</u> | **1.000** | intrinsic weight at the largest of its grid | 16 |

CleanRL PPO-RND's best configuration here has no complete unit on the confirmation seeds (phases `confirmation`): the confirmation units of this case hold intrinsic weight 2, learning rate 0.006.

Best configuration for $W$ at an end of its grid: slower predictor with divisor held after one sixty-fourth of the run (intrinsic weight at the smallest of its grid; learning rate at the largest); predictor at half speed with held divisor (intrinsic weight at the smallest of its grid). Best configuration for $M$ at an end of its grid: held divisor alone, held after one sixteenth of the run (intrinsic weight at the largest of its grid); held divisor alone, held after one sixty-fourth of the run (intrinsic weight at the largest of its grid); slower predictor with divisor held after one sixty-fourth of the run (intrinsic weight at the largest of its grid; learning rate at the largest); predictor at half speed with held divisor (intrinsic weight at the largest of its grid; learning rate at the largest); CleanRL PPO-RND (intrinsic weight at the largest of its grid). The edge rule of `program.md` asks for more values on that side before anything is confirmed.

#### Whole-curve mean on every configuration of the methods

Rows: a method with its settings other than the intrinsic weight; columns: the intrinsic weights. Over each method's whole grid the highest $W$ is bold and the second highest underlined.

| method and its other settings | weight 1 | weight 2 | weight 4 |
|---|---|---|---|
| held divisor alone, held after one sixteenth of the run, learning rate 0.003 | 0.934 | 0.932 | 0.926 |
| held divisor alone, held after one sixteenth of the run, learning rate 0.006 | 0.942 | **0.962** | <u>0.945</u> |
| held divisor alone, held after one sixteenth of the run, learning rate 0.01 | 0.934 | 0.941 | 0.942 |
| held divisor alone, held after one sixty-fourth of the run, learning rate 0.003 | 0.902 | 0.937 | 0.928 |
| held divisor alone, held after one sixty-fourth of the run, learning rate 0.006 | 0.929 | **0.964** | 0.948 |
| held divisor alone, held after one sixty-fourth of the run, learning rate 0.01 | 0.944 | <u>0.949</u> | 0.943 |
| slower predictor with divisor held after one sixty-fourth of the run, learning rate 0.003 | 0.940 | 0.929 | 0.915 |
| slower predictor with divisor held after one sixty-fourth of the run, learning rate 0.006 | 0.944 | <u>0.954</u> | 0.888 |
| slower predictor with divisor held after one sixty-fourth of the run, learning rate 0.01 | **0.956** | 0.939 | 0.904 |
| predictor at half speed with held divisor, learning rate 0.003 | 0.943 | 0.941 | 0.910 |
| predictor at half speed with held divisor, learning rate 0.006 | **0.959** | <u>0.953</u> | 0.916 |
| predictor at half speed with held divisor, learning rate 0.01 | 0.928 | 0.938 | 0.924 |

#### Mean capped first-goal time on every configuration of the methods

The same rows and columns, $M$ in thousands of steps; over each method's whole grid the smallest is bold and the second smallest underlined.

| method and its other settings | weight 1 | weight 2 | weight 4 |
|---|---|---|---|
| held divisor alone, held after one sixteenth of the run, learning rate 0.003 | 41.5 | 21.8 | 21.6 |
| held divisor alone, held after one sixteenth of the run, learning rate 0.006 | 20.7 | 19.2 | **16.2** |
| held divisor alone, held after one sixteenth of the run, learning rate 0.01 | 26.1 | 17.8 | <u>16.7</u> |
| held divisor alone, held after one sixty-fourth of the run, learning rate 0.003 | 70.9 | 29.5 | 21.8 |
| held divisor alone, held after one sixty-fourth of the run, learning rate 0.006 | 26.4 | <u>14.8</u> | **12.9** |
| held divisor alone, held after one sixty-fourth of the run, learning rate 0.01 | 29.4 | 20.7 | 15.4 |
| slower predictor with divisor held after one sixty-fourth of the run, learning rate 0.003 | 22.8 | 12.6 | 13.8 |
| slower predictor with divisor held after one sixty-fourth of the run, learning rate 0.006 | 13.1 | 15 | 21.1 |
| slower predictor with divisor held after one sixty-fourth of the run, learning rate 0.01 | 18.8 | <u>10.3</u> | **9.29** |
| predictor at half speed with held divisor, learning rate 0.003 | 16.6 | 20.7 | 21.2 |
| predictor at half speed with held divisor, learning rate 0.006 | 14.1 | 12.9 | 15 |
| predictor at half speed with held divisor, learning rate 0.01 | 10.7 | <u>10.6</u> | **10.5** |

## Validation seeds, case by case

A case is one environment with one reward setting. The validation seeds are neither the development seeds nor the confirmation seeds. They come in two blocks. On the first block each side stands at its best configuration among the few run there, so that block is a second selection step. Only complete units are read; nothing is estimated for the others.

### PointMaze Medium, reward 1 on arrival

#### First block: each side at its best configuration there

The units of the phase `validation-block` in PointMaze Medium, reward 1 on arrival. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006 | 0.9349 ± 0.0031 | 0.9950 ± 0.0010 | 25 ± 1.46 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.01 | 0.9275 ± 0.0027 | 0.9892 ± 0.0015 | 23.5 ± 1.28 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.006 | 0.9194 ± 0.0023 | 0.9794 ± 0.0023 | 19.9 ± 1.16 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 | 0.9382 ± 0.0027 | 0.9995 ± 0.0003 | 21 ± 1.22 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 | 0.9374 ± 0.0027 | <u>0.9999</u> ± 0.0000 | 28.6 ± 1.69 | 256 of 256 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.01 | 0.9388 ± 0.0026 | <u>0.9999</u> ± 0.0001 | 22.4 ± 1.27 | 256 of 256 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.006 | 0.9340 ± 0.0035 | <u>0.9999</u> ± 0.0001 | 30.6 ± 2.09 | 256 of 256 |
| slower predictor with divisor held after one sixty-fourth of the run | intrinsic weight 1, learning rate 0.01 | 0.9362 ± 0.0032 | **1.0000** ± 0.0000 | 19.8 ± 1.09 | 256 of 256 |
| slower predictor with divisor held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.006 | 0.9346 ± 0.0030 | **1.0000** ± 0.0000 | <u>17.8</u> ± 1.05 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.006 | **0.9438** ± 0.0030 | **1.0000** ± 0.0000 | 22.6 ± 1.75 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 2, learning rate 0.006 | <u>0.9399</u> ± 0.0028 | **1.0000** ± 0.0000 | **17.4** ± 1.07 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run), the finalist of batch 4 | intrinsic weight 2, learning rate 0.003 | 0.9348 ± 0.0030 | <u>0.9999</u> ± 0.0000 | 17.9 ± 1 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run), the finalist of batch 4 | intrinsic weight 0.5, learning rate 0.006 | 0.9284 ± 0.0040 | **1.0000** ± 0.0000 | 33.9 ± 2.64 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run), the finalist of batch 4 | intrinsic weight 0.25, learning rate 0.01 | 0.9195 ± 0.0042 | **1.0000** ± 0.0000 | 36.2 ± 2.53 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006 (best of 3) | 0.9349 ± 0.0031 | control | 0.9950 ± 0.0010 | control | 256 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.006 (best of 2) | **0.9438** ± 0.0030 | **+0.0089** [+0.0013, +0.0164] | **1.0000** ± 0.0000 | **+0.0050** [+0.0034, +0.0071] | 256 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.01 (best of 2) | <u>0.9388</u> ± 0.0026 | <u>+0.0039</u> [-0.0034, +0.0113] | <u>0.9999</u> ± 0.0001 | <u>+0.0049</u> [+0.0032, +0.0070] | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 (best of 2) | 0.9382 ± 0.0027 | +0.0033 [-0.0042, +0.0109] | 0.9995 ± 0.0003 | +0.0045 [+0.0027, +0.0066] | 256 |
| slower predictor with divisor held after one sixty-fourth of the run | intrinsic weight 1, learning rate 0.01 (best of 2) | 0.9362 ± 0.0032 | +0.0013 [-0.0070, +0.0097] | **1.0000** ± 0.0000 | **+0.0050** [+0.0034, +0.0071] | 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run), the finalist of batch 4 | intrinsic weight 2, learning rate 0.003 (best of 3) | 0.9348 ± 0.0030 | -0.0001 [-0.0084, +0.0083] | <u>0.9999</u> ± 0.0000 | <u>+0.0049</u> [+0.0033, +0.0070] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.006 (best of 3) | 19.9 ± 1.16 | control | 256 of 256 | 0.9194 | 0.9794 | 256 |
| predictor at half speed with held divisor | intrinsic weight 2, learning rate 0.006 (best of 2) | **17.4** ± 1.07 | **-2.54** [-5.53, +0.434] | 256 of 256 | **0.9399** | **1.0000** | 256 |
| slower predictor with divisor held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.006 (best of 2) | <u>17.8</u> ± 1.05 | <u>-2.1</u> [-5.16, +1.01] | 256 of 256 | 0.9346 | **1.0000** | 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run), the finalist of batch 4 | intrinsic weight 2, learning rate 0.003 (best of 3) | 17.9 ± 1 | -1.98 [-5.03, +0.955] | 256 of 256 | 0.9348 | <u>0.9999</u> | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 (best of 2) | 21 ± 1.22 | +1.08 [-1.61, +3.82] | 256 of 256 | 0.9382 | 0.9995 | 256 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.01 (best of 2) | 22.4 ± 1.27 | +2.46 [-0.627, +5.49] | 256 of 256 | <u>0.9388</u> | <u>0.9999</u> | 256 |

##### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:53 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 4 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:44 PT | none |
| 5 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:45 PT | none |
| 6 | held divisor alone, held after one sixty-fourth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:46 PT | none |
| 7 | held divisor alone, held after one sixty-fourth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:48 PT | none |
| 8 | slower predictor with divisor held after one sixty-fourth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:49 PT | none |
| 9 | slower predictor with divisor held after one sixty-fourth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:48 PT | none |
| 10 | predictor at half speed with held divisor | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:52 PT | none |
| 11 | predictor at half speed with held divisor | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:51 PT | none |
| 12 | slower predictor with held divisor (divisor held after one sixteenth of the run), the finalist of batch 4 | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:56 PT | none |
| 13 | slower predictor with held divisor (divisor held after one sixteenth of the run), the finalist of batch 4 | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 14 | slower predictor with held divisor (divisor held after one sixteenth of the run), the finalist of batch 4 | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:56 PT | none |

Unit names, by the number in the first column:

1. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-one-sixty-fourth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-one-sixty-fourth_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-arrival-plus-one_slow-predictor-held-sixty-fourth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-medium_reward-arrival-plus-one_slow-predictor-held-sixty-fourth_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-medium_reward-arrival-plus-one_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-medium_reward-arrival-plus-one_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-medium_reward-arrival-plus-one_slower-predictor-with-held-divisor_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-medium_reward-arrival-plus-one_slower-predictor-with-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`
14. `validation-block_pointmaze-medium_reward-arrival-plus-one_slower-predictor-with-held-divisor_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.25_clip-coefficient-0.1_seeds-10000-to-10255`

## Predictions

The predictions were written in `ideas.md` before the runs. Each is quoted, then read part by part: a part held or did not hold on the numbers given, or cannot be read yet because a unit it needs is not complete. A prediction held when every part held, and did not hold when at least one part did not. Where a prediction leaves a choice, it is read as follows. Every arm is read on the first validation block at its best configuration for the whole-curve mean $W$ among those run there, and the control at its own; a part waits until every configuration planned there for its arms and for the control is read. "Level with RND (within 0.01)" is read as: the mean difference in $W$ from the control is at most 0.01 in size. "Ahead" is read as: the 95% interval of the difference is entirely above zero. "Raises the return between 66 and 200 thousand steps" is read as: the mean training return over those steps, paired by seed, is higher with the divisor held after 1/64 of the run than with it held after 1/16, at the same predictor speed (mean difference above zero; the interval is given beside it, and a part that held with an interval across zero is a difference the seeds do not separate from none). "Between RND and the finalist at every point of the curve" is read as: at every logging window the seed-averaged return of the arm lies between that of the control and that of the slower predictor with held divisor.

### Prediction 1: the held divisor alone, held after 1/16 of the run

> **Held divisor alone (1/16)** is level with RND on the whole-curve mean (within 0.01 on the validation seeds) and ahead on the final level, as on PointMaze Large.

Observed: **held**.

- It is level with the control on the whole-curve mean (within 0.01): **held**. $W$ 0.9382, difference from the control +0.0033 [-0.0042, +0.0109].
- It is ahead of the control on the final level: **held**. $L$ 0.9995, difference from the control +0.0045 [+0.0027, +0.0066].

### Prediction 2: holding the divisor after 1/64 instead of 1/16

> **Holding the divisor after 1/64 instead of 1/16** raises the return between 66 and 200 thousand steps for both predictor speeds, and with the slower predictor it is ahead of RND on the whole-curve mean on the validation seeds (interval above zero).

Observed: **did not hold**.

- It raises the return between 66 and 200 thousand steps with the predictor at RND's speed: **held**. Mean return over steps 66,000 to 200,000, held after 1/64 minus held after 1/16, on 256 seeds: +0.0113 [-0.0236, +0.0454]; the interval stands across zero.
- It raises the return between 66 and 200 thousand steps with the slower predictor: **held**. Mean return over steps 66,000 to 200,000, held after 1/64 minus held after 1/16, on 256 seeds: +0.0333 [-0.0085, +0.0765]; the interval stands across zero.
- With the slower predictor it is ahead of the control on the whole-curve mean: **did not hold**. $W$ 0.9362, difference from the control +0.0013 [-0.0070, +0.0097].

### Prediction 3: the predictor at half speed

> **The predictor at half speed** lies between RND and the finalist at every point of the curve.

Observed: **did not hold**.

- Its curve lies between the control's and the finalist's at every logging window: **did not hold**. Of 256 logging windows the arm is between the two in 119, above both in 137 and below both in 0. Whole-curve means: the arm 0.9438, the control 0.9349, the slower predictor with held divisor 0.9348.

