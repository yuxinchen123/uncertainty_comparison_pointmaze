# Batch 4: the two finalists' development sweeps in the other three PointMaze cases

Generated 2026-11-12 19:55 PT by the command below. This file is generated: regenerate it, do not edit it. Times are Pacific, converted from machines running Eastern. Run folder: `2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days`; batch folder: `batches/04_two-finalists_development-sweeps-in-the-other-three-pointmaze-cases`.

```bash
cd /p/rlprojects/RND/10_jax_exploration_platform
bash "benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh" -s scripts/maze_campaign/analysis/batch_summary_job.py \
    --run-dir=runs/2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days \
    --batch=04_two-finalists_development-sweeps-in-the-other-three-pointmaze-cases
```

## Contents

| section | first written | last changed | status |
|---|---|---|---|
| [Measures](#measures) | 2026-11-12 19:43 PT | 2026-11-12 19:43 PT | unread |
| [Development sweeps against the baseline, case by case](#development-sweeps-against-the-baseline-case-by-case) | 2026-11-12 19:43 PT | 2026-11-12 19:55 PT | unread |
| [Predictions](#predictions) | 2026-11-12 19:43 PT | 2026-11-12 19:50 PT | unread |

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
| 1 | fast last layer with slower predictor and held divisor, step 0.1 | PointMaze Medium, reward 1 on arrival | 15 | complete | 256 of 256 | 2026-11-12 19:19 PT | none |
| 2 | slower predictor with held divisor | PointMaze Medium, reward 1 on arrival | 15 | complete | 256 of 256 | 2026-11-12 19:17 PT | none |
| 3 | fast last layer with slower predictor and held divisor, step 0.1 | PointMaze Medium, step cost | 15 | complete | 256 of 256 | 2026-11-12 19:21 PT | none |
| 4 | slower predictor with held divisor | PointMaze Medium, step cost | 15 | complete | 256 of 256 | 2026-11-12 19:16 PT | none |
| 5 | fast last layer with slower predictor and held divisor, step 0.1 | PointMaze Large, step cost | 15 | unfinished | 37 of 256 | not finished | none |
| 6 | slower predictor with held divisor | PointMaze Large, step cost | 15 | unfinished | 115 of 256 | not finished | none |

Unit names, by the number in the first column:

1. `development-at-the-budget_pointmaze-medium_reward-arrival-plus-one_fast-last-layer-step-0-point-1-slower-predictor-held-divisor_cells-01-to-15-of-15_seeds-0-to-15`
2. `development-at-the-budget_pointmaze-medium_reward-arrival-plus-one_slower-predictor-with-held-divisor_cells-01-to-15-of-15_seeds-0-to-15`
3. `development-at-the-budget_pointmaze-medium_reward-step-cost_fast-last-layer-step-0-point-1-slower-predictor-held-divisor_cells-01-to-15-of-15_seeds-0-to-15`
4. `development-at-the-budget_pointmaze-medium_reward-step-cost_slower-predictor-with-held-divisor_cells-01-to-15-of-15_seeds-0-to-15`
5. `development-at-the-budget_pointmaze-large_reward-step-cost_fast-last-layer-step-0-point-1-slower-predictor-held-divisor_cells-01-to-15-of-15_seeds-0-to-15`
6. `development-at-the-budget_pointmaze-large_reward-step-cost_slower-predictor-with-held-divisor_cells-01-to-15-of-15_seeds-0-to-15`

### PointMaze Medium, reward 1 on arrival

Read at 1,048,576 environment steps per policy. CleanRL PPO-RND: 16 of 16 configurations read (2 of 2 units complete). fast last layer with slower predictor and held divisor, step 0.1: 15 of 15 configurations read (1 of 1 units complete). slower predictor with held divisor: 15 of 15 configurations read (1 of 1 units complete).

#### Discovery and optimization: whole-curve mean

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the baseline [95% interval] | $L$ ± standard error | difference in $L$ from the baseline [95% interval] | best configuration at an end of its grid | seeds in common with the baseline |
|---|---|---|---|---|---|---|---|
| CleanRL PPO-RND | intrinsic weight 2, learning rate 0.01 (best of 16) | 0.944 ± 0.006 | baseline | <u>0.993</u> ± 0.004 | baseline | no | 16 |
| slower predictor with held divisor | intrinsic weight 0.5, learning rate 0.006 (best of 15) | **0.965** ± 0.006 | **+0.021** [+0.007, +0.036] | **1.000** ± 0.000 | +0.007 [+0.002, +0.016] | no | 16 |
| fast last layer with slower predictor and held divisor, step 0.1 | intrinsic weight 0.25, learning rate 0.01 (best of 15) | <u>0.946</u> ± 0.007 | <u>+0.002</u> [-0.016, +0.022] | **1.000** ± 0.000 | +0.007 [+0.001, +0.016] | intrinsic weight at the smallest of its grid; learning rate at the largest | 16 |

CleanRL PPO-RND's best configuration here has no complete unit on the confirmation seeds (phases `confirmation`): the confirmation units of this case hold intrinsic weight 2, learning rate 0.006.

#### Discovery: mean capped first-goal time

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the baseline [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | best configuration at an end of its grid | seeds in common with the baseline |
|---|---|---|---|---|---|---|---|---|
| CleanRL PPO-RND | intrinsic weight 4, learning rate 0.02 (best of 16) | **8.73** ± 1.31 | baseline | 16 of 16 | 0.655 | 0.812 | intrinsic weight at the largest of its grid | 16 |
| slower predictor with held divisor | intrinsic weight 2, learning rate 0.006 (best of 15) | <u>10.9</u> ± 1.94 | **+2.21** [-1.87, +6.77] | 16 of 16 | **0.935** | **1.000** | no | 16 |
| fast last layer with slower predictor and held divisor, step 0.1 | intrinsic weight 2, learning rate 0.003 (best of 15) | 13 ± 1.82 | <u>+4.25</u> [+0.375, +8.34] | 16 of 16 | <u>0.925</u> | <u>0.999</u> | learning rate at the smallest of its grid | 16 |

CleanRL PPO-RND's best configuration here has no complete unit on the confirmation seeds (phases `confirmation`): the confirmation units of this case hold intrinsic weight 2, learning rate 0.006.

Best configuration for $W$ at an end of its grid: fast last layer with slower predictor and held divisor, step 0.1 (intrinsic weight at the smallest of its grid; learning rate at the largest). Best configuration for $M$ at an end of its grid: fast last layer with slower predictor and held divisor, step 0.1 (learning rate at the smallest of its grid); CleanRL PPO-RND (intrinsic weight at the largest of its grid). The edge rule of `program.md` asks for more values on that side before anything is confirmed.

#### Whole-curve mean on every configuration of the methods

Rows: a method with its settings other than the intrinsic weight; columns: the intrinsic weights. Over each method's whole grid the highest $W$ is bold and the second highest underlined.

| method and its other settings | weight 0.25 | weight 0.5 | weight 1 | weight 2 | weight 4 |
|---|---|---|---|---|---|
| fast last layer with slower predictor and held divisor, step 0.1, learning rate 0.003 | 0.927 | 0.929 | 0.935 | 0.925 | 0.717 |
| fast last layer with slower predictor and held divisor, step 0.1, learning rate 0.006 | 0.935 | 0.895 | 0.896 | 0.867 | 0.813 |
| fast last layer with slower predictor and held divisor, step 0.1, learning rate 0.01 | **0.946** | <u>0.943</u> | 0.886 | 0.809 | 0.698 |
| slower predictor with held divisor, learning rate 0.003 | 0.918 | 0.950 | 0.951 | <u>0.959</u> | 0.902 |
| slower predictor with held divisor, learning rate 0.006 | 0.869 | **0.965** | 0.957 | 0.935 | 0.882 |
| slower predictor with held divisor, learning rate 0.01 | 0.958 | 0.935 | 0.952 | 0.935 | 0.908 |

#### Mean capped first-goal time on every configuration of the methods

The same rows and columns, $M$ in thousands of steps; over each method's whole grid the smallest is bold and the second smallest underlined.

| method and its other settings | weight 0.25 | weight 0.5 | weight 1 | weight 2 | weight 4 |
|---|---|---|---|---|---|
| fast last layer with slower predictor and held divisor, step 0.1, learning rate 0.003 | 23.3 | 30.2 | 14.8 | **13** | 28.5 |
| fast last layer with slower predictor and held divisor, step 0.1, learning rate 0.006 | 36.5 | 21.8 | 18.7 | 20.8 | 15 |
| fast last layer with slower predictor and held divisor, step 0.1, learning rate 0.01 | <u>13.6</u> | 19.8 | 15.7 | 15.8 | 14.1 |
| slower predictor with held divisor, learning rate 0.003 | 48 | 14.3 | 19.4 | 12.6 | 21.7 |
| slower predictor with held divisor, learning rate 0.006 | 76.2 | 14.5 | <u>12.5</u> | **10.9** | 19 |
| slower predictor with held divisor, learning rate 0.01 | 15.6 | 27.1 | 13.9 | 12.8 | 12.9 |

### PointMaze Medium, step cost

Read at 1,048,576 environment steps per policy. CleanRL PPO-RND: 18 of 18 configurations read (2 of 2 units complete). fast last layer with slower predictor and held divisor, step 0.1: 15 of 15 configurations read (1 of 1 units complete). slower predictor with held divisor: 15 of 15 configurations read (1 of 1 units complete).

#### Discovery and optimization: whole-curve mean

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the baseline [95% interval] | $L$ ± standard error | difference in $L$ from the baseline [95% interval] | best configuration at an end of its grid | seeds in common with the baseline |
|---|---|---|---|---|---|---|---|
| CleanRL PPO-RND | intrinsic weight 0.00195312, learning rate 0.006 (best of 18) | **-0.341** ± 0.003 | baseline | <u>-0.304</u> ± 0.000 | baseline | learning rate at the largest of its grid | 16 |
| fast last layer with slower predictor and held divisor, step 0.1 | intrinsic weight 0.0078125, learning rate 0.006 (best of 15) | **-0.341** ± 0.004 | **-0.001** [-0.011, +0.008] | **-0.303** ± 0.000 | **+0.001** [-0.000, +0.001] | no | 16 |
| slower predictor with held divisor | intrinsic weight 0.00195312, learning rate 0.01 (best of 15) | <u>-0.342</u> ± 0.003 | <u>-0.002</u> [-0.010, +0.007] | -0.305 ± 0.001 | <u>-0.001</u> [-0.002, +0.000] | learning rate at the largest of its grid | 16 |

On 1,024 confirmation seeds the same configuration of CleanRL PPO-RND has $W$ -0.352 ± 0.001.

#### Discovery: mean capped first-goal time

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the baseline [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | best configuration at an end of its grid | seeds in common with the baseline |
|---|---|---|---|---|---|---|---|---|
| CleanRL PPO-RND | intrinsic weight 0.00195312, learning rate 0.006 (best of 18) | <u>10.6</u> ± 1.56 | baseline | 16 of 16 | **-0.341** | **-0.304** | learning rate at the largest of its grid | 16 |
| slower predictor with held divisor | intrinsic weight 0.125, learning rate 0.01 (best of 15) | **10.3** ± 2.44 | **-0.259** [-4.59, +4.79] | 16 of 16 | -0.349 | <u>-0.308</u> | intrinsic weight at the largest of its grid; learning rate at the largest | 16 |
| fast last layer with slower predictor and held divisor, step 0.1 | intrinsic weight 0.00195312, learning rate 0.01 (best of 15) | **10.3** ± 2 | <u>-0.258</u> [-2.36, +1.95] | 16 of 16 | <u>-0.348</u> | -0.309 | learning rate at the largest of its grid | 16 |

On 1,024 confirmation seeds the same configuration of CleanRL PPO-RND has $M$ 20.8 ± 0.619 thousand steps.

Best configuration for $W$ at an end of its grid: slower predictor with held divisor (learning rate at the largest of its grid); CleanRL PPO-RND (learning rate at the largest of its grid). Best configuration for $M$ at an end of its grid: fast last layer with slower predictor and held divisor, step 0.1 (learning rate at the largest of its grid); slower predictor with held divisor (intrinsic weight at the largest of its grid; learning rate at the largest); CleanRL PPO-RND (learning rate at the largest of its grid). The edge rule of `program.md` asks for more values on that side before anything is confirmed.

#### Whole-curve mean on every configuration of the methods

Rows: a method with its settings other than the intrinsic weight; columns: the intrinsic weights. Over each method's whole grid the highest $W$ is bold and the second highest underlined.

| method and its other settings | weight 0.000488281 | weight 0.00195312 | weight 0.0078125 | weight 0.03125 | weight 0.125 |
|---|---|---|---|---|---|
| fast last layer with slower predictor and held divisor, step 0.1, learning rate 0.003 | -0.347 | -0.370 | -0.348 | -0.348 | -0.348 |
| fast last layer with slower predictor and held divisor, step 0.1, learning rate 0.006 | <u>-0.346</u> | <u>-0.346</u> | **-0.341** | -0.352 | <u>-0.346</u> |
| fast last layer with slower predictor and held divisor, step 0.1, learning rate 0.01 | <u>-0.346</u> | -0.348 | -0.349 | -0.362 | -0.354 |
| slower predictor with held divisor, learning rate 0.003 | -0.355 | -0.361 | -0.353 | <u>-0.348</u> | -0.368 |
| slower predictor with held divisor, learning rate 0.006 | -0.356 | -0.349 | -0.352 | -0.356 | -0.351 |
| slower predictor with held divisor, learning rate 0.01 | -0.358 | **-0.342** | -0.355 | -0.351 | -0.349 |

#### Mean capped first-goal time on every configuration of the methods

The same rows and columns, $M$ in thousands of steps; over each method's whole grid the smallest is bold and the second smallest underlined.

| method and its other settings | weight 0.000488281 | weight 0.00195312 | weight 0.0078125 | weight 0.03125 | weight 0.125 |
|---|---|---|---|---|---|
| fast last layer with slower predictor and held divisor, step 0.1, learning rate 0.003 | 17.4 | 17.4 | 17.5 | 13.3 | 13.5 |
| fast last layer with slower predictor and held divisor, step 0.1, learning rate 0.006 | 14.5 | 15.3 | 16.3 | 20.6 | <u>10.9</u> |
| fast last layer with slower predictor and held divisor, step 0.1, learning rate 0.01 | 13.7 | **10.3** | 14.5 | 16.2 | 13.5 |
| slower predictor with held divisor, learning rate 0.003 | 16.9 | 23.8 | 18.2 | 15.9 | 19.6 |
| slower predictor with held divisor, learning rate 0.006 | 23.3 | 12.5 | <u>12.3</u> | 17.6 | 14.2 |
| slower predictor with held divisor, learning rate 0.01 | 14 | <u>12.3</u> | 17.9 | 14.9 | **10.3** |

### PointMaze Large, step cost

Read at 33,554,432 environment steps per policy. CleanRL PPO-RND: 12 of 12 configurations read (1 of 1 units complete). fast last layer with slower predictor and held divisor, step 0.1: 0 of 15 configurations read (0 of 1 units complete). slower predictor with held divisor: 0 of 15 configurations read (0 of 1 units complete). Not read yet, no complete unit: fast last layer with slower predictor and held divisor, step 0.1; slower predictor with held divisor.

## Predictions

The predictions were written in `ideas.md` before the runs. Each is quoted, then read part by part: a part held or did not hold on the numbers given, or cannot be read yet because a unit it needs is not complete. A prediction held when every part held, and did not hold when at least one part did not. Where a prediction leaves a choice, it is read as follows. Each side is read at its own best configuration on the development seeds: for the whole-curve mean $W$ the one with the highest $W$, for "reach the goal sooner" the one with the smallest mean capped first-goal time $M$. "Sooner" is read as a negative mean difference in $M$ from the baseline; the interval is given beside it. "Level with or slightly ahead" is read as: the 95% interval of the difference in $W$ is not entirely below zero. "Not ahead" is read as: that interval is not entirely above zero. "The worse group" of prediction 3 is read as the seeds whose whole-curve mean is below -0.81, halfway between the two group values the prediction names.

### Prediction 1: PointMaze Medium, reward 1 on arrival

> **PointMaze Medium, reward 1 on arrival:** both finalists reach the goal sooner than RND and are level with or slightly ahead of it on the whole-curve mean (RND: 0.932 on fresh seeds), since the map is solved early and little is left to gain.

Observed: **did not hold**.

- Both finalists reach the goal sooner than the baseline: **did not hold**. Difference in $M$ from the baseline, in thousands of steps: fast last layer with slower predictor and held divisor, step 0.1 +4.25 [+0.375, +8.34]; slower predictor with held divisor +2.21 [-1.87, +6.77].
- Both are level with or slightly ahead of the baseline on the whole-curve mean: **held**. At the best configurations for $W$: fast last layer with slower predictor and held divisor, step 0.1 $W$ 0.946, difference from the baseline +0.002 [-0.016, +0.022]; slower predictor with held divisor $W$ 0.965, difference from the baseline +0.021 [+0.007, +0.036].

### Prediction 2: PointMaze Medium, step cost

> **PointMaze Medium, step cost:** RND's best weight there is nearly zero and PPO without a bonus is ahead of RND. A bonus-only change is not expected to be ahead of RND's nearly bonus-free configuration; if the finalists' best weight is also the smallest one, this case needs a change outside the bonus (days 6 to 10).

Observed: **held**.

- No finalist is ahead of the baseline on the whole-curve mean: **held**. At the best configurations for $W$: fast last layer with slower predictor and held divisor, step 0.1 $W$ -0.341, difference from the baseline -0.001 [-0.011, +0.008]; slower predictor with held divisor $W$ -0.342, difference from the baseline -0.002 [-0.010, +0.007].
- Beside: fast last layer with slower predictor and held divisor, step 0.1: best intrinsic weight 0.0078125, not the smallest of its grid; slower predictor with held divisor: best intrinsic weight 0.00195312, not the smallest of its grid. The prediction says that a best weight at the smallest of the grid would mean the case needs a change outside the bonus.

### Prediction 3: PointMaze Large, step cost

> **PointMaze Large, step cost:** RND's seeds split into two groups there (about −0.72 and −0.90). The held divisor should lower the share of seeds in the worse group, because that group is the one that keeps exploring after the first arrivals.

Observed: **cannot be read yet**.

- Each finalist has a smaller share of seeds in the worse group than the baseline: **cannot be read yet**. Waiting for complete units in PointMaze Large, step cost: fast last layer with slower predictor and held divisor, step 0.1; slower predictor with held divisor.
- Beside: Waiting for complete units in PointMaze Large, step cost: fast last layer with slower predictor and held divisor, step 0.1; slower predictor with held divisor.

