# Standing of the bonus arms against CleanRL PPO-RND on the validation seeds, the four PointMaze cases

Generated 2026-11-13 06:48 PT by the command below. This file is generated: regenerate it, do not edit it. Times are Pacific, converted from machines running Eastern. Run folder: `2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days`.

```bash
cd /p/rlprojects/RND/10_jax_exploration_platform
bash "benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/submit_tests.sh" -s scripts/maze_campaign/analysis/standing_job.py \
    --run-dir=runs/2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_reward-arrival-plus-1-and-step-cost_bonus-5-days_other-parts-5-days
```

## Contents

| section | first written | last changed | status |
|---|---|---|---|
| [Standing in one table](#standing-in-one-table) | 2026-11-13 00:10 PT | 2026-11-13 06:48 PT | unread |
| [Measures](#measures) | 2026-11-13 00:10 PT | 2026-11-13 00:10 PT | unread |
| [PointMaze Medium, reward 1 on arrival](#pointmaze-medium-reward-1-on-arrival) | 2026-11-13 00:10 PT | 2026-11-13 06:48 PT | unread |
| [PointMaze Medium, step cost](#pointmaze-medium-step-cost) | 2026-11-13 00:10 PT | 2026-11-13 06:48 PT | unread |
| [PointMaze Large, reward 1 on arrival](#pointmaze-large-reward-1-on-arrival) | 2026-11-13 00:10 PT | 2026-11-13 06:48 PT | unread |
| [PointMaze Large, step cost](#pointmaze-large-step-cost) | 2026-11-13 00:10 PT | 2026-11-13 06:48 PT | unread |

Times are Pacific, converted from machines running Eastern, and come from this report's own history of regenerations. [The same report with unread sections in blue and changed text in brown](standing.html). No section marks itself read.

## Standing in one table

Each cell gives a verdict word, the paired difference from the control (CleanRL PPO-RND) with its 95% interval, and the block of validation seeds it rests on. For the whole-curve mean $W$: "ahead" when the interval is entirely above zero, "behind" when entirely below, else "level". For the mean capped first-goal time $M$: "ahead" when the interval is entirely below zero. A cell is for the family's arm with the highest $W$ on the first block, among its arms run on the block the cell rests on. It rests on the second block when an arm of the family was run there (the method at the configuration fixed on the first block, against the control at the configuration that was its best on the first block), else on the first block (each side at its best configuration there, chosen on those same seeds, which is a second selection step). "no complete comparison yet" means units exist and the method or the control has none complete; "not run" means no unit; "not final, units still running" means the cell is read from the complete units while other configurations of the family or of the control are not complete, so a best configuration and the verdict can still change. None of this is the comparison on the confirmation seeds. The families are in rows and the four cases in columns.

| family of arms | PointMaze Medium, reward 1 on arrival | PointMaze Medium, step cost | PointMaze Large, reward 1 on arrival | PointMaze Large, step cost |
|---|---|---|---|---|
| held divisor alone (whole-curve mean) | ahead: +0.0109 [+0.0037, +0.0183], second block under equal rules | ahead: +0.0058 [+0.0007, +0.0115], second block under equal rules | level: -0.0023 [-0.0098, +0.0043], second block under equal rules | ahead: +0.0095 [+0.0051, +0.0141], second block under equal rules |
| divisor held from the first arrival (whole-curve mean) | ahead: +0.0095 [+0.0015, +0.0178], second block under equal rules | ahead: +0.0065 [+0.0012, +0.0122], second block under equal rules | level: -0.0003 [-0.0066, +0.0059], second block under equal rules | ahead: +0.0072 [+0.0013, +0.0131], second block under equal rules |
| fast last layer throughout with the first-arrival hold (whole-curve mean) | ahead: +0.0095 [+0.0019, +0.0171], second block under equal rules | level: +0.0041 [-0.0017, +0.0103], second block under equal rules | ahead: +0.0153 [+0.0096, +0.0209], second block under equal rules | level: +0.0046 [-0.0002, +0.0094], first block; not final, units still running |
| fast last layer until the first arrival, then the predictor's own divisor (whole-curve mean) | ahead: +0.0095 [+0.0023, +0.0169], first block | level: +0.0029 [-0.0013, +0.0072], first block | ahead: +0.0114 [+0.0054, +0.0174], first block | behind: -0.0099 [-0.0184, -0.0017], first block; not final, units still running |
| slower predictor with held divisor (whole-curve mean) | level: +0.0013 [-0.0070, +0.0097], first block | not run | ahead: +0.0102 [+0.0048, +0.0159], first block | ahead: +0.0081 [+0.0032, +0.0129], first block |
| predictor at half speed with held divisor (whole-curve mean) | ahead: +0.0102 [+0.0012, +0.0186], second block under equal rules | level: +0.0014 [-0.0033, +0.0059], first block | ahead: +0.0068 [+0.0007, +0.0129], first block | ahead: +0.0094 [+0.0048, +0.0140], first block |
| all three parts (whole-curve mean) | behind: -0.0288 [-0.0401, -0.0180], first block | not run | ahead: +0.0137 [+0.0071, +0.0202], first block | level: +0.0019 [-0.0041, +0.0077], first block |
| fast last layer (mean capped first-goal time) | ahead: -4.99 [-8.32, -2.37] thousand steps, first block of the discovery check | level: -1.04 [-2.35, +0.241] thousand steps, first block of the discovery check | ahead: -63.4 [-75.2, -52] thousand steps, first block of the discovery check | ahead: -61.3 [-73.5, -49.6] thousand steps, first block of the discovery check |

Which arm each cell is for:

- **held divisor alone, PointMaze Medium, reward 1 on arrival:** held divisor alone, held after one sixty-fourth of the run; the control's first-block best is also its best on the second block. Other arms of the family on the second block under equal rules: held divisor alone, held after one sixteenth of the run +0.0090 [+0.0023, +0.0159] (ahead).
- **held divisor alone, PointMaze Medium, step cost:** held divisor alone, held after one sixteenth of the run; with the control at its best on the second block the difference is +0.0033 [-0.0004, +0.0069] (level).
- **held divisor alone, PointMaze Large, reward 1 on arrival:** held divisor alone, held after one sixteenth of the run; the control's first-block best is also its best on the second block.
- **held divisor alone, PointMaze Large, step cost:** held divisor alone, held after one sixteenth of the run; the control's first-block best is also its best on the second block.
- **divisor held from the first arrival, PointMaze Medium, reward 1 on arrival:** divisor held from each policy's own first arrival; the control's first-block best is also its best on the second block.
- **divisor held from the first arrival, PointMaze Medium, step cost:** divisor held from each policy's own first arrival; with the control at its best on the second block the difference is +0.0040 [-0.0001, +0.0080] (level).
- **divisor held from the first arrival, PointMaze Large, reward 1 on arrival:** divisor held from each policy's own first arrival; the control's first-block best is also its best on the second block.
- **divisor held from the first arrival, PointMaze Large, step cost:** divisor held from each policy's own first arrival; the control's first-block best is also its best on the second block.
- **fast last layer throughout with the first-arrival hold, PointMaze Medium, reward 1 on arrival:** fast last layer at step 0.01 throughout, divisor held from the first arrival; the control's first-block best is also its best on the second block.
- **fast last layer throughout with the first-arrival hold, PointMaze Medium, step cost:** fast last layer at step 0.01 throughout, divisor held from the first arrival; with the control at its best on the second block the difference is +0.0016 [-0.0031, +0.0062] (level).
- **fast last layer throughout with the first-arrival hold, PointMaze Large, reward 1 on arrival:** fast last layer at step 0.03 throughout, divisor held from the first arrival; the control's first-block best is also its best on the second block. Other arms of the family on the second block under equal rules: fast last layer at step 0.01 throughout, divisor held from the first arrival +0.0103 [+0.0049, +0.0158] (ahead).
- **fast last layer throughout with the first-arrival hold, PointMaze Large, step cost:** fast last layer at step 0.1 throughout, divisor held from the first arrival. Not final: fast last layer at step 0.03 throughout, divisor held from the first arrival: 0 of 1 configurations read (0 of 1 units complete); fast last layer at step 0.01 throughout, divisor held from the first arrival: 0 of 1 configurations read (0 of 1 units complete).
- **fast last layer until the first arrival, then the predictor's own divisor, PointMaze Medium, reward 1 on arrival:** fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held. Other arms of the family on the first block: fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, left running +0.0036 [-0.0041, +0.0114] (level).
- **fast last layer until the first arrival, then the predictor's own divisor, PointMaze Medium, step cost:** fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held.
- **fast last layer until the first arrival, then the predictor's own divisor, PointMaze Large, reward 1 on arrival:** fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, left running. Other arms of the family on the first block: fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held +0.0025 [-0.0043, +0.0093] (level).
- **fast last layer until the first arrival, then the predictor's own divisor, PointMaze Large, step cost:** fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held. Not final: fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held: 1 of 3 configurations read (1 of 3 units complete); fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, left running: 0 of 1 configurations read (0 of 1 units complete).
- **slower predictor with held divisor, PointMaze Medium, reward 1 on arrival:** slower predictor with divisor held after one sixty-fourth of the run. Other arms of the family on the first block: slower predictor with held divisor (divisor held after one sixteenth of the run) -0.0001 [-0.0084, +0.0083] (level).
- **slower predictor with held divisor, PointMaze Large, reward 1 on arrival:** slower predictor with held divisor (divisor held after one sixteenth of the run).
- **slower predictor with held divisor, PointMaze Large, step cost:** slower predictor with held divisor (divisor held after one sixteenth of the run).
- **predictor at half speed with held divisor, PointMaze Medium, reward 1 on arrival:** predictor at half speed with held divisor; the control's first-block best is also its best on the second block.
- **predictor at half speed with held divisor, PointMaze Medium, step cost:** predictor at half speed with held divisor.
- **predictor at half speed with held divisor, PointMaze Large, reward 1 on arrival:** predictor at half speed with held divisor.
- **predictor at half speed with held divisor, PointMaze Large, step cost:** predictor at half speed with held divisor.
- **all three parts, PointMaze Medium, reward 1 on arrival:** all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run.
- **all three parts, PointMaze Large, reward 1 on arrival:** all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run. Other arms of the family on the first block: all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run +0.0078 [+0.0014, +0.0143] (ahead).
- **all three parts, PointMaze Large, step cost:** all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run.
- **fast last layer, PointMaze Medium, reward 1 on arrival:** fast last layer inside the rollout, step 0.1, the method of the check with the smallest $M$.
- **fast last layer, PointMaze Medium, step cost:** fast last layer inside the rollout, step 0.1, the method of the check with the smallest $M$.
- **fast last layer, PointMaze Large, reward 1 on arrival:** fast last layer inside the rollout, step 0.1, the method of the check with the smallest $M$. Other methods of the check: all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run -54 [-65.8, -42.7] thousand steps (ahead).
- **fast last layer, PointMaze Large, step cost:** fast last layer inside the rollout, step 0.1, the method of the check with the smallest $M$.

## Measures

Two metrics are reported, each in its own table with its own best configuration per method: the whole-curve mean $W$ measures discovery and optimization together, and the mean capped first-goal time $M$ measures discovery alone. The two are never mixed in one choice.

Define $K$ as the number of policies (seeds) of one configuration, $B$ as the budget in environment steps per policy (named with every table), $R_i(s)$ as the mean return of the training episodes of policy $i$ that end in the logging window containing step $s$, and $\Delta s$ as the number of that window's steps inside the range being averaged. Define $H$ as the episode step limit and $t$ as the step of an episode at which the goal is reached. With reward 1 on arrival an episode returns 1 if it reaches the goal and 0 otherwise; with the step cost it returns $-(t-1)/H$ if it reaches the goal and $-1$ otherwise. A window in which no episode ends takes the value of the window before it; before a policy's first finished episode the value is the return of an episode that never reaches the goal.

1. **Whole-curve mean:** $W(B) = \dfrac{1}{K}\sum_{i=1}^{K} \dfrac{1}{B}\sum_{s \le B} R_i(s) \thinspace \Delta s$, the area under the training curve divided by the budget.
2. **Final level:** $L(B) = \dfrac{1}{K}\sum_{i=1}^{K} \dfrac{1}{0.1 B}\sum_{0.9 B \lt s \le B} R_i(s) \thinspace \Delta s$, the same mean over the last 10% of the budget.
3. **Mean capped first-goal time:** define $T_i$ as the training step of policy $i$'s first arrival at the goal ($T_i = \infty$ if it has none by step $B$). $M(B) = \dfrac{1}{K}\sum_{i=1}^{K} \min(T_i, B)$. Smaller is better. The tables give $M$ in thousands of steps.
4. **Best configuration:** per method and case, for $W$ the configuration with the highest $W$ (ties: the higher $L$), for $M$ the one with the smallest $M$ (ties: more policies reach the goal). The two can differ.
5. **Difference from the baseline:** define $d_i$ as the value of seed $i$ under the method at its best configuration minus its value under the baseline at the baseline's own best configuration for the same metric. The tables give $\bar d = \dfrac{1}{K}\sum_{i=1}^{K} d_i$ over the seeds both have, with a 95% interval: the 2.5% and 97.5% points of the means of 10,000 resamples of those seeds, drawn with replacement. For $W$ and $L$ a method is ahead when the interval is entirely above zero; for $M$ it is ahead when the interval is entirely below zero.

Standard errors are over the $K$ policies.

## PointMaze Medium, reward 1 on arrival

### First block: each side at its best configuration there

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
| divisor held from each policy's own first arrival | intrinsic weight 4, learning rate 0.006 | 0.9435 ± 0.0027 | <u>0.9999</u> ± 0.0000 | 18.5 ± 1.13 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 2, learning rate 0.01 | 0.9412 ± 0.0024 | **1.0000** ± 0.0000 | 23.1 ± 1.34 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 2, learning rate 0.006 | 0.9334 ± 0.0029 | **1.0000** ± 0.0000 | 29.4 ± 1.85 | 256 of 256 |
| fast last layer at step 0.03 throughout, divisor held from the first arrival | intrinsic weight 4, learning rate 0.006 | 0.9194 ± 0.0034 | 0.9965 ± 0.0005 | 17.3 ± 0.903 | 256 of 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 4, learning rate 0.006 | 0.9400 ± 0.0031 | 0.9993 ± 0.0002 | 18.5 ± 1 | 256 of 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 4, learning rate 0.006 | 0.8424 ± 0.0063 | 0.9747 ± 0.0037 | 15.7 ± 0.84 | 256 of 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 4, learning rate 0.01 | 0.7759 ± 0.0077 | 0.9490 ± 0.0060 | 14.7 ± 0.957 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.006 | **0.9445** ± 0.0028 | **1.0000** ± 0.0000 | 16 ± 1.08 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 4, learning rate 0.006 | 0.9416 ± 0.0026 | <u>0.9999</u> ± 0.0000 | 14.7 ± 0.807 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, left running | intrinsic weight 2, learning rate 0.006 | 0.9385 ± 0.0028 | 0.9959 ± 0.0006 | 15.8 ± 0.993 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 2, learning rate 0.003 | 0.9348 ± 0.0030 | <u>0.9999</u> ± 0.0000 | 17.9 ± 1 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 0.5, learning rate 0.006 | 0.9284 ± 0.0040 | **1.0000** ± 0.0000 | 33.9 ± 2.64 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 0.25, learning rate 0.01 | 0.9195 ± 0.0042 | **1.0000** ± 0.0000 | 36.2 ± 2.53 | 256 of 256 |
| slower predictor with divisor held after one sixty-fourth of the run | intrinsic weight 1, learning rate 0.01 | 0.9362 ± 0.0032 | **1.0000** ± 0.0000 | 19.8 ± 1.09 | 256 of 256 |
| slower predictor with divisor held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.006 | 0.9346 ± 0.0030 | **1.0000** ± 0.0000 | 17.8 ± 1.05 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.006 | <u>0.9438</u> ± 0.0030 | **1.0000** ± 0.0000 | 22.6 ± 1.75 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 2, learning rate 0.006 | 0.9399 ± 0.0028 | **1.0000** ± 0.0000 | 17.4 ± 1.07 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.01 | 0.9061 ± 0.0051 | <u>0.9999</u> ± 0.0000 | 25.4 ± 1.49 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 0.25, learning rate 0.01 | 0.9061 ± 0.0054 | <u>0.9999</u> ± 0.0000 | 36.3 ± 2.46 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.003 | 0.8928 ± 0.0059 | 0.9997 ± 0.0001 | 24 ± 1.78 | 256 of 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.006 | 0.9256 ± 0.0029 | 0.9962 ± 0.0003 | 31.9 ± 2.06 | 256 of 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 0.25, learning rate 0.006 | 0.9177 ± 0.0032 | 0.9994 ± 0.0001 | 47 ± 2.3 | 256 of 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 0.125, learning rate 0.006 | 0.8794 ± 0.0053 | 0.9998 ± 0.0001 | 72.6 ± 3.84 | 256 of 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.006 | 0.8766 ± 0.0031 | 0.9801 ± 0.0009 | 23.8 ± 1.2 | 256 of 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 | 0.7321 ± 0.0034 | 0.9197 ± 0.0022 | 19.8 ± 0.945 | 256 of 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 | 0.5416 ± 0.0030 | 0.7647 ± 0.0050 | 18.2 ± 0.836 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | intrinsic weight 4, learning rate 0.006 | 0.7001 ± 0.0055 | 0.9414 ± 0.0033 | <u>14.6</u> ± 0.737 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | intrinsic weight 4, learning rate 0.01 | 0.5548 ± 0.0067 | 0.8353 ± 0.0069 | **14** ± 0.777 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores left running (batch 11) | intrinsic weight 2, learning rate 0.006 | 0.6403 ± 0.0077 | 0.8320 ± 0.0089 | 17.7 ± 1.08 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores left running (batch 11) | intrinsic weight 4, learning rate 0.006 | 0.4487 ± 0.0065 | 0.6323 ± 0.0099 | 17.4 ± 1.07 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held after one sixteenth of the run (batch 11) | intrinsic weight 4, learning rate 0.006 | 0.6610 ± 0.0071 | 0.9205 ± 0.0045 | 16.8 ± 1.41 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006 (best of 3) | 0.9349 ± 0.0031 | control | 0.9950 ± 0.0010 | control | 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.006 (best of 2) | **0.9445** ± 0.0028 | **+0.0095** [+0.0023, +0.0169] | **1.0000** ± 0.0000 | **+0.0050** [+0.0034, +0.0071] | 256 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.006 (best of 2) | <u>0.9438</u> ± 0.0030 | <u>+0.0089</u> [+0.0013, +0.0164] | **1.0000** ± 0.0000 | **+0.0050** [+0.0034, +0.0071] | 256 |
| divisor held from each policy's own first arrival | intrinsic weight 4, learning rate 0.006 (best of 3) | 0.9435 ± 0.0027 | +0.0085 [+0.0010, +0.0160] | <u>0.9999</u> ± 0.0000 | <u>+0.0049</u> [+0.0033, +0.0070] | 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 4, learning rate 0.006 (the only configuration read) | 0.9400 ± 0.0031 | +0.0051 [-0.0028, +0.0127] | 0.9993 ± 0.0002 | +0.0043 [+0.0026, +0.0064] | 256 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.01 (best of 2) | 0.9388 ± 0.0026 | +0.0039 [-0.0034, +0.0113] | <u>0.9999</u> ± 0.0001 | <u>+0.0049</u> [+0.0032, +0.0070] | 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, left running | intrinsic weight 2, learning rate 0.006 (the only configuration read) | 0.9385 ± 0.0028 | +0.0036 [-0.0041, +0.0114] | 0.9959 ± 0.0006 | +0.0009 [-0.0012, +0.0033] | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 (best of 2) | 0.9382 ± 0.0027 | +0.0033 [-0.0042, +0.0109] | 0.9995 ± 0.0003 | +0.0045 [+0.0027, +0.0066] | 256 |
| slower predictor with divisor held after one sixty-fourth of the run | intrinsic weight 1, learning rate 0.01 (best of 2) | 0.9362 ± 0.0032 | +0.0013 [-0.0070, +0.0097] | **1.0000** ± 0.0000 | **+0.0050** [+0.0034, +0.0071] | 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 2, learning rate 0.003 (best of 3) | 0.9348 ± 0.0030 | -0.0001 [-0.0084, +0.0083] | <u>0.9999</u> ± 0.0000 | <u>+0.0049</u> [+0.0033, +0.0070] | 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.006 (best of 6) | 0.9256 ± 0.0029 | -0.0093 [-0.0168, -0.0017] | 0.9962 ± 0.0003 | +0.0012 [-0.0005, +0.0034] | 256 |
| fast last layer at step 0.03 throughout, divisor held from the first arrival | intrinsic weight 4, learning rate 0.006 (the only configuration read) | 0.9194 ± 0.0034 | -0.0155 [-0.0243, -0.0068] | 0.9965 ± 0.0005 | +0.0015 [-0.0004, +0.0037] | 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.01 (best of 3) | 0.9061 ± 0.0051 | -0.0288 [-0.0401, -0.0180] | <u>0.9999</u> ± 0.0000 | <u>+0.0049</u> [+0.0033, +0.0070] | 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 4, learning rate 0.006 (best of 2) | 0.8424 ± 0.0063 | -0.0925 [-0.1068, -0.0785] | 0.9747 ± 0.0037 | -0.0203 [-0.0287, -0.0135] | 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | intrinsic weight 4, learning rate 0.006 (best of 2) | 0.7001 ± 0.0055 | -0.2348 [-0.2463, -0.2229] | 0.9414 ± 0.0033 | -0.0536 [-0.0605, -0.0469] | 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held after one sixteenth of the run (batch 11) | intrinsic weight 4, learning rate 0.006 (the only configuration read) | 0.6610 ± 0.0071 | -0.2740 [-0.2888, -0.2596] | 0.9205 ± 0.0045 | -0.0745 [-0.0836, -0.0658] | 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores left running (batch 11) | intrinsic weight 2, learning rate 0.006 (best of 2) | 0.6403 ± 0.0077 | -0.2946 [-0.3109, -0.2783] | 0.8320 ± 0.0089 | -0.1630 [-0.1809, -0.1462] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.006 (best of 3) | 19.9 ± 1.16 | control | 256 of 256 | 0.9194 | 0.9794 | 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | intrinsic weight 4, learning rate 0.01 (best of 2) | **14** ± 0.777 | **-5.91** [-8.46, -3.5] | 256 of 256 | 0.5548 | 0.8353 | 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 4, learning rate 0.01 (best of 2) | <u>14.7</u> ± 0.957 | <u>-5.24</u> [-7.97, -2.55] | 256 of 256 | 0.7759 | 0.9490 | 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 4, learning rate 0.006 (best of 2) | <u>14.7</u> ± 0.807 | -5.16 [-7.59, -2.72] | 256 of 256 | <u>0.9416</u> | <u>0.9999</u> | 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, left running | intrinsic weight 2, learning rate 0.006 (the only configuration read) | 15.8 ± 0.993 | -4.15 [-6.79, -1.49] | 256 of 256 | 0.9385 | 0.9959 | 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held after one sixteenth of the run (batch 11) | intrinsic weight 4, learning rate 0.006 (the only configuration read) | 16.8 ± 1.41 | -3.08 [-6.4, +0.361] | 256 of 256 | 0.6610 | 0.9205 | 256 |
| fast last layer at step 0.03 throughout, divisor held from the first arrival | intrinsic weight 4, learning rate 0.006 (the only configuration read) | 17.3 ± 0.903 | -2.61 [-5.19, +0.0209] | 256 of 256 | 0.9194 | 0.9965 | 256 |
| predictor at half speed with held divisor | intrinsic weight 2, learning rate 0.006 (best of 2) | 17.4 ± 1.07 | -2.54 [-5.53, +0.434] | 256 of 256 | 0.9399 | **1.0000** | 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores left running (batch 11) | intrinsic weight 4, learning rate 0.006 (best of 2) | 17.4 ± 1.07 | -2.54 [-5.32, +0.24] | 256 of 256 | 0.4487 | 0.6323 | 256 |
| slower predictor with divisor held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.006 (best of 2) | 17.8 ± 1.05 | -2.1 [-5.16, +1.01] | 256 of 256 | 0.9346 | **1.0000** | 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 2, learning rate 0.003 (best of 3) | 17.9 ± 1 | -1.98 [-5.03, +0.955] | 256 of 256 | 0.9348 | <u>0.9999</u> | 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 (best of 6) | 18.2 ± 0.836 | -1.72 [-4.47, +0.948] | 256 of 256 | 0.5416 | 0.7647 | 256 |
| divisor held from each policy's own first arrival | intrinsic weight 4, learning rate 0.006 (best of 3) | 18.5 ± 1.13 | -1.4 [-4.23, +1.44] | 256 of 256 | **0.9435** | <u>0.9999</u> | 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 4, learning rate 0.006 (the only configuration read) | 18.5 ± 1 | -1.4 [-4.1, +1.3] | 256 of 256 | 0.9400 | 0.9993 | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 (best of 2) | 21 ± 1.22 | +1.08 [-1.61, +3.82] | 256 of 256 | 0.9382 | 0.9995 | 256 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.01 (best of 2) | 22.4 ± 1.27 | +2.46 [-0.627, +5.49] | 256 of 256 | 0.9388 | <u>0.9999</u> | 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.003 (best of 3) | 24 ± 1.78 | +4.09 [+0.448, +7.95] | 256 of 256 | 0.8928 | 0.9997 | 256 |

#### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:53 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 4 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:44 PT | none |
| 5 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:45 PT | none |
| 6 | held divisor alone, held after one sixty-fourth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:46 PT | none |
| 7 | held divisor alone, held after one sixty-fourth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:48 PT | none |
| 8 | divisor held from each policy's own first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:32 PT | none |
| 9 | divisor held from each policy's own first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:32 PT | none |
| 10 | divisor held from each policy's own first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:34 PT | none |
| 11 | fast last layer at step 0.03 throughout, divisor held from the first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:15 PT | none |
| 12 | fast last layer at step 0.01 throughout, divisor held from the first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:15 PT | none |
| 13 | fast last layer at step 0.1 throughout, divisor held from the first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 03:06 PT | none |
| 14 | fast last layer at step 0.1 throughout, divisor held from the first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 03:08 PT | none |
| 15 | fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:52 PT | none |
| 16 | fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:48 PT | none |
| 17 | fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, left running | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:55 PT | none |
| 18 | slower predictor with held divisor (divisor held after one sixteenth of the run) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:56 PT | none |
| 19 | slower predictor with held divisor (divisor held after one sixteenth of the run) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:52 PT | none |
| 20 | slower predictor with held divisor (divisor held after one sixteenth of the run) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:56 PT | none |
| 21 | slower predictor with divisor held after one sixty-fourth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:49 PT | none |
| 22 | slower predictor with divisor held after one sixty-fourth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:48 PT | none |
| 23 | predictor at half speed with held divisor | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:52 PT | none |
| 24 | predictor at half speed with held divisor | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 21:51 PT | none |
| 25 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 20:03 PT | none |
| 26 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 20:02 PT | none |
| 27 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 20:02 PT | none |
| 28 | error length as the bonus, with the divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:55 PT | none |
| 29 | error length as the bonus, with the divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:55 PT | none |
| 30 | error length as the bonus, with the divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:56 PT | none |
| 31 | error length as the bonus, with the divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:35 PT | none |
| 32 | error length as the bonus, with the divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:34 PT | none |
| 33 | error length as the bonus, with the divisor held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:35 PT | none |
| 34 | fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:05 PT | none |
| 35 | fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:04 PT | none |
| 36 | fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores left running (batch 11) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:18 PT | none |
| 37 | fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores left running (batch 11) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:18 PT | none |
| 38 | fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held after one sixteenth of the run (batch 11) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:21 PT | none |

Unit names, by the number in the first column:

1. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `validation-block_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-after-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-one-sixty-fourth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-one-sixty-fourth_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-medium_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-step-0-point-03-with-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-step-0-point-01-with-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
14. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
15. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-then-predictor-divisor-held_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
16. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-then-predictor-divisor-held_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
17. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-then-predictor-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
18. `validation-block_pointmaze-medium_reward-arrival-plus-one_slower-predictor-with-held-divisor_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
19. `validation-block_pointmaze-medium_reward-arrival-plus-one_slower-predictor-with-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`
20. `validation-block_pointmaze-medium_reward-arrival-plus-one_slower-predictor-with-held-divisor_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.25_clip-coefficient-0.1_seeds-10000-to-10255`
21. `validation-block_pointmaze-medium_reward-arrival-plus-one_slow-predictor-held-sixty-fourth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
22. `validation-block_pointmaze-medium_reward-arrival-plus-one_slow-predictor-held-sixty-fourth_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
23. `validation-block_pointmaze-medium_reward-arrival-plus-one_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
24. `validation-block_pointmaze-medium_reward-arrival-plus-one_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
25. `validation-block_pointmaze-medium_reward-arrival-plus-one_three-parts-held-one-sixteenth_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
26. `validation-block_pointmaze-medium_reward-arrival-plus-one_three-parts-held-one-sixteenth_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.25_clip-coefficient-0.1_seeds-10000-to-10255`
27. `validation-block_pointmaze-medium_reward-arrival-plus-one_three-parts-held-one-sixteenth_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`
28. `validation-block_pointmaze-medium_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.125_clip-coefficient-0.1_seeds-10000-to-10255`
29. `validation-block_pointmaze-medium_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.25_clip-coefficient-0.1_seeds-10000-to-10255`
30. `validation-block_pointmaze-medium_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`
31. `validation-block_pointmaze-medium_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
32. `validation-block_pointmaze-medium_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
33. `validation-block_pointmaze-medium_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
34. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-until-arrival-then-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
35. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-until-arrival-then-hold_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
36. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-until-arrival-no-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
37. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-until-arrival-no-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
38. `validation-block_pointmaze-medium_reward-arrival-plus-one_fast-until-arrival-fixed-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

### Second block: seeds that chose nothing

The units of the phase `validation-block-two` in PointMaze Medium, reward 1 on arrival. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.006 | 0.9308 ± 0.0030 | 0.9907 ± 0.0039 | 26.3 ± 1.88 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.01 | 0.9276 ± 0.0026 | 0.9798 ± 0.0059 | 23.6 ± 1.4 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.006 | 0.9141 ± 0.0030 | 0.9821 ± 0.0018 | 19.6 ± 1.14 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 | 0.9398 ± 0.0025 | 0.9995 ± 0.0002 | 19.7 ± 1.13 | 256 of 256 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.01 | **0.9417** ± 0.0026 | **1.0000** ± 0.0000 | 22.9 ± 1.46 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 4, learning rate 0.006 | 0.9404 ± 0.0030 | <u>0.9999</u> ± 0.0000 | 20.6 ± 1.12 | 256 of 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 4, learning rate 0.006 | 0.9403 ± 0.0025 | 0.9995 ± 0.0001 | <u>17.8</u> ± 0.934 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 2, learning rate 0.006 | <u>0.9416</u> ± 0.0027 | **1.0000** ± 0.0000 | **17.2** ± 0.962 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.006 | 0.9410 ± 0.0035 | <u>0.9999</u> ± 0.0001 | 22.8 ± 1.68 | 256 of 256 |

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
| divisor held from each policy's own first arrival | intrinsic weight 4, learning rate 0.006 (its best on the first block) | 0.9404 ± 0.0030 | +0.0095 [+0.0015, +0.0178] | +0.0095 [+0.0015, +0.0178] | <u>0.9999</u> ± 0.0000 | <u>+0.0092</u> [+0.0043, +0.0177] | 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 4, learning rate 0.006 (its best on the first block) | 0.9403 ± 0.0025 | +0.0095 [+0.0019, +0.0171] | +0.0095 [+0.0019, +0.0171] | 0.9995 ± 0.0001 | +0.0088 [+0.0039, +0.0173] | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 (its best on the first block) | 0.9398 ± 0.0025 | +0.0090 [+0.0023, +0.0159] | +0.0090 [+0.0023, +0.0159] | 0.9995 ± 0.0002 | +0.0088 [+0.0038, +0.0173] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.006 (its best on the first block and on this block) | 19.6 ± 1.14 | control | control | 256 of 256 | 256 |
| predictor at half speed with held divisor | intrinsic weight 2, learning rate 0.006 (its best on the first block) | **17.2** ± 0.962 | **-2.45** [-5.23, +0.244] | **-2.45** [-5.23, +0.244] | 256 of 256 | 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 4, learning rate 0.006 (its best on the first block) | <u>17.8</u> ± 0.934 | <u>-1.84</u> [-4.58, +0.809] | <u>-1.84</u> [-4.58, +0.809] | 256 of 256 | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 (its best on the first block) | 19.7 ± 1.13 | +0.0469 [-2.73, +2.71] | +0.0469 [-2.73, +2.71] | 256 of 256 | 256 |
| divisor held from each policy's own first arrival | intrinsic weight 4, learning rate 0.006 (its best on the first block) | 20.6 ± 1.12 | +0.99 [-1.78, +3.78] | +0.99 [-1.78, +3.78] | 256 of 256 | 256 |
| held divisor alone, held after one sixty-fourth of the run | intrinsic weight 2, learning rate 0.01 (its best on the first block) | 22.9 ± 1.46 | +3.28 [+0.108, +6.56] | +3.28 [+0.108, +6.56] | 256 of 256 | 256 |

#### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:16 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:19 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:16 PT | none |
| 4 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:22 PT | none |
| 5 | held divisor alone, held after one sixty-fourth of the run | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:23 PT | none |
| 6 | divisor held from each policy's own first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:31 PT | none |
| 7 | fast last layer at step 0.01 throughout, divisor held from the first arrival | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:33 PT | none |
| 8 | predictor at half speed with held divisor | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:19 PT | none |
| 9 | predictor at half speed with held divisor | PointMaze Medium, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:21 PT | none |

Unit names, by the number in the first column:

1. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_divisor-held-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
5. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_divisor-held-one-sixty-fourth_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
6. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
7. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_fast-step-0-point-01-with-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
8. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_half-speed-predictor-held_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-20000-to-20255`
9. `validation-block-two_pointmaze-medium_reward-arrival-plus-one_half-speed-predictor-held_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`

### Discovery check: the leading configurations for the first-goal time

The units of the phase `discovery-validation` in PointMaze Medium, reward 1 on arrival. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which these configurations were picked as the leading ones of each method's discovery sweep, nor the confirmation seeds. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $M$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 | 0.0073 ± 0.0003 | 0.0000 ± 0.0000 | 17.5 ± 1.41 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | **0.8718** ± 0.0034 | **0.9537** ± 0.0028 | 17.6 ± 0.96 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.01 | <u>0.6440</u> ± 0.0073 | <u>0.7891</u> ± 0.0078 | 18.3 ± 0.963 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.03 | 0.1132 ± 0.0017 | 0.0368 ± 0.0029 | **12.5** ± 0.565 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.02, entropy coefficient 0.01 | 0.1440 ± 0.0029 | 0.1639 ± 0.0056 | <u>13.1</u> ± 0.859 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.02, entropy coefficient 0.03 | 0.0699 ± 0.0012 | 0.0070 ± 0.0006 | 13.5 ± 0.778 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.01 | 0.1550 ± 0.0030 | 0.1846 ± 0.0068 | 13.6 ± 0.825 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | 0.3785 ± 0.0072 | 0.5111 ± 0.0121 | 13.8 ± 0.763 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 | 0.2504 ± 0.0046 | 0.0739 ± 0.0078 | 13.8 ± 0.655 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.03 | 0.1141 ± 0.0025 | 0.0071 ± 0.0008 | 14.2 ± 0.758 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.01 | 0.2536 ± 0.0067 | 0.2941 ± 0.0104 | 14.8 ± 1.11 | 256 of 256 |

Then, for $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.1 (best of 3) | <u>17.5</u> ± 1.41 | control | 256 of 256 | <u>0.0073</u> | <u>0.0000</u> | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.03 (best of 8) | **12.5** ± 0.565 | -4.99 [-8.32, -2.37] | 256 of 256 | **0.1132** | **0.0368** | 256 |

#### Units

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

## PointMaze Medium, step cost

### First block: each side at its best configuration there

The units of the phase `validation-block` in PointMaze Medium, step cost. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006 | -0.3505 ± 0.0016 | -0.3078 ± 0.0007 | 20 ± 1.17 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.0078125, learning rate 0.006 | -0.3514 ± 0.0018 | -0.3074 ± 0.0011 | 20.9 ± 1.27 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.01 | -0.3549 ± 0.0018 | -0.3078 ± 0.0011 | 20.1 ± 1 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 | **-0.3463** ± 0.0012 | -0.3066 ± 0.0005 | 13.9 ± 0.528 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.006 | -0.3495 ± 0.0018 | -0.3076 ± 0.0011 | 16.5 ± 0.757 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 0.03125, learning rate 0.006 | -0.3499 ± 0.0016 | <u>-0.3061</u> ± 0.0005 | 20.1 ± 0.96 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 0.125, learning rate 0.006 | -0.3518 ± 0.0020 | -0.3074 ± 0.0012 | 21.3 ± 1.14 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 0.00195312, learning rate 0.006 | -0.3518 ± 0.0021 | -0.3089 ± 0.0015 | 21.5 ± 1.01 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 | -0.3520 ± 0.0014 | -0.3070 ± 0.0005 | 14.6 ± 0.653 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 0.03125, learning rate 0.01 | -0.3522 ± 0.0014 | -0.3066 ± 0.0004 | 19.7 ± 0.991 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 0.0078125, learning rate 0.006 | -0.3534 ± 0.0020 | -0.3086 ± 0.0009 | 20.9 ± 1.02 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 0.125, learning rate 0.01 | -0.3537 ± 0.0015 | -0.3072 ± 0.0005 | 19.1 ± 0.864 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.006 | -0.3667 ± 0.0018 | -0.3095 ± 0.0010 | 15.7 ± 0.771 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 16, learning rate 0.006 | -0.3935 ± 0.0025 | -0.3142 ± 0.0006 | 17.6 ± 1.05 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 1, learning rate 0.006 | -0.3476 ± 0.0014 | -0.3073 ± 0.0007 | 15.4 ± 0.693 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 2, learning rate 0.006 | -0.3484 ± 0.0014 | **-0.3058** ± 0.0004 | 14.2 ± 0.617 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 4, learning rate 0.006 | -0.3504 ± 0.0015 | -0.3066 ± 0.0005 | 13.4 ± 0.562 | 256 of 256 |
| fast last layer at step 0.03 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.006 | -0.3491 ± 0.0014 | -0.3067 ± 0.0003 | 14 ± 0.628 | 256 of 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.006 | <u>-0.3470</u> ± 0.0013 | -0.3066 ± 0.0006 | 13.8 ± 0.588 | 256 of 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.006 | -0.3692 ± 0.0030 | -0.3132 ± 0.0008 | <u>12.9</u> ± 0.654 | 256 of 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 4, learning rate 0.01 | -0.4057 ± 0.0034 | -0.3299 ± 0.0015 | 15 ± 1.74 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.006 | -0.3476 ± 0.0013 | -0.3068 ± 0.0006 | **12.7** ± 0.612 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 1, learning rate 0.006 | -0.3489 ± 0.0022 | -0.3070 ± 0.0006 | 13.1 ± 0.66 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.006 | -0.3491 ± 0.0018 | -0.3076 ± 0.0005 | 13.7 ± 0.573 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 0.0078125, learning rate 0.006 | -0.3539 ± 0.0020 | -0.3079 ± 0.0011 | 22 ± 1.17 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 0.00195312, learning rate 0.006 | -0.3544 ± 0.0020 | -0.3071 ± 0.0007 | 24.3 ± 1.84 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 2, learning rate 0.006 | -0.3547 ± 0.0021 | -0.3089 ± 0.0012 | 13.4 ± 0.627 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 4, learning rate 0.006 | -0.3637 ± 0.0025 | -0.3096 ± 0.0010 | 13 ± 0.585 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | intrinsic weight 2, learning rate 0.006 | -0.3821 ± 0.0018 | -0.3153 ± 0.0006 | **12.7** ± 0.593 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | intrinsic weight 4, learning rate 0.006 | -0.4307 ± 0.0026 | -0.3296 ± 0.0013 | <u>12.9</u> ± 0.684 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006 (best of 3) | -0.3505 ± 0.0016 | control | -0.3078 ± 0.0007 | control | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 (best of 11) | **-0.3463** ± 0.0012 | **+0.0042** [+0.0005, +0.0080] | **-0.3066** ± 0.0005 | **+0.0013** [-0.0004, +0.0030] | 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.006 (the only configuration read) | <u>-0.3470</u> ± 0.0013 | <u>+0.0035</u> [-0.0008, +0.0077] | **-0.3066** ± 0.0006 | <u>+0.0012</u> [-0.0005, +0.0029] | 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.006 (best of 2) | -0.3476 ± 0.0013 | +0.0029 [-0.0013, +0.0072] | -0.3068 ± 0.0006 | +0.0010 [-0.0008, +0.0028] | 256 |
| divisor held from each policy's own first arrival | intrinsic weight 1, learning rate 0.006 (best of 3) | -0.3476 ± 0.0014 | +0.0029 [-0.0014, +0.0071] | -0.3073 ± 0.0007 | +0.0005 [-0.0014, +0.0024] | 256 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.006 (best of 5) | -0.3491 ± 0.0018 | +0.0014 [-0.0033, +0.0059] | -0.3076 ± 0.0005 | +0.0003 [-0.0015, +0.0020] | 256 |
| fast last layer at step 0.03 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.006 (the only configuration read) | -0.3491 ± 0.0014 | +0.0014 [-0.0028, +0.0055] | <u>-0.3067</u> ± 0.0003 | +0.0011 [-0.0003, +0.0027] | 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.006 (best of 2) | -0.3692 ± 0.0030 | -0.0187 [-0.0257, -0.0120] | -0.3132 ± 0.0008 | -0.0054 [-0.0076, -0.0034] | 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | intrinsic weight 2, learning rate 0.006 (best of 2) | -0.3821 ± 0.0018 | -0.0316 [-0.0361, -0.0270] | -0.3153 ± 0.0006 | -0.0074 [-0.0092, -0.0056] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006 (best of 3) | 20 ± 1.17 | control | 256 of 256 | -0.3505 | -0.3078 | 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.006 (best of 2) | **12.7** ± 0.612 | **-7.32** [-9.92, -5.08] | 256 of 256 | -0.3476 | -0.3068 | 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | intrinsic weight 2, learning rate 0.006 (best of 2) | **12.7** ± 0.593 | **-7.32** [-9.86, -5.2] | 256 of 256 | -0.3821 | -0.3153 | 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.006 (best of 2) | <u>12.9</u> ± 0.654 | <u>-7.06</u> [-9.78, -4.74] | 256 of 256 | -0.3692 | -0.3132 | 256 |
| predictor at half speed with held divisor | intrinsic weight 4, learning rate 0.006 (best of 5) | 13 ± 0.585 | -6.97 [-9.67, -4.65] | 256 of 256 | -0.3637 | -0.3096 | 256 |
| divisor held from each policy's own first arrival | intrinsic weight 4, learning rate 0.006 (best of 3) | 13.4 ± 0.562 | -6.63 [-9.28, -4.43] | 256 of 256 | -0.3504 | **-0.3066** | 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.006 (the only configuration read) | 13.8 ± 0.588 | -6.19 [-8.79, -4.06] | 256 of 256 | <u>-0.3470</u> | **-0.3066** | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 (best of 11) | 13.9 ± 0.528 | -6.13 [-8.63, -4.1] | 256 of 256 | **-0.3463** | **-0.3066** | 256 |
| fast last layer at step 0.03 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.006 (the only configuration read) | 14 ± 0.628 | -5.94 [-8.68, -3.65] | 256 of 256 | -0.3491 | <u>-0.3067</u> | 256 |

#### Units

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
| 15 | divisor held from each policy's own first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:35 PT | none |
| 16 | divisor held from each policy's own first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:35 PT | none |
| 17 | divisor held from each policy's own first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 01:34 PT | none |
| 18 | fast last layer at step 0.03 throughout, divisor held from the first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:15 PT | none |
| 19 | fast last layer at step 0.01 throughout, divisor held from the first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:19 PT | none |
| 20 | fast last layer at step 0.1 throughout, divisor held from the first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:11 PT | none |
| 21 | fast last layer at step 0.1 throughout, divisor held from the first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:13 PT | none |
| 22 | fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:52 PT | none |
| 23 | fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:52 PT | none |
| 24 | predictor at half speed with held divisor | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:02 PT | none |
| 25 | predictor at half speed with held divisor | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:02 PT | none |
| 26 | predictor at half speed with held divisor | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 00:13 PT | none |
| 27 | predictor at half speed with held divisor | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 00:12 PT | none |
| 28 | predictor at half speed with held divisor | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 00:12 PT | none |
| 29 | fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 04:02 PT | none |
| 30 | fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 04:02 PT | none |

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
15. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
16. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
17. `validation-block_pointmaze-medium_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
18. `validation-block_pointmaze-medium_reward-step-cost_fast-step-0-point-03-with-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
19. `validation-block_pointmaze-medium_reward-step-cost_fast-step-0-point-01-with-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
20. `validation-block_pointmaze-medium_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
21. `validation-block_pointmaze-medium_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
22. `validation-block_pointmaze-medium_reward-step-cost_fast-then-predictor-divisor-held_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
23. `validation-block_pointmaze-medium_reward-step-cost_fast-then-predictor-divisor-held_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
24. `validation-block_pointmaze-medium_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-10000-to-10255`
25. `validation-block_pointmaze-medium_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.0078125_clip-coefficient-0.1_seeds-10000-to-10255`
26. `validation-block_pointmaze-medium_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
27. `validation-block_pointmaze-medium_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
28. `validation-block_pointmaze-medium_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
29. `validation-block_pointmaze-medium_reward-step-cost_fast-until-arrival-then-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
30. `validation-block_pointmaze-medium_reward-step-cost_fast-until-arrival-then-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

### Second block: seeds that chose nothing

The units of the phase `validation-block-two` in PointMaze Medium, step cost. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.0078125, learning rate 0.006 | -0.3510 ± 0.0016 | -0.3078 ± 0.0007 | 20.4 ± 0.931 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006 | -0.3535 ± 0.0024 | -0.3082 ± 0.0012 | 19.9 ± 1.01 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.01 | -0.3541 ± 0.0014 | -0.3073 ± 0.0006 | 20.6 ± 0.841 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 | -0.3477 ± 0.0012 | <u>-0.3062</u> ± 0.0004 | 15 ± 0.68 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.006 | -0.3514 ± 0.0014 | -0.3069 ± 0.0004 | <u>13.8</u> ± 0.593 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.006 | -0.3680 ± 0.0021 | -0.3089 ± 0.0003 | 15.9 ± 0.715 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 1, learning rate 0.006 | **-0.3470** ± 0.0015 | -0.3069 ± 0.0006 | 16.9 ± 0.742 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 2, learning rate 0.006 | <u>-0.3472</u> ± 0.0015 | **-0.3059** ± 0.0005 | 14.7 ± 0.632 | 256 of 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.006 | -0.3494 ± 0.0018 | -0.3068 ± 0.0005 | **13.5** ± 0.628 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND), at its first-block best | intrinsic weight 0.00195312, learning rate 0.006 (its best on the first block) | -0.3535 ± 0.0024 | control | -0.0025 [-0.0080, +0.0025] | -0.3082 ± 0.0012 | control | 256 |
| control (CleanRL PPO-RND), at its best on this block | intrinsic weight 0.0078125, learning rate 0.006 (best of 3 on this block) | -0.3510 ± 0.0016 | +0.0025 [-0.0026, +0.0080] | control | -0.3078 ± 0.0007 | +0.0004 [-0.0023, +0.0035] | 256 |
| divisor held from each policy's own first arrival | intrinsic weight 1, learning rate 0.006 (its best on the first block) | **-0.3470** ± 0.0015 | **+0.0065** [+0.0012, +0.0122] | **+0.0040** [-0.0001, +0.0080] | -0.3069 ± 0.0006 | +0.0013 [-0.0011, +0.0041] | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 (its best on the first block) | <u>-0.3477</u> ± 0.0012 | <u>+0.0058</u> [+0.0007, +0.0115] | <u>+0.0033</u> [-0.0004, +0.0069] | **-0.3062** ± 0.0004 | **+0.0020** [-0.0002, +0.0047] | 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.006 (its best on the first block) | -0.3494 ± 0.0018 | +0.0041 [-0.0017, +0.0103] | +0.0016 [-0.0031, +0.0062] | <u>-0.3068</u> ± 0.0005 | <u>+0.0014</u> [-0.0009, +0.0042] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 0.00195312, learning rate 0.006 (its best on the first block and on this block) | 19.9 ± 1.01 | control | control | 256 of 256 | 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.006 (its best on the first block) | **13.5** ± 0.628 | **-6.43** [-8.63, -4.26] | **-6.43** [-8.63, -4.26] | 256 of 256 | 256 |
| divisor held from each policy's own first arrival | intrinsic weight 2, learning rate 0.006 (its best on this block: its first-block best, intrinsic weight 4, learning rate 0.006, was not run here) | <u>14.7</u> ± 0.632 | <u>-5.26</u> [-7.49, -3.17] | <u>-5.26</u> [-7.49, -3.17] | 256 of 256 | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.006 (its best on the first block) | 15 ± 0.68 | -4.9 [-7.17, -2.73] | -4.9 [-7.17, -2.73] | 256 of 256 | 256 |

#### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:36 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:38 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:38 PT | none |
| 4 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:39 PT | none |
| 5 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:40 PT | none |
| 6 | held divisor alone, held after one sixteenth of the run | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:40 PT | none |
| 7 | divisor held from each policy's own first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 02:31 PT | none |
| 8 | divisor held from each policy's own first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 02:31 PT | none |
| 9 | fast last layer at step 0.01 throughout, divisor held from the first arrival | PointMaze Medium, step cost | 1 | complete | 256 of 256 | 2026-11-13 06:34 PT | none |

Unit names, by the number in the first column:

1. `validation-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.0078125_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-medium_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-medium_reward-step-cost_divisor-held-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
5. `validation-block-two_pointmaze-medium_reward-step-cost_divisor-held-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
6. `validation-block-two_pointmaze-medium_reward-step-cost_divisor-held-one-sixteenth_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`
7. `validation-block-two_pointmaze-medium_reward-step-cost_held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-20000-to-20255`
8. `validation-block-two_pointmaze-medium_reward-step-cost_held-from-first-arrival_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
9. `validation-block-two_pointmaze-medium_reward-step-cost_fast-step-0-point-01-with-hold_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`

### Discovery check: the leading configurations for the first-goal time

The units of the phase `discovery-validation` in PointMaze Medium, step cost. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which these configurations were picked as the leading ones of each method's discovery sweep, nor the confirmation seeds. Read at 1,048,576 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $M$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 | -0.5526 ± 0.0109 | -0.6884 ± 0.0206 | 12.4 ± 0.471 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.01 | <u>-0.4220</u> ± 0.0028 | <u>-0.3519</u> ± 0.0025 | 13.9 ± 0.762 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | **-0.3680** ± 0.0015 | **-0.3244** ± 0.0010 | 14.5 ± 0.63 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.03 | -0.9430 ± 0.0011 | -0.9971 ± 0.0002 | **11.4** ± 0.506 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.01 | -0.5857 ± 0.0044 | -0.4946 ± 0.0060 | <u>11.7</u> ± 0.564 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 | -0.8423 ± 0.0023 | -0.9989 ± 0.0001 | 11.9 ± 0.532 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.02, entropy coefficient 0.03 | -0.9723 ± 0.0008 | -0.9991 ± 0.0001 | 12.4 ± 0.701 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.01 | -0.6902 ± 0.0064 | -0.5906 ± 0.0091 | 12.5 ± 0.682 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.02, entropy coefficient 0.03 | -0.9095 ± 0.0021 | -0.9992 ± 0.0001 | 13.1 ± 0.659 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.01 | -0.8694 ± 0.0032 | -0.8254 ± 0.0062 | 13.3 ± 0.883 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.02, entropy coefficient 0.01 | -0.9183 ± 0.0043 | -0.9059 ± 0.0064 | 14 ± 1.05 | 256 of 256 |

Then, for $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 (best of 3) | <u>12.4</u> ± 0.471 | control | 256 of 256 | **-0.5526** | **-0.6884** | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.01, entropy coefficient 0.03 (best of 8) | **11.4** ± 0.506 | -1.04 [-2.35, +0.241] | 256 of 256 | <u>-0.9430</u> | <u>-0.9971</u> | 256 |

#### Units

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

## PointMaze Large, reward 1 on arrival

### First block: each side at its best configuration there

The units of the phase `validation-block` in PointMaze Large, reward 1 on arrival. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 8,388,608 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 | 0.8781 ± 0.0021 | 0.9920 ± 0.0005 | 269 ± 10.8 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 1, learning rate 0.002 | 0.8759 ± 0.0022 | 0.9955 ± 0.0004 | 385 ± 13.1 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.003 | 0.8701 ± 0.0023 | 0.9906 ± 0.0006 | 203 ± 8.06 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.001 | 0.8695 ± 0.0029 | 0.9911 ± 0.0007 | 344 ± 17.8 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 | 0.8626 ± 0.0025 | 0.9765 ± 0.0019 | 205 ± 8.2 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002 | 0.8876 ± 0.0021 | 0.9976 ± 0.0003 | 270 ± 10.5 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002 | 0.8821 ± 0.0020 | 0.9970 ± 0.0002 | 213 ± 8.15 | 256 of 256 |
| held divisor alone, held after one quarter of the run | intrinsic weight 4, learning rate 0.002 | 0.8718 ± 0.0024 | 0.9946 ± 0.0005 | 205 ± 8.2 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 2, learning rate 0.002 | 0.8874 ± 0.0023 | 0.9977 ± 0.0002 | 259 ± 10.6 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 4, learning rate 0.002 | 0.8852 ± 0.0023 | 0.9976 ± 0.0003 | 201 ± 9.11 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 1, learning rate 0.002 | 0.8680 ± 0.0033 | 0.9979 ± 0.0003 | 375 ± 12.4 | 256 of 256 |
| fast last layer at step 0.03 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.002 | **0.8976** ± 0.0021 | 0.9967 ± 0.0003 | 128 ± 6.96 | 256 of 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.002 | <u>0.8928</u> ± 0.0025 | 0.9971 ± 0.0003 | 180 ± 8.38 | 256 of 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.002 | 0.8707 ± 0.0058 | 0.9872 ± 0.0055 | 90.7 ± 5.08 | 256 of 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 16, learning rate 0.002 | 0.5725 ± 0.0106 | 0.8495 ± 0.0123 | 86.1 ± 6.37 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 4, learning rate 0.002 | 0.8806 ± 0.0029 | 0.9979 ± 0.0002 | 96.2 ± 6.95 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.002 | 0.8728 ± 0.0055 | 0.9945 ± 0.0028 | 85.2 ± 5.18 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 1, learning rate 0.002 | 0.8284 ± 0.0118 | 0.9591 ± 0.0121 | 136 ± 14.9 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, left running | intrinsic weight 2, learning rate 0.002 | 0.8894 ± 0.0022 | 0.9931 ± 0.0005 | 91.7 ± 5.75 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 4, learning rate 0.002 | 0.8883 ± 0.0021 | **0.9982** ± 0.0002 | 153 ± 9.69 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 2, learning rate 0.002 | 0.8851 ± 0.0029 | <u>0.9981</u> ± 0.0003 | 182 ± 13.9 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 4, learning rate 0.002 | 0.8849 ± 0.0023 | 0.9978 ± 0.0003 | 161 ± 10.6 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 2, learning rate 0.002 | 0.8821 ± 0.0027 | <u>0.9981</u> ± 0.0002 | 217 ± 13.5 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 1, learning rate 0.002 | 0.8597 ± 0.0042 | 0.9978 ± 0.0004 | 360 ± 22.8 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.002 | 0.8917 ± 0.0026 | 0.9969 ± 0.0004 | 160 ± 11.2 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.002 | 0.8779 ± 0.0021 | 0.9948 ± 0.0004 | 124 ± 9.81 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 0.25, learning rate 0.002 | 0.8778 ± 0.0052 | 0.9974 ± 0.0004 | 284 ± 20.3 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run | intrinsic weight 0.5, learning rate 0.002 | 0.8859 ± 0.0025 | 0.9954 ± 0.0005 | 172 ± 11.1 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run | intrinsic weight 0.25, learning rate 0.002 | 0.8853 ± 0.0031 | 0.9969 ± 0.0005 | 243 ± 13.5 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run | intrinsic weight 1, learning rate 0.002 | 0.8712 ± 0.0021 | 0.9935 ± 0.0004 | 120 ± 7.9 | 256 of 256 |
| separate predictor step size, predictor learning rate 0.25 times 1e-4 | intrinsic weight 2, learning rate 0.002 | 0.8859 ± 0.0024 | 0.9954 ± 0.0007 | 171 ± 11 | 256 of 256 |
| separate predictor step size, predictor learning rate 0.25 times 1e-4 | intrinsic weight 4, learning rate 0.002 | 0.8783 ± 0.0022 | 0.9927 ± 0.0005 | 133 ± 7.88 | 256 of 256 |
| separate predictor step size, predictor learning rate 0.25 times 1e-4 | intrinsic weight 1, learning rate 0.002 | 0.8756 ± 0.0030 | 0.9977 ± 0.0002 | 263 ± 15.4 | 256 of 256 |
| separate predictor step size, predictor learning rate 1 times 1e-4 | intrinsic weight 2, learning rate 0.002 | 0.8804 ± 0.0024 | 0.9954 ± 0.0004 | 238 ± 12.4 | 256 of 256 |
| separate predictor step size, predictor learning rate 1 times 1e-4 | intrinsic weight 4, learning rate 0.002 | 0.8762 ± 0.0022 | 0.9931 ± 0.0003 | 185 ± 10.1 | 256 of 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 0.25, learning rate 0.002 | 0.8714 ± 0.0016 | 0.9943 ± 0.0003 | 259 ± 8.89 | 256 of 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 0.125, learning rate 0.002 | 0.8695 ± 0.0019 | 0.9965 ± 0.0003 | 399 ± 12 | 256 of 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.002 | 0.8278 ± 0.0015 | 0.9883 ± 0.0005 | 176 ± 6.38 | 256 of 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.002 | 0.6557 ± 0.0026 | 0.9162 ± 0.0020 | 125 ± 5 | 256 of 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002 | 0.3521 ± 0.0030 | 0.6369 ± 0.0049 | 103 ± 4.22 | 256 of 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002 | 0.1189 ± 0.0018 | 0.2435 ± 0.0045 | 93 ± 4.07 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | intrinsic weight 2, learning rate 0.002 | 0.8496 ± 0.0036 | 0.9919 ± 0.0011 | 106 ± 11.2 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | intrinsic weight 4, learning rate 0.002 | 0.7871 ± 0.0045 | 0.9770 ± 0.0021 | **82.9** ± 4.46 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | intrinsic weight 16, learning rate 0.002 | 0.3370 ± 0.0092 | 0.6110 ± 0.0137 | 89.1 ± 8.37 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores left running (batch 11) | intrinsic weight 2, learning rate 0.002 | 0.4258 ± 0.0056 | 0.6341 ± 0.0071 | 96.2 ± 8.72 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores left running (batch 11) | intrinsic weight 4, learning rate 0.002 | 0.1356 ± 0.0039 | 0.2376 ± 0.0071 | 91.2 ± 6.69 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores left running (batch 11) | intrinsic weight 16, learning rate 0.002 | 0.0230 ± 0.0004 | 0.0305 ± 0.0016 | <u>83</u> ± 6.26 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held after one sixteenth of the run (batch 11) | intrinsic weight 4, learning rate 0.002 | 0.5718 ± 0.0045 | 0.8776 ± 0.0045 | 83.3 ± 5.16 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 (best of 5) | 0.8781 ± 0.0021 | control | 0.9920 ± 0.0005 | control | 256 |
| fast last layer at step 0.03 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.002 (the only configuration read) | **0.8976** ± 0.0021 | **+0.0195** [+0.0138, +0.0254] | 0.9967 ± 0.0003 | +0.0047 [+0.0035, +0.0060] | 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.002 (the only configuration read) | <u>0.8928</u> ± 0.0025 | <u>+0.0148</u> [+0.0083, +0.0213] | 0.9971 ± 0.0003 | +0.0051 [+0.0040, +0.0061] | 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.002 (best of 3) | 0.8917 ± 0.0026 | +0.0137 [+0.0071, +0.0202] | 0.9969 ± 0.0004 | +0.0049 [+0.0036, +0.0063] | 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, left running | intrinsic weight 2, learning rate 0.002 (the only configuration read) | 0.8894 ± 0.0022 | +0.0114 [+0.0054, +0.0174] | 0.9931 ± 0.0005 | +0.0011 [-0.0003, +0.0024] | 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 4, learning rate 0.002 (best of 2) | 0.8883 ± 0.0021 | +0.0102 [+0.0048, +0.0159] | **0.9982** ± 0.0002 | **+0.0062** [+0.0052, +0.0073] | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002 (best of 2) | 0.8876 ± 0.0021 | +0.0095 [+0.0040, +0.0151] | 0.9976 ± 0.0003 | +0.0056 [+0.0045, +0.0067] | 256 |
| divisor held from each policy's own first arrival | intrinsic weight 2, learning rate 0.002 (best of 3) | 0.8874 ± 0.0023 | +0.0094 [+0.0031, +0.0156] | 0.9977 ± 0.0002 | +0.0057 [+0.0046, +0.0068] | 256 |
| separate predictor step size, predictor learning rate 0.25 times 1e-4 | intrinsic weight 2, learning rate 0.002 (best of 3) | 0.8859 ± 0.0024 | +0.0078 [+0.0016, +0.0141] | 0.9954 ± 0.0007 | +0.0033 [+0.0015, +0.0050] | 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run | intrinsic weight 0.5, learning rate 0.002 (best of 3) | 0.8859 ± 0.0025 | +0.0078 [+0.0014, +0.0143] | 0.9954 ± 0.0005 | +0.0033 [+0.0019, +0.0047] | 256 |
| predictor at half speed with held divisor | intrinsic weight 4, learning rate 0.002 (best of 3) | 0.8849 ± 0.0023 | +0.0068 [+0.0007, +0.0129] | 0.9978 ± 0.0003 | <u>+0.0058</u> [+0.0047, +0.0070] | 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 4, learning rate 0.002 (best of 3) | 0.8806 ± 0.0029 | +0.0025 [-0.0043, +0.0093] | <u>0.9979</u> ± 0.0002 | <u>+0.0058</u> [+0.0047, +0.0070] | 256 |
| separate predictor step size, predictor learning rate 1 times 1e-4 | intrinsic weight 2, learning rate 0.002 (best of 2) | 0.8804 ± 0.0024 | +0.0024 [-0.0035, +0.0085] | 0.9954 ± 0.0004 | +0.0034 [+0.0022, +0.0046] | 256 |
| held divisor alone, held after one quarter of the run | intrinsic weight 4, learning rate 0.002 (the only configuration read) | 0.8718 ± 0.0024 | -0.0063 [-0.0126, -0.0001] | 0.9946 ± 0.0005 | +0.0026 [+0.0012, +0.0039] | 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 0.25, learning rate 0.002 (best of 6) | 0.8714 ± 0.0016 | -0.0066 [-0.0117, -0.0017] | 0.9943 ± 0.0003 | +0.0023 [+0.0012, +0.0034] | 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.002 (best of 2) | 0.8707 ± 0.0058 | -0.0074 [-0.0201, +0.0034] | 0.9872 ± 0.0055 | -0.0048 [-0.0169, +0.0035] | 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | intrinsic weight 2, learning rate 0.002 (best of 3) | 0.8496 ± 0.0036 | -0.0284 [-0.0367, -0.0208] | 0.9919 ± 0.0011 | -0.0001 [-0.0027, +0.0019] | 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held after one sixteenth of the run (batch 11) | intrinsic weight 4, learning rate 0.002 (the only configuration read) | 0.5718 ± 0.0045 | -0.3063 [-0.3159, -0.2965] | 0.8776 ± 0.0045 | -0.1144 [-0.1234, -0.1056] | 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores left running (batch 11) | intrinsic weight 2, learning rate 0.002 (best of 3) | 0.4258 ± 0.0056 | -0.4522 [-0.4641, -0.4405] | 0.6341 ± 0.0071 | -0.3579 [-0.3718, -0.3445] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.003 (best of 5) | 203 ± 8.06 | control | 256 of 256 | 0.8701 | 0.9906 | 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | intrinsic weight 4, learning rate 0.002 (best of 3) | **82.9** ± 4.46 | **-120** [-139, -102] | 256 of 256 | 0.7871 | 0.9770 | 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores left running (batch 11) | intrinsic weight 16, learning rate 0.002 (best of 3) | <u>83</u> ± 6.26 | **-120** [-139, -100] | 256 of 256 | 0.0230 | 0.0305 | 256 |
| fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held after one sixteenth of the run (batch 11) | intrinsic weight 4, learning rate 0.002 (the only configuration read) | 83.3 ± 5.16 | **-120** [-138, -102] | 256 of 256 | 0.5718 | 0.8776 | 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 2, learning rate 0.002 (best of 3) | 85.2 ± 5.18 | <u>-118</u> [-137, -99] | 256 of 256 | 0.8728 | 0.9945 | 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 16, learning rate 0.002 (best of 2) | 86.1 ± 6.37 | -117 [-137, -96.5] | 256 of 256 | 0.5725 | 0.8495 | 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, left running | intrinsic weight 2, learning rate 0.002 (the only configuration read) | 91.7 ± 5.75 | -111 [-129, -93.7] | 256 of 256 | 0.8894 | 0.9931 | 256 |
| error length as the bonus, with the divisor held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002 (best of 6) | 93 ± 4.07 | -110 [-128, -92.4] | 256 of 256 | 0.1189 | 0.2435 | 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run | intrinsic weight 1, learning rate 0.002 (best of 3) | 120 ± 7.9 | -82.9 [-104, -61.3] | 256 of 256 | 0.8712 | 0.9935 | 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.002 (best of 3) | 124 ± 9.81 | -79 [-102, -54.7] | 256 of 256 | 0.8779 | 0.9948 | 256 |
| fast last layer at step 0.03 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.002 (the only configuration read) | 128 ± 6.96 | -74.8 [-94.9, -54.9] | 256 of 256 | **0.8976** | 0.9967 | 256 |
| separate predictor step size, predictor learning rate 0.25 times 1e-4 | intrinsic weight 4, learning rate 0.002 (best of 3) | 133 ± 7.88 | -70.1 [-90.7, -48.3] | 256 of 256 | 0.8783 | 0.9927 | 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 4, learning rate 0.002 (best of 2) | 153 ± 9.69 | -50.3 [-75.1, -25.4] | 256 of 256 | 0.8883 | **0.9982** | 256 |
| predictor at half speed with held divisor | intrinsic weight 4, learning rate 0.002 (best of 3) | 161 ± 10.6 | -42.3 [-66.9, -18.1] | 256 of 256 | 0.8849 | <u>0.9978</u> | 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.002 (the only configuration read) | 180 ± 8.38 | -23 [-45.8, -0.465] | 256 of 256 | <u>0.8928</u> | 0.9971 | 256 |
| separate predictor step size, predictor learning rate 1 times 1e-4 | intrinsic weight 4, learning rate 0.002 (best of 2) | 185 ± 10.1 | -18 [-43.8, +8.55] | 256 of 256 | 0.8762 | 0.9931 | 256 |
| divisor held from each policy's own first arrival | intrinsic weight 4, learning rate 0.002 (best of 3) | 201 ± 9.11 | -2.08 [-25.1, +21.6] | 256 of 256 | 0.8852 | 0.9976 | 256 |
| held divisor alone, held after one quarter of the run | intrinsic weight 4, learning rate 0.002 (the only configuration read) | 205 ± 8.2 | +1.77 [-21.3, +24.3] | 256 of 256 | 0.8718 | 0.9946 | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002 (best of 2) | 213 ± 8.15 | +10.2 [-11, +30.4] | 256 of 256 | 0.8821 | 0.9970 | 256 |

#### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 20:19 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 15:26 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 15:26 PT | none |
| 4 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 15:27 PT | none |
| 5 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 20:18 PT | none |
| 6 | held divisor alone, held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 16:41 PT | none |
| 7 | held divisor alone, held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 17:02 PT | none |
| 8 | held divisor alone, held after one quarter of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 16:37 PT | none |
| 9 | divisor held from each policy's own first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:55 PT | none |
| 10 | divisor held from each policy's own first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:55 PT | none |
| 11 | divisor held from each policy's own first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:07 PT | none |
| 12 | fast last layer at step 0.03 throughout, divisor held from the first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:35 PT | none |
| 13 | fast last layer at step 0.01 throughout, divisor held from the first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:08 PT | 1 continued from a checkpoint |
| 14 | fast last layer at step 0.1 throughout, divisor held from the first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 03:44 PT | none |
| 15 | fast last layer at step 0.1 throughout, divisor held from the first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:46 PT | none |
| 16 | fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:39 PT | none |
| 17 | fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:12 PT | none |
| 18 | fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:01 PT | none |
| 19 | fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, left running | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:31 PT | none |
| 20 | slower predictor with held divisor (divisor held after one sixteenth of the run) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 16:29 PT | none |
| 21 | slower predictor with held divisor (divisor held after one sixteenth of the run) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 16:37 PT | none |
| 22 | predictor at half speed with held divisor | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:34 PT | none |
| 23 | predictor at half speed with held divisor | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:32 PT | none |
| 24 | predictor at half speed with held divisor | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 22:37 PT | none |
| 25 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 18:49 PT | none |
| 26 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:03 PT | none |
| 27 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:02 PT | none |
| 28 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:04 PT | none |
| 29 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:01 PT | none |
| 30 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:02 PT | none |
| 31 | separate predictor step size, predictor learning rate 0.25 times 1e-4 | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 15:57 PT | none |
| 32 | separate predictor step size, predictor learning rate 0.25 times 1e-4 | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 15:42 PT | none |
| 33 | separate predictor step size, predictor learning rate 0.25 times 1e-4 | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 17:04 PT | none |
| 34 | separate predictor step size, predictor learning rate 1 times 1e-4 | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 15:42 PT | none |
| 35 | separate predictor step size, predictor learning rate 1 times 1e-4 | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 15:42 PT | none |
| 36 | error length as the bonus, with the divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:04 PT | none |
| 37 | error length as the bonus, with the divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:04 PT | none |
| 38 | error length as the bonus, with the divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 01:11 PT | none |
| 39 | error length as the bonus, with the divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:40 PT | none |
| 40 | error length as the bonus, with the divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:45 PT | none |
| 41 | error length as the bonus, with the divisor held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 00:45 PT | none |
| 42 | fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:22 PT | none |
| 43 | fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:11 PT | none |
| 44 | fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held (batch 11) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 04:23 PT | none |
| 45 | fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores left running (batch 11) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:09 PT | none |
| 46 | fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores left running (batch 11) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:10 PT | none |
| 47 | fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores left running (batch 11) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:09 PT | none |
| 48 | fast last layer (step 0.1) until the first arrival, then the divisor of the fast scores held after one sixteenth of the run (batch 11) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 05:09 PT | none |

Unit names, by the number in the first column:

1. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.001_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-after-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-after-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-after-one-quarter_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-large_reward-arrival-plus-one_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-step-0-point-03-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-step-0-point-01-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
14. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
15. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-layer-first-arrival-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
16. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-then-predictor-divisor-held_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
17. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-then-predictor-divisor-held_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
18. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-then-predictor-divisor-held_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
19. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-then-predictor-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
20. `validation-block_pointmaze-large_reward-arrival-plus-one_slower-predictor-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
21. `validation-block_pointmaze-large_reward-arrival-plus-one_slower-predictor-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
22. `validation-block_pointmaze-large_reward-arrival-plus-one_predictor-half-speed-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
23. `validation-block_pointmaze-large_reward-arrival-plus-one_predictor-half-speed-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
24. `validation-block_pointmaze-large_reward-arrival-plus-one_predictor-half-speed-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
25. `validation-block_pointmaze-large_reward-arrival-plus-one_three-parts-held-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-0.25_clip-coefficient-0.1_seeds-10000-to-10255`
26. `validation-block_pointmaze-large_reward-arrival-plus-one_three-parts-held-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`
27. `validation-block_pointmaze-large_reward-arrival-plus-one_three-parts-held-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
28. `validation-block_pointmaze-large_reward-arrival-plus-one_three-parts-held-one-eighth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-0.25_clip-coefficient-0.1_seeds-10000-to-10255`
29. `validation-block_pointmaze-large_reward-arrival-plus-one_three-parts-held-one-eighth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`
30. `validation-block_pointmaze-large_reward-arrival-plus-one_three-parts-held-one-eighth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
31. `validation-block_pointmaze-large_reward-arrival-plus-one_predictor-own-step-size-0-point-25_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
32. `validation-block_pointmaze-large_reward-arrival-plus-one_predictor-own-step-size-0-point-25_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
33. `validation-block_pointmaze-large_reward-arrival-plus-one_predictor-own-step-size-0-point-25_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
34. `validation-block_pointmaze-large_reward-arrival-plus-one_predictor-own-step-size-1_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
35. `validation-block_pointmaze-large_reward-arrival-plus-one_predictor-own-step-size-1_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
36. `validation-block_pointmaze-large_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-0.125_clip-coefficient-0.1_seeds-10000-to-10255`
37. `validation-block_pointmaze-large_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-0.25_clip-coefficient-0.1_seeds-10000-to-10255`
38. `validation-block_pointmaze-large_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`
39. `validation-block_pointmaze-large_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
40. `validation-block_pointmaze-large_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
41. `validation-block_pointmaze-large_reward-arrival-plus-one_error-length-with-held-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
42. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-until-arrival-then-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
43. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-until-arrival-then-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
44. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-until-arrival-then-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
45. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-until-arrival-no-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
46. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-until-arrival-no-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
47. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-until-arrival-no-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
48. `validation-block_pointmaze-large_reward-arrival-plus-one_fast-until-arrival-fixed-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

### Second block: seeds that chose nothing

The units of the phase `validation-block-two` in PointMaze Large, reward 1 on arrival. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 8,388,608 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 | 0.8842 ± 0.0019 | 0.9927 ± 0.0005 | 266 ± 10.6 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002 | 0.8819 ± 0.0030 | 0.9969 ± 0.0005 | 259 ± 10 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 2, learning rate 0.002 | 0.8839 ± 0.0025 | **0.9976** ± 0.0003 | 266 ± 11.6 | 256 of 256 |
| fast last layer at step 0.03 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.002 | **0.8995** ± 0.0022 | 0.9968 ± 0.0004 | **118** ± 6.21 | 256 of 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.002 | <u>0.8945</u> ± 0.0023 | <u>0.9975</u> ± 0.0002 | <u>180</u> ± 8.79 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 (its best on the first block and on this block) | 0.8842 ± 0.0019 | control | control | 0.9927 ± 0.0005 | control | 256 |
| fast last layer at step 0.03 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.002 (its best on the first block) | **0.8995** ± 0.0022 | **+0.0153** [+0.0096, +0.0209] | **+0.0153** [+0.0096, +0.0209] | 0.9968 ± 0.0004 | +0.0041 [+0.0030, +0.0054] | 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.002 (its best on the first block) | <u>0.8945</u> ± 0.0023 | <u>+0.0103</u> [+0.0049, +0.0158] | <u>+0.0103</u> [+0.0049, +0.0158] | <u>0.9975</u> ± 0.0002 | <u>+0.0048</u> [+0.0037, +0.0058] | 256 |
| divisor held from each policy's own first arrival | intrinsic weight 2, learning rate 0.002 (its best on the first block) | 0.8839 ± 0.0025 | -0.0003 [-0.0066, +0.0059] | -0.0003 [-0.0066, +0.0059] | **0.9976** ± 0.0003 | **+0.0049** [+0.0038, +0.0060] | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002 (its best on the first block) | 0.8819 ± 0.0030 | -0.0023 [-0.0098, +0.0043] | -0.0023 [-0.0098, +0.0043] | 0.9969 ± 0.0005 | +0.0042 [+0.0028, +0.0055] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND), at its best on this block | intrinsic weight 2, learning rate 0.002 (the only configuration read on this block) | 266 ± 10.6 | the control's first-block best (intrinsic weight 2, learning rate 0.003) has no complete unit on this block | control | 256 of 256 | 256 |
| fast last layer at step 0.03 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.002 (its best on the first block) | **118** ± 6.21 | the control's first-block best (intrinsic weight 2, learning rate 0.003) has no complete unit on this block | **-148** [-172, -125] | 256 of 256 | 256 |
| fast last layer at step 0.01 throughout, divisor held from the first arrival | intrinsic weight 2, learning rate 0.002 (its best on the first block) | <u>180</u> ± 8.79 | the control's first-block best (intrinsic weight 2, learning rate 0.003) has no complete unit on this block | <u>-86.3</u> [-112, -60.6] | 256 of 256 | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 2, learning rate 0.002 (its best on this block: its first-block best, intrinsic weight 4, learning rate 0.002, was not run here) | 259 ± 10 | the control's first-block best (intrinsic weight 2, learning rate 0.003) has no complete unit on this block | -6.95 [-34.8, +20.8] | 256 of 256 | 256 |
| divisor held from each policy's own first arrival | intrinsic weight 2, learning rate 0.002 (its best on this block: its first-block best, intrinsic weight 4, learning rate 0.002, was not run here) | 266 ± 11.6 | the control's first-block best (intrinsic weight 2, learning rate 0.003) has no complete unit on this block | -0.353 [-29.6, +30.8] | 256 of 256 | 256 |

#### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:45 PT | none |
| 2 | held divisor alone, held after one sixteenth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:53 PT | none |
| 3 | divisor held from each policy's own first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 02:53 PT | none |
| 4 | fast last layer at step 0.03 throughout, divisor held from the first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:19 PT | none |
| 5 | fast last layer at step 0.01 throughout, divisor held from the first arrival | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-13 06:17 PT | none |

Unit names, by the number in the first column:

1. `validation-block-two_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-large_reward-arrival-plus-one_divisor-held-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-large_reward-arrival-plus-one_held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
4. `validation-block-two_pointmaze-large_reward-arrival-plus-one_fast-step-0-point-03-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`
5. `validation-block-two_pointmaze-large_reward-arrival-plus-one_fast-step-0-point-01-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-20000-to-20255`

### Discovery check: the leading configurations for the first-goal time

The units of the phase `discovery-validation` in PointMaze Large, reward 1 on arrival. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which these configurations were picked as the leading ones of each method's discovery sweep, nor the confirmation seeds. Read at 8,388,608 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $M$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 | 0.0054 ± 0.0002 | 0.0000 ± 0.0000 | 119 ± 5.46 | **256 of 256** |
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.01, entropy coefficient 0.01 | <u>0.0063</u> ± 0.0002 | <u>0.0015</u> ± 0.0003 | 135 ± 6.06 | **256 of 256** |
| control (CleanRL PPO-RND) | intrinsic weight 256, learning rate 0.001, entropy coefficient 0.03 | 0.0006 ± 0.0000 | 0.0000 ± 0.0000 | 245 ± 45.9 | <u>254 of 256</u> |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.006, entropy coefficient 0.03 | 0.0010 ± 0.0000 | 0.0000 ± 0.0000 | **55.8** ± 2.19 | **256 of 256** |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 4, learning rate 0.002, entropy coefficient 0.03 | 0.0025 ± 0.0001 | 0.0000 ± 0.0000 | 66.3 ± 3.17 | **256 of 256** |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.01 | **0.0360** ± 0.0012 | **0.0417** ± 0.0021 | 77.7 ± 4.54 | **256 of 256** |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run | intrinsic weight 16, learning rate 0.006, entropy coefficient 0.03 | 0.0014 ± 0.0000 | 0.0000 ± 0.0000 | <u>65.3</u> ± 2.63 | **256 of 256** |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run | intrinsic weight 64, learning rate 0.002, entropy coefficient 0.03 | 0.0025 ± 0.0000 | 0.0000 ± 0.0000 | 72.8 ± 2.96 | **256 of 256** |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run | intrinsic weight 4, learning rate 0.01, entropy coefficient 0.03 | 0.0009 ± 0.0000 | 0.0000 ± 0.0000 | 75.6 ± 2.79 | **256 of 256** |

Then, for $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 64, learning rate 0.006, entropy coefficient 0.01 (best of 3) | 119 ± 5.46 | control | 256 of 256 | **0.0054** | 0.0000 | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.006, entropy coefficient 0.03 (best of 3) | **55.8** ± 2.19 | **-63.4** [-75.2, -52] | 256 of 256 | 0.0010 | 0.0000 | 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run | intrinsic weight 16, learning rate 0.006, entropy coefficient 0.03 (best of 3) | <u>65.3</u> ± 2.63 | <u>-54</u> [-65.8, -42.7] | 256 of 256 | <u>0.0014</u> | 0.0000 | 256 |

#### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:11 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:11 PT | none |
| 3 | control (CleanRL PPO-RND) | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:08 PT | none |
| 4 | fast last layer inside the rollout, step 0.1 | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:45 PT | none |
| 5 | fast last layer inside the rollout, step 0.1 | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:46 PT | none |
| 6 | fast last layer inside the rollout, step 0.1 | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:46 PT | none |
| 7 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:45 PT | none |
| 8 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:46 PT | none |
| 9 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one eighth of the run | PointMaze Large, reward 1 on arrival | 1 | complete | 256 of 256 | 2026-11-12 19:45 PT | none |

Unit names, by the number in the first column:

1. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.001_entropy-coefficient-0.03_intrinsic-weight-256.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `discovery-validation_pointmaze-large_reward-arrival-plus-one_control-cleanrl-ppo-rnd_learning-rate-0.01_entropy-coefficient-0.01_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `discovery-validation_pointmaze-large_reward-arrival-plus-one_fast-last-layer-step-0-point-1_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `discovery-validation_pointmaze-large_reward-arrival-plus-one_fast-last-layer-step-0-point-1_learning-rate-0.002_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `discovery-validation_pointmaze-large_reward-arrival-plus-one_fast-last-layer-step-0-point-1_learning-rate-0.006_entropy-coefficient-0.03_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `discovery-validation_pointmaze-large_reward-arrival-plus-one_three-parts-held-one-eighth_learning-rate-0.002_entropy-coefficient-0.03_intrinsic-weight-64.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `discovery-validation_pointmaze-large_reward-arrival-plus-one_three-parts-held-one-eighth_learning-rate-0.006_entropy-coefficient-0.03_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `discovery-validation_pointmaze-large_reward-arrival-plus-one_three-parts-held-one-eighth_learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`

## PointMaze Large, step cost

### First block: each side at its best configuration there

The units of the phase `validation-block` in PointMaze Large, step cost. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which the configurations run here were picked from each side's sweeps and screens, nor the confirmation seeds. Read at 33,554,432 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $W$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 | -0.7310 ± 0.0018 | -0.7163 ± 0.0017 | 191 ± 10.1 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 2, learning rate 0.002 | -0.7411 ± 0.0036 | -0.7229 ± 0.0033 | 241 ± 13.4 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002 | **-0.7215** ± 0.0008 | **-0.7025** ± 0.0009 | 182 ± 9.19 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 4, learning rate 0.002 | -0.7255 ± 0.0023 | -0.7098 ± 0.0022 | 204 ± 10.3 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.003 | -0.7270 ± 0.0003 | -0.7055 ± 0.0002 | 147 ± 6.83 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 8, learning rate 0.003 | -0.7236 ± 0.0021 | -0.7079 ± 0.0021 | 163 ± 8.72 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 8, learning rate 0.002 | -0.7268 ± 0.0025 | -0.7115 ± 0.0025 | 189 ± 9.38 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 4, learning rate 0.002 | -0.7335 ± 0.0034 | -0.7198 ± 0.0035 | 206 ± 11.5 | 256 of 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 8, learning rate 0.002 | -0.7264 ± 0.0016 | -0.7031 ± 0.0018 | <u>71.5</u> ± 4.37 | 256 of 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 8, learning rate 0.0045 | -0.7638 ± 0.0030 | -0.7281 ± 0.0032 | 78.6 ± 5.91 | 256 of 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 4, learning rate 0.002 | -0.7409 ± 0.0040 | -0.7277 ± 0.0041 | **69.4** ± 5.2 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 8, learning rate 0.003 | -0.7230 ± 0.0016 | -0.7058 ± 0.0015 | 127 ± 7.73 | 256 of 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 16, learning rate 0.003 | -0.7243 ± 0.0006 | <u>-0.7027</u> ± 0.0003 | 125 ± 7.29 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 8, learning rate 0.003 | <u>-0.7216</u> ± 0.0014 | -0.7053 ± 0.0014 | 132 ± 9.08 | 256 of 256 |
| predictor at half speed with held divisor | intrinsic weight 4, learning rate 0.003 | -0.7370 ± 0.0036 | -0.7208 ± 0.0035 | 142 ± 9.75 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.002 | -0.7291 ± 0.0025 | -0.7116 ± 0.0025 | 122 ± 8.85 | 256 of 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 0.5, learning rate 0.0045 | -0.7320 ± 0.0028 | -0.7142 ± 0.0028 | 139 ± 10 | 256 of 256 |

Then, for $W$ and $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $W$ (the highest is the best):

| method | best configuration for $W$ | $W$ ↑ ± standard error | difference in $W$ from the control [95% interval] | $L$ ± standard error | difference in $L$ from the control [95% interval] | seeds in common with the control |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 (best of 2) | -0.7310 ± 0.0018 | control | -0.7163 ± 0.0017 | control | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002 (best of 3) | **-0.7215** ± 0.0008 | **+0.0095** [+0.0057, +0.0137] | **-0.7025** ± 0.0009 | **+0.0139** [+0.0102, +0.0178] | 256 |
| predictor at half speed with held divisor | intrinsic weight 8, learning rate 0.003 (best of 2) | <u>-0.7216</u> ± 0.0014 | <u>+0.0094</u> [+0.0048, +0.0140] | -0.7053 ± 0.0014 | +0.0111 [+0.0067, +0.0155] | 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 8, learning rate 0.003 (best of 2) | -0.7230 ± 0.0016 | +0.0081 [+0.0032, +0.0129] | -0.7058 ± 0.0015 | +0.0105 [+0.0061, +0.0149] | 256 |
| divisor held from each policy's own first arrival | intrinsic weight 8, learning rate 0.003 (best of 3) | -0.7236 ± 0.0021 | +0.0074 [+0.0017, +0.0128] | -0.7079 ± 0.0021 | +0.0084 [+0.0030, +0.0138] | 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 8, learning rate 0.002 (best of 2) | -0.7264 ± 0.0016 | +0.0046 [-0.0002, +0.0094] | <u>-0.7031</u> ± 0.0018 | <u>+0.0132</u> [+0.0082, +0.0184] | 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.002 (best of 2) | -0.7291 ± 0.0025 | +0.0019 [-0.0041, +0.0077] | -0.7116 ± 0.0025 | +0.0047 [-0.0013, +0.0102] | 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 4, learning rate 0.002 (the only configuration read) | -0.7409 ± 0.0040 | -0.0099 [-0.0184, -0.0017] | -0.7277 ± 0.0041 | -0.0114 [-0.0202, -0.0031] | 256 |

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 (best of 2) | 191 ± 10.1 | control | 256 of 256 | -0.7310 | -0.7163 | 256 |
| fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | intrinsic weight 4, learning rate 0.002 (the only configuration read) | **69.4** ± 5.2 | **-122** [-144, -100] | 256 of 256 | -0.7409 | -0.7277 | 256 |
| fast last layer at step 0.1 throughout, divisor held from the first arrival | intrinsic weight 8, learning rate 0.002 (best of 2) | <u>71.5</u> ± 4.37 | <u>-120</u> [-142, -98.5] | 256 of 256 | -0.7264 | <u>-0.7031</u> | 256 |
| all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | intrinsic weight 1, learning rate 0.002 (best of 2) | 122 ± 8.85 | -69 [-95, -42.2] | 256 of 256 | -0.7291 | -0.7116 | 256 |
| slower predictor with held divisor (divisor held after one sixteenth of the run) | intrinsic weight 16, learning rate 0.003 (best of 2) | 125 ± 7.29 | -66.3 [-90.8, -41.5] | 256 of 256 | -0.7243 | **-0.7027** | 256 |
| predictor at half speed with held divisor | intrinsic weight 8, learning rate 0.003 (best of 2) | 132 ± 9.08 | -58.9 [-85.7, -32] | 256 of 256 | **-0.7216** | -0.7053 | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.003 (best of 3) | 147 ± 6.83 | -43.8 [-68, -20.2] | 256 of 256 | -0.7270 | -0.7055 | 256 |
| divisor held from each policy's own first arrival | intrinsic weight 8, learning rate 0.003 (best of 3) | 163 ± 8.72 | -28.1 [-52.5, -3.09] | 256 of 256 | <u>-0.7236</u> | -0.7079 | 256 |

Not complete, left out: fast last layer at step 0.03 throughout, divisor held from the first arrival: 0 of 1 configurations read (0 of 1 units complete); fast last layer at step 0.01 throughout, divisor held from the first arrival: 0 of 1 configurations read (0 of 1 units complete); fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held: 1 of 3 configurations read (1 of 3 units complete); fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, left running: 0 of 1 configurations read (0 of 1 units complete).

#### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:25 PT | none |
| 2 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:03 PT | none |
| 3 | held divisor alone, held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:50 PT | none |
| 4 | held divisor alone, held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 00:19 PT | none |
| 5 | held divisor alone, held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:54 PT | none |
| 6 | divisor held from each policy's own first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:32 PT | 1 continued from a checkpoint |
| 7 | divisor held from each policy's own first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:04 PT | none |
| 8 | divisor held from each policy's own first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 03:05 PT | none |
| 9 | fast last layer at step 0.03 throughout, divisor held from the first arrival | PointMaze Large, step cost | 1 | unfinished | 144 of 256 | not finished | none |
| 10 | fast last layer at step 0.01 throughout, divisor held from the first arrival | PointMaze Large, step cost | 1 | not started (no shard) | 0 of 256 | not finished | none |
| 11 | fast last layer at step 0.1 throughout, divisor held from the first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:16 PT | 1 continued from a checkpoint |
| 12 | fast last layer at step 0.1 throughout, divisor held from the first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 05:30 PT | 1 continued from a checkpoint |
| 13 | fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 06:42 PT | none |
| 14 | fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | PointMaze Large, step cost | 1 | unfinished | 153 of 256 | not finished | none |
| 15 | fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, held | PointMaze Large, step cost | 1 | unfinished | 89 of 256 | not finished | none |
| 16 | fast last layer (step 0.1) until the first arrival, then the predictor's own divisor, left running | PointMaze Large, step cost | 1 | unfinished | 88 of 256 | not finished | none |
| 17 | slower predictor with held divisor (divisor held after one sixteenth of the run) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:24 PT | none |
| 18 | slower predictor with held divisor (divisor held after one sixteenth of the run) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:25 PT | none |
| 19 | predictor at half speed with held divisor | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:51 PT | none |
| 20 | predictor at half speed with held divisor | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:27 PT | none |
| 21 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 00:07 PT | none |
| 22 | all three parts (fast last layer with slower predictor and held divisor, step 0.1), divisor held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-12 23:50 PT | none |

Unit names, by the number in the first column:

1. `validation-block_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-2.0_clip-coefficient-0.1_seeds-10000-to-10255`
2. `validation-block_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
3. `validation-block_pointmaze-large_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
4. `validation-block_pointmaze-large_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
5. `validation-block_pointmaze-large_reward-step-cost_divisor-held-after-one-sixteenth_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
6. `validation-block_pointmaze-large_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
7. `validation-block_pointmaze-large_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
8. `validation-block_pointmaze-large_reward-step-cost_divisor-held-from-first-arrival_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
9. `validation-block_pointmaze-large_reward-step-cost_fast-step-0-point-03-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
10. `validation-block_pointmaze-large_reward-step-cost_fast-step-0-point-01-with-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
11. `validation-block_pointmaze-large_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
12. `validation-block_pointmaze-large_reward-step-cost_fast-layer-first-arrival-hold_learning-rate-0.0045_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
13. `validation-block_pointmaze-large_reward-step-cost_fast-then-predictor-divisor-held_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
14. `validation-block_pointmaze-large_reward-step-cost_fast-then-predictor-divisor-held_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
15. `validation-block_pointmaze-large_reward-step-cost_fast-then-predictor-divisor-held_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
16. `validation-block_pointmaze-large_reward-step-cost_fast-then-predictor-divisor_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
17. `validation-block_pointmaze-large_reward-step-cost_slower-predictor-with-held-divisor_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-16.0_clip-coefficient-0.1_seeds-10000-to-10255`
18. `validation-block_pointmaze-large_reward-step-cost_slower-predictor-with-held-divisor_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
19. `validation-block_pointmaze-large_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-10000-to-10255`
20. `validation-block_pointmaze-large_reward-step-cost_predictor-half-speed-held-divisor_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-10000-to-10255`
21. `validation-block_pointmaze-large_reward-step-cost_three-parts-held-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-1.0_clip-coefficient-0.1_seeds-10000-to-10255`
22. `validation-block_pointmaze-large_reward-step-cost_three-parts-held-one-sixteenth_learning-rate-0.0045_entropy-coefficient-0.01_intrinsic-weight-0.5_clip-coefficient-0.1_seeds-10000-to-10255`

### Second block: seeds that chose nothing

The units of the phase `validation-block-two` in PointMaze Large, step cost. Each trains one configuration on 256 seeds, from 20000 to 20255. Nothing was chosen on these seeds before they were read. Read at 33,554,432 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration run on this block, the control first; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 | -0.7328 ± 0.0020 | -0.7175 ± 0.0021 | 203 ± 10.8 | 256 of 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002 | **-0.7232** ± 0.0011 | **-0.7045** ± 0.0011 | <u>184</u> ± 9.22 | 256 of 256 |
| divisor held from each policy's own first arrival | intrinsic weight 8, learning rate 0.003 | <u>-0.7255</u> ± 0.0023 | <u>-0.7101</u> ± 0.0023 | **152** ± 7.71 | 256 of 256 |

Then one table per metric. A method stands at the configuration that was its best for that metric on the first block; a row says so when that configuration was not run here, and the method then stands at its best on this block, which favours it. Each method is compared twice with the control, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds:

1. **Equal rules:** against the control at the configuration that was its best on the first block. Both sides were fixed before this block was read.
2. **Favouring the control:** against the control at its best configuration on this block, chosen among those run here after the block was read.

The control's rows come first, then the methods from the best value on.

Second block, $W$ (the highest is the best):

| method | configuration and how it was fixed | $W$ ↑ ± standard error | difference in $W$, equal rules [95% interval] | difference in $W$, favouring the control [95% interval] | $L$ ± standard error | difference in $L$, equal rules [95% interval] | seeds in common |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 (its best on the first block and on this block) | -0.7328 ± 0.0020 | control | control | -0.7175 ± 0.0021 | control | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002 (its best on the first block) | **-0.7232** ± 0.0011 | **+0.0095** [+0.0051, +0.0141] | **+0.0095** [+0.0051, +0.0141] | **-0.7045** ± 0.0011 | **+0.0130** [+0.0084, +0.0179] | 256 |
| divisor held from each policy's own first arrival | intrinsic weight 8, learning rate 0.003 (its best on the first block) | <u>-0.7255</u> ± 0.0023 | <u>+0.0072</u> [+0.0013, +0.0131] | <u>+0.0072</u> [+0.0013, +0.0131] | <u>-0.7101</u> ± 0.0023 | <u>+0.0074</u> [+0.0013, +0.0135] | 256 |

Second block, $M$ (the smallest is the best):

| method | configuration and how it was fixed | $M$ ↓ ± standard error, thousand steps | difference in $M$, equal rules [95% interval], thousand steps | difference in $M$, favouring the control [95% interval], thousand steps | policies that reach the goal | seeds in common |
|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 4, learning rate 0.002 (its best on the first block and on this block) | 203 ± 10.8 | control | control | 256 of 256 | 256 |
| divisor held from each policy's own first arrival | intrinsic weight 8, learning rate 0.003 (its best on the first block) | **152** ± 7.71 | **-51.1** [-78, -24.9] | **-51.1** [-78, -24.9] | 256 of 256 | 256 |
| held divisor alone, held after one sixteenth of the run | intrinsic weight 8, learning rate 0.002 (its best on this block: its first-block best, intrinsic weight 8, learning rate 0.003, was not run here) | <u>184</u> ± 9.22 | <u>-19.3</u> [-46.1, +7] | <u>-19.3</u> [-46.1, +7] | 256 of 256 | 256 |

#### Units

| unit | method | case | configurations | state | logging windows written | finished | restarts |
|---|---|---|---|---|---|---|---|
| 1 | control (CleanRL PPO-RND) | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 04:00 PT | none |
| 2 | held divisor alone, held after one sixteenth of the run | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 04:25 PT | none |
| 3 | divisor held from each policy's own first arrival | PointMaze Large, step cost | 1 | complete | 256 of 256 | 2026-11-13 04:25 PT | none |

Unit names, by the number in the first column:

1. `validation-block-two_pointmaze-large_reward-step-cost_control-cleanrl-ppo-rnd_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-4.0_clip-coefficient-0.1_seeds-20000-to-20255`
2. `validation-block-two_pointmaze-large_reward-step-cost_divisor-held-one-sixteenth_learning-rate-0.002_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`
3. `validation-block-two_pointmaze-large_reward-step-cost_held-from-first-arrival_learning-rate-0.003_entropy-coefficient-0.01_intrinsic-weight-8.0_clip-coefficient-0.1_seeds-20000-to-20255`

### Discovery check: the leading configurations for the first-goal time

The units of the phase `discovery-validation` in PointMaze Large, step cost. Each trains one configuration on 256 seeds, from 10000 to 10255: neither the development seeds, on which these configurations were picked as the leading ones of each method's discovery sweep, nor the confirmation seeds. Read at 33,554,432 environment steps per policy. Define $W$, $L$ and $M$ as in the section Measures. First table: every configuration, the control first, each method from its best on for $M$; per column the best value is bold and the second best underlined (the smallest for $M$).

| method | configuration | $W$ ± standard error | $L$ ± standard error | $M$ ± standard error, thousand steps | policies that reach the goal |
|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006, entropy coefficient 0.01 | <u>-0.9669</u> ± 0.0017 | <u>-0.9803</u> ± 0.0022 | 120 ± 5.6 | 256 of 256 |
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.0045, entropy coefficient 0.01 | **-0.8831** ± 0.0022 | **-0.8849** ± 0.0026 | 131 ± 6.45 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.006, entropy coefficient 0.03 | -1.0000 ± 0.0000 | -1.0000 ± 0.0000 | **58.9** ± 2.49 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.002, entropy coefficient 0.03 | -0.9999 ± 0.0000 | -1.0000 ± 0.0000 | <u>62.1</u> ± 2.52 | 256 of 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 8, learning rate 0.0045, entropy coefficient 0.01 | -0.9950 ± 0.0002 | -0.9970 ± 0.0003 | 76.4 ± 4.83 | 256 of 256 |

Then, for $M$: each method at its best configuration on these seeds against the control at its own, paired on the seeds both have, with the 95% interval from 10,000 resamples of those seeds. Each best configuration was chosen on these same seeds, among the few run here: a second selection step, not the comparison on the confirmation seeds.

Best against best for $M$ (the smallest is the best):

| method | best configuration for $M$ | $M$ ↓ ± standard error, thousand steps | difference in $M$ from the control [95% interval], thousand steps | policies that reach the goal | $W$ at that configuration | $L$ at that configuration | seeds in common with the control |
|---|---|---|---|---|---|---|---|
| control (CleanRL PPO-RND) | intrinsic weight 8, learning rate 0.006, entropy coefficient 0.01 (best of 2) | <u>120</u> ± 5.6 | control | 256 of 256 | **-0.9669** | **-0.9803** | 256 |
| fast last layer inside the rollout, step 0.1 | intrinsic weight 16, learning rate 0.006, entropy coefficient 0.03 (best of 3) | **58.9** ± 2.49 | -61.3 [-73.5, -49.6] | 256 of 256 | <u>-1.0000</u> | <u>-1.0000</u> | 256 |

#### Units

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

