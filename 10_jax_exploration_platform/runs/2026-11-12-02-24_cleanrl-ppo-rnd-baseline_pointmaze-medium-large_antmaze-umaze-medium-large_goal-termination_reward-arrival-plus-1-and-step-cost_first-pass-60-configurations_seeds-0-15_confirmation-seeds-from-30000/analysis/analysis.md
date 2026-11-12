# Training-return report

Generated 2026-11-12 10:44 PT; times are Pacific, converted from machines running Eastern. Run folder: `2026-11-12-02-24_cleanrl-ppo-rnd-baseline_pointmaze-medium-large_antmaze-umaze-medium-large_goal-termination_reward-arrival-plus-1-and-step-cost_first-pass-60-configurations_seeds-0-15_confirmation-seeds-from-30000`.

## Contents

| section | first written | last changed | status |
|---|---|---|---|
| [Measures](#measures) | 2026-11-12 10:40 PT | 2026-11-12 10:40 PT | unread |
| [Records read](#records-read) | 2026-11-12 10:40 PT | 2026-11-12 10:40 PT | unread |
| [Development sweeps](#development-sweeps) | 2026-11-12 10:40 PT | 2026-11-12 10:40 PT | unread |
| [Budget](#budget) | 2026-11-12 10:40 PT | 2026-11-12 10:45 PT | unread |
| [Training curves](#training-curves) | 2026-11-12 10:40 PT | 2026-11-12 10:40 PT | unread |
| [Final level against the training length](#final-level-against-the-training-length) | 2026-11-12 10:40 PT | 2026-11-12 10:40 PT | unread |
| [Decline check](#decline-check) | 2026-11-12 10:40 PT | 2026-11-12 10:40 PT | unread |

Times are Pacific, converted from machines running Eastern, and come from this report's own history of regenerations. [The same report with unread sections in blue and changed text in brown](analysis.html). No section marks itself read.

## Measures

Define $K$ as the number of policies (seeds) of a configuration, $B$ as the budget in environment steps per policy, $R_i(s)$ as the mean return of the training episodes of policy $i$ that end in the logging window containing step $s$, and $\Delta s$ as the number of a logging window's steps that lie inside the range being averaged. The return is of the reward the learner sees. With reward 1 on arrival an episode returns 1 if it reaches the goal and 0 otherwise. With the step cost it returns $-(t-1)/H$ when it reaches the goal at step $t$ of an episode limited to $H$ steps, and $-1$ otherwise. Define $R_0$ as the return of an episode that never reaches the goal ($R_0 = 0$ with reward 1 on arrival, $R_0 = -1$ with the step cost).

A window in which no episode ends takes the value of the window before it; before a policy's first finished episode the value is $R_0$. A window that straddles an end of a range counts in proportion to its steps inside the range.

1. **Training curve:** $\bar R(s) = \dfrac{1}{K}\sum_{i=1}^{K} R_i(s)$. Bands in the figures are one standard error over the $K$ policies.
2. **Whole-curve mean, which chooses configurations:** $W(B) = \dfrac{1}{K}\sum_{i=1}^{K} \dfrac{1}{B}\sum_{s \le B} R_i(s) \thinspace \Delta s$, the area under the curve divided by the budget.
3. **Final level:** $L(B) = \dfrac{1}{K}\sum_{i=1}^{K} \dfrac{1}{0.1 B}\sum_{0.9 B \lt s \le B} R_i(s) \thinspace \Delta s$, the same mean over the last 10% of the budget. The final level at a share $f$ of the budget is $L(f B)$ read from the same runs: the mean over the steps from $0.9 f B$ to $f B$.
4. **Final level over peak:** define $S(s)$ as the mean of the seed-averaged curve $\bar R$ over the $0.1 B$ steps that end at step $s$, and $P(B) = \max_{0.1 B \le s \le B} S(s)$ as its highest value. The ratio is $\dfrac{L(B) - R_0}{P(B) - R_0}$; both are measured from $R_0$, so the ratio means the same in both reward settings. The smoothing has the width of the final level, so $S(B) = L(B)$ and the ratio is at most 1. The decline check also gives the ratio with the highest single window of $\bar R$ in place of $P(B)$; that ratio is the stricter one, because one window of a few seeds is raised by noise.
5. **Mean capped first-goal step:** define $T_i$ as the training step of policy $i$'s first arrival at the goal ($T_i = \infty$ if it has none). The measure is $M(B) = \dfrac{1}{K}\sum_{i=1}^{K} \min(T_i, B)$.
6. **Policies that reach the goal:** $F(B) = \dfrac{1}{K}\sum_{i=1}^{K} \mathbf{1}[T_i \le B]$.
7. **Maze cells reached:** the share of open maze cells a policy has reached in training by step $B$, averaged over the policies.
8. **Mean steps to the goal, last 10%:** the mean length of the configuration's training episodes that reach the goal in the last 10% of the budget, over all its policies together.
9. **Episodes that reach the goal, last 10%:** per policy, the share of its training episodes ending in the last 10% of the budget that reach the goal; then the mean over the policies.

Standard errors are over the $K$ policies. Measures 4 to 9 are reported beside $W$ and $L$ and never choose a configuration.

**Choice and comparison.** A configuration is chosen on the development seeds (below 30000) by the highest $W(B)$; ties go to the higher $L(B)$. On the confirmation seeds (30000 on), define $d_i$ as the value of seed $i$ under a method minus its value under the baseline, on the $K$ seeds both have. The tables give $\bar d = \dfrac{1}{K}\sum_{i=1}^{K} d_i$ for $W$ and for $L$, each with a 95% interval: the 2.5% and 97.5% points of the means of 10,000 resamples of the seeds, drawn with replacement. A difference is ahead when its interval is entirely above zero, behind when it is entirely below zero, and level otherwise. A method beats the baseline in one environment and reward setting when $W$ is ahead and $L$ is not behind. It succeeds when it beats the baseline in all five environments in one reward setting.

**Unfinished runs.** A table compares its rows at one step count: the planned steps per policy when at least one row has trained that far, else the largest step count every policy of the table has reached. Rows that have not reached it are listed under the table without measures. The learning rate falls to zero over the planned steps, so the measures of an unfinished run are not those of a run trained for fewer steps.

## Records read

16 units, 16 complete; 3,840 policies with at least one logging window. A unit that was continued from a checkpoint keeps one unbroken sequence of windows; a unit that was started again from zero is read from its last start. Episodes restarted after a non-finite physics state count as episodes that reach the step limit, and stay in every sum (the records hold sums per window, not single episodes).

| unit | environment | reward setting | method | environments per policy | policies | steps per policy so far | planned steps per policy | state |
|---|---|---|---|---|---|---|---|---|
| 1 | PointMaze Large | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 256 | 50,331,648 | 50,331,648 | complete |
| 2 | PointMaze Large | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 256 | 50,331,648 | 50,331,648 | complete |
| 3 | PointMaze Large | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 256 | 50,331,648 | 50,331,648 | complete |
| 4 | PointMaze Large | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 256 | 50,331,648 | 50,331,648 | complete |
| 5 | PointMaze Large | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 256 | 50,331,648 | 50,331,648 | complete |
| 6 | PointMaze Large | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 256 | 50,331,648 | 50,331,648 | complete |
| 7 | PointMaze Large | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 256 | 50,331,648 | 50,331,648 | complete |
| 8 | PointMaze Large | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 128 | 50,331,648 | 50,331,648 | complete |
| 9 | PointMaze Medium | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 256 | 50,331,648 | 50,331,648 | complete |
| 10 | PointMaze Medium | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 256 | 50,331,648 | 50,331,648 | complete |
| 11 | PointMaze Medium | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 256 | 50,331,648 | 50,331,648 | complete |
| 12 | PointMaze Medium | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 256 | 50,331,648 | 50,331,648 | complete |
| 13 | PointMaze Medium | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 256 | 50,331,648 | 50,331,648 | complete |
| 14 | PointMaze Medium | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 256 | 50,331,648 | 50,331,648 | complete |
| 15 | PointMaze Medium | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 256 | 50,331,648 | 50,331,648 | complete |
| 16 | PointMaze Medium | reward 1 on arrival and step cost | CleanRL PPO-RND | 4 | 128 | 50,331,648 | 50,331,648 | complete |

Unit names, by the number in the first column:

1. `development-first-pass_pointmaze-large_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-01-to-08-of-60_seeds-0-to-15`
2. `development-first-pass_pointmaze-large_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-09-to-16-of-60_seeds-0-to-15`
3. `development-first-pass_pointmaze-large_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-17-to-24-of-60_seeds-0-to-15`
4. `development-first-pass_pointmaze-large_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-25-to-32-of-60_seeds-0-to-15`
5. `development-first-pass_pointmaze-large_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-33-to-40-of-60_seeds-0-to-15`
6. `development-first-pass_pointmaze-large_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-41-to-48-of-60_seeds-0-to-15`
7. `development-first-pass_pointmaze-large_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-49-to-56-of-60_seeds-0-to-15`
8. `development-first-pass_pointmaze-large_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-57-to-60-of-60_seeds-0-to-15`
9. `development-first-pass_pointmaze-medium_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-01-to-08-of-60_seeds-0-to-15`
10. `development-first-pass_pointmaze-medium_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-09-to-16-of-60_seeds-0-to-15`
11. `development-first-pass_pointmaze-medium_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-17-to-24-of-60_seeds-0-to-15`
12. `development-first-pass_pointmaze-medium_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-25-to-32-of-60_seeds-0-to-15`
13. `development-first-pass_pointmaze-medium_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-33-to-40-of-60_seeds-0-to-15`
14. `development-first-pass_pointmaze-medium_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-41-to-48-of-60_seeds-0-to-15`
15. `development-first-pass_pointmaze-medium_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-49-to-56-of-60_seeds-0-to-15`
16. `development-first-pass_pointmaze-medium_reward-arrival-plus-one-and-step-cost_cleanrl-ppo-rnd_cells-57-to-60-of-60_seeds-0-to-15`

## Development sweeps

Define $W$, $L$ and the other measures as in the section Measures. Each table holds one method's configurations in one environment and reward setting, trained on the development seeds (below 30000).

### PointMaze Medium, reward 1 on arrival, CleanRL PPO-RND

60 configurations with 16 seeds each, compared at 50,331,648 environment steps per policy. Rows are sorted by $W$; the best value of each column is bold and the second best underlined. For the mean capped first-goal step and the mean steps to the goal the smallest value is the best.

| configuration | whole-curve mean $W$ ↑ ± standard error | final level $L$ ± standard error | final level over peak | mean capped first-goal step | policies that reach the goal | maze cells reached | mean steps to the goal, last 10% | episodes that reach the goal, last 10% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01 | **0.998** ± 0.001 | **1.000** ± 0.000 | **100.0%** | 76.2K | **100.0%** | 94.7% | 124.9 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.01 | <u>0.997</u> ± 0.001 | **1.000** ± 0.000 | **100.0%** | 35.1K | **100.0%** | 97.6% | 126.3 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.001 | 0.995 ± 0.001 | **1.000** ± 0.000 | **100.0%** | 90.4K | **100.0%** | 97.4% | 129.9 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.01 | 0.995 ± 0.001 | 0.996 ± 0.001 | 99.8% | 19.5K | **100.0%** | 98.3% | 136.5 | 99.6% | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.001 | 0.995 ± 0.001 | <u>0.998</u> ± 0.002 | 99.8% | 91.0K | **100.0%** | 99.3% | 130.1 | <u>99.8%</u> | 16 |
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.01 | 0.994 ± 0.001 | 0.997 ± 0.001 | <u>99.9%</u> | 53.2K | **100.0%** | 96.9% | 131.7 | 99.7% | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.01 | 0.992 ± 0.002 | **1.000** ± 0.000 | **100.0%** | 320.7K | **100.0%** | 95.2% | 123.2 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.01 | 0.991 ± 0.001 | 0.995 ± 0.001 | 99.8% | 37.1K | **100.0%** | 99.0% | 135.0 | 99.5% | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.01 | 0.988 ± 0.004 | **1.000** ± 0.000 | **100.0%** | 174.0K | **100.0%** | <u>99.8%</u> | 126.4 | **100.0%** | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.01 | 0.982 ± 0.006 | **1.000** ± 0.000 | **100.0%** | 689.9K | **100.0%** | 97.1% | 123.1 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.001 | 0.978 ± 0.005 | 0.982 ± 0.008 | 98.5% | 73.6K | **100.0%** | <u>99.8%</u> | 148.8 | 98.3% | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.01 | 0.977 ± 0.006 | **1.000** ± 0.000 | **100.0%** | 789.0K | **100.0%** | 96.4% | 121.8 | **100.0%** | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.001 | 0.974 ± 0.011 | **1.000** ± 0.000 | **100.0%** | 814.2K | **100.0%** | 96.2% | 129.7 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 0.968 ± 0.027 | 0.980 ± 0.013 | 98.2% | 83.6K | **100.0%** | 97.6% | 136.5 | 98.4% | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.001 | 0.959 ± 0.011 | **1.000** ± 0.000 | **100.0%** | 1.50M | **100.0%** | 95.9% | 126.2 | **100.0%** | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.001 | 0.956 ± 0.014 | 0.995 ± 0.004 | 99.6% | 1.46M | **100.0%** | 98.1% | 130.7 | 99.6% | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.01 | 0.953 ± 0.003 | 0.961 ± 0.004 | 99.2% | 39.7K | **100.0%** | **100.0%** | 153.8 | 96.2% | 16 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 0.951 ± 0.013 | 0.943 ± 0.035 | 95.5% | <u>18.3K</u> | **100.0%** | 98.3% | 153.5 | 94.8% | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.01 | 0.947 ± 0.004 | 0.945 ± 0.010 | 97.7% | 36.4K | **100.0%** | **100.0%** | 156.0 | 94.5% | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.01 | 0.946 ± 0.018 | **1.000** ± 0.000 | **100.0%** | 1.79M | **100.0%** | 98.8% | 123.5 | **100.0%** | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.01 | 0.932 ± 0.008 | 0.928 ± 0.010 | 97.6% | 24.9K | **100.0%** | **100.0%** | 168.1 | 92.9% | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.01 | 0.914 ± 0.061 | 0.938 ± 0.062 | **100.0%** | 1.68M | **100.0%** | 92.3% | 121.8 | 93.8% | 16 |
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.001 | 0.885 ± 0.055 | 0.810 ± 0.088 | 82.6% | 41.2K | **100.0%** | **100.0%** | 148.9 | 82.4% | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.001 | 0.844 ± 0.017 | 0.793 ± 0.054 | 88.9% | 37.8K | **100.0%** | **100.0%** | 172.7 | 79.7% | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.01 | 0.816 ± 0.082 | 0.875 ± 0.085 | **100.0%** | 5.22M | <u>93.8%</u> | 96.6% | 122.5 | 87.5% | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.01 | 0.813 ± 0.092 | 0.834 ± 0.091 | <u>99.9%</u> | 4.59M | <u>93.8%</u> | 88.9% | 124.1 | 83.4% | 16 |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.01 | 0.784 ± 0.087 | 0.835 ± 0.091 | 99.8% | 5.98M | <u>93.8%</u> | 93.5% | 125.0 | 83.5% | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.001 | 0.773 ± 0.037 | 0.710 ± 0.086 | 81.5% | 26.5K | **100.0%** | **100.0%** | 162.6 | 73.4% | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 0.767 ± 0.054 | 0.728 ± 0.067 | 82.5% | 55.0K | **100.0%** | **100.0%** | 187.1 | 74.9% | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.001 | 0.765 ± 0.082 | 0.875 ± 0.085 | 99.4% | 5.50M | <u>93.8%</u> | 98.1% | 131.6 | 87.5% | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.001 | 0.763 ± 0.097 | 0.812 ± 0.101 | **100.0%** | 5.03M | <u>93.8%</u> | 90.6% | 126.3 | 81.2% | 16 |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.001 | 0.741 ± 0.091 | 0.870 ± 0.085 | 99.7% | 1.99M | **100.0%** | 94.5% | 135.5 | 87.0% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.01 | 0.720 ± 0.018 | 0.674 ± 0.017 | 84.2% | 44.4K | **100.0%** | **100.0%** | 190.2 | 67.5% | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.01 | 0.717 ± 0.087 | 0.875 ± 0.085 | **100.0%** | 6.01M | **100.0%** | 97.4% | 123.9 | 87.5% | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.001 | 0.688 ± 0.105 | 0.750 ± 0.112 | **100.0%** | 8.20M | 87.5% | 89.9% | 125.8 | 75.0% | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.01 | 0.681 ± 0.117 | 0.689 ± 0.119 | 99.1% | 6.84M | <u>93.8%</u> | 95.0% | <u>121.7</u> | 68.9% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.01 | 0.677 ± 0.018 | 0.603 ± 0.027 | 79.7% | 26.9K | **100.0%** | **100.0%** | 216.2 | 60.4% | 16 |
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.01 | 0.666 ± 0.093 | 0.813 ± 0.101 | **100.0%** | 11.27M | <u>93.8%</u> | 96.2% | 123.6 | 81.3% | 16 |
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.001 | 0.585 ± 0.119 | 0.625 ± 0.125 | **100.0%** | 19.96M | 62.5% | 85.8% | 130.3 | 62.5% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.001 | 0.564 ± 0.023 | 0.449 ± 0.052 | 68.9% | 65.8K | **100.0%** | **100.0%** | 240.0 | 45.2% | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.001 | 0.547 ± 0.113 | 0.625 ± 0.125 | **100.0%** | 15.88M | 75.0% | 88.0% | 129.4 | 62.5% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.01 | 0.533 ± 0.113 | 0.625 ± 0.125 | **100.0%** | 15.17M | 75.0% | 83.9% | **121.2** | 62.5% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.001 | 0.521 ± 0.120 | 0.562 ± 0.128 | **100.0%** | 17.75M | 68.8% | 80.0% | 131.7 | 56.2% | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 0.503 ± 0.120 | 0.562 ± 0.128 | **100.0%** | 18.41M | 68.8% | 88.0% | 140.5 | 56.2% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.001 | 0.469 ± 0.124 | 0.500 ± 0.129 | **100.0%** | 22.31M | 56.2% | 83.9% | 130.9 | 50.0% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.01 | 0.467 ± 0.123 | 0.500 ± 0.129 | **100.0%** | 23.66M | 56.2% | 81.7% | 124.1 | 50.0% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.01 | 0.466 ± 0.109 | 0.570 ± 0.126 | 99.8% | 12.10M | 81.2% | 89.4% | 125.4 | 57.0% | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.01 | 0.459 ± 0.114 | 0.562 ± 0.128 | **100.0%** | 16.00M | 75.0% | 90.4% | 123.3 | 56.2% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.001 | 0.431 ± 0.126 | 0.438 ± 0.128 | **100.0%** | 25.46M | 50.0% | 74.8% | 124.9 | 43.8% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.001 | 0.403 ± 0.120 | 0.438 ± 0.128 | **100.0%** | 26.18M | 50.0% | 80.3% | 124.6 | 43.8% | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.001 | 0.402 ± 0.120 | 0.494 ± 0.128 | **100.0%** | 23.10M | 62.5% | 88.5% | 138.9 | 49.4% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.001 | 0.378 ± 0.037 | 0.242 ± 0.056 | 46.1% | 32.3K | **100.0%** | **100.0%** | 213.2 | 24.9% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.01 | 0.378 ± 0.113 | 0.454 ± 0.125 | **100.0%** | 17.64M | 68.8% | 90.6% | 124.3 | 45.4% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.01 | 0.372 ± 0.050 | 0.232 ± 0.069 | 32.2% | **17.7K** | **100.0%** | **100.0%** | 226.7 | 23.3% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.01 | 0.353 ± 0.119 | 0.375 ± 0.125 | **100.0%** | 26.23M | 50.0% | 86.3% | 123.2 | 37.5% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.001 | 0.353 ± 0.118 | 0.380 ± 0.124 | 96.5% | 16.47M | 75.0% | 85.6% | 123.7 | 38.1% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.001 | 0.323 ± 0.037 | 0.271 ± 0.070 | 54.0% | 54.3K | **100.0%** | **100.0%** | 206.2 | 27.9% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.01 | 0.312 ± 0.120 | 0.312 ± 0.120 | **100.0%** | 31.46M | 37.5% | 79.3% | 123.2 | 31.2% | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.001 | 0.312 ± 0.119 | 0.312 ± 0.120 | **100.0%** | 18.89M | 62.5% | 83.4% | 124.1 | 31.2% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.001 | 0.298 ± 0.115 | 0.312 ± 0.120 | **100.0%** | 25.60M | 50.0% | 81.5% | 125.1 | 31.2% | 16 |

Chosen (highest $W$, ties by $L$): **intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01**. Place in the sweep: intrinsic weight 1 is inside its 10 swept values; learning rate 0.001 is the largest of its 3 swept values; entropy coefficient 0.01 is the largest of its 2 swept values.

### PointMaze Medium, step cost, CleanRL PPO-RND

60 configurations with 16 seeds each, compared at 50,331,648 environment steps per policy. Rows are sorted by $W$; the best value of each column is bold and the second best underlined. For the mean capped first-goal step and the mean steps to the goal the smallest value is the best.

| configuration | whole-curve mean $W$ ↑ ± standard error | final level $L$ ± standard error | final level over peak | mean capped first-goal step | policies that reach the goal | maze cells reached | mean steps to the goal, last 10% | episodes that reach the goal, last 10% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.001 | **-0.303** ± 0.001 | <u>-0.302</u> ± 0.001 | **100.0%** | 20.7K | 100.0% | 98.6% | <u>121.8</u> | **100.0%** | 16 |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.01 | <u>-0.304</u> ± 0.001 | **-0.301** ± 0.001 | **100.0%** | 18.9K | 100.0% | 95.9% | **121.5** | **100.0%** | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.01 | <u>-0.304</u> ± 0.001 | <u>-0.302</u> ± 0.001 | **100.0%** | 28.9K | 100.0% | 97.8% | <u>121.8</u> | **100.0%** | 16 |
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.01 | <u>-0.304</u> ± 0.001 | <u>-0.302</u> ± 0.001 | **100.0%** | 23.1K | 100.0% | 95.7% | 121.9 | **100.0%** | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.001 | -0.305 ± 0.001 | -0.303 ± 0.001 | **100.0%** | 27.0K | 100.0% | 97.4% | 122.0 | **100.0%** | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.001 | -0.305 ± 0.002 | <u>-0.302</u> ± 0.001 | **100.0%** | **13.8K** | 100.0% | 95.7% | 121.9 | **100.0%** | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.001 | -0.305 ± 0.001 | -0.303 ± 0.001 | **100.0%** | 23.6K | 100.0% | 99.3% | 122.1 | **100.0%** | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.01 | -0.305 ± 0.001 | -0.303 ± 0.001 | **100.0%** | <u>14.1K</u> | 100.0% | 94.0% | 122.2 | **100.0%** | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.01 | -0.305 ± 0.001 | -0.304 ± 0.001 | **100.0%** | 22.6K | 100.0% | 98.1% | 122.5 | **100.0%** | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.001 | -0.306 ± 0.002 | -0.304 ± 0.002 | **100.0%** | 23.3K | 100.0% | **100.0%** | 122.4 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.01 | -0.306 ± 0.001 | -0.304 ± 0.001 | **100.0%** | 21.7K | 100.0% | 99.3% | 122.6 | **100.0%** | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.01 | -0.306 ± 0.001 | -0.303 ± 0.001 | **100.0%** | 21.4K | 100.0% | 99.5% | 122.4 | **100.0%** | 16 |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.001 | -0.306 ± 0.001 | -0.306 ± 0.002 | 99.6% | 14.5K | 100.0% | 94.5% | 123.2 | **100.0%** | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.01 | -0.306 ± 0.001 | -0.303 ± 0.001 | **100.0%** | 16.2K | 100.0% | 94.0% | 122.1 | **100.0%** | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.01 | -0.306 ± 0.001 | -0.304 ± 0.001 | **100.0%** | 17.5K | 100.0% | 97.4% | 122.4 | **100.0%** | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.01 | -0.306 ± 0.002 | -0.303 ± 0.001 | **100.0%** | 24.0K | 100.0% | 99.5% | 122.2 | **100.0%** | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.001 | -0.306 ± 0.001 | -0.305 ± 0.001 | 99.8% | 15.9K | 100.0% | 93.5% | 123.0 | **100.0%** | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.01 | -0.306 ± 0.003 | -0.303 ± 0.003 | **100.0%** | 17.3K | 100.0% | 90.6% | 122.2 | **100.0%** | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.01 | -0.306 ± 0.001 | -0.304 ± 0.002 | **100.0%** | 17.7K | 100.0% | 99.0% | 122.7 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.01 | -0.307 ± 0.001 | -0.305 ± 0.001 | <u>99.9%</u> | 17.9K | 100.0% | 99.5% | 123.2 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.001 | -0.307 ± 0.001 | -0.306 ± 0.002 | 99.7% | 22.0K | 100.0% | **100.0%** | 123.4 | **100.0%** | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.001 | -0.307 ± 0.001 | -0.306 ± 0.002 | 99.7% | 14.9K | 100.0% | 97.4% | 123.4 | **100.0%** | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.001 | -0.307 ± 0.002 | -0.305 ± 0.002 | <u>99.9%</u> | 23.1K | 100.0% | 99.0% | 123.1 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01 | -0.308 ± 0.003 | -0.304 ± 0.002 | **100.0%** | 16.4K | 100.0% | 98.3% | 122.6 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | -0.308 ± 0.001 | -0.309 ± 0.002 | 99.3% | 15.6K | 100.0% | 99.0% | 124.3 | **100.0%** | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.001 | -0.308 ± 0.002 | -0.307 ± 0.002 | 99.7% | 16.7K | 100.0% | 94.7% | 123.9 | **100.0%** | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.001 | -0.309 ± 0.002 | -0.307 ± 0.002 | **100.0%** | 25.4K | 100.0% | 98.8% | 123.8 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.001 | -0.310 ± 0.001 | -0.311 ± 0.002 | 99.4% | 15.4K | 100.0% | 98.1% | 125.3 | <u>99.9%</u> | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.01 | -0.310 ± 0.004 | -0.304 ± 0.002 | **100.0%** | 17.4K | 100.0% | 98.3% | 122.5 | **100.0%** | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.01 | -0.310 ± 0.003 | -0.305 ± 0.002 | **100.0%** | 30.7K | 100.0% | 99.5% | 122.9 | **100.0%** | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.001 | -0.310 ± 0.005 | -0.304 ± 0.001 | **100.0%** | 23.3K | 100.0% | <u>99.8%</u> | 122.6 | **100.0%** | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.01 | -0.311 ± 0.003 | -0.306 ± 0.002 | **100.0%** | 15.3K | 100.0% | 97.1% | 123.4 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.01 | -0.311 ± 0.001 | -0.314 ± 0.003 | 99.0% | 16.3K | 100.0% | 98.6% | 126.4 | <u>99.9%</u> | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | -0.313 ± 0.009 | -0.304 ± 0.001 | **100.0%** | 19.2K | 100.0% | 98.1% | 122.4 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.01 | -0.315 ± 0.002 | -0.313 ± 0.003 | 99.6% | 18.5K | 100.0% | 99.5% | 125.9 | **100.0%** | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.01 | -0.317 ± 0.008 | -0.305 ± 0.001 | **100.0%** | 193.6K | 100.0% | 98.6% | 122.8 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.01 | -0.317 ± 0.001 | -0.318 ± 0.002 | 99.1% | 19.0K | 100.0% | **100.0%** | 127.7 | <u>99.9%</u> | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.01 | -0.321 ± 0.017 | -0.318 ± 0.017 | **100.0%** | 14.2K | 100.0% | 95.9% | 125.3 | **100.0%** | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.001 | -0.321 ± 0.016 | -0.318 ± 0.015 | **100.0%** | 20.7K | 100.0% | <u>99.8%</u> | 125.7 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.001 | -0.322 ± 0.005 | -0.328 ± 0.007 | 97.9% | 25.0K | 100.0% | **100.0%** | 129.7 | 99.6% | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.01 | -0.325 ± 0.018 | -0.320 ± 0.017 | **100.0%** | 18.9K | 100.0% | **100.0%** | 126.0 | **100.0%** | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.001 | -0.328 ± 0.022 | -0.310 ± 0.005 | **100.0%** | 19.9K | 100.0% | **100.0%** | 124.4 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.001 | -0.334 ± 0.008 | -0.328 ± 0.012 | 98.2% | 15.1K | 100.0% | **100.0%** | 129.0 | 99.4% | 16 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | -0.334 ± 0.015 | -0.334 ± 0.015 | 98.4% | 17.0K | 100.0% | <u>99.8%</u> | 130.5 | 99.5% | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.01 | -0.341 ± 0.003 | -0.353 ± 0.006 | 96.6% | 24.3K | 100.0% | <u>99.8%</u> | 138.4 | 98.8% | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.01 | -0.346 ± 0.003 | -0.347 ± 0.005 | 97.9% | 29.9K | 100.0% | **100.0%** | 137.2 | 99.2% | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.001 | -0.346 ± 0.027 | -0.338 ± 0.022 | **100.0%** | 18.2K | 100.0% | 93.5% | 130.7 | **100.0%** | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.001 | -0.349 ± 0.043 | -0.346 ± 0.044 | **100.0%** | 24.0K | 100.0% | **100.0%** | <u>121.8</u> | 93.7% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.01 | -0.351 ± 0.043 | -0.349 ± 0.043 | **100.0%** | 23.3K | 100.0% | **100.0%** | 123.1 | 93.7% | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.01 | -0.355 ± 0.003 | -0.351 ± 0.005 | 98.6% | 36.7K | 100.0% | **100.0%** | 137.4 | 98.6% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.001 | -0.370 ± 0.038 | -0.352 ± 0.033 | **100.0%** | 30.9K | 100.0% | 96.9% | 132.6 | **100.0%** | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | -0.408 ± 0.024 | -0.410 ± 0.027 | 92.0% | 24.9K | 100.0% | **100.0%** | 146.5 | 96.0% | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.001 | -0.410 ± 0.018 | -0.448 ± 0.047 | 86.3% | 19.7K | 100.0% | **100.0%** | 146.5 | 90.2% | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.001 | -0.423 ± 0.011 | -0.508 ± 0.046 | 75.9% | 31.6K | 100.0% | **100.0%** | 162.5 | 86.3% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.01 | -0.465 ± 0.006 | -0.509 ± 0.009 | 85.6% | 24.7K | 100.0% | **100.0%** | 175.0 | 87.6% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.01 | -0.471 ± 0.009 | -0.469 ± 0.009 | 92.5% | 33.4K | 100.0% | **100.0%** | 162.3 | 89.3% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.01 | -0.524 ± 0.015 | -0.562 ± 0.017 | 78.5% | 22.2K | 100.0% | **100.0%** | 191.7 | 84.6% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.001 | -0.612 ± 0.015 | -0.702 ± 0.042 | 61.5% | 53.8K | 100.0% | **100.0%** | 203.3 | 65.8% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.001 | -0.675 ± 0.029 | -0.725 ± 0.038 | 62.9% | 48.6K | 100.0% | **100.0%** | 221.5 | 67.5% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.001 | -0.715 ± 0.033 | -0.798 ± 0.045 | 49.1% | 39.0K | 100.0% | **100.0%** | 228.5 | 50.2% | 16 |

Chosen (highest $W$, ties by $L$): **intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.001**. Place in the sweep: intrinsic weight 0.015625 is inside its 10 swept values; learning rate 0.0003 is inside its 3 swept values; entropy coefficient 0.001 is the smallest of its 2 swept values.

### PointMaze Large, reward 1 on arrival, CleanRL PPO-RND

60 configurations with 16 seeds each, compared at 50,331,648 environment steps per policy. Rows are sorted by $W$; the best value of each column is bold and the second best underlined. For the mean capped first-goal step and the mean steps to the goal the smallest value is the best.

| configuration | whole-curve mean $W$ ↑ ± standard error | final level $L$ ± standard error | final level over peak | mean capped first-goal step | policies that reach the goal | maze cells reached | mean steps to the goal, last 10% | episodes that reach the goal, last 10% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01 | **0.967** ± 0.003 | **0.999** ± 0.000 | **100.0%** | 524.4K | **100.0%** | 98.2% | 286.3 | **99.9%** | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.01 | <u>0.963</u> ± 0.004 | <u>0.998</u> ± 0.001 | **100.0%** | 855.7K | **100.0%** | 95.1% | 287.1 | <u>99.8%</u> | 16 |
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.01 | 0.957 ± 0.004 | 0.980 ± 0.006 | <u>99.8%</u> | 307.9K | **100.0%** | 97.1% | 294.7 | 98.1% | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.01 | 0.951 ± 0.005 | 0.994 ± 0.003 | **100.0%** | 1.01M | **100.0%** | 97.8% | 285.1 | 99.4% | 16 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.01 | 0.941 ± 0.005 | 0.951 ± 0.014 | 96.9% | 567.2K | **100.0%** | 97.4% | 297.2 | 95.2% | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.01 | 0.925 ± 0.006 | 0.969 ± 0.005 | **100.0%** | 481.8K | **100.0%** | 97.6% | 294.7 | 96.9% | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.01 | 0.888 ± 0.060 | 0.937 ± 0.062 | **100.0%** | 4.70M | <u>93.8%</u> | 95.8% | 290.9 | 93.7% | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.01 | 0.848 ± 0.043 | **0.999** ± 0.000 | **100.0%** | 5.30M | **100.0%** | 96.2% | 288.7 | **99.9%** | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.001 | 0.828 ± 0.059 | 0.923 ± 0.062 | 99.2% | 2.15M | **100.0%** | 97.8% | 317.6 | 92.4% | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.001 | 0.823 ± 0.029 | 0.829 ± 0.037 | 88.9% | 607.6K | **100.0%** | 98.1% | 328.2 | 83.1% | 16 |
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.001 | 0.796 ± 0.054 | 0.848 ± 0.085 | 93.9% | 1.01M | **100.0%** | 99.0% | 296.2 | 85.1% | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.01 | 0.751 ± 0.064 | 0.936 ± 0.062 | **100.0%** | 10.56M | <u>93.8%</u> | 93.3% | 292.5 | 93.6% | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.01 | 0.696 ± 0.024 | 0.658 ± 0.048 | 84.1% | 405.3K | **100.0%** | **100.0%** | 335.0 | 65.9% | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.01 | 0.661 ± 0.022 | 0.546 ± 0.036 | 70.3% | <u>260.1K</u> | **100.0%** | **100.0%** | 347.0 | 54.7% | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.01 | 0.630 ± 0.028 | 0.666 ± 0.046 | 89.3% | 997.0K | **100.0%** | <u>99.9%</u> | 334.6 | 66.6% | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 0.596 ± 0.106 | 0.642 ± 0.117 | 97.1% | 13.46M | 75.0% | 91.3% | 303.0 | 64.2% | 16 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 0.596 ± 0.090 | 0.631 ± 0.104 | 87.5% | 4.46M | <u>93.8%</u> | 92.5% | 321.1 | 63.2% | 16 |
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.001 | 0.582 ± 0.070 | 0.531 ± 0.091 | 80.7% | 868.3K | **100.0%** | 99.2% | 337.5 | 53.4% | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.001 | 0.358 ± 0.034 | 0.410 ± 0.064 | 93.2% | 524.6K | **100.0%** | **100.0%** | 341.5 | 41.1% | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.001 | 0.335 ± 0.101 | 0.432 ± 0.126 | **100.0%** | 22.02M | 62.5% | 88.3% | 299.7 | 43.2% | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.01 | 0.229 ± 0.102 | 0.250 ± 0.112 | **100.0%** | 23.60M | 56.2% | 77.4% | <u>284.4</u> | 25.0% | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 0.206 ± 0.046 | 0.242 ± 0.071 | 77.0% | 639.1K | **100.0%** | 99.3% | 352.8 | 24.3% | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.001 | 0.197 ± 0.089 | 0.249 ± 0.111 | 99.7% | 32.34M | 43.8% | 67.7% | 291.8 | 24.9% | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.01 | 0.191 ± 0.087 | 0.250 ± 0.112 | **100.0%** | 39.81M | 25.0% | 71.6% | 290.1 | 25.0% | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.01 | 0.162 ± 0.065 | 0.309 ± 0.118 | **100.0%** | 39.55M | 37.5% | 66.3% | 296.0 | 30.9% | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.001 | 0.161 ± 0.043 | 0.102 ± 0.052 | 35.5% | 300.4K | **100.0%** | **100.0%** | 333.0 | 10.4% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.01 | 0.155 ± 0.024 | 0.167 ± 0.038 | 74.2% | 839.7K | **100.0%** | **100.0%** | 359.1 | 16.7% | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.001 | 0.133 ± 0.073 | 0.185 ± 0.099 | **100.0%** | 26.37M | 56.2% | 74.6% | 318.6 | 18.5% | 16 |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.01 | 0.118 ± 0.080 | 0.125 ± 0.085 | **100.0%** | 41.33M | 18.8% | 57.9% | 288.1 | 12.5% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.01 | 0.117 ± 0.022 | 0.093 ± 0.026 | 54.7% | 496.0K | **100.0%** | **100.0%** | 367.8 | 9.3% | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.001 | 0.085 ± 0.060 | 0.119 ± 0.081 | 95.6% | 37.29M | 37.5% | 68.8% | 300.5 | 11.9% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.001 | 0.061 ± 0.061 | 0.062 ± 0.062 | **100.0%** | 47.22M | 6.2% | 29.5% | 286.0 | 6.2% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.001 | 0.060 ± 0.060 | 0.062 ± 0.062 | 98.8% | 47.19M | 6.2% | 22.8% | 288.1 | 6.2% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.001 | 0.052 ± 0.010 | 0.066 ± 0.010 | 74.1% | 1.92M | **100.0%** | **100.0%** | 359.0 | 6.6% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.01 | 0.036 ± 0.010 | 0.021 ± 0.009 | 23.2% | **167.6K** | **100.0%** | **100.0%** | 373.2 | 2.1% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.001 | 0.030 ± 0.009 | 0.012 ± 0.005 | 26.4% | 420.0K | **100.0%** | **100.0%** | 349.0 | 1.2% | 16 |
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.01 | 0.029 ± 0.029 | 0.062 ± 0.062 | **100.0%** | 46.14M | 12.5% | 45.7% | **282.7** | 6.2% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.001 | 0.028 ± 0.007 | 0.031 ± 0.013 | 71.3% | 2.76M | **100.0%** | 98.8% | 371.3 | 3.1% | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.001 | 0.024 ± 0.024 | 0.062 ± 0.062 | **100.0%** | 42.67M | 18.8% | 44.6% | 333.0 | 6.2% | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.001 | 0.015 ± 0.015 | 0.062 ± 0.062 | **100.0%** | 49.37M | 6.2% | 35.7% | 329.3 | 6.2% | 16 |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.001 | 0.014 ± 0.014 | 0.027 ± 0.027 | 51.4% | 45.92M | 12.5% | 40.8% | 383.2 | 2.8% | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.01 | 0.011 ± 0.011 | 0.059 ± 0.059 | **100.0%** | 47.19M | 6.2% | 30.8% | 320.7 | 5.9% | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.003 ± 0.003 | **100.0%** | 45.40M | 18.8% | 39.9% | 391.0 | 0.3% | 16 |
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0% | 47.26M | 6.2% | 29.2% | none | 0.0% | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0% | 47.19M | 6.2% | 24.0% | none | 0.0% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0% | 47.19M | 6.2% | 25.3% | none | 0.0% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0% | 47.19M | 6.2% | 29.3% | none | 0.0% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0% | 47.19M | 6.2% | 32.7% | none | 0.0% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 50.33M | 0.0% | 26.2% | none | 0.0% | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 50.33M | 0.0% | 35.6% | none | 0.0% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 50.33M | 0.0% | 18.8% | none | 0.0% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 50.33M | 0.0% | 22.8% | none | 0.0% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 50.33M | 0.0% | 21.5% | none | 0.0% | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 50.33M | 0.0% | 37.8% | none | 0.0% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 50.33M | 0.0% | 20.1% | none | 0.0% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 50.33M | 0.0% | 37.1% | none | 0.0% | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 50.33M | 0.0% | 43.1% | none | 0.0% | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 50.33M | 0.0% | 45.9% | none | 0.0% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 50.33M | 0.0% | 26.4% | none | 0.0% | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 50.33M | 0.0% | 49.7% | none | 0.0% | 16 |

Chosen (highest $W$, ties by $L$): **intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01**. Place in the sweep: intrinsic weight 1 is inside its 10 swept values; learning rate 0.001 is the largest of its 3 swept values; entropy coefficient 0.01 is the largest of its 2 swept values.

### PointMaze Large, step cost, CleanRL PPO-RND

60 configurations with 16 seeds each, compared at 50,331,648 environment steps per policy. Rows are sorted by $W$; the best value of each column is bold and the second best underlined. For the mean capped first-goal step and the mean steps to the goal the smallest value is the best.

| configuration | whole-curve mean $W$ ↑ ± standard error | final level $L$ ± standard error | final level over peak | mean capped first-goal step | policies that reach the goal | maze cells reached | mean steps to the goal, last 10% | episodes that reach the goal, last 10% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.01 | **-0.717** ± 0.002 | **-0.706** ± 0.004 | 99.8% | 232.3K | **100.0%** | 98.5% | **282.8** | 99.8% | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.01 | <u>-0.734</u> ± 0.002 | <u>-0.716</u> ± 0.003 | 99.5% | 629.1K | **100.0%** | 98.6% | <u>286.4</u> | 99.1% | 16 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.01 | -0.736 ± 0.011 | -0.717 ± 0.011 | **100.0%** | 298.7K | **100.0%** | 98.9% | <u>286.4</u> | 99.6% | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.01 | -0.742 ± 0.014 | -0.725 ± 0.014 | **100.0%** | 364.0K | **100.0%** | 97.7% | 289.8 | <u>99.9%</u> | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.01 | -0.748 ± 0.015 | -0.724 ± 0.016 | **100.0%** | 1.49M | **100.0%** | 96.5% | 288.8 | **100.0%** | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.01 | -0.752 ± 0.018 | -0.735 ± 0.019 | **100.0%** | 1.13M | **100.0%** | 95.8% | 292.5 | **100.0%** | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.01 | -0.754 ± 0.006 | -0.753 ± 0.013 | 94.5% | **122.9K** | **100.0%** | 97.6% | 295.9 | 95.2% | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.01 | -0.764 ± 0.006 | -0.757 ± 0.009 | 94.6% | 401.1K | **100.0%** | 98.5% | 299.9 | 96.5% | 16 |
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01 | -0.768 ± 0.021 | -0.753 ± 0.021 | **100.0%** | 506.4K | **100.0%** | 97.8% | 299.0 | <u>99.9%</u> | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.01 | -0.772 ± 0.021 | -0.757 ± 0.021 | **100.0%** | 1.04M | **100.0%** | 97.7% | 300.4 | <u>99.9%</u> | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.01 | -0.774 ± 0.014 | -0.750 ± 0.015 | 98.2% | 1.66M | **100.0%** | 97.1% | 298.7 | 99.3% | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.01 | -0.782 ± 0.015 | -0.737 ± 0.015 | **100.0%** | 6.49M | **100.0%** | 96.5% | 294.3 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | -0.783 ± 0.010 | -0.796 ± 0.021 | 82.7% | 535.4K | **100.0%** | 99.0% | 307.9 | 89.5% | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.01 | -0.803 ± 0.006 | -0.810 ± 0.017 | 83.0% | 794.8K | **100.0%** | <u>99.9%</u> | 309.3 | 83.7% | 16 |
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.001 | -0.804 ± 0.014 | -0.798 ± 0.019 | 92.8% | 530.6K | **100.0%** | 98.5% | 312.7 | 93.9% | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.01 | -0.808 ± 0.032 | -0.790 ± 0.034 | **100.0%** | 14.63M | 75.0% | 91.0% | 287.6 | 75.0% | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.001 | -0.813 ± 0.024 | -0.790 ± 0.025 | 99.6% | 4.42M | **100.0%** | 98.4% | 312.4 | 99.7% | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.01 | -0.814 ± 0.019 | -0.758 ± 0.016 | **100.0%** | 7.38M | **100.0%** | 96.2% | 302.1 | <u>99.9%</u> | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.01 | -0.818 ± 0.034 | -0.806 ± 0.036 | **100.0%** | 13.65M | 75.0% | 85.2% | 286.6 | 68.7% | 16 |
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.001 | -0.819 ± 0.023 | -0.793 ± 0.026 | 99.8% | 3.45M | <u>93.8%</u> | 93.5% | 303.9 | 88.4% | 16 |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.01 | -0.820 ± 0.030 | -0.806 ± 0.032 | **100.0%** | 10.64M | 81.2% | 87.5% | 301.8 | 81.2% | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | -0.822 ± 0.024 | -0.816 ± 0.029 | 93.6% | 4.57M | <u>93.8%</u> | 98.4% | 308.0 | 81.2% | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.001 | -0.822 ± 0.021 | -0.838 ± 0.022 | 84.2% | 677.3K | **100.0%** | 99.0% | 320.3 | 83.1% | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.01 | -0.835 ± 0.035 | -0.825 ± 0.037 | **100.0%** | 19.32M | 62.5% | 81.8% | 287.6 | 62.5% | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.001 | -0.851 ± 0.028 | -0.832 ± 0.031 | 99.3% | 13.20M | 75.0% | 87.0% | 300.7 | 67.9% | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.001 | -0.865 ± 0.027 | -0.828 ± 0.031 | **100.0%** | 16.64M | 81.2% | 90.2% | 310.1 | 78.8% | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.001 | -0.872 ± 0.009 | -0.900 ± 0.016 | 58.5% | 958.8K | **100.0%** | 99.5% | 338.6 | 65.3% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.01 | -0.873 ± 0.032 | -0.860 ± 0.035 | **100.0%** | 23.11M | 56.2% | 65.8% | 298.7 | 56.2% | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.001 | -0.878 ± 0.029 | -0.870 ± 0.034 | 91.2% | 15.20M | 75.0% | 89.5% | 311.9 | 61.8% | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.001 | -0.878 ± 0.024 | -0.855 ± 0.030 | 95.2% | 5.10M | <u>93.8%</u> | 98.9% | 313.1 | 68.3% | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.001 | -0.882 ± 0.029 | -0.859 ± 0.033 | 98.5% | 18.10M | 75.0% | 89.0% | 310.6 | 65.6% | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.001 | -0.890 ± 0.031 | -0.877 ± 0.034 | 98.2% | 17.98M | 68.8% | 80.4% | 300.1 | 49.9% | 16 |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.001 | -0.890 ± 0.027 | -0.873 ± 0.031 | **100.0%** | 16.27M | 68.8% | 92.0% | 315.0 | 62.0% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.01 | -0.895 ± 0.030 | -0.882 ± 0.032 | **100.0%** | 23.55M | 56.2% | 69.8% | 312.8 | 56.2% | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.001 | -0.903 ± 0.013 | -0.931 ± 0.019 | 55.9% | <u>169.3K</u> | **100.0%** | 98.6% | 328.0 | 39.1% | 16 |
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.01 | -0.904 ± 0.029 | -0.889 ± 0.031 | **100.0%** | 25.15M | 56.2% | 68.8% | 317.5 | 56.2% | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.01 | -0.908 ± 0.023 | -0.863 ± 0.030 | <u>99.9%</u> | 23.56M | 68.8% | 79.6% | 317.4 | 68.5% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.01 | -0.918 ± 0.027 | -0.910 ± 0.028 | **100.0%** | 23.63M | 56.2% | 64.9% | 324.8 | 49.9% | 16 |
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.001 | -0.919 ± 0.022 | -0.905 ± 0.025 | 98.6% | 21.83M | 62.5% | 71.1% | 335.6 | 61.2% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.01 | -0.919 ± 0.006 | -0.933 ± 0.009 | 68.3% | 393.6K | **100.0%** | **100.0%** | 348.4 | 51.4% | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.001 | -0.921 ± 0.025 | -0.903 ± 0.028 | 98.9% | 23.35M | 62.5% | 86.8% | 324.6 | 53.4% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.001 | -0.925 ± 0.028 | -0.917 ± 0.030 | **100.0%** | 29.64M | 43.8% | 55.8% | 310.3 | 37.5% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.01 | -0.930 ± 0.026 | -0.917 ± 0.030 | **100.0%** | 33.42M | 37.5% | 57.1% | 309.9 | 37.5% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.01 | -0.934 ± 0.007 | -0.945 ± 0.009 | 63.4% | 1.12M | **100.0%** | **100.0%** | 345.8 | 40.5% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.001 | -0.938 ± 0.027 | -0.932 ± 0.029 | 99.8% | 29.75M | 43.8% | 61.3% | 322.8 | 37.5% | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | -0.938 ± 0.010 | -0.969 ± 0.012 | 24.8% | 3.40M | <u>93.8%</u> | 96.9% | 359.0 | 30.5% | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.01 | -0.941 ± 0.027 | -0.933 ± 0.029 | **100.0%** | 36.28M | 31.2% | 48.5% | 311.2 | 31.2% | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | -0.942 ± 0.023 | -0.939 ± 0.024 | 95.6% | 26.61M | 50.0% | 76.4% | 340.6 | 43.6% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.001 | -0.944 ± 0.024 | -0.936 ± 0.025 | **100.0%** | 30.80M | 43.8% | 59.2% | 328.4 | 37.1% | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.001 | -0.945 ± 0.022 | -0.939 ± 0.024 | **100.0%** | 27.71M | 50.0% | 60.3% | 339.9 | 43.1% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.001 | -0.947 ± 0.027 | -0.946 ± 0.029 | 90.2% | 32.91M | 37.5% | 58.0% | <u>286.4</u> | 18.8% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.001 | -0.947 ± 0.024 | -0.943 ± 0.025 | 97.8% | 32.89M | 37.5% | 57.2% | 331.6 | 35.0% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.01 | -0.948 ± 0.025 | -0.940 ± 0.028 | **100.0%** | 38.33M | 31.2% | 50.3% | 302.1 | 25.0% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.001 | -0.955 ± 0.025 | -0.955 ± 0.025 | 97.7% | 40.94M | 18.8% | 66.0% | 302.2 | 18.7% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.01 | -0.960 ± 0.004 | -0.976 ± 0.003 | 29.8% | 342.0K | **100.0%** | 99.7% | 365.4 | 26.9% | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.001 | -0.961 ± 0.020 | -0.944 ± 0.027 | <u>99.9%</u> | 38.00M | 25.0% | 50.8% | 308.8 | 25.0% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.01 | -0.966 ± 0.020 | -0.964 ± 0.021 | **100.0%** | 37.92M | 25.0% | 45.7% | 321.3 | 18.7% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.001 | -0.969 ± 0.003 | -0.974 ± 0.004 | 58.1% | 1.81M | **100.0%** | **100.0%** | 353.7 | 21.8% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.001 | -0.984 ± 0.003 | -0.989 ± 0.004 | 36.5% | 1.23M | **100.0%** | **100.0%** | 366.2 | 12.8% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.001 | -0.989 ± 0.003 | -0.997 ± 0.001 | 13.8% | 343.2K | **100.0%** | **100.0%** | 359.9 | 3.2% | 16 |

Chosen (highest $W$, ties by $L$): **intrinsic weight 4, learning rate 0.001, entropy coefficient 0.01**. Place in the sweep: intrinsic weight 4 is inside its 10 swept values; learning rate 0.001 is the largest of its 3 swept values; entropy coefficient 0.01 is the largest of its 2 swept values.

## Budget

Define $\bar R$, $W$, $L$ and $R_0$ as in the section Measures, $N$ as the steps per policy a sweep was trained for, and $S_5(s)$ as the mean of $\bar R$ over the $0.05 N$ steps that end at step $s$. The rule reads the three CleanRL PPO-RND configurations with the highest $W(N)$ on the development seeds. For each, define $t$ as the first logged step $s \ge 0.05 N$ with $S_5(s) - R_0 \ge 0.95 \thinspace (L(N) - R_0)$. The proposed budget is twice the largest $t$, rounded up to a power of two, and never above $N$. A configuration that never reaches 95% of its final level counts as $t = N$.

### PointMaze Medium, reward 1 on arrival

The sweep was trained for 50,331,648 steps per policy with 16 seeds per configuration.

| configuration | whole-curve mean $W$ ± standard error | final level $L$ | 95% of the final level, measured from $R_0$ | first step at which the smoothed curve reaches it |
|---|---|---|---|---|
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01 | 0.998 ± 0.001 | 1.000 | 0.950 | 2,621,440 (the earliest step the rule can read) |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.01 | 0.997 ± 0.001 | 1.000 | 0.950 | 2,621,440 (the earliest step the rule can read) |
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.001 | 0.995 ± 0.001 | 1.000 | 0.950 | 2,621,440 (the earliest step the rule can read) |

Proposed budget: **8,388,608 steps per policy**: twice the largest step count is 5,242,880, rounded up to a power of two. Every step count is the earliest the rule can read from this run (2,621,440: one full range of 5% of its steps), so the curves rise faster than this run can show, and a shorter run would give a smaller budget.

The measures of every configuration that trained the whole length, recomputed at 8,388,608 steps and sorted by $W$. The runs were trained for 50,331,648 steps, so their learning rate has not reached zero at the budget.

| configuration | whole-curve mean $W$ ↑ ± standard error | final level $L$ ± standard error | final level over peak | mean capped first-goal step | policies that reach the goal | maze cells reached | mean steps to the goal, last 10% | episodes that reach the goal, last 10% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.01 | **0.987** ± 0.002 | 0.995 ± 0.001 | 99.8% | 19.5K | **100.0%** | 98.3% | 131.7 | 99.5% | 16 |
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01 | <u>0.986</u> ± 0.004 | **1.000** ± 0.000 | **100.0%** | 76.2K | **100.0%** | 94.5% | 125.3 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.01 | 0.984 ± 0.003 | <u>0.997</u> ± 0.001 | <u>99.9%</u> | 53.2K | **100.0%** | 96.4% | 128.0 | <u>99.7%</u> | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.01 | 0.982 ± 0.007 | **1.000** ± 0.000 | **100.0%** | 35.1K | **100.0%** | 97.6% | 125.6 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.001 | 0.979 ± 0.006 | **1.000** ± 0.000 | **100.0%** | 90.4K | **100.0%** | 96.9% | 127.4 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.01 | 0.975 ± 0.004 | 0.994 ± 0.002 | 99.5% | 37.1K | **100.0%** | 98.6% | 133.5 | 99.4% | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.001 | 0.974 ± 0.007 | **1.000** ± 0.000 | **100.0%** | 91.0K | **100.0%** | 99.3% | 124.9 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.001 | 0.972 ± 0.007 | 0.995 ± 0.002 | 99.6% | 73.6K | **100.0%** | <u>99.8%</u> | 134.2 | 99.6% | 16 |
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.001 | 0.969 ± 0.005 | 0.983 ± 0.004 | 99.4% | 41.2K | **100.0%** | **100.0%** | 136.8 | 98.4% | 16 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 0.952 ± 0.021 | 0.987 ± 0.007 | 99.5% | <u>18.3K</u> | **100.0%** | 98.3% | 136.9 | 98.8% | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.01 | 0.951 ± 0.012 | **1.000** ± 0.000 | **100.0%** | 320.7K | **100.0%** | 95.2% | 124.7 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.01 | 0.929 ± 0.024 | **1.000** ± 0.000 | **100.0%** | 174.0K | **100.0%** | <u>99.8%</u> | 125.1 | **100.0%** | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.01 | 0.925 ± 0.007 | 0.949 ± 0.008 | 98.5% | 24.9K | **100.0%** | **100.0%** | 146.4 | 95.0% | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 0.924 ± 0.062 | 0.937 ± 0.062 | **100.0%** | 83.6K | **100.0%** | 96.9% | 126.4 | 93.7% | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.01 | 0.902 ± 0.005 | 0.965 ± 0.002 | **100.0%** | 39.7K | **100.0%** | **100.0%** | 145.8 | 96.5% | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.01 | 0.891 ± 0.033 | **1.000** ± 0.000 | **100.0%** | 689.9K | **100.0%** | 97.1% | 126.3 | **100.0%** | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.01 | 0.882 ± 0.015 | 0.949 ± 0.010 | **100.0%** | 36.4K | **100.0%** | **100.0%** | 158.9 | 95.0% | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.01 | 0.863 ± 0.035 | **1.000** ± 0.000 | **100.0%** | 789.0K | **100.0%** | 96.4% | 123.1 | **100.0%** | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 0.848 ± 0.037 | 0.834 ± 0.071 | 88.0% | 55.0K | **100.0%** | **100.0%** | 167.0 | 84.1% | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.001 | 0.847 ± 0.064 | 0.938 ± 0.061 | **100.0%** | 814.2K | **100.0%** | 96.2% | 134.0 | 93.8% | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.001 | 0.835 ± 0.026 | 0.876 ± 0.036 | 97.7% | 26.5K | **100.0%** | **100.0%** | 158.5 | 88.2% | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.001 | 0.816 ± 0.033 | 0.841 ± 0.042 | 91.9% | 37.8K | **100.0%** | <u>99.8%</u> | 175.7 | 85.3% | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.01 | 0.795 ± 0.078 | 0.934 ± 0.062 | **100.0%** | 1.68M | **100.0%** | 92.3% | 123.7 | 93.6% | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.001 | 0.753 ± 0.068 | 0.965 ± 0.035 | **100.0%** | 1.50M | **100.0%** | 95.7% | 127.1 | 97.1% | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.001 | 0.748 ± 0.081 | 0.979 ± 0.019 | **100.0%** | 1.46M | **100.0%** | 97.8% | 133.1 | 98.0% | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.01 | 0.727 ± 0.101 | 0.812 ± 0.101 | **100.0%** | 1.53M | 87.5% | 87.5% | 124.2 | 81.2% | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.01 | 0.710 ± 0.090 | 0.875 ± 0.085 | **100.0%** | 1.65M | 87.5% | 97.4% | 124.5 | 87.5% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.01 | 0.655 ± 0.013 | 0.676 ± 0.023 | 91.1% | **17.7K** | **100.0%** | **100.0%** | 202.8 | 67.7% | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.01 | 0.634 ± 0.113 | 0.688 ± 0.120 | **100.0%** | 2.54M | 75.0% | 87.7% | 123.1 | 68.8% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.01 | 0.612 ± 0.012 | 0.743 ± 0.018 | 96.9% | 26.9K | **100.0%** | **100.0%** | 182.8 | 74.5% | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.001 | 0.565 ± 0.111 | 0.688 ± 0.120 | **100.0%** | 2.31M | 87.5% | 88.5% | 125.5 | 68.8% | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.01 | 0.550 ± 0.102 | 0.812 ± 0.101 | **100.0%** | 2.60M | <u>93.8%</u> | 96.6% | 124.2 | 81.2% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.01 | 0.543 ± 0.026 | 0.662 ± 0.039 | 93.2% | 44.4K | **100.0%** | **100.0%** | 196.1 | 66.8% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.001 | 0.530 ± 0.036 | 0.700 ± 0.043 | 97.0% | 65.8K | **100.0%** | **100.0%** | 183.8 | 71.6% | 16 |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.01 | 0.528 ± 0.099 | 0.756 ± 0.109 | **100.0%** | 2.82M | 87.5% | 91.6% | 125.2 | 75.6% | 16 |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.001 | 0.504 ± 0.117 | 0.625 ± 0.124 | **100.0%** | 1.69M | <u>93.8%</u> | 89.2% | 129.3 | 62.5% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.001 | 0.469 ± 0.042 | 0.522 ± 0.066 | 91.2% | 32.3K | **100.0%** | **100.0%** | 198.1 | 52.8% | 16 |
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.001 | 0.460 ± 0.124 | 0.500 ± 0.129 | **100.0%** | 4.00M | 56.2% | 85.3% | 123.3 | 50.0% | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.001 | 0.457 ± 0.116 | 0.562 ± 0.128 | **100.0%** | 2.78M | 75.0% | 83.7% | 128.6 | 56.2% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.001 | 0.439 ± 0.033 | 0.531 ± 0.061 | 93.2% | 54.3K | **100.0%** | **100.0%** | 179.0 | 54.6% | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.001 | 0.438 ± 0.119 | 0.557 ± 0.127 | **100.0%** | 2.59M | 81.2% | 96.4% | 129.7 | 55.7% | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.01 | 0.410 ± 0.121 | 0.466 ± 0.125 | **100.0%** | 3.22M | 68.8% | 88.5% | 125.2 | 47.8% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.001 | 0.409 ± 0.121 | 0.438 ± 0.128 | **100.0%** | 3.96M | 56.2% | 83.9% | 128.7 | 43.8% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.001 | 0.398 ± 0.119 | 0.438 ± 0.128 | **100.0%** | 4.49M | 50.0% | 74.8% | 125.2 | 43.8% | 16 |
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.01 | 0.386 ± 0.119 | 0.438 ± 0.128 | **100.0%** | 3.48M | 62.5% | 86.8% | 123.5 | 43.8% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.01 | 0.385 ± 0.120 | 0.437 ± 0.128 | **100.0%** | 4.61M | 50.0% | 76.0% | 125.5 | 43.7% | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 0.378 ± 0.114 | 0.454 ± 0.125 | **100.0%** | 4.12M | 62.5% | 84.9% | 128.2 | 45.5% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.001 | 0.374 ± 0.125 | 0.377 ± 0.125 | **100.0%** | 4.23M | 56.2% | 78.4% | 126.5 | 37.7% | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.001 | 0.355 ± 0.119 | 0.375 ± 0.125 | **100.0%** | 4.49M | 50.0% | 83.9% | 125.7 | 37.5% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.01 | 0.339 ± 0.115 | 0.375 ± 0.125 | **100.0%** | 3.45M | 62.5% | 81.5% | 125.1 | 37.5% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.01 | 0.311 ± 0.119 | 0.312 ± 0.120 | **100.0%** | 5.25M | 37.5% | 79.3% | 123.1 | 31.2% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.001 | 0.309 ± 0.118 | 0.312 ± 0.120 | 99.4% | 3.61M | 62.5% | 83.7% | **122.1** | 31.2% | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.001 | 0.308 ± 0.118 | 0.312 ± 0.120 | **100.0%** | 3.16M | 62.5% | 83.4% | 125.2 | 31.2% | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.001 | 0.306 ± 0.112 | 0.417 ± 0.123 | **100.0%** | 4.25M | 56.2% | 82.7% | 140.9 | 42.2% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.001 | 0.301 ± 0.115 | 0.312 ± 0.120 | **100.0%** | 4.82M | 43.8% | 77.9% | 123.2 | 31.2% | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.01 | 0.296 ± 0.114 | 0.312 ± 0.120 | **100.0%** | 3.69M | 56.2% | 80.3% | 127.5 | 31.2% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.01 | 0.281 ± 0.111 | 0.312 ± 0.120 | <u>99.9%</u> | 4.97M | 43.8% | 80.3% | 132.2 | 31.2% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.001 | 0.249 ± 0.111 | 0.250 ± 0.112 | **100.0%** | 4.63M | 50.0% | 79.3% | 123.4 | 25.0% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.01 | 0.241 ± 0.108 | 0.250 ± 0.112 | **100.0%** | 3.87M | 56.2% | 83.4% | <u>122.5</u> | 25.0% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.01 | 0.234 ± 0.105 | 0.250 ± 0.112 | **100.0%** | 2.76M | 68.8% | 79.3% | 124.2 | 25.0% | 16 |

At the budget the highest $W$ belongs to **intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.01**, while at the run's length it belongs to **intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01**: the choice changes. At the budget their $W$ are 0.987 ± 0.002 and 0.986 ± 0.004.

### PointMaze Medium, step cost

The sweep was trained for 50,331,648 steps per policy with 16 seeds per configuration.

| configuration | whole-curve mean $W$ ± standard error | final level $L$ | 95% of the final level, measured from $R_0$ | first step at which the smoothed curve reaches it |
|---|---|---|---|---|
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.001 | -0.303 ± 0.001 | -0.302 | -0.337 | 2,621,440 (the earliest step the rule can read) |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.01 | -0.304 ± 0.001 | -0.301 | -0.336 | 2,621,440 (the earliest step the rule can read) |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.01 | -0.304 ± 0.001 | -0.302 | -0.337 | 2,621,440 (the earliest step the rule can read) |

Proposed budget: **8,388,608 steps per policy**: twice the largest step count is 5,242,880, rounded up to a power of two. Every step count is the earliest the rule can read from this run (2,621,440: one full range of 5% of its steps), so the curves rise faster than this run can show, and a shorter run would give a smaller budget.

The measures of every configuration that trained the whole length, recomputed at 8,388,608 steps and sorted by $W$. The runs were trained for 50,331,648 steps, so their learning rate has not reached zero at the budget.

| configuration | whole-curve mean $W$ ↑ ± standard error | final level $L$ ± standard error | final level over peak | mean capped first-goal step | policies that reach the goal | maze cells reached | mean steps to the goal, last 10% | episodes that reach the goal, last 10% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.01 | **-0.308** ± 0.001 | <u>-0.304</u> ± 0.001 | **100.0%** | 18.9K | 100.0% | 95.9% | 122.8 | **100.0%** | 16 |
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.001 | **-0.308** ± 0.002 | **-0.303** ± 0.001 | **100.0%** | 20.7K | 100.0% | 98.6% | **122.0** | **100.0%** | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.001 | **-0.308** ± 0.002 | -0.307 ± 0.002 | 99.7% | **13.8K** | 100.0% | 95.7% | 123.6 | **100.0%** | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.01 | <u>-0.310</u> ± 0.002 | -0.306 ± 0.001 | **100.0%** | <u>14.1K</u> | 100.0% | 94.0% | 123.4 | **100.0%** | 16 |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.001 | <u>-0.310</u> ± 0.002 | <u>-0.304</u> ± 0.001 | **100.0%** | 14.5K | 100.0% | 93.5% | 122.5 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.01 | <u>-0.310</u> ± 0.001 | -0.306 ± 0.001 | **100.0%** | 21.7K | 100.0% | 99.3% | 123.2 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.001 | -0.311 ± 0.001 | -0.309 ± 0.002 | 99.7% | 15.4K | 100.0% | 97.6% | 124.3 | **100.0%** | 16 |
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.01 | -0.311 ± 0.003 | -0.305 ± 0.001 | **100.0%** | 23.1K | 100.0% | 95.7% | 122.9 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | -0.311 ± 0.003 | <u>-0.304</u> ± 0.001 | **100.0%** | 15.6K | 100.0% | 99.0% | 122.7 | **100.0%** | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.01 | -0.312 ± 0.002 | -0.307 ± 0.001 | **100.0%** | 17.5K | 100.0% | 97.4% | 123.9 | **100.0%** | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.01 | -0.312 ± 0.001 | -0.307 ± 0.001 | **100.0%** | 22.6K | 100.0% | 98.1% | 123.7 | **100.0%** | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.001 | -0.312 ± 0.003 | -0.307 ± 0.003 | **100.0%** | 15.9K | 100.0% | 92.8% | 123.6 | **100.0%** | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.01 | -0.312 ± 0.002 | -0.308 ± 0.002 | **100.0%** | 16.2K | 100.0% | 94.0% | 124.1 | **100.0%** | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.01 | -0.312 ± 0.004 | -0.309 ± 0.003 | **100.0%** | 17.3K | 100.0% | 90.6% | 124.3 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.01 | -0.313 ± 0.001 | -0.305 ± 0.000 | **100.0%** | 17.9K | 100.0% | 99.5% | 123.0 | **100.0%** | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.001 | -0.313 ± 0.003 | -0.308 ± 0.002 | <u>99.9%</u> | 16.7K | 100.0% | 94.7% | 124.2 | **100.0%** | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.01 | -0.313 ± 0.003 | -0.305 ± 0.001 | **100.0%** | 28.9K | 100.0% | 97.8% | 122.9 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.001 | -0.313 ± 0.001 | <u>-0.304</u> ± 0.001 | **100.0%** | 22.0K | 100.0% | **100.0%** | 122.7 | **100.0%** | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.001 | -0.314 ± 0.005 | **-0.303** ± 0.001 | **100.0%** | 27.0K | 100.0% | 97.4% | **122.0** | **100.0%** | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.001 | -0.314 ± 0.003 | -0.307 ± 0.002 | **100.0%** | 25.4K | 100.0% | 98.8% | 123.8 | **100.0%** | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.001 | -0.314 ± 0.005 | -0.306 ± 0.002 | **100.0%** | 14.9K | 100.0% | 97.4% | 123.4 | **100.0%** | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.01 | -0.314 ± 0.001 | -0.306 ± 0.001 | **100.0%** | 17.7K | 100.0% | 99.0% | 123.3 | **100.0%** | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.001 | -0.314 ± 0.002 | <u>-0.304</u> ± 0.001 | **100.0%** | 23.3K | 100.0% | **100.0%** | 122.7 | **100.0%** | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.001 | -0.315 ± 0.003 | **-0.303** ± 0.000 | **100.0%** | 23.6K | 100.0% | 99.3% | 122.4 | **100.0%** | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.01 | -0.315 ± 0.003 | -0.306 ± 0.001 | **100.0%** | 21.4K | 100.0% | 99.5% | 123.3 | **100.0%** | 16 |
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01 | -0.317 ± 0.004 | -0.309 ± 0.003 | **100.0%** | 16.4K | 100.0% | 98.3% | 124.6 | **100.0%** | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.001 | -0.318 ± 0.004 | -0.305 ± 0.001 | **100.0%** | 23.1K | 100.0% | 99.0% | 123.2 | **100.0%** | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.01 | -0.320 ± 0.008 | -0.305 ± 0.001 | **100.0%** | 24.0K | 100.0% | 99.5% | 122.8 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.01 | -0.320 ± 0.003 | -0.311 ± 0.001 | 99.8% | 16.3K | 100.0% | 98.6% | 125.1 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.001 | -0.321 ± 0.003 | -0.321 ± 0.005 | 98.8% | 15.1K | 100.0% | 98.8% | 127.8 | 99.7% | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.01 | -0.322 ± 0.001 | -0.316 ± 0.002 | 99.1% | 19.0K | 100.0% | **100.0%** | 127.1 | <u>99.9%</u> | 16 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.01 | -0.326 ± 0.005 | -0.317 ± 0.004 | 99.8% | 18.5K | 100.0% | 99.5% | 127.4 | <u>99.9%</u> | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.01 | -0.328 ± 0.019 | -0.323 ± 0.017 | **100.0%** | 14.2K | 100.0% | 95.9% | 127.1 | **100.0%** | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.01 | -0.329 ± 0.010 | -0.310 ± 0.003 | **100.0%** | 15.3K | 100.0% | 97.1% | 124.9 | **100.0%** | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.01 | -0.332 ± 0.022 | -0.323 ± 0.017 | **100.0%** | 17.4K | 100.0% | 98.3% | 127.1 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.001 | -0.332 ± 0.010 | -0.313 ± 0.003 | **100.0%** | 25.0K | 100.0% | **100.0%** | 125.8 | **100.0%** | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.001 | -0.333 ± 0.020 | -0.321 ± 0.017 | <u>99.9%</u> | 20.7K | 100.0% | <u>99.8%</u> | 126.3 | **100.0%** | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.01 | -0.335 ± 0.015 | -0.307 ± 0.002 | **100.0%** | 30.7K | 100.0% | 99.5% | 123.5 | **100.0%** | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.001 | -0.341 ± 0.030 | <u>-0.304</u> ± 0.001 | **100.0%** | 23.3K | 100.0% | <u>99.8%</u> | 122.7 | **100.0%** | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.01 | -0.345 ± 0.026 | -0.322 ± 0.017 | **100.0%** | 18.9K | 100.0% | **100.0%** | 126.6 | **100.0%** | 16 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | -0.345 ± 0.023 | -0.322 ± 0.014 | <u>99.9%</u> | 17.0K | 100.0% | 99.5% | 127.4 | <u>99.9%</u> | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.01 | -0.349 ± 0.003 | -0.331 ± 0.003 | 99.5% | 24.3K | 100.0% | <u>99.8%</u> | 131.3 | 99.3% | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.001 | -0.353 ± 0.043 | -0.348 ± 0.043 | <u>99.9%</u> | 19.9K | 100.0% | **100.0%** | 122.8 | 93.8% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.001 | -0.354 ± 0.043 | -0.348 ± 0.044 | <u>99.9%</u> | 24.0K | 100.0% | **100.0%** | <u>122.3</u> | 93.6% | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | -0.355 ± 0.043 | -0.349 ± 0.043 | <u>99.9%</u> | 19.2K | 100.0% | 98.1% | 123.1 | 93.7% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.01 | -0.361 ± 0.043 | -0.350 ± 0.043 | **100.0%** | 23.3K | 100.0% | **100.0%** | 123.7 | 93.7% | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.001 | -0.365 ± 0.043 | -0.355 ± 0.036 | **100.0%** | 18.2K | 100.0% | 93.5% | 131.8 | 99.5% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.01 | -0.372 ± 0.042 | -0.308 ± 0.002 | **100.0%** | 193.6K | 100.0% | 98.6% | 124.3 | **100.0%** | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.01 | -0.375 ± 0.008 | -0.344 ± 0.005 | 99.8% | 29.9K | 100.0% | **100.0%** | 136.0 | 99.2% | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.001 | -0.382 ± 0.009 | -0.351 ± 0.008 | 99.8% | 19.7K | 100.0% | 98.6% | 134.5 | 97.9% | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | -0.385 ± 0.014 | -0.370 ± 0.028 | 95.6% | 24.9K | 100.0% | **100.0%** | 139.7 | 98.6% | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.001 | -0.389 ± 0.011 | -0.361 ± 0.012 | 97.6% | 31.6K | 100.0% | **100.0%** | 140.7 | 99.1% | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.01 | -0.390 ± 0.009 | -0.353 ± 0.005 | 99.2% | 36.7K | 100.0% | **100.0%** | 137.9 | 98.6% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.001 | -0.427 ± 0.061 | -0.394 ± 0.053 | **100.0%** | 30.9K | 100.0% | 96.9% | 132.0 | 93.6% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.01 | -0.477 ± 0.008 | -0.467 ± 0.016 | 93.7% | 22.2K | 100.0% | **100.0%** | 162.2 | 90.3% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.01 | -0.494 ± 0.006 | -0.425 ± 0.008 | 98.8% | 24.7K | 100.0% | **100.0%** | 155.0 | 93.8% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.01 | -0.585 ± 0.021 | -0.471 ± 0.025 | 96.7% | 33.4K | 100.0% | **100.0%** | 157.7 | 89.4% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.001 | -0.596 ± 0.032 | -0.541 ± 0.035 | **100.0%** | 48.6K | 100.0% | **100.0%** | 178.2 | 86.7% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.001 | -0.620 ± 0.023 | -0.579 ± 0.048 | 93.6% | 39.0K | 100.0% | **100.0%** | 165.0 | 75.6% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.001 | -0.674 ± 0.025 | -0.558 ± 0.043 | 93.8% | 53.8K | 100.0% | **100.0%** | 173.5 | 81.1% | 16 |

At the budget the highest $W$ belongs to **intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.01**, while at the run's length it belongs to **intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.001**: the choice changes. At the budget their $W$ are -0.308 ± 0.001 and -0.308 ± 0.002.

### PointMaze Large, reward 1 on arrival

The sweep was trained for 50,331,648 steps per policy with 16 seeds per configuration.

| configuration | whole-curve mean $W$ ± standard error | final level $L$ | 95% of the final level, measured from $R_0$ | first step at which the smoothed curve reaches it |
|---|---|---|---|---|
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01 | 0.967 ± 0.003 | 0.999 | 0.949 | 4,456,448 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.01 | 0.963 ± 0.004 | 0.998 | 0.948 | 4,980,736 |
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.01 | 0.957 ± 0.004 | 0.980 | 0.931 | 4,456,448 |

Proposed budget: **16,777,216 steps per policy**: twice the largest step count is 9,961,472, rounded up to a power of two.

The measures of every configuration that trained the whole length, recomputed at 16,777,216 steps and sorted by $W$. The runs were trained for 50,331,648 steps, so their learning rate has not reached zero at the budget.

| configuration | whole-curve mean $W$ ↑ ± standard error | final level $L$ ± standard error | final level over peak | mean capped first-goal step | policies that reach the goal | maze cells reached | mean steps to the goal, last 10% | episodes that reach the goal, last 10% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.01 | **0.913** ± 0.006 | 0.980 ± 0.004 | 99.7% | 307.9K | **100.0%** | 96.9% | 290.8 | 98.1% | 16 |
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01 | <u>0.911</u> ± 0.007 | <u>0.987</u> ± 0.004 | **100.0%** | 524.4K | **100.0%** | 98.1% | 291.0 | <u>98.7%</u> | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.01 | 0.899 ± 0.011 | **0.993** ± 0.001 | **100.0%** | 855.7K | **100.0%** | 95.1% | 292.0 | **99.3%** | 16 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.01 | 0.887 ± 0.008 | 0.960 ± 0.008 | 98.5% | 567.2K | **100.0%** | 97.4% | 294.5 | 96.0% | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.01 | 0.867 ± 0.015 | **0.993** ± 0.003 | <u>99.9%</u> | 1.01M | **100.0%** | 97.0% | 290.7 | **99.3%** | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.01 | 0.867 ± 0.009 | 0.959 ± 0.008 | **100.0%** | 481.8K | **100.0%** | 97.3% | 299.3 | 96.0% | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.001 | 0.824 ± 0.023 | 0.940 ± 0.009 | 99.0% | 607.6K | **100.0%** | 97.6% | 305.7 | 94.2% | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.01 | 0.794 ± 0.057 | 0.928 ± 0.062 | 99.5% | 2.61M | <u>93.8%</u> | 95.8% | 295.3 | 92.8% | 16 |
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.001 | 0.689 ± 0.058 | 0.815 ± 0.082 | 91.4% | 1.01M | **100.0%** | 99.0% | 302.1 | 81.8% | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.01 | 0.688 ± 0.015 | 0.733 ± 0.023 | 92.9% | <u>260.1K</u> | **100.0%** | **100.0%** | 328.5 | 73.5% | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.001 | 0.643 ± 0.069 | 0.892 ± 0.064 | **100.0%** | 2.15M | **100.0%** | 97.8% | 315.2 | 89.4% | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.01 | 0.623 ± 0.020 | 0.784 ± 0.025 | 97.7% | 405.3K | **100.0%** | **100.0%** | 323.5 | 78.7% | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.01 | 0.623 ± 0.071 | 0.873 ± 0.085 | **100.0%** | 5.30M | **100.0%** | 95.7% | 290.0 | 87.3% | 16 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | 0.598 ± 0.083 | 0.562 ± 0.116 | 75.4% | 2.36M | <u>93.8%</u> | 92.0% | 308.8 | 56.3% | 16 |
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.001 | 0.570 ± 0.077 | 0.658 ± 0.088 | 96.7% | 868.3K | **100.0%** | 98.5% | 317.7 | 66.2% | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | 0.533 ± 0.100 | 0.626 ± 0.113 | 92.9% | 5.07M | 75.0% | 90.1% | 298.8 | 62.9% | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.01 | 0.488 ± 0.023 | 0.720 ± 0.039 | **100.0%** | 997.0K | **100.0%** | 99.6% | 326.1 | 72.2% | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.01 | 0.469 ± 0.078 | 0.800 ± 0.099 | **100.0%** | 7.30M | 81.2% | 86.7% | 303.8 | 80.0% | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.001 | 0.320 ± 0.034 | 0.422 ± 0.069 | 87.2% | 524.6K | **100.0%** | **100.0%** | 337.4 | 42.4% | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | 0.230 ± 0.040 | 0.174 ± 0.049 | 49.4% | 639.1K | **100.0%** | 99.3% | 351.7 | 17.6% | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.001 | 0.207 ± 0.082 | 0.318 ± 0.110 | **100.0%** | 9.42M | 56.2% | 87.5% | 318.8 | 32.0% | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.001 | 0.203 ± 0.044 | 0.148 ± 0.048 | 47.4% | 300.4K | **100.0%** | <u>99.9%</u> | 341.6 | 15.1% | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.01 | 0.187 ± 0.084 | 0.247 ± 0.110 | 99.0% | 8.92M | 56.2% | 77.0% | 289.1 | 24.7% | 16 |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.01 | 0.103 ± 0.072 | 0.124 ± 0.085 | 99.4% | 14.07M | 18.8% | 56.7% | 289.8 | 12.4% | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.001 | 0.102 ± 0.062 | 0.228 ± 0.102 | **100.0%** | 12.95M | 37.5% | 64.3% | 297.8 | 22.9% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.01 | 0.098 ± 0.012 | 0.155 ± 0.024 | 85.3% | 496.0K | **100.0%** | **100.0%** | 361.3 | 15.6% | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.01 | 0.084 ± 0.050 | 0.187 ± 0.101 | **100.0%** | 14.65M | 25.0% | 67.5% | 294.6 | 18.7% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.01 | 0.071 ± 0.008 | 0.135 ± 0.022 | 97.9% | 839.7K | **100.0%** | **100.0%** | 357.2 | 13.5% | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.001 | 0.064 ± 0.045 | 0.125 ± 0.084 | **100.0%** | 11.69M | 56.2% | 72.4% | 307.0 | 12.5% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.01 | 0.060 ± 0.011 | 0.046 ± 0.016 | 46.3% | **167.6K** | **100.0%** | **100.0%** | 370.1 | 4.7% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.001 | 0.058 ± 0.058 | 0.062 ± 0.062 | **100.0%** | 15.76M | 6.2% | 29.5% | **286.2** | 6.2% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.001 | 0.056 ± 0.056 | 0.062 ± 0.062 | **100.0%** | 15.73M | 6.2% | 22.6% | <u>287.0</u> | 6.2% | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.001 | 0.038 ± 0.038 | 0.047 ± 0.047 | 74.6% | 14.68M | 18.8% | 60.9% | 307.1 | 4.8% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.001 | 0.033 ± 0.010 | 0.034 ± 0.011 | 69.0% | 1.24M | <u>93.8%</u> | 96.9% | 358.8 | 3.4% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.001 | 0.027 ± 0.009 | 0.047 ± 0.030 | **100.0%** | 420.0K | **100.0%** | **100.0%** | 329.2 | 4.9% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.001 | 0.023 ± 0.005 | 0.039 ± 0.018 | 97.6% | 1.92M | **100.0%** | 99.5% | 358.7 | 3.9% | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.01 | 0.001 ± 0.001 | 0.014 ± 0.014 | **100.0%** | 15.61M | 12.5% | 45.1% | 341.5 | 1.5% | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0% | 14.76M | 12.5% | 40.5% | none | 0.0% | 16 |
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0% | 15.81M | 6.2% | 27.4% | none | 0.0% | 16 |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0% | 16.14M | 6.2% | 28.8% | none | 0.0% | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0% | 15.73M | 6.2% | 27.2% | none | 0.0% | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0% | 15.73M | 6.2% | 22.7% | none | 0.0% | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0% | 15.86M | 6.2% | 35.3% | none | 0.0% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0% | 15.73M | 6.2% | 24.7% | none | 0.0% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0% | 15.73M | 6.2% | 28.4% | none | 0.0% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0% | 15.73M | 6.2% | 31.3% | none | 0.0% | 16 |
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.0% | 16.16M | 6.2% | 39.4% | none | 0.0% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 16.78M | 0.0% | 25.7% | none | 0.0% | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 16.78M | 0.0% | 35.3% | none | 0.0% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 16.78M | 0.0% | 17.8% | none | 0.0% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 16.78M | 0.0% | 22.8% | none | 0.0% | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 16.78M | 0.0% | 33.3% | none | 0.0% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 16.78M | 0.0% | 21.5% | none | 0.0% | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 16.78M | 0.0% | 33.8% | none | 0.0% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 16.78M | 0.0% | 19.4% | none | 0.0% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 16.78M | 0.0% | 37.1% | none | 0.0% | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 16.78M | 0.0% | 39.8% | none | 0.0% | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 16.78M | 0.0% | 40.6% | none | 0.0% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 16.78M | 0.0% | 23.8% | none | 0.0% | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.01 | 0.000 ± 0.000 | 0.000 ± 0.000 | none | 16.78M | 0.0% | 43.5% | none | 0.0% | 16 |

At the budget the highest $W$ belongs to **intrinsic weight 4, learning rate 0.001, entropy coefficient 0.01**, while at the run's length it belongs to **intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01**: the choice changes. At the budget their $W$ are 0.913 ± 0.006 and 0.911 ± 0.007.

### PointMaze Large, step cost

The sweep was trained for 50,331,648 steps per policy with 16 seeds per configuration.

| configuration | whole-curve mean $W$ ± standard error | final level $L$ | 95% of the final level, measured from $R_0$ | first step at which the smoothed curve reaches it |
|---|---|---|---|---|
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.01 | -0.717 ± 0.002 | -0.706 | -0.720 | 11,534,336 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.01 | -0.734 ± 0.002 | -0.716 | -0.730 | 9,699,328 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.01 | -0.736 ± 0.011 | -0.717 | -0.731 | 14,155,776 |

Proposed budget: **33,554,432 steps per policy**: twice the largest step count is 28,311,552, rounded up to a power of two.

The measures of every configuration that trained the whole length, recomputed at 33,554,432 steps and sorted by $W$. The runs were trained for 50,331,648 steps, so their learning rate has not reached zero at the budget.

| configuration | whole-curve mean $W$ ↑ ± standard error | final level $L$ ± standard error | final level over peak | mean capped first-goal step | policies that reach the goal | maze cells reached | mean steps to the goal, last 10% | episodes that reach the goal, last 10% | seeds |
|---|---|---|---|---|---|---|---|---|---|
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.01 | **-0.723** ± 0.001 | **-0.709** ± 0.003 | 99.5% | 232.3K | **100.0%** | 98.5% | **283.9** | 99.5% | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.01 | <u>-0.741</u> ± 0.003 | <u>-0.719</u> ± 0.003 | 98.2% | 629.1K | **100.0%** | 98.6% | <u>287.0</u> | 98.8% | 16 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.01 | -0.744 ± 0.011 | -0.721 ± 0.011 | 99.1% | 298.7K | **100.0%** | 98.9% | 287.6 | 99.2% | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.01 | -0.748 ± 0.014 | -0.731 ± 0.015 | 99.3% | 364.0K | **100.0%** | 97.7% | 291.5 | 99.4% | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.01 | -0.754 ± 0.005 | -0.742 ± 0.009 | 98.0% | **122.9K** | **100.0%** | 97.6% | 293.5 | 96.7% | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.01 | -0.759 ± 0.018 | -0.739 ± 0.019 | **100.0%** | 1.13M | **100.0%** | 95.8% | 294.0 | **99.9%** | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.01 | -0.760 ± 0.015 | -0.730 ± 0.016 | **100.0%** | 1.49M | **100.0%** | 96.5% | 291.1 | <u>99.8%</u> | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.01 | -0.769 ± 0.006 | -0.743 ± 0.008 | **100.0%** | 401.1K | **100.0%** | 98.5% | 295.0 | 97.4% | 16 |
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01 | -0.775 ± 0.021 | -0.761 ± 0.021 | 99.7% | 506.4K | **100.0%** | 97.8% | 301.5 | 99.5% | 16 |
| intrinsic weight 4, learning rate 0.0003, entropy coefficient 0.001 | -0.779 ± 0.008 | -0.779 ± 0.012 | 89.1% | 535.4K | **100.0%** | 99.0% | 307.4 | 95.6% | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.01 | -0.779 ± 0.021 | -0.760 ± 0.022 | **100.0%** | 1.04M | **100.0%** | 97.7% | 301.5 | <u>99.8%</u> | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.01 | -0.787 ± 0.014 | -0.748 ± 0.016 | **100.0%** | 1.66M | **100.0%** | 97.1% | 297.6 | 99.3% | 16 |
| intrinsic weight 4, learning rate 0.001, entropy coefficient 0.001 | -0.803 ± 0.016 | -0.823 ± 0.020 | 80.8% | 530.6K | **100.0%** | 98.5% | 316.5 | 86.9% | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.01 | -0.804 ± 0.018 | -0.739 ± 0.015 | 99.6% | 6.49M | **100.0%** | 96.5% | 294.9 | 99.7% | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.01 | -0.806 ± 0.004 | -0.775 ± 0.007 | 97.9% | 794.8K | **100.0%** | <u>99.9%</u> | 305.1 | 94.4% | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.01 | -0.816 ± 0.032 | -0.792 ± 0.034 | **100.0%** | 10.44M | 75.0% | 90.9% | 288.3 | 75.0% | 16 |
| intrinsic weight 4, learning rate 0.0001, entropy coefficient 0.001 | -0.822 ± 0.021 | -0.812 ± 0.027 | 97.2% | 677.3K | **100.0%** | 99.0% | 304.1 | 79.4% | 16 |
| intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.001 | -0.824 ± 0.025 | -0.791 ± 0.024 | 99.3% | 4.42M | **100.0%** | 98.4% | 313.0 | 99.3% | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.01 | -0.824 ± 0.033 | -0.808 ± 0.036 | **100.0%** | 9.46M | 75.0% | 85.2% | 287.6 | 68.7% | 16 |
| intrinsic weight 1, learning rate 0.0003, entropy coefficient 0.001 | -0.825 ± 0.025 | -0.815 ± 0.028 | 93.9% | 3.52M | <u>93.8%</u> | 98.4% | 306.8 | 80.9% | 16 |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.01 | -0.826 ± 0.029 | -0.809 ± 0.031 | **100.0%** | 7.49M | 81.2% | 87.5% | 303.0 | 81.2% | 16 |
| intrinsic weight 1, learning rate 0.001, entropy coefficient 0.001 | -0.828 ± 0.022 | -0.793 ± 0.026 | **100.0%** | 2.40M | <u>93.8%</u> | 93.5% | 304.3 | 88.7% | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.01 | -0.840 ± 0.034 | -0.827 ± 0.036 | **100.0%** | 13.03M | 62.5% | 81.7% | 288.7 | 62.5% | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.01 | -0.841 ± 0.024 | -0.780 ± 0.020 | **100.0%** | 7.38M | **100.0%** | 96.2% | 307.7 | 96.9% | 16 |
| intrinsic weight 0.0625, learning rate 0.001, entropy coefficient 0.001 | -0.858 ± 0.027 | -0.847 ± 0.030 | 93.4% | 9.00M | 75.0% | 87.0% | 304.0 | 64.4% | 16 |
| intrinsic weight 16, learning rate 0.0001, entropy coefficient 0.001 | -0.863 ± 0.010 | -0.874 ± 0.019 | 72.1% | 958.8K | **100.0%** | 99.5% | 324.6 | 68.3% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.01 | -0.880 ± 0.031 | -0.862 ± 0.035 | **100.0%** | 15.77M | 56.2% | 65.8% | 299.8 | 56.1% | 16 |
| intrinsic weight 0.25, learning rate 0.0001, entropy coefficient 0.001 | -0.882 ± 0.028 | -0.846 ± 0.030 | **100.0%** | 13.49M | 81.2% | 89.8% | 309.0 | 69.4% | 16 |
| intrinsic weight 0.25, learning rate 0.0003, entropy coefficient 0.001 | -0.884 ± 0.028 | -0.866 ± 0.032 | 98.1% | 11.01M | 75.0% | 89.5% | 317.3 | 68.2% | 16 |
| intrinsic weight 0.25, learning rate 0.001, entropy coefficient 0.001 | -0.890 ± 0.023 | -0.860 ± 0.028 | **100.0%** | 4.05M | <u>93.8%</u> | 98.8% | 318.6 | 70.9% | 16 |
| intrinsic weight 0.000244141, learning rate 0.001, entropy coefficient 0.001 | -0.893 ± 0.028 | -0.867 ± 0.033 | 99.2% | 13.89M | 68.8% | 88.3% | 302.8 | 55.6% | 16 |
| intrinsic weight 0.015625, learning rate 0.001, entropy coefficient 0.001 | -0.894 ± 0.026 | -0.890 ± 0.028 | 87.2% | 11.02M | 68.8% | 92.0% | 323.4 | 59.9% | 16 |
| intrinsic weight 0.00390625, learning rate 0.001, entropy coefficient 0.001 | -0.895 ± 0.030 | -0.879 ± 0.034 | **100.0%** | 12.74M | 68.8% | 79.6% | 297.6 | 47.8% | 16 |
| intrinsic weight 16, learning rate 0.001, entropy coefficient 0.001 | -0.896 ± 0.012 | -0.889 ± 0.020 | 88.1% | <u>169.3K</u> | **100.0%** | 98.6% | 324.8 | 60.5% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.01 | -0.901 ± 0.029 | -0.886 ± 0.031 | 97.8% | 16.21M | 56.2% | 68.3% | 315.0 | 55.7% | 16 |
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.01 | -0.911 ± 0.028 | -0.890 ± 0.031 | **100.0%** | 17.81M | 56.2% | 68.8% | 318.0 | 56.2% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.01 | -0.918 ± 0.006 | -0.913 ± 0.009 | 85.9% | 393.6K | **100.0%** | **100.0%** | 342.5 | 60.0% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.01 | -0.922 ± 0.026 | -0.912 ± 0.028 | **100.0%** | 16.29M | 56.2% | 64.9% | 326.1 | 49.8% | 16 |
| intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.001 | -0.925 ± 0.022 | -0.910 ± 0.024 | 95.6% | 15.54M | 62.5% | 71.1% | 339.1 | 60.8% | 16 |
| intrinsic weight 16, learning rate 0.0003, entropy coefficient 0.001 | -0.925 ± 0.012 | -0.965 ± 0.011 | 27.6% | 2.35M | <u>93.8%</u> | 96.9% | 353.2 | 30.1% | 16 |
| intrinsic weight 0.000976562, learning rate 0.001, entropy coefficient 0.001 | -0.928 ± 0.025 | -0.928 ± 0.027 | 82.3% | 17.06M | 62.5% | 85.9% | 314.4 | 34.8% | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.01 | -0.928 ± 0.022 | -0.886 ± 0.029 | **100.0%** | 18.32M | 68.8% | 75.5% | 323.4 | 61.8% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.001 | -0.929 ± 0.027 | -0.918 ± 0.030 | <u>99.9%</u> | 20.20M | 43.8% | 55.8% | 310.6 | 37.5% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.01 | -0.935 ± 0.007 | -0.928 ± 0.010 | 80.6% | 1.12M | **100.0%** | **100.0%** | 341.4 | 49.2% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.01 | -0.936 ± 0.024 | -0.918 ± 0.030 | **100.0%** | 22.94M | 37.5% | 57.1% | 310.5 | 37.3% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0001, entropy coefficient 0.001 | -0.941 ± 0.026 | -0.933 ± 0.028 | **100.0%** | 20.31M | 43.8% | 61.3% | 323.2 | 37.4% | 16 |
| intrinsic weight 0.0625, learning rate 0.0003, entropy coefficient 0.001 | -0.943 ± 0.023 | -0.946 ± 0.022 | 83.3% | 18.23M | 50.0% | 75.7% | 348.0 | 43.8% | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.01 | -0.945 ± 0.026 | -0.934 ± 0.029 | **100.0%** | 24.75M | 31.2% | 48.5% | 312.0 | 31.2% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.001 | -0.947 ± 0.023 | -0.941 ± 0.025 | 98.7% | 21.36M | 43.8% | 58.8% | 323.1 | 31.3% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0003, entropy coefficient 0.001 | -0.947 ± 0.027 | -0.946 ± 0.029 | 89.8% | 22.42M | 37.5% | 58.0% | 290.5 | 19.8% | 16 |
| intrinsic weight 0.0625, learning rate 0.0001, entropy coefficient 0.001 | -0.948 ± 0.021 | -0.941 ± 0.023 | 98.7% | 19.32M | 50.0% | 60.3% | 341.5 | 42.7% | 16 |
| intrinsic weight 0.000244141, learning rate 0.0003, entropy coefficient 0.001 | -0.949 ± 0.023 | -0.943 ± 0.024 | 96.2% | 22.40M | 37.5% | 57.2% | 324.8 | 30.8% | 16 |
| intrinsic weight 0.000976562, learning rate 0.0001, entropy coefficient 0.01 | -0.952 ± 0.023 | -0.941 ± 0.028 | **100.0%** | 25.82M | 25.0% | 45.7% | 303.3 | 25.0% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.01 | -0.953 ± 0.004 | -0.973 ± 0.004 | 33.9% | 342.0K | **100.0%** | 99.7% | 365.7 | 31.2% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0003, entropy coefficient 0.001 | -0.956 ± 0.025 | -0.955 ± 0.025 | 97.8% | 27.31M | 18.8% | 66.0% | 301.1 | 18.6% | 16 |
| intrinsic weight 0.00390625, learning rate 0.0001, entropy coefficient 0.01 | -0.968 ± 0.019 | -0.964 ± 0.021 | **100.0%** | 25.34M | 25.0% | 45.7% | 321.6 | 18.7% | 16 |
| intrinsic weight 0.015625, learning rate 0.0001, entropy coefficient 0.001 | -0.969 ± 0.017 | -0.947 ± 0.026 | 97.1% | 25.42M | 25.0% | 50.7% | 313.7 | 25.0% | 16 |
| intrinsic weight 64, learning rate 0.0001, entropy coefficient 0.001 | -0.971 ± 0.004 | -0.962 ± 0.006 | 82.6% | 1.81M | **100.0%** | **100.0%** | 347.9 | 29.0% | 16 |
| intrinsic weight 64, learning rate 0.0003, entropy coefficient 0.001 | -0.980 ± 0.004 | -0.988 ± 0.003 | 39.0% | 1.23M | **100.0%** | **100.0%** | 364.6 | 13.6% | 16 |
| intrinsic weight 64, learning rate 0.001, entropy coefficient 0.001 | -0.987 ± 0.003 | -0.988 ± 0.006 | 47.4% | 343.2K | **100.0%** | **100.0%** | 351.4 | 9.8% | 16 |

At the budget the highest $W$ belongs to **intrinsic weight 4, learning rate 0.001, entropy coefficient 0.01**, the configuration that is also the best at the run's length, so the choice does not change.

## Training curves

Define $\bar R(s)$ as in the section Measures: the mean over a configuration's policies of the mean return of the training episodes that end in the logging window containing step $s$.

### Development seeds, reward 1 on arrival, CleanRL PPO-RND

Seed-mean training curve of the chosen configuration with a band of one standard error, and the other configurations of the sweep in grey.

![training curves, development seeds](plots/training-curves_development-seeds_reward-1-on-arrival_cleanrl-ppo-rnd.png)

### Development seeds, step cost, CleanRL PPO-RND

Seed-mean training curve of the chosen configuration with a band of one standard error, and the other configurations of the sweep in grey.

![training curves, development seeds](plots/training-curves_development-seeds_step-cost_cleanrl-ppo-rnd.png)

## Final level against the training length

Define $L(f B)$ as in the section Measures: the mean of the training curve over the last 10% of the first share $f$ of the planned steps $B$, read from the runs trained for $B$ steps. Each cell gives $L$; a method's cell adds its paired difference to the baseline (method minus baseline, same seeds) with the 95% interval and its side. A run trained for another number of steps is a separate table, because its learning rate falls to zero over its own length; its curve is in the section Training curves.

### PointMaze Medium, reward 1 on arrival, 50,331,648 planned steps (development seeds, chosen configurations)

| row | 25% of the planned steps | 50% of the planned steps | 75% of the planned steps | 100% of the planned steps |
|---|---|---|---|---|
| CleanRL PPO-RND, 4 environments per policy, intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01 | 1.000 | 1.000 | 1.000 | 1.000 |

### PointMaze Medium, step cost, 50,331,648 planned steps (development seeds, chosen configurations)

| row | 25% of the planned steps | 50% of the planned steps | 75% of the planned steps | 100% of the planned steps |
|---|---|---|---|---|
| CleanRL PPO-RND, 4 environments per policy, intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.001 | -0.302 | -0.303 | -0.302 | -0.302 |

### PointMaze Large, reward 1 on arrival, 50,331,648 planned steps (development seeds, chosen configurations)

| row | 25% of the planned steps | 50% of the planned steps | 75% of the planned steps | 100% of the planned steps |
|---|---|---|---|---|
| CleanRL PPO-RND, 4 environments per policy, intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01 | 0.985 | 0.995 | 0.993 | 0.999 |

### PointMaze Large, step cost, 50,331,648 planned steps (development seeds, chosen configurations)

| row | 25% of the planned steps | 50% of the planned steps | 75% of the planned steps | 100% of the planned steps |
|---|---|---|---|---|
| CleanRL PPO-RND, 4 environments per policy, intrinsic weight 4, learning rate 0.001, entropy coefficient 0.01 | -0.715 | -0.710 | -0.705 | -0.706 |

## Decline check

Define $P(B)$ as the highest value of the seed-averaged curve smoothed over 10% of the budget and $R_0$ as the return of an episode that never reaches the goal (section Measures). The final level over the smoothed peak is $\dfrac{L(B) - R_0}{P(B) - R_0}$; the ratio with the highest single window puts that window's value in place of $P(B)$ and is the stricter of the two. A chosen configuration declines when the ratio with the smoothed peak is below 90%. The table gives both ratios of each of the 4 chosen configurations (the choice of every development sweep and every confirmation row): 0 below 90% of the smoothed peak, 0 below 90% of the highest single window only.

| environment and reward setting | seeds | configuration | final level $L$ | peak smoothed over 10% of the budget, and the step its range ends at | final level over the smoothed peak | highest single window, and its step | final level over the highest single window | verdict | read at step |
|---|---|---|---|---|---|---|---|---|---|
| PointMaze Medium, reward 1 on arrival | development | CleanRL PPO-RND, intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01 | 1.000 | 1.000 at 15.20M | 100.0% | 1.000 at 4.98M | 100.0% | no decline | 50.33M |
| PointMaze Medium, step cost | development | CleanRL PPO-RND, intrinsic weight 0.015625, learning rate 0.0003, entropy coefficient 0.001 | -0.302 | -0.302 at 50.33M | 100.0% | -0.302 at 50.33M | 100.0% | no decline | 50.33M |
| PointMaze Large, reward 1 on arrival | development | CleanRL PPO-RND, intrinsic weight 1, learning rate 0.001, entropy coefficient 0.01 | 0.999 | 0.999 at 50.33M | 100.0% | 1.000 at 49.81M | 99.9% | no decline | 50.33M |
| PointMaze Large, step cost | development | CleanRL PPO-RND, intrinsic weight 4, learning rate 0.001, entropy coefficient 0.01 | -0.706 | -0.705 at 39.85M | 99.8% | -0.703 at 36.70M | 99.2% | no decline | 50.33M |

