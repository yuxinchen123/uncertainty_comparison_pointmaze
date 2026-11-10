# Batch 5 summary: an uneven start with a scheduled decay

Written 2026-11-10 04:43 PT, with all 64 units finished and none failed. Times are Pacific. Scores are the mean capped
first-goal time in thousands of transitions at 2^22 transitions, 32 seeds (field seeds 0-7), best of 28 settings;
± is the standard error over field-seed groups. The goal was found by every seed in every row except where noted.

## Result

Each pair is the same count rule on the frozen tabular-RND field (start max/min 3.65 on the straight grid and 4.23
on the snake grid, the control's own start) and on the even field (start max/min 1.00).

| rule | field | straight, tabular-PPO | straight, coordinate-PPO | snake, tabular-PPO | snake, coordinate-PPO |
|---|---|---|---|---|---|
| exponent −1, every step | uneven | 6.91 ± 0.60 | 1.94 ± 0.25 | 15.06 ± 1.44 | 51.5 ± 7.1 |
| exponent −1, every step | even | 6.47 ± 0.26 | 1.70 ± 0.26 | 13.31 ± 0.74 | 41.3 ± 2.8 |
| exponent −1 at rate 0.25, every step | uneven | 7.58 ± 0.60 | 2.12 ± 0.25 | 15.60 ± 1.02 | 103.9 ± 38.2 |
| exponent −1 at rate 0.25, every step | even | not finished | 2.27 ± 0.20 | 13.42 ± 1.07 | 39.5 ± 3.0 |
| exponent −1 at rate 0.25, per rollout | uneven | 7.89 ± 0.64 | 3.13 ± 0.27 | 18.76 ± 1.80 | 129.2 ± 31.9 |
| exponent −1 at rate 0.25, per rollout | even | 7.08 ± 0.37 | 3.20 ± 0.22 | 15.92 ± 0.79 | 63.8 ± 9.7 |
| exponent −0.5, every step | uneven | 14.23 ± 3.80 | 2.56 ± 0.28 | 67.7 ± 12.7 | 558 ± 208 (goal found by 97%) |
| exponent −0.5, every step | even | 8.42 ± 0.39 | 2.38 ± 0.22 | 32.3 ± 4.5 | 52.7 ± 7.8 |
| control (uneven start, decay left to training) | | 7.22 ± 0.28 | 3.90 ± 0.20 | 18.74 ± 0.96 | 82.1 ± 12.0 |

## Predictions against outcomes

1. **Each row is within 30% of the same rule on the even field.** Holds for exponent −1 every step (7%, 14%, 13%
   and 25% slower). Fails on the snake grid with coordinate-PPO for exponent −1 at a quarter rate (2.6 and 2.0 times
   slower) and fails for exponent −0.5 (69% slower, 8%, 2.1 times and 10.6 times).
2. **Most of the gain over the control comes from the decay and its refresh.** Holds at exponent −1 every step:
   with the uneven start it is already ahead of the control in all four cases, and the even start adds 7% to 25%.
3. **The rows with exponent −1 are at or below the control in all four cases even with the uneven start.** Holds for
   exponent −1 every step. Fails at a quarter rate: on the snake grid with coordinate-PPO both quarter-rate rows are
   behind the control (103.9 and 129.2 against 82.1), and on the straight grid with tabular-PPO they are 5% and 9%
   behind it.

## What the batch shows

1. Both parts matter, and they interact. Define $f$ as the ratio of two cells' starting bonuses and $-a$ as the
   decay exponent. Under a count rule the cell with the higher start needs $f^{1/a}$ times the visits before its
   bonus falls to the other's: 4 times at exponent −1 and 16 times at −0.5 for $f = 4$. So an uneven start costs
   more when the decay is slow or the count rate is low, which is what the rows show.
2. The 2×2 comparison at the screen, with the scheduled decay at exponent −1 every step:

| | decay left to training | decay set by the schedule |
|---|---|---|
| **uneven start** | control: 7.22, 3.90, 18.74, 82.1 | 6.91, 1.94, 15.06, 51.5 |
| **even start** | unit start alone: 8.57, 4.66, 21.74, 1,905 | 6.47, 1.70, 13.31, 41.3 (schedule-fit RND: 6.33, 1.96, 13.51, 42.1) |

   The even start alone is behind the control in all four cases; the scheduled decay alone is ahead in all four;
   both together are best in all four.
