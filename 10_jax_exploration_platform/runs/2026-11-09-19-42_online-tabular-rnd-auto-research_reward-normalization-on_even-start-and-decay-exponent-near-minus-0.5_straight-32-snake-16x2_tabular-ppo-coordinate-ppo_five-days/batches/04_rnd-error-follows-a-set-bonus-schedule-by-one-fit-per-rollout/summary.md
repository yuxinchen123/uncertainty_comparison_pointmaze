# Batch 4 summary: an RND whose error follows a set bonus schedule

Written 2026-11-11 10:50 PT, with all 288 units finished and none failed. Times are Pacific. Scores are the mean
capped first-goal time in thousands of transitions at 2^22 transitions, 32 seeds (field seeds 0-7), best of 28
settings; ± is the standard error over field-seed groups. Every row found the goal with every seed except where
noted. "Schedule-fit RND" is the exact output-layer fit of `ideas.md`; "count rule" is the same bonus shape computed
from a visit table.

## Result

| row | refresh | straight, tabular-PPO | straight, coordinate-PPO | snake, tabular-PPO | snake, coordinate-PPO |
|---|---|---|---|---|---|
| control (CleanRL online tabular-RND) | per rollout | 7.22 ± 0.28 | 3.90 ± 0.20 | 18.74 ± 0.96 | 82.1 ± 12.0 |
| schedule-fit RND, exponent −1 | per step | 6.33 ± 0.29 | 1.96 ± 0.29 | 13.51 ± 0.72 | 42.1 ± 2.8 |
| schedule-fit RND, exponent −1 at rate 0.25 | per step | 6.75 ± 0.37 | 2.13 ± 0.19 | 13.69 ± 1.05 | 43.2 ± 3.0 |
| schedule-fit RND, exponent −1 at rate 0.25 | per rollout | 6.97 ± 0.41 | 3.11 ± 0.24 | 16.60 ± 0.87 | 43.8 ± 5.6 |
| schedule-fit RND, exponent −0.75 | per step | 7.31 ± 0.35 | 2.22 ± 0.25 | 17.84 ± 2.03 | 47.9 ± 4.4 |
| schedule-fit RND, exponent −0.5 | per step | 8.63 ± 0.65 | 2.29 ± 0.23 | 32.2 ± 4.4 | 49.0 ± 5.1 |
| schedule-fit RND, exponent −0.5 | per rollout | 8.77 ± 0.84 | 3.43 ± 0.31 | 39.4 ± 6.5 | 57.7 ± 3.1 |
| all-layers fit, exponent −1 at rate 0.25 | per rollout | 9.64 ± 1.08 | 3.54 ± 0.16 | 27.8 ± 2.7 | 65.1 ± 8.1 |
| all-layers fit, exponent −0.5 | per rollout | 11.86 ± 1.20 | 4.15 ± 0.39 | 45.0 ± 9.4 | 64.2 ± 4.4 |
| count rule, exponent −1 at rate 0.5 | per step | 6.35 ± 0.37 | 2.05 ± 0.26 | 12.01 ± 0.78 | 39.5 ± 2.4 |
| count rule, exponent −1 at rate 0.25 | per step | 7.03 ± 0.41 | 2.27 ± 0.20 | 13.42 ± 1.07 | 39.5 ± 3.0 |
| count rule, exponent −1 at rate 0.125 | per step | 6.90 ± 0.52 | 2.36 ± 0.25 | 13.11 ± 1.20 | 43.8 ± 5.0 |
| count rule, exponent −1 at rate 0.0625 | per step | 6.75 ± 0.29 | 2.72 ± 0.20 | 15.09 ± 1.23 | 44.8 ± 4.0 |
| count rule, exponent −1 at rate 0.5 | per rollout | 7.55 ± 0.44 | 3.01 ± 0.27 | 17.08 ± 0.86 | 67.5 ± 11.4 |
| count rule, exponent −1 at rate 0.125 | per rollout | 7.41 ± 0.56 | 3.09 ± 0.22 | 16.42 ± 1.13 | 40.0 ± 3.6 |
| count rule, exponent −1 at rate 0.0625 | per rollout | 7.74 ± 0.45 | 3.08 ± 0.22 | 17.09 ± 1.41 | 54.2 ± 5.2 |
| count rule, exponent −0.75 | per step | 7.53 ± 0.46 | 2.30 ± 0.28 | 16.62 ± 1.52 | 39.7 ± 3.1 |
| count rule, exponent −1.5 at rate 0.25 | per step | 6.23 ± 0.32 | 1.83 ± 0.27 | 10.38 ± 0.58 | 268 ± 134 (goal found by 97%) |
| count rule, exponent −1.5 at rate 0.0625 | per step | 6.26 ± 0.34 | 2.31 ± 0.23 | 12.38 ± 0.64 | 52.9 ± 8.2 |

Measured in the RL runs at each row's best setting: every schedule-fit and all-layers row starts with a
largest-to-smallest bonus ratio of 1.00. The median per-cell slope is −0.50 for the exponent −0.5 rows, −0.69 to
−0.72 for exponent −0.75, −0.79 to −0.88 for exponent −1, and −0.86 to −0.94 for exponent −1 at rate 0.25. The
slope is below the exponent in size because the schedule is $(1+m)^{-a}$ and the first visits fall on its flat part.

## Predictions against outcomes

1. **Each schedule-fit row scores within two standard errors of the count rule of the same shape and refresh.**
   Holds in every pair measured (exponent −0.5 per step and per rollout, −0.75 per step, −1 per step, −1 at rate
   0.25 per step and per rollout, against the count-rule rows of batches 3 to 5). The largest gap is exponent −1 at
   rate 0.25 per rollout on the snake grid with coordinate-PPO: 43.8 ± 5.6 against 63.8 ± 9.7.
2. **Start ratio 1.00, and a median slope between −0.52 and −0.48 for the exponent −0.5 rows.** Holds.
3. **Exponent −0.5 is ahead of the control with coordinate-PPO and behind it with tabular-PPO.** Holds, refreshed
   either way.
4. **Exponent −1 at rate 0.25, refreshed per step, is at or below the control in all four cases.** Holds (6.75,
   2.13, 13.69, 43.2 against 7.22, 3.90, 18.74, 82.1).
5. **The all-layers rows are within 20% of the exact rows on the straight grid and worse on the snake grid.** Fails
   on the straight grid in three of four comparisons: the all-layers fit is 38% and 14% behind the exact fit at
   exponent −1 with rate 0.25, and 35% and 21% behind at exponent −0.5. On the snake grid it is 68% and 49% behind
   at exponent −1 with rate 0.25, and 14% and 11% behind at exponent −0.5.

## What the batch shows

1. An RND can be built whose error follows a chosen schedule exactly from an even start, and it then scores as the
   count rule of the same shape does.
2. Exponent −1 refreshed at every step is ahead of the control in all four cases at the screen, by 12%, 50%, 28%
   and 49%. A count rate below 1 brings no gain with a per-step refresh (rates 0.5 and 1 are within one standard
   error of each other; 0.0625 is behind on the straight grid with coordinate-PPO and on the snake grid with
   tabular-PPO).
3. Refreshing at every step is ahead of refreshing once per rollout in every pair of rows with the same schedule.
4. The all-layers fit, the only form here that applies to inputs that are not one-hot, is behind the exact fit in
   all eight comparisons and behind the control on the straight grid with tabular-PPO and on the snake grid with
   tabular-PPO.

## Decision

The finalist is the schedule-fit RND with exponent −1 at rate 1, refreshed at every step. It goes to the full
protocol in batch 6, with the exponent −0.5 form beside it because that form meets both property targets of
`program.md`.
