# Batch 3 summary: count-rule bonus shapes

Written 2026-11-10 03:44 PT, with all 224 units finished and none failed. Times are Pacific. Scores are the mean capped
first-goal time in thousands of transitions at 2^22 transitions, 32 seeds, best of 28 settings (policy learning rate,
entropy, weight); ± is the standard error over field-seed groups. Full tables: `analysis/analysis.md`.

## Result

Define $m$ as a cell's earlier visits and $c$ as the count rate; the bonus is $(1 + c\thinspace m)^{-a}$ for decay
exponent $-a$, refreshed at every step or once per rollout.

| row | straight, tabular-PPO | straight, coordinate-PPO | snake, tabular-PPO | snake, coordinate-PPO |
|---|---|---|---|---|
| control (CleanRL online tabular-RND) | 7.22 ± 0.28 | 3.90 ± 0.20 | 18.74 ± 0.96 | 82.1 ± 12.0 |
| −1, per step | **6.47 ± 0.26** | **1.70 ± 0.26** | **13.31 ± 0.74** | **41.3 ± 2.8** |
| −1 at rate 0.25, per rollout | 7.08 ± 0.37 | 3.20 ± 0.22 | 15.92 ± 0.79 | 63.8 ± 9.7 |
| −1, per rollout | 7.73 ± 0.47 | 3.09 ± 0.42 | 16.25 ± 0.88 | 116.9 ± 41.6 |
| −1 at rate 4, per rollout | 7.18 ± 0.42 | 3.97 ± 0.17 | 15.21 ± 0.67 | 319.6 ± 51.6 |
| −0.75, per rollout | 8.67 ± 0.70 | 3.21 ± 0.38 | 18.46 ± 1.68 | 59.6 ± 7.2 |
| −0.5, per step (batch 2) | 8.42 ± 0.39 | 2.38 ± 0.22 | 32.3 ± 4.5 | 52.7 ± 7.8 |
| −0.5, per rollout (batch 2) | 9.70 ± 0.72 | 3.52 ± 0.32 | 42.8 ± 6.3 | 57.0 ± 4.4 |
| −1.5, per rollout | 7.21 ± 0.44 | 4.35 ± 0.29 | 12.94 ± 1.13 | 2,429 ± 325 |
| −2, per step | 5.76 ± 0.30 | 3.96 ± 0.45 | 11.15 ± 1.19 | 2,839 ± 439 |
| −2, per rollout | 7.56 ± 0.61 | 5.77 ± 0.29 | 14.14 ± 1.62 | 3,288 ± 282 |
| −2 at rate 0.25, per rollout | 6.79 ± 0.49 | 3.89 ± 0.48 | 12.75 ± 0.77 | 2,613 ± 249 |
| −3, per rollout | 8.03 ± 1.05 | 10.31 ± 2.95 | 16.12 ± 1.38 | 3,283 ± 249 |
| −1 plus 0.1 of −0.5, per rollout | 8.84 ± 0.66 | 3.62 ± 0.42 | 22.38 ± 1.42 | 59.7 ± 7.0 |
| −1 plus 0.3 of −0.5, per rollout | 9.76 ± 1.17 | 3.65 ± 0.40 | 26.15 ± 2.15 | 56.8 ± 3.8 |
| −2 plus 0.03 of −0.5, per rollout | 9.28 ± 0.79 | 5.13 ± 0.26 | 19.24 ± 1.72 | 109.0 ± 15.1 |
| −2 plus 0.1 of −0.5, per rollout | 10.55 ± 1.52 | 4.27 ± 0.33 | 27.87 ± 2.26 | 73.3 ± 8.1 |
| −2 plus 0.3 of −0.5, per rollout | 10.90 ± 1.56 | 4.14 ± 0.32 | 37.27 ± 4.14 | 62.0 ± 6.0 |

## Answer to the batch's question

Yes: there is one shape that is good in all four cases. **Exponent −1 refreshed at every step** is ahead of the
control in every case (by 10%, 56%, 29% and 50%). Exponent −1 at a quarter of the count rate, refreshed per rollout,
is level with the control on the straight grid with tabular-PPO and ahead in the other three.

## Predictions against outcomes

1. **Tabular-PPO: −1.5 and −3 score like −2 (within 10%).** Holds for −1.5 (7.21 against 7.56; 12.94 against 14.14)
   and on the straight grid for −3 (8.03); −3 on the snake grid is 14% slower (16.12).
2. **Coordinate-PPO on the snake grid worsens steadily from −0.75 to −3.** Holds: 59.6, 116.9, 2,429, 3,288, 3,283.
3. **A sum with a slow tail keeps tabular-PPO within 15% of its best and brings the snake grid with coordinate-PPO
   under the control.** Half fails. The sums are under the control with coordinate-PPO on the snake grid (56.8 to
   73.3 at tail weights 0.1 and 0.3), but with tabular-PPO they are 23% to 190% behind the best shape. A sum is not
   better than −1 alone.

## What the batch changes

1. The decay target moves from −0.5 to about −1. Exponent −0.5 is ahead of the control only with coordinate-PPO;
   exponents of −1.5 and faster fail on the snake grid with coordinate-PPO (the goal is found by few seeds).
2. The refresh timing matters as much as the exponent: at −1, per step against per rollout is 6.47 against 7.73,
   1.70 against 3.09, 13.31 against 16.25 and 41.3 against 116.9.
3. A slower count rate helps the per-rollout rule with coordinate-PPO on the snake grid (63.8 at rate 0.25 against
   116.9 at rate 1) and a faster one hurts it (319.6 at rate 4).
