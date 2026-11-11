# Batch 7 summary: the decay exponent and the refresh on the full protocol

Written 2026-11-11 10:58 PT, with all 80 development units and all 20 confirmation units finished and none failed.
Times are Pacific. Every row is the schedule-fit RND with an even start. Each row's configuration was chosen on the
development seeds 0-15 from 52 configurations at 2^24 transitions, then trained on the confirmation seeds 20000-20511
(512 seeds; the exponent −1 and −0.5 rows of batch 6 have 1,024). Scores are the mean capped first-goal time at 2^24
transitions; the percentage is the difference to the baseline on the seeds both rows have, with the verdict of
`analysis/analysis.md`, section "confirmation".

## Result

| decay exponent | refresh | straight, tabular-PPO | straight, coordinate-PPO | snake, tabular-PPO | snake, coordinate-PPO | cases ahead of the baseline |
|---|---|---|---|---|---|---|
| baseline (decay left to training) | per rollout | 7.56K | 81.7K (trimmed 4.74K) | 18.47K | 157.0K (trimmed 91.0K) | — |
| −0.25 | every step | 20.54K, +173%, behind | 5.72K, −94%, ahead | 218.1K, +1,112%, behind | 67.4K, −52%, ahead | 2 of 4 |
| −0.5 (batch 6) | every step | 8.04K, +6%, behind | 3.09K, −96%, ahead | 32.0K, +73%, behind | 60.1K, −62%, ahead | 2 of 4 |
| −0.75 | every step | 6.81K, −10%, ahead | 2.19K, −98%, ahead | 15.99K, −11%, ahead | 52.4K, −63%, ahead | 4 of 4 |
| −1 (batch 6) | every step | 6.12K, −19%, ahead | 2.41K, −97%, ahead | 12.84K, −31%, ahead | 67.6K, −57%, ahead | 4 of 4 |
| −1.5 | every step | 5.38K, −29%, ahead | 15.64K, −84%, ahead (trimmed 2.58K) | 10.10K, −44%, ahead | 1.32M, +834%, behind | 3 of 4 |
| −2 | every step | 5.59K, −26%, ahead | 230.2K, +135%, level (trimmed 3.80K) | 10.49K, −42%, ahead | 5.53M, behind (goal found by 86%) | 2 of 4 |
| −1 | per rollout | 7.13K, −5%, ahead | 4.55K, −95%, ahead | 14.78K, −18%, ahead | 120.8K, −14%, level | 3 of 4 |

Rows compared with exponent −1 refreshed every step, on the same seeds (first row minus second row):

| comparison | straight, tabular-PPO | straight, coordinate-PPO | snake, tabular-PPO | snake, coordinate-PPO |
|---|---|---|---|---|
| exponent −0.75 against −1 | +11%, behind | −6%, level | +24%, behind | −28%, ahead |
| exponent −0.5 against −1 | +31%, behind | +28%, behind | +149%, behind | −11%, level |
| refreshed every step against once per rollout, both at −1 | −14%, ahead | −49%, ahead | −13%, ahead | −40%, ahead |

Final-policy goal rate on the snake grid with coordinate-PPO (sampled actions): baseline 98%, exponent −0.25 90%,
−0.5 85%, −0.75 57%, −1 55%, −1.5 42%, −2 36%. In the other three cases every row is at 96.7% or above.

## Predictions against outcomes

1. **Tabular-PPO, both grids: −1.5 and −2 are within 10% of −1 or ahead of it; −0.75 is 10% to 40% behind −1; −0.25
   is more than twice as slow as −1.** Holds: the means of −1.5 and −2 are 9% to 21% below the mean of −1; −0.75 is 11% and 24% behind;
   −0.25 is 3.4 and 17 times as slow.
2. **Coordinate-PPO, snake grid: −1.5 and −2 fail (goal found by fewer than 80% of seeds, score above 1,000K);
   −0.75 is within 20% of −1; −0.25 is within 50% of −1.** Holds in part. Both scores are above 1,000K (1.32M and
   5.53M), but the goal is found by 98% and 86% of seeds. −0.75 is 28% ahead of −1 on the same seeds, more than
   predicted. −0.25 has the same score as −1 (67.4K against 67.6K).
3. **Coordinate-PPO, straight grid: −2 is at least 50% behind −1; the others are within 30% of −1.** Holds for −2
   (230.2K against 2.41K; trimmed 3.80K against 2.20K), −0.75 (9% ahead) and −0.5 (28% behind). Fails for −0.25
   (5.72K, 2.4 times −1) and for −1.5 on the mean (15.64K; trimmed 2.58K, 17% behind).
4. **Refreshing once per rollout at −1 is behind the per-step refresh in all four cases, by 10% to 30% in three and
   by more than 50% on the snake grid with coordinate-PPO.** Holds in direction in all four cases. The per-rollout
   row is 16%, 94%, 15% and 67% slower on the same 512 seeds, so the straight grid with coordinate-PPO is larger than
   predicted.
5. **Exponent −1 refreshed every step stays the only row ahead of the baseline in all four cases.** Fails: exponent
   −0.75 refreshed every step is also ahead in all four.

## What the batch shows

1. The exponents ahead of the baseline in all four cases are −0.75 and −1. Slower schedules (−0.25, −0.5) are behind
   with tabular-PPO; faster ones (−1.5, −2) are behind or level with coordinate-PPO.
2. Tabular-PPO is fastest with a fast decay (−1.5) and coordinate-PPO with a slower one (−0.75 on both grids).
   Between −0.75 and −1 neither is ahead in all cases: −1 is ahead with tabular-PPO, −0.75 on the snake grid with
   coordinate-PPO.
3. On the snake grid with coordinate-PPO a slower decay keeps more of the final policy: −0.5 reaches the goal first
   at 60.1K (level with −1) and its final policy reaches it in 85% of evaluation episodes against 55% for −1.
4. The per-step refresh is ahead of the per-rollout refresh at the same schedule in all four cases.
