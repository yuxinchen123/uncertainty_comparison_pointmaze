# Batch 6 summary: the finalists and the ablation on the full protocol

Written 2026-11-11 10:55 PT, with all 80 development units and all 112 confirmation units finished and none failed.
Times are Pacific. Each row's configuration was chosen on the development seeds 0-15 from 52 configurations at 2^24
transitions, then trained on the confirmation seeds from 20000 on. Scores are the mean capped first-goal time at 2^24
transitions. The difference to the baseline is on the seeds both rows have, as a percentage of the baseline's mean;
a row is ahead when the whole 95% interval of the difference (resampling field seeds) is below zero, behind when it is
above zero, level when it contains zero. Full tables: `analysis/analysis.md`, section "confirmation".

## Result

| row | seeds | straight, tabular-PPO | straight, coordinate-PPO | snake, tabular-PPO | snake, coordinate-PPO | cases ahead |
|---|---|---|---|---|---|---|
| baseline: CleanRL online tabular-RND (uneven start, decay left to training) | 1,024 | 7.56K | 81.7K (trimmed 4.74K) | 18.47K | 157.0K (trimmed 91.0K) | — |
| schedule-fit RND, exponent −1, refreshed every step (even start, scheduled decay) | 1,024 | 6.12K, −19%, ahead | 2.41K, −97%, ahead | 12.84K, −31%, ahead | 67.6K, −57%, ahead | 4 of 4 |
| schedule-fit RND, exponent −0.5, refreshed every step | 1,024 | 8.04K, +6%, behind | 3.09K, −96%, ahead | 32.0K, +73%, behind | 60.1K, −62%, ahead | 2 of 4 |
| count rule on the frozen tabular-RND field, exponent −1, refreshed every step (uneven start, scheduled decay) | 512 | 6.48K, −14%, ahead | 2.39K, −98%, ahead | 14.07K, −22%, ahead | 128.1K, −9%, level | 3 of 4 |
| unit start alone (even start, decay left to training) | 512 | 8.67K, +15%, behind | 5.23K, −95%, ahead on the mean only (trimmed 4.88K) | 20.35K, +13%, behind | 8.37M, behind (goal found by 52%) | 1 of 4 |

Chosen configurations of the finalist (exponent −1, every step): policy learning rate 1e-3 in all four cases; entropy
0.001, 0.01, 0.01, 0.01; intrinsic weight 2^-6, 2^-5, 2^-6, 2^6. Baseline: learning rate 1e-3, 1e-3, 1e-3, 1e-4;
entropy 0.001, 0.001, 0.001, 0.01; weight 2^-4, 2^-6, 2^5, 2^5.

The two parts compared with each other on the same 512 seeds (first row minus second row):

| comparison | straight, tabular-PPO | straight, coordinate-PPO | snake, tabular-PPO | snake, coordinate-PPO |
|---|---|---|---|---|
| even start against uneven start, both with the scheduled decay | −5%, ahead | −2%, level | −9%, ahead | −43%, ahead |
| scheduled decay against decay left to training, both with the even start | −29%, ahead | −55%, ahead | −37%, ahead | −99%, ahead |

## Predictions against outcomes

1. **The finalist's chosen configuration is at policy learning rate 1e-3 in all four cases.** Holds.
2. **The finalist is ahead of the baseline in all four cases on the 1,024 seeds, the straight grid with tabular-PPO
   being the closest.** Holds, and the interval excludes zero in all four, that case included (−1.44K, interval
   −1.62K to −1.26K).
3. **The exponent −0.5 form is ahead with coordinate-PPO and behind with tabular-PPO.** Holds.
4. **The scheduled decay gives most of the gain; the even start adds 7% to 25%; the even start alone is behind the
   baseline.** Holds in part. The scheduled decay is ahead of the decay left to training in all four cases. The
   even start adds 5% and 9% with tabular-PPO, nothing measurable on the straight grid with coordinate-PPO, and 43%
   on the snake grid with coordinate-PPO. The even start alone is behind the baseline in three cases; on the
   straight grid with coordinate-PPO its mean is lower only because three baseline seeds never reach the goal.

## Notes for the report

1. On the straight grid with coordinate-PPO three of the baseline's 1,024 seeds never reach the goal and carry 60%
   of its mean. With the lowest and highest 80 seeds removed the finalist is at 2.20K against 4.74K.
2. On the snake grid with coordinate-PPO the finalist's final policy reaches the goal in 55% of evaluation episodes
   against the baseline's 98%; its chosen weight is 2^6, the largest of the grid. Three of its four chosen weights
   are at an edge of the weight grid.
3. The baseline's development grid was trained twice (by the extension run and here). The same seed does not give
   the same run on another card model: 6.9% of pairs are identical, and the two copies agree on average.

## Against the success rule of `program.md`

1. **Below the baseline in each of the four cases on the same 1,024 seeds:** met.
2. **Even-start target (ratio at most 1.5):** met, the ratio is 1.00.
3. **Decay target (median slope between −0.6 and −0.4):** not met by the finalist. Its schedule has exponent −1 and
   its measured slope is −0.67 to −0.77. The exponent −0.5 form meets the target (−0.47 to −0.48) and is ahead in two
   of four cases.
4. **The ablation shows whether the gain needs both properties:** it does in three of four cases, and the decay is
   the larger part.
5. **Novelty:** limited. With one-hot inputs the method computes the bonus of a count rule (`literature/novelty.md`).
