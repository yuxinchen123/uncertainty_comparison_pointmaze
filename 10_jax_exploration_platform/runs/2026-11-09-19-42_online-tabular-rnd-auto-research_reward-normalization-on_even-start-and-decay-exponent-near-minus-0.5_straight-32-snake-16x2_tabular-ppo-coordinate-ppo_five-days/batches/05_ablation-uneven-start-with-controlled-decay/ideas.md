# Batch 5: the ablation arm with an uneven start and a controlled decay

Written 2026-11-10 03:25 PT, before any unit of the batch had finished. Times are Pacific. Exact settings: `rl_ideas.json`.
Reward normalization is on. Policy learning rates 1e-3 and 1e-4, entropy 0.001 and 0.01, weights 2^-6 to 2^6 in steps
of 4×, seeds 0-31 (field seeds 0-7), 2^22 transitions.

## Why this batch

The goal names two parts: an even start and a controlled decay. Whether a gain needs both is a 2×2 comparison:

| | decay left to the predictor's training | decay set by a schedule |
|---|---|---|
| **uneven start** | control (CleanRL online tabular-RND) | this batch |
| **even start** | unit start alone, and the initial-ratio readout (batch 1) | count rule on the uniform field (batches 2 to 4) and the schedule-fit RND (batch 4) |

The missing cell is the count rule applied to each cell's untrained tabular-RND error: the bonus starts as uneven as
the control's (max/min 3.65 on the straight grid and 4.23 on the snake grid on the bench) and then follows the
schedule exactly. The field is scaled to average 1 over cells, as every count-rule field is.

## Rows

| row | compared with (same rule on the uniform field) |
|---|---|
| exponent −1, refreshed every step | batch 3 row of the same rule |
| exponent −1 at a quarter of the count rate, refreshed every step | batch 4 row |
| exponent −1 at a quarter of the count rate, refreshed per rollout | batch 3 row |
| exponent −0.5, refreshed every step | batch 2 row |

## Predictions

1. Each row is within 30% of the same rule on the uniform field in all four cases: with one-hot inputs the start is
   uneven by a factor of about 4, far less than the coordinate-network fields of the 2026-10-31 study.
2. So most of the gain over the control comes from the decay and its refresh, and little from the even start.
3. The rows with exponent −1 are at or below the control in all four cases even with the uneven start.

If prediction 1 fails in a case (the uneven field is much slower there), the even start matters in that case and the
2×2 table will show it.
