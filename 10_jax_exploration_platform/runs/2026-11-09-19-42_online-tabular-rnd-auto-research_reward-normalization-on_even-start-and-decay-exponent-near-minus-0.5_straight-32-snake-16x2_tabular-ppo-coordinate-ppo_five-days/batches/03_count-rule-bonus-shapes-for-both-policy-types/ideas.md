# Batch 3: which bonus shape works for both policy types (count-rule runs)

Written 2026-11-10 01:35 PT, before any unit of the batch had finished. Times are Pacific. Exact settings:
`rl_ideas.json`. These are count-rule runs on the uniform field, not RND variants: they take minutes each, and batch 2
showed that the stable RND candidate matches the count rule of the same form. Reward normalization is on. Policy
learning rates 1e-3 and 1e-4, entropy 0.001 and 0.01, weights 2^-6 to 2^6 in steps of 4×, seeds 0-31, 2^22 transitions.

## Why this batch

Batch 2's count-rule rows (bonus fixed during a rollout, 32 seeds, best setting) show that the best decay exponent
depends on the policy type:

| decay exponent | straight, tabular-PPO | snake, tabular-PPO | snake, coordinate-PPO |
|---|---|---|---|
| −2 | 7.56K ± 0.61K | 14.1K ± 1.6K | 3,288K ± 282K (goal found by 25%) |
| −1 | 7.73K ± 0.47K | 16.3K ± 0.9K | 116.9K ± 41.6K |
| −0.5 | 9.70K ± 0.72K | 40.3K ± 5.5K (quarter-speed count; full speed not finished) | 57.0K ± 4.4K |
| −0.25 | 22.2K ± 3.2K | 227K ± 55K | 65.8K ± 8.0K |
| control (CleanRL tabular-RND) | not finished | 18.7K ± 1.0K | 82.1K ± 12.0K |

So with tabular-PPO a faster decay is better, and with coordinate-PPO on the snake grid −0.5 is best and −2 fails.
A method fixed at −0.5 cannot be best in all four cases.

## The question

Is there one bonus shape that is good in all four cases? Define $m$ as a cell's earlier visits (those before the
current rollout) and $c$ as the count rate.

| rows | shape | question |
|---|---|---|
| exponents −0.75, −1.5, −3 | $(1+m)^{-a}$ | where between −0.5 and −2 does each policy type turn, and is −3 better still for tabular-PPO? |
| per step, exponents −1 and −2 | $(1+m)^{-a}$ with $m$ counted at every arrival | does fixing the bonus during a rollout matter at fast exponents? |
| −2 plus 0.03, 0.1, 0.3 of −0.5 | $(1+m)^{-2} + \lambda (1+m)^{-1/2}$ | a sharp drop for tabular-PPO with a slow tail for coordinate-PPO |
| −1 plus 0.1, 0.3 of −0.5 | $(1+m)^{-1} + \lambda (1+m)^{-1/2}$ | the same with a milder drop |
| −2 at count rates 0.25 and 4; −1 at 0.25 and 4 | $(1 + c\thinspace m)^{-a}$ | how fast the drop must be |

## Predictions

- Tabular-PPO: −1.5 and −3 score like −2 (within 10%).
- Coordinate-PPO on the snake grid: the score worsens steadily from −0.75 to −3.
- A mixture with a slow tail of weight 0.1 to 0.3 keeps tabular-PPO within 15% of its best and brings the snake grid
  with coordinate-PPO under the control's 82K.

If a mixture does that, the RND method gets the same shape: the shrink map generalizes to any decreasing bonus
schedule $b(m)$ through $\rho \to b(b^{-1}(\rho) + 1)$.
