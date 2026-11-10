# Batch 6: the finalists on the full protocol (development stage)

Written 2026-11-10 04:24 PT, before any unit of the batch had finished. Times are Pacific. Exact settings: `rl_ideas.json`.

## The choice

At the screen (32 seeds, best of 28 settings, 2^22 transitions, thousands of transitions) the RND whose error follows
the schedule with exponent −1, refreshed at every step, is ahead of the control in all four cases:

| row | straight, tabular-PPO | straight, coordinate-PPO | snake, tabular-PPO | snake, coordinate-PPO |
|---|---|---|---|---|
| control (CleanRL online tabular-RND) | 7.22 ± 0.28 | 3.90 ± 0.20 | 18.74 ± 0.96 | 82.1 ± 12.0 |
| schedule-fit RND, exponent −1, refreshed every step | 6.33 ± 0.29 | 1.96 ± 0.29 | 13.51 ± 0.72 | 42.1 ± 2.8 |
| count rule of the same shape (even field) | 6.47 ± 0.26 | 1.70 ± 0.26 | 13.31 ± 0.74 | 41.3 ± 2.8 |
| the same count rule on the uneven frozen tabular-RND field | 6.91 ± 0.60 | 2.03 ± 0.26 (settings unfinished) | 15.06 ± 1.44 | 51.5 ± 7.1 |
| unit start alone (even start, decay left to training; batch 1) | 8.57 ± 0.64 | 4.66 ± 0.22 | 21.74 ± 1.24 | 1,905 ± 292 |
| schedule-fit RND, exponent −0.5, refreshed every step | not finished (its count rule: 8.42) | 2.29 ± 0.23 | 32.2 ± 4.4 | 49.0 ± 5.1 |

## Rows of this batch

Each row runs the fixed-settings study's own development protocol: policy learning rate 1e-4 and 1e-3, entropy 0.001
and 0.01, 13 weights from 2^-6 to 2^6 (52 configurations), seeds 0-15, 2^24 transitions, reward normalization on.
The baseline's development grid under the same protocol is in the extension run.

| row | role |
|---|---|
| schedule-fit RND, exponent −1, refreshed every step | finalist |
| schedule-fit RND, exponent −0.5, refreshed every step | the form that meets both property targets of `program.md` (even start, slope −0.5) |
| count rule, exponent −1 every step, on the frozen tabular-RND field | ablation: uneven start, scheduled decay |
| unit start alone | ablation: even start, decay left to training |

After this stage each row's configuration is chosen per case by the study's rule (smallest mean capped first-goal
time, then the larger discovery rate) and confirmed on seeds 20000-21023, the baseline's own confirmation seeds.

## Predictions

1. The finalist's chosen configuration is at policy learning rate 1e-3 in all four cases, as at the screen.
2. On the 1,024 confirmation seeds the finalist is ahead of the baseline in all four cases. The straight grid with
   tabular-PPO is the closest (12% at the screen); the paired 95% interval excludes zero in the other three.
3. The exponent −0.5 form is ahead of the baseline with coordinate-PPO and behind it with tabular-PPO.
4. The 2×2 comparison on the confirmation seeds: the scheduled decay gives most of the gain; the even start adds
   7% to 25% on top of it; the even start with the decay left to training is behind the baseline.

## Addition of 2026-11-10 05:44 PT: the baseline's development grid is also run here

The extension run trains the baseline's development grid on UVA cards, and it is slower than estimated: at 05:42 the
baseline's own units needed up to 12 more hours, and two other units of that run sit on Tesla P100 cards with 22 and
37 hours left; the extension's confirmation stage starts only when all of them have finished. So the same grid is
queued here as the row `control` (online tabular-RND, reward normalization on, no change), on H100 cards:

- same protocol, seeds and settings as the extension's row, so each (configuration, seed) pair trains the same run;
- when the extension's own units finish, the two sets are compared pair by pair as a check of both paths;
- the 1,024-seed confirmation of the baseline's chosen configuration then runs here as well, so the final comparison
  does not wait for the extension.

## Addition of 2026-11-10 08:24 PT: the two copies of the baseline's grid agree on average, not run by run

Check on the first finished case (snake grid, coordinate-PPO; 832 pairs of the same configuration and seed, script
`analysis/compare_baseline_paths.py`, job in `logs/compare_baseline_paths_*.out`):

- only 57 of 832 pairs (6.9%) have the same first-goal time in both copies;
- the means over all 832 pairs are 2,002K (extension copy, UVA cards) and 2,085K (this run's copy, MIT cards);
- the extension copy chooses learning rate 1e-4, entropy 0.01, weight 2^-6 (76.4K; 83.9K in this run's copy), and this
  run's copy chooses learning rate 1e-4, entropy 0.01, weight 2^5 (74.8K; 89.9K in the extension copy).

Reading:

1. The earlier statement that each pair "trains the same run" is wrong across cards. A different card model or a
   different grouping of copies changes floating-point rounding, and over 131,072 updates the runs part. Each copy
   is a valid draw of the same protocol; they are equal in distribution, not seed by seed.
2. The baseline's choice between weights at learning rate 1e-4 and entropy 0.01 is within the noise of 16 seeds: the
   two copies pick different weights with scores 2% apart.
3. For the final comparison the baseline row confirmed here uses this run's choice. When the extension confirms its
   own choice, that row is added beside it, so the finalist is compared with both.
4. The difference on the same seeds stays a valid comparison (same field seeds and networks on both sides), but
   pairing removes less noise than it would if runs were reproducible across cards.
