# Batch 7: decay exponent and refresh timing on the full protocol

Written 2026-11-10 12:23 PT, before any unit of the batch had finished. Times are Pacific. Exact settings: `rl_ideas.json`.

## Why this batch

Batch 6 runs two exponents on the full protocol (−1 and −0.5, both refreshed every step). The screens of batches 2
to 4 covered more exponents and both refresh timings, but only at 2^22 transitions on 32 seeds with a reduced grid.
To state how the decay exponent and the refresh timing matter with the same weight as the main comparison, the other
exponents and the per-rollout refresh get the full protocol too: the 52-configuration development grid on seeds 0-15
at 2^24 transitions, then confirmation of the chosen configuration on seeds from 20000 on.

Every row is the schedule-fit RND with the even start and the exact output-layer fit; only the schedule's exponent
or the refresh timing changes. Define $m$ as the visits of a cell so far; the bonus is $(1+m)^{-a}$ for decay
exponent $-a$.

## Rows

| row | changes against the finalist (exponent −1, refreshed every step) |
|---|---|
| exponent −0.25, every step | slower decay |
| exponent −0.75, every step | slower decay |
| exponent −1.5, every step | faster decay |
| exponent −2, every step | faster decay |
| exponent −1, once per rollout | refresh timing |

## Predictions (from the screen rows of batches 2 to 4)

1. **Tabular-PPO, both grids:** −1.5 and −2 are within 10% of −1 or ahead of it; −0.75 is 10% to 40% behind −1;
   −0.25 is more than twice as slow as −1.
2. **Coordinate-PPO, snake grid:** −1.5 and −2 fail (the goal is found by fewer than 80% of seeds and the score is
   above 1,000K); −0.75 is within 20% of −1; −0.25 is within 50% of −1.
3. **Coordinate-PPO, straight grid:** −2 is at least 50% behind −1; the others are within 30% of −1.
4. **Refresh once per rollout at −1** is behind the per-step refresh in all four cases, by 10% to 30% in three and
   by more than 50% on the snake grid with coordinate-PPO.
5. So −1 refreshed every step stays the only row that is ahead of the baseline in all four cases.
