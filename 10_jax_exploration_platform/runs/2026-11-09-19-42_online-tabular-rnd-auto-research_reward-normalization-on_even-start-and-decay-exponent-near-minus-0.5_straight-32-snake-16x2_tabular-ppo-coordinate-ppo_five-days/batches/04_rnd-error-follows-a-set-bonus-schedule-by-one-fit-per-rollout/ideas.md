# Batch 4: an RND whose error follows a set bonus schedule, fitted once per rollout

Written 2026-11-10 02:40 PT, before any unit of the batch had finished. Times are Pacific. Exact settings: `rl_ideas.json`.
Reward normalization is on. Policy learning rates 1e-3 and 1e-4, entropy 0.001 and 0.01, weights 2^-6 to 2^6 in steps
of 4×, seeds 0-31, 2^22 transitions.

## Why this batch

Batch 3 (137 of 224 units finished at 02:35) compares count-rule shapes. Define $m$ as a cell's earlier visits and
$c$ as the count rate; the bonus is $(1 + c\thinspace m)^{-a}$ for decay exponent $-a$. Mean capped first-goal
time, 32 seeds, best of 28 settings, in thousands of transitions:

| row | straight, tabular-PPO | straight, coordinate-PPO | snake, tabular-PPO | snake, coordinate-PPO |
|---|---|---|---|---|
| control (CleanRL online tabular-RND) | 7.22 ± 0.28 | 3.90 ± 0.20 | 18.74 ± 0.96 | 82.1 ± 12.0 |
| count rule, −0.5, refreshed per rollout | 9.70 ± 0.72 | 3.52 ± 0.32 | 42.8 ± 6.3 | 57.0 ± 4.4 |
| count rule, −0.5, refreshed per step | 8.42 ± 0.39 | 2.38 ± 0.22 | 32.3 ± 4.5 | 52.7 ± 7.8 |
| count rule, −1, per rollout | 7.73 ± 0.47 | 3.09 ± 0.42 | 16.25 ± 0.88 | 116.9 ± 41.6 |
| count rule, −1 at rate 0.25, per rollout | 7.08 ± 0.37 | not finished | 15.92 ± 0.79 | 63.8 ± 9.7 |
| count rule, −1, per step | not finished | not finished | not finished | 41.3 ± 2.8 |
| count rule, −2, per rollout | 7.56 ± 0.61 | 5.77 ± 0.29 | 14.1 ± 1.6 | 3,288 ± 282 |
| count rule, −1 plus 0.3 of −0.5, per rollout | 9.76 ± 1.17 | not finished | 26.2 ± 2.2 | 56.8 ± 3.8 |

Reading:

1. The −0.5 rule is ahead of the control with coordinate-PPO and behind it with tabular-PPO, refreshed either way.
2. Exponent −1 at a quarter of the count rate is level with or ahead of the control in the three cases finished.
3. Refreshing the bonus at every step is better than once per rollout in every pair measured.
4. Sums of a fast and a slow power law are not better than −1 alone.

## The method tested here

An RND with one-hot inputs, built so that its error follows a chosen schedule exactly:

1. **Equal start.** The predictor's output layer starts at zero and the target's features have unit length, so every
   cell's error length starts at exactly 1.
2. **A schedule carried by the error.** Define $b(m)$ as the wanted bonus after $m$ visits, with $b(0) = 1$, and
   $\rho$ as a cell's current error length. The cell's place on the schedule is $b^{-1}(\rho)$; no count is stored.
3. **One fit per rollout.** After a rollout in which a cell was a state $n$ times, the cell's output is moved toward
   the target so that its error length becomes $b(b^{-1}(\rho) + n)$. Cells not seen keep their outputs. All cells
   are fitted together:
   - **exact output-layer fit:** only the output layer changes, by the smallest change that gives every cell exactly
     its new output (one linear solve of the size of the cell count);
   - **all-layers fit:** every layer changes, by one least-squares step on the linearized predictor (12
     conjugate-gradient iterations), so it is approximate.
4. **Refresh per step (exact fit only).** During the rollout an arrival earns the error the predictor would have after
   also training on the states already seen in that rollout, $b(b^{-1}(\rho) + k)$ for $k$ such states. With the
   exact fit this is what a predictor updated at every step would give; the weights are still written once per
   rollout.

What it reads: how many times each cell appears in the rollout it is training on. It keeps no visit table and never
reads the step counter.

**Limit of the claim.** With one-hot inputs and the exact fit, the error length is a function of the cell's visit
count alone, so the method computes the same bonus as a count rule of the same shape. The test
`test_exact_output_layer_fit_is_the_count_rule_in_every_cell` checks this. Its place is as the RND that has both
wanted properties exactly; whether such an RND beats the control is what the batch measures.

## Bench and speed (serval05, 2026-11-10 02:20 to 02:29, card free)

| idea | start max/min | slope, uniform visits (straight, snake) | slope, one-cell rollouts | spread between cells, uniform visits | unvisited cells kept | seconds per iteration at 512 copies |
|---|---|---|---|---|---|---|
| control | 3.65, 4.23 | −3.1, −1.5 | — | — | — | 0.105 |
| exact fit, −0.5 | 1.00 | −0.49, −0.49 | −0.50 | 0.00 | 1.000 | 0.015 |
| exact fit, −1 | 1.00 | −0.97, −0.97 | −1.00 | 0.01 | 1.000 | 0.015 |
| exact fit, −1 plus 0.3 of −0.5 | 1.00 | −0.59, −0.60 | −0.53 | 0.00 | 1.000 | 0.016 |
| all-layers fit, −0.5 | 1.00 | −0.45, −0.40 | −1.03 | 0.12, 0.10 | 0.998, 1.002 | 0.106 |

- The exact fit is 7 times as fast as the control, and the all-layers fit costs the same as the control.
- Two variants were measured and dropped from the code:
  - all-layers fit with every visited cell weighted equally: slope −0.19 and −0.12, spread 1.4;
  - all-layers fit with 24 iterations and two steps: 3.7 times the control's cost.
- The all-layers fit decays twice as fast when each rollout sits in one cell (−1.03 against −0.50), because a change
  made for one cell also moves the others toward their targets. The exact fit does not.

## Rows

| rows | what they answer |
|---|---|
| exact fit, −0.5, refreshed per rollout and per step | the RND with both wanted properties exactly: does it match its count rule, and does it beat the control? |
| exact fit, −1 at rate 0.25, per rollout and per step | the shape batch 3 points to |
| exact fit, −1 and −0.75, per step | neighbours of that shape |
| all-layers fit, −0.5 and −1 at rate 0.25 | does training every layer cost anything? |
| count rule, −1 per step at rates 0.5, 0.25, 0.125, 0.0625 | how slow the count rate should be |
| count rule, −1 per rollout at rates 0.5, 0.125, 0.0625 | the same, refreshed per rollout |
| count rule, −0.75 per step; −1.5 per step at rates 0.25 and 0.0625 | other exponents with a per-step refresh or a slow rate |

## Predictions

1. Each exact-fit row scores within two standard errors of the count rule of the same shape and refresh, in all four
   cases.
2. Measured in RL, every exact-fit row has a start ratio of 1.00, and the −0.5 rows a median per-cell slope between
   −0.52 and −0.48.
3. Exact fit, −0.5: ahead of the control with coordinate-PPO and behind it with tabular-PPO (as its count rule).
4. Exact fit, −1 at rate 0.25, refreshed per step: at or below the control in all four cases.
5. The all-layers rows are within 20% of the exact rows on the straight grid and worse on the snake grid.

Prediction 4 is the one that matters for the goal. If it holds, the candidate method is this RND with the −1
schedule at a quarter rate. That schedule meets the even-start target and not the decay target of −0.5 in
`program.md`; the −0.5 form meets both targets and, by the batch 3 rows, is not ahead with tabular-PPO.
