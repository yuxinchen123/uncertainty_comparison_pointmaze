# Batch 1: the start, the scale and the decay of the bonus on continuous inputs

Written 2026-11-12 11:50 PT, before any run of the batch. Bonus-only changes (program.md, days 1 to 5): PPO, the
policy and value networks and the sweep are those of the RND baseline.

## What the baseline shows (PointMaze, first pass, 16 seeds)

1. **The bonus is needed on PointMaze Large with reward 1 on arrival:** with a negligible intrinsic weight no policy
   reaches the goal; with weight 1 all 16 do, the median first arrival is at 421 thousand steps, and the share of
   episodes that reach the goal passes 0.9 near 2 million steps.
2. **Most of the baseline's loss of whole-curve mean is before and just after the first arrival:** the curve needs
   about 1.7 million steps after the median first arrival to pass 0.9.
3. **A too-large weight finds the goal in every seed but ends low and falls late** (weight 16: final level 0.22 to
   0.45; weight 64 worse). The intrinsic reward is divided by a running standard deviation, so its size relative to
   the task reward does not fall as the maze becomes familiar.
4. **PointMaze Medium is nearly solved without a bonus**, so only the first few hundred thousand steps can differ.

## Screen

Each idea is one change against the control, on PointMaze Large with reward 1 on arrival, at that case's budget, 16
seeds (0 to 15), with the baseline's chosen learning rate and entropy coefficient and the intrinsic weight swept over
the 10 values of the first pass (the most important setting first). An idea with its own setting adds at most three
values of it. AntMaze U-maze joins the screen when its baseline exists. Measure: whole-curve mean, with the final
level, the median first-arrival step and the two property measures below beside it.

Two property measures, read on a fixed grid of maze positions (velocity zero) every logging window:

1. **Evenness of the start:** the largest over the smallest initial bonus on the grid.
2. **Decay:** per grid cell, the slope of the logarithm of the bonus against the logarithm of the visits to that
   cell, and how far the bonus has fallen at cells with no visit yet (how much one cell's training lowers another's
   bonus).

## The control and the ideas

| name | change against the control | own setting |
|---|---|---|
| control | CleanRL RND as in the baseline | none |
| diagnostic: bonus recorded on the maze grid | no change to training; the two property measures above are logged | none |
| even start | the predictor's output layer starts at zero and the target's output is scaled to unit length per input, so the first error is the same at every input | none |
| fixed scale | the divisor of the intrinsic reward is frozen at its value after the random-action rollouts, instead of following the running standard deviation | none |
| even start with no divisor | even start, and the intrinsic reward is the error itself (it starts at 0.5 everywhere and only falls) | none |
| separate predictor step size | the predictor gets its own Adam and its own learning rate, not annealed with the policy's | predictor learning rate 0.25, 1 and 4 times 1e-4 |
| error length as the bonus | the bonus is the length of the error vector instead of half its square | none |
| fast last layer inside the rollout | between two policy updates, the predictor's output layer takes one small step at every visited state, so a state seen twice in one rollout pays less the second time | step size 0.1, 0.3, 1 of the exact fit |
| scheduled last layer on fixed features | the predictor's body is fixed; its output layer is fitted in closed form once per rollout so that each visited input's error moves along $(1+m)^{-a}$, with $m$ read from the error itself; bonus refreshed every step | decay $a$ = 0.5, 0.75, 1 |

## How the screen is run (settled 2026-11-12 13:20 PT, before any run)

- **Learning rate 2e-3, entropy 0.01, clip 0.1:** RND's chosen configuration for this case (development runs at the
  budget).
- **Intrinsic weights, 12 values:** the 10 of the first pass ($4^{-6}$ to $4^{3}$) plus 0.5 and 2, so that the
  control's own best weight (2) is in the grid and the ideas are read at factor 2 around it. One unit per row of
  the table: 12 weights × 16 seeds = 192 policies in one process.
- **The grid diagnostic is on in every unit** (it does not change training): per logging window and policy, the
  bonus at one probe per open cell (cell centre, velocity zero), the divisor, and the visits to each cell.
- **Fixed scale:** each random-action rollout at the start is scored and its discounted sum enters the divisor's
  statistics, as a training rollout's would; the divisor is what those statistics hold when the random-action
  rollouts end. (Before this change those rollouts never touched the divisor, so "its value after them" was 1.)
- **The output layer in the two last-layer ideas includes its bias:** a constant 1 is appended to the last hidden
  features before they are scaled to unit length, so an input whose hidden units are all off still has a feature
  vector.
- **Both last-layer ideas work step by step inside the rollout,** not once per rollout: after the states of one
  step are scored, the layer takes the smallest change that moves their errors. Environments of one policy in the
  same state at the same step count as one visit.
- **Scheduled last layer:** the squared error at an input never visited can rise slightly above 1 when other inputs
  are fitted; it is not clipped. The input statistics keep moving as in the baseline, so the schedule is exact at
  the moment of fitting and drifts afterwards. If the diagnostic shows that the drift matters, frozen input
  statistics are the follow-up.
- **Decay slope:** computed in the analysis from the recorded probes and visits.

## Predictions (written before the runs)

1. **Even start alone** changes the whole-curve mean by less than its interval; its start ratio is 1.00 against well
   above 2 for the control.
2. **Fixed scale** and **even start with no divisor** raise the final level at large weights (no late fall) and move
   the best weight upward; at the best weight the whole-curve mean rises by 0.02 to 0.05, because the bonus stops
   competing with the task reward once the maze is familiar.
3. **Separate predictor step size:** a faster predictor (4 times) lowers the median first-arrival step but lowers
   the final level; a slower one does the opposite; the best whole-curve mean is at 1 or 4 times.
4. **Error length** behaves like a slower decay: later first arrival, same final level.
5. **Fast last layer inside the rollout** lowers the median first-arrival step by at least 15%.
6. **Scheduled last layer on fixed features** has the earliest median first arrival of the batch at $a$ = 1, and the
   highest final level at $a$ = 0.5; whether its whole-curve mean is ahead of the control depends on how much one
   input's fit lowers the error at inputs not yet visited, which the diagnostic measures.

## What would make an idea new, and what would not

Zero-initialised output layers, frozen normalisation, separate optimisers and other loss powers are general tools.
The claim, if the batch supports it, is their combination into a bonus whose start and decay are set by
construction and that needs no running normalisation. Published variants (NGU, NovelD, DRND, coin-flip networks,
E3B, random distribution distillation, self-supervised network distillation) are not re-implemented; the closest
ones to the last two rows are E3B (an elliptical bonus from a feature covariance) and coin-flip networks (a learned
count), and `literature/novelty.md` will state the exact difference before a finalist is chosen.
