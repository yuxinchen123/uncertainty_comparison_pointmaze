# Batch 6: the predictor at half speed with held divisor, a second validation block, and the PointMaze Large cases

Written 2026-11-12 22:02 PT, before any run of the batch. Bonus-only changes (program.md, days 1 to 5); every option
exists in the frozen code of batch 2 (commit d7054945). No new idea: the batch checks, on seeds not used for any
choice so far, what batches 4 and 5 found.

## What batches 4 and 5 showed

**PointMaze Medium, reward 1 on arrival, validation block (256 seeds, 10000 to 10255).** Each arm at the better of
its two configurations against CleanRL PPO-RND at the best of its three (0.9349):

| arm | whole-curve mean ↑ | difference from RND [95%] | difference in the final level [95%] |
|---|---|---|---|
| predictor at half speed with held divisor (learning rate 6e-3, weight 1) | **0.9438** | +0.0089 [+0.0012, +0.0166] | +0.0050 [+0.0034, +0.0071] |
| held divisor alone, held after 1/64 of the run (1e-2, weight 2) | <u>0.9388</u> | +0.0039 [−0.0035, +0.0111] | +0.0049 [+0.0033, +0.0070] |
| held divisor alone, held after 1/16 of the run (6e-3, weight 4) | 0.9382 | +0.0033 [−0.0043, +0.0108] | +0.0044 [+0.0027, +0.0066] |
| slower predictor, divisor held after 1/64 of the run (1e-2, weight 1) | 0.9362 | +0.0013 [−0.0069, +0.0096] | +0.0050 [+0.0033, +0.0072] |
| slower predictor with held divisor, the finalist of batch 4 (3e-3, weight 2) | 0.9348 | −0.0001 [−0.0082, +0.0085] | +0.0050 [+0.0033, +0.0070] |

One arm of five has its interval above zero, and its lower end is +0.001. It is the best of five arms and of two
configurations on these seeds, so the difference is too favourable by an unknown amount.

**PointMaze Large, step cost, 16 development seeds** (against RND's best of 12, −0.7247): all three parts: −0.7187,
+0.0060 [+0.0044, +0.0075]; slower predictor with held divisor: −0.7222, +0.0025 [−0.0042, +0.0072], with its best
weight now inside its grid (8; 16 gives −0.723, 32 gives −0.737).

## What is run

1. **A second validation block, seeds 20000 to 20255, PointMaze Medium with reward 1 on arrival.** No choice has
   been made on these seeds. RND at its three configurations; the predictor at half speed with held divisor at its
   two; the held divisor alone (after 1/16 and after 1/64) at one each. 7 units.
2. **The predictor at half speed with held divisor on PointMaze Large with reward 1 on arrival**, first
   validation block, learning rate 2e-3, weights 1, 2 and 4 (the grid the other arms had there). 3 units.
3. **PointMaze Large with the step cost on the first validation block:** RND at its two best development
   configurations; all three parts at its two best; the slower predictor with held divisor at its two best; the
   predictor at half speed with held divisor at learning rate 3e-3 and weights 4 and 8. 8 units.

## Predictions

1. **Second block, PointMaze Medium:** the predictor at half speed with held divisor is again above RND, by less
   than on the first block (between 0 and +0.009); its interval is above zero only if the true difference is at
   least about 0.008, which I do not expect.
2. **PointMaze Large, reward 1 on arrival:** the predictor at half speed with held divisor is within 0.005 of the
   slower predictor with held divisor (0.8883 at weight 4) at its best weight, and ahead of RND (0.8781) with its
   interval above zero.
3. **PointMaze Large, step cost:** all three parts stays ahead of RND with its interval above zero on 256 seeds;
   the slower predictor with held divisor is level.
