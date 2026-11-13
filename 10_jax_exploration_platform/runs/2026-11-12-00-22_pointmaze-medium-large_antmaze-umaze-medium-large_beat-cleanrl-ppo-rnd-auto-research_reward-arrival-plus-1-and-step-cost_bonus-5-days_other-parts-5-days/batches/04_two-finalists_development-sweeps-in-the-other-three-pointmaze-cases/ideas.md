# Batch 4: the two finalists in the other three PointMaze cases

Written 2026-11-12 19:08 PT, before any run of the batch. Bonus-only changes (program.md, days 1 to 5); the
options exist in the frozen code of batch 2 (commit d7054945). No new idea is introduced here: the batch asks
whether what is ahead on PointMaze Large with reward 1 on arrival is also ahead in the other cases.

## What the validation block showed (256 seeds, PointMaze Large, reward 1 on arrival)

Each method at its best intrinsic weight of those run, against the control (CleanRL PPO-RND) at its best weight
of 1, 2 and 4 (weight 2: whole-curve mean 0.8781, final level 0.9920, mean capped first-goal time 269 thousand
steps). Differences are on the same seeds, with 95% intervals from resampling seeds.

| method | intrinsic weight | whole-curve mean ↑ | difference from the control | difference in the final level | difference in the first-goal time, thousand steps |
|---|---|---|---|---|---|
| all three parts (fast last layer at step 0.1, predictor at 0.25 times 1e-4 with its own optimizer, divisor held after 1/16 of the run) | 0.5 | **0.8917** | +0.0137 [+0.0072, +0.0203] | +0.0049 [+0.0036, +0.0063] | −109 [−139, −79] |
| slower predictor with held divisor (the first without the fast last layer) | 4 | <u>0.8883</u> | +0.0102 [+0.0047, +0.0159] | +0.0062 [+0.0052, +0.0073] | −117 [−146, −87] |
| divisor held after 1/16 of the run | 2 | 0.8876 | +0.0095 [+0.0040, +0.0150] | +0.0056 [+0.0045, +0.0068] | +1 [−27, +28] |
| predictor at 0.25 times 1e-4 with its own optimizer | 2 | 0.8859 | +0.0078 [+0.0015, +0.0144] | +0.0034 [+0.0014, +0.0051] | −99 [−129, −68] |
| all three parts, divisor held after 1/8 of the run | 0.5 | 0.8859 | +0.0078 [+0.0011, +0.0142] | +0.0034 [+0.0020, +0.0048] | −97 [−127, −67] |

Each method's weight was the best of two or three on these same seeds, so the differences are slightly
favourable to the methods; the control's weight was chosen the same way.

## The two finalists

1. **All three parts** (divisor held after 1/16): the largest difference on the whole-curve mean.
2. **Slower predictor with held divisor**: two parts instead of three and within the first one's interval.

## Sweeps (16 development seeds, learning rate annealed to zero over each case's budget, entropy 0.01, clip 0.1)

The learning rates are those of RND's sweep at the budget in the same case. The weights start two factors of 2
below RND's grid where the grid allows, because on PointMaze Large the first finalist's best weight (0.5) is a
quarter of RND's (2).

| case | budget in steps per policy | learning rates | intrinsic weights | RND's chosen configuration |
|---|---|---|---|---|
| PointMaze Medium, reward 1 on arrival | 1,048,576 | 3e-3, 6e-3, 1e-2 | 0.25, 0.5, 1, 2, 4 | learning rate 6e-3, weight 2 |
| PointMaze Medium, step cost | 1,048,576 | 3e-3, 6e-3, 1e-2 | 0.00049, 0.00195, 0.0078, 0.03125, 0.125 | learning rate 6e-3, weight 0.00195 |
| PointMaze Large, step cost | 33,554,432 | 2e-3, 3e-3, 4.5e-3 | 0.5, 1, 2, 4, 8 | learning rate 2e-3, weight 4 |

15 configurations per finalist and case, 6 units. The edge rule applies: a best weight or learning rate at an
end of its grid is extended before anything is confirmed.

## Predictions

1. **PointMaze Medium, reward 1 on arrival:** both finalists reach the goal sooner than RND and are level with
   or slightly ahead of it on the whole-curve mean (RND: 0.932 on fresh seeds), since the map is solved early
   and little is left to gain.
2. **PointMaze Medium, step cost:** RND's best weight there is nearly zero and PPO without a bonus is ahead of
   RND. A bonus-only change is not expected to be ahead of RND's nearly bonus-free configuration; if the
   finalists' best weight is also the smallest one, this case needs a change outside the bonus (days 6 to 10).
3. **PointMaze Large, step cost:** RND's seeds split into two groups there (about −0.72 and −0.90). The held
   divisor should lower the share of seeds in the worse group, because that group is the one that keeps
   exploring after the first arrivals.
