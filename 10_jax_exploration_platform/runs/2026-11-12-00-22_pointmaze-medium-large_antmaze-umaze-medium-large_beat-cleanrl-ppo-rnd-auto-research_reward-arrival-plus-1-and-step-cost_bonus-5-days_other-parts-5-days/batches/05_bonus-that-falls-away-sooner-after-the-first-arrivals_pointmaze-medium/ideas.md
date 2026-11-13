# Batch 5: a bonus that falls away sooner after the first arrivals (PointMaze Medium, reward 1 on arrival)

Written 2026-11-12 21:03 PT, before any run of the batch. Bonus-only changes (program.md, days 1 to 5); every option
exists in the frozen code of batch 2 (commit d7054945), so no code changes.

## What the earlier runs showed

On the 256 validation seeds of PointMaze Medium with reward 1 on arrival, the slower predictor with held divisor
is level with CleanRL PPO-RND on the whole-curve mean (0.9348 against 0.9349) although it reaches the goal sooner
and ends higher. The mean training return of the 256 policies shows where it loses:

| environment steps, thousand | 16 | 33 | 49 | 66 | 98 | 131 | 197 | 262 | 1,049 |
|---|---|---|---|---|---|---|---|---|---|
| CleanRL PPO-RND (learning rate 6e-3, weight 2) | 0.132 | 0.279 | 0.453 | 0.613 | 0.816 | 0.916 | 0.969 | 0.983 | 0.995 |
| slower predictor with held divisor (learning rate 3e-3, weight 2) | 0.184 | 0.382 | 0.499 | 0.584 | 0.715 | 0.840 | 0.953 | 0.986 | 1.000 |

It is ahead until about 50 thousand steps and behind from 66 to 200 thousand. In that stretch the intrinsic
reward the learner sees (after the divisor) is 30 to 60 times RND's: 0.41 against 0.013 at 66 thousand steps,
0.18 against 0.006 at 98 thousand, 0.09 against 0.003 at 131 thousand. Two parts of the method cause it:

1. the predictor at 0.25 times its learning rate keeps the error high for longer (that is what finds the goal
   sooner);
2. the divisor is held after 1/16 of the run, which is 66 thousand steps here: later than the first arrivals
   (median 12 thousand steps), so the divisor has already shrunk with the bonus when it is frozen.

## Arms (one change each against the slower predictor with held divisor; RND is the control)

| arm | predictor learning rate, times 1e-4 | divisor held after | why |
|---|---|---|---|
| held divisor alone | 1 (joint optimizer, as RND) | 1/16 of the run | without the slower predictor the bonus falls as fast as RND's; never run in this case |
| held divisor alone, held earlier | 1 (joint optimizer, as RND) | 1/64 of the run | an earlier hold keeps the larger early divisor, so the later bonus is smaller |
| slower predictor, divisor held earlier | 0.25 (own optimizer) | 1/64 of the run | keeps the earlier first arrival and shrinks the later bonus |
| predictor at half speed with held divisor | 0.5 (own optimizer) | 1/16 of the run | between RND and the finalist |

Sweep per arm: learning rates 3e-3, 6e-3, 1e-2; intrinsic weights 1, 2, 4; entropy 0.01; clip 0.1; 16 development
seeds; 1,048,576 steps per policy. 36 configurations, 4 units (one per arm). The best two configurations of each arm then go to
the validation block, because a difference of 0.02 on 16 seeds is noise in this case.

## Predictions

1. **Held divisor alone (1/16)** is level with RND on the whole-curve mean (within 0.01 on the validation
   seeds) and ahead on the final level, as on PointMaze Large.
2. **Holding the divisor after 1/64 instead of 1/16** raises the return between 66 and 200 thousand steps for
   both predictor speeds, and with the slower predictor it is ahead of RND on the whole-curve mean on the
   validation seeds (interval above zero).
3. **The predictor at half speed** lies between RND and the finalist at every point of the curve.
