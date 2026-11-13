# Batch 7: the held divisor alone in the two step-cost cases

Written 2026-11-12 22:47 PT, before any run of the batch. Bonus-only (program.md, days 1 to 5); the option exists
in the frozen code of batch 2 (commit d7054945). One change against CleanRL PPO-RND: the divisor of the intrinsic
reward (the running standard deviation of its discounted sum) is no longer updated after 1/16 of the run.

## What batches 5 and 6 showed

**Second validation block (seeds 20000 to 20255, used for no choice before), PointMaze Medium, reward 1 on
arrival.** The arms ran at configurations fixed on the first block; RND is at the best of its three on this block
(0.9308), which favours RND.

| arm | whole-curve mean ↑ | difference from RND [95%] | difference in the final level [95%] |
|---|---|---|---|
| held divisor alone, held after 1/64 of the run (learning rate 1e-2, weight 2) | **0.9417** | +0.0109 [+0.0039, +0.0182] | +0.0092 [+0.0043, +0.0181] |
| predictor at half speed with held divisor (6e-3, weight 2) | <u>0.9416</u> | +0.0108 [+0.0031, +0.0186] | +0.0093 [+0.0043, +0.0178] |
| held divisor alone, held after 1/16 of the run (6e-3, weight 4) | 0.9398 | +0.0090 [+0.0021, +0.0160] | +0.0087 [+0.0037, +0.0178] |

**First validation block, PointMaze Large, reward 1 on arrival** (RND's best of five: 0.8781): held divisor alone
after 1/16, weight 2: 0.8876, +0.0095 [+0.0039, +0.0152]; predictor at half speed with held divisor, weight 4:
0.8849, +0.0068 [+0.0005, +0.0128]; slower predictor with held divisor, weight 4: 0.8883, +0.0102
[+0.0047, +0.0159]; all three parts, weight 0.5: 0.8917, +0.0137 [+0.0068, +0.0203].

What the arms that are ahead in both cases share is the held divisor. Slowing the predictor adds nothing
consistent to the whole-curve mean (it helps in one case and not in the other); it shortens the first-goal time.
So the simplest candidate for the whole-curve mean is the held divisor alone. It has not been run with the step
cost.

## What is run (first validation block, 256 seeds)

| case | steps per policy | CleanRL PPO-RND | held divisor alone (after 1/16 of the run) | predictor at half speed with held divisor |
|---|---|---|---|---|
| PointMaze Medium, step cost | 1,048,576 | learning rate 6e-3 at weights 0.00195 and 0.0078; 1e-2 at 0.00195 | 6e-3 at weights 0.00195, 0.0078 and 0.03125 | 6e-3 at weights 0.00195 and 0.0078 |
| PointMaze Large, step cost | 33,554,432 | already queued in batch 6 | 2e-3 at weights 4 and 8; 3e-3 at weight 8 | already queued in batch 6 |

11 units. These configurations come from RND's own best ones and their neighbours, not from a development sweep
of the arm, so the arm is not favoured by a choice on 16 seeds.

## Predictions

1. **PointMaze Large, step cost:** the held divisor alone is ahead of RND on the whole-curve mean with its
   interval above zero, because RND's seeds there split into two groups and the worse group is the one that
   keeps exploring late.
2. **PointMaze Medium, step cost:** level with RND (within 0.005), because RND's best weight there is nearly
   zero and a held divisor changes little when the bonus hardly counts; with the larger weight 0.03125 the held
   divisor is behind RND's best.
