# Batch 12: smaller steps of the fast last layer, with the first-arrival hold (four PointMaze cases)

Written 2026-11-13 05:07 PT, before any run of the batch. Bonus-only change (program.md, days 1 to 5); existing options, no
code change (frozen copy 2017e2b8).

## What the earlier runs showed

The fast last layer has been run at step 0.1 only since batch 1 (which tried 0.1, 0.3 and 1). With the
first-arrival hold, at the hold's own best configuration (256 validation seeds, first block):

| case and configuration | hold alone (step 0): whole-curve mean, first-goal time | step 0.1: whole-curve mean, first-goal time |
|---|---|---|
| Large, reward 1 on arrival (2e-3, 0.01, 2) | 0.8874, 259 thousand steps | 0.8707, 91 thousand steps |
| Medium, reward 1 on arrival (6e-3, 0.01, 4) | 0.9435, 18.5 | 0.8424, 15.7 |
| Medium, step cost (6e-3, 0.01, 2) | −0.3484, 14.2 | −0.3692, 12.9 |

On Large with reward 1 on arrival the step 0.1 finds the goal about three times sooner and loses 0.017 of
whole-curve mean; RND's best there is 0.8781. Between step 0 and step 0.1 there may be a step that keeps most of
the earlier arrival and loses less afterwards. Stopping the fast last layer at the first arrival made things
worse (batch 11), so the step itself is the next thing to vary.

## Arms

The fast last layer throughout with the first-arrival hold
(`fast_last_layer_step=<step>,reward_divisor=held_after_first_arrival`), at steps 0.03 and 0.01, one
configuration per case (learning rate, entropy, intrinsic weight), first validation block (seeds 10000 to 10255):

1. Large, reward 1 on arrival: (2e-3, 0.01, 2).
2. Large, step cost: (2e-3, 0.01, 8) (hold alone: −0.7268, 189 thousand steps; the unit at step 0.1 is running).
3. Medium, reward 1 on arrival: (6e-3, 0.01, 4).
4. Medium, step cost: (6e-3, 0.01, 2).

Eight units. Each has the hold alone and the step 0.1 on the same seeds.

## Predictions

1. **Ordered by the step.** In every case the first-goal time and the whole-curve mean both rise as the step
   falls from 0.1 to 0.03 to 0.01, toward the hold alone.
2. **Large with reward 1 on arrival.** At step 0.03 the first-goal time is at least 60 thousand steps below the
   hold alone's (interval below zero) and the whole-curve mean is within 0.005 of the hold alone's. At step 0.03 or
   0.01 the whole-curve mean is above RND's best (0.8781) with the interval above zero. I am unsure of the second
   sentence: the hold alone is itself only level with RND on the second block.
3. **Medium with reward 1 on arrival.** Step 0.03 is behind the hold alone by 0.02 to 0.05; step 0.01 is within
   0.01 of it.

## What decides the next step

A step that is ahead of RND's best on the whole-curve mean in a case, or level with it while clearly ahead on the
first-goal time, goes to the second block at fixed configuration with RND and the hold alone on the same seeds,
and to the ant U-maze with 16 environments per policy.
