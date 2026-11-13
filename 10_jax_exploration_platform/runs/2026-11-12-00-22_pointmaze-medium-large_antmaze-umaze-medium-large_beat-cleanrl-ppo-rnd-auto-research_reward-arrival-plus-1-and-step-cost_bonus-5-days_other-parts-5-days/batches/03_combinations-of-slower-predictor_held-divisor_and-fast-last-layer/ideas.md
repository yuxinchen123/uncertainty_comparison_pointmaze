# Batch 3: combinations of the slower predictor, the held divisor and the fast last layer

Written 2026-11-12 16:10 PT, before any run of the batch. Bonus-only changes (program.md, days 1 to 5). Every arm
combines parts that were each run alone; the options exist in the frozen code of batch 2 (commit d7054945).

## What the earlier runs showed (PointMaze Large, reward 1 on arrival)

1. **A slower predictor with its own optimizer finds the goal sooner and is slightly ahead on the whole-curve mean.**
   Validation block (256 seeds), predictor at 0.25 times 1e-4, intrinsic weight 2: whole-curve mean 0.8859 ± 0.0024
   against the control's 0.8781 ± 0.0021 at the same weight (+0.0078 ± 0.0032 on the same seeds); mean capped
   first-goal time 171 thousand steps against 269 thousand (−99 ± 16 thousand); final level 0.9954 against 0.9920.
2. **Holding the divisor after 1/16 of the run keeps larger weights usable** (16 development seeds, first reading):
   whole-curve mean 0.895 ± 0.005 at weight 4 and 0.892 ± 0.009 at weight 2, final levels 0.998 and 0.997; the
   control has 0.867 and a final level of 0.990 at weight 4, and 0.885 at its best weight 2. Its first-goal times
   are those of the control (about 180 thousand steps).
3. **The fast last layer inside the rollout finds the goal two to three times sooner but ends lower.** At step 0.1
   (16 development seeds): mean capped first-goal time 62 ± 13 thousand steps at weight 16 against the control's
   190 ± 51 thousand at its best weight for that measure; its final level is 0.96 at weight 1 and 0.75 at weight 2
   (control: 0.997 and 0.995), and its best whole-curve mean is 0.862 ± 0.016 at weight 0.5.
4. **In the control the reward the learner sees rises again late** (batch 2, observation 7); the held divisor is the
   part that removes this.

Parts 1 and 3 act before and around the first arrival; part 2 acts after it. So they may add up.

## Screen

As in batches 1 and 2: PointMaze Large with reward 1 on arrival at 8,388,608 steps, seeds 0 to 15, learning rate
2e-3, entropy 0.01, clip 0.1, 12 intrinsic weights, the maze-grid record on. Both measures are read, each arm at its
own best weight for it. The control is batch 2's control (same code).

## The arms

| name | parts | options |
|---|---|---|
| slower predictor with held divisor | predictor with its own optimizer at 0.25 times 1e-4; divisor held after 1/16 of the run | `predictor_learning_rate_multiple=0.25`, `reward_divisor=held_after_fraction`, `divisor_hold_fraction=0.0625` |
| fast last layer with held divisor, step 0.1 | fast last layer inside the rollout at step 0.1; divisor held after 1/16 | `fast_last_layer_step=0.1`, held divisor as above |
| fast last layer with held divisor, step 0.3 | the same at step 0.3 | `fast_last_layer_step=0.3`, held divisor as above |
| fast last layer with slower predictor, step 0.1 | fast last layer at step 0.1; predictor at 0.25 times 1e-4 | `fast_last_layer_step=0.1`, `predictor_learning_rate_multiple=0.25` |
| all three, step 0.1 | fast last layer at step 0.1; predictor at 0.25 times 1e-4; divisor held after 1/16 | the three options together |
| all three, step 0.1, divisor held after 1/8 | the same with the divisor held after 1/8 of the run | `divisor_hold_fraction=0.125` |

That is 6 units.

## Predictions (written before the runs)

1. **Slower predictor with held divisor:** the two gains add: whole-curve mean at the best weight at least 0.015
   above the control's best on these seeds (a difference the validation block must then confirm), final level at
   least the control's.
2. **Fast last layer with held divisor:** the final level at weights 1 and 2 rises to 0.99 or more (from 0.96 and
   0.75 without the held divisor), because the reward can no longer rise again after the hold; the mean capped
   first-goal time stays below 100 thousand steps at its best weight; the whole-curve mean at its best weight is
   above the control's best. If the final level does not recover, the fast last layer's late cost is not the
   divisor's rise, and the records of the reward at the probes will show what it is.
3. **All three:** the smallest mean capped first-goal time of the batch and the highest whole-curve mean of the
   batch; holding after 1/8 instead of 1/16 changes the whole-curve mean by less than 0.01.
4. **If a combination is refused by the option checks or fails,** it is reported as not runnable and the reason is
   read from the unit's log before anything is changed.
