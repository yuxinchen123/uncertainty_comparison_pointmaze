# Batch 13: the fast last layer until the first arrival, then the divisor of the predictor's own scores (four PointMaze cases)

Written 2026-11-13 05:30 PT, before any run of the batch. Bonus-only change (program.md, days 1 to 5). New option
`predictor_divisor_after_first_arrival` (commit 9aab504b; 4 new tests and the 73 bonus, resume and integration
tests pass; with the option off the trainer is unchanged).

## What the diagnosis found

Notes: `diagnosis/2026-11-13_slow-settling-after-the-fast-last-layer/notes.md`.

1. **RND's divisor remembers the predictor's first errors.** On PointMaze Medium the bonus averages 1.42 in the
   first 2,048 steps; the divisor is 93 there and still 60 at the first arrival.
2. **The fast last layer's scores are about 14 times smaller from the start** (0.098), so the divisor built on them
   is about 4.
3. **When the fast last layer stops at the first arrival, the predictor's own scores are divided by that small
   divisor:** the learner sees 10 to 20 times RND's intrinsic reward, and the weighted intrinsic advantage is 6
   to 9 times the extrinsic one for the next 16 logging windows (about equal in RND). That is why the policies of
   batch 11 went on covering the maze. Holding the divisor only keeps it from shrinking further.
4. **Ruled out:** a larger action spread after the fast phase (it is smaller), a predictor left in another state
   (its own score follows RND's), the switch itself.
5. **Intervention** (64 seeds, short runs): with a second divisor that follows the predictor's own scores from
   the first rollout and is used after the arrival, the failure is gone: on Medium at weight 4 the short-run
   return is 0.734 against RND's 0.733 with the first arrival after 16 against 23 thousand steps; on Large at
   weight 2 it is 0.353 against RND's 0.329 with the first arrival after 101 against 249 thousand steps.

## Arms

| arm | options |
|---|---|
| fast last layer until the first arrival, then the predictor's own divisor, held from the arrival | `fast_last_layer_step=0.1,fast_last_layer_until_first_arrival=true,predictor_divisor_after_first_arrival=true,reward_divisor=held_after_first_arrival` |
| the same with the divisor left running | the same without `reward_divisor=held_after_first_arrival` |

Before its first arrival a policy is exactly the fast last layer of batches 1, 9 and 10. After it the policy is
scored as RND and divided as RND would have divided it (held, in the first arm).

**Configurations** (learning rate, entropy, intrinsic weight), first validation block (seeds 10000 to 10255):

1. Held: Medium with reward 1 on arrival (6e-3, 0.01, 2) and (6e-3, 0.01, 4); Medium with the step cost
   (6e-3, 0.01, 1) and (6e-3, 0.01, 2); Large with reward 1 on arrival (2e-3, 0.01, 1), (2e-3, 0.01, 2) and
   (2e-3, 0.01, 4); Large with the step cost (2e-3, 0.01, 4), (2e-3, 0.01, 8) and (3e-3, 0.01, 8).
2. Running: Medium with reward 1 on arrival (6e-3, 0.01, 2); Large with reward 1 on arrival (2e-3, 0.01, 2);
   Large with the step cost (2e-3, 0.01, 8).

Thirteen units. The weights stay at 1 to 8: before the arrival the learner of a fast-last-layer arm sees 2 to 3
times RND's intrinsic reward at the same weight, and no arm settled at weight 16.

## Predictions

1. **First-goal time as the fast last layer's at the same configuration** (the same program until the arrival):
   about 15 to 17 thousand steps on Medium with reward 1 on arrival, about 100 on Large with reward 1 on arrival
   at weight 2 and about 85 at weight 4, about 70 on Large with the step cost at (2e-3, 0.01, 8).
2. **The failure of batch 11 is gone at full length:** in every case the held arm's final level is within 0.01 of
   the hold alone's at the same configuration.
3. **PointMaze Large with reward 1 on arrival:** the held arm is ahead of RND's best (0.8781) with the interval
   above zero at weight 1 or 2, by about 0.01, because the first arrival comes about 2% of the budget sooner. I
   am unsure: after the arrival the arm at weight w settles like RND at a weight between w and 2w.
4. **PointMaze Large with the step cost:** the held arm is ahead of RND's best (−0.7310) with the interval above
   zero at one of its three configurations, and not behind batch 10's arm (−0.7264).
5. **PointMaze Medium:** within 0.01 of the hold alone on the whole-curve mean in both cases, so ahead of or level
   with RND's best; the first arrival is only a few thousand steps sooner there.
6. **Running against held:** the held arm has the higher final level in all three cases where both run.

## What decides the next step

A configuration ahead of RND's best on the first block goes to the second block, fixed, with RND, the hold alone
and batch 10's arm on the same seeds. An arm that is ahead in the Large cases and level on Medium is screened on
the ant U-maze at both environment counts.
