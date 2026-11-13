# Batch 11: the fast last layer until each policy's first arrival, with the divisor held from then on (four PointMaze cases)

Written 2026-11-13 03:53 PT, before any run of the batch. Bonus-only change (program.md, days 1 to 5). New option
`fast_last_layer_until_first_arrival` (commit 3abfe847, 7 new tests and the 62 existing bonus, resume and
configuration tests pass; the saved training state is unchanged).

## What the earlier runs showed

1. **The fast last layer finds the goal sooner and then costs return** (batch 10, 256 validation seeds). Added to
   the first-arrival hold at the hold's own best configuration, it lowers the whole-curve mean by 0.101 on
   PointMaze Medium with reward 1 on arrival and by 0.021 on Medium with the step cost, while the first-goal time
   falls by 2.8 and 1.3 thousand steps.
2. **On PointMaze Large the time before the first arrival is where return can be gained.** With reward 1 on
   arrival RND's best configuration for the whole-curve mean first arrives after 269 thousand steps, 3.2% of the
   budget; the fast last layer with the hold at weight 16 arrives after 86 thousand, and then keeps the goal
   slowly (final level 0.85 against 0.99).
3. **The hold alone is level with RND on Large with reward 1 on arrival on the second validation block** (−0.0003
   [−0.007, +0.006]); its gain is in the final level only.

## The arm

Until a policy's first arrival: the fast last layer (step 0.1), as in batches 1, 9 and 10. From the rollout after
that arrival: the bonus is scored by the predictor alone, as in RND, and the divisor is held. Both switches are per
policy and read the same signal, the policy's own first arrival. Nothing is set besides the step 0.1.

| arm | options |
|---|---|
| fast last layer until the first arrival, then the hold | `fast_last_layer_step=0.1,fast_last_layer_until_first_arrival=true,reward_divisor=held_after_first_arrival` |
| the same without the hold (to see what the hold adds) | `fast_last_layer_step=0.1,fast_last_layer_until_first_arrival=true` |

**Configurations** (learning rate, entropy, intrinsic weight), first validation block (seeds 10000 to 10255). Every
one has the first-arrival hold alone, or the fast last layer with the hold of batch 10, on the same seeds:

1. Medium, reward 1 on arrival: (6e-3, 0.01, 4) and (1e-2, 0.01, 4).
2. Medium, step cost: (6e-3, 0.01, 2) and (6e-3, 0.01, 4).
3. Large, reward 1 on arrival: (2e-3, 0.01, 2), (2e-3, 0.01, 4) and (2e-3, 0.01, 16).
4. Large, step cost: (3e-3, 0.01, 8) and (2e-3, 0.01, 8).
5. Without the hold: Medium with reward 1 on arrival (6e-3, 0.01, 4) and Large with reward 1 on arrival
   (2e-3, 0.01, 4).

Items 1 to 3 run first (7 units); items 4 and 5 (4 units) follow when no job of an older code copy is left that
could take them.

## Predictions

1. **The first-goal time is that of batch 10's arm at the same configuration** (the same program until the first
   arrival): 15.7 and 14.7 thousand steps at the two configurations of Medium with reward 1 on arrival, 12.9 on
   Medium with the step cost at (6e-3, 0.01, 2), 86 on Large with reward 1 on arrival at weight 16; each within
   the interval of the difference.
2. **The cost after the arrival is gone.** At the hold's own best configurations the whole-curve mean is not
   behind the hold alone by more than 0.003 in any case (batch 10's arm was behind by 0.101 and 0.021).
3. **PointMaze Medium:** level with the hold alone on the whole-curve mean (the goal is found only about 3
   thousand steps sooner, 0.3% of the budget) and ahead of RND's best as the hold is.
4. **PointMaze Large with reward 1 on arrival:** ahead of RND's best (0.8781) with the interval above zero at
   weight 4 or 16, by 0.01 to 0.02, because the first arrival comes 1 to 2% of the budget sooner. I am least sure
   of this one: after the switch a weight of 16 may still be too large for the policy to settle, as the hold alone
   already loses 0.002 from weight 2 to weight 4.
5. **Without the hold** the arm is behind the arm with the hold on the final level in both cases and level with
   it on the first-goal time.

## What decides the next step

- A configuration ahead of RND's best on the first block goes to the second block, fixed, together with RND and
  the hold alone on the same seeds.
- If prediction 4 holds, the arm is screened on the ant U-maze with 16 environments per policy, where the fast
  last layer looks ahead in the partial reading of batch 10.
