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

## Addendum 1, written 2026-11-13 04:09 PT after the four PointMaze Medium units and before the units it adds

**What the Medium units showed (256 validation seeds).** Prediction 2 failed. With reward 1 on arrival at learning
rate 6e-3 and weight 4 the arm's whole-curve mean is 0.700, against 0.9435 for the hold alone and 0.842 for batch
10's arm (the fast last layer throughout, with the hold); with the step cost it is behind the hold alone by 0.034
and 0.080 at weights 2 and 4. Prediction 1 held (first-goal time 14.6 against 15.7 thousand steps, −1.1 [−3.0,
+0.8]).

**What the records say about the cause.** The logged mean intrinsic reward of a policy, by logging window counted
from the window of its first arrival (median over the 256 policies, same configuration):

| arm | 1 before | at | 1 after | 4 after | 16 after | 64 after | mean return 32 windows into the run |
|---|---|---|---|---|---|---|---|
| fast last layer until the first arrival, then the hold | 0.028 | 0.048 | 0.058 | 0.031 | 0.0125 | 0.0048 | 0.35 |
| fast last layer throughout, with the hold | 0.029 | 0.025 | 0.018 | 0.010 | 0.0046 | 0.0014 | 0.60 |
| first-arrival hold alone | 0.103 | 0.103 | 0.054 | 0.023 | 0.0031 | 0.0006 | 0.91 |
| RND | 0.090 | 0.098 | 0.052 | 0.024 | 0.0033 | 0.0010 | 0.91 |

1. At the switch the reward the learner sees doubles (0.028 to 0.058): the divisor was built on the fast scores,
   which are about half the predictor's own, and is then frozen.
2. One window after the arrival the reward is the same size as the hold's alone (0.058 against 0.054), but it then
   falls four to eight times more slowly, and the return rises slowly with it.

Two readings fit: (a) the frozen divisor of the wrong scale keeps the bonus too large for the policy to settle;
(b) the fast phase leaves the policy or its value heads in a state that settles slowly whatever the divisor does.

**Units added to tell them apart** (first validation block; existing options only):

| arm | options | configurations |
|---|---|---|
| fast last layer until the first arrival, then RND's running divisor | `fast_last_layer_step=0.1,fast_last_layer_until_first_arrival=true` | Medium with reward 1 on arrival: (6e-3, 0.01, 2), (6e-3, 0.01, 4); Large with reward 1 on arrival: (2e-3, 0.01, 2), (2e-3, 0.01, 4), (2e-3, 0.01, 16) |
| fast last layer until the first arrival, divisor held after 1/16 of the run | the same with `reward_divisor=held_after_fraction,divisor_hold_fraction=0.0625` | Medium with reward 1 on arrival: (6e-3, 0.01, 4); Large with reward 1 on arrival: (2e-3, 0.01, 4) |

In the second arm the divisor follows the predictor's own scores from the switch (about 15 thousand steps on
Medium) until it is held (66 thousand steps on Medium, 524 thousand on Large). These units replace the second
description of this batch (the arm with the first-arrival hold on Large with the step cost is not run).

**Predictions of the addendum.**

6. Under reading (a), the arm with RND's running divisor is within 0.03 of RND at the same configuration on
   Medium with reward 1 on arrival (RND: 0.9349 at weight 2), where the arm with the first-arrival hold is 0.24
   behind. Under reading (b) it stays more than 0.05 behind. I expect (a), with a small loss left from the
   rollouts in which the running divisor is still catching up.
7. The arm with the divisor held after 1/16 is at least as good as the arm with the running divisor at the same
   configuration, as the fixed hold is for RND.
8. On Large with reward 1 on arrival the arm with the running divisor keeps the early first arrival (below 150
   thousand steps at weights 4 and 16, against 269 for RND's best) and its whole-curve mean is ahead of RND's
   best (0.8781) at one of the three weights. I am unsure: at weight 16 RND itself may settle too slowly.
