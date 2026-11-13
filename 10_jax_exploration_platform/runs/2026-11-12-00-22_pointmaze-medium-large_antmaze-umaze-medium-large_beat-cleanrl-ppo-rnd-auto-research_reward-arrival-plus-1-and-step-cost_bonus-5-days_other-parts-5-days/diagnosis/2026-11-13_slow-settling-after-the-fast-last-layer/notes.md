# Why a policy settles slowly after the fast last layer stops

Written 2026-11-13 05:03 PT. Times are Pacific, converted from machines running Eastern.

## The question

PointMaze Medium, reward 1 on arrival, learning rate 6e-3, entropy 0.01, intrinsic weight 4, 4 environments per
policy. A policy explores with the fast last layer (step 0.1) until its first arrival and is scored as plain
random network distillation (RND) from the next rollout on. On the 256 validation seeds its whole-curve mean is
0.4487 with RND's running divisor and 0.7001 with the divisor held from the first arrival, against 0.9194 for RND.
After the first arrival it keeps covering the whole maze instead of turning to the goal.

## The mechanism

**The divisor is 12 to 17 times too small after the switch, so the learner sees an intrinsic reward 10 to 20 times
the size RND would show it.**

Define the divisor as the running standard deviation of the discounted sum of the bonus, kept since the first
rollout (discount 0.99); the learner sees the bonus divided by it, times the intrinsic weight.

1. RND's divisor is large because it remembers the predictor's first errors: the bonus averages 1.42 in the first
   2,048 steps and the divisor is 93 there, 65 at the first arrival and still 20 near the end of a 262,144-step run.
2. The fast last layer lowers the error of every state as a rollout goes on (its rows share the bias input), so its
   scores are 14 times smaller from the start (0.098 in the first 2,048 steps) and the divisor built on them is
   about 4 and never larger.
3. At the switch the bonus becomes the predictor's own score again, 0.09 one window after the arrival, the same as
   RND's at that point (0.093). It is divided by 4.7 where RND divides by 56. The weighted intrinsic advantage is
   then 6 to 9 times the extrinsic one for the next 16 windows; in RND the two are about equal.
4. That is RND at an intrinsic weight of roughly 50 to 70 instead of 4, and RND at weight 64 does not keep the
   goal either (whole-curve mean 0.005 on PointMaze Large in the discovery check).
5. The running divisor is worse than the held one because it shrinks further after the switch (4.7 to 2.5 over
   64 windows) while the held one stays at 3.4.

## Measurements

All runs: the unit's own configuration with 64 policies (seeds 10000 to 10063), 512 iterations (262,144 steps, a
quarter of the unit), the learning rate annealed over the unit's 2,048 iterations, logging windows of 4 iterations
(2,048 steps), and the trainer's diagnostics on. Scripts: `scripts/`; configurations: `configurations/`; records:
`runs/data/` (not in git).

### 1. Around the first arrival (intrinsic weight 4)

Mean over the 64 policies, by logging window counted from each policy's own first arrival (0 is the window that
contains it). "Reward seen" is the bonus divided by the divisor, before the weight. The two advantage rows are
root mean squares over the rollout, the intrinsic one times its weight 4, the extrinsic one times its weight 2.

| quantity | arm | 1 before | 0 | 1 after | 4 after | 8 after | 16 after | 32 after | 64 after |
|---|---|---|---|---|---|---|---|---|---|
| divisor | RND | 65.0 | 59.9 | 55.8 | 47.6 | 41.0 | 33.6 | 26.3 | 19.8 |
| divisor | fast until the arrival | 3.58 | 3.85 | 4.73 | 4.53 | 4.07 | 3.56 | 3.04 | 2.48 |
| bonus | RND | 0.373 | 0.162 | 0.093 | 0.051 | 0.027 | 0.014 | 0.0054 | 0.0034 |
| bonus | fast until the arrival | 0.035 | 0.066 | 0.088 | 0.048 | 0.035 | 0.022 | 0.012 | 0.0086 |
| reward seen | RND | 0.0046 | 0.0026 | 0.0017 | 0.0011 | 0.00072 | 0.00042 | 0.00022 | 0.00019 |
| reward seen | fast until the arrival | 0.0100 | 0.0160 | 0.0193 | 0.0116 | 0.0097 | 0.0071 | 0.0047 | 0.0039 |
| intrinsic advantage | RND | 0.42 | 0.27 | 0.18 | 0.095 | 0.064 | 0.041 | 0.023 | 0.020 |
| intrinsic advantage | fast until the arrival | 0.63 | 1.01 | 1.15 | 0.63 | 0.47 | 0.32 | 0.19 | 0.15 |
| extrinsic advantage | RND | 0.087 | 0.138 | 0.110 | 0.090 | 0.090 | 0.068 | 0.034 | 0.014 |
| extrinsic advantage | fast until the arrival | 0.103 | 0.155 | 0.130 | 0.098 | 0.081 | 0.069 | 0.057 | 0.057 |
| action log standard deviation | RND | 0.064 | 0.071 | 0.079 | 0.103 | 0.143 | 0.197 | 0.267 | 0.421 |
| action log standard deviation | fast until the arrival | 0.016 | 0.014 | 0.017 | 0.041 | 0.074 | 0.137 | 0.284 | 0.580 |
| share of episodes that arrive | RND | 0 | 0.24 | 0.14 | 0.21 | 0.34 | 0.59 | 0.86 | 0.94 |
| share of episodes that arrive | fast until the arrival | 0 | 0.23 | 0.09 | 0.16 | 0.16 | 0.18 | 0.21 | 0.26 |

"Fast until the arrival" is the arm with RND's running divisor. With the first-arrival hold the divisor is 3.40
from the arrival on and the other rows are within a tenth of this arm's.

### 2. Where the two divisors come from (intrinsic weight 4, by window from the start of the run)

| quantity | arm | window 0 | 1 | 4 | 16 | 64 | 127 |
|---|---|---|---|---|---|---|---|
| bonus | RND | 1.42 | 0.278 | 0.091 | 0.027 | 0.0029 | 0.0026 |
| bonus | fast last layer throughout | 0.098 | 0.046 | 0.026 | 0.0099 | 0.0038 | 0.0025 |
| divisor | RND | 92.6 | 85.0 | 65.1 | 38.8 | 20.8 | 15.0 |
| divisor | fast last layer throughout | 3.96 | 4.23 | 3.57 | 2.46 | 1.52 | 1.15 |
| reward seen | RND | 0.0162 | 0.0036 | 0.0015 | 0.00072 | 0.00015 | 0.00019 |
| reward seen | fast last layer throughout | 0.0303 | 0.0116 | 0.0078 | 0.0042 | 0.0025 | 0.0022 |

So with the fast last layer the learner sees 2 to 3 times RND's intrinsic reward at the same weight early in a run
and more than 10 times late in it. "Weight 4" of a fast-last-layer arm is not RND's weight 4.

### 3. Smaller weights without any change of code

The arm with RND's running divisor at smaller intrinsic weights (one run, 64 policies per weight). If the reward's
size is the cause, a weight 16 times smaller should settle about as RND does at weight 4.

| intrinsic weight | mean first arrival, thousand steps | share arriving 16 windows after | 32 after | 64 after | mean return of the short run |
|---|---|---|---|---|---|
| 4 | 13.3 | 0.18 | 0.21 | 0.26 | 0.247 |
| 1 | 18.3 | 0.39 | 0.56 | 0.72 | 0.564 |
| 0.25 | 29.2 | 0.37 | 0.68 | 0.92 | 0.675 |
| 0.0625 | 81.8 | 0.37 | 0.58 | 0.78 | 0.389 |
| RND at weight 4, for comparison | 16.5 | 0.59 | 0.86 | 0.94 | 0.733 |

It settles better as the weight falls to 0.25 and finds the goal later; at 0.0625 it hardly explores.

### 4. The intervention: a second divisor on the predictor's own scores

New option `predictor_divisor_after_first_arrival=true`: a second filter and running standard deviation follow
the predictor's own scores of every rollout from the first one on, which are the statistics RND keeps, and a policy
that has arrived is divided by that second divisor. Before its arrival nothing changes. One run per row group,
64 policies per weight.

| arm | intrinsic weight | mean first arrival, thousand steps | divisor 1 window after | share arriving 8 windows after | 16 after | 32 after | 64 after | mean return of the short run |
|---|---|---|---|---|---|---|---|---|
| RND | 2 | 31.0 | 50.2 | 0.30 | 0.51 | 0.82 | 0.96 | 0.730 |
| RND | 4 | 22.8 | 50.5 | 0.33 | 0.56 | 0.82 | 0.91 | 0.733 |
| RND | 8 | 19.5 | 55.9 | 0.24 | 0.34 | 0.57 | 0.73 | 0.597 |
| RND | 16 | 17.5 | 56.6 | 0.16 | 0.26 | 0.35 | 0.40 | 0.396 |
| first-arrival hold alone | 2 | 25.2 | 57.6 | 0.40 | 0.64 | 0.84 | 0.98 | 0.771 |
| first-arrival hold alone | 4 | 22.0 | 55.6 | 0.36 | 0.56 | 0.79 | 0.92 | 0.749 |
| first-arrival hold alone | 8 | 16.1 | 62.3 | 0.31 | 0.42 | 0.61 | 0.87 | 0.705 |
| first-arrival hold alone | 16 | 18.1 | 57.9 | 0.26 | 0.30 | 0.40 | 0.63 | 0.557 |
| fast until the arrival, second divisor, running | 2 | 17.0 | 58.4 | 0.32 | 0.50 | 0.79 | 0.97 | 0.770 |
| fast until the arrival, second divisor, running | 4 | 16.4 | 59.2 | 0.28 | 0.47 | 0.77 | 0.91 | 0.734 |
| fast until the arrival, second divisor, running | 8 | 15.8 | 57.8 | 0.18 | 0.30 | 0.50 | 0.72 | 0.605 |
| fast until the arrival, second divisor, running | 16 | 20.6 | 59.5 | 0.19 | 0.26 | 0.29 | 0.42 | 0.379 |
| fast until the arrival, second divisor, held from the arrival | 2 | 23.3 | 58.8 | 0.35 | 0.56 | 0.80 | 0.91 | 0.738 |
| fast until the arrival, second divisor, held from the arrival | 4 | 12.9 | 65.2 | 0.29 | 0.47 | 0.72 | 0.92 | 0.757 |
| fast until the arrival, second divisor, held from the arrival | 8 | 17.7 | 60.9 | 0.21 | 0.29 | 0.56 | 0.84 | 0.665 |
| fast until the arrival, second divisor, held from the arrival | 16 | 14.8 | 66.8 | 0.15 | 0.24 | 0.37 | 0.70 | 0.557 |
| fast until the arrival, one divisor, running (the failing arm) | 4 | 13.3 | 4.73 | 0.16 | 0.18 | 0.21 | 0.26 | 0.247 |
| fast until the arrival, one divisor, held from the arrival | 4 | 16.6 | 3.40 | 0.18 | 0.16 | 0.24 | 0.36 | 0.307 |

With the second divisor the arm settles as RND does at the same weight (0.47 against 0.56 of the episodes arrive
16 windows after the first arrival at weight 4, 0.77 against 0.82 after 32; without it 0.18 and 0.21), and its
short-run return equals RND's (0.734 against 0.733; without it 0.247). Changing the divisor alone removes the
failure.

### 5. PointMaze Large with reward 1 on arrival (added 05:23 PT, after the coordinator's readings on 256 seeds)

On the validation seeds the failure is stronger on Large: with RND's running divisor after the switch the final
level is 0.24 at weight 4 where RND reaches 0.99. Four runs with the unit's configuration (learning rate 2e-3,
entropy 0.01), 64 policies per weight (seeds 10000 to 10063), 2,048 iterations (1,048,576 steps, one eighth of
the unit), logging windows of 8 iterations (4,096 steps), the learning rate annealed over the unit's 16,384
iterations.

**Is the predictor trained as in RND?** Yes. Its loss is the mean squared error of its own output on the update
batch (`bonuses/rnd/implementation.py`, `loss`); the fast last layer is a temporary copy that is never trained
on. Measured at weight 4: the predictor's own score during the fast phase follows RND's bonus at the same step.

| quantity, by window from the start of the run | window 0 | 1 | 4 | 16 | 64 | 127 | 255 |
|---|---|---|---|---|---|---|---|
| RND: bonus | 1.08 | 0.177 | 0.046 | 0.020 | 0.0070 | 0.0033 | 0.0010 |
| fast until the arrival: the predictor's own score | 1.11 | 0.214 | 0.057 | 0.013 | 0.0047 | 0.0027 | 0.0009 |
| fast until the arrival: what the policy is rewarded with | 0.101 | 0.049 | 0.023 | 0.011 | 0.0048 | 0.0029 | 0.0018 |

**The divisor and the reward the learner sees over time** (weight 4):

| quantity | arm | window 0 | 4 | 16 | 64 | 127 | 255 |
|---|---|---|---|---|---|---|---|
| divisor | RND | 110 | 63.4 | 35.8 | 18.8 | 13.4 | 9.5 |
| divisor | fast until the arrival, one divisor (the failing arm) | 6.2 | 4.3 | 2.9 | 1.75 | 1.33 | 0.99 |
| divisor | fast until the arrival, second divisor | 6.2 | 6.1 | 22.1 | 18.8 | 13.7 | 9.7 |
| reward seen | RND | 0.0108 | 0.00076 | 0.00057 | 0.00039 | 0.00025 | 0.00011 |
| reward seen | fast until the arrival, one divisor | 0.0194 | 0.0055 | 0.0038 | 0.0028 | 0.0022 | 0.0019 |
| reward seen | fast until the arrival, second divisor | 0.0193 | 0.0047 | 0.0013 | 0.00028 | 0.00021 | 0.00010 |
| intrinsic advantage | RND | 0.95 | 0.081 | 0.059 | 0.032 | 0.017 | 0.0082 |
| intrinsic advantage | fast until the arrival, one divisor | 1.22 | 0.29 | 0.19 | 0.15 | 0.10 | 0.086 |
| intrinsic advantage | fast until the arrival, second divisor | 1.22 | 0.25 | 0.079 | 0.024 | 0.015 | 0.0082 |
| share of episodes that arrive | RND | 0 | 0 | 0.001 | 0.021 | 0.19 | 0.71 |
| share of episodes that arrive | fast until the arrival, one divisor | 0 | 0.010 | 0.011 | 0.014 | 0.023 | 0.029 |
| share of episodes that arrive | fast until the arrival, second divisor | 0 | 0 | 0.002 | 0.045 | 0.30 | 0.76 |

In the failing arm the divisor falls from 6.2 to 0.99 within the first 1.05 million steps, the reward the learner
sees is 7 times RND's by window 4 and 17 times by window 255, and it stays there: the policy never turns to the
goal. With the second divisor the reward seen and the intrinsic advantage meet RND's (0.00010 against 0.00011;
0.0082 against 0.0082 at window 255), and more episodes arrive than in RND at the same weight.

**All three weights:**

| arm | intrinsic weight | mean first arrival, thousand steps | divisor 1 window after the arrival | share arriving 64 windows after | mean return of the short run |
|---|---|---|---|---|---|
| RND | 2 | 249 | 19.7 | 0.29 | 0.329 |
| RND | 4 | 223 | 23.5 | 0.15 | 0.257 |
| RND | 16 | 151 | 25.6 | 0.03 | 0.040 |
| fast until the arrival, one divisor, running (the failing arm) | 2 | 96 | 2.9 | 0.02 | 0.050 |
| fast until the arrival, one divisor, running (the failing arm) | 4 | 74 | 3.0 | 0.03 | 0.021 |
| fast until the arrival, one divisor, running (the failing arm) | 16 | 90 | 2.8 | 0.02 | 0.017 |
| fast until the arrival, second divisor, running | 2 | 101 | 33.0 | 0.14 | 0.353 |
| fast until the arrival, second divisor, running | 4 | 76 | 39.0 | 0.12 | 0.348 |
| fast until the arrival, second divisor, running | 16 | 108 | 36.2 | 0.02 | 0.042 |
| fast until the arrival, second divisor, held from the arrival | 2 | 81 | 38.5 | 0.19 | 0.333 |
| fast until the arrival, second divisor, held from the arrival | 4 | 114 | 38.8 | 0.10 | 0.285 |
| fast until the arrival, second divisor, held from the arrival | 16 | 85 | 39.4 | 0.02 | 0.086 |

With the second divisor the first arrival still comes 2 to 3 times sooner than RND's and the short-run return at
weights 2 and 4 is at RND's best or above it (0.353 and 0.348 against 0.329). After the arrival the arm settles
more slowly than RND at the same weight (0.14 against 0.29 of the episodes arrive 64 windows later at weight 2),
as on Medium; the earlier arrival makes up for it within this short run. Weight 16 does not settle in any arm.
These are 64 seeds and one eighth of the budget: a direction, not a result.

## What is ruled out

1. **An inflated action standard deviation.** The log standard deviation is lower after the fast phase than after
   the same stretch of RND (0.016 against 0.064 one window before the arrival) and rises afterwards in both.
2. **A predictor left in another state.** One window after the arrival the predictor's own score is the same in
   both arms (0.088 and 0.093). It falls more slowly afterwards only because the policy keeps roaming, and with
   the second divisor the arm settles normally with the same predictor.
3. **The switch itself.** The unit tests show that an arrived policy is scored exactly as RND scores it, and the
   intervention changes nothing but the divisor.
4. **The frozen divisor as such.** The hold is not the cause; the scale of what is held, or kept running, is.

## The fix, and what to know before using it

Call: `--bonus-options fast_last_layer_step=0.1,fast_last_layer_until_first_arrival=true,predictor_divisor_after_first_arrival=true`,
with `reward_divisor=held_after_first_arrival` added for the hold (the second divisor is then the one held from the
first arrival). Commit 9aab504b. It is a change of how the bonus is scaled, so it belongs to the bonus-only days.

1. **Tests:** 4 new ones in `tests/bonuses/test_rnd_predictor_divisor_after_first_arrival.py` pass (job 6902460):
   a policy is bit for bit unchanged through the iteration of its first arrival; from the next one it is divided
   by the second divisor; the second statistics after the first iteration equal RND's; with the hold they stay
   from the first arrival; saved units continue exactly before and after the switch; what is refused. The golden
   comparison is bit-identical with the option off (job 6902632). All 73 tests of `tests/bonuses`, the resume
   tests and the two integration files for bonus options and configuration files pass (job 6902612). With the
   option on and the trainer's diagnostics on, each window record also holds `predictor_score_mean`, the
   predictor's own score of the rollout (commit bb07d8fa; the 4 new tests pass again with it, job 6902859).
2. **The training state grows with the option on** (a second filter and the mean, variance and count of its
   statistics, per policy). A unit that uses the option can only be continued by a code copy that has it; units
   without the option are unchanged.
3. **Refused:** the option without `fast_last_layer_until_first_arrival`; with the divisors frozen after priming,
   held after a fraction, or none; with the record on the maze grid.
4. **The weights of fast-last-layer arms are not RND's weights** (measurement 2). Before the arrival the learner
   still sees 2 to 3 times RND's intrinsic reward at the same weight, and one window after the arrival the
   weighted intrinsic advantage with the second divisor is about twice RND's at the same weight (0.36 against
   0.17 at weight 4), so the arm at weight w settles between RND at w and RND at 2w.
5. **What these runs do not show:** whether the arm is ahead of RND on the whole-curve mean. They have 64 seeds
   and a quarter of the budget, enough for effects of a factor 3 and not for differences of 0.01. The first
   arrival comes earlier at weights 2 and 4 (17.0 and 16.4 against RND's 31.0 and 22.8 thousand steps) with
   about the same settling, which is what the validation block has to test.

## Jobs

Diagnosis runs (ids in the folder's `slurm/submitted_jobids_diagnosis_sl5nw.txt`, all ended): PointMaze Medium:
6902210 to 6902213, 6902407, 6902611, 6902613, 6902614, 6902615; PointMaze Large: 6902855, 6902857, 6902858,
6902885. Each ran 6 to 10 minutes on one RTX 5080 of nekomata01 or nekomata02, the last one on an RTX 4000 Ada of
cheetah02. Job 6902856 waited more than two minutes for a card that had been taken and was cancelled by its id.
