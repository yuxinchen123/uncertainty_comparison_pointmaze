# Reward divisor held after a policy's first arrival: the card check

Times are Pacific (PT), read from the unit's log and record; the machines run on Eastern time.

## The option

`reward_divisor=held_after_first_arrival` (random network distillation bonus, added 2026-11-13).
The divisor of the intrinsic reward is the running standard deviation of the reward's discounted
sum, one per policy. With this option each policy's divisor is moved on by every rollout up to and
including the rollout in which the policy first reaches the goal in training, and by no later
rollout. A policy that never arrives keeps the running divisor for the whole run. The trainer
reads each policy's first arrival from the episode log of the training state
(`first_goal_step`, −1 until the first arrival).

## The unit

| part | value |
|---|---|
| machine and card | serval02 card 1, NVIDIA H100 NVL, 95,830 MiB |
| started, ended | 2026-11-13 00:49:31 PT, 00:54:07 PT (276 s in all) |
| other processes on the card | none during the unit |
| why not serval05 | its card held processes of another account at 00:40 PT |
| environment | PointMaze Medium, reward 1 on arrival, the episode ends at the goal |
| policies | 256 (seeds 10000 to 10255), 4 environments each |
| rollout | 128 steps, so 512 environment steps per policy and iteration |
| settings | learning rate 6e-3, entropy coefficient 0.01, intrinsic weight 2, clip 0.1 |
| budget | 2,048 iterations, 1,048,576 steps per policy |
| configuration file | `units/unit_configuration_pointmaze-medium_256-policies_learning-rate-0.006_intrinsic-weight-2.toml`, the study's validation unit of these settings |
| driver | `units/check_hold_and_speed.py`, started by `units/serval02-card-1_unit-1_hold-against-first-arrival_and_speed-against-fraction-hold.sh` |

The record is `records/serval02-card-1_unit-1_hold-against-first-arrival_and_speed-against-fraction-hold.json`
(not in git; it holds both step lists per policy and the seconds of every block).

## Hold step against first-arrival step

The hold step of a policy is the end of the last rollout that moved its divisor. The whole budget
was trained with the new option, and the divisor's variance of every policy was read after every
iteration.

| measure | value |
|---|---|
| policies with an arrival | 256 of 256 |
| first-arrival step: smallest, median, largest | 1,300; 16,306; 169,568 |
| hold step minus first-arrival step: smallest, median, largest | 0; 288; 508 |
| policies whose hold step is at or after the arrival and less than one rollout (512 steps) after it | 256 of 256 |
| policies whose last moved rollout is the rollout of the arrival | 256 of 256 |
| policies whose first arrival is after 1/16 of the run (65,536 steps) | 25 |

No policy's divisor moved again after its hold began. All 256 policies arrived, so this unit has
no policy without an arrival; that case is covered by the tests
(`tests/bonuses/test_rnd_divisor_held_after_first_arrival.py`).

## Speed against the divisor held after 1/16 of the run

Both options were trained from fresh states in alternating blocks of 128 iterations, 15 blocks
each, after two untimed iterations each. In every pair of blocks the new option ran first.

Define $s$ as the seconds per iteration, $C = 256$ as the number of policies, and
$r = \dfrac{512}{s}$ as the environment steps per second one policy gets. The total is $C \times r$
and the hours per million steps per policy are $\dfrac{10^{6}}{3600 \times r}$.

| option | copies | seconds per iteration | total steps per second (millions) | steps per second per copy (thousands) | hours per million steps per copy |
|---|---|---|---|---|---|
| held after the first arrival | 256 | 0.015582 | 8.412 | 32.86 | 0.00845 |
| held after 1/16 of the run | 256 | 0.015601 | 8.401 | 32.82 | 0.00846 |

The new option's blocks were 0.12% shorter on average (difference of the block means −1.9e-5 s,
standard error 6.1e-6 s; shorter in 14 of 15 pairs). The order inside each pair was fixed, so this
small difference is not evidence that the new option is faster. It shows no slowdown.

## A source change while the check ran

The check's process started at 00:49:35 PT. At 00:51:02 PT (the file's modification time) one
line of the option's refusal check in `training/train_step.py` was changed: the environment's
`continuing_task` is now read directly. That line runs once while the program is built, gives the
same answer for this environment in both versions, and is not part of the compiled iteration, so
the numbers above do not depend on which version the process read.
