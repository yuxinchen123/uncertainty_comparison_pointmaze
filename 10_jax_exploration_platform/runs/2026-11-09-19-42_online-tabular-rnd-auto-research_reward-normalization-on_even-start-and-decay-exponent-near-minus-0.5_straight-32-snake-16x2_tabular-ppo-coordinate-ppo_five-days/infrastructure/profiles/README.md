# Speed profiles of online tabular-RND ideas

Times are Pacific. Snake grid (50 cells), coordinate-PPO, reward normalization on, 20 timed iterations after two
warm-up iterations. Script: `scripts/grid_initialization_campaign/profile_online_tabular.py`.

## Reference profile: serval02 card 1 (H100 NVL), 2026-11-09 23:15 PT, card held by this job alone

`serval02_card1_h100_nvl_snake_coordinate-ppo_exact_step.json`. Define hours per million steps per copy as
$\dfrac{10^6}{3600 \times r}$, where $r$ is the steps per second one copy gets.

| idea | copies | seconds per iteration | total steps per second (millions) | steps per second per copy (thousands) | hours per million steps per copy |
|---|---|---|---|---|---|
| control | 256 | 0.055 | 0.59 | 2.32 | 0.120 |
| control | 512 | 0.106 | 0.62 | 1.21 | 0.230 |
| unit start + shrink target, plain SGD | 256 | 0.034 | 0.95 | 3.72 | 0.075 |
| unit start + shrink target, plain SGD | 512 | 0.065 | 1.01 | 1.97 | 0.141 |
| unit start + exact-step shrink target, 1 step per minibatch | 256 | 0.060 | 0.55 | 2.13 | 0.130 |
| unit start + exact-step shrink target, 1 step per minibatch | 512 | 0.114 | 0.57 | 1.12 | 0.248 |
| exact step, 2 steps per minibatch | 512 | 0.234 | 0.28 | 0.55 | 0.507 |
| exact step, 4 steps per minibatch | 512 | 0.442 | 0.15 | 0.29 | 0.862 |

Reading:
- Total throughput does not grow with copies beyond 256: the card is the limit.
- Ideas with their own plain SGD are 1.6 times faster than the control, which carries Adam moments for the predictor.
- The exact-step form with one step per minibatch costs 1.08 times the control: accepted as it is.
- Two and four steps per minibatch cost 2.2 and 4.2 times the control; they are not used in RL.

## Earlier profile: serval05 (H100 NVL), 2026-11-09 20:02 PT — taken on a shared card

`serval05_h100_nvl_snake_coordinate-ppo.json`. Another account's processes were on the card during this profile
(they had started at about 19:15 PT; found at 22:31 PT), so its absolute values are about half of the reference
above (control 0.215 s per iteration at 512 copies). Its ratios between ideas agree with the reference. It is kept
as a record and not used for estimates.

## Estimates for the other servers (calculated, no probe there: user rule 2026-11-09)

From the reference profile: an H100 runs the control at about 0.6 million steps per second and the plain-SGD ideas at
about 1.0 million, at 224 to 512 copies per process. Measured afterwards from real units (not probes): Gautschi's
H100 SXM finished 224-copy screen units of 0.94 billion steps in about 1,490 s (0.63 million steps per second).
