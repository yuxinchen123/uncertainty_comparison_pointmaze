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

## Copies per process: serval02 card 1 (H100 NVL), 2026-11-10 10:35 to 10:41 PT

`serval02_card1_h100_nvl_snake_coordinate-ppo_copies_per_card.json` (snake grid, coordinate-PPO, 20 timed
iterations per row; serval05 had another account's process on its card, so the profile ran on serval02 card 1; no
other process was on that card during or after the measurement). Asked by the user on 2026-11-10: are the H100 and
H200 cards fully used, and how many copies should one process hold. Rule used (user, 2026-11-10): keep doubling the
copies while a doubling raises the total throughput by more than 10%.

**Baseline's kind of unit (predictor trained by gradient steps: control, unit start alone, shrink target):**

| copies | seconds per iteration | total steps per second (millions) | steps per second per copy (thousands) | hours per million steps per copy | gain from doubling |
|---|---|---|---|---|---|
| 64 | 0.0195 | 0.419 | 6.55 | 0.042 | |
| 128 | 0.0318 | 0.515 | 4.03 | 0.069 | +23% |
| 256 | 0.0546 | 0.600 | 2.35 | 0.118 | +17% |
| 512 | 0.1052 | 0.623 | 1.22 | 0.228 | +4% |
| 1024 | 0.2055 | 0.638 | 0.62 | 0.446 | +2% |
| 2048 | 0.4073 | 0.644 | 0.31 | 0.884 | +1% |
| 4096 | out of memory on the 94 GB card | | | | |

**Finalist's kind of unit (schedule-fit RND with the exact output-layer fit; count-rule units are lighter still):**

| copies | seconds per iteration | total steps per second (millions) | steps per second per copy (thousands) | hours per million steps per copy | gain from doubling |
|---|---|---|---|---|---|
| 64 | 0.0072 | 1.135 | 17.73 | 0.016 | |
| 128 | 0.0085 | 1.936 | 15.12 | 0.018 | +71% |
| 256 | 0.0109 | 2.997 | 11.71 | 0.024 | +55% |
| 512 | 0.0158 | 4.145 | 8.10 | 0.034 | +38% |
| 1024 | 0.0265 | 4.937 | 4.82 | 0.058 | +19% |
| 2048 | 0.0477 | 5.492 | 2.68 | 0.104 | +11% |
| 4096 | 0.0908 | 5.774 | 1.41 | 0.197 | +5% |

Reading:
- **Trained-predictor units: 256 copies per process.** The units run so far held 208 copies (development, about
  0.57 million steps per second) or 128 copies (confirmation, 0.515 million): 5% and 14% below the card's 0.60 million.
  The cards showed 100% utilization and about 5 GB of 80 GB memory in use; memory is not the limit, the card's
  arithmetic is.
- **Exact-fit and count-rule units: 2,048 copies per process.** The units run so far held 128 or 208 copies (about
  1.9 to 2.6 million steps per second) against 5.5 million at 2,048: the card could have done 2.1 to 2.8 times as
  much. Their units finished within minutes to an hour, so the loss was small in hours, but larger units are the rule
  from now on.
- **When cards outnumber the waiting units**, smaller units still finish sooner on the clock (more cards at once);
  the knee sizes apply when units wait for cards, as the baseline's units do now.
- From 2026-11-10 10:45 PT: confirmation units of trained-predictor rows hold 256 seeds
  (`auto_research_units.py --seeds-per-unit 256`); exact-fit and count-rule rows use up to 2,048 copies per unit.

## Earlier profile: serval05 (H100 NVL), 2026-11-09 20:02 PT — taken on a shared card

`serval05_h100_nvl_snake_coordinate-ppo.json`. Another account's processes were on the card during this profile
(they had started at about 19:15 PT; found at 22:31 PT), so its absolute values are about half of the reference
above (control 0.215 s per iteration at 512 copies). Its ratios between ideas agree with the reference. It is kept
as a record and not used for estimates.

## Estimates for the other servers (calculated, no probe there: user rule 2026-11-09)

From the reference profile: an H100 runs the control at about 0.6 million steps per second and the plain-SGD ideas at
about 1.0 million, at 224 to 512 copies per process. Measured afterwards from real units (not probes): Gautschi's
H100 SXM finished 224-copy screen units of 0.94 billion steps in about 1,490 s (0.63 million steps per second).
