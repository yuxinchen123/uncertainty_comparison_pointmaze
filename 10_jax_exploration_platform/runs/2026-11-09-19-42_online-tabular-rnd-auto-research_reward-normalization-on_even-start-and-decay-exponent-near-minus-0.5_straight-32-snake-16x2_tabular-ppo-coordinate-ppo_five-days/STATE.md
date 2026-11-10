# Online tabular-RND auto research: state and log

Times are Pacific. The protocol is in `program.md`; the plan is
`~/.claude/plans/2026-11-09-19-03_tabular-rnd-extension-and-online-tabular-rnd-auto-research.md`. Deadline 2026-11-14
20:00 PT.

## Where things are

- **Code:** repository commit `9cd855bb` (pushed). The frozen copy is `code/source_9cd855bb…/`.
- **Records:**
  - bench results: `bench/batch_NN/<idea>.json`;
  - RL units: `data/research/<unit>/summary.json`;
  - ledgers: `slurm/submitted_jobids_*.txt`.
- **Baseline:** the extension run `runs/2026-11-09-19-40_fixed-…-extension_…`, condition "online tabular-RND, reward
  normalization on". It is confirmed on 1,024 seeds once its development finishes.

## Log

- **2026-11-09 19:42.** Folder created. The literature helper downloaded 27 papers to `literature/pdfs/` (Ciosek et
  al. 2020 could not be downloaded) and wrote `notes.md`, `sources.md` and a first `novelty.md`.
- **2026-11-09 20:00-20:30. Bench, batch 1.** 35 idea variants were run on jaguar03 (4 RTX A4500 cards, 15 minutes):
  8 field seeds, uniform and start-heavy visits, 4,096 iterations of the RL update shape. Findings:
  1. **The CleanRL baseline is uneven and decays far too fast.** The starting bonus differs 3.65× (straight) and
     4.23× (snake) between cells. Per-cell decay slopes are −3.1 / −2.5 (straight, uniform / start-heavy visits) and
     −1.5 / −1.3 (snake). The late count consistency is 1.0-1.9 (the sd of log b√N).
  2. **The decay is set by the optimizer's fixed-size steps.** A separate Adam at a constant learning rate of 1e-3
     equals the baseline. At 1e-4 the slopes are −1.2 / −0.7, still too fast on the straight grid. AdaGrad gives
     −3 to −5 at 1e-3 and 1e-2, and about −0.2 at 1e-1.
  3. **Even-start constructions alone fix the start, not the decay.**
     - The initial-ratio readout and the zero output with a unit-length target start at exactly 1. Their decay is
       the baseline's (−3.1) or faster (−5).
     - Random biases with standard deviation 0.5 start nearly even (1.04×) and decay at about −1.1.
     - A wider input layer (64) reduces the spread only to 2.6-3.0×.
  4. **Loss-set decay works with plain SGD.**
     - The shrink-target loss (learning rate 1) and the power-law loss q = 4 (0.1) give slopes −0.2 to −0.46.
     - Their late count consistency is 0.07-0.17: every cell follows the same law of its own visits.
     - The power-law loss diverges at learning rate 1 (q = 4) and at 0.1 (q = 6).
  5. **Both properties together:**
     - zero output + unit-length target + shrink-target, learning rate 1: start 1.00, slopes −0.46 / −0.44 / −0.41 /
       −0.40, consistency 0.07-0.12;
     - the same with power-law q = 4 at learning rate 0.1: −0.40 to −0.33.
     - A learning-rate refinement bench (power-law 0.2-0.5, shrink-target 2-5) is running.
  6. **Training only the input rows of a frozen trunk does not fit:**
     - frozen random deeper layers: slope about −0.1;
     - the target-copy trunk: −0.1 to −0.16 at learning rates up to 1.
- **2026-11-09 20:07. serval05 profile (H100 NVL, snake grid, coordinate-PPO).** Seconds per iteration at 512 copies:
  - control 0.215;
  - power-law and shrink-target with plain SGD 0.136;
  - target-copy trunk 0.035.

  Total throughput is flat above 512 copies (0.30 million steps per second for the control). So the new ideas are
  faster than the baseline and need no speed work. One H100 NVL is about 2 RTX A4500 cards. The table is in
  `infrastructure/profiles/serval05_h100_nvl_snake_coordinate-ppo.json`.
- **2026-11-09 20:47.** The control was also queued at policy learning rate 1e-4 (8 units), so the screen's control
  is not weak where the baseline prefers that rate. The batch's ideas and predictions were written to
  `batches/01_…/ideas.md` before any RL unit finished.
- **2026-11-09 20:45-21:10. Three-law bench** (`bench/batch_01_three_laws/`; serval05, and serval02 card 1 while
  another account held serval05). The third law visits only the first half of the cells, about 33K visits per visited
  cell on the straight grid. It measures what never-visited cells keep. The count rule would end with a
  never-visited / visited bonus ratio of about 180 (straight) and 145 (snake).

  | variant | never-visited cells keep, straight / snake | never-visited / visited bonus at the end |
  |---|---|---|
  | control | 29% / 32% | over 10^11: the visited bonus falls to almost nothing |
  | unit start only | 52% / 55% | over 10^13 |
  | separate Adam 1e-4 | 32% / 34% | 10^9 / 10^4 |
  | length readout | 54% / 56% | 10^6 / 10^5 |
  | power-law q = 4, SGD 0.1, no unit start | 50% / 54% | 12 / 6.5 |
  | shrink target, SGD 1, no unit start | 52% / 54% | 20 / 9.5 |
  | unit start + AdaGrad 0.1 | 86% / 74% | 10^12 / 10^3: cells do not follow one law |
  | unit start + power-law q = 4, SGD 0.3 | 71% / 75% | 64 / 48 |
  | unit start + shrink target, SGD 2 | 71% / 75% | 82 / 62 |
  | unit start + shrink target, SGD 3 | 73% / 74% | 94 / 74 |

  Reading: the two full candidates behave like a count rule with about a quarter to a half of a count per visit, and
  never-visited cells keep about three quarters of their start. The control instead drives visited cells' bonus to
  nearly zero within a few visits. Loss powers for other exponents: q = 6 at learning rate 0.1 gives slopes −0.19 /
  −0.16 (the target is −0.25); q = 3 at learning rate 1 diverges, so it needs a smaller rate.
- **2026-11-09 21:02-21:07. Remote fleet started** (`scripts/grid_initialization_campaign/remote_fleet.py`, log in
  `slurm/fleet_log.txt`). 46 units were staged from the end of the queue: Gautschi 6, ACES 16, Gilbreth 12, MIT 12.
  20 worker jobs were submitted; the 2 Gautschi H100 jobs started at once and passed the first-unit check (all 656
  arrays identical to UVA's). The others wait on priority. Units staged for 2 hours on a server where no job has
  started go back to UVA at the next tick. MIT is reached through the connection open on portal11.

## Idea list

| idea | targets | status |
|---|---|---|
| initial-copy ratio readout | even start | bench: start even, decay unchanged; RL batch 1 |
| zero output + unit-length target ("unit start") | even start | bench: start even, decay −5; RL batch 1 |
| random bias start, standard deviation 0.5 | even start | bench: 1.04×, −1.1; RL batch 1 |
| wide one-hot input layer (64) | even start | bench: 2.6-3.0×; RL batch 1 |
| separate Adam, constant learning rate | decay (diagnostic) | bench: 1e-3 equals the baseline, 1e-4 slows it; RL batch 1 at 1e-4 |
| length readout | decay (halves the exponent) | bench: start 1.9-2.1×, −1.5 / −0.7; RL batch 1 |
| power-law loss q = 4, plain SGD | decay | bench: right law, learning rate decides the slope; refinement running |
| power-law loss q = 6, squared readout | decay | bench: −0.15 at 1e-2, diverges at 1e-1; dropped |
| shrink-target loss, plain SGD | decay | bench: right law; refinement running |
| AdaGrad | decay | bench: −3 to −5 or −0.2, poor consistency; with unit start at 1e-1: −0.49 / −0.28; RL batch 1 |
| SGD with a 1/t learning rate | decay | bench: about −0.2; dropped |
| input rows only (frozen trunk or target-copy trunk) | per-cell decay | bench: does not fit (−0.1); dropped |
| unit start + power-law q = 4 | both | candidate; RL batch 1 |
| unit start + shrink-target | both | candidate; RL batch 1 |
