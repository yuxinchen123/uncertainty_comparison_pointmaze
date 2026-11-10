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
- **Correction, 2026-11-09 23:20:** the serval05 profile below was taken while another account's processes were on
  the card, so its absolute values are about half the true ones. The reference is the serval02 card 1 profile of
  23:15 (`infrastructure/profiles/README.md`): control 0.106 s per iteration at 512 copies (0.62 million steps per
  second), plain-SGD ideas 0.065 s, exact-step with one step per minibatch 0.114 s.
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
- **2026-11-09 21:26. Policy learning rate matters for the control, so the main ideas also run at 1e-4.** Partial
  progress on the snake grid with coordinate-PPO: the control at policy learning rate 1e-4 has 99-100% of its copies
  at the goal by 262K transitions, against 65-70% for the control at 1e-3 after 2M to 3M. In the control the policy's
  learning rate is also the predictor's, so a lower rate slows its decay (bench: slope −0.7 instead of −1.5 on the
  snake grid). 64 units were added: the eight main ideas at policy learning rate 1e-4 in all four cases. An idea's
  screen score is now its best setting over both policy learning rates, compared with the control's best over both.
- **2026-11-09 22:25. The power-law loss diverges in RL** (`diagnosis/diagnosis.md`). Unit start + power-law q = 4
  had 53 and 59 of 224 copies with a non-finite bonus on the snake grid with coordinate-PPO. Its step factor
  $2\rho^2$ has no upper limit, and early in RL a cell fills whole minibatches. The shrink-target loss has a step
  factor between 0 and 1 and is the method's loss from here on. Eight power-law units that had not started are held in
  `queue/deferred_unstable_power_law/`. The screen report now lists copies with a non-finite bonus.
- **2026-11-09 22:30-23:05. Batch 2 code** (commits `00b86271`, `b264ed38`, `19581d33`; 104 tests and both
  unchanged-path gates pass):
  - generalized shrink map $\rho \to \rho(1+\rho^p)^{-1/p}$, exponent $-1/p$;
  - exact-step optimizer: no learning rate, the step that best realizes every kept cell's shrink target;
  - diagnostic count-rule forms: bonus fixed during a rollout, and a count rate;
  - bench: a fourth law in which each rollout sits in one cell.
- **2026-11-09 22:50. Batch 2 bench** (`bench/batch_02_four_laws/`, serval02 card 1). With unit start and the shrink
  target, SGD 1 and 2 have no non-finite copy under one-cell rollouts; SGD 3 has 3 of 16, SGD 4 has 8 of 16 and SGD 6
  all. Exponent −1 is stable at SGD 1 (slopes −0.93 / −0.86), exponent −0.25 at SGD 2 and 4 (−0.22 to −0.24). The
  exact-step form has slopes −0.31 / −0.25 with one step per minibatch: it under-counts when a minibatch holds many
  cells. The one-cell law did not reproduce the power-law failure, so the RL non-finite count stays the gate.
- **2026-11-09 23:00. Batch 1, partial RL results** (32 seeds, best setting of each idea so far):
  - straight grid, coordinate-PPO: control 3.90K ± 0.20K; separate Adam 1e-4 3.78K; shrink target SGD 1 without unit
    start 6.20K; unit start alone 21.1K; unit start + shrink target only finished at policy learning rate 1e-4
    (32K to 35K), where every idea is slow;
  - snake grid, coordinate-PPO: shrink target SGD 1 without unit start 55.2K ± 8.3K; control 82.1K ± 12.0K (at
    policy learning rate 1e-4); even-start-only variants 1.1M;
  - straight grid, tabular-PPO: initial-ratio readout 7.95K; unit start 8.57K; length readout 9.79K; unit start +
    shrink target SGD 3 14.9K ± 2.5K; control not finished.
  Reading: the control is already fast on the straight grid; the slower decay helps on the snake grid; the stable
  full candidate is not ahead anywhere yet.
- **2026-11-09 23:08. Code rollout for batch 2.** New frozen copy `code/source_19581d33…/` on UVA and on Gautschi,
  Gilbreth, ACES and MIT. Running workers launch each unit from the folder named `source_9cd855bb…`, so on every
  machine that name is now a link to the new copy and the original is kept as `source_9cd855bb….original`. Batch 1
  settings compute the same numbers under the new code (new keys default off; unchanged-path gates).
- **2026-11-09 23:09. Batch 2 queued:** 224 units (`batches/02_…/ideas.md`): 10 diagnostic count-rule rows and 4
  stable shrink-target ideas, at both policy learning rates.
- **2026-11-09 23:20. Exact-step profile and bench** (serval02 card 1). One step per minibatch costs 1.08 times the
  control and is accepted; two and four steps cost 2.2 and 4.2 times and are not used. Bench slopes under uniform
  visits: one step −0.31 / −0.25, two −0.33 / −0.28, four −0.36 / −0.31; with 4 counts per kept visit −0.41 / −0.37.
  No non-finite copy in any law. So the exact step is stable but under-counts when a minibatch holds many cells.
- **2026-11-09 23:20. Gilbreth started:** 7 of its 12 standby jobs began at about 23:15 and passed the first-unit
  check with the corrected tolerance.
- **2026-11-10 00:45. Batches 1 and 2, partial** (170 of 400 units; 32 seeds; best setting per idea; `analysis/`):
  1. **The candidate imitates the handicapped count rule.** Unit start + shrink target (SGD 2) against the count rule
     fixed per rollout at half speed: 10.5K ± 0.8K against 9.0K ± 0.6K (straight, tabular-PPO) and 57.0K ± 5.9K
     against 51.2K ± 6.6K (snake, coordinate-PPO).
  2. **The handicaps cost little.** Per-rollout or per-step, and count rates from 0.25 to 4, all score 47K to 57K on
     the snake grid with coordinate-PPO and 8.1K to 9.0K on the straight grid with tabular-PPO.
  3. **The preferred decay depends on the policy type.** Snake grid, coordinate-PPO: count exponent −0.5 forms 47K to
     57K, control 82.1K ± 12.0K, exponent −1 117K, exponent −2 3.3M (goal found by 25%). Tabular-PPO so far prefers
     variants whose error collapses fast: unit start alone 21.7K on the snake grid; wide input layer 7.4K and
     initial-ratio readout 7.95K on the straight grid.
  4. **Straight grid, coordinate-PPO:** every reasonable variant scores 3.8K to 6.3K; control 3.90K ± 0.20K; the
     candidate 5.07K ± 0.33K.
  5. **Stability:** SGD 3 has 179 of 896 non-finite copies on the snake grid with coordinate-PPO; SGD 2 has 2 of 672
     there and 14 of 448 on the snake grid with tabular-PPO.
- **2026-11-10 01:03. Placement.** MIT and ACES had emptied their queues (count units take minutes), so 82 of the
  units waiting at UVA went to them, with 8 new ACES jobs and 5 new MIT jobs.
- **2026-11-10 01:22. Main finding of batches 1 and 2 (261 of 400 units): the best decay exponent depends on the
  policy type.** Count-rule rows with the bonus fixed during a rollout (32 seeds, best setting):

  | decay exponent | straight, tabular-PPO | snake, tabular-PPO | snake, coordinate-PPO |
  |---|---|---|---|
  | −2 | 7.56K ± 0.61K | 14.1K ± 1.6K | 3,288K ± 282K (goal found by 25%) |
  | −1 | 7.73K ± 0.47K | 16.3K ± 0.9K | 116.9K ± 41.6K |
  | −0.5 | 9.70K ± 0.72K | 40.3K ± 5.5K (quarter speed) | 57.0K ± 4.4K |
  | −0.25 | 22.2K ± 3.2K | 227K ± 55K | 65.8K ± 8.0K |
  | control | lr 1e-4 only: 16.8K | 18.7K ± 1.0K | 82.1K ± 12.0K |
  | unit start + shrink target, SGD 2 | 10.5K ± 0.8K | 61.3K ± 8.1K (14 non-finite copies) | 57.0K ± 5.9K |

  Reading:
  1. With tabular-PPO a faster decay is better; with coordinate-PPO on the snake grid −0.5 is best and −2 fails.
     The hypothesis "−0.5 is the right decay" holds for coordinate-PPO on the snake grid and not for tabular-PPO.
  2. The even start alone does not help: the initial-ratio readout (same decay as the control) scores 24.4K against
     the control's 18.7K on the snake grid with tabular-PPO, and 4.27K against 3.90K on the straight grid with
     coordinate-PPO. Unit start alone scores 2.1M on the snake grid with coordinate-PPO.
  3. The candidate matches its count rule with coordinate-PPO (57.0K against 51K to 57K) and on the straight grid
     with tabular-PPO (10.5K against 9.0K to 9.7K), but trails it on the snake grid with tabular-PPO (61.3K against
     32K to 40K), where it also has non-finite copies.
  4. Straight grid, coordinate-PPO: control 3.90K; every reasonable variant 3.8K to 6.3K; candidate 5.07K (SGD 2),
     5.53K (SGD 1); shrink exponent −1 4.27K.
- **2026-11-10 01:27. Batch 3 queued** (`batches/03_…/ideas.md`, 224 count-rule units; commit `9d038926`, 105 tests
  and both unchanged-path gates pass): exponents −0.75, −1.5, −3; per-step forms at −1 and −2; sums of a fast and a
  slow power law; count rates at −1 and −2. Question: is there one bonus shape that is good for both policy types?
  Code rollout as before; `code/current` now names the newest copy on every machine.
- **2026-11-10 02:35. Batch 3, 137 of 224 units** (32 seeds, best of 28 settings, thousands of transitions; cases
  in the order straight tabular-PPO, straight coordinate-PPO, snake tabular-PPO, snake coordinate-PPO):

  | row | straight, tabular-PPO | straight, coordinate-PPO | snake, tabular-PPO | snake, coordinate-PPO |
  |---|---|---|---|---|
  | control | 7.22 ± 0.28 | 3.90 ± 0.20 | 18.74 ± 0.96 | 82.1 ± 12.0 |
  | count rule −0.5, per step | 8.42 ± 0.39 | 2.38 ± 0.22 | 32.3 ± 4.5 | 52.7 ± 7.8 |
  | count rule −1 at rate 0.25, per rollout | 7.08 ± 0.37 | not finished | 15.92 ± 0.79 | 63.8 ± 9.7 |
  | count rule −1, per step | not finished | not finished | not finished | 41.3 ± 2.8 |
  | count rule −1.5, per rollout | 7.21 ± 0.44 | 15.9 (settings unfinished) | 12.9 ± 1.1 | 2,429 ± 325 |
  | count rule −1 plus 0.3 of −0.5, per rollout | 9.76 ± 1.17 | not finished | 26.2 ± 2.2 | 56.8 ± 3.8 |

  Reading:
  1. The −0.5 rule stays ahead of the control with coordinate-PPO and behind it with tabular-PPO.
  2. Exponent −1 at a quarter of the count rate is level with or ahead of the control in the three cases finished.
  3. A per-step refresh of the bonus beats a per-rollout refresh in every pair measured.
  4. Sums of a fast and a slow power law are not better than −1 alone, so batch 3's third prediction fails.
- **2026-11-10 02:20-02:29. Bench and profile of the rollout fit** (serval05, card free; `bench/batch_04_four_laws/`,
  `infrastructure/profiles/serval05_h100_nvl_snake_coordinate-ppo_rollout_fit.json`). New optimizer `rollout_fit`:
  once per rollout every cell gets an output target (visited cells move along the bonus schedule by their visits,
  the others stay) and the predictor is fitted to all of them.
  - Exact output-layer solve: start 1.00, slope −0.49 on both grids, zero spread between cells, unvisited cells kept
    exactly, under all four visit patterns; 0.015 s per iteration at 512 copies against the control's 0.105.
  - All layers by 12 conjugate-gradient iterations: slope −0.45 / −0.40, spread 0.1, but −1.03 when each rollout
    sits in one cell; same cost as the control.
  - Dropped after measurement: equal weights per visited cell (slope −0.19), and 24 iterations with two steps
    (3.7 times the control's cost).
- **2026-11-10 02:43. Batch 4 queued** (`batches/04_…/ideas.md`, 288 units; commit `84723a80`, 110 tests and both
  unchanged-path gates pass, jobs 6861020 and 6861021). 8 RND rows whose error follows a set schedule (exact fit at
  −0.5 and at −1 with rate 0.25, each refreshed per rollout and per step; −1 and −0.75 per step; all-layers fit at
  −0.5 and at −1 with rate 0.25) and 10 count-rule rows around −1 with slower count rates. Third code rollout as
  before. Units staged: MIT 110, ACES 50, Gautschi 40, Gilbreth 30, UVA 58.
  - With one-hot inputs the exact fit computes the same bonus as a count rule of the same shape (tested). It is the
    RND that has both wanted properties exactly.
  - The −0.5 schedule meets both property targets of `program.md`; the −1 schedule at a quarter rate meets the
    even-start target only. Both are run so the comparison is direct.
- **2026-11-10 03:45. Batch 3 complete (224 of 224, none failed); summary in `batches/03_…/summary.md`.** The count
  rule with exponent −1 refreshed at every step is ahead of the control in all four cases: 6.47 ± 0.26 against
  7.22 ± 0.28 (straight, tabular-PPO), 1.70 ± 0.26 against 3.90 ± 0.20 (straight, coordinate-PPO), 13.31 ± 0.74
  against 18.74 ± 0.96 (snake, tabular-PPO), 41.3 ± 2.8 against 82.1 ± 12.0 (snake, coordinate-PPO). Exponent −0.5
  is ahead only with coordinate-PPO; −1.5 and faster fail on the snake grid with coordinate-PPO. So the data put the
  decay target near −1, with the refresh at every step, and not at −0.5.
- **2026-11-10 03:26. Batch 5 queued** (`batches/05_…/ideas.md`, 64 count-rule units on the frozen tabular-RND
  field; commit `accf9bde`; no trainer change). It is the cell of the 2×2 comparison with an uneven start and a
  scheduled decay.
- **2026-11-10 03:45. Batch 4, 118 of 288.** Every exact-fit RND row finished so far is within one standard error of
  the count rule of the same shape: −0.5 per rollout 3.43 against 3.52 and 39.4 against 42.8; −0.5 per step 2.29
  against 2.38 and 32.2 against 32.3; −1 at rate 0.25 per rollout 6.97 against 7.08; −1 per step 6.33 against 6.47.
  The all-layers fit at −0.5 scores 64.2 ± 4.4 on the snake grid with coordinate-PPO (exact form and count rule: 57.0).
- **2026-11-10 03:05 and 03:23. Rebalances** with the new fleet command `take-back` (waiting units returned at once):
  42 units from Gautschi and 27 from Gilbreth to MIT and ACES, then 24 from MIT to Gilbreth and Gautschi when
  Gilbreth's waiting jobs started.
- **2026-11-10 04:22. Batch 4, 215 of 288; batch 5, 63 of 64** (32 seeds, best of 28 settings, thousands of
  transitions; cases in the order straight tabular-PPO, straight coordinate-PPO, snake tabular-PPO, snake
  coordinate-PPO):

  | row | straight, tabular-PPO | straight, coordinate-PPO | snake, tabular-PPO | snake, coordinate-PPO |
  |---|---|---|---|---|
  | control | 7.22 ± 0.28 | 3.90 ± 0.20 | 18.74 ± 0.96 | 82.1 ± 12.0 |
  | schedule-fit RND, −1, every step | 6.33 ± 0.29 | 1.96 ± 0.29 | 13.51 ± 0.72 | 42.1 ± 2.8 |
  | schedule-fit RND, −1 at rate 0.25, every step | 6.75 ± 0.37 | 2.13 ± 0.19 | 13.69 ± 1.05 | 43.2 ± 3.0 |
  | schedule-fit RND, −1 at rate 0.25, per rollout | 6.97 ± 0.41 | 3.11 ± 0.24 | 16.60 ± 0.87 | 43.8 ± 5.6 |
  | count rule −1 every step, even field | 6.47 ± 0.26 | 1.70 ± 0.26 | 13.31 ± 0.74 | 41.3 ± 2.8 |
  | count rule −1 every step, frozen tabular-RND field | 6.91 ± 0.60 | 2.03 ± 0.26 (unfinished) | 15.06 ± 1.44 | 51.5 ± 7.1 |
  | count rule −0.5 every step, even field | 8.42 ± 0.39 | 2.38 ± 0.22 | 32.3 ± 4.5 | 52.7 ± 7.8 |
  | count rule −0.5 every step, frozen tabular-RND field | 14.23 ± 3.80 | 2.56 ± 0.28 | 67.7 ± 12.7 | 558 ± 208 |
  | unit start alone (batch 1) | 8.57 ± 0.64 | 4.66 ± 0.22 | 21.74 ± 1.24 | 1,905 ± 292 |

  Reading:
  1. **The schedule-fit RND with exponent −1, refreshed every step, is ahead of the control in all four cases**
     (by 12%, 50%, 28% and 49%), and equals its count rule within one standard error.
  2. **2×2 of the two parts.** Scheduled decay with the uneven start is ahead of the control in all four cases;
     the even start adds 7% to 25% on top of it at −1; the even start with the decay left to training (unit start
     alone) is behind the control in all four.
  3. **The even start matters more when the decay is slow:** at −0.5 the uneven field is 69%, 8%, 110% and about
     10 times slower than the even field. With exponent −a, a cell whose start is f times higher needs f^(1/a)
     times the visits to draw level: 4 times at −1 and 16 times at −0.5 for f = 4.
  4. Batch 5's first prediction (uneven start within 30% of the even field) holds at −1 every step and fails at
     −0.5 and at −1 with a quarter rate on the snake grid with coordinate-PPO (104 ± 38 against 39.5).
- **2026-11-10 04:27. Batch 6 queued: finalists on the full protocol, development stage** (`batches/06_…/ideas.md`,
  64 units, commit `47c6d296`; unit builder stages `development` and `confirmation`, commit `3fefc804`). Rows: the
  finalist (schedule-fit RND, −1, every step), the −0.5 form that meets both property targets, and the two
  ablation arms (count rule −1 every step on the frozen tabular-RND field; unit start alone). The 16 long
  unit-start units and the finalist's 16 units are on MIT; the rest on UVA.

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
| count-rule bonus shapes (exponents, count rates, sums of power laws, per-step or per-rollout refresh) | decay (diagnostic) | RL batches 2 and 3: −0.5 best with coordinate-PPO, faster decay best with tabular-PPO; −1 at rate 0.25 level with or ahead of the control in three cases; batch 4 continues |
| shrink-target with other exponents; exact-step optimizer | decay | RL batch 2: matches the per-rollout count rule in two cases, behind it on the snake grid with tabular-PPO |
| rollout fit, exact output-layer solve, schedule −0.5 or −1 at rate 0.25 | both, exactly | bench: start 1.00, slope on the schedule, zero spread; RL batch 4 |
| per-step refresh of the bonus (exact fit) | decay timing | RL batch 4 |
| rollout fit, all layers (12 conjugate-gradient iterations) | both, approximately | bench: −0.45 / −0.40, −1.03 under one-cell rollouts; RL batch 4 |
