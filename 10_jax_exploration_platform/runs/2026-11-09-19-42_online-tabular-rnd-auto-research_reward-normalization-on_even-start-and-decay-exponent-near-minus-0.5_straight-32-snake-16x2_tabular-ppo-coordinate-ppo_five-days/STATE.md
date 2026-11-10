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
