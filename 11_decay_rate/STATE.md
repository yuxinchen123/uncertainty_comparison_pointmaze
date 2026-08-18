# 11_decay_rate campaign state

All times Pacific (PT), converted from machines running Eastern.

## Status

- 2026-08-17 21:50 PT — campaign start. 36-hour budget. Cloned karpathy/autoresearch for
  reference (gitignored). Literature sweep running in the background (6 agents; notes land in
  `literature/`). Building the fixed harness next.

## Standing facts

- Reservation `sl5nw_156` (jaguar03 + puma01) ends 2026-08-19 20:59 PT — usable for the whole
  campaign.
- Prior art: 07_reconstruction "Convergence rate runs" section. Best aggregate slope -0.503
  (zero-bias init, SGD-1/t eta0=1e-2 t0=1e2); Adam best -0.83. Per-position heterogeneity is
  unsolved — that is this campaign's goal metric (dev_worst).
- 55 Slurm jobs under uid sl5nw belong to OTHER sessions. Never touch any job id not in
  `experiments/submitted_jobids.txt`.

## Method idea backlog (updated as literature notes arrive)

- N1 initial-copy normalization: keep frozen copy of the predictor at init; readout
  ||g_n - f|| / ||g_0 - f||. Starts at exactly 1 everywhere.
- N2 unit-norm target + zero-init predictor head: g_0 = 0 and f normalized per input to
  ||f(x)|| = 1, so the initial residual is exactly 1 at every x.
- O-A SGD-1/t baseline (prior work's best).
- O-B iterate averaging (Polyak/tail): averaged-iterate residual should decay as 1/n per
  eigenmode uniformly under constant-step full-batch GD -> sqrt readout gives -1/2.
- O-C AdaGrad (count-like accumulator).
- O-D per-position loss reweighting w_i = 1/(||e_i||^2 + eps) with stop-grad, to equalize
  relative decay across positions; combine with a global schedule for the -1/2 shape.
- O-E coin-flip auxiliary head (CFN, Lobel et al.): magnitude of the learned mean of random
  +-1 targets behaves as 1/sqrt(n) by construction; works per position under non-uniform
  visitation.
- O-F architecture: residual MLP, LayerNorm, Fourier features on the 2-D input (flattens the
  NTK spectrum -> more uniform per-position rates).
- O-G Gauss-Newton / full-matrix preconditioning: makes every residual mode decay at the SAME
  geometric rate; then a step-size schedule eta_t ~ 1/(2t) shapes the common decay into
  n^{-1/2} exactly, at every position at once.
- O-H ensemble over several predictors (variance reduction of the init draw).

## Experiments

Ledger: `results.tsv`. Campaign folder:
`experiments/2026-08-17-22-05_autoresearch_uniform-fullbatch_cellmid108-center100_4096step_seed0-9`.
One experiment runs in ~22 s on jaguar03 (job ids in the folder's `slurm/submitted_jobids.txt`;
20-minute cron monitor id 8ab1c6de is armed).

- 2026-08-17 22:20 PT — exps 001–003 done. Findings so far:
  1. exp 001 (Adam): slope_mean -0.86 (matches prior work's -0.83), homogeneous-ish
     (slope_std 0.13) but wrong rate everywhere; start_dev 1.37.
  2. exp 002 (SGD-1/t, prior best): normalized aggregate slope is -0.80 on cell_midpoints but
     -0.38 on center_square — the prior "-0.503 aggregate" was an average over heterogeneous
     point sets. slope_std 0.78; worst position slope -4.55.
  3. exp 003 (SGD-1/t + initial-copy readout normalization): start_dev EXACTLY 0 —
     requirement (1) is solved by one frozen network copy. Slopes unchanged; dev_worst now
     dominated by a nearly-flat position (slope -0.001). Requirement (2) is the open battle.
- Harness note (outside the loop, before exp 004): added `agg_slope_norm` (the prior work's
  normalize-then-average convention) to metrics.py for comparability; all three metrics.json
  recomputed.
- 2026-08-17 23:00 PT — exps 004–010 done. The findings that changed the plan:
  1. exp 005 (functional shrink, inner Adam): inner Adam's fixed-size steps overshoot the
     nearby target — acts like fast distillation (slope -1.8). Inner fitting must use SGD.
  2. exp 007/009 (AdaGrad): the best gradient-descent family. lr 3e-3: dev_mean 0.32,
     slope_std 0.20, both envs' aggregate slopes near -0.5. Two defects: an early first-step
     drop (target stays at 1, curves drop to ~0.55–0.8) and corner positions decaying ~0.17
     too fast (radial pattern = tangent-kernel diagonal grows with input radius).
  3. exp 008 (sphere-projected inputs): fixes the radial pattern but collapses nearby points
     into near-identical inputs — worse overall. Discarded.
  4. **exp 010 (linear head + exact min-change interpolation + residual-encoded shrink)
     SOLVES the uniform benchmark**: per-visit map r -> r/sqrt(1+r^2) (bonus value itself
     encodes the count; no counter, no clock), realized exactly by a minimum-Frobenius-change
     head update. Slope -0.500000 at every position, dev 0.038 = the (n+1 vs n) off-by-one,
     identical at all 208 positions.
- 2026-08-18 00:30 PT — waves 4–7 done (exps 011–031; exps 032–033 in flight). The picture:
  1. UNIFORM regime is solved twice over: the residual-encoded shrink (exp 010/023/024/026,
     dev 0.038 = the (n+1 vs n) off-by-one, slope -0.500000, self-correcting under capacity
     stress) and coin-flip + adaptive dictionary (exp 019/021, dev 0.053, statistical floor).
  2. NONUNIFORM regime ranking: coin-flip adaptive 0.455 > elliptical closed form 0.535 >
     shrink adaptive 1.31 (the multiplicative recurrence integrates minibatch interference
     drift; least-squares statistics average it out).
  3. Deep-RL-practical gradient-only methods: AdaGrad 3e-3 best of the optimizer family
     (dev_mean 0.32 uniform); quartic loss = slope -0.4-ish with level spread (exps 030/031
     pending); MLP inner-SGD shrink fails on slow kernel modes (exp 013).
  4. Literature sweep digested (6 agents, notes in literature/): the per-position law
     b_i^2 = sum_j w_ij (1-eta lambda_j)^{2n}; no global schedule can equalize (mode
     log-decay ratios are fixed at lambda_k/lambda_j); measured NTK condition 5.6e5;
     coin-flip = CFN (Lobel et al. ICML 2023); the level-vs-slope decomposition; DRND names
     "initial bonus inconsistency"; PINN line (Chen/Howard/Stinis) is the only prior work
     targeting per-point rate equalization; Li et al. landscape/ResNet story does NOT apply
     at this width/depth (answer to the user's pointer).
- 2026-08-18 04:00 PT — validations, ablations, horizon stress done (ledger rows through the
  abl/hor block). Highlights: collision-free Hadamard coins (val_117) are the new uniform
  champion (dev_worst 0.0335, dev_mean 0.0031 — below the shrink's off-by-one floor); the
  Hadamard family needed two bug fixes (block-sum collapse, spike collisions), both diagnosed
  from theory and verified; ablations confirm the chi-floor scaling, the insertion-radius
  requirement, and the bandwidth-overlap trade; the shrink holds -0.500000 over 4.5 decades
  while AdaGrad drifts. Development document: 16 pages, compiled, bibliography verified (18
  new entries via the collector+2-verifier workflow), literature section written.
- RL PILOT LAUNCHED 2026-08-17 23:30 PT: new `coinflip_count` intrinsic model in the parent
  package (additive registry row; all 241 package tests pass after the pin-test update).
  Smoke passed (20k steps, completed record, bonus decays per episode, 1458 steps/min).
  Sweep `2026-08-17-23-27_cfpilot`: 60 runs = 4 configs (coinflip_count beta 0.3/1/3 +
  gt_position_velocity beta 1 control) x 15 seeds x 1M steps on
  PointMaze_Large-v3_start_bottom_left (run-8.1's setting; its winner table is the
  reference). Worker jobs 6539941 (32x1 bigcat06) + 6539942 (30x1 affogato02), AllocCPUS
  verified, open cpu partition, --time 4-00:00:00. Expected finish ~16:00-20:00 PT Aug 18.
  Interim table: `analysis/code/pilot_table.py`. Monitoring cron c7cdaf7e covers
  campaign + pilot.
- Remaining: pilot smoke -> fleet launch -> monitor to completion -> pilot analysis + doc
  section; final document regeneration (val_117/118 + hor_302 rows), page QA, final report.
