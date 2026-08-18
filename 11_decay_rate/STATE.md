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
