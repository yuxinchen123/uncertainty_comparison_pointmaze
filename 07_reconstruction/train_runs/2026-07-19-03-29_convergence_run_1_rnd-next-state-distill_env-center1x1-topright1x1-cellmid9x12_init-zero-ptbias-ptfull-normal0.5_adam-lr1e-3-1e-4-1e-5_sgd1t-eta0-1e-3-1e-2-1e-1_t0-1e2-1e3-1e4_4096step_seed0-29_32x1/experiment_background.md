# Convergence run 1 — experiment background

## Purpose

Measure the convergence RATE of the RND intrinsic bonus in isolation from the RL loop. Each run
holds a fixed set of maze points, forwards ALL of them through the frozen target and the
predictor in one batch, takes one optimizer step on the standard squared distillation objective,
and repeats for 4096 steps — recording the per-point l2 bonus at log-spaced checkpoints. The
analysis fits a power decay with a floor, y(n) = c + a*(n + n0)^(-alpha), per seed and on the
seed-averaged curve (reported separately), and ranks configurations by closeness of the fitted
slope (-alpha) to -1/2 — the slope a perfect one-over-square-root-of-visit-count bonus would
show. Full design, methodology, and both hyperparameter tables:
`development_document/main.tex` Section "Convergence rate runs"
(`sec:convergence-rate-runs`, subsections `sec:convergence-fit-method` and
`sec:convergence-run-1`).

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| trainer | `07_reconstruction/convergence_train.py` (pure distillation; no SAC, no env stepping) |
| networks | 4 -> 256 -> ReLU -> 128, target frozen (the run-3.2.x architecture) |
| training objective | squared distillation loss (mean of half the squared residual norm); full batch |
| steps per run | 4096 full-batch optimizer steps |
| point sets | env 1.1 center square (100 pts, half over a wall cell — deliberate); env 1.2 top-right goal cell (100 pts); env 2 all 9x12 cell midpoints (108 pts, walls included) |
| velocities | vx = vy = 0 for every point |
| input | raw [x, y, 0, 0]; observation normalization OFF; reward normalization OFF |
| initializations | I1 zero (orthogonal sqrt(2) + zero bias = RND paper = CleanRL); I2 ptbias (orthogonal + PyTorch-default bias); I3 ptfull (full PyTorch default, NEW `rnd_weight_init` switch); I4 normal0.5 (orthogonal + N(0, 0.25) bias = run-3.2.4 winner V1's init) |
| optimizers | Adam lr in {1e-5, 1e-4, 1e-3 (default)}; SGD-1/t eta0 in {1e-3, 1e-2, 1e-1} x t0 in {1e2, 1e3, 1e4}; 12 configs |
| seeds | `a_seed` 0-29 (30 seeds; each re-draws ONLY the initialization of both nets) |
| total runs | 3 x 4 x 12 x 30 = 4320 (about 15 s each single-threaded; ~18 core-hours) |
| logging | per-point l2 bonus (exact norm, clamp dropped in logging) at 80 checkpoint steps: {0} + round(2^(j/8)), j = 0..96, deduplicated; atomic JSON per run with completed/diverged flags |
| run ids | seed OUTERMOST: run_id = 144*seed + 48*point_set + 12*init + opt (pinned by `slurm/test_run_id_convention.py`) |
| env python | `/p/rlprojects/RND/.venvs/exploration/bin/python` (shared canonical env) |

## Code and config changes

- NEW `07_reconstruction/convergence_train.py` — the standalone distillation trainer (calls the
  exact `RND.update()` code path; divergence guard writes `diverged: true` and stops early).
- `src/rnd_exploration/methods/rnd.py` — NEW `weight_init` constructor arg
  (`orthogonal` default, bit-identical to all prior runs | `pytorch_default` =
  U(-1/sqrt(fan_in), +1/sqrt(fan_in)) weights from a new keyed stream `weight-uniform`).
- `train.py` + `methods/__init__.py` — thread `rnd_weight_init` through Config/argparse/factory
  and record it in run JSONs.
- NEW tests: `tests/methods/rnd/test_weight_init.py`, `tests/convergence/test_convergence_run1.py`
  (checkpoint grid, point sets, end-to-end mini run, divergence guard) — all passing, plus the
  existing `tests/methods` suite (81 passed).
- NEW figure `development_document/code/2026-07-19-03-21_convergence-run1-point-sets/` (the three
  point sets on the true 9x12 maze grid), included in the new main.tex section.
- This run folder's `slurm/` scripts are adapted from run 3.2.4 (same queue/worker mechanics;
  `build_cmd` targets `convergence_train.py`).

## Git state

Commit: `55eb6bb3e21415e309deda48418d48dca3cc524e` (branch `Use-RLexplore-RND`)

Working tree dirty — the convergence-run-1 changes above are uncommitted at launch time
(plus unrelated in-flight edits to `.claude/` rules, `development_document/`, and prior run
folders; `code/` in this folder holds the exact `src/rnd_exploration` + `convergence_train.py`
snapshot used by the sweep).

## Completion and exclusions (2026-07-19)

Sweep `2026-07-19-03-32_convergence-run1` completed the same night: 4320/4320 runs
`completed: true`, 0 failed, 13 minutes wall on 192 workers (jobs 6512782-6512787, all
COMPLETED). Exclusion record (analysis-convention): 180 runs have `diverged: true` and are
excluded from every fit — exactly the 6 cells x 30 seeds of SGD-1/t at learning rate 0.1 on
env 2 (cell midpoints) for initializations I3 (full PyTorch default) and I4 (normal-0.5 bias),
at every t0. All other cells have post-exclusion n = 30. Details: `analysis/analysis.md`.
