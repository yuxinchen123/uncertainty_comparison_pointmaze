# Experiment background — 11_decay_rate autoresearch campaign (iteration phase)

## Purpose

Find a training method whose RND-style bonus starts at 1 at every maze position and decays as
one over the square root of the visit count at every position simultaneously (program.md at the
project root states the full goal and rules). This campaign folder holds the ITERATION-phase
experiments: one subfolder `exp_NNN_<method>/` per experiment, each produced by editing
`code/method.py`, committing, and running the fixed protocol below. The tracked ledger of all
experiments is `/p/rlprojects/RND/11_decay_rate/results.tsv`; the per-experiment method code is
recoverable from the commit hash in that ledger.

## Key hyperparameters (the fixed protocol; per-experiment knobs live in method.py per commit)

| Parameter | Value |
|-----------|-------|
| point sets | `cell_midpoints` (108 points, walls included), `center_square` (100 points) |
| input | raw `[x, y, 0, 0]`, no observation normalization |
| regime | `uniform_fullbatch`: one update per step on the full point set |
| steps | 4096 |
| checkpoints | step 0 + the 79 log-spaced steps `round(2^(j/8))` |
| seeds | 0–9 (10 per experiment; validation runs use 30) |
| metric window | checkpoint steps 2–4096 |
| primary metric | `dev_worst` = seed-mean of the per-record max over positions of the RMS log deviation from `min(1, m^{-1/2})` |
| runner | `code/run_experiment.py` (fixed), 20 worker processes, 1 torch thread each |

## Code and config changes

- New subproject `11_decay_rate/`: fixed harness (`code/decay_harness/`), editable
  `code/method.py`, fixed runner `code/run_experiment.py`, all new in this campaign's first
  commit. Point sets, checkpoint grid, and the floored power fit are ported unchanged from
  `07_reconstruction/convergence_train.py` and convergence run 1's `fit_convergence.py`.

## Git state

Recorded per experiment in `results.tsv` (column 1 = the commit whose `method.py` ran). The
campaign scaffold commit is the first commit touching `11_decay_rate/` on branch
`Use-RLexplore-RND`; the tree is committed before every submission (commit-before-submit rule).
