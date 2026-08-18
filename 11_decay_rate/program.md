# 11_decay_rate — research program

Adapted from the loop of `autoresearch/program.md` (karpathy/autoresearch, cloned here for
reference). One agent iterates on ONE method file, runs a fixed-protocol experiment per
iteration, logs every experiment to `results.tsv`, and commits per experiment.

## Goal

Find a training method for the Random Network Distillation (RND) bonus such that, when a
predictor network is trained by repeated full-batch distillation against a frozen random
target on a fixed set of maze positions, the per-position bonus $b_i(n)$ after $n$ optimizer
steps satisfies, at EVERY position $i$ simultaneously:

1. **Start at one.** $b_i(0) = 1$ for every $i$ (exactly, or as close as a legitimate online
   normalization allows).
2. **Decay as one over the square root of the visit count.** $b_i(n) \approx m_i(n)^{-1/2}$
   where $m_i(n)$ is the number of times position $i$ has been trained on after $n$ steps
   (in the full-batch regime $m_i(n) = n$ for every $i$).

Prior work (07_reconstruction development document, "Convergence rate runs"): constant-rate
Adam decays too fast (aggregate slope about $-0.83$); SGD with a $1/t$ learning-rate schedule
reaches aggregate slope $-0.503$, but different positions still decay at different rates and
start from different values. The aggregate is solved; the per-position problem is the goal.

## The fixed experiment protocol (never modified by the loop)

- Point sets: `cell_midpoints` (the 108 midpoints of every PointMaze_Large-v3 cell, walls
  included — the primary set) and `center_square` (100 sub-cell midpoints of the 1x1 square at
  the maze center). Raw inputs $[x, y, 0, 0]$, no observation normalization.
- Training regime `uniform_fullbatch`: one update per step on the full point set, 4096 steps.
- Training regime `nonuniform` (extension): each step trains on a 32-point batch sampled
  i.i.d. from a fixed non-uniform distribution over the point set; the target curve at each
  position uses that position's own realized visit count.
- Checkpoints: per-position bonus stored at step 0 and the 79 log-spaced steps
  $\mathrm{round}(2^{j/8})$ up to 4096 (the prior work's grid).
- Seeds: 10 for iteration experiments, 30 for validation runs. Each seed re-draws every
  network initialization; nothing else is random in the uniform regime.
- One experiment = `run_experiment.py` over (point set x seed) cells; per-cell JSON records
  with atomic checkpointed flushes; re-running skips completed cells (resumable).

## Metrics (computed by the fixed harness, `code/decay_harness/metrics.py`)

Define the target curve $T_i(n) = \min(1, m_i(n)^{-1/2})$ and the fit window as checkpoint
steps $2 \le n \le 4096$ (the prior work's burn-in). For each seed $s$ and position $i$ the
per-position deviation is the root mean square of $\log b_{s,i}(n) - \log T_i(n)$ over
in-window checkpoints. Metrics, in ranking order:

- **dev_worst** (primary): mean over seeds of the maximum per-position deviation. This is the
  goal metric — "every position at the same time" means the worst position matters.
- **dev_mean**: mean over seeds and positions of the deviation (tiebreak).
- **start_dev**: root mean square over seeds and positions of $\log b_{s,i}(0)$ — requirement
  (1) in one number.
- **slope battery**: per-position decay exponents from the floored power fit
  $y = c + a\,(n+n_0)^{-\alpha}$ on seed-mean curves — mean, standard deviation, min, max of
  $-\alpha_i$, plus the aggregate-curve slope (comparable to the prior work's $-0.503$).
- **diverged / floor-hit counts**: excluded curves are reported, never silently dropped.

Lower dev_worst is better. An experiment "wins" if it lowers dev_worst without degrading
dev_mean or start_dev materially and without breaking the sanity rules below.

## What the loop CAN do

- Modify `code/method.py` — the ONLY file the loop edits. Everything inside is fair game:
  predictor and target architecture, initialization, optimizer, learning-rate schedule, loss,
  bonus readout, ensembles, auxiliary networks (e.g. coin-flip heads), running statistics,
  frozen copies of past networks.

## What the loop CANNOT do

- Modify `code/decay_harness/` or `code/run_experiment.py` (the protocol and the metrics are
  the ground truth). A genuine harness bug is fixed outside the loop and noted in `STATE.md`,
  never as part of an experiment.
- **No oracle counts in the readout.** The bonus readout may depend on the input, the
  networks, and running statistics that an online RL implementation could maintain — but it
  must NOT multiply or divide by an explicit function of the step counter or of per-position
  visit counts. (A learning-RATE schedule may use the step counter; that is standard
  optimizer practice. The distinction: the optimizer may know the time, the readout may not.)
  Reason: `bonus = anything / sqrt(n)` scores perfectly and learns nothing.
- No per-position lookup tables keyed by training-set index (the method must be a function of
  the input coordinates, so it generalizes off the training set).

## The experiment loop

1. Read `STATE.md` (campaign state) and `results.tsv` (what has been tried).
2. Edit `code/method.py` with ONE idea (or a controlled combination).
3. Commit (message: `11_decay_rate exp NNN: <short description>`).
4. Submit the experiment to Slurm (jaguar03 + the active reservation; job id appended to
   `experiments/submitted_jobids.txt` at submit time) and wait for it.
5. Read `experiments/<stamp>_<name>/metrics.json`, decide keep / discard, append one row to
   `results.tsv`, update `STATE.md`.
6. Repeat. Validation runs (30 seeds, both regimes) for any method that leads the board.

## results.tsv columns (tab-separated)

```
commit	dev_worst	dev_mean	start_dev	slope_mean	slope_std	status	description
```

`status` is `keep`, `discard`, or `crash`; crashed experiments log `inf` metrics. Every
experiment gets a row, failures included.

## Simplicity criterion (kept from autoresearch)

All else equal, simpler is better. A tiny dev_worst gain that needs a fragile stack of tricks
is worth less than a clean method with one mechanism; a simplification that keeps the score is
a win.

## Deliverable

`development_document/11_decay_rate_development_document.tex` in the style of
`07_reconstruction/development_document/RND_development_document.tex`: notation, methods,
metric definitions, literature, experiment platform, runs with tables and figures, findings.
