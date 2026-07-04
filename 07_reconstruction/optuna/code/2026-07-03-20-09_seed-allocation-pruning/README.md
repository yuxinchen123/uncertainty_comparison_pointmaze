# Optuna seed-allocation and pruning verification

Date: 2026-07-03. Optuna 4.9.0, numpy 1.26.4, scipy 1.16.0, Python 3.11 (conda env `exploration`).

These scripts verify, empirically, how the next RND sweep's seed-allocation rule is expressed in
Optuna and what Optuna offers natively. The user's rule: run every hyperparameter combination for 10
seeds first, then spend the remaining computation only on combinations whose mean plus the upper end
of the 95% confidence interval is still at or above 35 (i.e. `mean + 1.96*sd/sqrt(n) >= 35`).

## Shared simulation

`sim.py` builds 12 hyperparameter "cells" mirroring a 12-cell sweep. Each cell has a bimodal per-seed
reward: with probability `p = true_mean/80` a seed succeeds and draws `Normal(80, 10)`, else it fails
and draws `Normal(0, 3)`. The 12 true means are `[52.8, 35.2, 29, 26.1, 26, 17.9, 17.3, 9.6, 4.9, 1,
0.5, 0.1]`, so the true survivors (mean at or above 35) are cells 0 and 1. Every `(cell, seed)`
outcome is fixed via a per-quantity keyed substream `substream(base_seed, "cell", c, "seed", s)`. One
seed run costs 1 unit; the full grid is 12 cells x 50 seeds = 600 seed-runs.

## Scripts and how to rerun each

Run each from this directory. Each command tees its real printed output to `<name>_output.txt`.

- `sim.py` — sanity check that empirical means track the 12 targets.
  `conda run -n exploration python sim.py 2>&1 | tee sim_output.txt`
- `01_custom_pruner.py` — point 1: one trial per cell, intermediate seed reports, and a custom
  `BasePruner` subclass applying `mean + 1.96*sd/sqrt(n) < 35` after 10 seeds; run at 3 base seeds.
  `conda run -n exploration python 01_custom_pruner.py 2>&1 | tee 01_custom_pruner_output.txt`
- `02_manual_prune.py` — point 2: the same rule inside the objective with no pruner object, raising
  `optuna.TrialPruned()` directly.
  `conda run -n exploration python 02_manual_prune.py 2>&1 | tee 02_manual_prune_output.txt`
- `03_wilcoxon.py` — point 3: built-in `WilcoxonPruner`, reporting one seed per step, comparing
  against the best completed cell.
  `conda run -n exploration python 03_wilcoxon.py 2>&1 | tee 03_wilcoxon_output.txt`
- `04_successive_halving_hyperband.py` — point 4: `SuccessiveHalvingPruner` and `HyperbandPruner`
  with resource = seed count.
  `conda run -n exploration python 04_successive_halving_hyperband.py 2>&1 | tee 04_successive_halving_hyperband_output.txt`
- `05_facts.py` — point 5: the pruning interface facts (cooperative pruning, PRUNED state, partial
  curve survival, default pruner type).
  `conda run -n exploration python 05_facts.py 2>&1 | tee 05_facts_output.txt`

## What the runs showed (summary)

- The user's rule is expressed two equivalent ways: a custom `BasePruner` subclass (only `prune`
  must be implemented) checked via `trial.should_prune()`, or a plain `if ...: raise
  optuna.TrialPruned()` inside the objective. Both gave the identical result at base seed 0 (252
  seed-runs, survivors cells 0 and 1). The manual version is the simplest and needs no pruner object.
- Cost across 3 base seeds: 238-308 seed-runs vs the full grid's 600. Surviving sets were `{0,1}`,
  `{0,2,3,4}`, `{0,2}` — because the per-seed reward is strongly bimodal, early lucky or unlucky
  seeds move the 95% bound enough to keep a truly-below-35 cell or prune a truly-at-35 cell (cell 1,
  true mean 35.2, sat right on the bar and was pruned in 2 of 3 base seeds).
- Optuna's native pruners answer a different question: `SuccessiveHalvingPruner`, `HyperbandPruner`
  and `WilcoxonPruner` all prune relative to the OTHER cells (top fraction, or statistically worse
  than the best), not against an absolute bar at 35. None of them implements the user's absolute rule.
