# TPE behavior under high per-seed variance (Optuna 4.9.0)

Date: 2026-07-04
Environment: `conda run -n exploration python` (Python 3.11, optuna 4.9.0, numpy 1.26.4, scipy 1.16.0).

These scripts answer, empirically, how Optuna's default sampler (`TPESampler`) behaves when each trial's
value is one draw from a high-variance, strongly bimodal per-seed reward. Every printed number is the real
output of the script next to it (`<script>_output.txt`), produced by driving the installed optuna 4.9.0
code, not a re-implementation.

## The noise regime the toys are calibrated to

Per-seed reward is bimodal: a configuration whose true mean is `m` (in 0..80) succeeds with probability
`m/80` giving a draw near `Normal(80, 10)`, else collapses to a draw near `Normal(0, 3)`. The per-seed
standard deviation is large at mid-range means (about 35-41). `common.py` holds this reward model, the
keyed random-number helper, the landscapes, and the routine that drives the sampler internals.

## What each script verifies and how to rerun it

1. `01_acquisition_dent.py` — the mechanism at small scale. A study whose good group is 20 observations
   near x=8 (values 45-55), plus 180 background trials so the split size gamma = ceil(0.1*n) = 20 puts
   exactly those 20 in the good group; then 0/1/3/6 bad observations (value ~0) are added at exactly x=8.
   For each count it drives `_split_trials` + `_build_parzen_estimator` + `_compute_acquisition_func` to
   get the acquisition log l(x) - log g(x) on a 0..10 grid and reports the argmax and the notch at x=8.
   Result: the argmax stays at the peak (x≈7.96) with 0-1 bad draws and moves to the shoulder (x≈8.76)
   by 3-6 bad draws; the region is dented, not excluded.
   Rerun: `conda run -n exploration python -u 01_acquisition_dent.py 2>&1 | tee 01_acquisition_dent_output.txt`

2. `02_recovery.py` — does TPE return to the true-best region after unlucky early failures there? Landscape
   m(x) = 50*exp(-(x-8)^2/2), peak at x=8, bimodal single-seed rewards. Seeds a study with 20 informative
   random observations (and, in the treated arm, 3 forced failures at x=8), runs TPE for 100 trials, and
   reports the fraction of proposals within |x-8|<1 across four 25-trial windows, at 3 base seeds, with and
   without the forced failures. Result: the forced failures lower the first-window near-peak fraction only
   slightly (about 0.52 vs 0.56) and the two arms match within 25-50 trials.
   Rerun: `conda run -n exploration python -u 02_recovery.py 2>&1 | tee 02_recovery_output.txt`

3. `03_top_quantile_target.py` — the central demonstration: single-seed TPE targets the top-quantile draw,
   not the mean. Two regions: A near x=3 is a consistent good learner (Normal(60,5), true mean 60); B near
   x=7 is the project's own bimodal regime (Normal(80,10) w.p. 0.5 else Normal(0,3), true mean 40, lower).
   150 single-seed trials vs 15 trials of 10-seed means (same 150 seed-runs), 3 base seeds plus a 12-seed
   tally. Result: single-seed concentrates on the lower-mean jackpot region B in the majority of seeds
   (8/12) with a best value inflated to ~100; 10-seed means shift the majority to the true-best region A
   (7/12) with a best value ~60. The script also runs the task's original numbers (A Normal(30,2); B 80
   w.p. 0.15 else Normal(15,3)) as a contrast that does NOT flip (single-seed stays on A, 12/12), because
   A's steady 30s fill the top decile and B's 15%-rare 80s are too few to take over the good group.
   Rerun: `conda run -n exploration python -u 03_top_quantile_target.py 2>&1 | tee 03_top_quantile_target_output.txt`

4. `04_budget_matched_fix.py` — spend 300 seed-runs three ways on the single-peak landscape: (a) 300 single
   -seed trials, (b) 100 trials of 3-seed means, (c) 30 trials of 10-seed means; 3 base seeds. Reports the
   true mean at the best-by-value x, the inflation (best value minus that true mean), and a belief measure
   (mean true reward over TPE's next 25 proposals). Result: inflation falls monotonically (about +63, +46,
   +17) and the true mean of the picked x rises (about 41, 45, 47); the belief measure is reported with the
   caveat that 300 trials concentrate TPE more than 30 trials, so it partly reflects trial count.
   Rerun: `conda run -n exploration python -u 04_budget_matched_fix.py 2>&1 | tee 04_budget_matched_fix_output.txt`

5. `05_source_facts.py` — confirms two source facts. (a) Prints the installed source of `TPESampler._sample`,
   `_split_complete_trials`, and `_split_complete_trials_single_objective`, and shows two trials at the SAME
   x (values 90 and 10) landing in different split groups: nothing averages duplicate or nearby params or
   models noise; each trial is placed by its own value. (b) Builds 12 draws at x=8 (4 in the good group, 8
   in the rest) and 12 at x=2 (all good), and compares the acquisition at x=8 (4/12 good, acq -0.95) with
   x=2 (12/12 good, acq +3.77): the acquisition at a repeated x reflects how many of that x's draws reached
   the top quantile.
   Rerun: `conda run -n exploration python -u 05_source_facts.py 2>&1 | tee 05_source_facts_output.txt`

## Self-contained snippets

`snippets/` holds standalone versions of each demonstration (own imports, inline toy data, no imports from
`common.py`), each with its captured output as `<snippet>_output.txt`. Their numbers differ slightly from
the full scripts because a snippet uses its own seed set and keyed-RNG names, but each conclusion holds.

- `snippet_01_acquisition_dent.py` — argmax off the peak vs number of bad draws at x=8.
- `snippet_02_recovery.py` — near-peak proposal fraction per window, with and without forced early failures.
- `snippet_03_top_quantile.py` — 8-seed tally: single-seed favours region B, 10-seed means favour region A.
- `snippet_04_budget_matched.py` — inflation and picked-x true mean for the three budget-matched designs.
- `snippet_05_repeated_x.py` — acquisition at a repeated x reflects its good/rest draw split.

Rerun any snippet with, e.g.:
`conda run -n exploration python -u snippets/snippet_03_top_quantile.py 2>&1 | tee snippets/snippet_03_top_quantile_output.txt`

## Source files read (installed package)

- `optuna/samplers/_tpe/sampler.py` (`_sample`, `_split_trials`, `_split_complete_trials`,
  `_split_complete_trials_single_objective`, `default_gamma`, `_compute_acquisition_func`).
- `optuna/samplers/_tpe/parzen_estimator.py` (Parzen estimators driven via `_build_parzen_estimator`).

## Caveats

- Scripts 01 and 05 drive private API (`study._get_trials`, `_split_trials`, `sampler._build_parzen_estimator`,
  `sampler._compute_acquisition_func`) to expose the mechanism; these are not stable public interfaces.
- Script 01 and 05 set the good-group size directly (via gamma sizing or an explicit `n_below`) to construct a
  specific split; the point is the acquisition given a split, not the default split size.
- The point-3 landscape is deliberately tuned so the flip is observable: the task's literal numbers do not
  flip (verified in the contrast run). The failure mode needs a lower-mean region whose single draws reach
  the top quantile MORE often than a higher-mean region's — that requires a variance difference between
  regions, not just a mean difference.
