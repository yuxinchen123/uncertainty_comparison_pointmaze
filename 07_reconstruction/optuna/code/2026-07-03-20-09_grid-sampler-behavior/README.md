# Optuna GridSampler behavior verification

Date: 2026-07-03
Optuna version observed: 4.9.0 (conda env `exploration`, Python 3.11)

These scripts empirically verify how Optuna's basic workflow and `GridSampler` behave, for a
tutorial written against the RND maze sweep (feature normalization x ridge lambda x bonus clip x
beta x seed). Every script prints text only (no plots). Each script has its saved output in
`<script>_output.txt`.

Each Optuna sampler is given an explicit `seed=` for reproducibility. Objectives are trivial and
instant, so every script finishes in a few seconds.

## Scripts and rerun commands

Run each from this folder.

1. Minimal workflow (objective, create_study, optimize, read best_trial / best_params /
   trials_dataframe columns):
   `conda run -n exploration python 01_minimal_workflow.py 2>&1 | tee 01_minimal_workflow_output.txt`

2. GridSampler basics on a 36-combination grid: (a) n_trials=None self-stop, (b) n_trials larger
   than the grid, (c) full single-visit coverage, (d) a second optimize call after exhaustion:
   `conda run -n exploration python 02_gridsampler_basic.py 2>&1 | tee 02_gridsampler_basic_output.txt`

3. GridSampler with the seed as an extra grid axis (360 combinations), full coverage:
   `conda run -n exploration python 03_gridsampler_seed_axis.py 2>&1 | tee 03_gridsampler_seed_axis_output.txt`

4. GridSampler visit order: deterministic across fresh studies at the same seed; shuffled vs
   Cartesian-product; different seed gives different order:
   `conda run -n exploration python 04_gridsampler_order.py 2>&1 | tee 04_gridsampler_order_output.txt`

5. study.enqueue_trial: the first trials use the enqueued params, in order, on a TPE study:
   `conda run -n exploration python 05_enqueue_trial.py 2>&1 | tee 05_enqueue_trial_output.txt`

6. GridSampler mismatches: (6a) objective suggests a param not in the grid; (6a-variant) grid has an
   extra key the objective never suggests; (6b) suggest_float low/high vs a grid value inside and
   outside that range:
   `conda run -n exploration python 06_gridsampler_mismatch.py 2>&1 | tee 06_gridsampler_mismatch_output.txt`

7. Trial states: produce COMPLETE / PRUNED / FAIL trials and count them from `trial.state`:
   `conda run -n exploration python 07_trial_states.py 2>&1 | tee 07_trial_states_output.txt`

## Note on stderr in the saved outputs

Optuna logs trial failures at WARNING level to stderr, and `GridSampler` prints a UserWarning when it
re-evaluates an exhausted grid or when a grid value is outside a suggest_float range. Because stdout is
buffered, these stderr lines appear at the TOP of some `_output.txt` files (before the script's own
stdout), not inline. They are expected and are part of what the test observes.
