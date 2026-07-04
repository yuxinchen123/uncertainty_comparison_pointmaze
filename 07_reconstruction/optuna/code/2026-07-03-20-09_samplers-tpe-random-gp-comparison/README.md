# Optuna sampler behavior checks (for the RL tutorial)

Date: 2026-07-03
Optuna version observed: 4.9.0 (numpy 1.26.4, scipy 1.16.0, torch 2.10.0+cu128).
Everything ran locally on the login node; no Slurm, no wandb, no plots.

All scripts share one toy reward landscape (`landscape.py`) that mirrors the project's SAC
exploration sweep: two log-scale knobs (ridge lambda in [1e-6, 1e-2], beta in [1e-3, 1e-1]),
a single smooth peak of height ~50 at (ridge=1e-6, beta=1e-2) about one decade wide in each
axis, plus additive noise of standard deviation ~10. Noise is drawn with the keyed-substream
pattern so a given point + noise seed always returns the same value.

## What each script verifies and how to rerun it

Run every command from this folder.

- `01_suggest_types.py` — the four value-suggestion calls: `suggest_categorical`,
  `suggest_float` (linear), `suggest_float(log=True)`, `suggest_int`; prints returned values
  over 6 trials.
  `conda run -n exploration python 01_suggest_types.py 2>&1 | tee 01_suggest_types_output.txt`

- `02_landscape_check.py` — sanity-checks the toy landscape (peak height, decay, noise SD,
  repeatability).
  `conda run -n exploration python 02_landscape_check.py 2>&1 | tee 02_landscape_check_output.txt`

- `03_sampler_comparison.py` — the main comparison at 60 trials each: RandomSampler(seed=0),
  TPESampler(seed=0, n_startup_trials=10), GPSampler(seed=0), GridSampler over a 4x3 log grid
  with a rep dimension of 5 (=60). Reports best value, best params, and near-peak allocation
  (trials within half a decade of the peak in both axes), across three noise seeds.
  `conda run -n exploration python -W ignore::UserWarning 03_sampler_comparison.py 2>&1 | tee 03_sampler_comparison_output.txt`

- `03b_gridsampler_stops_early.py` — shows that a plain 12-cell GridSampler runs only 12
  trials when 60 are requested (it does not pad by repeating cells).
  `conda run -n exploration python 03b_gridsampler_stops_early.py 2>&1 | tee 03b_gridsampler_stops_early_output.txt`

- `04_tpe_mixed_space.py` — TPE on a mixed space (one categorical + one log float); prints
  category counts in the first 15 vs last 15 of 60 trials.
  `conda run -n exploration python 04_tpe_mixed_space.py 2>&1 | tee 04_tpe_mixed_space_output.txt`

- `05_tpe_determinism.py` — two fresh TPE(seed=0) studies produce an identical first-10
  suggestion sequence; a different seed does not.
  `conda run -n exploration python 05_tpe_determinism.py 2>&1 | tee 05_tpe_determinism_output.txt`

- `06_gp_vs_tpe_runtime.py` — wall time of a 60-trial GP study vs a 60-trial TPE study on the
  same objective.
  `conda run -n exploration python -W ignore::UserWarning 06_gp_vs_tpe_runtime.py 2>&1 | tee 06_gp_vs_tpe_runtime_output.txt`

- `07_cmaes_sampler.py` — tries to construct and run CmaEsSampler; reports the exact error
  when the optional `cmaes` package is missing (does not install it).
  `conda run -n exploration python 07_cmaes_sampler.py 2>&1 | tee 07_cmaes_sampler_output.txt`

- `08_n_startup_trials.py` — confirms TPE's first `n_startup_trials` are random (with
  n_startup=60 of 60 the suggestions equal RandomSampler exactly) and that shrinking the
  startup phase lets TPE concentrate near the peak.
  `conda run -n exploration python 08_n_startup_trials.py 2>&1 | tee 08_n_startup_trials_output.txt`

## Shared module

- `landscape.py` — the toy reward landscape (`clean_reward`, `noisy_reward`, `near_peak`) and
  the `substream` keyed-seed helper. Imported by the numbered scripts; not run directly.
