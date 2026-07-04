# Optuna GPSampler internals — verification scripts

Date: 2026-07-03. Verified against the installed **optuna 4.9.0** (numpy 1.26.4, torch 2.10.0+cu128),
Python 3.11, conda env `exploration`. Everything ran locally on the login node; no Slurm, no plots.

These scripts back a tutorial section on how Optuna's `GPSampler` works: the Matern 5/2 kernel, input
and objective normalization, kernel-hyperparameter fitting, the single-objective acquisition function
(logEI), and how the acquisition is optimized. A from-scratch numpy Gaussian process and Expected
Improvement reproduce the mechanism, and Optuna's `GPSampler` is shown to ask where that landscape is
highest.

## Scripts and how to rerun

Each script writes its real printed output to `<name>_output.txt` next to it.

- `01_read_gp_source.py` — quotes the exact installed source and docstrings for the kernel, input/
  objective normalization, hyperparameter fitting, logEI, and the acquisition optimizer; numerically
  confirms `optuna._gp.gp.Matern52Kernel` equals the closed-form Matern 5/2.
  - `conda run -n exploration python -u 01_read_gp_source.py 2>&1 | tee 01_read_gp_source_output.txt`
- `02_gp_ei_from_scratch.py` — a numpy-only Gaussian process posterior + closed-form Expected
  Improvement on a 1-D toy (reward vs `x = log10(ridge)`), prints the `x, mu, sigma, z, EI` grid,
  states where `argmax EI` falls, and decomposes EI into its exploit and explore terms at one
  sigma-driven point and one mu-driven point.
  - `conda run -n exploration python -u 02_gp_ei_from_scratch.py 2>&1 | tee 02_gp_ei_from_scratch_output.txt`
- `03_optuna_gpsampler_compare.py` — imports the same six observations into a `GPSampler` study and
  asks where to sample; compares the asked `x` against the from-scratch high-EI band. Test A: ten
  independent single asks (fresh study per seed). Test B: ten sequential asks without telling results
  (pending trials spread the asks via the constant-liar strategy).
  - `conda run -n exploration python -u 03_optuna_gpsampler_compare.py 2>&1 | tee 03_optuna_gpsampler_compare_output.txt`
- `04_ask_time_scaling.py` — confirms the torch requirement and measures GPSampler per-ask wall time
  at 20 vs 200 completed trials (one untimed warmup ask first).
  - `conda run -n exploration python -u 04_ask_time_scaling.py 2>&1 | tee 04_ask_time_scaling_output.txt`

## Trimmed, self-contained snippets (pasted into the tutorial)

Each `snippet_*.py` is a shorter, fully self-contained version of the matching script.

- `snippet_kernel_check.py` — installed Matern 5/2 kernel vs closed form.
- `snippet_gp_ei.py` — from-scratch GP posterior + EI (tasks 2 and 3).
- `snippet_gpsampler_ask.py` — GPSampler ask vs the from-scratch EI region (task 4).
- `snippet_ask_time.py` — per-ask time at 20 vs 200 completed trials (task 5).

Rerun any snippet with, e.g.:
`conda run -n exploration python -u snippet_gp_ei.py 2>&1 | tee snippet_gp_ei_output.txt`

## Note on the toy objective

The task's example curve `50*exp(-((x+6)/1.2)^2)` peaks at the left edge `x=-6`. To make the
explore/exploit split visible "between observations", the scripts center the bump in the interior at
`x=-4` (`50*exp(-((x+4)/1.2)^2)`) and leave the region between `x=-5` and `x=-3` unsampled. Kernel
hyperparameters in the from-scratch GP are fixed (signal variance `s2=400`, lengthscale `L=0.8`,
noise variance `1.0`) and stated in the output; the point is the mechanism, not the fitting. Optuna
fits its own hyperparameters, so exact agreement with the from-scratch landscape is not expected.
