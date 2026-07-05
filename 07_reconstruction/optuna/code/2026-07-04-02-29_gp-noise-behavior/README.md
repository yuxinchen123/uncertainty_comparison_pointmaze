# GPSampler noise behavior under high across-seed variance

Date: 2026-07-04. Environment: `conda run -n exploration python` (Python 3.11), optuna 4.9.0,
numpy 1.26.4, scipy 1.16.0, torch 2.10.0.

Question being answered (for an RL researcher): results have high variance across seeds; how does
Optuna's Gaussian-process sampler (`GPSampler`) handle a configuration that draws a very bad seed
first but actually has a high mean? Everything here is measured empirically on optuna 4.9.0, plus
exact source quotes from the installed package where noted.

Project noise regime the toys are calibrated to: per-seed reward is strongly bimodal. A configuration
with mean `m` realizes each seed as: success with probability `m/80` giving `Normal(80, 10)`, else
`Normal(0, 3)`. Per-seed standard deviation is ~35 at mid-range means. One seed = one 15-hour run.

## What each script verifies and how to rerun

All commands are run from this folder. Each writes its own `<script>_output.txt`.

- `01_noise_model_default_and_fit.py` — Point 1. Confirms `GPSampler`'s default
  `deterministic_objective=False`, the noise floor `1e-6`, the `Gamma(1.1, 30)` noise prior, and that
  fitting optuna's own GP (`optuna._gp.gp.fit_kernel_params`) on bimodal data gives a fitted noise
  variance far above the floor by default, versus exactly the floor when `deterministic_objective=True`.
  Rerun: `conda run -n exploration python -u 01_noise_model_default_and_fit.py 2>&1 | tee 01_noise_model_default_and_fit_output.txt`

- `02_posterior_averages_repeats.py` — Point 2. Stacks k=1,3,10 bimodal draws (true mean 30) at one
  location plus 3 anchors, reads the GP posterior mean at that location, and compares it to the running
  sample mean and the last draw. Includes the `deterministic_objective=True` interpolation contrast on
  distinct nearby points.
  Rerun: `conda run -n exploration python -u 02_posterior_averages_repeats.py 2>&1 | tee 02_posterior_averages_repeats_output.txt`

- `03_unlucky_early_draws_gp.py` — Point 3. Landscape `m(x)=50*exp(-(x-8)^2/2)` on `[0,10]`, pre-loaded
  with 20 informative points plus 3 forced ~0 draws at the optimum x=8, then 60 honest single-seed
  `GPSampler` trials over 3 base seeds. Reports near-optimum (`|x-8|<1`) proposals per 20-trial window,
  and the posterior mean at x=8 with fitted vs pinned noise.
  Rerun: `conda run -n exploration python -u 03_unlucky_early_draws_gp.py 2>&1 | tee 03_unlucky_early_draws_gp_output.txt`

- `04_homoscedastic_two_region.py` — Point 4. Two-region landscape (A: `Normal(30,2)`, true mean 30;
  B: `80` w.p. 0.15 else `Normal(15,3)`, true mean 24.75). 150 single-seed `GPSampler` trials over 3
  seeds. Reports last-50 allocation A vs B, best-posterior-mean location, per-region posterior mean, and
  the single fitted (shared) noise variance.
  Rerun: `conda run -n exploration python -u 04_homoscedastic_two_region.py 2>&1 | tee 04_homoscedastic_two_region_output.txt`

- `05_practical_facts.py` — Point 5. (a) per-ask wall-clock cost at 150/200/300 completed trials;
  (b) re-check that running trials get a best-value constant-liar imputation; (c) failure-mode probes
  with bimodal / degenerate values (extreme bimodal, all-failures, exactly-constant, plus a full-sampler
  constant-objective run).
  Rerun: `conda run -n exploration python -u 05_practical_facts.py 2>&1 | tee 05_practical_facts_output.txt`

## Tutorial snippets (trimmed, self-contained; output is the snippet's own observed output)

- `snippet_noise_fit.py` — Point 1: default is `deterministic_objective=False`; fitted noise_var vs floor.
- `snippet_posterior_averages.py` — Point 2: posterior mean tracks the sample mean of repeated draws.
- `snippet_unlucky_optimum.py` — Point 3: fitted noise keeps the optimum's estimate high despite 3 zeros.
- `snippet_two_region.py` — Point 4: the head-to-head; the GP averages each region and prefers the
  higher-mean region A.
- `snippet_constant_liar.py` — Point 5(b): running trials are imputed with the best completed value.

Rerun any snippet: `conda run -n exploration python -u <snippet>.py 2>&1 | tee <snippet>_output.txt`

## Source references (installed optuna 4.9.0)

- Noise variance is a fitted kernel parameter: `optuna/_gp/gp.py`, `GPRegressor._fit_kernel_params`,
  lines 320-324 and 348-352 — `noise_var = torch.exp(raw_params[n_params+1]) + minimum_noise` when not
  deterministic, else `torch.tensor(minimum_noise)`.
- Noise prior and floor: `optuna/_gp/prior.py` — `DEFAULT_MINIMUM_NOISE_VAR = 1e-6` (line 16) and
  `gamma_log_prior(gpr.noise_var, 1.1, 30)` (line 32).
- Default `deterministic_objective=False`: `optuna/samplers/_gp/sampler.py`, `GPSampler.__init__`
  line 185; `self._minimum_noise = prior.DEFAULT_MINIMUM_NOISE_VAR` line 203.
- Posterior mean formula `mean = cov_fx_fX @ inv(cov_fX_fX + noise_var*I) @ y`:
  `optuna/_gp/gp.py`, `GPRegressor.posterior` docstring lines 223-225.
- Best-value constant liar for running trials: `optuna/_gp/acqf.py`, `LogEI.__init__` lines 134-149 —
  `constant_liar_value = self._gpr._y_train.max()`.
