# TPE internals and mini-TPE verification (optuna 4.9.0)

Date: 2026-07-03
Environment: `conda run -n exploration python` (Python 3.11, optuna 4.9.0, numpy 1.26.4, scipy 1.16.0).

These scripts verify how optuna's `TPESampler` works, by reading the installed source and by
running the sampler and small numpy re-implementations. Every printed number is the real output of
the script next to it (`<script>_output.txt`).

## What each script verifies and how to rerun it

- `01_tpe_defaults_and_internals.py` — TPESampler constructor defaults; the `default_gamma` split
  size; the `default_weights` down-weighting; the numeric mixture (one truncated-normal per
  observation plus one prior component with mean at the range center and sigma equal to the full
  range); the magic-clip lower bound on sigma; categorical weights with additive prior smoothing.
  Rerun: `conda run -n exploration python -u 01_tpe_defaults_and_internals.py 2>&1 | tee 01_tpe_defaults_and_internals_output.txt`

- `02_candidate_mechanics.py` — drives the real sampler's internal methods to confirm that each ask
  draws `n_ei_candidates` (24) samples from the below/good density l(x), that the acquisition equals
  `log l(x) - log g(x)` exactly, and that the returned point is the candidate at the argmax.
  Rerun: `conda run -n exploration python -u 02_candidate_mechanics.py 2>&1 | tee 02_candidate_mechanics_output.txt`

- `03_mini_tpe.py` — a from-scratch numpy mini-TPE (10 random startup, top-25% good/rest split,
  fixed-bandwidth 1-D Gaussian KDE per parameter per group, 24 candidates drawn from the good
  density, argmax of summed log l - log g), run on a noisy 2-D bump landscape for 60 trials, and
  compared against pure random and the real `TPESampler` at three base seeds.
  Rerun: `conda run -n exploration python -u 03_mini_tpe.py 2>&1 | tee 03_mini_tpe_output.txt`

- `04_ei_theorem.py` — numerical check of the Bergstra et al. (2011) result: on a dense x-grid,
  compute l(x)/g(x) and compute the expected improvement EI(x) by integrating numerically over y
  under the two-density model, then report the Spearman rank correlation between them.
  Rerun: `conda run -n exploration python -u 04_ei_theorem.py 2>&1 | tee 04_ei_theorem_output.txt`

- `05_constant_liar.py` — seed a study with 30 completed trials, ask a batch of 8 without reporting
  results, once with `TPESampler(seed=0)` and once with `TPESampler(seed=0, constant_liar=True)`,
  and report the batch's mean pairwise distance and point locations both ways.
  Rerun: `conda run -n exploration python -u 05_constant_liar.py 2>&1 | tee 05_constant_liar_output.txt`

- `06_source_quotes.py` — prints verbatim `inspect.getsource` of the functions that back the claims
  (`default_gamma`, `default_weights`, `_sample`, `_compute_acquisition_func`, `_compare`, the
  numeric and categorical Parzen distribution builders).
  Rerun: `conda run -n exploration python -u 06_source_quotes.py 2>&1 | tee 06_source_quotes_output.txt`

## Source files read (installed package)

- `optuna/samplers/_tpe/sampler.py`
- `optuna/samplers/_tpe/parzen_estimator.py`
