# Optuna: best configuration PER method, not one overall winner

Date: 2026-07-03. Verified on optuna 4.9.0, numpy 1.26.4, scipy 1.16.0, Python 3.11
(conda env `exploration`).

These scripts verify how Optuna expresses the train-run-3.2.1 question: sweep THREE optimizer
methods (O1 Adam, O2 AdaGrad, O3 SGD-with-1/t schedule) that have partly different
hyperparameters (O3 alone has `eta0` and `t0`), and pick the best configuration PER method
(three winners), not one overall winner.

## Shared toy simulation

`toy.py` defines `reward(method, readout, log10_beta, eta0, t0, base_seed)` used by every
script. The noise-free mean is a Gaussian bump in `log10(beta)` per method:

- adam: peak 20 at beta=1e2, readout `l2` adds +3 (best adam cell mean = 23).
- adagrad: peak 27 at beta=1e1.
- sgd1t: peak 45 at beta=1e2, with a mild extra penalty away from eta0=1e-2 and t0=1e4.

Additive Gaussian noise, standard deviation 6, is drawn from a per-configuration keyed
generator (`substream`) so every call is reproducible. So sgd1t is clearly the best method,
adagrad second, adam third; this ordering is what drives the allocation experiment.

## Scripts and how to rerun each

All commands run from this folder. Each writes its real printed output to
`<script>_output.txt`.

- `01_conditional_search_space.py` — one study, `method` categorical, `eta0`/`t0` suggested
  only inside the sgd1t branch; shows the branch-only params are missing (NaN) for
  adam/adagrad rows in `trials_dataframe`.
  `conda run -n exploration python -u 01_conditional_search_space.py 2>&1 | tee 01_conditional_search_space_output.txt`
- `02_allocation_problem.py` — one shared 90-trial TPE study vs three 30-trial per-method
  studies, at 3 seeds; prints per-method trial counts, noise-free best reached, and the noisy
  best. This is the central evidence for splitting into one study per method.
  `conda run -n exploration python -u 02_allocation_problem.py 2>&1 | tee 02_allocation_problem_output.txt`
- `03_per_method_best_from_mixed.py` — recover the per-method winner from one mixed study by
  filtering COMPLETE trials on `params["method"]` and taking the per-method argmax.
  `conda run -n exploration python -u 03_per_method_best_from_mixed.py 2>&1 | tee 03_per_method_best_from_mixed_output.txt`
- `04_journal_many_studies.py` — three per-method studies in ONE JournalStorage file
  (current import paths `optuna.storages.journal.JournalFileBackend` / `JournalFileOpenLock`
  / `JournalStorage`); list names, reload by name, confirm independence.
  `conda run -n exploration python -u 04_journal_many_studies.py 2>&1 | tee 04_journal_many_studies_output.txt`
- `05_grid_per_method.py` — three GridSampler studies with different search-space dicts in
  one storage; confirms exact grid sizes 16 / 16 / 160.
  `conda run -n exploration python -u 05_grid_per_method.py 2>&1 | tee 05_grid_per_method_output.txt`
- `06_enqueue_reference_config.py` — `study.enqueue_trial` seeds the canonical Adam-mse cell
  into the adam study only; runs as trial 0 and never appears in the sgd1t study.
  `conda run -n exploration python -u 06_enqueue_reference_config.py 2>&1 | tee 06_enqueue_reference_config_output.txt`

Scripts 04, 05, 06 write a journal log under `journal_demo/`; each script deletes and
recreates its own file at the start so reruns are reproducible. `journal_demo/` is a
generated artifact, not source.

`snippets/` holds the six self-contained tutorial blocks (`s1`..`s6`) with each block's
saved output. Each block has its own inline toy landscape and no dependence on `toy.py` or
any sibling file, so it can be pasted into the tutorial and run on its own. The snippet toy
uses a simpler noise key than `toy.py`, so its noise realization differs from the numbered
scripts; that is why the snippet allocation study (s2) concentrates on a different method
than script 02 (see the caveat about noise-dependent concentration). Rerun one with, e.g.,
`conda run -n exploration python -u snippets/s2_allocation.py`.
