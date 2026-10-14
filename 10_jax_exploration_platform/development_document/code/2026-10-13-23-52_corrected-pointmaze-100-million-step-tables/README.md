# Corrected PointMaze report generator

Generated table: `tab:pm-jax-corrected-results` (Table 1.6), using its named marker block in the main document. Scores come from the run's `summary.json`, published by `scripts/aggregate_selected_run.py`; the shared ranking function selects each algorithm's largest whole-run reward among the 11 retrained settings. Captions and prose remain outside the markers.

The same call overlays corrected and uncorrected curves in Figures 1.2 and 1.3. Corrected curves come from the published summary; historical curves come from `historical_curves.json`, extracted without changing their original windows or seed standard errors by `scripts/extract_historical_comparison_curves.py`. Matching settings keep their color and line pattern; lighter thin lines show historical training, and darker thick lines show corrected training. The old 1000-million-step and corrected 100-million-step budgets use different learning-rate annealing schedules. The historical no-bonus intrinsic-advantage weight is one; the corrected weight is zero. It requires all 11 settings to finish and validates finite measurements and full step budgets.

Run on an allocated CPU node:

```bash
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python /p/rlprojects/RND/10_jax_exploration_platform/development_document/code/2026-10-13-23-52_corrected-pointmaze-100-million-step-tables/make_all.py
```

Then compile `RND_JAX_exploration_development_document.tex` with `latexmk -pdf` in its directory and inspect the rendered table and figure pages.
