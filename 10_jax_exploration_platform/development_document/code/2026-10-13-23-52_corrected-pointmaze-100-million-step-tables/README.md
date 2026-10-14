# Corrected PointMaze report generator

Generated table: `tab:pm-jax-corrected-results` (Table 1.6), using its named marker block in the main document. Scores come from the run's `summary.json`, published by `scripts/aggregate_selected_run.py`; the shared ranking function selects each algorithm's largest whole-run reward among the 11 retrained settings. Captions and prose remain outside the markers.

The same call creates Figures 1.4 and 1.5 from the aggregate curves. It requires all 11 settings to finish and validates finite measurements and full step budgets.

Run on an allocated CPU node:

```bash
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python /p/rlprojects/RND/10_jax_exploration_platform/development_document/code/2026-10-13-23-52_corrected-pointmaze-100-million-step-tables/make_all.py
```

Then compile `RND_JAX_exploration_development_document.tex` with `latexmk -pdf` in its directory and inspect the rendered table and figure pages.
