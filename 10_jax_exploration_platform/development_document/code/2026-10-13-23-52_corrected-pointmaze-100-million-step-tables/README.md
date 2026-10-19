# Corrected PointMaze report generator

Generated table: `tab:pm-jax-corrected-results` (Table 1.6), using its named marker block in the main document. Scores come from the run's `summary.json`, published by `scripts/aggregate_selected_run.py`; the shared ranking function selects each algorithm's largest whole-run reward among the 11 retrained settings. Captions and prose remain outside the markers.

The same call creates Figure 1.4.1 from the published corrected summary. It shows exactly the six settings from historical Figure 1.3.1, keeping the same colors and learning-rate line patterns. Figures 1.2, 1.3.1 and 1.4.1 show extrinsic reward only. The current document uses the reward-component generator below to add the two raw combined-reward figures. The generator requires all 11 trained settings to finish and validates finite measurements and full step budgets.

Run on an allocated CPU node:

```bash
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python /p/rlprojects/RND/10_jax_exploration_platform/development_document/code/2026-10-13-23-52_corrected-pointmaze-100-million-step-tables/make_all.py
```

Then compile `RND_JAX_exploration_development_document.tex` with `latexmk -pdf` in its directory and inspect the rendered table and figure pages.

To regenerate all five current reward plots, run `scripts/write_reward_component_figures.py --analysis-dir analysis/2026-10-19-12-50_extrinsic-and-raw-intrinsic-reward-curves` on a compute host. Its paired total plots add raw intrinsic episode rewards without normalization or weights; these are not the normalized quantities used for training.
