# 2026-08-16-21-00_pm-jax-run11-tables

The table generator of train run 1.1 (`runs/2026-08-16-20-39_jax-ppo_pointmaze-large-nonoise_...`).

`make_all.py` writes one table into `platform_development_document.tex`:

| marker name | what it is |
|---|---|
| `pm-jax-run11-results` | one row per algorithm arm at its best configuration, ranked by whole-run reward |

Everything between `% >>> AUTO-GENERATED TABLE START: pm-jax-run11-results` and
`% <<< AUTO-GENERATED TABLE END: pm-jax-run11-results` is replaced wholesale. The caption, the
label and the `\resizebox` stay outside the markers and are hand-edited. A missing or duplicated
marker is a hard failure, not a silent no-op.

Every number comes from `runs/<this run>/code/aggregate.py`, imported: `cell_table` scores each
(arm, learning rate, intrinsic weight) cell over the run's phase-blocked episode windows and
`best_per_arm` picks each arm's winner. The generator itself loads no shard, scores no copy and
ranks nothing, so the writeup and the 20-minute status table cannot disagree.

The training-curve figure is not this folder's: it is drawn by
`runs/<this run>/analysis/code/make_reward_curves.py`, which imports the same two functions so that
the curve shown is the configuration the table names.

## Running it

```bash
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python make_all.py
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
latexmk -cd -pdf -interaction=nonstopmode \
  /p/rlprojects/RND/10_jax_exploration_platform/development_document/platform_development_document.tex
grep -nE 'Overfull \\hbox' ../../platform_development_document.log
```

Then rasterize the table's page and look at it, per the development document's README.
