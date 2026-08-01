# Point maze + ant maze train run 1 — reward-curve figure (final data)

The figure `plots/reward_curves_8env.pdf` / `.png` is the 4x2 reward plot defined in the run
plan (`~/.claude/plans/2026-07-22-17-40_antmaze-section-4-and-hyperparameter-table-skill.md`,
step 8 item 3): one panel per environment, PointMaze column left and AntMaze column right,
maze sizes UMaze / Open / Medium / Large top to bottom — the same environment order as the
writeup's per-environment result tables.

## What each panel shows

- One curve per algorithm, for ONLY that algorithm's **best configuration** in the panel's
  environment. Best = highest mean whole-run per-episode extrinsic return over the
  configuration's completed runs (`prune_controller.score_of_record`) — the same selection rule
  as the writeup tables, so figure and tables name the same winners.
- Each curve point is the mean over seeds of $\bar R_{100}(t)$, where $\bar R_{100}(t)$ is one
  run's mean per-episode extrinsic return over the 100 training episodes before step $t$
  (the record's `train_history` `train/mean_extrinsic_reward` column, logged every 50k steps).
  The shaded band is $\pm$ one standard error over seeds ($s/\sqrt{n}$; the legend's $n$ is the
  configuration's completed-seed count).
- Colors are fixed per algorithm across every panel (Okabe-Ito colorblind-safe palette):
  gray `no_exploration`, blue `rnd_next_state`, green `gt_position_velocity`, orange
  `gt_position_maze_cell`, pink `gt_position_1m`.

## Data

- Sweep `2026-07-23-02-05_pm-am-run1`; the sweep was ended at the 2026-07-31 snapshot (see
  `../infra_history.md`). The figure was built 2026-08-01 from **14154 completed records**
  (`completed=true` only) — slightly more than the writeup tables' 2026-07-31 snapshot (13782),
  because collaborator workers finished their in-flight runs after the owner's jobs stopped.
- In cells where every configuration is flat at the episode-limit return (the three bonus cells
  of AntMaze_Large, the two ground-truth cells of AntMaze_Medium), the "best configuration" is
  not meaningful — the racing score cannot separate identical means; the AntMaze_Large panel is
  annotated accordingly.

## Regenerate

    /p/rlprojects/RND/.venvs/exploration/bin/python code/make_reward_curves.py

The first run scans every per-run JSON (~minutes) and writes `data/curves_cache.json`; later
runs re-plot from the cache instantly. Delete the cache to pick up newly completed records.
