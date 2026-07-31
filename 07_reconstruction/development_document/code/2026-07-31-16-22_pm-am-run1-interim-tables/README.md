# Interim-results tables for subsection 8.1 (point maze + ant maze train run 1)

Generator for the three tables of the dark-brown interim update (2026-07-31, sweep still
running) in `RND_development_document.tex` subsection "Train run 1". Only the `\begin{tabular}...\end{tabular}`
blocks between the marker comments are regenerated; captions, labels, sizing and the surrounding
prose are hand-edited in `RND_development_document.tex` (generate-latex-table skill convention).

## Generated tables

- `tab:pm-am-run1-interim-status` — racing state per (environment, algorithm) cell: configuration
  counts (still racing / pruned / winner phase) and completed runs, from the sweep's queue
  directories and prune decision log.
- `tab:pm-am-run1-interim-pointmaze` — per-environment metrics blocks of the 4 PointMaze
  environments (no 1 m coverage column: identical to maze-cell coverage there, the PointMaze
  maze cell is 1 m).
- `tab:pm-am-run1-interim-antmaze` — per-environment metrics blocks of the 4 AntMaze
  environments (both coverage columns: 4 m cells vs 1 m squares).

Numbers come from the run's completed per-run JSON records
(`train_runs/2026-07-23-01-35_run_8_1_.../data/2026-07-23-02-05_pm-am-run1/local/`), counted and
scored by the run's own monitoring modules (`20_mins_monitoring/{status_table,
env_metrics_tables, monitoring_report}.py` + `slurm/{build_queue, prune_controller}.py`) — the
same source of truth as the 20-minute monitoring report, so the writeup tables cannot drift from
the monitoring tables. Marking (per metric column, per environment block: best bold, second best
underlined; higher is better except steps to goal; N never marked) follows
`monitoring_report.MARK_SPEC`.

To regenerate after new runs complete:

    /p/rlprojects/RND/.venvs/exploration/bin/python make_all.py
    latexmk -cd -pdf ../../RND_development_document.tex

(reads every completed per-run JSON of the sweep — takes a few minutes)
