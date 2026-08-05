# Analysis of this run

Nothing is analyzed here yet — the sweep was submitted 2026-08-05 and the first 1,000,000-step runs
take 17–20 hours each.

## Where the live numbers come from meanwhile

`../20_mins_monitoring/monitoring_report.py --sweep_id <id>` prints, and with `--snapshot` also
writes to `../20_mins_monitoring/outputs/`, the two tables this run will be reported on:

1. **Status by environment** — configurations, verdicts (undecided, truncated, survivor) and
   completed runs, one row per environment.
2. **Scores by environment** — every bonus weight ranked by its mean score under that environment's
   own rule, with the standard error, the completed-seed count, the one-sided 99% upper bound the
   truncation rule uses, and the verdict. Each block restates its frozen bar and how many of its 15
   configurations sit at or above it.

Those functions are the single source of the numbers: the writeup's columns are generated from them,
so the document and the monitoring report cannot disagree. The score rule per environment lives in
`../slurm/score_rules.py` (PointMaze: the final training-episode reward; AntMaze: the whole-run mean
per-episode return), and the bars in `../slurm/FROZEN_BARS.json`.

## What the finished run is meant to answer

One number per environment: does the train-run-5 "original-small" RND stack do better or worse at
predictor learning rate 1e-3 than at the paper's 1e-4? The comparison is the frozen bar — the mean
of the best 1e-4 configuration on that same environment — so a surviving bonus weight is one whose
99% upper confidence bound never fell below what 1e-4 already achieved.

## Folder convention

`analysis.md` and `plots/` are committed when they exist; `data/` and `code/` are gitignored per the
project's analysis-folder rule.
