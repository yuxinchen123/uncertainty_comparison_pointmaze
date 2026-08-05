# 8.1.2 interim-results table generator (train run 1.2)

## Generated tables

- `tab:pm-am-trainrun12-status`  — markers in RND_development_document.tex; counts come from the
  run folder's queue + decision log via its `20_mins_monitoring/status_table.py`.
- `tab:pm-am-trainrun12-interim` — markers in RND_development_document.tex; scores come from the
  completed per-run JSONs via the run folder's `20_mins_monitoring/env_metrics_tables.py` (the
  same modules the 20-minute monitoring report uses, so the writeup can never drift from it).

To regenerate after new data lands (then rebuild the PDF):

    /p/rlprojects/RND/.venvs/exploration/bin/python make_all.py
    latexmk -cd -pdf ../../RND_development_document.tex

Run folder: `train_runs/2026-08-01-01-44_run_8_1_2_...`, sweep `2026-08-01-02-03_run812`.
