# Train run 2 — cancellation & resume record

**Cancelled: 2026-06-25 ~16:40 EDT** (by user request, to finalize the writeup on the runs finished so
far). The fleet (64-job local work queue) was `scancel`-ed via the tracked IDs in
`slurm/submitted_jobids.txt`; the 20-min step-distribution loop and the milestone monitor cron were
stopped. No new runs are submitted.

## Status at cancellation

| Algorithm | β★ | finished | incomplete | total |
|---|---|---|---|---|
| `gt_position_velocity` | 1 | 97 | 103 | 200 |
| `rnd_elliptical` | 0.01 | 116 | 84 | 200 |
| `rnd_state` | 100 | 99 | 101 | 200 |
| **all** | | **312** | **288** | **600** |

- **finished** = a valid per-run JSON exists in `data/local/` (one file per completed run). These are
  the runs the writeup (Train run 2 section of `development_document/main.tex`) is computed from; the
  reward / distance tables state `n` per algorithm.
- **incomplete** = claimed-but-interrupted (their slurm job was cancelled mid-training; no JSON written)
  or never-claimed. They produced no data.

## Exact lists (one config per line)
- `resume_completed_configs.txt` — the **312** finished configs (queue/done names). Do NOT re-run these.
- `resume_incomplete_configs.txt` — the **288** configs still owed (algorithm + seed are in each name,
  e.g. `00081_gt_position_velocity_seed81.json`).

These snapshot lists use the **pre-resume** queue-name format (`NNNNN_<algo>_seed<seed>.json`,
algo-outermost idx). After `resume_setup.py` regenerates the queue under the revised convention, pending
configs are renamed `NNN_of_600_<algo>_seed<seed>.json` (seed-outermost id); resume keys on
`(algorithm, a_seed)`, so the old snapshot names still map correctly and nothing is double-run.

The queue folders are the durable source of truth: `queue/done/` (312 completed), `queue/running/` (288
interrupted at cancellation), `queue/pending/` (0), `queue/failed/` (0).

## How to resume (complete the remaining 288)
Run the resume helper, then relaunch. The helper rebuilds the queue under the **revised** convention
(per-run integer id, seed-outermost order) and pre-marks the 312 already-finished runs as done — keyed on
`(algorithm, a_seed)`, so it works even though those JSONs use the old descriptive names — leaving only
the 288 owed in `pending/`:

```bash
cd /p/rlprojects/RND/07_reconstruction
RUN=train_runs/2026-06-24-21-24_run_2_after_reorganization
RUN_DIR="$PWD/$RUN" conda run -n exploration python "$RUN/slurm/resume_setup.py"
conda run -n exploration bash "$RUN/slurm/launch_queue.sh"   # exploration env active in the launching shell
```

Notes:
- The 288 resumed runs write the **revised** convention (id-based filename `NNN_of_600.json`, plus
  `run_id`/`run_total` and the `train_history` group); the 312 already-finished runs keep their old
  descriptive filenames. `analysis/code/common.py` `load_records` reads every `data/local/*.json`
  regardless of filename, so a mixed-convention `data/local/` still analyzes correctly.
- Do **not** simply `mv queue/running/* queue/pending/` and relaunch: those interrupted configs predate
  the `run_id` field, and the revised `worker.py` requires `run_id` — `resume_setup.py` is the supported
  path (it regenerates configs that carry `run_id`).
- For a fully convention-consistent run instead, `build_queue.py --reset` and relaunch all 600 — this
  re-runs everything (deterministic seeds reproduce the same eval results and add `train_history`).
- Cancellation safety: only IDs in `slurm/submitted_jobids.txt` were cancelled; never `scancel -u`.
