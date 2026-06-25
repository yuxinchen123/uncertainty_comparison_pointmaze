# Train run 2 (after reorganization) — experiment background

## Goal
Two things at once: (1) confirm the reorganized `rnd_exploration` package + `train.py` reproduce the Train-run-1
results, and (2) test whether a new logging mode — **use wandb only to fetch sweep parameters, then log metrics
locally** — reduces wall-clock vs full wandb logging (many concurrent processes logging to wandb add waiting time).

## Design
One W&B grid sweep (`config/sweep_run2.yaml`), `program: train.py`, entity `catresearch`, project
`rnd_07_reconstruction`. Grid axes (alphabetical → sweep nesting; `a_seed` outermost, `z_logging_mode` innermost
so the two modes alternate every run and accrue evenly):
- `a_seed`: 0–199 (200 seeds)
- `g_algo_beta`: the three best (algorithm, β) pairs from run 1, encoded as one param so a single grid sweep pins
  each algorithm to its β: `gt_position_velocity|1`, `rnd_elliptical|0.01`, `rnd_state|100`. `train.py` splits it
  back into `algorithm` + `beta` and records both to `wandb.config`.
- `z_logging_mode`: `wandb_full` (metrics logged to wandb) vs `wandb_param_only` (wandb.init only to receive the
  sweep params; metrics logged locally, no `wandb.log`).
Total = 200 × 3 × 2 = **1200 runs** (600 per logging mode). Fixed: `total_timesteps=1e6` (run 1 used 2e6),
`n_eval_episodes_final=100` (run 1 used 10000), `eval_freq=50000`, `n_eval_episodes=100`, `device=cpu`,
`discount=0.999`, `env_max_episode=400`, `goal_position=top_right`, PointMaze_Large-v3.

## Data layout (clear separation of the two modes)
Every run writes a per-run JSON via `train.py --local_log_dir` to
`data/<z_logging_mode>/<run_name>.json` — i.e. `data/wandb_full/` and `data/wandb_param_only/`. Each JSON holds
the config, the eval-reward history, the distance-to-GT history, and `runtime_seconds`. `wandb_full` runs also
log to the wandb cloud (incurring the overhead being measured). The analysis reads the local JSONs.

## Runtime measurement
`runtime_seconds` is measured in `train.py` from the start of `run()` (parameters already fetched from wandb —
the param-wait is **excluded**) to program end. The wandb-logging overhead during training is the difference
between the two modes.

## Analysis + reporting (code in `analysis/code/`, outputs in `analysis/plots/`)
Tables/plots POOL both modes (training is identical across modes): a per-algorithm reward table (β⋆, R̄, SE, n),
a final-reward bar, a reward-vs-step curve, and a distance-to-GT table — mirroring the Train-run-1 section. The
**timing bar plot** is the one artifact that separates the modes: two bars (wandb_full vs wandb_param_only), each
the mean `runtime_seconds` ± SE over the first 100 runs per algorithm per mode (300 runs/bar).

`main.tex` "Train run 2" subsection updates: at ≥50 finished/algorithm (pooled) → reward tables+plots; at ≥100
finished/algorithm/mode → add the timing bar; at ≥150 finished/algorithm/mode → refresh. Hyperparameters that
differ from run 1 are shown in blue.

## Reproduce
- Package/env: `conda run -n exploration`, editable-installed `rnd_exploration` (commit after the reorganization).
- Create the sweep: `wandb sweep config/sweep_run2.yaml` → `<sweep_id>`.
- Launch: from a shell with `exploration` active, `bash slurm/00_batch_slurm.sh catresearch/rnd_07_reconstruction/<sweep_id>`
  (reservation-aware; 10 nolim + 20 cpu + 10 gpu + reservation fill; per project rule `.claude/rules/slurm-submission.md`).
