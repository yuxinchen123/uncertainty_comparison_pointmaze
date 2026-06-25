# RND run-id & local-JSON logging convention (project rule)

How run-2+ sweeps assign run ids and how each run logs locally (no wandb). The 07_reconstruction local
work queue replaced the W&B sweep as the config source; this rule fixes the id and logging conventions so
the writeup's `\section{Logging}` (`development_document/main.tex`) and the analysis code stay in sync.

## Run id (sweep position) — seed outermost
- Each run carries an integer `run_id` (its 0-based position in the sweep) and `run_total` (the sweep
  size, 600 for 3 algorithms x 200 seeds). Together `run_id`/`run_total` is the run's identity ("i / total").
- **The sweep iterates the seed OUTERMOST**, algorithm inner (in fixed `ALGO_BETA` order
  `gt_position_velocity, rnd_elliptical, rnd_state`). So seed 0 owns ids 0–2, seed 1 owns 3–5, …, seed
  199 owns 597–599. An earlier seed always has smaller ids than a later seed.
- `build_queue.py` stamps `run_id`/`run_total` into each queue config; `worker.py` passes them through as
  `--run_id`/`--run_total`; `train.py` uses them.
- When changing the sweep, keep seed outermost and keep `run_id` a gap-free `0..run_total-1`. The unit
  test `slurm/test_run_id_convention.py` pins this ordering — update it with any change.

## Per-run JSON filename
- A sweep run (`run_total > 0`) is named **`<run_id zero-padded to run_total's width>_of_<run_total>.json`**
  — e.g. `042_of_600.json` — by `train.py` `_run_name`. The descriptive fields (algorithm, seed, beta)
  live INSIDE the JSON, not in the filename.
- A standalone run (`run_total == 0`, a manual `train.py` invocation) keeps the descriptive pipe-delimited
  name. Do not remove that branch.

## Logging: own file per run, group-style, single write
- **One JSON per run** under `train_runs/<run>/data/local/`. Because every run owns a distinct path, the
  512 concurrent workers never write the same file — no file-lock contention. Never write a shared log file.
- **Group-style, written once.** File access is slow, so every metric is accumulated in memory by its
  callback at the eval cadence (`eval_freq`) during training, and the whole record — all groups — is
  flushed in a SINGLE file write at run end (`_write_local_log`). Do NOT write incrementally per step.
- The record has four groups: the top-level config/identity fields (`run_id`, `run_total`, `algorithm`,
  `beta`, `a_seed`, `z_logging_mode`, `total_timesteps`, `eval_freq`, `runtime_seconds`), and three
  per-eval history lists:
  - `eval_history` — `eval/mean_*reward`, `eval/n_eval_episodes`, `visit_counts/*` (all algorithms).
  - `train_history` — `train/mean_*reward`, `train/mean_episode_length`, `train/n_episodes_averaged`,
    averaged over the past `n_eval_episodes` training episodes (all algorithms). Plotted as the DASHED
    training curve alongside the solid eval curve.
  - `distance_history` — `distance_to_gt/*` (6 metrics). **Logged ONLY for algorithms in
    `ALGORITHMS_NO_ACTION`** (state-only bonus fields, e.g. `gt_position_velocity`, `rnd_state`); NOT for
    `rnd_elliptical`. This is the one group with an algorithm-dependent trigger.
- The analysis (`analysis/code/common.py` `load_records`) reads every `data/local/*.json` by content, not
  filename, so old descriptive-name JSONs and new id-named JSONs analyze together.

## When a run/sweep is cancelled
- Document it in `train_runs/<run>/CANCELLATION_AND_RESUME.md` (status per algorithm, completed vs
  incomplete config lists). Resume with `slurm/resume_setup.py` (rebuilds the id-based queue and pre-marks
  finished `(algorithm, a_seed)` pairs as done), then `slurm/launch_queue.sh`.
- Cancellation safety is the global rule: only `scancel` ids in `slurm/submitted_jobids.txt`; never
  `scancel -u`. See `.claude/rules/slurm-submission.md`.
