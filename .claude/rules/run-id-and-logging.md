# RND run-id & local-JSON logging convention (project rule)

How run-2+ sweeps assign run ids and how each run logs locally (no wandb). The 07_reconstruction local
work queue replaced the W&B sweep as the config source; this rule fixes the id and logging conventions so
the writeup's `\section{Logging}` (`development_document/main.tex`) and the analysis code stay in sync.

## Sweep id — many sweeps per run folder, no legacy-data collisions
- A run folder (`train_runs/<run-slug>/`) can hold **many sweeps**: reruns of the same config (which EXTEND
  coverage) or different configs. Each sweep launch gets a **`sweep_id` = `<YYYY-MM-DD-HH-MM>_<tag>`** (the tag
  names the config version, e.g. `obsrms-distoff`). Reruns of the same config share the tag, differ by timestamp.
- Everything a sweep produces is scoped under its id, so sweeps never collide and old/invalid data never
  pollutes a new analysis:
  - queue:  `queue/<sweep_id>/{pending,running,done,failed}/`
  - data:   `data/<sweep_id>/local/<run_id>_of_<total>.json`
  - job ids: `slurm/submitted_jobids_<sweep_id>.txt` (cancel only a sweep's own ids; never `scancel -u`)
- `slurm/build_queue.py --sweep_id <id>` builds one sweep's queue and appends a row to **`data/SWEEPS.md`**
  (the manifest: id, size, config, `log_distance` on/off, status). `slurm/launch_queue.sh [tag]` generates the
  id, builds the queue, and injects `SWEEP_ID` into every job (`sbatch --export=ALL,SWEEP_ID=...`); `worker.py`
  reads `SWEEP_ID` and only touches that sweep's subtree.
- **Analysis loads by sweep_id.** `load_records(data/<sweep_id>)` reads exactly that sweep (its `local/`
  subdir). Pool multiple sweep_ids that share a config tag to combine reruns; mark superseded sweeps `legacy`
  in `SWEEPS.md` to exclude them. Because the data path is `data/<sweep_id>/local/`, the existing
  `load_records` glob (`<dir>/*/*.json`) works unchanged when pointed at one sweep's dir.
- The `run_id` below is **sweep-local** (0..total-1 within one sweep); the `(sweep_id, run_id)` pair is unique
  within the run folder.

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
- **One JSON per run** under `train_runs/<run>/data/<sweep_id>/local/`. Because every run owns a distinct path
  (sweep-scoped), the 512 concurrent workers never write the same file — no file-lock contention, and runs from
  different sweeps never collide. Never write a shared log file.
- **Group-style, checkpointed at the eval cadence (revised 2026-07-02, run 3.1.2).** Every metric is
  accumulated in memory by its callback; the whole record — all groups — is flushed by `_write_local_log`
  atomically (write `<name>.json.tmp`, then `os.replace`) at EACH eval cadence with `"completed": false`,
  and once at run end with `"completed": true` (`LocalLogCheckpointCallback`, appended last so a flush
  sees the same step's fresh history rows). Do NOT write incrementally per step. Why the revision: under
  the original write-once-at-end convention, 488 run-3.1.1 runs killed at the Slurm job walltime left ZERO
  data despite 10+ trained hours each; ~20 checkpoint writes per 1M-step run to the run's OWN file are
  negligible filesystem load and leave usable partial curves. Analyses and progress counters MUST treat
  `completed=false` records as partial (count separately; usable for truncated curves, never for final-
  reward aggregates); a missing `completed` field means an old write-once record, i.e. complete.
- The record has five groups: the top-level config/identity fields (`run_id`, `run_total`, `algorithm`,
  `beta`, `a_seed`, `z_logging_mode` or `trainer`, `total_timesteps`, `eval_freq`, `runtime_seconds`), and
  four history lists:
  - `eval_history` — `eval/mean_*reward`, `eval/n_eval_episodes`, `visit_counts/*` (all algorithms).
  - `train_history` — `train/mean_*reward`, `train/mean_episode_length`, `train/n_episodes_averaged`,
    averaged over the past `n_eval_episodes` training episodes (all algorithms). Plotted as the DASHED
    training curve alongside the solid eval curve.
  - `train_episode_history` — **one record per COMPLETED training episode** (`step`,
    `train/extrinsic_reward`, `train/intrinsic_reward`, `train/total_reward`, `train/episode_length`), so
    the full first-to-last training-episode trajectory is saved. Episode-level (NOT step-level) to save
    space; all algorithms. The per-eval training-evaluation curve = mean over the past `n_eval_episodes` of
    these. Appended by `TrainEpisodeStatsCallback` on each episode end (this is the run-4 logging addition,
    mirrored into the run-2 python repo so both trainers log it).
  - `distance_history` — `distance_to_gt/*` (6 metrics). **OFF by default** (`log_distance` switch, default
    False): gridding the whole maze through the intrinsic model at every eval is overhead and most runs only
    need the reward curves. When enabled, logged ONLY for algorithms in `ALGORITHMS_NO_ACTION` (state-only
    bonus fields, e.g. `gt_position_velocity`, `rnd_state`); NOT for `rnd_elliptical`. The switch is a
    `Config.log_distance` field (python `train.py`) / `--log_distance` arg (JAX `run4_train.py`); the shared
    `DistanceLoggingCallback` takes `enabled=` and is a no-op (empty history) when off.
- **No final-eval special case.** Every evaluation — including the final one at `total_timesteps` — uses the
  same `n_eval_episodes`; the `n_eval_episodes_final` knob is retired (run-2 set it to 100 = `n_eval_episodes`,
  so this is no numeric change for run-2, just a cleaner convention).
- The analysis (`analysis/code/common.py` `load_records`) reads every `data/local/*.json` by content, not
  filename, so old descriptive-name JSONs and new id-named JSONs analyze together.

## When a run/sweep is cancelled
- Document it in `train_runs/<run>/CANCELLATION_AND_RESUME.md` (status per algorithm, completed vs
  incomplete config lists). Resume with `slurm/resume_setup.py` (rebuilds the id-based queue and pre-marks
  finished `(algorithm, a_seed)` pairs as done), then `slurm/launch_queue.sh`.
- **Requeueing configs whose attempt was killed mid-run (added 2026-07-02):** a rerun OVERWRITES the same
  per-run JSON path on its first checkpoint, destroying the killed attempt's partial record. Before moving
  a killed config's marker from `running/` back to `pending/`, ARCHIVE its partial JSON (completed=false)
  to `data/<sweep_id>/killed_attempts_<YYYY-MM-DD-HH-MM>/` — that preserves the lost-work evidence (steps
  reached, runtime at last checkpoint) for the infrastructure accounting while keeping `local/` truthful.
  Job-level timing needs nothing extra: sacct keeps every job's Start/End permanently, and each attempt's
  `runtime_seconds` restarts from its own clock, so paces never mix attempts.
- Cancellation safety is the global rule: only `scancel` ids in that sweep's own
  `slurm/submitted_jobids_<sweep_id>.txt`; never `scancel -u`. See `.claude/rules/slurm-submission.md`.
