# RND run-id & local-JSON logging convention — generic part in the SHARED submit-cpu-sweep skill (2026-07-11); RND specifics kept here

The generic work-queue convention lives in the SHARED skill
`/p/rlprojects/.claude/skills/submit-cpu-sweep/SKILL.md` (moved 2026-07-11; readable by every
rlprojects member). It covers: `sweep_id` = `<YYYY-MM-DD-HH-MM>_<tag>` scoping
(`queue/<sweep_id>/{pending,running,done,failed}/`, `data/<sweep_id>/local/`,
`slurm/submitted_jobids_<sweep_id>.txt`, the `data/SWEEPS.md` manifest), seed-outermost run ids
with execution following id order, per-run JSON filenames (`<id>_of_<total>.json`), one-file-per-run
checkpointed logging with the `completed` flag, and the requeue/archive procedure after kills.

Short version of the logging convention (full version in the user rule): each run writes ONE JSON
under `train_runs/<run>/data/<sweep_id>/local/`, flushed atomically (`.tmp` + `os.replace`) at each
eval cadence with `"completed": false` and once at run end with `"completed": true`; analyses must
treat `completed=false` records as partial (usable for truncated curves, never for final-reward
aggregates); a missing `completed` field means an old write-once record, i.e. complete.

This file keeps the RND-specific parts, so the writeup's `\section{Logging}`
(`development_document/RND_development_document.tex`) and the analysis code stay in sync.

## RND sweep order
- `run_total` 600 = 3 algorithms x 200 seeds; seed OUTERMOST, algorithm inner in fixed `ALGO_BETA`
  order `gt_position_velocity, rnd_elliptical, rnd_state` — seed 0 owns ids 0–2, seed 1 owns 3–5,
  …, seed 199 owns 597–599.
- The unit test `slurm/test_run_id_convention.py` pins this ordering — update it with any change.

## RND record format (five groups)
The record has five groups: the top-level config/identity fields (`run_id`, `run_total`, `algorithm`,
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

The whole record is flushed by `_write_local_log` at the checkpoint cadence above
(`LocalLogCheckpointCallback`, appended last so a flush sees the same step's fresh history rows).

## No final-eval special case
Every evaluation — including the final one at `total_timesteps` — uses the same `n_eval_episodes`;
the `n_eval_episodes_final` knob is retired (run-2 set it to 100 = `n_eval_episodes`, so this is no
numeric change for run-2, just a cleaner convention).

## Analysis loader
The analysis (`analysis/code/common.py` `load_records`) reads every `data/<sweep_id>/local/*.json`
by content, not filename, so old descriptive-name JSONs and new id-named JSONs analyze together.
