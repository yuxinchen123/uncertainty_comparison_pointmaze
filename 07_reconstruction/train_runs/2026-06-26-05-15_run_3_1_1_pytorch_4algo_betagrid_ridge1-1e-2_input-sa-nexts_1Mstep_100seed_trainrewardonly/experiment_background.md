## Purpose

Train run 3.1.1 — the first actual run of the Train-run-3 family (Part I: elliptical bonuses), using the
live **PyTorch** trainer (`07_reconstruction/train.py`, SB3 SAC), not the JAX run-4 trainer. It compares
four intrinsic-bonus methods on `PointMaze_Large-v3` (fixed start, `top_right` goal), each tuned over an
intrinsic-coefficient (β) grid, and — for the three elliptical methods — over a ridge-λ grid and a
feature-input grid. Scoring is on the **training-episode extrinsic reward** (standalone eval is OFF, the new
default) to save compute. The research question: among batch vs global elliptical memory (and add- vs
sample-time covariance updates) and RND-next-state, which best drives extrinsic reward, and how do the ridge
and the encoder input (state+action vs next-state-only) matter.

The four methods (columns of the hyperparameter table):

| # | Description | `--algorithm` | distinguishing knob |
|---|---|---|---|
| A1 | Batch elliptical, unit-norm | `rnd_elliptical` | batch covariance, `elliptical_update_timing=sample` |
| A2 | Global elliptical, unit-norm, update at buffer-sample time | `rnd_elliptical_global` | `elliptical_update_timing=sample` |
| A3 | Global elliptical, unit-norm, update at buffer-add time | `rnd_elliptical_global` | `elliptical_update_timing=add` |
| A4 | RND next-state | `rnd_next_state` | RND, next-state input, no ridge |

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| env | `PointMaze_Large-v3`, `top_right` goal, `continuing_task=True`, `reset_target=False` |
| `env_max_episode` | 400 |
| `apply_termination_wrapper` | False |
| `discount_factor` (γ) | 0.999 |
| policy / agent | SAC (`MlpPolicy`), SB3 defaults (lr 3e-4, buffer 1e6, batch 256, τ 0.005, ent_coef auto, net [256,256], train_freq/gradient_steps 1) |
| `total_timesteps` | 1,000,000 |
| `device` | cpu |
| `eval_standalone` | **False** (new default; no deterministic eval rollout) |
| `eval_freq` | 50,000 (cadence of the windowed train-reward snapshot + visit-count coverage) |
| `n_eval_episodes` | 100 (window size for the windowed train-reward mean) |
| `log_distance` | False |
| logging | local JSON only (`z_logging_mode=local`, `use_wandb=False`) |
| β grid (all 4 methods) | {1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3} (7) |
| ridge λ (`elliptical_regularization`, A1/A2/A3) | {1, 1e-2} |
| feature input (`elliptical_feature_input`, A1/A2/A3) | {state_action, next_state} |
| `elliptical_feature_normalization` (A1/A2/A3) | unit |
| elliptical feature_dim / bonus_method / bonus_clip | 128 / cholesky / 5.0 |
| RND (`rnd_next_state`): output_dim / obs_norm / distance / n_predictors / lr | 128 / True / mse / 1 / 1e-3 |
| `a_seed` | 0–99 (100 seeds) |

Sweep size: 3 elliptical × 2 input × 2 ridge × 7 β = 84, plus 7 RND = **91 configs/seed × 100 seeds = 9100
runs**. The file work-queue holds all 9100 pending configs; the number of Slurm jobs is small and fixed
(`launch_queue.sh`, per `.claude/rules/slurm-submission.md`) and workers drain the queue. The writeup is
updated once every one of the 91 configs has ≥30 finished seeds.

## Code and config changes

Relative to the live `07_reconstruction` baseline:

- `src/rnd_exploration/methods/elliptical_bonus.py`: added `feature_input` ("state_action" | "next_state")
  to `EllipticalBonus`; `next_state` builds φ from `next_observations` only (`_input_dim = obs_dim`, action
  dropped). Inherited by `GlobalEllipticalBonus`.
- `train.py`: added `Config`/argparse knobs `elliptical_regularization`, `elliptical_feature_normalization`,
  `elliptical_update_timing`, `elliptical_feature_input`, and `eval_standalone` (default **False** — the new
  no-standalone-eval standard). `_write_local_log` now records `eval_standalone` and (for elliptical algos)
  the four elliptical knobs so the analysis can group runs.
- `src/rnd_exploration/methods/__init__.py`: the elliptical factory branch now forwards
  `feature_normalization` and `feature_input` from cfg (previously omitted).
- `src/rnd_exploration/callbacks/wandb_eval_logging.py`: `eval_standalone` gates the expensive deterministic
  rollout; the cheap visit-count coverage is still logged when off.
- Tests: extended `tests/methods/test_elliptical_bonus.py` (next-state cases); updated stale callback tests
  (`tests/callbacks/test_callbacks.py`); replaced `tests/integration/test_final_eval_n_episodes.py` with
  `tests/integration/test_eval_standalone.py`.
- Slurm: this run's `slurm/` (build_queue.py / worker.py / worker.slurm / launch_queue.sh /
  test_run_id_convention.py) drives `train.py` under the `exploration` conda env, using the run-4 SWEEP_ID
  isolation (`data/<sweep_id>/local/`).

## Git state

Commit: `8622c33d5dc10f52898192459494fe7d02ef3777` (`8622c33 07_reconstruction: Train run 4 ...`)

Working tree is **dirty**. The elliptical-bonus implementation (batch + global classes, `update_timing`,
feature normalization, the buffer `observe`/`update` hooks) was already in-progress uncommitted work at
session start (`buffers/vector_intrinsic_replay_buffer.py`, `methods/{__init__,base,elliptical_bonus}.py`,
`tests/methods/test_elliptical_bonus.py`); run 3.1.1 builds on it and adds the changes listed above. The
run uses the live package at this state (not a frozen snapshot), so do not edit the elliptical/train code
while the sweep is running.
