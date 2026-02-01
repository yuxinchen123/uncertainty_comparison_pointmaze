# 06rl_integration — Documentation

This folder implements **reinforcement learning (RL) training with intrinsic rewards** on the PointMaze environment, using Stable Baselines 3 (SB3) and uncertainty-based exploration bonuses. The code integrates uncertainty methods from other project folders (01–05) and supports both single-goal and multi-goal setups, with optional ground-truth (GT) intrinsic rewards for baselines.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [Main Components](#main-components)
5. [Training Flow](#training-flow)
6. [Evaluation](#evaluation)
7. [File Reference](#file-reference)
8. [Usage](#usage)

---

## Overview

### Purpose

- Train RL agents (SAC or PPO) on **PointMaze** with:
  - **Extrinsic reward**: task reward (e.g. +1 for reaching the goal).
  - **Intrinsic reward**: exploration bonus from an uncertainty method (e.g. RND, RND-linear, or ground truth 1/√n).
- Total reward: `r_total = r_extrinsic + beta * r_intrinsic`.
- Support **single-goal** (one fixed goal) or **multi-goal** (N goals, sampled per episode).
- Log metrics and visit-count heatmaps to **Weights & Biases (WandB)**.

### Key Concepts

- **Uncertainty method**: Produces a scalar “uncertainty” per state (e.g. RND prediction error or GT 1/√n). Used as intrinsic reward.
- **Visit counts**: Per grid cell (and per goal in multi-goal). Used for GT intrinsic reward and for updating some uncertainty methods.
- **Beta**: Coefficient scaling intrinsic reward; `beta=0` disables intrinsic rewards.

---

## Architecture

### High-Level Stack

```
CLI (main) → train(config)
    → create_env() → PointMaze → GoalWrapper → IntrinsicRewardWrapper → Monitor → DummyVecEnv
    → create_uncertainty_method() from uncertainty/integration.py (wraps methods from 01–05)
    → SAC/PPO agent (optionally with custom IntrinsicReplayBuffer for SAC)
    → Callbacks: EnhancedEvalCallback, WandBLoggingCallback, CheckpointCallback, VisitCountHeatmapCallback
```

### Environment Wrapping Order (inner → outer)

1. **PointMaze** (Gymnasium): base maze environment.
2. **GoalWrapper**: sets goal(s); single-goal removes `desired_goal` from obs; multi-goal keeps it and samples a goal per episode.
3. **IntrinsicRewardWrapper**: maps state to grid, computes intrinsic reward, maintains visit counts, combines extrinsic + beta * intrinsic.
4. **Monitor**: records episode returns/lengths for SB3.
5. **DummyVecEnv**: SB3 vectorized env interface (single env).

Evaluation envs are built similarly but without intrinsic reward affecting the reported metric (e.g. `create_eval_env` with `track_visit_counts` only for heatmaps).

---

## Configuration

### `config.py` — `RLConfig`

Central dataclass for all settings.

| Category   | Key examples | Description |
|-----------|----------------|-------------|
| Algorithm | `algorithm`   | `'sac'` or `'ppo'` |
| Env       | `env_name`, `grid_rows`, `grid_cols` | PointMaze and grid size (e.g. 9×12) |
| Goals     | `goal_mode`, `num_goals`, `fixed_goal_cell` | `'single'` or `'multi'`; for single, optional fixed (row,col) |
| Uncertainty | `uncertainty_method`, `uncertainty_config` | Method name and params (e.g. `feature_dim`, `theta_seed`, `regularization`, `update_frequency`, `num_epochs_per_update`) |
| Intrinsic | `beta`, `use_gt_baseline` | Intrinsic weight; if True, use GT 1/√n as intrinsic |
| Training  | `total_timesteps`, `learning_rate`, `batch_size`, `buffer_size` | Main training knobs |
| SAC/PPO   | `sac_config`, `ppo_config` | Algorithm-specific (tau, gamma, n_steps, etc.) |
| Eval      | `eval_freq`, `n_eval_episodes`, `dual_eval` | How often and how many episodes to evaluate |
| Logging   | `log_dir`, `tensorboard_log`, `wandb_switch`, `heatmap_log_freq` | Logs and WandB/heatmaps |
| Misc      | `device`, `a_seed` | Device and global seed |

- **`get_default_config()`**: returns default `RLConfig`.
- **`update_config_from_dict(config, config_dict)`**: overwrites or merges config from a dict (e.g. WandB sweep).

---

## Main Components

### 1. `train_rl.py` — Training Script

**Entry:** `main()` parses CLI (and optionally `wandb.config` in sweep mode), builds `RLConfig`, then calls `train(config)`.

**Important functions:**

- **`create_env(config, uncertainty_method, gt_tracker)`**
  - Builds training env: `make_pointmaze_env` → `GoalWrapper` → `IntrinsicRewardWrapper`.
  - Goal selection: single-goal uses `select_fixed_goal`; multi-goal uses `select_diverse_goals`.
  - If `use_gt_baseline`: uses `GTIntrinsicReward` as the uncertainty method; otherwise uses the provided `uncertainty_method`.

- **`create_eval_env(config, continuing_task, track_visit_counts)`**
  - Builds evaluation env with same goals as training. No intrinsic reward in the metric; optional `IntrinsicRewardWrapper` with `beta=0` only to track visit counts for heatmaps.

- **`train(config)`**
  - Seeds, creates uncertainty method (unless GT baseline), creates train/eval envs, builds SAC/PPO (with `MlpPolicy` or `MultiInputPolicy` depending on obs space).
  - For SAC with `beta != 0`: can replace the replay buffer with `IntrinsicReplayBuffer` or `DictIntrinsicReplayBuffer` so that **intrinsic rewards are recomputed at sample time** using current visit counts.
  - Registers callbacks, runs `model.learn()`, saves final model and prints (and optionally logs) wrapper and GT stats.

**Callbacks (all in `train_rl.py`):**

- **`IntrinsicRewardCallback`**: Legacy; triggers uncertainty model updates every N steps (wrapper already updates in `step()`).
- **`DualEvalCallback`**: Evaluates on two envs (e.g. single-goal vs continuation); results can be logged by a custom WandB callback.
- **`EnhancedEvalCallback`** (extends SB3 `EvalCallback`):
  - Runs evaluation at `eval_freq`; can use deterministic seeds (`eval_seed_base + episode_idx`).
  - For each eval episode, records extrinsic, intrinsic (using **training** visit counts), and total reward, plus length.
  - Stores results in `evaluations_results`, `evaluations_timesteps`, `evaluations_intrinsic_rewards`, `evaluations_total_rewards`, `evaluations_length` for logging.
- **`WandBLoggingCallback`**:
  - Logs to WandB at the right steps: eval metrics (from `EnhancedEvalCallback` or dual eval), train episode metrics (from Monitor + wrapper’s extrinsic/intrinsic), and periodic `global_step`.
  - Reads from the training env’s Monitor and `IntrinsicRewardWrapper` to get per-episode extrinsic/intrinsic/total and length.
- **`CheckpointCallback`**: Saves model every `eval_freq`.
- **`VisitCountHeatmapCallback`**: Every `heatmap_log_freq` steps, gets visit counts from the training (and optionally eval) wrapper and logs heatmap images to WandB.

---

### 2. Wrappers

**`wrappers/goal_wrapper.py` — `GoalWrapper`**

- **Single-goal**: fixes one goal cell; removes `desired_goal` from observation; same goal every reset.
- **Multi-goal**: keeps a list of goal cells; on each reset samples one uniformly and sets it; observation keeps `desired_goal` for goal conditioning.
- Exposes `get_current_goal_cell()`, `get_current_goal_idx()`, `get_all_goals()`.

**`wrappers/intrinsic_reward_wrapper.py` — `IntrinsicRewardWrapper`**

- Wraps env after `GoalWrapper`.
- Maps continuous state to grid with `observation_to_grid_notebook_exact` (from `01sweep_uncertainty/utilities/evaluation.py`).
- **Visit counts**: single-goal `(grid_rows, grid_cols)`; multi-goal `(num_goals, grid_rows, grid_cols)`; only open cells (maze_map[row,col]==0) are incremented.
- In `step()`:
  1. Calls `env.step(action)` → gets extrinsic reward.
  2. Maps new state to grid, appends to `visited_states`.
  3. Updates uncertainty method’s visit counts (so intrinsic reward uses current counts).
  4. Computes intrinsic reward **before** incrementing visit count for current cell (bonus = “before this visit”).
  5. Increments visit count for current cell.
  6. Optionally calls `update_with_batch` on the uncertainty method (online learning).
  7. Returns `reward_total = reward_extrinsic + beta * intrinsic_reward` and puts `intrinsic_reward`, `extrinsic_reward`, `total_reward` in `info`.
- On reset: saves last episode extrinsic/intrinsic into `last_episode_*` for callbacks, then clears episode stats and `visited_states`.
- Exposes `get_visit_counts()`, `get_visit_counts_for_goal(goal_idx)`, `get_visited_states()`, `get_statistics()`, `get_episode_statistics()`, and internally `_get_uncertainty(observation)`.

---

### 3. Uncertainty Layer

**`uncertainty/integration.py`**

- **`UncertaintyMethodAdapter`**: Wraps any method that has `get_uncertainty(coordinates)` and optionally `train_on_positions(...)`. Buffers states and calls `update_with_batch(states)` every `update_frequency` steps with `num_epochs_per_update`.
- **`create_uncertainty_method(method_name, method_config, device)`**: Factory that builds the right method from folders 01–05 (e.g. `rnd`, `rnd_linear_sgd`, `rnd_linear_ls`, `elliptical`, ensemble and scalar-ensemble variants) and wraps it in `UncertaintyMethodAdapter`.

**`uncertainty/gt_intrinsic.py` — `GTIntrinsicReward`**

- No trainable parameters. Holds a visit-count matrix; `update_visit_counts(visit_counts)` syncs it from the wrapper.
- `get_uncertainty(coordinates)`: for each (x,y) → grid (row,col); if open and in bounds, returns `1/sqrt(count)` (or 1.0 if count 0); walls/out-of-bounds → NaN. Used as the “ground truth” exploration bonus for baselines.

---

### 4. Buffers (SAC only)

**`buffers/intrinsic_replay_buffer.py` — `IntrinsicReplayBuffer`**

- Extends SB3 `ReplayBuffer`. Stores transitions; in `add()`, also stores extrinsic reward from `info['extrinsic_reward']`.
- In `sample()`: gets a batch from the parent, then **recomputes** intrinsic reward for the batch using `intrinsic_reward_fn(observations)` (which uses current visit counts from the wrapper), and returns rewards = extrinsic + beta * intrinsic. So the agent is trained on bonuses that reflect **current** exploration state, not stale at collection time.

**`buffers/dict_intrinsic_replay_buffer.py` — `DictIntrinsicReplayBuffer`**

- Same idea for dict observation spaces (e.g. `observation` = dict with `achieved_goal`, etc.): stores extrinsic, recalculates intrinsic on sample.

---

### 5. Evaluation and Metrics

**`evaluate.py`**

- **`evaluate_model(model, env, n_episodes, deterministic)`**: Runs the policy, records episode return/length and intrinsic/extrinsic from `info`; fills `RLPerformanceMetrics` and returns its statistics.
- **`compare_methods(method_configs, config_base, n_episodes)`**: For each method config (name, model path, uncertainty method, beta, etc.), builds env with `create_env` (from `train_rl`), loads model, runs `evaluate_model`, and optionally prints visit stats from `GroundTruthTracker`. Used for comparing trained checkpoints.

**`evaluation/gt_tracker.py` — `GroundTruthTracker`**

- Holds a copy of visit counts; `update_from_visit_counts(visit_counts)` updates from wrapper. Provides `compute_gt_uncertainty_matrix()` (1/√n per cell) and `get_statistics()` (e.g. total visits, visited/unvisited cells).

**`evaluation/metrics.py` — `RLPerformanceMetrics`**

- Deque-based window of episode returns, lengths, intrinsic/extrinsic sums, success. `record_episode(...)` and `get_statistics()` for means, stds, success rate, etc.

---

### 6. Utils

- **`utils/env_utils.py`**: `make_pointmaze_env(env_name, seed, continuing_task)`, `get_env_info(env)`.
- **`utils/goal_utils.py`**: `get_valid_cells(env, maze_map)`, `cell_to_continuous_coords(cell, grid_rows, grid_cols)`, `select_fixed_goal(env, seed, goal_cell, maze_map)`, `select_diverse_goals(env, n_goals, seed, maze_map)` (k-means on valid cells).
- **`utils/heatmap_utils.py`**: Used by `VisitCountHeatmapCallback` to build visit-count heatmap images for WandB.

---

## Training Flow

1. **Parse args** (and WandB sweep config if present); build `RLConfig`.
2. **Seed** numpy/torch (and CUDA if used).
3. **Uncertainty method**: unless `use_gt_baseline`, create via `create_uncertainty_method(...)`; GT baseline uses `GTIntrinsicReward`.
4. **Envs**: `create_env` → training env with GoalWrapper + IntrinsicRewardWrapper; `create_eval_env` → eval env (same goals, no intrinsic in metric).
5. **Agent**: SAC or PPO with policy type from observation space (dict → MultiInputPolicy). Optionally replace SAC buffer with `IntrinsicReplayBuffer` / `DictIntrinsicReplayBuffer` and `intrinsic_reward_fn` from the training wrapper.
6. **Callbacks**: EnhancedEvalCallback, WandBLoggingCallback (if WandB on), CheckpointCallback, and optionally VisitCountHeatmapCallback.
7. **Learn**: `model.learn(total_timesteps=..., callback=callbacks)`.
8. **Save** final model; print and optionally log wrapper and GT stats.

During training, each step:

- Env step goes through GoalWrapper → IntrinsicRewardWrapper: intrinsic is computed (with up-to-date visit counts), visit count updated, reward = extrinsic + beta * intrinsic.
- If SAC with custom buffer, sampling recalculates intrinsic from current visit counts.
- At `eval_freq`, EnhancedEvalCallback runs eval episodes and stores extrinsic/intrinsic/total and lengths; WandBLoggingCallback logs them and train episode stats.

---

## Evaluation

- **Online**: `EnhancedEvalCallback` evaluates at `eval_freq`; metrics are “eval/extrinsic_reward”, “eval/intrinsic_reward”, “eval/total_reward”, “eval/episode_length”, etc., logged by `WandBLoggingCallback`.
- **Offline**: `evaluate.py`’s `compare_methods()` loads saved models and runs `evaluate_model()` on envs created with the same config (and same uncertainty wrapper for visit stats). Use for comparing methods or sweeps.

Evaluation intrinsic reward is computed using **training** visit counts (passed via `training_wrapper` into `EnhancedEvalCallback`) so that eval reflects what the policy was trained on.

---

## File Reference

| Path | Role |
|------|------|
| `train_rl.py` | Main training script, env creation, callbacks, SAC/PPO setup, custom buffer wiring |
| `config.py` | `RLConfig`, defaults, `update_config_from_dict` |
| `evaluate.py` | Offline evaluation and `compare_methods` |
| `wrappers/goal_wrapper.py` | Goal setting and observation filtering (single/multi goal) |
| `wrappers/intrinsic_reward_wrapper.py` | Visit counts, intrinsic reward, combined reward |
| `uncertainty/integration.py` | Adapter and `create_uncertainty_method` (01–05) |
| `uncertainty/gt_intrinsic.py` | GT intrinsic reward 1/√n |
| `buffers/intrinsic_replay_buffer.py` | SAC buffer that recomputes intrinsic on sample |
| `buffers/dict_intrinsic_replay_buffer.py` | Same for dict observations |
| `evaluation/gt_tracker.py` | Visit count copy and GT stats |
| `evaluation/metrics.py` | `RLPerformanceMetrics` |
| `callbacks/visit_count_heatmap_callback.py` | Visit count heatmaps to WandB |
| `utils/env_utils.py` | PointMaze creation and env info |
| `utils/goal_utils.py` | Valid cells, coords, fixed/diverse goal selection |
| `utils/heatmap_utils.py` | Heatmap image creation |
| `gt_baseline_*.yaml`, `plain_rl_wandb_sweep.yaml` | WandB sweep configs |
| `slurm/*` | Job scripts for running on clusters |

---

## Usage

**Single run (e.g. SAC, RND-linear LS, beta=1.0, single goal):**

```bash
python train_rl.py --algorithm sac --uncertainty_method rnd_linear_ls --beta 1.0 --goal_mode single --total_timesteps 100000
```

**GT baseline (intrinsic = 1/√n):**

```bash
python train_rl.py --use_gt_baseline --beta 1.0 --goal_mode single
```

**Multi-goal, 5 goals:**

```bash
python train_rl.py --goal_mode multi --num_goals 5
```

**WandB sweep:** run the trainer with a sweep that sets `wandb.config`; the script reads options from `wandb.config` when `wandb.run` is not None (see `main()` in `train_rl.py`). Use the provided YAML sweep configs with `wandb sweep` and `wandb agent`.

**Evaluate saved models:**

```bash
python evaluate.py --model_paths path/to/model1 path/to/model2 --method_names method1 method2 --n_episodes 10
```

(Optional: `--uncertainty_methods`, `--betas` for each model.)

---

This documentation describes the current behavior of the code without modifying any of it. For implementation details, refer to the docstrings and source files listed above.
