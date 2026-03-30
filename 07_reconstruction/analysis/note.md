# Sweep analysis — background

This folder is for notes and figures tied to the W&B sweep defined in `07_reconstruction/04_wandb_sweep.yaml`.

## W&B sweeps used for analysis

The runs analyzed here come from **two** sweeps in project **`rnd_07_reconstruction`** (same sweep name, different IDs):

| Sweep name     | Sweep ID   | Team (API entity) | Notes |
|----------------|------------|-------------------|--------|
| `sweep_my_rnd` | **`y24vyh06`** | `catresearch` | Larger estimated grid (~8k runs per dashboard). |
| `sweep_my_rnd` | **`j40bkl6y`** | `catresearch` | Earlier sweep (~4k est. runs). |

Dashboard creator shown: **yuxinchen-claire**. The **entity** passed to the W&B API is usually the team slug (here **`catresearch`**, as in `catresearch/rnd_07_reconstruction/...`); set `WANDB_ENTITY` if your team differs.

**Download local copies:** `analysis/script/download_wandb_sweeps.py` → **`analysis/data/`**. Working analysis notes: **`analysis/analysis.md`**.

## Sweep configuration (summary)

| Field | Value |
|--------|--------|
| **Program** | `04_many_exploration_method.py` |
| **W&B project** | `rnd_07_reconstruction` |
| **Sweep name** | `sweep_my_rnd` |
| **Method** | grid |
| **Optimization metric** | `eval/mean_extrinsic_reward` (maximize) |

**Setting:** SAC on **PointMaze_Large-v3** with fixed **bottom_right** goal, **100 seeds** (`a_seed` 0–99), **2M** `total_timesteps`, **400** max steps per episode, **γ = 0.999**. Eval every **50k** steps (**100** mid-run eval episodes; **10k** episodes on the final eval step). **Device:** CPU in the YAML (change if you launch on GPU). **W&B:** on.

**Intrinsic scaling:** `beta` ∈ `{0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000}` (eight values), crossed with every **algorithm** below — a large grid.

**Algorithms** (see `_algorithm_to_config` in `04_many_exploration_method.py`):

- `no_exploration` — baseline (β forced to 0 in code).
- `gt_position`, `gt_position_velocity` — visit-count–style bonuses.
- `rnd_next_state`, `rnd_next_state_position_only`, `rnd_state`, `rnd_state_action`, `rnd_state_action_next_state` — RND variants on different features.
- `rnd_linear_next_state` — linear RND (frozen shared body, trainable predictor head vs frozen target).
- `rnd_elliptical` — elliptical bonus.

**RND hyperparameters held fixed in the sweep:** `rnd_obs_norm: true`, `rnd_distance: mse`, `rnd_output_dim: 128`, `n_predictors: 1`.

**Environment flag:** `apply_termination_wrapper: false` — time limits stay as **truncation** so Stable-Baselines3 can treat timeouts correctly for bootstrapping (see project debug notes under `debug_claude/env/` if needed).

## Code revision used for this note

When this file was written, the local repo **HEAD** was:

- **Commit:** `9c46143042054bda2776b507f7fd237c99b126d3`
- **Subject:** `fixed linear RND to have differnt head`
- **Date:** 2026-03-21 (author timezone in git log)

Recent commits that touched this sweep or the training entrypoint include `9cb21da` (termination wrapper flag), `2ff924b` (multi-algorithm / distance logging line), and earlier sweep shape changes.

**If you launched the sweep from another machine or an older W&B sweep ID**, replace the commit above with the SHA you actually used (`git rev-parse HEAD` at launch time, or the commit recorded in W&B run metadata).

## How to run the sweep (reminder)

From `07_reconstruction/`, use W&B’s sweep CLI against `04_wandb_sweep.yaml` (e.g. `wandb sweep` then agents), ensuring the working directory and Python path match how your Slurm or local jobs invoke `04_many_exploration_method.py`.

---

## Factors to record for replication and readable analysis

Use this as a checklist when you write up results or share this folder. Others need more than the YAML to match your numbers and understand your conclusions.

### Code and environment

- **Exact git commit** (full SHA) for every run or sweep; note if `RLeXplore` / other submodules are pinned at a specific commit too (`git submodule status`).
- **Conda / venv name** or a frozen **`requirements.txt` / `conda env export`** (Python version, `torch`, `gymnasium`, `gymnasium-robotics`, `stable-baselines3`, `wandb` versions).
- **Working directory** used to launch training (e.g. must be `07_reconstruction/` so imports resolve).
- **Uncommitted local changes** (`git status` / patch) if you ran with edits not on the recorded SHA.

### W&B and job orchestration

- **W&B entity / team** and **sweep ID** (or URL) for the run set you analyze, not only project name.
- **How jobs were started** (e.g. `wandb agent …`, Slurm script path, number of parallel agents, partition/GPU vs CPU as actually used — YAML says `cpu` but clusters may override).
- **Any CLI overrides** not in `04_wandb_sweep.yaml` (extra env vars, `CUDA_VISIBLE_DEVICES`, etc.).

### Training algorithm details (defaults not in the sweep)

Stable-Baselines3 **SAC defaults** you did *not* sweep: learning rate, buffer size, batch size, tau, ent_coef, policy architecture (`MlpPolicy` hidden sizes), learning starts, train frequency — list these if you changed them in code or they differ from SB3 docs.
- **RND internals** fixed in code: predictor LR (`0.001`), RND batch size (`256`), linear vs non-linear RND construction (document the commit where linear RND target/predictor split changed).

### Environment and task semantics

- **PointMaze** version / `gymnasium-robotics` behavior: `continuing_task=True`, `reset_target=False`, fixed start/goal cells per `goal_position` and seed (Manhattan distance is logged at startup).
- **`apply_termination_wrapper`** value and why (affects truncation → bootstrap in the replay buffer).
- **Observation** after wrappers: flattened `[x, y, vx, vy]` (4D) after `RemoveGoal` + `FlattenObservation`; which RND **feature** each algorithm uses (state vs next state vs action concat).

### Randomness and scale

- **Seeds:** `a_seed` grid and whether PyTorch / NumPy / env seeding is fully fixed (SB3 `seed`, `DummyVecEnv.seed`).
- **Expected variance** across seeds; whether analysis reports mean ± std / CIs and over how many seeds.
- **Total number of runs** in the sweep (algorithms × β × seeds) and whether any runs failed or were dropped.

### Metrics and logging (what “results” mean)

- **Primary metric** in the YAML (`eval/mean_extrinsic_reward`) vs any **secondary metrics** you use in the paper or plots (`eval/mean_intrinsic_reward`, visit-count coverage, distance-to-GT callbacks, final eval with `n_eval_episodes_final`).
- **Definition** of each logged key (e.g. intrinsic summed per episode with `beta` already applied in train/eval callbacks — align with code when interpreting magnitudes).
- **Step axis** for curves: `step` = env steps vs gradient steps; eval logged every `eval_freq`.

### Analysis artifacts (this folder)

- **Raw export:** run `analysis/script/download_wandb_sweeps.py` (records time in each sweep’s `manifest.json`); data under **`analysis/data/<sweep_id>/`** (`runs_index.csv` + per-run `config.json` / `summary.json`).
- **Analysis code:** notebook or script **commit + path** used to produce each figure/table; **random seed** for any subsampling or bootstrap.
- **Filtering rules** (e.g. exclude incomplete runs, min `total_timesteps`).
- **Figure captions** in prose: what is averaged, error bars, and which β / algorithm subset each plot shows.

### Optional but high value

- **Wall-clock and hardware** (GPU model, CPU count) for cost/comparability.
- **Known bugs** fixed mid-sweep (e.g. linear RND init, termination wrapper) and whether you **split** results before/after that commit.

---

*Add your analysis (tables, plots, conclusions) alongside this file or in subfolders as you prefer.*
