# 06rl_integration — Reconstruction Plan

**Audience:** New maintainer taking over the project.  
**Goal:** Rebuild the codebase incrementally (“start small”) with a clear plan.  
**Status:** Plan only — no code changes in this document.

---

## 1. Current State Summary

### 1.1 What the Codebase Does

- Trains RL agents (SAC or PPO) on **PointMaze** with **intrinsic rewards** (exploration bonuses).
- Total reward: `r_extrinsic + beta * r_intrinsic`. Intrinsic reward comes from:
  - **Learned uncertainty methods** (RND, RND-linear, ensembles, etc.) from folders 01–05, or
  - **Ground-truth baseline**: 1/√n per grid cell (visit-count based).
- Supports **single-goal** (one fixed goal) and **multi-goal** (N goals, sampled per episode).
- Uses **Stable Baselines 3**; optional **WandB** for logging and sweeps; optional **custom replay buffer** for SAC that recomputes intrinsic rewards on sample.

### 1.2 Main Pain Points

| Issue | Where it shows up |
|-------|-------------------|
| **Monolithic training script** | `train_rl.py` ~1650 lines: env creation, 5+ callback classes, buffer replacement, WandB logic, and CLI/sweep handling in one file. |
| **Fragile imports** | Many `sys.path.insert(...)` and `importlib.util.spec_from_file_location(...)` to pull from `01sweep_uncertainty`, `02–05` utilities; naming conflict with local `evaluation/` vs `01sweep_uncertainty/utilities/evaluation.py`. |
| **Tight coupling** | Callbacks reach into env wrappers (Monitor, IntrinsicRewardWrapper), eval callback depends on training wrapper for “training visit counts,” buffer replacement depends on finding the wrapper in the env stack. |
| **Scattered env creation** | `create_env` and `create_eval_env` in `train_rl.py`; `get_maze_map` and `observation_to_grid` imported from 01sweep inside functions; maze_map fetched in multiple places. |
| **Many modes in one path** | Single vs multi-goal, GT vs learned uncertainty, SAC vs PPO, with/without custom buffer, with/without WandB/heatmaps — all branching inside the same functions. |
| **Unclear public API** | Entry is `main()` + `train(config)`; no clear “run one training” vs “run sweep” vs “evaluate only” API for a new maintainer. |

### 1.3 External Dependencies (outside 06rl_integration)

- **01sweep_uncertainty/utilities**: `environment.py` (e.g. `get_maze_map`), `evaluation.py` (e.g. `observation_to_grid_notebook_exact`), `uncertainty_methods.py` (RND, RND-linear, etc.).
- **02–05 folders**: ensemble and scalar-ensemble uncertainty methods (optional; `uncertainty/integration.py` tries to import them).
- **Stable Baselines 3**, **Gymnasium**, **gymnasium_robotics** (PointMaze), **WandB**, **PyTorch**, **NumPy**, **scikit-learn** (goal selection).

---

## 2. Reconstruction Goals

- **Start small:** Get a minimal, runnable pipeline first (one algorithm, one goal mode, one uncertainty source), then add options.
- **Clear boundaries:** Separate “env + wrappers,” “uncertainty,” “training loop,” “logging,” “evaluation” so each can be tested and replaced independently.
- **Stable dependencies:** Prefer a single way to depend on 01 (and optionally 02–05): e.g. package install or a small shim layer in 06, not ad-hoc path/importlib in every file.
- **Testable:** Unit tests for config, env building, reward shaping, and a short integration test that runs a few hundred steps without touching 01–05 internals if possible.
- **Documented:** One place that says “how to run one experiment,” “how to add a new uncertainty method,” “how to add a new callback.”

---

## 3. Phased Reconstruction Plan

### Phase 0: Preparation (no code rewrite yet)

- **0.1** Freeze current behavior: ensure existing `train_rl.py` + evaluate path runs and produces the metrics you care about (e.g. one SAC single-goal run, one GT sweep). Document the exact command and expected outputs.
- **0.2** List all entry points: `train_rl.py` (main + sweep), `evaluate.py` (compare_methods). Note which CLI flags and WandB sweep keys are required.
- **0.3** List all “external” symbols 06 uses from 01 (and 02–05): function names, file paths. This becomes the contract for the new dependency layer.
- **0.4** Decide where the new code will live: e.g. a new top-level folder `06rl_integration_v2/` or a branch with a new subfolder `06rl_integration/minimal/` so the old code keeps working until the new one is validated.

**Deliverable:** A short “current behavior checklist” and “external dependency contract” (could be a section in this doc or a separate DEPENDENCIES.md).

---

### Phase 1: Minimal runnable pipeline (“small”)

**Scope:** One algorithm (e.g. SAC), one goal mode (e.g. single-goal), one intrinsic source (e.g. GT only). No WandB, no heatmaps, no custom buffer, no multi-goal, no PPO.

**1.1 Config**

- Single module or dataclass for “minimal config”: env name, grid size, seed, total_timesteps, eval_freq, n_eval_episodes, algorithm=SAC, goal_mode=single, use_gt_baseline=True, beta, log_dir. No sweep-specific or WandB-specific keys yet.
- Load from CLI (argparse) and optionally from one YAML/JSON file. No WandB config injection yet.

**1.2 Dependency shim (01sweep only)**

- One module in 06 that is the **only** place that imports from 01 (or from a thin wrapper around 01). Expose only what 06 needs, e.g.:
  - `get_maze_map()`
  - `observation_to_grid(obs, grid_rows, grid_cols)` (or the exact name from 01)
- Prefer a single `sys.path` or package setup in that one module (or install 01 as a package). No `importlib.spec_from_file_location` in multiple files.
- If 01 cannot be installed as a package, document “run from repo root” or “set PYTHONPATH” and keep path logic in this shim only.

**1.3 Env construction**

- One function: `build_env(config)` that returns the training env (PointMaze → GoalWrapper → IntrinsicRewardWrapper → Monitor → DummyVecEnv).
- Goal selection (fixed goal for single-goal) and maze_map loading go through the dependency shim. No direct imports of 01 inside train loop or callbacks.
- One function: `build_eval_env(config)` that returns eval env (same goals, no intrinsic in reward, or beta=0 wrapper). No visit-count tracking for heatmaps yet.

**1.4 Intrinsic reward (GT only)**

- One class or function that, given (grid_rows, grid_cols, maze_map, visit_counts), returns intrinsic reward for a state (e.g. 1/√n). This can live in 06 (reimplement or wrap 01’s logic via the shim). No RND/ensemble yet.
- IntrinsicRewardWrapper (or a minimal version) only depends on this GT intrinsic and the dependency shim for observation→grid.

**1.5 Training loop**

- Minimal `train(config)`:
  - Build envs with `build_env` / `build_eval_env`.
  - Create SAC with standard SB3 API.
  - Use SB3’s built-in `EvalCallback` only (no custom eval callback yet).
  - No custom replay buffer; no WandB; no heatmaps.
  - Run `model.learn(total_timesteps=...)`, save model to `config.log_dir`.

**1.6 CLI**

- Single script: e.g. `train_minimal.py` or `python -m 06rl_integration.train_minimal` with args for the minimal config. No sweep, no WandB.

**1.7 Validation**

- Run a short run (e.g. 1000 steps), check that training completes and a model is saved. Optionally compare eval return with one run from the old code (same seed, same config) to spot regressions.

**Deliverable:** A minimal pipeline that runs SAC, single-goal, GT intrinsic, no WandB, with a single entry script and clear modules for config, shim, env, intrinsic, and train.

---

### Phase 2: Core features (same structure, more options)

Add one dimension at a time; keep the same layering (config → shim → env → intrinsic → train).

- **2.1** Add **learned uncertainty** (e.g. one method: RND or RND-linear from 01). The “uncertainty” layer in 06 has a single interface (e.g. `get_uncertainty(coords)`, `update_with_batch(states)`); GT and RND both implement it. `create_uncertainty_method(name, config)` in one place, using the dependency shim to import 01’s methods.
- **2.2** Add **custom replay buffer** for SAC (recompute intrinsic on sample). Hide behind a flag in config; buffer construction uses the same intrinsic interface so it stays decoupled from the wrapper.
- **2.3** Add **multi-goal** mode: goal selection (diverse goals) and wrapper behavior (per-goal visit counts) without changing the rest of the pipeline. Config: `goal_mode`, `num_goals`; env builder branches only in goal selection and wrapper init.
- **2.4** Add **evaluation script** (load model, run N episodes, report metrics). Reuse `build_eval_env` and the same config so evaluation is consistent with training.
- **2.5** Optional: **PPO** support (no replay buffer). Same env and intrinsic interface; only the algorithm and default hyperparameters differ.

**Deliverable:** Same architecture as Phase 1, with support for learned uncertainty, custom buffer, multi-goal, evaluation script, and optionally PPO. Still no WandB/heatmaps in the core.

---

### Phase 3: Logging and experiments

- **3.1** **WandB:** One logging module or callback that receives “events” (e.g. eval results, episode end stats) and pushes to WandB. Training loop and eval callback do not import WandB; they just call a logger interface. WandB switch in config.
- **3.2** **Eval callback:** Replace default EvalCallback with an enhanced one that records extrinsic/intrinsic/total and episode lengths, and passes them to the logger. No direct WandB calls inside the callback.
- **3.3** **Sweeps:** A separate script or entry that reads WandB sweep config and calls the same `train(config)` with config built from `wandb.config`. No sweep logic inside the core training function.
- **3.4** **Heatmaps:** Optional callback that, every N steps, gets visit counts from the wrapper and sends an image to the logger (WandB or file). Again, callback talks to logger interface, not WandB directly.

**Deliverable:** Full logging and sweep capability without cluttering the core train/env/intrinsic code.

---

### Phase 4: Parity and cleanup

- **4.1** Support all uncertainty methods from 01 (and optionally 02–05) through the same interface and dependency shim. Document how to add a new method (new branch in `create_uncertainty_method` + optional new path in shim).
- **4.2** Dual evaluation (e.g. single-goal vs continuation) if still needed: second eval env and a small wrapper around the eval callback or logger that reports two sets of metrics.
- **4.3** Slurm/scripts: point batch scripts at the new entry point(s) and document required env vars and args.
- **4.4** Deprecate old `train_rl.py`: once new pipeline is validated (same metrics on a few seeds/configs), rename or remove the old script and point docs and scripts to the new one.
- **4.5** Add a short “Contributing” or “Architecture” section: how to add an algorithm, a callback, an uncertainty method, a new env.

**Deliverable:** Feature parity with current codebase, single recommended entry point, and clear extension points.

---

## 4. Suggested Layout (after reconstruction)

Keep this as a target structure; implement step by step in the phases above.

```
06rl_integration/
  config/
    __init__.py
    schema.py          # Config dataclass(es), validation, load from CLI/YAML
  deps/
    __init__.py
    sweep_01.py        # Only place that imports 01 (get_maze_map, observation_to_grid, uncertainty methods)
  envs/
    __init__.py
    build.py           # build_env(config), build_eval_env(config)
    wrappers/
      goal.py
      intrinsic.py
  intrinsic/
    __init__.py
    gt.py              # GT 1/sqrt(n)
    factory.py         # create_uncertainty_method(name, config) -> interface
  training/
    __init__.py
    train.py           # train(config) — no WandB, no sweep
    agent.py           # build SAC/PPO from config (optional: buffer replacement)
  logging/
    __init__.py
    wandb_logger.py    # Implements logger interface; forwards to WandB
    callbacks.py       # Eval callback, heatmap callback — use logger interface
  scripts/
    train.py           # CLI → config → train(config)
    sweep.py           # WandB agent → config from wandb.config → train(config)
    evaluate.py        # Load model, build_eval_env, run episodes, print metrics
  tests/
    test_config.py
    test_env_build.py
    test_intrinsic_gt.py
    test_train_short.py  # Few hundred steps, no 01 if possible or with mock)
  docs/
    RECONSTRUCTION_PLAN.md  # This file
    DOCUMENTATION.md       # User-facing: how to run, extend, dependencies
    DEPENDENCIES.md        # Contract with 01 (and 02–05) — Phase 0
```

You can start with a flatter layout (e.g. `config.py`, `env_build.py`, `train_minimal.py`) and refactor into the above as you add features.

---

## 5. Dependency and Import Strategy

- **Single dependency boundary:** All use of 01sweep_uncertainty (and 02–05) goes through one module (e.g. `deps/sweep_01.py`). That module is responsible for `sys.path` or package imports and re-exports only what 06 needs.
- **No `evaluation` name clash:** Either:
  - Import 01’s evaluation helpers under a different name in the shim (e.g. `observation_to_grid_from_01`), or
  - Install 01 as a package with a top-level name (e.g. `sweep_utilities`) and use `from sweep_utilities.evaluation import ...`. Do not use `importlib.spec_from_file_location` in multiple places.
- **Config and envs:** No direct imports of 01 or WandB inside config or env-building code. Config is plain dataclasses/dicts; env builder receives config and uses the shim for maze_map and observation→grid.

---

## 6. Testing and Validation Strategy

- **Unit:** Config load/save and override; env build (mock or minimal PointMaze); GT intrinsic (fixed visit counts → expected 1/√n).
- **Integration:** “Short run” (e.g. 500–2000 steps) with GT intrinsic, single-goal, SAC; check no crash and saved model. Optionally compare eval return with old code (same seed/config).
- **Regression:** Before removing old code, run a small matrix (e.g. 2 seeds × 2 configs) on old and new pipeline and compare key metrics (eval mean return, mean length); document any acceptable differences (e.g. SB3 version).

---

## 7. Order of Work (checklist)

Use this as a high-level checklist; details live in the phases above.

- [ ] **Phase 0:** Freeze current behavior, document entry points and external dependency contract; decide new code location.
- [ ] **Phase 1:** Minimal pipeline (config, shim, env build, GT intrinsic, SAC, single script, short run).
- [ ] **Phase 2:** Learned uncertainty, custom buffer, multi-goal, eval script, optional PPO.
- [ ] **Phase 3:** Logging interface, WandB callback, enhanced eval callback, sweep entry, heatmaps.
- [ ] **Phase 4:** All uncertainty methods, dual eval, Slurm/docs, deprecate old script, extension guide.

---

## 8. What Not to Do (until the new pipeline is stable)

- Do not refactor the existing `train_rl.py` in place while also adding new structure; use a new folder or branch for the new code.
- Do not add new features (e.g. new algorithms or envs) to the old monolith; add them only to the new pipeline once Phase 1–2 are done.
- Do not scatter imports from 01 or 02–05 across multiple files in the new code; keep them behind the single dependency shim.
- Do not change any code as part of this plan; this document is planning only.

---

*End of reconstruction plan. Update this document as you complete phases or adjust scope.*
