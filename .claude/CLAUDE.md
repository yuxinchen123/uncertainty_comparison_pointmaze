# RND exploration project — Claude instructions

This is `/p/rlprojects/RND/` — a series of classic deep-RL experiments on **intrinsic exploration**
(Random Network Distillation and variants). Current focus: `07_reconstruction/`. The code is fully
independent of the tinker negotiation work under `/p/rlprojects/tinker/`.

## Python environment
- Run all Python through **`conda run -n exploration python …`** (Python 3.11) — the real cluster env
  (`/u/sl5nw/.conda/envs/exploration`).
- NOTE: `07_reconstruction/SETUP.md` documents a generic `conda create -n rnd python=3.10`, but the
  actual runs use **`exploration` / 3.11** — prefer `exploration`. Dependencies in `requirements.txt`
  (torch, gymnasium, gymnasium-robotics, stable-baselines3, wandb).

## Experiment tracking — Weights & Biases
- Entity **`catresearch`**, project **`rnd_07_reconstruction`**. Live metrics go to wandb; local
  `wandb/` run folders are gitignored. Primary metric `eval/mean_extrinsic_reward`; oracle-distance
  under `distance_to_gt/*`. Per-run logging convention: see `.claude/rules/wandb-logging.md`.

## Slurm
- **RND's Slurm usage convention is the project rule `.claude/rules/slurm.md`** — numbered
  `0X_run_<partition>.slurm` agent scripts fanned out at one wandb sweep by `00_batch_slurm.sh`
  (partitions `gpu`/`gnolim`/`cpu`/`nolim`); the launching shell must have `exploration` active
  before `sbatch`. This convention is intentionally project-level (RND's Slurm usage differs from
  the other projects on this cluster).
- Shared cluster hardware/network facts (jaguar03 specs, gpu-partition CPU sizing): the global
  `cluster-slurm.md` rule.

## Reproducible seeding
- Seed arg is **`--a_seed`** (sweep range 0–99); it seeds `random`, `numpy`, `torch`, and CUDA
  together. Follow the global `rng-seeding.md` rule for any new randomness.

## Code conventions
- Numbered scripts pair with sweep configs: `0X_<name>.py` ↔ `0X_wandb_sweep.yaml`; the highest
  number is the live driver (currently `04_many_exploration_method.py`; `--algorithm` switches the
  intrinsic bonus). Oracle visit-count reference uses `gt_*` naming (`gt_position`,
  `gt_position_velocity`); distance metrics under `distance_to_gt/*`.
- Intrinsic-bonus library in `intrinsic/`; PointMaze wrappers in `env_wrapper/`; distance metrics in
  `distance_to_GT/`; tests in `tests/` (`pytest`).

## RL training run folders (same structure + naming as the tinker project)
- A training run is organized as **`train_runs/<run-slug>/`** with subfolders **`analysis/`**,
  **`code/`** (a snapshot of the code used for the run), the run's outputs, and an
  **`experiment_background.md`** written by the `experiment-background` skill at launch.
- Run-slug naming follows the global convention **`YYYY-MM-DD-HH-MM_<slug>`** with **every
  load-bearing knob spelled out** and **no shorthand/abstraction** — e.g.
  `2026-06-23-14-05_sac_algorithm-rnd_beta-0.01_a-seed-0_pointmaze-large`.
- The `analysis/` subfolder follows the global `analysis-folder.md` rule: `analysis.md` + `plots/`
  committed; `data/` + `code/` gitignored.
- wandb remains the live-metrics surface; `train_runs/` is the durable, reproducible record.

## Domain (07_reconstruction)
- SAC (Stable-Baselines3, `MlpPolicy`) on Gymnasium-Robotics **`PointMaze_Large-v3`** (fixed start +
  `top_right` goal). Reward = sparse extrinsic + `beta * intrinsic`. Research question: which
  intrinsic bonus (**RND** variants / oracle **VisitCount** / **EllipticalBonus**) best drives
  exploration, scored by the distance of its learned bonus field to the oracle visit-count field.

## Paper / writeup
- `development_document/` holds the NeurIPS-style writeup (fresh template). Build it with the global
  paper skills: `ml-paper`, `latex-build-tinytex`, `generate-latex-table`, `claude-edit-latex`.

## Collaboration
- `/p/rlprojects/RND` is one git repo shared with a collaborator (`slurm_yuxin/`, owner
  `yuxinchen`). This `.claude/CLAUDE.md` and `.claude/rules/` are committed (shared conventions);
  `.claude/settings.local.json` is personal and gitignored.
