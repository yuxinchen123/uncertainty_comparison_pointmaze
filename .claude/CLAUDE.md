# RND exploration project — Claude instructions

This is `/p/rlprojects/RND/` — a series of classic deep-RL experiments on **intrinsic exploration**
(Random Network Distillation and variants). Current focus: `07_reconstruction/`. The code is fully
independent of the tinker negotiation work under `/p/rlprojects/tinker/`.

## Python environment
- Run all Python through the SHARED canonical env **`/p/rlprojects/RND/.venvs/exploration/bin/python …`**
  (Python 3.11; registry `.venvs/ENVS.md`). `conda run -n exploration` (the private
  `/u/sl5nw/.conda/envs/exploration`) is RETIRED for new work — identical clone, kept only to
  reproduce runs launched <= 3.2.3.
- NOTE: `07_reconstruction/SETUP.md` documents a generic `conda create -n rnd python=3.10`, but the
  actual runs use **`exploration` / 3.11** — prefer `exploration`. Dependencies in `requirements.txt`
  (torch, gymnasium, gymnasium-robotics, stable-baselines3, wandb).

## Experiment tracking — Weights & Biases
- Entity **`catresearch`**, project **`rnd_07_reconstruction`**. Live metrics go to wandb; local
  `wandb/` run folders are gitignored. Primary metric `eval/mean_extrinsic_reward`; oracle-distance
  under `distance_to_gt/*`. Per-run logging convention: see `.claude/rules/wandb-logging.md`.

## Slurm
- **Large CPU sweeps follow the SHARED skill
  `/p/rlprojects/.claude/skills/uva-submit-cpu-sweep/SKILL.md`** (moved there 2026-07-11; readable by
  every rlprojects member): how to submit (reservation discovery per user, partition buckets, gpu
  allowlist filled lowest-GPU-capability first, job shapes, `srun --wait=0`, capacity planning
  over each user's own pools) plus the local file work queue (sweep ids, seed-outermost run ids,
  per-run JSON logging). The project stubs `.claude/rules/slurm-submission.md` and
  `.claude/rules/run-id-and-logging.md` keep the RND-specific facts (paths, scripts, record
  format). Collaborators adding workers to a live run follow that run's
  `for_collaborator/README.md`.
- The launch and worker scripts call the env's python by full path, so no conda activation is
  needed at `sbatch` time. The canonical project env is the SHARED
  `/p/rlprojects/RND/.venvs/exploration/bin/python` (registry: `.venvs/ENVS.md`; the old private
  `/u/sl5nw/.conda/envs/exploration` is retired — kept only to reproduce runs launched <= 3.2.3).
- The old `wandb agent` fan-out convention (`.claude/rules/slurm.md`) was deleted 2026-07-08 —
  superseded by the work-queue convention above.
- Shared cluster hardware/network facts (jaguar03 specs, gpu-partition CPU sizing): the global
  `uva-cluster-slurm.md` rule.

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
- Launching a run also means adding its writeup block to the development document — the SHARED
  skill `rnd-experiment-tex-track` (see "Paper / writeup" below).

## Domain (07_reconstruction)
- SAC (Stable-Baselines3, `MlpPolicy`) on Gymnasium-Robotics **`PointMaze_Large-v3`** (fixed start +
  `top_right` goal). Reward = sparse extrinsic + `beta * intrinsic`. Research question: which
  intrinsic bonus (**RND** variants / oracle **VisitCount** / **EllipticalBonus**) best drives
  exploration, scored by the distance of its learned bonus field to the oracle visit-count field.

## Paper / writeup
- `development_document/` holds the NeurIPS-style writeup (fresh template). Build it with the global
  paper skills: `latex-build-tinytex`, `generate-latex-table`, `claude-edit-latex`.
- **Every new train run adds its writeup block per the SHARED skill
  `/p/rlprojects/.claude/skills/rnd-experiment-tex-track/SKILL.md`**: an env-spec table (new
  environments only), a what-is-swept table, a best-configuration results table, and a
  training-curve figure, in the grid table style; the skill's `example.tex` is the compilable
  style reference. Load it at run launch (together with `experiment-background`) and whenever a
  run's results go into the document.

## Collaboration
- `/p/rlprojects/RND` is one git repo shared with a collaborator (`slurm_yuxin/`, owner
  `yuxinchen`). This `.claude/CLAUDE.md` and `.claude/rules/` are committed (shared conventions);
  `.claude/settings.local.json` is personal and gitignored.
