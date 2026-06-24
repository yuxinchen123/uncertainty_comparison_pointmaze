---
name: wandb-logging
description: What every RND training run logs to Weights & Biases and how to inspect a run, so RND runs are reproducible from artifacts.
---

# RND wandb logging convention

Every RND training run must be reproducible from its wandb run plus its `train_runs/<slug>/` folder.

## What to log to wandb
- Config: the full argparse/sweep config (algorithm, beta, a_seed, total_timesteps, env, discount).
- Metrics: `eval/mean_extrinsic_reward` (primary), training episode stats, and `distance_to_gt/*`
  (each intrinsic bonus field vs the oracle visit-count field).
- Entity `catresearch`, project `rnd_07_reconstruction`.

## How to inspect a run
- Pull sweeps with `analysis/script/download_wandb_sweeps.py`; combine + analyze under `analysis/`.
- The durable per-run record (analysis, code snapshot, `experiment_background.md`) lives in
  `train_runs/<slug>/` (see CLAUDE.md run-folder convention), not only in wandb.
