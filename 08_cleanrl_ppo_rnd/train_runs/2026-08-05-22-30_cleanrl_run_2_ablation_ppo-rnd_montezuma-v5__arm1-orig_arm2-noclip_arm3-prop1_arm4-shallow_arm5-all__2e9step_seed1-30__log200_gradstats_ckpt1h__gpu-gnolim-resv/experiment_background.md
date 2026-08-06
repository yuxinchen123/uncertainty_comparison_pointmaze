## Purpose

Ablate three of CleanRL's Random Network Distillation choices against this project's own RND, on
Montezuma's Revenge, so that a difference in exploration performance can be attributed to a named
implementation choice rather than to the two codebases differing in a dozen ways at once.

Arm 1 is CleanRL's `ppo_rnd_envpool.py` as published (with one defect corrected, below). Arms 2 to 4
each change exactly one knob toward this project's RND; arm 5 changes all three, so comparing arm 5
against the sum of arms 2 to 4 measures whether the three interact.

| arm | what it changes from arm 1 |
|---|---|
| `arm1_original` | nothing — the reference |
| `arm2_no_rnd_grad_clip` | the RND predictor's gradient is not clipped; **the policy still is** |
| `arm3_update_proportion_1` | the predictor trains on the whole batch, not a random quarter |
| `arm4_shallower_predictor` | the predictor is one block deeper than the target, not two |
| `arm5_all` | all three together |

30 seeds per arm, 150 runs, 2,000,000,000 environment steps each.

The gradient statistics logged alongside are the second question: how often CleanRL's clip actually
touches the predictor. The twenty-run rehearsal already showed the answer is "on 92% to 97% of
optimizer steps, halving the predictor's gradient, for reasons entirely to do with the policy" — the
predictor contributes 0.006% to 0.030% of the joint squared norm. Arms 2 and 5 remove that.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| environment | `MontezumaRevenge-v5` via envpool 0.6.6, 128 parallel copies |
| algorithm | PPO + Random Network Distillation |
| total timesteps per run | 2,000,000,000 |
| rollout | `num_envs` 128 × `num_steps` 128 = 16,384 steps per policy update |
| minibatches / epochs | 4 / 4 |
| learning rate | 1e-4, Adam, eps 1e-5 |
| discount, extrinsic / intrinsic | 0.999 / 0.99 |
| GAE lambda | 0.95 |
| clip coefficient | 0.1 |
| entropy / value coefficients | 0.001 / 0.5 |
| intrinsic / extrinsic advantage weights | 1.0 / 2.0 |
| policy gradient clip | global norm 0.5, **in every arm** |
| RND predictor clip | joint with the policy (arms 1, 3, 4); none (arms 2, 5) |
| predictor update proportion | 0.25 (arms 1, 2, 4); 1.0 (arms 3, 5) |
| predictor blocks beyond the target | 2 (arms 1, 2, 3); 1 (arms 4, 5) |
| observation-normalization warm-up | 50 rollouts of a random agent = 6,400 steps |
| seeds | 1 … 30 per arm |
| logging cadence | every 200 policy updates = 3,276,800 steps |
| extrinsic reward reported | mean over the last 200 finished episodes |
| checkpoint cadence | every 3,600 s, plus on SIGTERM at the next update boundary |
| per-run resources | 1 GPU, 8 CPU threads, ~4.9 GB GPU memory, ~6 GB host memory |

## Code and config changes

- **Entry point** `src/ppo_rnd_envpool_shuze.py`, derived from CleanRL `ppo_rnd_envpool.py` at
  upstream commit `fe8d8a0`.
- **The envpool auto-reset defect is corrected** (`--fix_envpool_autoreset`, on for every run).
  envpool resets on the step *after* the one that ends an episode, and that step's submitted action
  is discarded — so one row per episode in the rollout buffer pairs an action with a state it never
  acted on. The correction drops those rows and stops the intrinsic advantage carrying backwards
  across the boundary. Measured directly on the installed envpool rather than taken from the docs.
- **No wandb and no tensorboard.** Each run writes one JSON record
  `data/<sweep_id>/local/<run_id>_of_150.json` in train run 8.1.2's shape, flushed atomically at
  every logging step, plus an append-only episode sidecar `<run>.episodes.jsonl`.
- **Only the mean is logged.** Whether a run's result is its final performance or its average across
  the run is decided afterwards by `analysis/compute_run_metrics.py`, not baked into training.
- **Gradient statistics** accumulate on the GPU and are read to the host once per logging interval,
  so they add no synchronisation to the update loop.
- **`PYTHONNOUSERSITE=1` in `slurm/worker_env.sh`**: a torch 2.10+cu128 install in `~/.local` would
  otherwise shadow the environment's torch 2.6.0+cu124 and remove Pascal support, killing every
  `gnolim` run.
- **Environment**: `/p/rlprojects/RND/.venvs/cleanrl_rnd` (Python 3.10, torch 2.6.0+cu124,
  envpool 0.6.6, gym 0.23.1) — shared, so a collaborator can submit against it.

## Git state

Repository root `/p/rlprojects/RND`, commit `221a32309855b96dfdb61654d76ddc32f4146545`.
Working tree clean at submission time except for the sweep folder created by this launch.
