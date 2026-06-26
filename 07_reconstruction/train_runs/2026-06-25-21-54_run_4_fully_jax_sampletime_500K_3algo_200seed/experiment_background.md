## Purpose

Train run 4 is the **fully-JAX, sample-time reproduction of Train run 2**. Run 2 trained SAC (Stable-Baselines3,
torch) with the torch intrinsic models on PointMaze_Large-v3. Run 4 trains the SAME task with the SAC backend in
JAX (sbx) and, for `rnd_state`, the RND predictor/target/obs-norm folded into JAX (`jax_rnd.JaxRND`). The
research question is unchanged (which intrinsic bonus best drives exploration); run 4 adds a JAX implementation
whose per-step intrinsic path has no torch, to (a) reproduce run 2's learning curves and (b) measure the
end-to-end speedup. The comparison plot overlays run-2 eval curves (dashed) and run-4 eval curves (solid), same
algorithm = same color, plus run-4 training-evaluation curves (darker same color).

**Algorithmic parity is exact** — the only intended difference from run 2 is `total_timesteps` (500K vs run-2's
1M). In particular the intrinsic is recomputed at **sample time** (on each sampled gradient-step batch), matching
run-2's `VectorIntrinsicReplayBuffer`, not at step time.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| env | PointMaze_Large-v3, fixed start + `top_right` goal, `env_max_episode=400`, `apply_termination_wrapper=False` |
| algorithms × beta | `gt_position_velocity`×1, `rnd_elliptical`×0.01, `rnd_state`×100 (== run-2 ALGO_BETA) |
| seeds | 0–199 (200 seeds); sweep = 3 × 200 = **600 runs**, seed-outermost ids 0..599 |
| total_timesteps | **500,000** (run-2 used 1,000,000 — the one intended difference) |
| eval_freq / n_eval_episodes | 50,000 / 100 (no final-eval special case: final eval also uses 100) |
| SAC | sbx `MlpPolicy`, lr 3e-4, buffer 1e6, learning_starts 100, batch 256, tau 0.005, train_freq 1, gradient_steps 1, ent_coef auto, policy_delay 1, net 256×256, γ=0.999, device cpu — sbx defaults == SB3 defaults == run-2 |
| RND (rnd_state) | JAX: net 256→relu→128 orthogonal std=√2, lr 1e-3, obs-norm clip[-5,5], distance mse, n_predictors 1, feature=observations — mirrors torch `rnd.py` |
| intrinsic timing | **sample time** — `SACWithIntrinsic.train` recomputes reward = extrinsic + beta·intrinsic on the sampled batch each gradient step and trains the predictor then (== run-2 `VectorIntrinsicReplayBuffer.sample`) |
| per-worker threads | 2 CPUs (srun cpus-per-task=2); OMP/MKL/OPENBLAS=2 |

## Code and config changes

- New trainer `code/run4_train.py` (+ `code/jax_rnd.py`): sbx SAC subclass `SACWithIntrinsic` that recomputes the
  intrinsic at the numpy hook in `train()` before the jitted `_train`; `InfoIntrinsicWrapper` computes the
  step-time intrinsic for the info (train/intrinsic logging) only, buffer stores extrinsic reward.
- Logging (run-4 convention, `.claude/rules/run-id-and-logging.md`): per JSON the record holds `eval_history`,
  `train_history` (per-eval mean over past n_eval_episodes), and `train_episode_history` — one row per completed
  training episode (episode-level), so the full first-to-last training-episode trajectory is saved. No final-eval
  special case.
- Python repo (run-2 code) updated to the SAME logging convention: `train_episode_stats.py` adds
  `episode_history`; `wandb_eval_logging.py` removes the final-eval special case; `train.py` writes
  `train_episode_history`. (Run-2 data already collected; only run-4 runs with this code.)
- Slurm: `slurm/build_queue.py` (600 configs, seed-outermost), `slurm/worker.py` (runs `run4_train.py` under the
  JAX venv `/p/rlprojects/RND/.venvs/jax_bench`), `slurm/worker.slurm` (XLA single-thread + 2-thread BLAS),
  `slurm/launch_queue.sh` (reservation-aware, per `.claude/rules/slurm-submission.md`).

## Git state

Commit: `37f775a70aba7ae5405e88047bd9e47aacf49f40` (branch `Use-RLexplore-RND`)

Working tree dirty — run-4 trainer/queue are untracked; the logging-convention edits are uncommitted:
```
 M 07_reconstruction/src/rnd_exploration/callbacks/train_episode_stats.py
 M 07_reconstruction/src/rnd_exploration/callbacks/wandb_eval_logging.py
 M 07_reconstruction/train.py
?? 07_reconstruction/analysis/2026-06-25-run-profiling/
?? 07_reconstruction/train_runs/2026-06-25-21-54_run_4_fully_jax_sampletime_500K_3algo_200seed/
```
