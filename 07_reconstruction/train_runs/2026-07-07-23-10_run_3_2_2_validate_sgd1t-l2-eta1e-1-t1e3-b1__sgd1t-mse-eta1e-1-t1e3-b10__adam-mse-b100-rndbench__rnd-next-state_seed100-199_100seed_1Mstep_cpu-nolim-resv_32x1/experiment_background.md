## Purpose

Out-of-sample validation of Train run 3.2.1's two best configurations against the RND benchmark,
with 100 **brand-new seeds** each, to settle whether the top sgd1t configs genuinely beat the
benchmark once the winner's-curse of the run-3.2.1 selection is removed. In run 3.2.1 the top config
(sgd1t·l2·η0=0.1·t0=1e3·β=1) reached mean 49.2 at n=26, but that config was the argmax over 35 racing
survivors, so its mean is upward-biased; a two-sample test against the benchmark (35.15) gave only
~95% one-sided confidence, not 99%. This run re-measures the same configs on seeds never used before,
so the comparison is unbiased.

Three configurations, each 100 seeds (a_seed 100..199), all `rnd_next_state`, run-3.1.1 stack, no
pruning (all 300 runs execute to completion):

1. **C0** sgd1t · l2 · η0=0.1 · t0=1000 · β=1 — run-3.2.1 rank-1 racing survivor.
2. **C1** sgd1t · mse · η0=0.1 · t0=1000 · β=10 — run-3.2.1 rank-2 racing survivor.
3. **C2** adam · mse · β=100 — the RND benchmark (= run-3.1.1 `rnd_next_state` best arm: Adam at
   PyTorch defaults, MSE readout, no clipping). Re-run here on fresh seeds so the benchmark is measured
   under the identical fleet/timing as the two candidates, not only from run-3.1.1's old data.

Seeds 100..199 are disjoint from every prior use: the two sgd1t configs were run in 3.2.1 on seeds
0..49; the benchmark was run in 3.1.1 on seeds 0..99. So all 300 runs are genuinely new draws.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| configs | 3 (C0 sgd1t-l2-β1, C1 sgd1t-mse-β10, C2 adam-mse-β100 benchmark) |
| seeds (`a_seed`) | 100..199 (100 fresh seeds per config), seed-outermost run ids 0..299 |
| runs | 3 × 100 = 300, run_total=300, NO pruning (all run to completion) |
| algorithm | `rnd_next_state` (all three) |
| C0/C1 optimizer | sgd1t: eta_t = eta0/(1 + t/t0), eta0=0.1, t0=1000, no floor, momentum 0 |
| C2 optimizer | adam, PyTorch defaults (lr 1e-3, betas (0.9,0.999), eps 1e-8) |
| readout | C0 l2 (‖e‖₂); C1, C2 mse (½‖e‖²) |
| beta | C0=1, C1=10, C2=100 |
| RND fixed | `rnd_output_dim` 128, `rnd_obs_norm` True, `rnd_distance` mse, `n_predictors` 1, no clipping |
| env | PointMaze_Large-v3, goal `top_right`, `env_max_episode` 400, no termination wrapper |
| SAC | SB3 MlpPolicy defaults: lr 3e-4, buffer 1e6, batch 256, tau 0.005, gamma 0.999, train_freq 1, gradient_steps 1, device cpu |
| run length | `total_timesteps` 1e6, `eval_freq` 50000, `n_eval_episodes` 100 |
| logging | `eval_standalone` False, `log_distance` False, `use_wandb` False, `z_logging_mode` local, `sb3_verbose` 0 (silent console) |
| Slurm pools | cpu (open) + nolim (open) + jaguar03 & puma01 (reservation). GPU and GNOLIM open partitions avoided entirely (user rule for this submission). |
| Slurm job shape | 32×1 (>=32-core nodes) / 16×1 (fragments), `--ntasks-per-core=2`, `--mem-per-cpu=2G`, `srun --wait=0`; reservation jobs `--time` bounded by the reservation window |
| job names | valid1 (cpu), valid2 (nolim), check1 (jaguar03), check2 (puma01) — 2 bases + numeric suffix (§3) |

## Code and config changes

- No source changes vs run 3.2.1 — reuses the four RND switches (`rnd_optimizer`, `rnd_bonus_readout`,
  `rnd_sgd_eta0`, `rnd_sgd_t0`) and the `sb3_verbose=0` silent console, both already committed.
- `worker.py` is the timeout-free copy (the 24 h `PER_RUN_TIMEOUT` was removed 2026-07-06 after it
  killed 71 near-complete run-3.2.1 runs); the only per-run limit is the Slurm walltime.
- New folder machinery: `build_queue.py` writes the 300-entry seed-outermost queue (seeds 100..199);
  `worker_32x1.slurm` / `worker_16x1.slurm` point RUN_DIR at this folder; `launch_queue.sh` submits
  cpu+nolim+reservation, no controller (fixed 100 seeds, no bar/pruning).

## Git state

Commit at launch: `55eb6bb3e21415e309deda48418d48dca3cc524e` (branch `Use-RLexplore-RND`). Working tree
carries the run-3.2.1 in-flight artifacts plus this new run folder and the worker.py timeout removal
(uncommitted at build time; folded into the next commit).
