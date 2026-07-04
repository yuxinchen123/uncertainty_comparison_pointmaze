# Train run 3.2.1 — RND optimizer variants (O1 Adam, O2 AdaGrad, O3 SGD-1/t), optuna-allocated seeds

## Purpose

Which optimizer for the RND predictor gives the best exploration on PointMaze_Large-v3, and does the
bonus readout scale (mse vs l2) matter? First live run of the writeup's Part II optimizer design
(`development_document/main.tex`, section `sec:train-run-3-2-1`). Three research questions:

1. **Readout scale**: does switching the bonus readout from $B = \tfrac{1}{2}\lVert e\rVert^2$ (mse) to
   $B = \lVert e\rVert_2$ (l2) alone change exploration? (Adam-l2 from this run vs the run-3.1.1
   Adam-mse reference arm — the Adam-mse cell is NOT re-swept here, by user decision, because it is
   exactly run 3.1.1's `rnd_next_state` arm.)
2. **Count memory**: does AdaGrad's cumulative squared-gradient memory beat constant-gain Adam at
   matched readout?
3. **Decreasing step**: does SGD with the shifted 1/t schedule beat both constant-gain optimizers, and
   at which (eta0, t0)?

Scored on the final training-episode extrinsic reward at 1M steps (no standalone eval, run-3.1.1
standard). Seed allocation is adaptive (optuna-bookkept, controller-driven): every configuration gets
seeds in step (seed-outermost queue); once a configuration has n >= 10 finished seeds it is stopped as
soon as $U = \bar{R} + 2.576\,s/\sqrt{n} < \text{bar}$ (its 99% upper confidence limit falls below the
bar), else it keeps receiving seeds to a 50-seed cap.

**Bar = 36.95** — run 3.1.1's best `rnd_next_state` arm (beta=100): mean final training-episode reward
36.95 (SE 3.92, n=48) at the 2026-07-02 aggregation. The controller recomputes this from the run-3.1.1
JSONs at its first start and freezes the recomputed value in `optuna/decisions.jsonl` (this file
records the frozen value next to the 36.95 reference).

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| algorithm | `rnd_next_state` (all configurations) |
| swept: optimizer (`rnd_optimizer`) | O1 `adam` / O2 `adagrad` / O3 `sgd1t` |
| swept: readout (`rnd_bonus_readout`) | O1: {l2} only (mse cell = run-3.1.1 reference, not re-run); O2, O3: {mse, l2} |
| swept: beta | 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4 (8 values, all methods) |
| swept: eta0 (`rnd_sgd_eta0`, O3 only) | 1e-3, 3e-3, 1e-2, 3e-2, 1e-1 |
| swept: t0 (`rnd_sgd_t0`, O3 only) | 1e3, 1e4 |
| configurations | O1 8 + O2 16 + O3 160 = **184** |
| seeds (`a_seed`) | 0–49 cap; adaptive allocation, floor 10 before any stop decision |
| queue size | 184 x 50 = 9200 entries, run ids 0..9199 **seed-outermost** |
| optimizer constants | Adam: lr 1e-3, betas (0.9, 0.999), eps 1e-8; AdaGrad: lr 1e-2, eps 1e-10, accumulator 0 (PyTorch defaults); SGD-1/t: eta_t = eta0/(1 + t/t0), t = predictor updates (0-based), no floor, momentum 0 |
| gradient clipping | none (g_max = infinity) |
| RND fixed | `rnd_output_dim` 128, `rnd_obs_norm` True, `rnd_distance` mse, `n_predictors` 1 |
| env | PointMaze_Large-v3, goal `top_right`, `env_max_episode` 400, no termination wrapper |
| SAC | SB3 MlpPolicy defaults: lr 3e-4, buffer 1e6, batch 256, tau 0.005, gamma (`discount_factor`) 0.999, train_freq 1, gradient_steps 1, device cpu |
| run length | `total_timesteps` 1e6, `eval_freq` 50000, `n_eval_episodes` 100 |
| logging | `eval_standalone` False, `log_distance` False, `use_wandb` False, `z_logging_mode` local, JSON checkpointed at eval cadence |
| Slurm job shape | 32 tasks x 1 CPU, `--ntasks-per-core=2`, `--mem-per-cpu=2G` (64G per job = 2G per worker; ai01–04 exception 1900M), `srun --wait=0`, `--time=4-00:00:00` |
| user headroom | at most 6 x 32-thread jobs on jaguar03 (leaves >= 16 CPUs for the user's own jobs; rule `slurm-submission.md` section 10) |

## Code and config changes

- Four new `train.py` switches, added for this run (defaults reproduce the pre-change behavior
  byte-identically; no new RNG draws): `rnd_optimizer` (adam/adagrad/sgd1t), `rnd_bonus_readout`
  (mse/l2), `rnd_sgd_eta0`, `rnd_sgd_t0`. Implementation in
  `src/rnd_exploration/methods/rnd.py` (`RND._build_optimizer`, `RND._sgd_1t_lr`, the l2 readout
  branch in `compute`, the per-step lr assignment in `update`), forwarding in
  `src/rnd_exploration/methods/__init__.py`, JSON recording (gated on `kind == "rnd"`) in
  `train.py` `_write_local_log`. Training always uses the mse objective; only the reward readout
  changes. New unit tests in `tests/methods/rnd/test_rnd.py` and `tests/methods/test_registry.py`
  (159 tests pass, 2026-07-04).
- Seed allocation is new for this run: full 9200-entry queue built up front; a controller process
  (`slurm/optuna_controller.py`, 1-CPU job) applies the bar rule every 10 minutes and moves stopped
  configurations' remaining `pending/` entries to `queue/<sweep_id>/pruned/`. Optuna 4.9.0 journal
  file `optuna/optuna_journal.log` (three studies: `3_2_1_adam`, `3_2_1_adagrad`, `3_2_1_sgd1t`) is
  the decision ledger; the sampler decides nothing (all trials enqueued; the grid is fixed).
- Worker job shape is new: 32 tasks x 1 CPU per job (previous runs used 8x2 / 16x1), with
  `--ntasks-per-core=2` and `srun --wait=0` (rule sections 7 and 9 of `slurm-submission.md`).
- `worker.py` copied from run 3.1.2 unchanged (ordered claim window: sorted pending names, random
  pick within the first 32 — early seeds finish first).

## Git state

Commit at code-change time: `37507fa9dea03c579c4de56f1f58b07a336c437d` (branch `Use-RLexplore-RND`),
with the four-switch code changes uncommitted on top. The changes are committed before the sweep
launch; the launch-time commit hash is appended below when the launch happens.

Working tree dirty at write time (the run-3.2.1 code changes plus unrelated in-flight run-3.1.2
follow-up artifacts):

```
 M 07_reconstruction/src/rnd_exploration/methods/__init__.py
 M 07_reconstruction/src/rnd_exploration/methods/rnd.py
 M 07_reconstruction/tests/methods/rnd/test_rnd.py
 M 07_reconstruction/tests/methods/test_registry.py
 M 07_reconstruction/train.py
 M .claude/rules/slurm-submission.md   (new section 10: 16-CPU user headroom)
```
