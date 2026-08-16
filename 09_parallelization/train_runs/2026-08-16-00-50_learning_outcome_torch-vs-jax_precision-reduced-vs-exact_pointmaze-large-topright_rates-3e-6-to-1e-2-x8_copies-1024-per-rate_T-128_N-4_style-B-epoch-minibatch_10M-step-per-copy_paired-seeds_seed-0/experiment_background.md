## Purpose

Everything measured in this project so far is speed. This run asks whether the two
implementations of the same algorithm — `ppo/torch_ppo/torch_ppo_rnd.py` and
`ppo/jax_ppo/jax_ppo_rnd.py` — actually **learn the same thing**, and whether the card's
reduced-precision matrix mode changes where either of them ends up.

Two questions:

1. **PyTorch against JAX.** Same algorithm, same environment, same learning rates, same step
   budget. With 1,024 copies per learning rate the bar is not identical curves but overlapping
   seed distributions. A disagreement is treated first as a suspected defect in one of the two.
2. **Reduced precision against exact single precision**, within each framework, everything else
   held identical. Both trainers ship with the card's reduced-precision matrix mode on (PyTorch
   asks for it with `tf32=True`, JAX takes it by default). The two frameworks' forward passes
   differ by 2.2e-03 in that mode and by 1.2e-06 in exact single precision, which is far below
   anything that should move where training converges — so this is a real question, not a
   rhetorical one.

Specification: `analysis/QUEUED_learning_outcome_comparison.md`.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| environment | `PointMaze_Large-v3` reimplementation, map `large`, start cell (7, 1), goal cell (1, 10) |
| reward | sparse: 1 per step within 0.45 of the goal, `reward_shift` 0, continuing task, truncation at 400 steps |
| algorithm | PPO + Random Network Distillation, two value heads, per-copy running statistics |
| update style | `epoch_minibatch` (style B): 4 epochs x 4 shuffled minibatches of 128 rows |
| learning rates | 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 (8 groups) |
| copies per rate | 1,024 |
| total copies per run | 8,192 |
| `sweep_seed_mode` | `paired` — copy k of every rate group shares initial weights and environment reset noise |
| `num_steps` (T) | 128 |
| `n_envs` (N) | 4 |
| iterations | 19,531 (= 9,999,872 environment steps per copy, "10 million") |
| learning-rate annealing | linear to zero over the 19,531 iterations, both frameworks |
| `base_seed` | 0 |
| PyTorch RNG | `torch.manual_seed(0)` in the driver, so the action noise and shuffles are reproducible |
| JAX RNG | `PRNGKey(0)` folded with the iteration index |
| history cadence | every 200 iterations (98 records per run), per-copy reward, intrinsic reward and maze coverage |
| checkpoint cadence | every 1,000 iterations |
| configurations | {PyTorch, JAX} x {reduced precision, exact single precision} = 4 runs |
| precision knob (PyTorch) | `PPOConfig(tf32=True)` / `tf32=False` |
| precision knob (JAX) | `jax_default_matmul_precision` default / `"highest"` |
| machine | serval05, one H100 NVL 95 GB, exclusive lock (`locks/gpu_run.sh`) |

## Code and config changes

- **`ppo/torch_ppo/torch_ppo_rnd.py`**: the matrix-precision knob is now set in both directions.
  It previously read `if cfg.tf32: torch.set_float32_matmul_precision("high")`, so a trainer
  built with `tf32=False` inherited whatever the process had already been left in — in a process
  that had built a `tf32=True` trainer first, a run asking for exact single precision would
  silently have got the reduced one. It now sets `"high"` or `"highest"` explicitly, and asserts
  the setting took. Test: `tests/test_tf32_knob.py`.
- **New**: `code/train_learning_outcome.py` — the campaign driver. One process per
  (framework, precision), chunked and resumable, per-chunk progress flushed to a file, and an
  in-process precision probe that refuses to train if the requested precision is not the one the
  card is actually using.
- **New**: `code/precision_effect_check.py` — runs the real trainer for a few iterations at a
  small copy count and dumps its parameters, so two processes at different precisions can be
  differenced. This is the decisive check that the knob reaches the compiled and graph-captured
  path, not only a bare matrix multiplication.
- **New**: `code/run_campaign.sh` — the four runs in sequence on the machine itself, under one
  hold of the exclusive lock.
- No change to the algorithm, the environment, or any hyperparameter of either trainer.

## Git state

Commit: `001ffd727341542a281feb758c4db6df4e6596fe` at run-folder creation; the code commit that
launched the campaign is recorded in each run's `record.json` under `git`.

Working tree at creation carried unrelated changes from other projects in this repository
(`07_reconstruction`, `08_cleanrl_ppo_rnd`); nothing under `09_parallelization` was uncommitted.
