# jax_e2e — experiment ledger (Module 3: env + training fused, JAX)

The jax end-to-end path lives in `ppo/jax_ppo/jax_ppo_rnd.py`: the environment step
(`pointmaze/jax_env`) is traced INSIDE the jitted rollout `lax.scan`, so env and trainer
already compile into one XLA program per phase. This folder tracks e2e-level experiments.

| # | change | result (C=128, T=128, N=4 unless labeled) | verdict |
|---|---|---|---|
| 0 | baseline (correction: the whole iteration — rollout scan + statistics + GAE + update — was ALREADY one jitted function in v0; only buffer donation was missing) | 74.4 iter/s style A (4.87e6 env-steps/s), 49.8 iter/s style B (3.27e6 env-steps/s); C=8: 111.4 / 86.2 | baseline |
| 1 | donate_argnums on the whole-iteration TrainState (params + optimizer moments + env state donated) | C=128: style A 74.4 (unchanged), style B 50.4 (+1%); C=8: 109.9 (-1%), 87.3 (+1%) — within noise. CPU test suite still passes bitwise. | KEEP (free; removes a state copy; matters as C grows) |
| 2 | LABELED VARIANT: rollout shape T=32, N=16 (same 512 rows/copy/iteration; GAE horizon 32 — a different algorithm per spec section 14, never a silent swap) | C=128: style A 225.8 iter/s = 1.48e7 env-steps/s (3.0x), style B 92.9 iter/s = 6.09e6 env-steps/s (1.9x); C=8: 411.8 / 207.1 | recorded as variant (default stays T=128, N=4) || 3 | C-scaling sweep, C = 128..4096 doubling (default T=128 N=4, donated whole-iteration jit; JSON tag _cscale) | style A total env-steps/s: 4.87e6 / 7.61e6 / 1.14e7 / 1.51e7 / 1.75e7 / 1.90e7 at C=128/256/512/1024/2048/4096 — total throughput NEVER falls, but is nearly flat past C=2048 (+9% for the last doubling). Per-copy throughput halves vs C=128 between C=512 and C=1024 (59% -> 39%). Style B: 3.37e6 -> 9.10e6 total; per-copy halves at C~512. GPU memory 72.5 GB at C=4096 (near the 95 GB limit; C=8192 would likely not fit). | measured (knee recorded) |

## Pairings

- jax env + jax PPO: THIS path (native, fused in XLA).
- torch env or CUDA-kernel env + jax PPO (cross-framework): dlpack boundary every rollout
  step; breaks the scan fusion. Measured once for the record, not a production path.

## Candidate experiments

- Fuse rollout + post + update into ONE jitted function per iteration (mirror of the torch
  one-graph mode).
- donate_argnums on parameters and optimizer state in the update.
- The T=32, N=16 rollout shape (labeled algorithm variant, spec section 14).
