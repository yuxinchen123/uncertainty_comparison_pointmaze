# jax_e2e — experiment ledger (Module 3: env + training fused, JAX)

The jax end-to-end path lives in `ppo/jax_ppo/jax_ppo_rnd.py`: the environment step
(`pointmaze/jax_env`) is traced INSIDE the jitted rollout `lax.scan`, so env and trainer
already compile into one XLA program per phase. This folder tracks e2e-level experiments.

| # | change | result (C=128, T=128, N=4 unless labeled) | verdict |
|---|---|---|---|
| 0 | baseline (correction: the whole iteration — rollout scan + statistics + GAE + update — was ALREADY one jitted function in v0; only buffer donation was missing) | 74.4 iter/s style A (4.87e6 env-steps/s), 49.8 iter/s style B (3.27e6 env-steps/s); C=8: 111.4 / 86.2 | baseline |
| 1 | donate_argnums on the whole-iteration TrainState (params + optimizer moments + env state donated) | C=128: style A 74.4 (unchanged), style B 50.4 (+1%); C=8: 109.9 (-1%), 87.3 (+1%) — within noise. CPU test suite still passes bitwise. | KEEP (free; removes a state copy; matters as C grows) |
| 2 | LABELED VARIANT: rollout shape T=32, N=16 (same 512 rows/copy/iteration; GAE horizon 32 — a different algorithm per spec section 14, never a silent swap) | C=128: style A 225.8 iter/s = 1.48e7 env-steps/s (3.0x), style B 92.9 iter/s = 6.09e6 env-steps/s (1.9x); C=8: 411.8 / 207.1 | recorded as variant (default stays T=128, N=4) || 3 | C-scaling sweep, C = 128..4096 doubling (default T=128 N=4, donated whole-iteration jit; JSON tag _cscale) | style A total env-steps/s: 4.87e6 / 7.61e6 / 1.14e7 / 1.51e7 / 1.75e7 / 1.90e7 at C=128/256/512/1024/2048/4096 — total throughput NEVER falls, but is nearly flat past C=2048 (+9% for the last doubling). Per-copy throughput halves vs C=128 between C=512 and C=1024 (59% -> 39%). Style B: 3.37e6 -> 9.10e6 total; per-copy halves at C~512. GPU memory 72.5 GB at C=4096 (near the 95 GB limit; C=8192 would likely not fit). | measured (knee recorded) || 4 | ROUND 2 production numbers after the trainer changes (J1 hoist + J2 unroll=4; command buffers already on by default, worth 15% on their own) | iterations/second, T=128 N=4, same harness as the round-1 baseline row: style A 239.8 / 214.2 / 185.6 at C=8/32/128 (round 1: 111.4 / 107.6 / 74.4) = **2.15x / 1.99x / 2.49x**; style B 152.3 / 123.1 / 85.4 (round 1: 86.2 / 77.6 / 49.8) = **1.77x / 1.59x / 1.72x**. At C=128 that is 1.216e7 env-steps/s style A (was 4.87e6) and 5.60e6 style B (was 3.27e6) | KEEP (new jax production baseline) |

## Measurement protocol note (round 2)

Two harnesses measure this trainer and they do NOT produce the same number, so a row must say
which one it used:

- `bench_train_jax.py` lets iterations pipeline, syncing once per timing block. At C=128 style A
  it reports 185.6 iterations/second (5.4 ms each). This is what an actual run achieves, because
  nothing in a real loop forces a synchronization every iteration.
- `ab_compare_jax.py` calls `block_until_ready` after EVERY iteration, which serializes the host
  and the device. The same configuration measures 7.92 ms there. It is the conservative number,
  and it is the right one for A/B work because it cannot hide a change behind pipelining.

The round-2 speedup factors above compare like with like (both sides from the pipelined harness,
round 1 and round 2). The A/B rows in the trainer ledger are all from the serialized harness.

## Pairings

- jax env + jax PPO: THIS path (native, fused in XLA).
- torch env or CUDA-kernel env + jax PPO (cross-framework): dlpack boundary every rollout
  step; breaks the scan fusion. Measured once for the record, not a production path.

## Candidate experiments

- Fuse rollout + post + update into ONE jitted function per iteration (mirror of the torch
  one-graph mode).
- donate_argnums on parameters and optimizer state in the update.
- The T=32, N=16 rollout shape (labeled algorithm variant, spec section 14).
