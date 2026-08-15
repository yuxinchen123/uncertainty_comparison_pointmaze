# torch_e2e — experiment ledger (Module 3: env + training fused, PyTorch)

The torch end-to-end path lives in `ppo/torch_ppo/torch_ppo_rnd.py`: the environment step is
already INSIDE the compiled+captured rollout graph (Module 1 torch env's pure `step_core`
composes into the per-step function), so "stacking module 1 on module 2" is the trainer's
capture modes. This folder tracks the e2e-level experiments: fusing the three phases into
one graph, and the env-variant pairings.

| # | change | result (C=128, T=128, N=4, style B) | verdict |
|---|---|---|---|
| 0 | separate graphs: whole-rollout capture + whole-update capture (+ eager post between) | 45.6 ms/iter = 1.44e6 env-steps/s (C=8: 32.8 ms) | baseline |
| 1 | one-graph: rollout + post-processing + update captured as ONE CUDA graph per iteration (static _U batch buffers; post writes them in place; identity-perm build; warmup snapshots/restores env state, running statistics, parameters, optimizer state) | style B C=128: 41.5 ms/iter (vs 45.6 separate-graphs); C=8: 27.4 ms. Style A C=128: 26.9 ms = 2.43e6 env-steps/s; C=8: 20.2 ms | KEEP |
| 2 | TF32 matmuls (tf32 flag -> set_float32_matmul_precision("high")) | style B C=128: 41.5 -> 37.9 ms/iter = 1.73e6 env-steps/s (+9%) | KEEP (production config: one-graph + tf32 + fused adam) |

## Copy-count scaling (one-graph + TF32 + fused Adam, style B, T=128, N=4)

| copies C | ms/iter | total env-steps/s | per-copy env-steps/s |
|---|---|---|---|
| 8 | 27.4 | 1.50e5 | 1.87e4 |
| 128 | 37.9 | 1.73e6 | 1.35e4 |
| 256 | 48.2 | 2.72e6 | 1.06e4 |
| 512 | 68.9 | 3.81e6 | 7.4e3 |
| 1024 | 114.0 | 4.60e6 | 4.5e3 |
| 2048 | 204.4 | 5.13e6 | 2.5e3 |
| 4096 | 386.4 | 5.43e6 | 1.3e3 |
| 8192 | 761.6 | 5.51e6 | 672 |
| 16384 | 1499.6 | 5.59e6 | 341 |
| 32768 | 3009.6 | 5.57e6 | 170 |
| 65536 | OOM during iteration-graph build (95 GB H100) | — | — |

Answers to the task's scaling questions: per-copy throughput is flat only up to the knee
(~128-256 copies) and then decays roughly as 1/C; TOTAL throughput never decreases — it
rises monotonically and saturates near 5.6e6 env-steps/s from ~8192 copies. The maximum
copy count that fits is 32,768.

## Pairings (the task's module-1 x module-2 grid)

- torch env + torch PPO: THIS path (native, fused).
- CUDA-kernel env + torch PPO (`env_backend="cuda"`, MEASURED): style B 25.1 ms (C=8) /
  35.6 ms (C=128); style A 23.2 ms (C=128) — 3-6% over the torch-env backend (the env is a
  small share of the captured iteration, so the kernel's 24x env-only win compresses).
- jax env + torch PPO (cross-framework, MEASURED once): the dlpack boundary costs
  +6.2 to +7.0 ms PER ENV STEP on top of a 70-260 us native step (~40-90x) — every crossing
  synchronizes both runtimes, in addition to breaking capture and scan fusion. Not a
  production path (`benchmarks/bench_cross_pairing.py`).
