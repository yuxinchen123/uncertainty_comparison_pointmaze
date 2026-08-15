# torch_ppo — experiment ledger (autoresearch style)

Goal metric: milliseconds per training iteration (T=128 steps x N=4 envs per copy, style B =
4 epochs x 4 minibatches) on the serval05 H100, and env-steps/second; correctness gated by
tests/test_torch_ppo.py (CPU) and tests/test_capture_gpu.py (GPU). Bench JSONs in
`../../benchmarks/results/` (bench_train.py).

| # | change | result (C=128 unless noted) | verdict |
|---|---|---|---|
| 0 | v0 eager: batched [C,...] params, baddbmm forwards, per-copy statistics/clip, sum-over-copies loss, two update styles | 777 ms/iter (rollout 680 + update 95); FLAT from C=8 to C=128 — 128 copies cost the same as 8. 8.4e4 env-steps/s. Tests: same-seed bit-identical, copy isolation, predictor learns — PASS both styles | baseline |
| 1 | torch.compile the fused per-step rollout body + fused Adam | 226-231 ms/iter (rollout 118, update 95-113) — 3.4x | superseded by 3 |
| 2 | reduce-overhead mode on the per-step body | 189 ms/iter (rollout 100) — cudagraph partially skipped (warning) | superseded by 3 |
| 3 | whole-rollout CUDA-graph capture of the EAGER body (pure step_core, in-place RMS, static buffers, pre-drawn noise) | rollout 187-211 ms — WORSE than per-step compile: replaying thousands of tiny unfused eager kernels is kernel-time bound | superseded by 4 |
| 4 | capture the COMPILED per-step function (fused kernels x 128 in one replay, zero python) | rollout 35-40 ms. Capture bitwise-equal to uncaptured compiled step (test_capture_gpu) | KEEP |
| 5 | capture the whole UPDATE phase (capturable+fused Adam with tensor lr, static input buffers, identity-perm build so the graph build consumes no RNG, compiled loss fwd/bwd, in-graph per-copy clip) | update 125 -> 16.7 ms; TOTAL 45.6 ms/iter = 1.44e6 env-steps/s at C=128 (17x vs baseline; C=8: 32.8 ms). Captured update BITWISE equal to eager update (worst param diff 0.0, both styles) | KEEP |

## Notes

- Spec §11 correction found during implementation: style A must keep the probability ratio in
  the policy surrogate (it carries the gradient); only the clip machinery may be dropped.
  Recorded in the spec file; the jax twin implements the corrected form.
- Comparison point: the jax twin measures 20 ms/iter (style B) / 13 ms (style A) at C=128.
  Remaining torch gap is rollout kernel time (226 us/step incl. 10 buffer-copy kernels) —
  candidate next steps: TF32 matmuls, folding buffer writes into the compiled step,
  concatenating the four 4-input GEMMs (actor/critic/target/predictor first layers) into one.
- The T=32, N=16 rollout-shape alternative (spec §14) is measured separately for the report;
  it changes the algorithm (GAE horizon) so it is a labeled variant, never a silent swap.
