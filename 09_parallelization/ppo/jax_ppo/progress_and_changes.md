# jax_ppo — experiment ledger (autoresearch style)

Goal metric (correctness): the three test suites in `tests/` plus the cross-framework
forward-agreement test against the torch fixture.
Goal metric (speed): training iterations/second on the serval05 H100
(JSONs in `../../benchmarks/results/`, impl `jax_ppo`).

| # | change | result | verdict |
|---|---|---|---|
| 0 | v0: full twin of torch_ppo_rnd.py per the algorithm spec (incl. the section-11 correction). Batched params [C, ...], matmul forwards, float64 running statistics (x64 on, everything else explicit float32), lax.scan rollout, scan intrinsic filter, reverse-scan GAE, per-copy grad clip, sum-over-copies loss, hand-rolled torch-formula Adam, two separately jitted update styles (style B = one lax.scan over 16 pre-gathered minibatches). | CPU tests all pass: same-seed bit-identical (both styles), copy isolation (both styles), finite + falling intrinsic reward. Cross-framework forward agreement vs torch fixture: worst mixed error 8.6e-7 (gate 1e-5). GPU (H100): one compilation (~4 s), then steady per-iteration times; C=8 smoke 300 iterations in 8.8 s with intrinsic reward falling 4.3 -> 0.03. | KEEP — correctness baseline |
| 1 | first throughput rows (no tuning): full_batch 111/108/74 iter/s at C=8/32/128; epoch_minibatch 86/78/50 iter/s. At C=128 full_batch trains 4.87e6 env-steps/s, epoch_minibatch 3.27e6. Copies are close to free: 16x copies costs only ~1.5x wall time. | baseline numbers recorded (JSON 2026-08-15-01-37-53) | measured || 2 | donate_argnums=(0,) on the whole-iteration jit (TrainState donated) | neutral at current sizes (<=1.3%, within noise); tests still pass bitwise | KEEP |
| 3 | LABELED VARIANT T=32/N=16 (spec 14): 4x fewer sequential rollout steps, 4x wider per-step GEMMs | style A C=128: 3.0x (225.8 iter/s, 1.48e7 env-steps/s); style B 1.9x | recorded (not default) || 4 | C-scaling sweep (see e2e/jax_e2e ledger row 3 for the numbers; JSON 2026-08-15-02-49-51 tag _cscale) | per-copy throughput halves between C=512 and 1024 (style A) / at C~512 (style B); total saturates ~1.9e7 (A) / ~9.1e6 (B) env-steps/s | measured |

## Notes

- Deviations from the letter of the task/spec, with reasons:
  - The priming PRNG label is the constant 999999937 (fold_in rejects negative ints).
  - Test-1 tolerance is the standard mixed form (atol 1e-6 + rtol 1e-5) rather than pure
    relative 1e-5: the gain-0.01 actor head produces values near 1e-3 where float32 BLAS
    accumulation noise (~1e-7 absolute) exceeds 1e-5 pure-relative. All outputs agree to
    8.6e-7 mixed error.
  - Style B's 16 optimizer steps run as one lax.scan over pre-gathered minibatches
    (spec 12 allows "unrolled or lax.scan").
- Next optimization candidates (not yet tried): donate_argnums on the TrainState in
  `_iterate` (removes a state copy per iteration), fusing the priming loop into one jit,
  jnp.compress-free minibatch gather layouts, larger N with fewer T per spec section 14.
