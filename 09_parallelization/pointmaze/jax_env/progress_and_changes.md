# jax_env — experiment ledger (autoresearch style)

Goal metric (correctness): fixture checker (`../common/code/check_against_fixtures.py --impl jax`).
Goal metric (speed): env steps/second on the serval05 H100 (JSONs in `../../benchmarks/results/`).

| # | change | result | verdict |
|---|---|---|---|
| 0 | v0: pure-functional transliteration of the torch exact physics (unrolled 8-candidate contacts, 2-contact QP, frozen fmix32 reset RNG); EnvState NamedTuple, jit + donated state; `scan` benchmark mode for the fused rollout upper bound | fixture checker ALL PASS float64 (one-step <= 0, rollout <= 1.8e-15) and float32; resets bit-identical to torch (test_cross_impl_rng.py) | KEEP — correctness baseline |

| 1 | GPU baseline measurements (H100, float32) | jit mode: 78.6/77.3/71.0/145.7 us per batch-step at N=1e3/1e4/1e5/1e6 = 1.27e7/1.29e8/1.41e9/6.87e9 env-steps/s. scan mode (whole K-step block one program; minimal per-step outputs, an upper bound a trainer cannot fully reach): 14.8/15.2/19.1/119.2 us = 6.7e7/6.6e8/5.2e9/8.39e9 env-steps/s. Beats torch compile (248 us @1M) 1.7x in like-for-like jit mode. | baseline recorded || 2 | ROUND 2 (env row 1). Verify buffer donation actually applies on the jitted step, rather than being silently dropped | Ran the jit-mode benchmark with UserWarning promoted to an error: no "donated buffers were not usable" warning is raised, so the donation the benchmark asks for is real. No code change needed | VERIFIED (no change) |
| 3 | ROUND 2 (env row 4). `unroll` on the scan-mode benchmark's step loop | N=1000: 14.8 -> 9.5 us (unroll 4) -> 8.4 us (unroll 8), i.e. **1.77x at the launch-bound floor**; N=1e6: 119.1 -> 109.4 -> 107.0 us (**1.11x**), raising the 1e6 figure from 8.39e9 to 9.34e9 env-steps/s. The unroll=1 numbers reproduce the round-1 baseline (14.8 and 119.1 vs 119.2 us), which is the stability evidence for these single-shot measurements | KEEP unroll=8 for the scan curve (recorded as a benchmark knob, `--unroll`) |
| 4 | ROUND 2. Fixture gates re-run after the round-2 benchmark changes (the env source itself was not modified) | float64 ALL PASS (one-step 0, rollout <= 1.8e-15); float32 ALL PASS | green |


## Notes

- The scan benchmark stores only a reward sum per step, not per-step observations, so it is
  an upper bound; the e2e trainer's rollout must write observations and will land between
  the jit and scan curves.
- NOT attempted in round 2, and why: the four-scalar-planes layout (env row 2) and the
  rank-by-counting contact selection (env row 3) are real refactors of the env source with
  fixture gates attached, and the end-to-end value of either is bounded by a measurement the
  torch side already has — swapping in an environment 24 times faster moved the whole training
  iteration from 36.6 to 35.6 ms, because the environment is a small share of a training step.
  They remain the right next rows for anyone optimizing the env in isolation.
