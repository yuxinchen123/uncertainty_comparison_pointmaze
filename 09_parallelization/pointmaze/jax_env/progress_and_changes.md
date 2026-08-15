# jax_env — experiment ledger (autoresearch style)

Goal metric (correctness): fixture checker (`../common/code/check_against_fixtures.py --impl jax`).
Goal metric (speed): env steps/second on the serval05 H100 (JSONs in `../../benchmarks/results/`).

| # | change | result | verdict |
|---|---|---|---|
| 0 | v0: pure-functional transliteration of the torch exact physics (unrolled 8-candidate contacts, 2-contact QP, frozen fmix32 reset RNG); EnvState NamedTuple, jit + donated state; `scan` benchmark mode for the fused rollout upper bound | fixture checker ALL PASS float64 (one-step <= 0, rollout <= 1.8e-15) and float32; resets bit-identical to torch (test_cross_impl_rng.py) | KEEP — correctness baseline |

| 1 | GPU baseline measurements (H100, float32) | jit mode: 78.6/77.3/71.0/145.7 us per batch-step at N=1e3/1e4/1e5/1e6 = 1.27e7/1.29e8/1.41e9/6.87e9 env-steps/s. scan mode (whole K-step block one program; minimal per-step outputs, an upper bound a trainer cannot fully reach): 14.8/15.2/19.1/119.2 us = 6.7e7/6.6e8/5.2e9/8.39e9 env-steps/s. Beats torch compile (248 us @1M) 1.7x in like-for-like jit mode. | baseline recorded |

## Notes

- The scan benchmark stores only a reward sum per step, not per-step observations, so it is
  an upper bound; the e2e trainer's rollout must write observations and will land between
  the jit and scan curves.
