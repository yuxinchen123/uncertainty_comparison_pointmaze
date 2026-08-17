# integration

Tests that run a whole iteration or a whole short training and check the result end to end.

| file | what it checks |
|---|---|
| `test_jit_eager_equivalence.py` | one iteration of every registered bonus gives the same answer compiled and uncompiled: integer state exactly equal, floating-point state within the mixed rule `abs(a - b) <= 1e-6 + 1e-5*abs(b)` |

Compiling is an optimisation, so switching it off must not change the answer. It is not required
to give the same bits: without compilation there is no fusion, so a sum accumulates in a different
order. Integer state — a visit-count table, the optimizer's step counter — has no such excuse.

Run:

```
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu /p/rlprojects/RND/.venvs/platform_jax/bin/python \
  /p/rlprojects/RND/10_jax_exploration_platform/tests/integration/test_jit_eager_equivalence.py
```
