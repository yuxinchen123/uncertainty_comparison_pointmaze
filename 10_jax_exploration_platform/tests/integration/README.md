# integration

Tests that run a whole iteration or a whole short training and check the result end to end.

| file | what it checks |
|---|---|
| `test_jit_eager_equivalence.py` | one iteration of every registered bonus gives the same answer compiled and uncompiled: integer state exactly equal, floating-point state within the mixed rule `abs(a - b) <= 1e-6 + 1e-5*abs(b)` |
| `test_training_command_study_options.py` | the training command's options of 2026-11-12: CleanRL values, sweeps over more settings, the two reward settings, goal termination for AntMaze, the episode log's records, and a unit stopped twice and continued from its checkpoint against the same unit run in one go |
| `test_phase_profile_batch_shapes.py` | the phase profile's synthetic batch takes its widths from the environment |

Compiling is an optimisation, so switching it off must not change the answer. It is not required
to give the same bits: without compilation there is no fusion, so a sum accumulates in a different
order. Integer state — a visit-count table, the optimizer's step counter — has no such excuse.

Run:

```
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu /p/rlprojects/RND/.venvs/platform_jax/bin/python \
  /p/rlprojects/RND/10_jax_exploration_platform/tests/integration/test_jit_eager_equivalence.py
```
