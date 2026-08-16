# agents

The trainer's own tests, copied from `09_parallelization/ppo/jax_ppo/tests/` with their import paths
pointed at `src/exploration_platform/agents/ppo/`.

| file | what it checks | runs here? |
|---|---|---|
| `test_jax_ppo.py` | the trainer's per-part behaviour: statistics, generalized advantage, both update styles, priming | yes, on the processor |
| `test_sweep_jax.py` | the learning-rate sweep: equal rates reproduce a uniform run, a zero-rate group never moves, changing one group leaves the others bit-identical, paired seeding pairs | yes, on the processor |
| `test_hoist_equivalence.py` | computing the critic values, the log-probability and the bonus after the rollout gives the same numbers as computing them inside it | yes |
| `test_flat_params_equivalence.py` | holding all parameters in one array computes the same function as holding twenty-one named arrays (checked in double precision) | yes |
| `audit_float64.py` | which quantities are held in double precision and which in single | yes |
| `test_forward_fixture.py` | the JAX forward passes reproduce the PyTorch trainer's outputs | **no** — it needs a fixture file written by the PyTorch trainer, which was not copied into this folder (see `../../ORIGIN.md`) |

Run one with the platform environment, for example:

```
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu /p/rlprojects/RND/.venvs/platform_jax/bin/python \
  /p/rlprojects/RND/10_jax_exploration_platform/tests/agents/test_jax_ppo.py
```
