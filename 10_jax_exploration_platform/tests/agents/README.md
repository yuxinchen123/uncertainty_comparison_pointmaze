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

## Two checks that already fail in the frozen baseline

`test_sweep_jax.py` and `test_hoist_equivalence.py` each end on a failed assertion, and they fail in
exactly the same place when the originals are run from `09_parallelization` with the same
interpreter (checked 2026-08-16 16:45 PT). The copy did not break them:

| test | the failing check | why it fails |
|---|---|---|
| `test_sweep_jax.py` | `test_distinct_seeding_separates_every_copy`: `assert not (obs[:3] == obs[3:]).all()` | the environment's position noise was set to zero on 2026-08-16, so every episode now begins at exactly the same point. The check reads the observation right after a reset and expects two differently-seeded copies to differ there — with no noise they cannot, whatever the seed |
| `test_hoist_equivalence.py` | `test_isolation`: `assert d == 0.0` for the key `obs` | same cause: the check compares observations that are now identical by construction, and the comparison it does treats that as the hoist having changed them |

Every other check in both files passes. The fix belongs to the stage that splits the trainer: give
those two checks a configuration with a non-zero position noise, so they measure seeding and hoisting
rather than the environment's reset noise. `09_parallelization` is frozen and is not edited for this.

Run one with the platform environment, for example:

```
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu /p/rlprojects/RND/.venvs/platform_jax/bin/python \
  /p/rlprojects/RND/10_jax_exploration_platform/tests/agents/test_jax_ppo.py
```
