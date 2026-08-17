# agents

The agent's own tests, brought over from `09_parallelization/ppo/jax_ppo/tests/` and pointed at
the split package under `src/exploration_platform/`.

| file | what it checks | runs here? |
|---|---|---|
| `test_jax_ppo.py` | per-part behaviour: same-seed determinism, copy isolation (agent AND bonus parameters), both update styles, the distillation error falling | yes, on the processor |
| `test_sweep_jax.py` | the sweep: equal rates reproduce a uniform run, a zero-rate group never moves, changing one group leaves the others bit-identical, paired and distinct seeding | yes, on the processor |
| `test_hoist_equivalence.py` | computing the critic values, the log-probability and the bonus after the rollout gives the same numbers as computing them inside it | yes |
| `test_flat_params_equivalence.py` | holding all parameters in one array computes the same function as holding the named arrays (checked in double precision) | yes |
| `audit_float64.py` | which quantities are held in double precision and which in single | yes |
| `test_forward_fixture.py` | the JAX forward passes reproduce the PyTorch trainer's outputs | **no** — it needs a fixture file written by the PyTorch trainer, which was not copied into this folder (see `../../ORIGIN.md`) |

Run one with the platform environment, for example:

```
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu /p/rlprojects/RND/.venvs/platform_jax/bin/python \
  /p/rlprojects/RND/10_jax_exploration_platform/tests/agents/test_jax_ppo.py
```

## The two checks that were rewritten for the zero-noise environment

Both used to end on a failed assertion, and both failed in exactly the same place when the
originals were run from `09_parallelization` with the same interpreter (checked 2026-08-16
16:45 PT) — the copy did not break them. They were rewritten on 2026-08-16 to assert what they
mean now that the environment's position noise is zero; the reason is written into each test's own
docstring.

| test | what it used to assert | what it asserts now |
|---|---|---|
| `test_sweep_jax.py::test_distinct_seeding_separates_every_copy` | the observation right after a reset differs between differently-seeded copies | the environment starts every copy at the same point (stated as its own assertion, since it is now true by construction), and the seeding shows in the POLICY: every copy's action at that one shared starting observation differs, pairwise, and paired seeding makes the paired copies' actions identical |
| `test_hoist_equivalence.py::test_isolation` | the observations, actions and whitened bonus input are bit-identical between the hoisted and in-scan rollouts | the log-probability — the one stored field that is a pure function of the shared action noise and the shared log standard deviation — is bit-identical, the first rollout step's observation and action are bit-identical (both forms still act from a byte-identical state there), and the rest is bounded by the float32 reassociation the two loop shapes cause |

The second rewrite followed a measurement rather than a guess: re-run against the frozen
`09_parallelization` baseline with the same interpreter, the deviation is 4.968e-08 relative on
the observations there too. Moving the critic out of the rollout scan leaves the compiler a
different loop to fuse, the sampled action moves in its last float32 bit, and the environment
carries that into every later observation. That is reassociation, not the zero-noise environment
and not anything the split changed — so the exact requirement was moved to the fields where
exactness is actually provable.
