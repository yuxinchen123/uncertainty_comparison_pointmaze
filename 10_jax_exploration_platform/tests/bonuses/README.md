# bonuses

Checks that belong to a bonus family rather than to the agent.

| file | what it checks |
|---|---|
| `test_registry.py` | every preset name builds a family, `07_reconstruction`'s older names select the same families, an unknown name is refused with the available names in the message, and the runner ends up holding the family its name asked for |
| `test_none_has_no_bonus_arithmetic.py` | selecting no bonus removes the bonus's arithmetic from the compiled program, not merely its effect — and the arm still trains, with an intrinsic reward of exactly zero |
| `test_visit_count.py` | the fused visit-count implementation and its plain-python reference agree on the same supplied trajectories: the count table exactly, the bonus to 1e-6, including wall cells, positions outside the world, velocities past the clip, and repeat visits |
| `test_visit_count_learning_sanity.py` | on a short run of the composed program: a heavily visited state scores below an unvisited one, the mean intrinsic reward falls as the table fills, and the copies keep reaching new cells |

Run one, for example:

```
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu /p/rlprojects/RND/.venvs/platform_jax/bin/python \
  /p/rlprojects/RND/10_jax_exploration_platform/tests/bonuses/test_registry.py
```
