# bonuses

Checks that belong to a bonus family rather than to the agent.

| file | what it checks |
|---|---|
| `test_registry.py` | every preset name builds a family, `07_reconstruction`'s older names select the same families, an unknown name is refused with the available names in the message, and the runner ends up holding the family its name asked for |
| `test_none_has_no_bonus_arithmetic.py` | selecting no bonus removes the bonus's arithmetic from the compiled program, not merely its effect — and the arm still trains, with an intrinsic reward of exactly zero |

Run one, for example:

```
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu /p/rlprojects/RND/.venvs/platform_jax/bin/python \
  /p/rlprojects/RND/10_jax_exploration_platform/tests/bonuses/test_registry.py
```
