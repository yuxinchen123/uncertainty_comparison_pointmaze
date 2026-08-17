# copy_isolation

Tests that one copy's numbers never depend on another copy's.

| file | what it checks |
|---|---|
| `test_bonus_copy_isolation.py` | with every registered bonus and in both update styles: perturbing copy 2's weights leaves copies 0, 1 and 3 bit-identical while copy 2 changes — across the agent's parameters, the bonus's parameters, the bonus's state and the agent's state; and, for the visit-count family, that no two copies end with the same count table |

Every new bonus family joins the loop in that file by being registered — there is nothing to add
by hand. That is the point: a scatter that indexes a table without the copy axis, or a statistic
reduced over the wrong axis, would join the copies together silently and leave every
parameter-only test still passing.

Run:

```
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu /p/rlprojects/RND/.venvs/platform_jax/bin/python \
  /p/rlprojects/RND/10_jax_exploration_platform/tests/copy_isolation/test_bonus_copy_isolation.py
```
