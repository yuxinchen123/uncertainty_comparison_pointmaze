"""The JAX exploration platform: PointMaze environments, a PPO agent, and swappable bonuses.

Double precision is switched on here rather than in one module because it is a property of the
whole platform: the running mean/variance accumulators are float64 (algorithm specification 4.2)
and jax has to be told before the first array is made. Everything else is explicitly float32,
which the F32 constant below names once for every module.
"""
import jax

jax.config.update("jax_enable_x64", True)   # float64 running statistics (spec 4.2)

import jax.numpy as jnp  # noqa: E402

F32 = jnp.float32
LOG2PI = 1.8378770664093453
