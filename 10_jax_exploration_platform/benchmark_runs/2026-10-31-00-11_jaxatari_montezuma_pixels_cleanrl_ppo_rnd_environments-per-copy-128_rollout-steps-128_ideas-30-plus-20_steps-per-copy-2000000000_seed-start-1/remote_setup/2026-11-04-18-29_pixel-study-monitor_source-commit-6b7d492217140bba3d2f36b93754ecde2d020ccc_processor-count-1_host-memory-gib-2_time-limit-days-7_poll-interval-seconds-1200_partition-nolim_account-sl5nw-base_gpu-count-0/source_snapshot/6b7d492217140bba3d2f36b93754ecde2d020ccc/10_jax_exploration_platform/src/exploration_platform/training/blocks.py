"""Run several unchanged training iterations with one host dispatch."""
import jax
import jax.numpy as jnp


def iteration_block(runner, length):
    """Compile a fixed-length block, returning sums needed for episode-window records."""
    if length < 1:
        raise ValueError("block length must be positive")

    # Before: length host calls and length additions; after: one compiled loop and two sums.
    def block(state, rates):
        """Carry the exact training state and supplied learning-rate schedule through a block."""
        zeros = jnp.zeros((runner.cfg.n_copies,), jnp.float32)

        def body(index, carry):
            """Advance one iteration and accumulate its per-copy reward measurements."""
            current, reward, intrinsic, _ = carry
            current, metrics = runner.iterate(current, rates[index])
            return (current, reward + metrics["reward_ext_sum"],
                    intrinsic + metrics["rint_mean"], metrics["loss"])

        state, reward, intrinsic, loss = jax.lax.fori_loop(
            0, length, body, (state, zeros, zeros, jnp.zeros((), jnp.float32)))
        return state, {"reward_ext_sum": reward, "rint_mean": intrinsic, "loss": loss}

    return jax.jit(block, donate_argnums=(0,))
