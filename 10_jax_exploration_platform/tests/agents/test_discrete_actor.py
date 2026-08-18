"""The discrete actor: Gumbel-argmax samples the categorical the logits define, and the
log-probability is the log-softmax gather.

Run: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <jax python> test_discrete_actor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
import exploration_platform  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from exploration_platform.agents.ppo.networks import (  # noqa: E402
    actor_logits, init_agent_params_discrete, logprob_discrete)


def test_gumbel_argmax_matches_softmax():
    """Sampling frequencies over many Gumbel draws match the softmax probabilities."""
    logits = jnp.asarray([[[1.0, 0.0, -1.0, 0.5]]])  # [C=1, M=1, A=4]
    n = 200_000
    u = jax.random.uniform(jax.random.PRNGKey(0), (n, 4), jnp.float32,
                           minval=1e-7, maxval=1.0 - 1e-7)
    g = -jnp.log(-jnp.log(u))
    counts = np.bincount(np.asarray(jnp.argmax(logits[0, 0] + g, axis=-1)), minlength=4) / n
    want = np.asarray(jax.nn.softmax(logits[0, 0]))
    assert np.abs(counts - want).max() < 5e-3, (counts, want)
    print("ok test_gumbel_argmax_matches_softmax")


def test_logprob_is_log_softmax_gather():
    """logprob_discrete equals log-softmax indexed at the action, for a real network."""
    params = init_agent_params_discrete(2, 0, [0, 1], obs_dim=7, n_actions=5)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 7), jnp.float32)
    logits = actor_logits(params["actor"], x)
    acts = jnp.asarray([[0, 4, 2], [1, 1, 3]], jnp.int32)
    got = logprob_discrete(logits, acts)
    want = np.take_along_axis(np.asarray(jax.nn.log_softmax(logits, axis=-1)),
                              np.asarray(acts)[..., None], axis=-1)[..., 0]
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-6)
    assert "logstd" not in params["actor"]
    print("ok test_logprob_is_log_softmax_gather")


if __name__ == "__main__":
    test_gumbel_argmax_matches_softmax()
    test_logprob_is_log_softmax_gather()
