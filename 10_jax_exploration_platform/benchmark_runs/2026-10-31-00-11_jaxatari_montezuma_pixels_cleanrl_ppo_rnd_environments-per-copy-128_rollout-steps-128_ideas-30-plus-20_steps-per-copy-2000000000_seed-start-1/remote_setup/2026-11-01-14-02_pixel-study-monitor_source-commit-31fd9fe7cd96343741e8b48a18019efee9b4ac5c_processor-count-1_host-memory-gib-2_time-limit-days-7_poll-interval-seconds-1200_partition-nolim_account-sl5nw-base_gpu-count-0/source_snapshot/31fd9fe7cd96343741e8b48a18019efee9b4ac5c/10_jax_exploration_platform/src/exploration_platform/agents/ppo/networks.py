"""The agent's two networks: a diagonal-Gaussian actor and a critic with two value heads.

Every forward takes x of shape [C, M, in] — the copy axis leads, so one matmul is C independent
matmuls and M is whatever batch the caller has (one rollout step, or a whole rollout at once).
"""
import jax
import jax.numpy as jnp

from ... import F32, LOG2PI
from ...networks import stacked_orthogonal


def init_agent_params(n_copies: int, base_seed: int, copy_seed_index,
                      obs_dim: int = 4, act_dim: int = 2):
    """The actor and critic weights of every copy: {"actor": {...}, "critic": {...}}.

    obs_dim and act_dim come from the environment (PointMaze 4/2, AntMaze 29/8); the defaults
    keep every draw of the existing PointMaze runs bit-identical.
    """
    # the actor: two hidden layers of 64 and an act_dim-dimensional mean, plus a
    # state-independent log standard deviation that starts at zero
    actor = stacked_orthogonal(
        "actor", [(64, obs_dim, 2 ** 0.5), (64, 64, 2 ** 0.5), (act_dim, 64, 0.01)],
        n_copies, base_seed, copy_seed_index)
    actor["logstd"] = jnp.zeros((n_copies, act_dim), F32)

    # the critic: a shared trunk and two heads, one for the extrinsic and one for the intrinsic
    # return. The heads are drawn as one two-layer stack so their keys stay what they were.
    critic = stacked_orthogonal("critic", [(64, obs_dim, 2 ** 0.5), (64, 64, 2 ** 0.5)],
                                n_copies, base_seed, copy_seed_index)
    heads = stacked_orthogonal("critic_heads", [(1, 64, 1.0), (1, 64, 1.0)],
                               n_copies, base_seed, copy_seed_index)
    critic["Wext"], critic["bext"] = heads["W0"], heads["b0"]
    critic["Wint"], critic["bint"] = heads["W1"], heads["b1"]
    return {"actor": actor, "critic": critic}


def init_agent_params_discrete(n_copies: int, base_seed: int, copy_seed_index,
                               obs_dim: int, n_actions: int):
    """The discrete-actor variant: the same two hidden layers, a logits head, no logstd.

    Draws use the same frozen network ids as the Gaussian actor, so a discrete run's weights
    are keyed exactly like a continuous run's of the same shapes.
    """
    actor = stacked_orthogonal(
        "actor", [(64, obs_dim, 2 ** 0.5), (64, 64, 2 ** 0.5), (n_actions, 64, 0.01)],
        n_copies, base_seed, copy_seed_index)
    critic = stacked_orthogonal("critic", [(64, obs_dim, 2 ** 0.5), (64, 64, 2 ** 0.5)],
                                n_copies, base_seed, copy_seed_index)
    heads = stacked_orthogonal("critic_heads", [(1, 64, 1.0), (1, 64, 1.0)],
                               n_copies, base_seed, copy_seed_index)
    critic["Wext"], critic["bext"] = heads["W0"], heads["b0"]
    critic["Wint"], critic["bint"] = heads["W1"], heads["b1"]
    return {"actor": actor, "critic": critic}


def actor_logits(actor, x):
    """Action logits [C, M, n_actions] — the discrete actor's forward, same trunk shape."""
    h = jnp.tanh(jnp.matmul(x, actor["W0"]) + actor["b0"][:, None, :])
    h = jnp.tanh(jnp.matmul(h, actor["W1"]) + actor["b1"][:, None, :])
    return jnp.matmul(h, actor["W2"]) + actor["b2"][:, None, :]


def logprob_discrete(logits, action):
    """Log-probability of int32 actions [C, M] under the categorical logits -> [C, M]."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp, action[..., None].astype(jnp.int32), axis=-1)[..., 0]


def actor_mean(actor, x):
    """Action mean [C, M, 2]."""
    h = jnp.tanh(jnp.matmul(x, actor["W0"]) + actor["b0"][:, None, :])
    h = jnp.tanh(jnp.matmul(h, actor["W1"]) + actor["b1"][:, None, :])
    return jnp.matmul(h, actor["W2"]) + actor["b2"][:, None, :]


def critic_values(critic, x):
    """(Vext, Vint) each [C, M]."""
    h = jnp.tanh(jnp.matmul(x, critic["W0"]) + critic["b0"][:, None, :])
    h = jnp.tanh(jnp.matmul(h, critic["W1"]) + critic["b1"][:, None, :])
    vext = jnp.matmul(h, critic["Wext"]) + critic["bext"][:, None, :]
    vint = jnp.matmul(h, critic["Wint"]) + critic["bint"][:, None, :]
    return vext[..., 0], vint[..., 0]


def logprob(mean, logstd, action):
    """Diagonal-Gaussian log-density summed over action dimensions -> [C, M]."""
    z = (action - mean) / jnp.exp(logstd)[:, None, :]
    return (-0.5 * z * z - logstd[:, None, :] - 0.5 * LOG2PI).sum(-1)
