"""Pieces of the reward-decline study (2026-10-19) that the iteration composes in when asked.

Each function is pure and per copy, so the iteration in `train_step.py` stays one program. None of
them is called by a configuration that leaves the study's switches at their defaults.
"""
import jax
import jax.numpy as jnp

from .. import F32
from ..agents.ppo.networks import actor_mean
from ..statistics import rms_init, rms_update


def tracker_init(n_copies: int, n_envs: int) -> dict:
    """An episode tracker at the start of every environment's episode: nothing seen yet.

    Every field is its own array: the iteration donates its input state, and one buffer passed
    as two donated fields is refused on the card.
    """
    zeros_i = lambda: jnp.zeros((n_copies, n_envs), jnp.int32)
    return {"reached": jnp.zeros((n_copies, n_envs), bool), "first_hit": zeros_i(),
            "inside": zeros_i(), "ret": jnp.zeros((n_copies, n_envs), F32), "length": zeros_i()}


def track_episodes(carry: dict, inside_buf, rext_buf, done_buf):
    """Walk one rollout forward and add up the episodes that ended inside it, per copy.

    inside_buf [T, C, N] bool: the step ended inside the goal radius. rext_buf [T, C, N]: the task
    reward the environment paid. done_buf [T, C, N]: the episode ended on that step.

    before: one environment, steps (inside, reward, done) = (0,0,0) (1,1,0) (1,1,1) (0,0,0)
    after:  one ended episode: reached 1, first step inside 2, steps inside 2, return 2,
            length 3; the carry holds the fourth step as the start of the next episode
    Returns (carry, sums) with sums a dict of [C] totals over the ended episodes:
    episodes, reached, first_hit (summed over episodes that reached), inside, ret, length.
    """
    def body(c, xs):
        """One step for every environment: extend the episode, then close it if it ended."""
        inside, reward, done = xs
        length = c["length"] + 1
        newly = inside & ~c["reached"]
        first_hit = jnp.where(newly, length, c["first_hit"])
        reached = c["reached"] | inside
        n_inside = c["inside"] + inside.astype(jnp.int32)
        ret = c["ret"] + reward

        # what an ending episode contributes; zero for an environment still running
        ended = {"episodes": done.astype(F32),
                 "reached": (done & reached).astype(F32),
                 "first_hit": jnp.where(done & reached, first_hit, 0).astype(F32),
                 "inside": jnp.where(done, n_inside, 0).astype(F32),
                 "ret": jnp.where(done, ret, 0.0),
                 "length": jnp.where(done, length, 0).astype(F32)}

        # an ended episode's environment starts a fresh one on the next step
        keep = ~done
        c = {"reached": reached & keep, "first_hit": jnp.where(keep, first_hit, 0),
             "inside": jnp.where(keep, n_inside, 0), "ret": jnp.where(keep, ret, 0.0),
             "length": jnp.where(keep, length, 0)}
        return c, ended

    carry, ended = jax.lax.scan(body, carry, (inside_buf, rext_buf, done_buf.astype(bool)))
    # before: each field [T, C, N]; after: [C], summed over the rollout and the environments
    return carry, {name: value.sum(axis=(0, 2)) for name, value in ended.items()}


def update_success_rate(rate, episodes, reached, step_size: float):
    """Move each copy's running goal-reaching rate toward this rollout's ended episodes.

    before: rate 0.2, 2 episodes ended, 1 reached, step 0.05 -> 0.2 + 0.05 x (0.5 - 0.2) = 0.215
    A copy that ended no episode in this rollout keeps its rate.
    """
    observed = reached / jnp.maximum(episodes, 1.0)
    return jnp.where(episodes > 0, rate + step_size * (observed - rate), rate)


def intrinsic_weight_factor(cfg, iteration, success_rate):
    """The [C] factor the intrinsic weight is multiplied by at this iteration, or None for 1.

    linear_decay: 1 - (k - 1) / total_iterations at iteration k (1 at the first iteration, 1/N
    at the last). success gating: 1 - the copy's running goal-reaching rate.
    """
    factor = None
    if cfg.int_coef_schedule == "linear_decay":
        factor = 1.0 - (iteration.astype(F32) - 1.0) / cfg.total_iterations
    if cfg.success_gate_rate > 0:
        gate = 1.0 - success_rate
        factor = gate if factor is None else factor * gate
    if factor is None:
        return None
    return jnp.broadcast_to(factor, (cfg.n_copies,)).astype(F32)


def normalize_task_reward(filter_state, rms, rext_buf, done_buf, gamma: float):
    """Divide the task reward by max(1, running std of its discounted sum), per copy.

    The discounted sum restarts after every episode end. The floor of 1 means the reward is only
    ever scaled down: while no copy has found the goal the sum is zero and its std near zero,
    and dividing by that would turn the first goal step into a reward of thousands.
    before: rewards of 0 and 1, discounted sums with std 25 -> rewards of 0 and 0.04
    Returns (filter_state, rms, normalized rewards [T, C, N], std [C]).
    """
    def body(f, xs):
        """One step of the discounted sum, reset after an episode ends."""
        r, done = xs
        f = gamma * f + r
        return f * (1.0 - done), f

    T, C, N = rext_buf.shape
    filter_state, sums = jax.lax.scan(body, filter_state, (rext_buf, done_buf))
    rms = rms_update(rms, sums.transpose(1, 0, 2).reshape(C, T * N, 1))
    std = jnp.maximum(jnp.sqrt(rms.var + 1e-8).astype(F32), 1.0).reshape(C)
    return filter_state, rms, rext_buf / std.reshape(1, C, 1), std


def ext_filter_init(n_copies: int, n_envs: int):
    """The task-reward normaliser at the start: zero discounted sums and fresh statistics."""
    return jnp.zeros((n_copies, n_envs), F32), rms_init(n_copies, 1)


def gaussian_kl(actor_old, actor_new, obs):
    """Mean over rows of KL(old policy || new policy) per copy, [C], for the diagonal Gaussian.

    The standard deviation is state independent, so per action dimension
    KL = log(s_new / s_old) + (s_old^2 + (mu_old - mu_new)^2) / (2 s_new^2) - 1/2.
    """
    mu_old, mu_new = actor_mean(actor_old, obs), actor_mean(actor_new, obs)
    ls_old = actor_old["logstd"][:, None, :]
    ls_new = actor_new["logstd"][:, None, :]
    per_dim = (ls_new - ls_old
               + (jnp.exp(2 * ls_old) + (mu_old - mu_new) ** 2) / (2 * jnp.exp(2 * ls_new)) - 0.5)
    return per_dim.sum(-1).mean(axis=1)


def explained_variance(values, returns):
    """1 - Var(returns - values) / Var(returns) over each copy's rows, [C]; 0 where Var is 0."""
    var_ret = returns.var(axis=1)
    ev = 1.0 - (returns - values).var(axis=1) / jnp.where(var_ret > 0, var_ret, 1.0)
    return jnp.where(var_ret > 0, ev, 0.0)
