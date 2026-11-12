"""The agent's own contribution to the loss, kept per copy so a bonus can be added to it.

Every term reduces over the batch axis and keeps the copy axis, so the returned [C] vector is
"copy c's loss". The composer adds the bonus's [C] vector to it and sums once — the same
arithmetic, in the same order, that the single-module trainer performed.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ... import LOG2PI
from .networks import (actor_logits, actor_mean, actor_mean_and_critic_values, critic_values,
                       logprob, logprob_discrete)


class LossCoefficients(NamedTuple):
    """The loss's clip, entropy and value coefficients, each a Python number or one per copy.

    A coefficient shared by every copy is the configuration's own Python number, so the program
    holds the constant it always did. A per-copy coefficient is a float32 array: the clip bounds
    [C, 1] against [C, M] rows, the entropy and value coefficients [C] against per-copy losses.
    """
    clip_low: object     # 1 - clip coefficient, the lower bound of the probability ratio
    clip_high: object    # 1 + clip coefficient, the upper bound of the probability ratio
    clip_neg: object     # -clip coefficient, the lower bound of the extrinsic value change
    clip: object         # the clip coefficient, the upper bound of the extrinsic value change
    ent: object          # entropy coefficient
    vf: object           # value coefficient


def loss_coefficients(cfg, sweep=None) -> LossCoefficients:
    """Read the three coefficients from the configuration, or per copy from the sweep.

    The clip bounds are formed from Python numbers before any rounding to float32, which is how
    the scalar program forms them, so a per-copy vector of equal values gives the same bounds.

    before: clip_coef 0.1, no sweep        -> clip_low 0.9, clip_high 1.1 (Python floats)
    after:  clip_coef (0.1, 0.2) per copy  -> clip_low [[0.9], [0.8]], clip_high [[1.1], [1.2]]
    """
    def value(name, shape=None, transform=None):
        """One coefficient, optionally transformed, as a scalar or broadcast along the copies."""
        if sweep is None:
            return transform(getattr(cfg, name)) if transform else getattr(cfg, name)
        return sweep.setting(cfg, name, shape, transform)

    rows = (cfg.n_copies, 1)
    return LossCoefficients(
        clip_low=value("clip_coef", rows, lambda c: 1 - c),
        clip_high=value("clip_coef", rows, lambda c: 1 + c),
        clip_neg=value("clip_coef", rows, lambda c: -c),
        clip=value("clip_coef", rows),
        ent=value("ent_coef"),
        vf=value("vf_coef"))


def ppo_loss_per_copy(coef: LossCoefficients, agent_params, batch, style_a: bool,
                      joint_trunk: bool = False):
    """Per-copy policy + value + entropy loss on one (mini)batch (spec 9 + the 11 correction).

    style_a is a Python constant chosen when the program is built, never a traced branch: style A
    (one update on the whole batch) keeps the probability ratio and drops the clip machinery,
    style B (epochs of minibatches) is the clipped objective. joint_trunk, also chosen when the
    program is built, runs the actor's and the critic's hidden layers as one batched network.
    """
    # advantages are standardised within each copy's own batch, so a copy's scale never leaks
    # into another copy's gradient
    a = batch["adv"]
    a_n = (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, ddof=1, keepdims=True) + 1e-8)

    # one forward for the policy and one for both value heads, or one for all three
    logstd = agent_params["actor"]["logstd"]
    if joint_trunk:
        mean, vext, vint = actor_mean_and_critic_values(
            agent_params["actor"], agent_params["critic"], batch["obs"])
        newlogp = logprob(mean, logstd, batch["actions"])
    else:
        mean = actor_mean(agent_params["actor"], batch["obs"])
        newlogp = logprob(mean, logstd, batch["actions"])
        vext, vint = critic_values(agent_params["critic"], batch["obs"])
    ratio = jnp.exp(newlogp - batch["old_logprob"])

    # the policy-gradient term and the extrinsic value term differ between the two styles
    if style_a:
        pg = (-a_n * ratio).mean(axis=1)
        v_ext = 0.5 * ((vext - batch["ret_ext"]) ** 2).mean(axis=1)
    else:
        pg = jnp.maximum(-a_n * ratio,
                         -a_n * jnp.clip(ratio, coef.clip_low, coef.clip_high)
                         ).mean(axis=1)
        vc = batch["vext_old"] + jnp.clip(vext - batch["vext_old"],
                                          coef.clip_neg, coef.clip)
        v_ext = 0.5 * jnp.maximum((vext - batch["ret_ext"]) ** 2,
                                  (vc - batch["ret_ext"]) ** 2).mean(axis=1)
    v_int = 0.5 * ((vint - batch["ret_int"]) ** 2).mean(axis=1)

    # a diagonal Gaussian's entropy depends only on the log standard deviation
    ent = (0.5 + 0.5 * LOG2PI + logstd).sum(axis=1)
    return pg - coef.ent * ent + coef.vf * (v_ext + v_int)


def ppo_loss_per_copy_discrete(coef: LossCoefficients, agent_params, batch, style_a: bool):
    """The categorical-actor counterpart of `ppo_loss_per_copy`: the same objective with the
    log-probability, ratio and entropy taken from logits instead of a Gaussian."""
    a = batch["adv"]
    a_n = (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, ddof=1, keepdims=True) + 1e-8)

    logits = actor_logits(agent_params["actor"], batch["obs"])
    newlogp = logprob_discrete(logits, batch["actions"])
    vext, vint = critic_values(agent_params["critic"], batch["obs"])
    ratio = jnp.exp(newlogp - batch["old_logprob"])

    if style_a:
        pg = (-a_n * ratio).mean(axis=1)
        v_ext = 0.5 * ((vext - batch["ret_ext"]) ** 2).mean(axis=1)
    else:
        pg = jnp.maximum(-a_n * ratio,
                         -a_n * jnp.clip(ratio, coef.clip_low, coef.clip_high)
                         ).mean(axis=1)
        vc = batch["vext_old"] + jnp.clip(vext - batch["vext_old"],
                                          coef.clip_neg, coef.clip)
        v_ext = 0.5 * jnp.maximum((vext - batch["ret_ext"]) ** 2,
                                  (vc - batch["ret_ext"]) ** 2).mean(axis=1)
    v_int = 0.5 * ((vint - batch["ret_int"]) ** 2).mean(axis=1)

    # categorical entropy from the same logits, averaged over the batch
    logp_all = jax.nn.log_softmax(logits, axis=-1)
    ent = (-(jnp.exp(logp_all) * logp_all).sum(-1)).mean(axis=1)
    return pg - coef.ent * ent + coef.vf * (v_ext + v_int)
