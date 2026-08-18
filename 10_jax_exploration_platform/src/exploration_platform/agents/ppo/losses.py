"""The agent's own contribution to the loss, kept per copy so a bonus can be added to it.

Every term reduces over the batch axis and keeps the copy axis, so the returned [C] vector is
"copy c's loss". The composer adds the bonus's [C] vector to it and sums once — the same
arithmetic, in the same order, that the single-module trainer performed.
"""
import jax
import jax.numpy as jnp

from ... import LOG2PI
from .networks import actor_logits, actor_mean, critic_values, logprob, logprob_discrete


def ppo_loss_per_copy(cfg, agent_params, batch, style_a: bool):
    """Per-copy policy + value + entropy loss on one (mini)batch (spec 9 + the 11 correction).

    style_a is a Python constant chosen when the program is built, never a traced branch: style A
    (one update on the whole batch) keeps the probability ratio and drops the clip machinery,
    style B (epochs of minibatches) is the clipped objective.
    """
    # advantages are standardised within each copy's own batch, so a copy's scale never leaks
    # into another copy's gradient
    a = batch["adv"]
    a_n = (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, ddof=1, keepdims=True) + 1e-8)

    # one forward for the policy and one for both value heads
    mean = actor_mean(agent_params["actor"], batch["obs"])
    logstd = agent_params["actor"]["logstd"]
    newlogp = logprob(mean, logstd, batch["actions"])
    vext, vint = critic_values(agent_params["critic"], batch["obs"])
    ratio = jnp.exp(newlogp - batch["old_logprob"])

    # the policy-gradient term and the extrinsic value term differ between the two styles
    if style_a:
        pg = (-a_n * ratio).mean(axis=1)
        v_ext = 0.5 * ((vext - batch["ret_ext"]) ** 2).mean(axis=1)
    else:
        pg = jnp.maximum(-a_n * ratio,
                         -a_n * jnp.clip(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                         ).mean(axis=1)
        vc = batch["vext_old"] + jnp.clip(vext - batch["vext_old"],
                                          -cfg.clip_coef, cfg.clip_coef)
        v_ext = 0.5 * jnp.maximum((vext - batch["ret_ext"]) ** 2,
                                  (vc - batch["ret_ext"]) ** 2).mean(axis=1)
    v_int = 0.5 * ((vint - batch["ret_int"]) ** 2).mean(axis=1)

    # a diagonal Gaussian's entropy depends only on the log standard deviation
    ent = (0.5 + 0.5 * LOG2PI + logstd).sum(axis=1)
    return pg - cfg.ent_coef * ent + cfg.vf_coef * (v_ext + v_int)


def ppo_loss_per_copy_discrete(cfg, agent_params, batch, style_a: bool):
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
                         -a_n * jnp.clip(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                         ).mean(axis=1)
        vc = batch["vext_old"] + jnp.clip(vext - batch["vext_old"],
                                          -cfg.clip_coef, cfg.clip_coef)
        v_ext = 0.5 * jnp.maximum((vext - batch["ret_ext"]) ** 2,
                                  (vc - batch["ret_ext"]) ** 2).mean(axis=1)
    v_int = 0.5 * ((vint - batch["ret_int"]) ** 2).mean(axis=1)

    # categorical entropy from the same logits, averaged over the batch
    logp_all = jax.nn.log_softmax(logits, axis=-1)
    ent = (-(jnp.exp(logp_all) * logp_all).sum(-1)).mean(axis=1)
    return pg - cfg.ent_coef * ent + cfg.vf_coef * (v_ext + v_int)
