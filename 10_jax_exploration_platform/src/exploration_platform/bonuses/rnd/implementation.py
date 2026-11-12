"""Random network distillation on the next observation, expressed through the bonus hooks.

The bonus of a state is half the squared distance between a fixed random target network's
features of that state and a trained predictor's. The predictor is trained on the states the
agent actually reached, so the distance shrinks where the agent has been and stays large where it
has not.

Trainable parameters: the predictor. Statistics: the running mean and variance of the network's
input, so a whitened input keeps the same scale as the agent moves through the maze. The
predictor's own error is also the loss the optimizer minimises, and it rides the agent's Adam —
one optimizer over {agent, bonus}, which is what the frozen 09_parallelization baseline did.

The options of `RNDConfig` change one part each, and every one is decided here, in Python, before
the program is compiled: an even start, the form of the bonus, a temporary output layer that
learns inside the rollout, and an output layer fitted in closed form on a body that is never
trained (`last_layer.py` holds the two fits and says why they run in float64). With every option
at its default the functions below trace exactly the operations they always did.
"""
import jax
import jax.numpy as jnp

from ... import F32
from ...statistics import rms_init, rms_update
from ..protocol import BonusFunctions
from .config import PREDICTOR_RATE_UNIT, RNDConfig, resolved_rnd_config
from .last_layer import (by_step, fast_layer_scores, flat_rollout, scheduled_layer_scores,
                         unit_rows_with_bias)
from .networks import (init_predictor, init_target, predictor_features, predictor_hidden,
                       target_features, unit_length, whiten)


def build(cfg, env_cfg, rnd_cfg: RNDConfig, n_copies: int, base_seed: int, copy_seed_index,
          obs_dim: int = 4, saved_constants=None):
    """Bind this bonus to one run's copy count, seeds and statistics setting.

    env_cfg is unused: this family reads the observation as obs_dim numbers and needs to know
    nothing else about the environment it came from. saved_constants, from a checkpoint, replaces
    the draw of the fixed networks, so a run resumed on another card model keeps the networks it
    started with to the last digit.
    """
    rnd_cfg = resolved_rnd_config(rnd_cfg)
    scheduled = rnd_cfg.scheduled_last_layer_decay > 0
    fast = rnd_cfg.fast_last_layer_step > 0
    steps, envs = cfg.num_steps, cfg.n_envs

    # the fixed networks are constants of the program, not parameters: they are drawn once here
    # (or taken from the checkpoint) and never appear in any gradient. The scheduled last layer
    # trains nothing, so the predictor's body (its two hidden layers) is such a constant too.
    constants = {"target": init_target(rnd_cfg, n_copies, base_seed, copy_seed_index, obs_dim)}
    if scheduled:
        drawn = init_predictor(rnd_cfg, n_copies, base_seed, copy_seed_index, obs_dim)
        constants["predictor_body"] = {name: drawn[name] for name in ("W0", "b0", "W1", "b1")}
    if saved_constants is not None:
        saved = {name: dict(tree) for name, tree in saved_constants.items()}
        shapes = jax.tree.map(lambda value: tuple(value.shape), constants)
        if jax.tree.map(lambda value: tuple(value.shape), saved) != shapes:
            raise ValueError("the saved fixed networks of the distillation bonus do not have "
                             f"this run's shapes {shapes}")
        constants = jax.tree.map(lambda value: jnp.asarray(value, value.dtype), saved)
    target = constants["target"]

    def target_of(x):
        """The target's output for whitened inputs; scaled to unit length under an even start."""
        tf = target_features(target, x)
        return unit_length(tf) if rnd_cfg.even_start else tf

    def bonus_of_error(error):
        """The bonus of error vectors [..., K]: half the squared length, or the length."""
        if rnd_cfg.bonus_form == "error_length":
            return jnp.sqrt((error ** 2).sum(-1))
        return 0.5 * (error ** 2).sum(-1)

    def init():
        """The predictor's weights, and fresh statistics for the observation dimensions.

        Even start: the predictor's output weights are zero (its bias already is), so its first
        output is zero at every input. Scheduled last layer: nothing is trainable, and the
        output layer [features + 1, outputs] (weights, then bias) is state, starting at zero.
        """
        state = {"obs_rms": rms_init(n_copies, obs_dim)}
        if scheduled:
            state["output_layer"] = jnp.zeros(
                (n_copies, rnd_cfg.feature_dim + 1, rnd_cfg.feature_dim), jnp.float64)
            return {}, state
        predictor = init_predictor(rnd_cfg, n_copies, base_seed, copy_seed_index, obs_dim)
        if rnd_cfg.even_start:
            predictor = {**predictor, "W2": jnp.zeros_like(predictor["W2"])}
        return {"predictor": predictor}, state

    def score(params, state, next_obs):
        """Intrinsic reward of [C, M, 4] next observations -> [C, M]."""
        x = whiten(next_obs, state["obs_rms"])
        tf = target_of(x)
        return bonus_of_error(predictor_features(params["predictor"], x) - tf)

    def last_layer_inputs(params, state, obs):
        """What a last-layer fit reads for [C, M, obs] observations, all float64: the unit-length
        last-hidden features with the bias input [C, M, F + 1], their lengths [C, M, 1], and the
        target [C, M, K]. The body and the target network run in float32 and are promoted."""
        x = whiten(obs, state["obs_rms"])
        body = constants["predictor_body"] if scheduled else params["predictor"]
        rows, length = unit_rows_with_bias(predictor_hidden(body, x))
        if scheduled:
            # the schedule starts from a squared error of exactly 1, so the target is scaled to
            # unit length after the promotion
            return rows, length, unit_length(target_features(target, x).astype(jnp.float64))
        return rows, length, target_of(x).astype(jnp.float64)

    def fast_scores(params, state, next_obs_flat):
        """Reward a whole rollout [C, T*N, obs] in time order with a temporary output layer that
        takes a step at every scored state; the predictor itself is not changed. -> [C, T*N]."""
        rows, length, tf = last_layer_inputs(params, state, next_obs_flat)
        predictor = params["predictor"]
        # before: weights [C, F, K] and bias [C, K]; after: one layer [C, F + 1, K], float64
        layer = jnp.concatenate([predictor["W2"], predictor["b2"][:, None, :]],
                                axis=1).astype(jnp.float64)
        bonus = fast_layer_scores(by_step(rows, steps, envs), by_step(length, steps, envs),
                                  by_step(tf, steps, envs), layer, rnd_cfg.fast_last_layer_step,
                                  bonus_of_error)
        return flat_rollout(bonus).astype(F32)

    def scheduled_score_alone(params, state, obs):
        """The squared error of [C, M, obs] observations under the kept output layer -> [C, M]."""
        rows, _, tf = last_layer_inputs(params, state, obs)
        error = jnp.matmul(rows, state["output_layer"]) - tf
        return (error * error).sum(-1).astype(F32)

    def prime(params, state, next_obs_flat):
        """Warm-up: fold a rollout of random actions into the input statistics, nothing else."""
        return {**state,
                "obs_rms": rms_update(state["obs_rms"], next_obs_flat, cfg.batch_stats_f32)}

    def post_rollout(params, state, next_obs_flat, scored):
        """Reward the rollout, then move the statistics on and whiten the update batch's input.

        The reward uses the statistics as they stood at the START of the iteration — the same
        whitening the agent's actions were taken under — while the batch the update will see is
        whitened with the statistics that now include this rollout.
        """
        if scheduled:
            # the kept output layer is read and fitted step by step; there is no update batch,
            # because nothing is trained by gradient
            rows, _, tf = last_layer_inputs(params, state, next_obs_flat)
            layer, squared = scheduled_layer_scores(
                by_step(rows, steps, envs), by_step(tf, steps, envs), state["output_layer"],
                rnd_cfg.scheduled_last_layer_decay)
            obs_rms = rms_update(state["obs_rms"], next_obs_flat, cfg.batch_stats_f32)
            return ({"obs_rms": obs_rms, "output_layer": layer},
                    flat_rollout(squared).astype(F32), {})
        if scored is not None:
            reward = scored
        elif fast:
            reward = fast_scores(params, state, next_obs_flat)
        else:
            reward = score(params, state, next_obs_flat)
        obs_rms = rms_update(state["obs_rms"], next_obs_flat, cfg.batch_stats_f32)
        extra = {"rnd_input": whiten(next_obs_flat, obs_rms)}
        # the target never changes, so its features of the update batch can be computed here,
        # once, and travel with the batch: [C, T*N, feature_dim] beside the [C, T*N, obs] input
        if cfg.target_features_once_per_iteration:
            extra["rnd_target"] = target_of(extra["rnd_input"])
        return {"obs_rms": obs_rms}, reward, extra

    def loss(params, batch):
        """Per-copy predictor error on the update batch, [C]; nothing for the scheduled layer."""
        if scheduled:
            return 0.0
        if cfg.target_features_once_per_iteration:
            tf = batch["rnd_target"]
        else:
            tf = target_of(batch["rnd_input"])
        pf = predictor_features(params["predictor"], batch["rnd_input"])
        return ((pf - tf) ** 2).mean(axis=2).mean(axis=1)

    def metrics(params, state):
        """This family reports nothing beyond the intrinsic reward the composer already logs."""
        return {}

    # a rollout scored with a last-layer fit depends on the order of its steps, so neither fit
    # can score one step at a time
    in_time_order = fast or scheduled
    own_rate = (rnd_cfg.predictor_learning_rate_multiple * PREDICTOR_RATE_UNIT
                if rnd_cfg.predictor_learning_rate_multiple > 0 else None)
    return BonusFunctions(name="rnd_next_state", init=init, prime=prime,
                          rollout_step=None if in_time_order else score,
                          post_rollout=post_rollout, loss=loss, metrics=metrics,
                          constants=constants,
                          score_alone=scheduled_score_alone if scheduled else score,
                          reward_divisor=rnd_cfg.reward_divisor, own_learning_rate=own_rate,
                          record_on_maze_grid=rnd_cfg.record_bonus_on_maze_grid)
