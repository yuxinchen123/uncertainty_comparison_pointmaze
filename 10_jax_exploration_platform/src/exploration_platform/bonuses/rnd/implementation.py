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
learns inside the rollout (for the whole run, or per policy until its first arrival at the
goal), and an output layer fitted in closed form on a body that is never
trained (`last_layer.py` holds the two fits and says why they run in float64); the bound of the
input clip, input statistics that stop moving after the priming rollouts, and random Fourier
features of the whitened input, read either by both networks or by the output layer fitted in
closed form. With every option at its default the functions below trace exactly the operations
they always did.
"""
import jax
import jax.numpy as jnp

from ... import F32
from ...statistics import rms_init, rms_update
from ..protocol import BonusFunctions
from .config import PREDICTOR_RATE_UNIT, RNDConfig, resolved_rnd_config
from .last_layer import (by_step, fast_layer_scores, flat_rollout, scheduled_layer_scores,
                         unit_rows_with_bias)
from .networks import (fourier_features, init_fourier_features, init_predictor, init_target,
                       predictor_features, predictor_hidden, target_features, unit_length, whiten)


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
    until_arrival = rnd_cfg.fast_last_layer_until_first_arrival
    frozen_statistics = rnd_cfg.input_statistics == "frozen_after_priming"
    fourier_input = rnd_cfg.input_features == "random_fourier"
    fourier_layer = rnd_cfg.scheduled_last_layer_features == "random_fourier"
    steps, envs = cfg.num_steps, cfg.n_envs
    if frozen_statistics and cfg.prime_iterations < 1:
        raise ValueError("input statistics frozen after priming need at least one priming "
                         "rollout; prime_iterations is 0")

    # the fixed networks are constants of the program, not parameters: they are drawn once here
    # (or taken from the checkpoint) and never appear in any gradient. The scheduled last layer
    # trains nothing, so what its output layer reads is such a constant too: the predictor's
    # body (its two hidden layers), or the random Fourier features in its place. Both networks
    # are as wide at their input as what they read.
    input_dim = rnd_cfg.fourier_feature_count if fourier_input else obs_dim
    constants = {"target": init_target(rnd_cfg, n_copies, base_seed, copy_seed_index, input_dim)}
    if scheduled and not fourier_layer:
        drawn = init_predictor(rnd_cfg, n_copies, base_seed, copy_seed_index, input_dim)
        constants["predictor_body"] = {name: drawn[name] for name in ("W0", "b0", "W1", "b1")}
    if fourier_input or fourier_layer:
        constants["fourier"] = init_fourier_features(rnd_cfg, n_copies, base_seed,
                                                     copy_seed_index, obs_dim)
    if saved_constants is not None:
        saved = {name: dict(tree) for name, tree in saved_constants.items()}
        shapes = jax.tree.map(lambda value: tuple(value.shape), constants)
        if jax.tree.map(lambda value: tuple(value.shape), saved) != shapes:
            raise ValueError("the saved fixed networks of the distillation bonus do not have "
                             f"this run's shapes {shapes}")
        constants = jax.tree.map(lambda value: jnp.asarray(value, value.dtype), saved)
    target = constants["target"]

    def network_input(obs, obs_rms):
        """What the target and the predictor read for [C, M, obs] observations: the whitened
        observation [C, M, obs], or its random Fourier features [C, M, D]."""
        x = whiten(obs, obs_rms, rnd_cfg.input_clip)
        return fourier_features(constants["fourier"], x) if fourier_input else x

    def statistics_after(state, next_obs_flat):
        """The input statistics after a training rollout: moved on by it, or, frozen after
        priming, exactly as they stood."""
        if frozen_statistics:
            return state["obs_rms"]
        return rms_update(state["obs_rms"], next_obs_flat, cfg.batch_stats_f32)

    def target_of(x):
        """The target's output for its inputs; scaled to unit length under an even start."""
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
        output layer is state, starting at zero: [features + 1, outputs] (weights, then bias) on
        the body's features, [D, outputs] on the random Fourier features.
        """
        state = {"obs_rms": rms_init(n_copies, obs_dim)}
        if scheduled:
            layer_rows = (rnd_cfg.fourier_feature_count if fourier_layer
                          else rnd_cfg.feature_dim + 1)
            state["output_layer"] = jnp.zeros((n_copies, layer_rows, rnd_cfg.feature_dim),
                                              jnp.float64)
            return {}, state
        predictor = init_predictor(rnd_cfg, n_copies, base_seed, copy_seed_index, input_dim)
        if rnd_cfg.even_start:
            predictor = {**predictor, "W2": jnp.zeros_like(predictor["W2"])}
        return {"predictor": predictor}, state

    def score(params, state, next_obs):
        """Intrinsic reward of [C, M, 4] next observations -> [C, M]."""
        x = network_input(next_obs, state["obs_rms"])
        tf = target_of(x)
        return bonus_of_error(predictor_features(params["predictor"], x) - tf)

    def fast_layer_inputs(params, state, obs):
        """What the fast last layer reads for [C, M, obs] observations, all float64: the
        unit-length last-hidden features with the bias input [C, M, F + 1], their lengths
        [C, M, 1], and the target [C, M, K]. The networks run in float32 and are promoted."""
        x = network_input(obs, state["obs_rms"])
        rows, length = unit_rows_with_bias(predictor_hidden(params["predictor"], x))
        return rows, length, target_of(x).astype(jnp.float64)

    def scheduled_layer_inputs(state, obs):
        """What the scheduled last layer reads for [C, M, obs] observations, both float64: the
        unit-length feature rows [C, M, F + 1] (the body's features with the bias input) or
        [C, M, D] (random Fourier features), and the unit-length target [C, M, K]. The features
        and the target network run in float32 and are promoted."""
        x = whiten(obs, state["obs_rms"], rnd_cfg.input_clip)
        if fourier_layer:
            # no bias input here: it would be half of every row's squared length, so each visit
            # would move the error of every input, near or far, by about half as much as the
            # visited input's
            rows = unit_length(fourier_features(constants["fourier"], x).astype(jnp.float64))
        else:
            rows, _ = unit_rows_with_bias(predictor_hidden(constants["predictor_body"], x))
        # the schedule starts from a squared error of exactly 1, so the target is scaled to
        # unit length after the promotion
        return rows, unit_length(target_features(target, x).astype(jnp.float64))

    def fast_scores(params, state, next_obs_flat):
        """Reward a whole rollout [C, T*N, obs] in time order with a temporary output layer that
        takes a step at every scored state; the predictor itself is not changed. -> [C, T*N]."""
        rows, length, tf = fast_layer_inputs(params, state, next_obs_flat)
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
        rows, tf = scheduled_layer_inputs(state, obs)
        error = jnp.matmul(rows, state["output_layer"]) - tf
        return (error * error).sum(-1).astype(F32)

    def prime(params, state, next_obs_flat):
        """Warm-up: fold a rollout of random actions into the input statistics, nothing else.
        Statistics frozen after priming are moved here and nowhere else."""
        return {**state,
                "obs_rms": rms_update(state["obs_rms"], next_obs_flat, cfg.batch_stats_f32)}

    def post_rollout(params, state, next_obs_flat, scored, arrived=None):
        """Reward the rollout, then move the statistics on and whiten the update batch's input.

        The reward uses the statistics as they stood at the START of the iteration — the same
        whitening the agent's actions were taken under — while the batch the update will see is
        whitened with the statistics that now include this rollout. Statistics frozen after
        priming are the same at both points.

        arrived, [C] booleans, is read only by the fast last layer that stops at the first
        arrival: a copy that had reached the goal before this rollout is scored by the predictor
        alone. The temporary layer is rebuilt from the predictor at every rollout and dropped
        after it, so nothing of it is left to undo when a copy stops using it.
        """
        if scheduled:
            # the kept output layer is read and fitted step by step; there is no update batch,
            # because nothing is trained by gradient
            rows, tf = scheduled_layer_inputs(state, next_obs_flat)
            layer, squared = scheduled_layer_scores(
                by_step(rows, steps, envs), by_step(tf, steps, envs), state["output_layer"],
                rnd_cfg.scheduled_last_layer_decay)
            obs_rms = statistics_after(state, next_obs_flat)
            return ({"obs_rms": obs_rms, "output_layer": layer},
                    flat_rollout(squared).astype(F32), {})
        if scored is not None:
            reward = scored
        elif fast:
            reward = fast_scores(params, state, next_obs_flat)
            if until_arrival:
                if arrived is None:
                    raise ValueError("fast_last_layer_until_first_arrival needs each copy's "
                                     "first arrival, which this caller of post_rollout did not "
                                     "hand over")
                # before: arrived [False, True]; fast scores and the predictor's own, [2, T*N]
                # after:  copy 0 keeps its fast scores, copy 1 takes the predictor's own scores
                reward = jnp.where(arrived[:, None], score(params, state, next_obs_flat), reward)
        else:
            reward = score(params, state, next_obs_flat)
        obs_rms = statistics_after(state, next_obs_flat)
        extra = {"rnd_input": network_input(next_obs_flat, obs_rms)}
        # the target never changes, so its features of the update batch can be computed here,
        # once, and travel with the batch: [C, T*N, feature_dim] beside the [C, T*N, obs] input
        # ([C, T*N, D] with random Fourier features)
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
                          reward_divisor=rnd_cfg.reward_divisor,
                          divisor_hold_fraction=(
                              rnd_cfg.divisor_hold_fraction
                              if rnd_cfg.reward_divisor == "held_after_fraction" else None),
                          own_learning_rate=own_rate,
                          record_on_maze_grid=rnd_cfg.record_bonus_on_maze_grid,
                          reads_first_arrival=until_arrival)
