"""CleanRL PPO with exact online counts and frozen state-dependent bonus fields."""

from typing import NamedTuple
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from .config import GridPPOConfig
from ..atari_pixels.networks import categorical_statistics, initialize_layer, quantity_keys
from ..atari_pixels.settings import scalar_setting, scalar_product
from ..atari_pixels.state import (AdamState, FlatParameterLayout, RunningStatistics, adam_step,
                                 initialize_adam, initialize_statistics, normalize_frames,
                                 update_observation_statistics)
from .online_rnd import jax_errors, jax_forward
from ..atari_pixels.training import normalize_intrinsic_rewards as cleanrl_normalize


class GridTrainState(NamedTuple):
    """Every mutable quantity belonging to an independent group of policies."""
    parameters: object
    optimizer: object
    positions: object
    episode_steps: object
    counts: object
    window_counts: object
    reached: object
    successes: object
    episodes: object
    first_goal_step: object
    maximum_progress: object
    blocked_moves: object
    reward_statistics: object
    reward_filter: object
    iteration: object
    policy_seeds: object
    # Online RND only (None for count bonuses): the frozen target and the input statistics.
    target: object = None
    observation_statistics: object = None
    # Online tabular-RND only: the [S,S] one-hot input table (None for coordinate inputs).
    rnd_table: object = None
    # Online RND with its own optimizer only: the predictor's optimizer state.
    predictor_optimizer: object = None
    # Online RND with an initial-ratio readout only: each cell's untrained readout [C,S].
    rnd_initial: object = None


class GridRollout(NamedTuple):
    """Time-major transitions retain terminal arrivals separately from reset states."""
    states: object
    arrivals: object
    arrival_counts: object
    actions: object
    log_probabilities: object
    values_ext: object
    values_int: object
    rewards: object
    intrinsic_rewards: object
    done: object
    succeeded: object


def gather_states(values, states, configuration):
    """Read only each policy's own state rows, preserving any trailing feature axis."""
    # Before: values [C,S,6], states [C,M]; after: selected rows [C,M,6].
    if configuration.flat_state_gather:
        offsets = jnp.arange(values.shape[0], dtype=jnp.int32).reshape(
            (values.shape[0],) + (1,) * (states.ndim - 1)) * values.shape[1]
        return values.reshape((-1,) + values.shape[2:])[states + offsets]
    return jax.vmap(lambda table, ids: table[ids])(values, states)


def network_outputs(parameters, states, coordinates, configuration):
    """Evaluate tabular outputs or the coordinate multilayer perceptron for selected states."""
    if configuration.policy_kind == "tabular":
        return gather_states(parameters["table"], states, configuration)
    # Preserve the CleanRL shared trunk, actor branch, and residual critic branch.
    values = coordinates[states]
    for layer in parameters["hidden"]:
        values = jax.nn.relu(jnp.matmul(values, layer["weight"]) + layer["bias"][:, None])
    actor = parameters["actor_hidden"]
    actor_features = jax.nn.relu(jnp.matmul(values, actor["weight"]) + actor["bias"][:, None])
    actor = parameters["actor_head"]
    logits = jnp.matmul(actor_features, actor["weight"]) + actor["bias"][:, None]
    extra = parameters["critic_extra"]
    critic_features = values + jax.nn.relu(jnp.matmul(values, extra["weight"]) + extra["bias"][:, None])
    critic = parameters["critic_head"]
    critics = jnp.matmul(critic_features, critic["weight"]) + critic["bias"][:, None]
    return jnp.concatenate((logits, critics), axis=-1)


def initialize(configuration, transition_table, coordinates, fields, policy_seeds, rnd=None):
    """Validate experiment inputs and initialize policy tables or coordinate networks."""
    # Validation happens on the host before tracing, including bounds and field finiteness.
    transitions, points, initial = map(np.asarray, (transition_table, coordinates, fields))
    seeds = np.asarray(policy_seeds, dtype=np.uint32)
    state_count = transitions.shape[0]
    if transitions.shape != (state_count, 4) or points.shape != (state_count, 2):
        raise ValueError("transition_table must be [states,4] and coordinates [states,2]")
    if initial.shape != (configuration.n_copies, state_count):
        raise ValueError("fields must have shape [independent copies,states]")
    if not np.isfinite(initial).all() or (initial < 0).any() or not np.isfinite(points).all():
        raise ValueError("coordinates must be finite and fields finite and nonnegative")
    if (transitions < 0).any() or (transitions >= state_count).any():
        raise ValueError("transition_table contains an invalid state")
    goal = state_count - 1 if configuration.goal_state == -1 else configuration.goal_state
    if not 0 <= configuration.start_state < state_count or not 0 <= goal < state_count:
        raise ValueError("start or goal is outside the state table")
    if goal == configuration.start_state:
        raise ValueError("start and goal must differ")
    if seeds.shape != (configuration.n_copies,):
        raise ValueError("policy_seeds must have one seed per independent copy")
    seeds = jnp.asarray(seeds)
    # Tabular policies and critics begin uniformly; neural conditions share the same seed.
    if configuration.policy_kind == "tabular":
        parameters = {"table": jnp.zeros((configuration.n_copies, state_count, 6), jnp.float32)}
    else:
        width = configuration.hidden_width
        parameters = {"hidden": (
            initialize_layer(seeds, "grid/policy/hidden/0", width, 2),
            initialize_layer(seeds, "grid/policy/hidden/1", width, width)),
            "actor_hidden": initialize_layer(seeds, "grid/policy/actor_hidden", width, width, gain=0.01),
            "actor_head": initialize_layer(seeds, "grid/policy/actor_head", 4, width, gain=0.01),
            "critic_extra": initialize_layer(seeds, "grid/policy/critic_extra", width, width, gain=0.1),
            "critic_head": initialize_layer(seeds, "grid/policy/critic_head", 2, width, gain=0.01)}
    # Online RND: the predictor joins the policy's parameters, so one Adam with one joint
    # gradient clip updates both, as in CleanRL; the target stays outside the optimizer.
    target = statistics = rnd_table = predictor_optimizer = rnd_initial = None
    if configuration.intrinsic_source == "online_rnd":
        if rnd is None:
            raise ValueError("online RND requires per-copy target and predictor networks")
        parameters = dict(parameters, predictor=jax.tree.map(jnp.asarray, rnd["predictor"]))
        target = jax.tree.map(jnp.asarray, rnd["target"])
        if configuration.normalize_rnd_inputs:
            statistics = RunningStatistics(jnp.asarray(rnd["statistics"]["mean"], jnp.float64),
                jnp.asarray(rnd["statistics"]["variance"], jnp.float64),
                jnp.asarray(rnd["statistics"]["count"], jnp.float64))
        if configuration.rnd_input == "one_hot":
            rnd_table = jnp.eye(state_count, dtype=jnp.float32)
        expected = state_count if configuration.rnd_input == "one_hot" else 2
        if parameters["predictor"][0]["weight"].shape[1] != expected:
            raise ValueError(f"the RND input layer must read {expected} values per cell")
    elif rnd is not None:
        raise ValueError("RND networks were supplied to a count-bonus configuration")
    # Before: no visits; after: the start is marked reached, with zero arrival counts.
    shape = (configuration.n_copies,)
    counts = jnp.zeros(shape + (state_count,), jnp.int32)
    reached = jnp.zeros_like(counts, dtype=jnp.bool_).at[:, configuration.start_state].set(True)
    filter_size = configuration.num_steps if configuration.upstream_reward_filter_axis else 1
    # A predictor with its own optimizer is left out of the policy's Adam and its joint clip.
    separate = configuration.intrinsic_source == "online_rnd" and configuration.rnd_optimizer != "shared_adam"
    policy_parameters = {name: value for name, value in parameters.items() if name != "predictor"} if separate else parameters
    if separate:
        zeros = jax.tree.map(jnp.zeros_like, parameters["predictor"])
        predictor_optimizer = AdamState(zeros if configuration.rnd_optimizer == "adam" else None,
            zeros if configuration.rnd_optimizer in {"adam", "adagrad"} else None, jnp.zeros((), jnp.int32))
    if configuration.intrinsic_source == "online_rnd" and configuration.rnd_readout.startswith("initial_ratio"):
        # Before: untrained networks; after: each cell's untrained readout, which later bonuses are divided by.
        table = rnd_table if rnd_table is not None else points
        ids = jnp.broadcast_to(jnp.arange(state_count, dtype=jnp.int32), (configuration.n_copies, state_count))
        inputs = jnp.asarray(table, jnp.float32)[ids]
        if statistics is not None:
            inputs = normalize_frames(inputs, statistics)
        rnd_initial = rnd_readout(parameters["predictor"], target, inputs, configuration)
    return GridTrainState(parameters, initialize_adam(policy_parameters),
        jnp.full(shape, configuration.start_state, jnp.int32), jnp.zeros(shape, jnp.int32),
        counts, jnp.zeros_like(counts), reached, jnp.zeros(shape, jnp.int32),
        jnp.zeros(shape, jnp.int32), jnp.full(shape, -1, jnp.int32),
        jnp.full(shape, configuration.start_state, jnp.int32),
        jnp.zeros(shape, jnp.int32),
        initialize_statistics(configuration.n_copies, ()),
        jnp.zeros(shape + (filter_size,), jnp.float32), jnp.zeros((), jnp.int32), seeds,
        target, statistics, rnd_table, predictor_optimizer, rnd_initial)


def rnd_inputs(state, coordinates, states):
    """Return the online RND inputs for selected states: coordinates (whitened when statistics exist) or one-hot rows."""
    # Before: state ids [C,M]; after: [C,M,2] coordinates, normalized and clipped if enabled, or [C,M,S] one-hot rows.
    values = (state.rnd_table if state.rnd_table is not None else coordinates)[states]
    if state.observation_statistics is not None:
        values = normalize_frames(values, state.observation_statistics)
    return values


def target_features(target, inputs, configuration):
    """Evaluate the frozen target, scaled to unit length per row when the configuration asks for it."""
    # Before: [C,M,512] features; after: the same rows divided by their own lengths.
    features = jax_forward(target, inputs)
    if configuration.rnd_unit_length_target:
        features = features / jnp.sqrt(jnp.sum(features ** 2, axis=-1, keepdims=True))
    return features


def rnd_readout(predictor, target, inputs, configuration):
    """Return the bonus before any initial-ratio division: half the summed squared error, or its square root."""
    if configuration.rnd_readout == "half_squared" and not configuration.rnd_unit_length_target:
        return jax_errors(predictor, target, inputs)
    squared = jnp.sum((jax_forward(predictor, inputs) - target_features(target, inputs, configuration)) ** 2, axis=-1)
    return jnp.sqrt(squared) if configuration.rnd_readout.endswith("length") else 0.5 * squared


def all_cell_errors(state, coordinates, configuration):
    """Evaluate the online RND bonus of every cell once; it is fixed during a rollout."""
    state_count = coordinates.shape[0]
    ids = jnp.broadcast_to(jnp.arange(state_count, dtype=jnp.int32), (configuration.n_copies, state_count))
    values = rnd_readout(state.parameters["predictor"], state.target, rnd_inputs(state, coordinates, ids), configuration)
    return values / state.rnd_initial if state.rnd_initial is not None else values


def count_bonus(field_values, visits, configuration):
    """Return field x visits^(-exponent); the exponent 0.5 keeps the original division by the square root."""
    # Before: field 1.0 and visits 4; after: 0.5 at exponent 0.5, 0.25 at exponent 1, 1/16 at exponent 2.
    visits = visits.astype(jnp.float32)
    if configuration.count_decay_exponent == 0.5:
        return field_values * jax.lax.rsqrt(visits) if configuration.rsqrt_bonus else field_values / jnp.sqrt(visits)
    return field_values * visits ** jnp.float32(-configuration.count_decay_exponent)


def collect_rollout(state, configuration, transition_table, coordinates, fields):
    """Update counts on every arrival before awarding its inverse-square-root bonus."""
    copies = configuration.n_copies
    rows = jnp.arange(copies, dtype=jnp.int32)
    goal = transition_table.shape[0] - 1 if configuration.goal_state == -1 else configuration.goal_state
    all_ids = jnp.broadcast_to(jnp.arange(transition_table.shape[0], dtype=jnp.int32), (copies, transition_table.shape[0]))
    cached = (network_outputs(state.parameters, all_ids, coordinates, configuration)
              if configuration.precompute_all_states else None)
    all_keys = (jax.vmap(lambda step: quantity_keys(state.policy_seeds, "grid/rollout/actions",
        state.iteration, step))(jnp.arange(configuration.num_steps, dtype=jnp.int32))
        if configuration.batch_random_keys else None)

    def step(carry, time):
        """Collect one transition, including goal arrivals and automatic episode reset."""
        position, elapsed, counts, reached, successes, episodes, first_goal, progress, blocked = carry
        outputs = (gather_states(cached, position[:, None], configuration)[:, 0]
            if cached is not None else network_outputs(
                state.parameters, position[:, None], coordinates, configuration)[:, 0])
        logits = outputs[:, :4]
        keys = (all_keys[time] if all_keys is not None else quantity_keys(
            state.policy_seeds, "grid/rollout/actions", state.iteration, time))
        actions = jax.vmap(lambda key, value: jax.random.categorical(key, value))(keys, logits).astype(jnp.int32)
        log_probability = (jnp.zeros((copies,), jnp.float32) if configuration.hoist_log_probabilities
                           else categorical_statistics(logits, actions)[0])
        # The arrived state is scored before replacing terminal observations with the start.
        arrived = transition_table[position, actions]
        counts = counts.at[rows, arrived].add(1)
        visits = counts[rows, arrived]
        if configuration.count_update != "per_step" or configuration.count_rate != 1.0 or configuration.count_mixture:
            # Diagnostic count forms: earlier visits m are those before this arrival, or before this rollout.
            earlier = (state.counts[rows, arrived] if configuration.count_update == "per_rollout" else visits - 1)
            scaled = 1.0 + jnp.float32(configuration.count_rate) * earlier.astype(jnp.float32)
            # Before: one exponent; after: the same, or a weighted sum of power laws of the scaled count.
            shape = (sum(jnp.float32(weight) * scaled ** jnp.float32(-exponent)
                         for weight, exponent in configuration.count_mixture) if configuration.count_mixture
                     else scaled ** jnp.float32(-configuration.count_decay_exponent))
            bonus = fields[rows, arrived] * shape
        else:
            bonus = fields[rows, arrived] if configuration.intrinsic_source == "online_rnd" else (
                jnp.zeros((copies,), jnp.float32) if configuration.hoist_bonus else
                count_bonus(fields[rows, arrived], visits, configuration))
        succeeded = arrived == goal
        elapsed = elapsed + 1
        done = succeeded | (elapsed >= configuration.episode_length)
        global_step = state.iteration * configuration.num_steps + time + 1
        first_goal = jnp.where((first_goal < 0) & succeeded, global_step, first_goal)
        reached = reached.at[rows, arrived].set(True)
        record = GridRollout(position, arrived, visits, actions, log_probability,
            jnp.zeros((copies,), jnp.float32) if configuration.hoist_values else outputs[:, 4],
            jnp.zeros((copies,), jnp.float32) if configuration.hoist_values else outputs[:, 5],
            succeeded.astype(jnp.float32), bonus, done, succeeded)
        return (jnp.where(done, configuration.start_state, arrived), jnp.where(done, 0, elapsed),
                counts, reached, successes + succeeded, episodes + done, first_goal,
                jnp.maximum(progress, arrived), blocked + (arrived == position)), record

    carry = (state.positions, state.episode_steps, state.counts, state.reached,
             state.successes, state.episodes, state.first_goal_step, state.maximum_progress, state.blocked_moves)
    carry, rollout = jax.lax.scan(step, carry, jnp.arange(configuration.num_steps, dtype=jnp.int32),
                                  unroll=configuration.rollout_unroll)
    position, elapsed, counts, reached, successes, episodes, first_goal, progress, blocked = carry
    # Before: rollout rows [T,C]; after: state ids [C,T] for one wide forward pass.
    ids = rollout.states.T
    if configuration.hoist_values or configuration.hoist_log_probabilities:
        outputs = (gather_states(cached, ids, configuration) if cached is not None
                   else network_outputs(state.parameters, ids, coordinates, configuration))
        if configuration.hoist_values:
            rollout = rollout._replace(values_ext=outputs[..., 4].T, values_int=outputs[..., 5].T)
        if configuration.hoist_log_probabilities:
            rollout = rollout._replace(log_probabilities=categorical_statistics(
                outputs[..., :4], rollout.actions.T)[0].T)
    if configuration.hoist_bonus:
        initial = gather_states(fields, rollout.arrivals.T, configuration).T
        rollout = rollout._replace(intrinsic_rewards=count_bonus(initial, rollout.arrival_counts, configuration))
    return state._replace(positions=position, episode_steps=elapsed, counts=counts,
        window_counts=counts - state.counts, reached=reached, successes=successes,
        episodes=episodes, first_goal_step=first_goal, maximum_progress=progress, blocked_moves=blocked), rollout


def generalized_advantages(rewards, intrinsic, rollout, final_values, configuration):
    """Stop task returns at done and continue intrinsic returns through reset observations."""
    # Shift values by one transition; the final values already observe the reset state.
    values_ext, values_int = final_values
    next_ext = jnp.concatenate((rollout.values_ext[1:], values_ext[None]))
    next_int = jnp.concatenate((rollout.values_int[1:], values_int[None]))
    mask = 1.0 - rollout.done.astype(jnp.float32)
    ext_difference = rewards + scalar_setting(configuration, "gamma_ext", (1, configuration.n_copies)) * next_ext * mask - rollout.values_ext
    int_difference = intrinsic + scalar_setting(configuration, "gamma_int", (1, configuration.n_copies)) * next_int - rollout.values_int
    ext_decay = scalar_product(configuration, ("gamma_ext", "gae_lambda"), (configuration.n_copies,))
    int_decay = scalar_product(configuration, ("gamma_int", "gae_lambda"), (configuration.n_copies,))
    if configuration.associative_gae:
        def compose(left, right):
            """Compose two adjacent affine return recurrences in reverse time."""
            a, b = left
            c, d = right
            return c + d * a, d * b

        ext, _ = jax.lax.associative_scan(compose, (ext_difference[::-1], (mask * ext_decay)[::-1]))
        intrinsic, _ = jax.lax.associative_scan(compose, (int_difference[::-1], jnp.broadcast_to(int_decay, int_difference.shape)[::-1]))
        return ext[::-1], intrinsic[::-1]

    def reverse(previous, inputs):
        """Accumulate one preceding pair of generalized advantages."""
        old_ext, old_int = previous
        difference_ext, difference_int, valid = inputs
        current = (difference_ext + ext_decay * valid * old_ext, difference_int + int_decay * old_int)
        return current, current

    zeros = jnp.zeros((configuration.n_copies,), jnp.float32)
    _, advantages = jax.lax.scan(reverse, (zeros, zeros),
        (ext_difference, int_difference, mask), reverse=True)
    return advantages


def prepare_batch(state, rollout, coordinates, configuration):
    """Build fixed PPO targets from raw count rewards or the explicit normalization sensitivity."""
    intrinsic = rollout.intrinsic_rewards
    if configuration.normalize_intrinsic_rewards:
        # Before: [T,C]; after: CleanRL's expected [T,C,one environment].
        intrinsic, statistics, filtered = cleanrl_normalize(intrinsic[..., None], state, configuration)
        intrinsic = intrinsic[..., 0]
        state = state._replace(reward_statistics=statistics, reward_filter=filtered)
    final = network_outputs(state.parameters, state.positions[:, None], coordinates, configuration)[:, 0]
    ext_advantage, int_advantage = generalized_advantages(rollout.rewards, intrinsic, rollout,
        (final[:, 4], final[:, 5]), configuration)
    batch = {"states": rollout.states.T, "actions": rollout.actions.T,
        "log_probabilities": rollout.log_probabilities.T, "values_ext": rollout.values_ext.T,
        "advantage_ext": ext_advantage.T, "advantage_int": int_advantage.T,
        "return_ext": (ext_advantage + rollout.values_ext).T,
        "return_int": (int_advantage + rollout.values_int).T}
    if configuration.intrinsic_source == "online_rnd":
        # CleanRL updates the input statistics with the rollout's states after GAE, then
        # trains the predictor on those states whitened with the updated statistics.
        if state.observation_statistics is not None:
            state = state._replace(observation_statistics=update_observation_statistics(
                state.observation_statistics, coordinates[batch["states"]], configuration.moment_reduction))
        batch["rnd_inputs"] = rnd_inputs(state, coordinates, batch["states"])
    return state, batch


def predictor_sample_losses(predictor, target, inputs, configuration):
    """Return each sample's predictor loss [C,M] under the configured loss and trainable layers."""
    if configuration.rnd_trainable == "input_layer":
        # Only the input weights train: with one-hot inputs each cell's row moves only when that cell is trained.
        # The input bias is shared by every cell, so it stays frozen with the deeper layers.
        predictor = ({"weight": predictor[0]["weight"], "bias": jax.lax.stop_gradient(predictor[0]["bias"])},) + tuple(
            jax.tree.map(jax.lax.stop_gradient, layer) for layer in predictor[1:])
    features = jax.lax.stop_gradient(target_features(target, inputs, configuration))
    outputs = jax_forward(predictor, inputs)
    if configuration.rnd_loss == "cleanrl":
        return ((outputs - features) ** 2).mean(axis=-1)
    if configuration.rnd_loss == "power":
        # (2/q) E^(q/2) of the summed squared error E: q = 2 is plain least squares.
        power = configuration.rnd_loss_power
        return (2.0 / power) * jnp.sum((outputs - features) ** 2, axis=-1) ** (power / 2.0)
    # Regress to the output moved toward the target so that the error length r becomes r (1 + r^p)^(-1/p);
    # p = 2 is r/sqrt(1+r^2). The step factor lies between 0 and 1 for every r, so the update cannot run away.
    residual = features - outputs
    squared = jnp.sum(residual ** 2, axis=-1, keepdims=True)
    power = configuration.rnd_shrink_power
    factor = (jax.lax.rsqrt(1.0 + squared) if power == 2.0 else
              (1.0 + squared ** (power / 2.0)) ** (-1.0 / power))
    moved = jax.lax.stop_gradient(outputs + (1.0 - factor) * residual)
    return 0.5 * jnp.sum((outputs - moved) ** 2, axis=-1)


def minibatch_loss(parameters, batch, coordinates, configuration, target=None):
    """Use CleanRL's clipped policy and task critic loss with an unclipped intrinsic critic."""
    if configuration.all_states_minibatch:
        ids = jnp.broadcast_to(jnp.arange(coordinates.shape[0], dtype=jnp.int32), (configuration.n_copies, coordinates.shape[0]))
        outputs = gather_states(network_outputs(parameters, ids, coordinates, configuration), batch["states"], configuration)
    else:
        outputs = network_outputs(parameters, batch["states"], coordinates, configuration)
    log_probability, entropy = categorical_statistics(outputs[..., :4], batch["actions"])
    ratio = jnp.exp(log_probability - batch["log_probabilities"])
    advantages = batch["advantage_ext"] * scalar_setting(configuration, "extrinsic_coefficient", (configuration.n_copies, 1))
    advantages += batch["advantage_int"] * scalar_setting(configuration, "intrinsic_coefficient", (configuration.n_copies, 1))
    advantages = (advantages - advantages.mean(axis=1, keepdims=True)) / (advantages.std(axis=1, ddof=1, keepdims=True) + 1e-8)
    clipping = scalar_setting(configuration, "clip_coefficient", (configuration.n_copies, 1))
    policy_loss = jnp.maximum(-advantages * ratio, -advantages * jnp.clip(ratio, 1 - clipping, 1 + clipping)).mean(axis=1)
    # Clip the task head only, matching the reference PPO/RND script.
    value_ext, value_int = outputs[..., 4], outputs[..., 5]
    clipped_value = batch["values_ext"] + jnp.clip(value_ext - batch["values_ext"], -clipping, clipping)
    ext_loss = 0.5 * jnp.maximum((value_ext - batch["return_ext"]) ** 2,
                                (clipped_value - batch["return_ext"]) ** 2).mean(axis=1)
    int_loss = 0.5 * ((value_int - batch["return_int"]) ** 2).mean(axis=1)
    entropy = entropy.mean(axis=1)
    total = policy_loss - scalar_setting(configuration, "entropy_coefficient") * entropy
    total += scalar_setting(configuration, "value_coefficient") * (ext_loss + int_loss)
    diagnostics = {"loss": total, "policy_loss": policy_loss, "value_loss": ext_loss + int_loss,
        "entropy": entropy, "approximate_kl": ((ratio - 1) - (log_probability - batch["log_probabilities"])).mean(axis=1)}
    if configuration.intrinsic_source == "online_rnd" and configuration.rnd_optimizer == "exact_step":
        # The exact-step optimizer computes its own direction and step, so the predictor adds nothing here.
        diagnostics.update(predictor_loss=jnp.zeros_like(total))
    elif configuration.intrinsic_source == "online_rnd":
        # CleanRL: mean squared feature error, averaged over the samples kept by the mask.
        errors = predictor_sample_losses(parameters["predictor"], target, batch["rnd_inputs"], configuration)
        mask = batch["predictor_mask"].astype(jnp.float32)
        predictor_loss = (errors * mask).sum(axis=1) / jnp.maximum(mask.sum(axis=1), 1.0)
        total = total + predictor_loss
        diagnostics.update(loss=total, predictor_loss=predictor_loss)
    return total.sum(), diagnostics


def update_policy(state, batch, coordinates, configuration):
    """Apply independently shuffled CleanRL epochs and per-policy Adam updates."""
    layout = FlatParameterLayout(state.parameters) if configuration.flat_parameters else None
    parameters = layout.pack(state.parameters) if layout else state.parameters
    optimizer = (AdamState(layout.pack(state.optimizer.first), layout.pack(state.optimizer.second), state.optimizer.step)
                 if layout else state.optimizer)
    rates = jnp.full((configuration.n_copies,), configuration.learning_rate, jnp.float32)
    explicit_rates = configuration.per_copy_values("learning_rate")
    if explicit_rates is not None:
        rates = jnp.asarray(explicit_rates, jnp.float32)
    if configuration.anneal_learning_rate:
        rates *= 1.0 - state.iteration.astype(jnp.float32) / configuration.total_iterations

    def permutation(epoch):
        """Derive each policy's shuffled rows without reference to its batch neighbors."""
        keys = quantity_keys(state.policy_seeds, "grid/update/permutation", state.iteration, epoch)
        rows = jax.vmap(lambda key: jax.random.permutation(key, configuration.num_steps))(keys)
        return rows.reshape(configuration.n_copies, configuration.num_minibatches,
            configuration.minibatch_size).transpose(1, 0, 2)

    def gather(rows):
        """Select matching transitions independently within each policy."""
        return jax.tree.map(lambda values: jax.vmap(lambda values, ids: values[ids])(values, rows), batch)

    separate = configuration.intrinsic_source == "online_rnd" and configuration.rnd_optimizer != "shared_adam"

    def minibatch(carry, rows_or_batch):
        """Differentiate one minibatch and update each policy using its own gradient norm."""
        parameters, optimizer, predictor_optimizer = carry
        if configuration.intrinsic_source == "online_rnd":
            rows, mask = rows_or_batch
            selected = dict(gather(rows), predictor_mask=mask)
        else:
            selected = rows_or_batch if configuration.pre_gather_minibatches else gather(rows_or_batch)

        def loss(values):
            """Unpack flat storage only at the policy evaluation boundary."""
            return minibatch_loss(layout.unpack(values) if layout else values, selected, coordinates,
                                  configuration, state.target)

        (_, metrics), gradients = jax.value_and_grad(loss, has_aux=True)(parameters)
        if separate:
            # The policy keeps Adam with its own clip; the predictor takes its own unclipped step.
            policy = {name: value for name, value in parameters.items() if name != "predictor"}
            policy, optimizer, norm = adam_step(policy, {name: value for name, value in gradients.items()
                if name != "predictor"}, optimizer, rates, scalar_setting(configuration, "max_gradient_norm"),
                scalar_setting(configuration, "adam_epsilon"), fuse_optimizer=configuration.fuse_optimizer)
            if configuration.rnd_optimizer == "exact_step":
                predictor = exact_shrink_step(parameters["predictor"], state.target, selected["rnd_inputs"],
                                              selected["predictor_mask"], configuration)
                predictor_optimizer = predictor_optimizer._replace(step=predictor_optimizer.step + 1)
            else:
                predictor, predictor_optimizer = predictor_step(parameters["predictor"], gradients["predictor"],
                                                                predictor_optimizer, configuration)
            parameters = dict(policy, predictor=predictor)
        else:
            parameters, optimizer, norm = adam_step(parameters, gradients, optimizer, rates,
                scalar_setting(configuration, "max_gradient_norm"), scalar_setting(configuration, "adam_epsilon"),
                fuse_optimizer=configuration.fuse_optimizer and layout is None)
        return (parameters, optimizer, predictor_optimizer), {**metrics, "gradient_norm": norm}

    if configuration.flatten_update_scan:
        rows = jax.vmap(permutation)(jnp.arange(configuration.update_epochs, dtype=jnp.int32)).reshape(
            configuration.update_epochs * configuration.num_minibatches, configuration.n_copies, configuration.minibatch_size)
        selected = jax.vmap(gather)(rows) if configuration.pre_gather_minibatches else rows
        (parameters, optimizer, predictor_optimizer), metrics = jax.lax.scan(
            minibatch, (parameters, optimizer, state.predictor_optimizer), selected, unroll=configuration.update_unroll)
        metrics = jax.tree.map(lambda values: values.mean(axis=0), metrics)
    else:
        def epoch(carry, index):
            """Run one complete shuffled epoch in its original update order."""
            rows = permutation(index)
            selected = jax.vmap(gather)(rows) if configuration.pre_gather_minibatches else rows
            if configuration.intrinsic_source == "online_rnd":
                # One uniform draw per sample per epoch keeps it with probability 0.25.
                keys = quantity_keys(state.policy_seeds, "grid/rnd/predictor_mask", state.iteration, index)
                kept = jax.vmap(lambda key: jax.random.uniform(key, (configuration.num_steps,)))(keys) < (
                    configuration.predictor_update_proportion)
                selected = (rows, jax.vmap(lambda ids: jax.vmap(lambda values, row: values[row])(kept, ids))(rows))
            return jax.lax.scan(minibatch, carry, selected, unroll=configuration.update_unroll)

        (parameters, optimizer, predictor_optimizer), metrics = jax.lax.scan(
            epoch, (parameters, optimizer, state.predictor_optimizer), jnp.arange(configuration.update_epochs, dtype=jnp.int32))
        metrics = jax.tree.map(lambda values: values.mean(axis=(0, 1)), metrics)
    if layout:
        parameters = layout.unpack(parameters)
        optimizer = AdamState(layout.unpack(optimizer.first), layout.unpack(optimizer.second), optimizer.step)
    return state._replace(parameters=parameters, optimizer=optimizer, predictor_optimizer=predictor_optimizer), {
        **metrics, "learning_rate": rates}


def exact_shrink_step(predictor, target, inputs, mask, configuration):
    """Move every kept cell of a minibatch to its shrink target for all its kept visits, with the best step length."""
    # Before: one-hot inputs [C,M,S] and a keep mask [C,M]; after: predictor weights after the exact steps.
    kept = mask.astype(jnp.float32)
    features = jax.lax.stop_gradient(target_features(target, inputs, configuration))
    # Kept visits of each sample's own cell in this minibatch: for one-hot rows the inner product is 1 or 0.
    same_cell = jnp.einsum("cmd,cnd->cmn", inputs, inputs, precision=jax.lax.Precision.HIGHEST)
    visits = (same_cell * kept[:, None, :]).sum(axis=-1)
    # Every distinct kept cell gets total weight one, however many of its visits were kept.
    weight = (kept / jnp.maximum(visits, 1.0))[..., None]
    outputs = jax_forward(predictor, inputs)
    residual = features - outputs
    squared = jnp.sum(residual ** 2, axis=-1, keepdims=True)
    power = configuration.rnd_shrink_power
    count = configuration.rnd_count_per_kept_visit * visits[..., None]
    # m kept visits at once: r -> r (1 + count * r^p)^(-1/p), the same as m single shrink steps in a row.
    factor = (1.0 + count * squared ** (power / 2.0)) ** (-1.0 / power)
    wanted = jax.lax.stop_gradient(outputs + (1.0 - factor) * residual)
    copies = kept.shape[0]

    def one_step(predictor, _):
        """Steepest descent on the weighted squared distance to the wanted outputs, with the exact step length."""
        def distance(values):
            return 0.5 * (weight * (jax_forward(values, inputs) - wanted) ** 2).sum()

        gradient = jax.grad(distance)(predictor)
        direction = jax.tree.map(lambda value: -value, gradient)
        # The outputs change by `step * moved` to first order; the best step is a one-dimensional least squares.
        current, moved = jax.jvp(lambda values: jax_forward(values, inputs), (predictor,), (direction,))
        numerator = (weight * (wanted - current) * moved).sum(axis=(1, 2))
        denominator = (weight * moved * moved).sum(axis=(1, 2))
        step = jnp.where(denominator > 0, numerator / jnp.where(denominator > 0, denominator, 1.0), 0.0)
        return jax.tree.map(lambda value, change: value + step.reshape((copies,) + (1,) * (value.ndim - 1)) * change,
                            predictor, direction), None

    predictor, _ = jax.lax.scan(one_step, predictor, None, length=configuration.rnd_exact_step_iterations)
    return predictor


def predictor_step(predictor, gradients, optimizer, configuration):
    """Apply the predictor's own optimizer to every copy: SGD, AdaGrad or Adam, with no clip."""
    # The learning rate is constant, or rate/(1 + t/T) after t earlier predictor steps.
    rate = jnp.float32(configuration.rnd_learning_rate)
    if configuration.rnd_learning_rate_time_scale > 0:
        rate = rate / (1.0 + optimizer.step.astype(jnp.float32) / configuration.rnd_learning_rate_time_scale)
    step = optimizer.step + 1
    if configuration.rnd_optimizer == "sgd":
        return jax.tree.map(lambda value, gradient: value - rate * gradient, predictor, gradients), optimizer._replace(step=step)
    if configuration.rnd_optimizer == "adagrad":
        # PyTorch AdaGrad: the accumulated squared gradient divides each step, epsilon 1e-10.
        second = jax.tree.map(lambda total, gradient: total + gradient ** 2, optimizer.second, gradients)
        return jax.tree.map(lambda value, gradient, total: value - rate * gradient / (jnp.sqrt(total) + 1e-10),
                            predictor, gradients, second), AdamState(None, second, step)
    rates = jnp.full((predictor[0]["weight"].shape[0],), rate, jnp.float32)
    predictor, optimizer, _ = adam_step(predictor, gradients, optimizer, rates, 0.0, configuration.adam_epsilon)
    return predictor, optimizer


def train_iteration(state, configuration, transition_table, coordinates, fields):
    """Collect one exact-count rollout and perform its complete CleanRL PPO update."""
    if configuration.intrinsic_source == "online_rnd":
        fields = all_cell_errors(state, coordinates, configuration)
    state, rollout = collect_rollout(state, configuration, transition_table, coordinates, fields)
    state, batch = prepare_batch(state, rollout, coordinates, configuration)
    state, metrics = update_policy(state, batch, coordinates, configuration)
    state = state._replace(iteration=state.iteration + 1)
    metrics.update(raw_intrinsic_reward=rollout.intrinsic_rewards.mean(axis=0),
        goal_arrivals=rollout.succeeded.sum(axis=0), successes=state.successes,
        episodes=state.episodes, first_goal_step=state.first_goal_step,
        maximum_progress=state.maximum_progress, coverage=state.reached.mean(axis=1),
        blocked_moves=state.blocked_moves,
        steps_per_policy=state.iteration * configuration.num_steps)
    return state, metrics


@lru_cache(maxsize=8)
def compiled_iteration(configuration):
    """Reuse program code while supplying each frozen field as an ordinary device input."""
    return jax.jit(lambda state, transitions, coordinates, fields: train_iteration(
        state, configuration, transitions, coordinates, fields),
        donate_argnums=(0,) if configuration.donate_state else ())


def evaluate_policy(state, configuration, transition_table, coordinates, episodes, greedy):
    """Play episodes from the start with the fixed final policy; return each goal step or -1."""
    # Before: parameters; after: first goal step [C,episodes], -1 when the limit is reached.
    copies, state_count = configuration.n_copies, coordinates.shape[0]
    goal = state_count - 1 if configuration.goal_state == -1 else configuration.goal_state
    ids = jnp.broadcast_to(jnp.arange(state_count, dtype=jnp.int32), (copies, state_count))
    logits = network_outputs(state.parameters, ids, coordinates, configuration)[..., :4]
    episode_ids = jnp.arange(episodes, dtype=jnp.uint32)

    def step(carry, time):
        """Move every unfinished evaluation episode one transition."""
        position, reached_at = carry
        values = jnp.take_along_axis(logits, position[..., None], axis=1)
        if greedy:
            actions = jnp.argmax(values, axis=-1)
        else:
            keys = quantity_keys(state.policy_seeds, "grid/evaluation/actions", time)
            keys = jax.vmap(lambda key: jax.vmap(lambda episode: jax.random.fold_in(key, episode))(episode_ids))(keys)
            actions = jax.vmap(jax.vmap(jax.random.categorical))(keys, values)
        arrived = transition_table[position, actions.astype(jnp.int32)]
        unfinished = reached_at < 0
        reached_at = jnp.where(unfinished & (arrived == goal), time + 1, reached_at)
        return (jnp.where(unfinished, arrived, position), reached_at), None

    start = jnp.full((copies, episodes), configuration.start_state, jnp.int32)
    (_, reached_at), _ = jax.lax.scan(step, (start, jnp.full((copies, episodes), -1, jnp.int32)),
                                      jnp.arange(configuration.episode_length, dtype=jnp.int32))
    return reached_at


class GridTrainer:
    """Bind immutable experiment inputs once and compile complete independent-policy updates."""

    def __init__(self, configuration, transition_table, coordinates, fields, rnd=None):
        """Keep one concrete geometry and frozen field batch outside mutable training state."""
        self.configuration = configuration
        self.transition_table = jnp.asarray(transition_table, jnp.int32)
        self.coordinates = jnp.asarray(coordinates, jnp.float32)
        self.fields = jnp.asarray(fields, jnp.float32)
        self.rnd = rnd
        self._iteration = compiled_iteration(configuration)

    def initialize(self, policy_seeds):
        """Create a fresh independent group after validating immutable inputs."""
        return initialize(self.configuration, self.transition_table, self.coordinates, self.fields,
                          policy_seeds, self.rnd)

    def all_cell_errors(self, state):
        """Return the online RND bonus of every cell for the decay record."""
        return jax.jit(lambda value: all_cell_errors(value, self.coordinates, self.configuration))(state)

    def evaluate(self, state, episodes, greedy):
        """Return each evaluation episode's goal step, or -1, for the fixed final policy."""
        return jax.jit(lambda value: evaluate_policy(value, self.configuration, self.transition_table,
                                                     self.coordinates, episodes, greedy))(state)

    def train_iteration(self, state):
        """Execute one compiled rollout and update without copying data to the host."""
        return self._iteration(state, self.transition_table, self.coordinates, self.fields)

    def capture(self, state):
        """Return a rollout for correctness or visualization without changing its input."""
        fields = (all_cell_errors(state, self.coordinates, self.configuration)
                  if self.configuration.intrinsic_source == "online_rnd" else self.fields)
        return collect_rollout(state, self.configuration, self.transition_table, self.coordinates, fields)

    def prepare_batch(self, state, rollout):
        """Expose fixed PPO targets for independent optimization comparisons."""
        return prepare_batch(state, rollout, self.coordinates, self.configuration)

    def update(self, state, batch):
        """Update a supplied fixed batch using the same production policy update."""
        return update_policy(state, batch, self.coordinates, self.configuration)

    def policy_outputs(self, state):
        """Return all action logits and both values for complete policy visualization."""
        ids = jnp.broadcast_to(jnp.arange(self.coordinates.shape[0], dtype=jnp.int32),
                               (self.configuration.n_copies, self.coordinates.shape[0]))
        return network_outputs(state.parameters, ids, self.coordinates, self.configuration)

    def train_steps(self, state, iterations):
        """Advance a known number of rollouts and return the final diagnostics."""
        if iterations < 1:
            raise ValueError("iterations must be positive")
        for _ in range(iterations):
            state, metrics = self.train_iteration(state)
        return state, metrics
