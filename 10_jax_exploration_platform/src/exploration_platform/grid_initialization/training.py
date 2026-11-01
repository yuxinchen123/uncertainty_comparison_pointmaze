"""CleanRL PPO with exact online counts and frozen state-dependent bonus fields."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import GridPPOConfig
from ..atari_pixels.networks import categorical_statistics, initialize_layer, quantity_keys
from ..atari_pixels.settings import scalar_setting, scalar_product
from ..atari_pixels.state import (AdamState, FlatParameterLayout, adam_step, initialize_adam,
                                 initialize_statistics)
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
    # Every neural policy has its own two 64-unit tanh layers and six-output head.
    values = coordinates[states]
    for layer in parameters["hidden"]:
        values = jnp.tanh(jnp.matmul(values, layer["weight"]) + layer["bias"][:, None])
    head = parameters["head"]
    return jnp.matmul(values, head["weight"]) + head["bias"][:, None]


def initialize(configuration, transition_table, coordinates, fields, policy_seeds):
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
            "head": initialize_layer(seeds, "grid/policy/head", 6, width, gain=0.01)}
    # Before: no visits; after: the start is marked reached, with zero arrival counts.
    shape = (configuration.n_copies,)
    counts = jnp.zeros(shape + (state_count,), jnp.int32)
    reached = jnp.zeros_like(counts, dtype=jnp.bool_).at[:, configuration.start_state].set(True)
    filter_size = configuration.num_steps if configuration.upstream_reward_filter_axis else 1
    return GridTrainState(parameters, initialize_adam(parameters),
        jnp.full(shape, configuration.start_state, jnp.int32), jnp.zeros(shape, jnp.int32),
        counts, jnp.zeros_like(counts), reached, jnp.zeros(shape, jnp.int32),
        jnp.zeros(shape, jnp.int32), jnp.full(shape, -1, jnp.int32),
        jnp.full(shape, configuration.start_state, jnp.int32),
        jnp.zeros(shape, jnp.int32),
        initialize_statistics(configuration.n_copies, ()),
        jnp.zeros(shape + (filter_size,), jnp.float32), jnp.zeros((), jnp.int32), seeds)


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
        bonus = jnp.zeros((copies,), jnp.float32) if configuration.hoist_bonus else (
            fields[rows, arrived] * jax.lax.rsqrt(visits.astype(jnp.float32)) if configuration.rsqrt_bonus
            else fields[rows, arrived] / jnp.sqrt(visits.astype(jnp.float32)))
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
        visits = rollout.arrival_counts.astype(jnp.float32)
        bonus = initial * jax.lax.rsqrt(visits) if configuration.rsqrt_bonus else initial / jnp.sqrt(visits)
        rollout = rollout._replace(intrinsic_rewards=bonus)
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
    return state, batch


def minibatch_loss(parameters, batch, coordinates, configuration):
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

    def minibatch(carry, rows_or_batch):
        """Differentiate one minibatch and update each policy using its own gradient norm."""
        parameters, optimizer = carry
        selected = rows_or_batch if configuration.pre_gather_minibatches else gather(rows_or_batch)

        def loss(values):
            """Unpack flat storage only at the policy evaluation boundary."""
            return minibatch_loss(layout.unpack(values) if layout else values, selected, coordinates, configuration)

        (_, metrics), gradients = jax.value_and_grad(loss, has_aux=True)(parameters)
        parameters, optimizer, norm = adam_step(parameters, gradients, optimizer, rates,
            scalar_setting(configuration, "max_gradient_norm"), scalar_setting(configuration, "adam_epsilon"),
            fuse_optimizer=configuration.fuse_optimizer and layout is None)
        return (parameters, optimizer), {**metrics, "gradient_norm": norm}

    if configuration.flatten_update_scan:
        rows = jax.vmap(permutation)(jnp.arange(configuration.update_epochs, dtype=jnp.int32)).reshape(
            configuration.update_epochs * configuration.num_minibatches, configuration.n_copies, configuration.minibatch_size)
        selected = jax.vmap(gather)(rows) if configuration.pre_gather_minibatches else rows
        (parameters, optimizer), metrics = jax.lax.scan(minibatch, (parameters, optimizer), selected,
                                                       unroll=configuration.update_unroll)
        metrics = jax.tree.map(lambda values: values.mean(axis=0), metrics)
    else:
        def epoch(carry, index):
            """Run one complete shuffled epoch in its original update order."""
            rows = permutation(index)
            selected = jax.vmap(gather)(rows) if configuration.pre_gather_minibatches else rows
            return jax.lax.scan(minibatch, carry, selected, unroll=configuration.update_unroll)

        (parameters, optimizer), metrics = jax.lax.scan(epoch, (parameters, optimizer), jnp.arange(configuration.update_epochs, dtype=jnp.int32))
        metrics = jax.tree.map(lambda values: values.mean(axis=(0, 1)), metrics)
    if layout:
        parameters = layout.unpack(parameters)
        optimizer = AdamState(layout.unpack(optimizer.first), layout.unpack(optimizer.second), optimizer.step)
    return state._replace(parameters=parameters, optimizer=optimizer), {**metrics, "learning_rate": rates}


def train_iteration(state, configuration, transition_table, coordinates, fields):
    """Collect one exact-count rollout and perform its complete CleanRL PPO update."""
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


class GridTrainer:
    """Bind immutable experiment inputs once and compile complete independent-policy updates."""

    def __init__(self, configuration, transition_table, coordinates, fields):
        """Keep one concrete geometry and frozen field batch outside mutable training state."""
        self.configuration = configuration
        self.transition_table = jnp.asarray(transition_table, jnp.int32)
        self.coordinates = jnp.asarray(coordinates, jnp.float32)
        self.fields = jnp.asarray(fields, jnp.float32)
        self._iteration = jax.jit(lambda state: train_iteration(state, configuration,
            self.transition_table, self.coordinates, self.fields),
            donate_argnums=(0,) if configuration.donate_state else ())

    def initialize(self, policy_seeds):
        """Create a fresh independent group after validating immutable inputs."""
        return initialize(self.configuration, self.transition_table, self.coordinates, self.fields, policy_seeds)

    def train_iteration(self, state):
        """Execute one compiled rollout and update without copying data to the host."""
        return self._iteration(state)

    def capture(self, state):
        """Return a rollout for correctness or visualization without changing its input."""
        return collect_rollout(state, self.configuration, self.transition_table, self.coordinates, self.fields)

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
