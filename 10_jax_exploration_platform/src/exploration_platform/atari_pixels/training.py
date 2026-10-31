"""Original CleanRL PPO/RND equations with independent copies and a pure JAX environment."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import PixelPPOConfig
from .networks import (actor_logits, categorical_statistics, critic_values, initialize_networks,
                       policy_features, policy_forward, quantity_keys, rnd_forward)
from .state import (AdamState, FlatParameterLayout, PixelTrainState, adam_step, initialize_adam, initialize_statistics,
                    normalize_frames, update_moments, update_observation_statistics)


class Rollout(NamedTuple):
    """Time-major transitions, with policy and environment axes kept distinct."""
    observations: object
    next_frames: object
    actions: object
    log_probabilities: object
    rewards: object
    intrinsic_rewards: object
    done_before: object
    values_ext: object
    values_int: object
    episode_returns: object
    episode_finished: object
    policy_features: object
    logits: object
    reset_frames: object
    frame_bank: object


def flatten_rollout(values):
    """Turn [time,copies,environments,...] into [copies,time*environments,...]."""
    # Before: [128,2,128,4,84,84]; after: [2,16384,4,84,84], without mixing policies.
    axes = (1, 0, 2) + tuple(range(3, values.ndim))
    return values.transpose(axes).reshape(values.shape[1], -1, *values.shape[3:])


def gather_indexed_frames(frame_bank, lanes, frame_indices):
    """Reconstruct only selected pixel stacks from each policy's per-environment frame bank."""
    # Before: bank [C,N,K,H,W], lane [C,B], indices [C,B,4]; after: [C,B,4,H,W].
    return jax.vmap(lambda bank, lane, indices: bank[lane[:, None], indices])(
        frame_bank, lanes, frame_indices)


def intrinsic_reward(parameters, target, frames, statistics, configuration=None):
    """Use half the SUM of feature errors for the rollout bonus, as upstream does."""
    normalized = normalize_frames(frames, statistics)
    differences = (rnd_forward(target, normalized, configuration)
                   - rnd_forward(parameters["predictor"], normalized, configuration))
    return jnp.sum(differences ** 2, axis=-1) * 0.5


def normalize_intrinsic_rewards(rewards, state, config):
    """Retain upstream's environment-axis filter and count, or explicitly select time order."""
    # Upstream transposes [T,N] and iterates N vectors of length T; each policy is independent.
    values = rewards.transpose(2, 1, 0) if config.upstream_reward_filter_axis else rewards

    def accumulate(previous, reward):
        """Carry the unreset discounted intrinsic filter along the selected upstream axis."""
        filtered = previous * jnp.asarray(config.gamma_int, jnp.float32) + reward
        return filtered, filtered

    if config.associative_reward_filter:
        coefficients = jnp.full_like(values, config.gamma_int)

        def compose(left, right):
            """Compose later affine filter updates after earlier updates."""
            left_value, left_coefficient = left
            right_value, right_coefficient = right
            return right_value + right_coefficient * left_value, right_coefficient * left_coefficient

        filtered, products = jax.lax.associative_scan(compose, (values, coefficients))
        filtered = filtered + products * state.reward_filter[None]
        filtered_state = filtered[-1]
    else:
        filtered_state, filtered = jax.lax.scan(accumulate, state.reward_filter, values)
    values_for_moments = filtered.astype(jnp.float64)
    batch_mean = values_for_moments.mean(axis=(0, 2))
    batch_variance = (values_for_moments.var(axis=(0, 2)) if config.moment_reduction == "two_pass"
                      else jnp.maximum((values_for_moments ** 2).mean(axis=(0, 2)) - batch_mean ** 2, 0.0))
    statistics = update_moments(
        state.reward_statistics,
        batch_mean, batch_variance,
        config.n_envs if config.upstream_reward_filter_axis else config.batch_size)
    normalized = rewards / jnp.sqrt(statistics.variance).astype(jnp.float32)[None, :, None]
    return normalized, statistics, filtered_state


def generalized_advantages(rewards, intrinsic_rewards, rollout, final_values,
                           done_after, config):
    """Mask extrinsic GAE at done while keeping intrinsic GAE non-episodic."""
    values_ext, values_int = final_values
    # Before: done_before[t] describes t-1; after: masks[t] describes transition t.
    masks = 1.0 - jnp.concatenate((rollout.done_before[1:], done_after[None]), axis=0).astype(jnp.float32)
    next_ext = jnp.concatenate((rollout.values_ext[1:], values_ext[None]), axis=0)
    next_int = jnp.concatenate((rollout.values_int[1:], values_int[None]), axis=0)
    delta_ext = rewards + config.gamma_ext * next_ext * masks - rollout.values_ext
    delta_int = intrinsic_rewards + config.gamma_int * next_int - rollout.values_int

    def reverse_step(previous, values):
        """Compute one earlier extrinsic/intrinsic advantage pair."""
        previous_ext, previous_int = previous
        difference_ext, difference_int, mask = values
        advantage_ext = difference_ext + config.gamma_ext * config.gae_lambda * mask * previous_ext
        advantage_int = difference_int + config.gamma_int * config.gae_lambda * previous_int
        return (advantage_ext, advantage_int), (advantage_ext, advantage_int)

    zeros = jnp.zeros_like(values_ext)
    if config.associative_gae:
        def compose(left, right):
            """Compose one earlier advantage recurrence after a later recurrence."""
            left_value, left_coefficient = left
            right_value, right_coefficient = right
            return right_value + right_coefficient * left_value, right_coefficient * left_coefficient

        coefficient_ext = config.gamma_ext * config.gae_lambda * masks
        coefficient_int = jnp.full_like(delta_int, config.gamma_int * config.gae_lambda)
        advantage_ext, _ = jax.lax.associative_scan(compose, (delta_ext[::-1], coefficient_ext[::-1]))
        advantage_int, _ = jax.lax.associative_scan(compose, (delta_int[::-1], coefficient_int[::-1]))
        return advantage_ext[::-1], advantage_int[::-1]
    _, advantages = jax.lax.scan(reverse_step, (zeros, zeros), (delta_ext, delta_int, masks), reverse=True)
    return advantages


def minibatch_loss(parameters, target, batch, predictor_mask, config,
                   intrinsic_coefficients, observation_statistics=None):
    """Return the sum of per-policy losses and independent loss diagnostics."""
    def evaluate_policy(parameters, observations):
        """Keep rematerialization local to the trainable policy's activation storage."""
        return policy_forward(parameters, observations, config)

    def evaluate_predictor(parameters, frames):
        """Keep rematerialization local to the trainable predictor's activation storage."""
        return rnd_forward(parameters, frames, config)

    policy = jax.checkpoint(evaluate_policy) if config.recompute_network_activations else evaluate_policy
    predictor = jax.checkpoint(evaluate_predictor) if config.recompute_network_activations else evaluate_predictor
    logits, value_ext, value_int = policy(parameters["agent"], batch["observations"])
    log_probability, entropy = categorical_statistics(logits, batch["actions"])
    ratio = jnp.exp(log_probability - batch["log_probabilities"])
    advantages = batch["advantage_ext"] * config.extrinsic_coefficient
    advantages = advantages + batch["advantage_int"] * intrinsic_coefficients[:, None]
    # PyTorch Tensor.std() uses correction=1; JAX's default population variance would differ.
    advantages = (advantages - advantages.mean(axis=1, keepdims=True)) / (
        advantages.std(axis=1, ddof=1, keepdims=True) + 1e-8)
    unclipped = -advantages * ratio
    clipped = -advantages * jnp.clip(ratio, 1.0 - config.clip_coefficient, 1.0 + config.clip_coefficient)
    policy_loss = jnp.maximum(unclipped, clipped).mean(axis=1)

    # Clip only the extrinsic value head; the intrinsic value head uses plain squared error.
    clipped_value = batch["values_ext"] + jnp.clip(
        value_ext - batch["values_ext"], -config.clip_coefficient, config.clip_coefficient)
    value_loss_ext = 0.5 * jnp.maximum((value_ext - batch["return_ext"]) ** 2,
                                     (clipped_value - batch["return_ext"]) ** 2).mean(axis=1)
    value_loss_int = 0.5 * ((value_int - batch["return_int"]) ** 2).mean(axis=1)
    normalized_frames = (normalize_frames(batch["raw_frames"], observation_statistics)
                         if config.fuse_minibatch_image_preparation else batch["normalized_frames"])
    target_features = (batch["target_features"] if config.cache_target_features
                       else rnd_forward(target, normalized_frames, config))
    target_features = jax.lax.stop_gradient(target_features)
    predictor_features = predictor(parameters["predictor"], normalized_frames)
    # Upstream trains with MEAN feature error, although the rollout bonus uses SUM/2.
    errors = ((predictor_features - target_features) ** 2).mean(axis=-1)
    predictor_loss = (errors * predictor_mask).sum(axis=1) / jnp.maximum(predictor_mask.sum(axis=1), 1.0)
    entropy_mean = entropy.mean(axis=1)
    losses = (policy_loss - config.entropy_coefficient * entropy_mean
              + config.value_coefficient * (value_loss_ext + value_loss_int) + predictor_loss)
    log_ratio = log_probability - batch["log_probabilities"]
    diagnostics = {"loss": losses, "policy_loss": policy_loss,
                   "value_loss": value_loss_ext + value_loss_int,
                   "predictor_loss": predictor_loss, "entropy": entropy_mean,
                   "approximate_kl": ((ratio - 1.0) - log_ratio).mean(axis=1)}
    # Summing independent policy means preserves each policy's original gradient magnitude.
    return losses.sum(), diagnostics


class PixelPPORND:
    """Construct compiled warmup and iteration functions for one fixed group shape."""

    def __init__(self, config: PixelPPOConfig, environment):
        """Keep environment and copy shape static while mutable state stays in a PyTree."""
        if not jax.config.x64_enabled:
            raise RuntimeError("the original running statistics require jax_enable_x64=True")
        if environment.action_count != config.action_count:
            raise ValueError("environment and learner action counts differ")
        self.config, self.environment = config, environment
        self.learning_rates = jnp.asarray(config.per_copy_learning_rates or
                                         (config.learning_rate,) * config.n_copies, jnp.float32)
        self.intrinsic_coefficients = jnp.asarray(config.per_copy_intrinsic_coefficients or
                                                (config.intrinsic_coefficient,) * config.n_copies, jnp.float32)
        self._compiled_iteration = jax.jit(self.iteration, donate_argnums=(0,) if config.donate_state else ())
        self._compiled_warmup = jax.jit(self._warmup)

    def train_iteration(self, state):
        """Execute the compiled iteration and separate observation/state buffers for donation."""
        state, metrics = self._compiled_iteration(state)
        # The environment stores the same pixels in its frame stack. A separate input buffer
        # prevents the next donated call from receiving that buffer twice.
        return state._replace(observations=jnp.array(state.observations, copy=True)), metrics

    def warmup(self, state):
        """Run compiled normalization warmup and retain a separately owned observation buffer."""
        state = self._compiled_warmup(state)
        return state._replace(observations=jnp.array(state.observations, copy=True))

    def initialize(self, copy_seeds=None):
        """Create networks, environment state, moments and stable per-copy seed identifiers."""
        config = self.config
        copy_seeds = jnp.asarray(tuple(range(1, config.n_copies + 1)) if copy_seeds is None
                                 else copy_seeds, jnp.uint32)
        if copy_seeds.shape != (config.n_copies,):
            raise ValueError("copy_seeds must have shape [n_copies]")
        parameters, target = initialize_networks(copy_seeds, config.action_count)
        environment_state, observations = self.environment.reset(copy_seeds)
        expected = (config.n_copies, config.n_envs, 4, 84, 84)
        if observations.shape != expected or observations.dtype != jnp.uint8:
            raise ValueError(f"environment observations must be uint8 {expected}, got {observations.shape}/{observations.dtype}")
        filter_size = config.num_steps if config.upstream_reward_filter_axis else config.n_envs
        return PixelTrainState(environment_state, jnp.array(observations, copy=True),
                               jnp.zeros((config.n_copies, config.n_envs), jnp.bool_), parameters,
                               target, initialize_adam(parameters),
                               initialize_statistics(config.n_copies, (1, 84, 84)),
                               initialize_statistics(config.n_copies, ()),
                               jnp.zeros((config.n_copies, filter_size), jnp.float32),
                               jnp.zeros((), jnp.int32), copy_seeds)

    def _warmup(self, state):
        """Collect fifty random rollouts for observation moments, preserving upstream's first row."""
        config = self.config

        def collect_rollout(carry, warmup_iteration):
            """Update observation moments once per 128-step random rollout."""
            environment_state, statistics, observations = carry

            def random_step(environment_state, step):
                """Advance the environment with independently keyed uniformly random actions."""
                keys = quantity_keys(state.copy_seeds, "warmup/actions", warmup_iteration, step)
                actions = jax.vmap(lambda key: jax.random.randint(
                    key, (config.n_envs,), 0, config.action_count, dtype=jnp.int32))(keys)
                environment_state, observations, _, _, _ = self.environment.step(environment_state, actions)
                return environment_state, observations

            environment_state, collected = jax.lax.scan(random_step, environment_state,
                                                        jnp.arange(config.num_steps))
            frames = flatten_rollout(collected)[..., 3:4, :, :]
            statistics = update_observation_statistics(statistics, frames, config.moment_reduction)
            return (environment_state, statistics, collected[-1]), None

        (environment_state, statistics, observations), _ = jax.lax.scan(
            collect_rollout, (state.environment, state.observation_statistics, state.observations),
            jnp.arange(config.observation_warmup_iterations))
        # Original CleanRL retains next_obs from before warmup while the environment advances.
        observations = state.observations if config.warmup_keeps_initial_observation else observations
        return state._replace(environment=environment_state, observations=observations,
                              observation_statistics=statistics)

    def collect(self, state):
        """Collect an on-policy rollout with frozen parameters and pre-update RND statistics."""
        config = self.config
        action_keys = (jax.vmap(lambda step: quantity_keys(state.copy_seeds, "rollout/actions",
            state.iteration, step))(jnp.arange(config.num_steps))
            if config.batch_quantity_keyed_random_generation else None)
        indexed = config.rollout_frame_storage == "indexed_frames"
        # Initial policy pixels intentionally differ from the true environment after upstream warmup.
        frame_bank = (jnp.concatenate((state.observations, self.environment.observe(state.environment)),
            axis=2) if indexed else None)
        initial_indices = jnp.broadcast_to(jnp.arange(4, 8, dtype=jnp.int32),
                                           state.observations.shape[:2] + (4,))

        def step(carry, step_index):
            """Record one transition and preserve its next pixels for the RND bonus."""
            environment_state, observations, done_before, frame_indices = carry
            hidden = policy_features(state.parameters["agent"], observations, config)
            logits = actor_logits(state.parameters["agent"], hidden, config)
            if config.hoist_critic:
                values_ext = values_int = jnp.zeros(observations.shape[:2], jnp.float32)
            else:
                values_ext, values_int = critic_values(state.parameters["agent"], hidden, config)
            keys = (action_keys[step_index] if config.batch_quantity_keyed_random_generation
                    else quantity_keys(state.copy_seeds, "rollout/actions", state.iteration, step_index))
            actions = jax.vmap(lambda key, value: jax.random.categorical(key, value).astype(jnp.int32))(keys, logits)
            log_probability = (jnp.zeros(actions.shape, jnp.float32)
                if config.hoist_rollout_log_probabilities else categorical_statistics(logits, actions)[0])
            next_environment, next_observations, reward, done, information = self.environment.step(
                environment_state, actions)
            # The environment resets in this same call, but a terminal transition's next
            # image is its pre-reset image, as returned by the original EnvPool transition.
            frames = information["final_observation"][..., 3:4, :, :]
            bonus = (jnp.zeros_like(reward) if config.hoist_rollout_bonus else intrinsic_reward(
                state.parameters, state.target, frames, state.observation_statistics, config))
            if indexed:
                stored_observations = jnp.where(step_index == 0, frame_indices - 4, frame_indices)
                normal_index = jnp.full(frame_indices.shape[:-1] + (1,), 8 + 2 * step_index, jnp.int32)
                reset_index = normal_index + 1
                after_transition = jnp.concatenate((frame_indices[..., 1:], normal_index), axis=-1)
                after_life_reset = jnp.concatenate((after_transition[..., 1:], reset_index), axis=-1)
                frame_indices = jnp.where(information["episode_done"][..., None], reset_index,
                    jnp.where(done[..., None], after_life_reset, after_transition))
            else:
                stored_observations = (observations.astype(jnp.float32)
                    if config.rollout_observation_storage == "float32" else observations)
            recorded = Rollout(stored_observations, frames, actions, log_probability, reward, bonus,
                               done_before, values_ext, values_int, information["episode_return"],
                               information["episode_done"], hidden if config.hoist_critic else
                               jnp.zeros(hidden.shape[:2] + (0,), jnp.float32),
                               logits if config.hoist_rollout_log_probabilities else
                               jnp.zeros(logits.shape[:2] + (0,), jnp.float32),
                               next_observations[..., 3:4, :, :] if indexed else
                               jnp.zeros(observations.shape[:2] + (0, 84, 84), jnp.uint8),
                               jnp.zeros(observations.shape[:2] + (0, 84, 84), jnp.uint8))
            return (next_environment, next_observations, done, frame_indices), recorded

        (environment_state, observations, done, _), rollout = jax.lax.scan(
            step, (state.environment, state.observations, state.done, initial_indices), jnp.arange(config.num_steps),
            unroll=config.rollout_unroll)
        if indexed:
            # Store two new frames per transition: the terminal image and any reset image.
            # Before: [T,C,N,2,H,W]; after: bank [C,N,8+2*T,H,W], shared across minibatches.
            added_frames = jnp.concatenate((rollout.next_frames, rollout.reset_frames), axis=3)
            added_frames = added_frames.transpose(1, 2, 0, 3, 4, 5).reshape(
                config.n_copies, config.n_envs, 2 * config.num_steps, 84, 84)
            rollout = rollout._replace(frame_bank=jnp.concatenate((frame_bank, added_frames), axis=2))
        if config.hoist_rollout_bonus:
            bonuses = intrinsic_reward(state.parameters, state.target,
                                       flatten_rollout(rollout.next_frames), state.observation_statistics, config)
            # Before: [C,T*N]; after: [T,C,N], restoring the exact temporal filter inputs.
            bonuses = bonuses.reshape(config.n_copies, config.num_steps, config.n_envs).transpose(1, 0, 2)
            rollout = rollout._replace(intrinsic_rewards=bonuses)
        if config.hoist_critic:
            values_ext, values_int = critic_values(state.parameters["agent"],
                flatten_rollout(rollout.policy_features), config)
            rollout = rollout._replace(values_ext=values_ext.reshape(config.n_copies,
                config.num_steps, config.n_envs).transpose(1, 0, 2),
                values_int=values_int.reshape(config.n_copies,
                config.num_steps, config.n_envs).transpose(1, 0, 2))
        if config.hoist_rollout_log_probabilities:
            log_probability, _ = categorical_statistics(flatten_rollout(rollout.logits),
                                                         flatten_rollout(rollout.actions))
            rollout = rollout._replace(log_probabilities=log_probability.reshape(
                config.n_copies, config.num_steps, config.n_envs).transpose(1, 0, 2))
        return state._replace(environment=environment_state, observations=observations, done=done), rollout

    def prepare_batch(self, state, rollout):
        """Compute upstream normalization, GAE, current-frame RND inputs and fixed-size batches."""
        config = self.config
        normalized_bonus, reward_statistics, reward_filter = normalize_intrinsic_rewards(
            rollout.intrinsic_rewards, state, config)
        _, final_ext, final_int = policy_forward(state.parameters["agent"], state.observations, config)
        advantage_ext, advantage_int = generalized_advantages(
            rollout.rewards, normalized_bonus, rollout, (final_ext, final_int), state.done, config)
        indexed = rollout.observations.ndim == 4
        observations = flatten_rollout(rollout.observations)
        if indexed:
            lanes = jnp.broadcast_to(jnp.tile(jnp.arange(config.n_envs, dtype=jnp.int32),
                                              config.num_steps), (config.n_copies, config.batch_size))
            frames = gather_indexed_frames(rollout.frame_bank, lanes, observations[..., -1:])
        else:
            frames = observations[..., 3:4, :, :]
        observation_statistics = update_observation_statistics(state.observation_statistics, frames, config.moment_reduction)
        batch = {"actions": flatten_rollout(rollout.actions),
                 "log_probabilities": flatten_rollout(rollout.log_probabilities),
                 "values_ext": flatten_rollout(rollout.values_ext),
                 "advantage_ext": flatten_rollout(advantage_ext),
                 "advantage_int": flatten_rollout(advantage_int),
                 "return_ext": flatten_rollout(advantage_ext + rollout.values_ext),
                 "return_int": flatten_rollout(advantage_int + rollout.values_int)}
        if indexed:
            batch.update(frame_bank=rollout.frame_bank, frame_lanes=lanes, frame_indices=observations)
        else:
            batch["observations"] = observations
        if config.fuse_minibatch_image_preparation:
            # Avoid a full float32 [C,16384,1,84,84] normalized-image buffer between phases.
            batch["raw_frames"] = frames.astype(jnp.uint8)
        else:
            batch["normalized_frames"] = normalize_frames(frames, observation_statistics)
        if config.cache_target_features:
            # Training targets use CURRENT frames and UPDATED statistics, unlike rollout targets.
            normalized = (normalize_frames(frames, observation_statistics)
                          if config.fuse_minibatch_image_preparation else batch["normalized_frames"])
            batch["target_features"] = jax.lax.stop_gradient(rnd_forward(state.target, normalized, config))
        return state._replace(observation_statistics=observation_statistics,
                              reward_statistics=reward_statistics, reward_filter=reward_filter), batch

    def update(self, state, batch):
        """Perform four independently shuffled epochs of four Adam steps per policy."""
        config = self.config
        fraction = (1.0 - state.iteration.astype(jnp.float32) / config.total_iterations
                    if config.anneal_learning_rate else jnp.asarray(1.0, jnp.float32))
        learning_rates = self.learning_rates * fraction
        layout = FlatParameterLayout(state.parameters) if config.flat_parameters else None
        starting_parameters = layout.pack(state.parameters) if layout is not None else state.parameters
        starting_optimizer = (AdamState(layout.pack(state.optimizer.first),
            layout.pack(state.optimizer.second), state.optimizer.step) if layout is not None else state.optimizer)

        def predictor_masks(epoch_index, minibatch_index):
            """Use exactly the same named mask keys in batched and per-step generation."""
            keys = quantity_keys(state.copy_seeds, "update/predictor-mask", state.iteration,
                                 epoch_index, minibatch_index)
            return jax.vmap(lambda key: (jax.random.uniform(key, (config.minibatch_size,),
                dtype=jnp.float32) < config.predictor_update_proportion).astype(jnp.float32))(keys)

        all_masks = (jax.vmap(lambda epoch: jax.vmap(lambda minibatch: predictor_masks(epoch, minibatch))(
            jnp.arange(config.num_minibatches)))(jnp.arange(config.update_epochs))
            if config.batch_quantity_keyed_random_generation else None)
        frame_bank = batch.get("frame_bank")
        row_batch = {name: value for name, value in batch.items() if name != "frame_bank"}

        def epoch_indices(epoch_index):
            """Build one independently shuffled permutation per policy and epoch."""
            keys = quantity_keys(state.copy_seeds, "update/permutation", state.iteration, epoch_index)
            permutations = jax.vmap(lambda key: jax.random.permutation(key, config.batch_size))(keys)
            return permutations.reshape(config.n_copies, config.num_minibatches,
                                         config.minibatch_size).transpose(1, 0, 2)

        def gather_minibatch(indices):
            """Gather one equal-sized minibatch from every independent policy."""
            selected = jax.tree.map(lambda value: jax.vmap(lambda rows, index: rows[index])(value, indices), row_batch)
            if frame_bank is not None:
                selected["observations"] = gather_indexed_frames(frame_bank, selected["frame_lanes"],
                                                                  selected["frame_indices"])
            return selected

        def minibatch(carry, values):
            """Differentiate and update one minibatch, retaining keyed epoch identities."""
            parameters, optimizer = carry
            epoch_index, minibatch_index, rows_or_indices = values
            selected = rows_or_indices if config.pre_gather_minibatches else gather_minibatch(rows_or_indices)
            masks = (all_masks[epoch_index, minibatch_index] if all_masks is not None
                     else predictor_masks(epoch_index, minibatch_index))

            def loss(parameters):
                """Differentiate trainable networks while keeping RND targets and statistics fixed."""
                named_parameters = layout.unpack(parameters) if layout is not None else parameters
                return minibatch_loss(named_parameters, state.target, selected, masks, config,
                                      self.intrinsic_coefficients, state.observation_statistics)

            (_, metrics), gradients = jax.value_and_grad(loss, has_aux=True)(parameters)
            parameters, optimizer, gradient_norm = adam_step(
                parameters, gradients, optimizer, learning_rates,
                config.max_gradient_norm, config.adam_epsilon,
                fuse_optimizer=config.fuse_optimizer and layout is None)
            return (parameters, optimizer), {**metrics, "gradient_norm": gradient_norm}

        def epoch(carry, epoch_index):
            """Give every policy a separate permutation before its four minibatches."""
            indices = epoch_indices(epoch_index)
            scan_values = jax.vmap(gather_minibatch)(indices) if config.pre_gather_minibatches else indices
            return jax.lax.scan(minibatch, carry, (jnp.full((config.num_minibatches,), epoch_index),
                                 jnp.arange(config.num_minibatches), scan_values),
                                 unroll=config.update_unroll)

        if config.flatten_update_scan:
            # Before: [epoch,minibatch,C,B]; after: one [epoch*minibatch,C,B] update scan.
            indices = jax.vmap(epoch_indices)(jnp.arange(config.update_epochs)).reshape(
                config.update_epochs * config.num_minibatches, config.n_copies, config.minibatch_size)
            scan_values = jax.vmap(gather_minibatch)(indices) if config.pre_gather_minibatches else indices
            epochs = jnp.repeat(jnp.arange(config.update_epochs), config.num_minibatches)
            minibatches = jnp.tile(jnp.arange(config.num_minibatches), config.update_epochs)
            (parameters, optimizer), diagnostics = jax.lax.scan(minibatch,
                (starting_parameters, starting_optimizer), (epochs, minibatches, scan_values),
                unroll=config.update_unroll)
            metrics = jax.tree.map(lambda value: value.mean(axis=0), diagnostics)
        else:
            (parameters, optimizer), diagnostics = jax.lax.scan(
                epoch, (starting_parameters, starting_optimizer), jnp.arange(config.update_epochs))
            metrics = jax.tree.map(lambda value: value.mean(axis=(0, 1)), diagnostics)
        if layout is not None:
            parameters = layout.unpack(parameters)
            optimizer = AdamState(layout.unpack(optimizer.first), layout.unpack(optimizer.second), optimizer.step)
        return state._replace(parameters=parameters, optimizer=optimizer), {**metrics, "learning_rate": learning_rates}

    def iteration(self, state):
        """Run one complete rollout/update, returning per-policy metrics without a host read."""
        next_state, rollout = self.collect(state)
        next_state, batch = self.prepare_batch(next_state, rollout)
        next_state, metrics = self.update(next_state, batch)
        episode_count = rollout.episode_finished.astype(jnp.int32).sum(axis=(0, 2))
        episode_return_sum = jnp.where(rollout.episode_finished, rollout.episode_returns, 0.0).sum(axis=(0, 2))
        metrics = {**metrics, "reward_ext_sum": rollout.rewards.sum(axis=(0, 2)),
                   "intrinsic_reward_mean": rollout.intrinsic_rewards.mean(axis=(0, 2)),
                   "episode_count": episode_count, "episode_return_sum": episode_return_sum,
                   "steps_per_policy": (state.iteration + 1) * self.config.batch_size}
        return next_state._replace(iteration=state.iteration + 1), metrics
