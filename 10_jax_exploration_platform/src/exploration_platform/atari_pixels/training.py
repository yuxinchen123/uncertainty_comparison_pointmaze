"""Original CleanRL PPO/RND equations with independent copies and a pure JAX environment."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import PixelPPOConfig
from .networks import (categorical_statistics, initialize_networks, policy_forward,
                       quantity_keys, rnd_forward)
from .state import (PixelTrainState, adam_step, initialize_adam, initialize_statistics,
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


def flatten_rollout(values):
    """Turn [time,copies,environments,...] into [copies,time*environments,...]."""
    # Before: [128,2,128,4,84,84]; after: [2,16384,4,84,84], without mixing policies.
    axes = (1, 0, 2) + tuple(range(3, values.ndim))
    return values.transpose(axes).reshape(values.shape[1], -1, *values.shape[3:])


def intrinsic_reward(parameters, target, frames, statistics):
    """Use half the SUM of feature errors for the rollout bonus, as upstream does."""
    normalized = normalize_frames(frames, statistics)
    differences = rnd_forward(target, normalized) - rnd_forward(parameters["predictor"], normalized)
    return jnp.sum(differences ** 2, axis=-1) * 0.5


def normalize_intrinsic_rewards(rewards, state, config):
    """Retain upstream's environment-axis filter and count, or explicitly select time order."""
    # Upstream transposes [T,N] and iterates N vectors of length T; each policy is independent.
    values = rewards.transpose(2, 1, 0) if config.upstream_reward_filter_axis else rewards

    def accumulate(previous, reward):
        """Carry the unreset discounted intrinsic filter along the selected upstream axis."""
        filtered = previous * jnp.asarray(config.gamma_int, jnp.float32) + reward
        return filtered, filtered

    filtered_state, filtered = jax.lax.scan(accumulate, state.reward_filter, values)
    statistics = update_moments(
        state.reward_statistics,
        filtered.astype(jnp.float64).mean(axis=(0, 2)),
        filtered.astype(jnp.float64).var(axis=(0, 2)),
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
    _, advantages = jax.lax.scan(reverse_step, (zeros, zeros), (delta_ext, delta_int, masks), reverse=True)
    return advantages


def minibatch_loss(parameters, target, batch, predictor_mask, config,
                   intrinsic_coefficients):
    """Return the sum of per-policy losses and independent loss diagnostics."""
    logits, value_ext, value_int = policy_forward(parameters["agent"], batch["observations"])
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
    normalized_frames = batch["normalized_frames"]
    target_features = (batch["target_features"] if config.cache_target_features
                       else rnd_forward(target, normalized_frames))
    target_features = jax.lax.stop_gradient(target_features)
    predictor_features = rnd_forward(parameters["predictor"], normalized_frames)
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
            statistics = update_observation_statistics(statistics, frames)
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

        def step(carry, step_index):
            """Record one transition and preserve its next pixels for the RND bonus."""
            environment_state, observations, done_before = carry
            logits, values_ext, values_int = policy_forward(state.parameters["agent"], observations)
            keys = quantity_keys(state.copy_seeds, "rollout/actions", state.iteration, step_index)
            actions = jax.vmap(lambda key, value: jax.random.categorical(key, value).astype(jnp.int32))(keys, logits)
            log_probability, _ = categorical_statistics(logits, actions)
            next_environment, next_observations, reward, done, information = self.environment.step(
                environment_state, actions)
            frames = next_observations[..., 3:4, :, :]
            bonus = (jnp.zeros_like(reward) if config.hoist_rollout_bonus else intrinsic_reward(
                state.parameters, state.target, frames, state.observation_statistics))
            recorded = Rollout(observations, frames, actions, log_probability, reward, bonus,
                               done_before, values_ext, values_int, information["episode_return"],
                               information["episode_done"])
            return (next_environment, next_observations, done), recorded

        (environment_state, observations, done), rollout = jax.lax.scan(
            step, (state.environment, state.observations, state.done), jnp.arange(config.num_steps),
            unroll=config.rollout_unroll)
        if config.hoist_rollout_bonus:
            bonuses = intrinsic_reward(state.parameters, state.target,
                                       flatten_rollout(rollout.next_frames), state.observation_statistics)
            # Before: [C,T*N]; after: [T,C,N], restoring the exact temporal filter inputs.
            bonuses = bonuses.reshape(config.n_copies, config.num_steps, config.n_envs).transpose(1, 0, 2)
            rollout = rollout._replace(intrinsic_rewards=bonuses)
        return state._replace(environment=environment_state, observations=observations, done=done), rollout

    def prepare_batch(self, state, rollout):
        """Compute upstream normalization, GAE, current-frame RND inputs and fixed-size batches."""
        config = self.config
        normalized_bonus, reward_statistics, reward_filter = normalize_intrinsic_rewards(
            rollout.intrinsic_rewards, state, config)
        _, final_ext, final_int = policy_forward(state.parameters["agent"], state.observations)
        advantage_ext, advantage_int = generalized_advantages(
            rollout.rewards, normalized_bonus, rollout, (final_ext, final_int), state.done, config)
        observations = flatten_rollout(rollout.observations)
        frames = observations[..., 3:4, :, :]
        observation_statistics = update_observation_statistics(state.observation_statistics, frames)
        batch = {"observations": observations, "actions": flatten_rollout(rollout.actions),
                 "log_probabilities": flatten_rollout(rollout.log_probabilities),
                 "values_ext": flatten_rollout(rollout.values_ext),
                 "advantage_ext": flatten_rollout(advantage_ext),
                 "advantage_int": flatten_rollout(advantage_int),
                 "return_ext": flatten_rollout(advantage_ext + rollout.values_ext),
                 "return_int": flatten_rollout(advantage_int + rollout.values_int),
                 "normalized_frames": normalize_frames(frames, observation_statistics)}
        if config.cache_target_features:
            # Training targets use CURRENT frames and UPDATED statistics, unlike rollout targets.
            batch["target_features"] = jax.lax.stop_gradient(rnd_forward(state.target, batch["normalized_frames"]))
        return state._replace(observation_statistics=observation_statistics,
                              reward_statistics=reward_statistics, reward_filter=reward_filter), batch

    def update(self, state, batch):
        """Perform four independently shuffled epochs of four Adam steps per policy."""
        config = self.config
        fraction = (1.0 - state.iteration.astype(jnp.float32) / config.total_iterations
                    if config.anneal_learning_rate else jnp.asarray(1.0, jnp.float32))
        learning_rates = self.learning_rates * fraction

        def epoch(carry, epoch_index):
            """Give every policy a separate permutation before its four minibatches."""
            keys = quantity_keys(state.copy_seeds, "update/permutation", state.iteration, epoch_index)
            permutations = jax.vmap(lambda key: jax.random.permutation(key, config.batch_size))(keys)
            # Before: [C,B]; after: [minibatches,C,minibatch_size], retaining each policy's rows.
            indices = permutations.reshape(config.n_copies, config.num_minibatches,
                                             config.minibatch_size).transpose(1, 0, 2)

            def minibatch(carry, values):
                """Gather one fixed-size minibatch, differentiate its losses and apply Adam."""
                parameters, optimizer = carry
                minibatch_index, indices = values
                selected = jax.tree.map(lambda value: jax.vmap(lambda rows, index: rows[index])(value, indices), batch)
                keys = quantity_keys(state.copy_seeds, "update/predictor-mask", state.iteration,
                                     epoch_index, minibatch_index)
                masks = jax.vmap(lambda key: (jax.random.uniform(key, (config.minibatch_size,),
                    dtype=jnp.float32) < config.predictor_update_proportion).astype(jnp.float32))(keys)

                def loss(parameters):
                    """Differentiate only the trainable networks, treating target features as constants."""
                    return minibatch_loss(parameters, state.target, selected, masks, config,
                                          self.intrinsic_coefficients)

                (_, metrics), gradients = jax.value_and_grad(loss, has_aux=True)(parameters)
                parameters, optimizer, gradient_norm = adam_step(
                    parameters, gradients, optimizer, learning_rates,
                    config.max_gradient_norm, config.adam_epsilon)
                return (parameters, optimizer), {**metrics, "gradient_norm": gradient_norm}

            return jax.lax.scan(minibatch, carry, (jnp.arange(config.num_minibatches), indices),
                                 unroll=config.update_unroll)

        (parameters, optimizer), diagnostics = jax.lax.scan(
            epoch, (state.parameters, state.optimizer), jnp.arange(config.update_epochs))
        metrics = jax.tree.map(lambda value: value.mean(axis=(0, 1)), diagnostics)
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
