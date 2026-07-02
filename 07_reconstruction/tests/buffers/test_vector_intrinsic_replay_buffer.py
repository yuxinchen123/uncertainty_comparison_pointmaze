"""Tests for VectorIntrinsicReplayBuffer (recomputes intrinsic reward on sample())."""
import numpy as np
import torch
import gymnasium as gym
import pytest

from rnd_exploration.buffers.vector_intrinsic_replay_buffer import (
    VectorIntrinsicReplayBuffer,
)

# Small fixed shapes so the whole suite runs in well under a second.
OBS_DIM = 3
ACT_DIM = 2
BUFFER_SIZE = 8


class FakeIntrinsicModel:
    """Fake intrinsic reward model: compute() returns a known constant per sample; update()/observe()
    record their calls (update on sample(), observe on add())."""

    def __init__(self, constant: float):
        # Constant intrinsic reward returned for every transition in a batch.
        self.constant = constant
        # Bookkeeping the tests assert against.
        self.update_called = False
        self.update_count = 0
        self.observe_called = False
        self.observe_count = 0
        self.last_observe_batch = None
        self.last_compute_keys = None
        self.last_compute_batch = None

    def compute(self, samples):
        # Return one constant intrinsic value per sampled transition; record the dict keys seen.
        n = samples["observations"].shape[0]
        self.last_compute_keys = set(samples.keys())
        self.last_compute_batch = n
        return torch.full((n,), float(self.constant), dtype=torch.float32)

    def update(self, samples):
        # Record that the buffer invoked update() during sample().
        self.update_called = True
        self.update_count += 1

    def observe(self, samples):
        # Record that the buffer invoked observe() during add() with the freshly added transition(s).
        self.observe_called = True
        self.observe_count += 1
        self.last_observe_batch = samples["observations"].shape[0]


def make_spaces():
    """Build small Box observation/action spaces for the buffer."""
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
    return obs_space, act_space


def make_buffer(beta, intrinsic_reward_model):
    """Construct a VectorIntrinsicReplayBuffer with the SB3 ReplayBuffer signature (n_envs=1, cpu)."""
    obs_space, act_space = make_spaces()
    return VectorIntrinsicReplayBuffer(
        buffer_size=BUFFER_SIZE,
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        n_envs=1,
        beta=beta,
        intrinsic_reward_model=intrinsic_reward_model,
    )


def add_transition(buffer, reward):
    """Add one (obs, next_obs, action, reward, done) transition with n_envs=1 shapes."""
    # SB3 ReplayBuffer.add expects leading n_envs axis: obs (1, OBS_DIM), action (1, ACT_DIM),
    # reward (1,), done (1,), infos a length-1 list of dicts.
    # before: scalar reward (e.g. 1.0)  ->  after: reward array shape (1,) -> stored at buffer.rewards[pos, 0]
    obs = np.zeros((1, OBS_DIM), dtype=np.float32)
    next_obs = np.ones((1, OBS_DIM), dtype=np.float32)
    action = np.zeros((1, ACT_DIM), dtype=np.float32)
    reward_arr = np.array([reward], dtype=np.float32)
    done = np.array([False])
    buffer.add(obs, next_obs, action, reward_arr, done, [{}])


def test_add_stores_raw_extrinsic_reward():
    """add() stores the raw env (extrinsic) reward unchanged, with no intrinsic bonus mixed in."""
    # Use a model + nonzero beta to prove add() does NOT pre-add the intrinsic term.
    model = FakeIntrinsicModel(constant=100.0)
    buffer = make_buffer(beta=1.0, intrinsic_reward_model=model)
    # Add transitions with distinct extrinsic rewards.
    extrinsics = [0.5, -2.0, 3.25]
    for r in extrinsics:
        add_transition(buffer, r)
    # buffer.rewards has shape (BUFFER_SIZE, n_envs); column 0 holds the stored raw extrinsic.
    stored = buffer.rewards[: len(extrinsics), 0]
    np.testing.assert_allclose(stored, np.array(extrinsics, dtype=np.float32), rtol=0, atol=1e-6)
    # Storing must not have run a sample-time update (that happens in sample(), not add()).
    assert model.update_called is False


def test_add_routes_to_observe():
    """add() feeds each freshly added transition to the model's observe() (the add-time update hook),
    once per add and with the n_envs batch size; sample-time update() is untouched by add()."""
    model = FakeIntrinsicModel(constant=0.0)
    buffer = make_buffer(beta=1.0, intrinsic_reward_model=model)
    # Golden path: three adds -> three observe() calls, each carrying the single (n_envs=1) transition.
    for r in (0.5, -2.0, 3.25):
        add_transition(buffer, r)
    assert model.observe_count == 3
    assert model.observe_called is True
    assert model.last_observe_batch == 1
    # add() must not have triggered the sample-time update.
    assert model.update_called is False


def test_add_skips_observe_when_no_model_or_beta_zero():
    """observe() is skipped when there is no model or beta <= 0 (the add-time hook mirrors the
    sample-time short-circuit), so neither configuration crashes on add()."""
    # beta = 0 with a model present: add() must not call observe().
    model = FakeIntrinsicModel(constant=1.0)
    buffer_zero_beta = make_buffer(beta=0.0, intrinsic_reward_model=model)
    add_transition(buffer_zero_beta, 1.0)
    assert model.observe_called is False
    # no model: add() simply stores (no observe routing, no crash).
    buffer_no_model = make_buffer(beta=1.0, intrinsic_reward_model=None)
    add_transition(buffer_no_model, 1.0)


def test_sample_reward_is_extrinsic_plus_beta_times_intrinsic():
    """sample() returns reward = extrinsic + beta*intrinsic when every transition shares one extrinsic value."""
    np.random.seed(0)
    extrinsic = 1.0
    constant = 4.0
    beta = 0.5
    model = FakeIntrinsicModel(constant=constant)
    buffer = make_buffer(beta=beta, intrinsic_reward_model=model)
    # Fill several transitions all with the same extrinsic, so any sampled index yields the same expected reward.
    for _ in range(5):
        add_transition(buffer, extrinsic)
    batch_size = 6
    samples = buffer.sample(batch_size)
    # Reward shape must be (batch_size, 1) so the SB3 critic MSE does not broadcast-mismatch.
    assert tuple(samples.rewards.shape) == (batch_size, 1)
    # Every sampled reward equals extrinsic + beta*constant = 1.0 + 0.5*4.0 = 3.0.
    expected = extrinsic + beta * constant
    got = samples.rewards.cpu().numpy().ravel()
    np.testing.assert_allclose(got, np.full(batch_size, expected, dtype=np.float32), atol=1e-6)
    # The intrinsic model was both queried and updated during sample().
    assert model.update_called is True
    assert model.update_count == 1
    # compute() received the minimal samples dict with exactly these three keys.
    assert model.last_compute_keys == {"observations", "next_observations", "actions"}
    assert model.last_compute_batch == batch_size


def test_sample_exact_value_single_transition():
    """With one buffered transition (all sampled indices = 0), sampled reward exactly equals extrinsic + beta*intrinsic."""
    np.random.seed(1)
    extrinsic = 2.5
    constant = -3.0
    beta = 2.0
    model = FakeIntrinsicModel(constant=constant)
    buffer = make_buffer(beta=beta, intrinsic_reward_model=model)
    # Only one transition -> upper_bound=1 -> every drawn index is 0 -> exact known extrinsic.
    add_transition(buffer, extrinsic)
    samples = buffer.sample(batch_size=4)
    # expected = 2.5 + 2.0*(-3.0) = -3.5 for all entries.
    expected = extrinsic + beta * constant
    got = samples.rewards.cpu().numpy().ravel()
    np.testing.assert_allclose(got, np.full(4, expected, dtype=np.float32), atol=1e-6)


def test_sample_beta_zero_returns_extrinsic_only():
    """Edge case: beta <= 0 short-circuits, so reward is the raw extrinsic and the intrinsic model is never touched."""
    np.random.seed(2)
    extrinsic = 1.75
    model = FakeIntrinsicModel(constant=999.0)
    buffer = make_buffer(beta=0.0, intrinsic_reward_model=model)
    # Single transition for an exact comparison.
    add_transition(buffer, extrinsic)
    samples = buffer.sample(batch_size=3)
    got = samples.rewards.cpu().numpy().ravel()
    np.testing.assert_allclose(got, np.full(3, extrinsic, dtype=np.float32), atol=1e-6)
    # beta<=0 must skip compute() and update() entirely.
    assert model.update_called is False
    assert model.last_compute_keys is None


def test_sample_no_intrinsic_model_returns_extrinsic_only():
    """Edge case: intrinsic_reward_model is None -> sample() returns the unmodified extrinsic batch."""
    np.random.seed(3)
    extrinsic = -0.25
    buffer = make_buffer(beta=1.0, intrinsic_reward_model=None)
    # Single transition for an exact comparison.
    add_transition(buffer, extrinsic)
    samples = buffer.sample(batch_size=2)
    got = samples.rewards.cpu().numpy().ravel()
    np.testing.assert_allclose(got, np.full(2, extrinsic, dtype=np.float32), atol=1e-6)
    # SB3 returns reward shape (batch_size, 1) even on the short-circuit path.
    assert tuple(samples.rewards.shape) == (2, 1)
