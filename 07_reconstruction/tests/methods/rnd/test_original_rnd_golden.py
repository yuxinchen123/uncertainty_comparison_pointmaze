"""Golden bit-identity regression + obs-RMS warmup-source tests for the train-run-5 RND switches.

The golden (`goldens/preedit_v0.pt`) was captured on the PRE-EDIT tree by `goldens/_generate.py`.
This test re-runs the SAME constructions (whose new knobs default to relu / 0 extra layers /
update_proportion 1.0) and asserts every captured tensor is byte-identical — proving the new
switches did not shift any RNG stream for the historical configs.
"""
import os
import types

import numpy as np
import pytest
import torch

from rnd_exploration.methods import EnvContext, build_intrinsic_model
from tests.methods.rnd.goldens import _generate as gen


def test_golden_bit_identity_defaults_unchanged():
    """Every representative pre-edit config reproduces its captured init / compute / post-5-update state."""
    if not os.path.exists(gen.OUT):
        pytest.skip("golden not generated (run goldens/_generate.py on the pre-edit tree)")
    golden = torch.load(gen.OUT, weights_only=True)
    # golden path: re-capture each config on the edited code and compare every tensor byte-for-byte
    for label, kwargs in gen.CONFIGS.items():
        rec = gen.capture_one(kwargs)
        for key, value in rec.items():
            ref = golden[label][key]
            assert torch.equal(value, ref), f"{label}/{key} changed after the edits"


class _FakeSpace:
    """A minimal action_space with sample()/seed() for the env-steps warmup test."""
    def __init__(self, dim):
        self.dim = dim
        self._rng = np.random.default_rng(0)

    def seed(self, s):
        """Seed the internal RNG (records that the warmup seeded the action space)."""
        self._rng = np.random.default_rng(s)
        self.seeded_with = s

    def sample(self):
        """Draw one random action vector."""
        return self._rng.standard_normal(self.dim).astype(np.float32)


class _FakeEnv:
    """A minimal gym-like env: reset()/step() over a fixed obs dim, counting steps and resets."""
    def __init__(self, obs_dim=4, action_dim=2, episode_len=5):
        self.obs_dim = obs_dim
        self.action_space = _FakeSpace(action_dim)
        self.episode_len = episode_len
        self.n_steps = 0
        self.n_resets = 0
        self._t = 0
        self._rng = np.random.default_rng(0)

    def reset(self, seed=None):
        """Reset the env; record the seed and bump the reset counter."""
        if seed is not None:
            self.reset_seed = seed
            self._rng = np.random.default_rng(seed)
        self.n_resets += 1
        self._t = 0
        return self._rng.standard_normal(self.obs_dim).astype(np.float32), {}

    def step(self, action):
        """Advance one step; terminate every episode_len steps to exercise the reset branch."""
        self.n_steps += 1
        self._t += 1
        obs = self._rng.standard_normal(self.obs_dim).astype(np.float32)
        truncated = self._t >= self.episode_len
        return obs, 0.0, False, truncated, {}


def _rnd_cfg(**kw):
    """A cfg stub with the RND fields the factory reads (obs-norm ON so warmup runs)."""
    base = dict(device="cpu", rnd_output_dim=8, rnd_obs_norm=True, rnd_distance="mse",
                n_predictors=1, a_seed=5)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_warmup_space_sample_uses_step_count():
    """space_sample warmup draws exactly rnd_obs_warmup_steps triples from the env spaces."""
    class _CountSpace:
        def __init__(self):
            self.n = 0
        def sample(self):
            self.n += 1
            return np.zeros(4, dtype=np.float32)
    obs_space, act_space = _CountSpace(), _CountSpace()
    cfg = _rnd_cfg(rnd_obs_warmup_mode="space_sample", rnd_obs_warmup_steps=37)
    ctx = EnvContext(obs_shape=(4,), action_dim=2, observation_space=obs_space,
                     action_space=act_space, position_wrapper=None, position_velocity_wrapper=None)
    model = build_intrinsic_model("rnd_next_state", cfg, ctx)
    # golden path: obs+next each sampled 37 times (2*37), actions 37 times; obs_rms saw 37 rows
    assert obs_space.n == 2 * 37 and act_space.n == 37
    assert model.obs_rms.count == pytest.approx(37 + 1e-4, abs=1e-6)  # RMS starts count at epsilon 1e-4


def test_warmup_env_steps_rolls_random_agent_with_keyed_seed():
    """env_steps warmup rolls the fresh env for rnd_obs_warmup_steps steps with a keyed seed."""
    import hashlib
    env = _FakeEnv(obs_dim=4, action_dim=2, episode_len=5)
    cfg = _rnd_cfg(rnd_obs_warmup_mode="env_steps", rnd_obs_warmup_steps=12, a_seed=5)
    ctx = EnvContext(obs_shape=(4,), action_dim=2, observation_space=None, action_space=None,
                     position_wrapper=None, position_velocity_wrapper=None, env=env)
    model = build_intrinsic_model("rnd_next_state", cfg, ctx)
    # golden path: the env was stepped exactly 12 times and obs_rms was updated (count grew)
    assert env.n_steps == 12
    assert model.obs_rms.count == pytest.approx(12 + 1e-4, abs=1e-6)
    # the warmup used the (a_seed, "rnd_obs_warmup") keyed seed, not the bare a_seed
    expected = int(hashlib.sha256(b"5::rnd_obs_warmup").hexdigest(), 16) & 0x7FFFFFFF
    assert env.reset_seed == expected and env.action_space.seeded_with == expected
    # it reset at least once (12 steps / episode_len 5) beyond the initial reset
    assert env.n_resets >= 3
    # edge case: env_steps mode with no env raises rather than silently skipping the warmup
    cfg_no_env = _rnd_cfg(rnd_obs_warmup_mode="env_steps", rnd_obs_warmup_steps=4)
    ctx_no_env = EnvContext(obs_shape=(4,), action_dim=2, observation_space=None, action_space=None,
                            position_wrapper=None, position_velocity_wrapper=None, env=None)
    with pytest.raises(ValueError):
        build_intrinsic_model("rnd_next_state", cfg_no_env, ctx_no_env)
