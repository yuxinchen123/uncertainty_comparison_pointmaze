"""Unit tests for checkpointing: a saved run restores to the same state, and resume is exact.

The tests use small stand-in modules rather than the real Agent and RNDModel, because what is under
test is the save/restore plumbing, not the networks.
"""

import os
import random
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from checkpointing import load_checkpoint, save_checkpoint  # noqa: E402


class TinyRMS:
    """A stand-in for gym's RunningMeanStd carrying the same three fields."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4


class TinyFilter:
    """A stand-in for RewardForwardFilter, which holds one running array."""

    def __init__(self):
        self.rewems = None


def build_pieces():
    """Build a small agent, RND model, optimizer and normalisers to save and restore."""
    agent = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    rnd = nn.ModuleDict({"predictor": nn.Linear(4, 3), "target": nn.Linear(4, 3)})
    opt = torch.optim.Adam(list(agent.parameters()) + list(rnd["predictor"].parameters()), lr=1e-3)
    return agent, rnd, opt, TinyRMS((1, 4)), TinyRMS(()), TinyFilter()


def test_save_then_load_restores_weights_and_counters(tmp_path):
    """Golden path: weights, optimizer moments, normalisers and counters all come back."""
    path = str(tmp_path / "ckpt.pt")
    agent, rnd, opt, obs_rms, rew_rms, filt = build_pieces()
    device = torch.device("cpu")

    # Take one optimizer step so Adam has non-trivial moment buffers to restore.
    loss = agent(torch.ones(2, 4)).sum() + rnd["predictor"](torch.ones(2, 4)).sum()
    loss.backward()
    opt.step()

    # Put recognisable values into the normalisers and the filter.
    obs_rms.mean[:] = 3.5
    obs_rms.var[:] = 2.25
    obs_rms.count = 1234.0
    rew_rms.var = np.float64(9.0)
    filt.rewems = np.array([0.1, 0.2, 0.3])

    info = save_checkpoint(
        path, agent=agent, rnd_model=rnd, optimizer=opt, obs_rms=obs_rms, reward_rms=rew_rms,
        discounted_reward=filt, global_step=524288, update=32, avg_returns=[1.0, 2.0], device=device,
    )
    assert info["bytes"] > 0 and info["replaced_previous"] is False
    assert os.stat(path).st_mode & 0o060 == 0o060

    # Restore into freshly built objects and check every field came back.
    agent2, rnd2, opt2, obs2, rew2, filt2 = build_pieces()
    state = load_checkpoint(
        path, agent=agent2, rnd_model=rnd2, optimizer=opt2, obs_rms=obs2, reward_rms=rew2,
        discounted_reward=filt2, device=device,
    )
    assert state["global_step"] == 524288 and state["update"] == 32
    assert state["avg_returns"] == [1.0, 2.0]
    for p1, p2 in zip(agent.parameters(), agent2.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(rnd.parameters(), rnd2.parameters()):
        assert torch.equal(p1, p2)
    assert np.allclose(obs2.mean, 3.5) and np.allclose(obs2.var, 2.25) and obs2.count == 1234.0
    assert np.isclose(rew2.var, 9.0)
    assert np.allclose(filt2.rewems, [0.1, 0.2, 0.3])
    # Adam's exponential moving averages must survive, or the resumed run takes different steps.
    m1 = opt.state_dict()["state"][0]["exp_avg"]
    m2 = opt2.state_dict()["state"][0]["exp_avg"]
    assert torch.equal(m1, m2)


def test_random_streams_continue_identically(tmp_path):
    """A restored run draws the same random numbers the uninterrupted run would have drawn."""
    path = str(tmp_path / "ckpt.pt")
    agent, rnd, opt, obs_rms, rew_rms, filt = build_pieces()
    device = torch.device("cpu")

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    save_checkpoint(
        path, agent=agent, rnd_model=rnd, optimizer=opt, obs_rms=obs_rms, reward_rms=rew_rms,
        discounted_reward=filt, global_step=0, update=1, avg_returns=[], device=device,
    )
    # before: draw three numbers straight after the save
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    # after: disturb every stream, then restore and draw again — the numbers must match
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    random.random(), np.random.rand(), torch.rand(1)

    agent2, rnd2, opt2, obs2, rew2, filt2 = build_pieces()
    load_checkpoint(
        path, agent=agent2, rnd_model=rnd2, optimizer=opt2, obs_rms=obs2, reward_rms=rew2,
        discounted_reward=filt2, device=device,
    )
    got = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert got == pytest.approx(expected)


def test_load_on_missing_file_reports_fresh_start(tmp_path):
    """Edge case: no checkpoint on disk is a fresh run, not an error."""
    agent, rnd, opt, obs_rms, rew_rms, filt = build_pieces()
    assert load_checkpoint(
        str(tmp_path / "absent.pt"), agent=agent, rnd_model=rnd, optimizer=opt, obs_rms=obs_rms,
        reward_rms=rew_rms, discounted_reward=filt, device=torch.device("cpu"),
    ) is None


def test_second_save_replaces_the_first(tmp_path):
    """Only one checkpoint file exists per run; the second save overwrites the first."""
    path = str(tmp_path / "ckpt.pt")
    agent, rnd, opt, obs_rms, rew_rms, filt = build_pieces()
    device = torch.device("cpu")
    common = dict(agent=agent, rnd_model=rnd, optimizer=opt, obs_rms=obs_rms, reward_rms=rew_rms,
                  discounted_reward=filt, avg_returns=[], device=device)
    save_checkpoint(path, global_step=100, update=1, **common)
    info = save_checkpoint(path, global_step=200, update=2, **common)
    assert info["replaced_previous"] is True
    assert sorted(os.listdir(tmp_path)) == ["ckpt.pt"]
    state = load_checkpoint(path, agent=agent, rnd_model=rnd, optimizer=opt, obs_rms=obs_rms,
                            reward_rms=rew_rms, discounted_reward=filt, device=device)
    assert state["global_step"] == 200
