"""Tests for the oracle visit-count intrinsic reward model (rnd_exploration.methods.visit_count.VisitCount).

VisitCount reads counts from a visit-count wrapper via observation_to_count(obs) -> int, then maps each
count to min(1, count**intrinsic_decay_rate); update() is a no-op. These tests use a tiny stub wrapper
exposing only that one method, so no full PointMaze env is needed.
"""
import numpy as np
import pytest
import torch

from rnd_exploration.methods.visit_count import VisitCount


class FakeVisitCountWrapper:
    """Stub standing in for PositionVisitCountWrapper; maps int(round(obs[0])) -> a preset count.

    Only exposes observation_to_count(obs) -> int, the single attribute VisitCount.compute reads.
    Records every observation it is handed so a test can assert compute iterates once per batch row.
    """

    def __init__(self, counts_by_first_coord):
        # counts_by_first_coord: {first_coord_int: count}; lets each batch row select its own count
        self.counts_by_first_coord = counts_by_first_coord
        self.calls = []

    def observation_to_count(self, obs):
        # record the row (copy so later in-place reuse can't mutate history) and look up its count
        arr = np.asarray(obs, dtype=float)
        self.calls.append(arr.copy())
        return self.counts_by_first_coord[int(round(arr[0]))]


def _batch(first_coords, obs_dim=2):
    """Build an (N, obs_dim) observation batch whose first column equals first_coords (rest zeros).

    BEFORE: first_coords = [0, 1, 2]
    AFTER:  array([[0,0],[1,0],[2,0]])  -> the stub maps row i to counts_by_first_coord[first_coords[i]]
    """
    obs = np.zeros((len(first_coords), obs_dim))
    obs[:, 0] = first_coords
    return obs


def test_compute_returns_min_one_count_decay_bonus():
    """Golden path: each count maps to min(1, count**-0.5); edge: count==1 sits exactly at the 1.0 cap."""
    # counts 4, 100, 1 -> 4**-0.5=0.5, 100**-0.5=0.1, 1**-0.5=1.0 (the boundary count)
    wrapper = FakeVisitCountWrapper({0: 4, 1: 100, 2: 1})
    model = VisitCount(wrapper, intrinsic_decay_rate=-0.5)
    out = model.compute({"next_observations": _batch([0, 1, 2])})
    np.testing.assert_allclose(out, [0.5, 0.1, 1.0], atol=1e-12)


def test_compute_zero_and_negative_count_give_max_bonus():
    """Golden path: an unvisited state (count 0) yields the max bonus 1.0; edge: a negative sentinel count also clamps to 1.0."""
    # count<=0 branch returns 1.0 without evaluating 0**decay (which would be inf/undefined)
    wrapper = FakeVisitCountWrapper({0: 0, 1: -3})
    model = VisitCount(wrapper, intrinsic_decay_rate=-0.5)
    out = model.compute({"next_observations": _batch([0, 1])})
    np.testing.assert_allclose(out, [1.0, 1.0], atol=1e-12)


def test_compute_positive_decay_always_caps_at_one():
    """Golden path: positive decay makes count**decay>=1 for visited states, so min(1,...) stays 1.0; edge: decay 0 (count**0==1) also gives 1.0."""
    # decay +0.5: 4**0.5=2 -> min(1,2)=1.0, 9**0.5=3 -> 1.0
    wrapper = FakeVisitCountWrapper({0: 4, 1: 9})
    model_pos = VisitCount(wrapper, intrinsic_decay_rate=0.5)
    np.testing.assert_allclose(model_pos.compute({"next_observations": _batch([0, 1])}), [1.0, 1.0], atol=1e-12)
    # decay 0: any positive count**0 == 1.0 -> capped to 1.0
    model_zero = VisitCount(wrapper, intrinsic_decay_rate=0.0)
    np.testing.assert_allclose(model_zero.compute({"next_observations": _batch([0, 1])}), [1.0, 1.0], atol=1e-12)


def test_compute_default_decay_is_inverse_sqrt():
    """Golden path: the constructor's default intrinsic_decay_rate (-0.5) gives 1/sqrt(n); edge: large count -> small but positive bonus."""
    # not passing intrinsic_decay_rate exercises the default; 10000**-0.5 = 0.01
    wrapper = FakeVisitCountWrapper({0: 4, 1: 10000})
    model = VisitCount(wrapper)
    out = model.compute({"next_observations": _batch([0, 1])})
    np.testing.assert_allclose(out, [0.5, 0.01], atol=1e-12)


def test_compute_reshapes_single_1d_observation():
    """Golden path: a 2D batch returns one bonus per row; edge: a flat 1D observation is reshaped to (1, obs_dim) and returns shape (1,)."""
    # 1D input (obs_dim,) must be reshaped to a single-row batch, not iterated element-wise
    wrapper = FakeVisitCountWrapper({3: 9})
    model = VisitCount(wrapper, intrinsic_decay_rate=-0.5)
    out = model.compute({"next_observations": np.array([3.0, 7.0])})
    assert out.shape == (1,)
    assert out[0] == pytest.approx(9.0 ** -0.5)


def test_compute_accepts_torch_tensor_next_observations():
    """Golden path: numpy next_observations work; edge: a torch tensor (to_numpy path) yields the same bonuses."""
    # to_numpy() should convert the tensor before indexing rows; result must match the numpy case
    wrapper = FakeVisitCountWrapper({0: 4, 1: 100})
    model = VisitCount(wrapper, intrinsic_decay_rate=-0.5)
    np_out = model.compute({"next_observations": _batch([0, 1])})
    torch_out = model.compute({"next_observations": torch.tensor(_batch([0, 1]), dtype=torch.float32)})
    np.testing.assert_allclose(torch_out, np_out, atol=1e-6)
    np.testing.assert_allclose(torch_out, [0.5, 0.1], atol=1e-6)


def test_compute_calls_observation_to_count_once_per_row():
    """Golden path: compute returns shape (N,) and queries the wrapper once per row with that row's values; edge: a single-row batch makes exactly one call."""
    # multi-row batch: one lookup per row, in order, passing the actual row contents
    wrapper = FakeVisitCountWrapper({0: 4, 1: 9, 2: 0})
    model = VisitCount(wrapper, intrinsic_decay_rate=-0.5)
    obs = _batch([0, 1, 2])
    out = model.compute({"next_observations": obs})
    assert out.shape == (3,)
    assert len(wrapper.calls) == 3
    for recorded, expected_row in zip(wrapper.calls, obs):
        np.testing.assert_allclose(recorded, expected_row)
    # edge: single-row batch -> exactly one wrapper query
    wrapper.calls.clear()
    single = model.compute({"next_observations": _batch([1])})
    assert single.shape == (1,) and len(wrapper.calls) == 1


def test_count_to_bonus_formula_directly():
    """Golden path: _count_to_bonus matches min(1, count**decay) for a visited count; edge: count<=0 short-circuits to 1.0."""
    # exercises the helper in isolation so the formula is pinned independent of compute's batching
    model = VisitCount(FakeVisitCountWrapper({}), intrinsic_decay_rate=-1.0)
    assert model._count_to_bonus(4) == pytest.approx(0.25)  # 4**-1 = 0.25
    assert model._count_to_bonus(0) == 1.0
    assert model._count_to_bonus(-5) == 1.0


def test_update_is_noop():
    """Golden path: update returns None and never touches the wrapper; edge: it is safe to call with empty samples."""
    # counts are maintained by the env wrapper, so update must not query or mutate it
    wrapper = FakeVisitCountWrapper({0: 4})
    model = VisitCount(wrapper, intrinsic_decay_rate=-0.5)
    assert model.update({"next_observations": _batch([0])}) is None
    assert model.update({}) is None
    assert wrapper.calls == []
