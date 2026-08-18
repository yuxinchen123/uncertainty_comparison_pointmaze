"""Unit tests for the coin-flip pseudo-count intrinsic model (11_decay_rate campaign winner)."""
import numpy as np
import pytest

from rnd_exploration.methods import ALGORITHM_NAMES, REGISTRY
from rnd_exploration.methods.coinflip_count import CoinFlipCount


def _samples(xy):
    """Build a samples dict whose next observations put (x, y) in the first two coordinates."""
    xy = np.asarray(xy, dtype=np.float32)
    obs = np.concatenate([xy, np.zeros_like(xy)], axis=1)
    return {"observations": obs, "next_observations": obs, "actions": np.zeros((len(xy), 2))}


def test_registered():
    """The algorithm is in the registry and inherits the standard name list."""
    assert "coinflip_count" in ALGORITHM_NAMES
    assert REGISTRY["coinflip_count"].kind == "coinflip"


def test_bonus_starts_at_one_everywhere():
    """Golden path: before any visit the bonus is exactly 1 at every input."""
    m = CoinFlipCount(obs_shape=(4,), seed=0)
    b = m.compute(_samples([[0.0, 0.0], [4.5, 3.0], [-5.5, -4.0]]))
    assert np.allclose(b, 1.0)


def test_visited_position_decays_with_its_own_count():
    """A visited position's bonus falls toward m^(-1/2); an unvisited far position stays ~1."""
    m = CoinFlipCount(obs_shape=(4,), solve_every=1, seed=0)
    visited, far = [0.0, 0.0], [5.5, 4.0]
    for _ in range(16):
        m.observe(_samples([visited]))
    b = m.compute(_samples([visited, far]))
    assert b[0] < 0.45  # 16 visits: ideal 0.25, allow chi noise at d=128
    assert b[1] > 0.8   # never visited, beyond the kernel bandwidth

def test_solve_cache_refreshes():
    """Edge case: with solve_every large, compute() still refreshes after dictionary growth."""
    m = CoinFlipCount(obs_shape=(4,), solve_every=10**9, seed=0)
    m.observe(_samples([[0.0, 0.0]]))
    b1 = m.compute(_samples([[0.0, 0.0]]))  # W solved lazily (was None after growth)
    assert b1.shape == (1,) and np.isfinite(b1).all()
