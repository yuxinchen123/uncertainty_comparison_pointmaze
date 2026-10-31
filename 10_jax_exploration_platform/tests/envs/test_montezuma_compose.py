"""Montezuma through the whole fused trainer, with the discrete actor chosen at compose time.

Run: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <jax python> test_montezuma_compose.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
import exploration_platform  # noqa: E402,F401
import numpy as np  # noqa: E402
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.envs.atari_montezuma.jax_montezuma import MontezumaConfig  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

SMALL = dict(n_copies=2, n_envs=2, num_steps=4, prime_iterations=1, track_coverage=True)


def test_rnd_trains_end_to_end():
    """Two full iterations with the distillation bonus: finite metrics, room coverage."""
    runner = Runner(PPOConfig(**SMALL), bonus="rnd_next_state", env_cfg=MontezumaConfig())
    assert runner.composition.env.action_kind == "discrete"
    state, stats = runner.train(2, log_every_seconds=0, history_every=1)
    h = stats["history"][-1]
    assert np.isfinite(h["reward_ext_sum_per_copy"]).all()
    assert np.isfinite(h["rint_mean_per_copy"]).all()
    cov = np.asarray(h["coverage_per_copy"])
    assert (cov >= 1.0 / 24).all(), cov  # at least the start room is counted
    print("ok test_rnd_trains_end_to_end")


def test_none_trains_end_to_end():
    """The no-bonus arm composes and runs with the discrete actor too."""
    runner = Runner(PPOConfig(**SMALL), bonus="none", env_cfg=MontezumaConfig())
    state, stats = runner.train(2, log_every_seconds=0, history_every=2)
    assert all(v == 0.0 for v in stats["history"][-1]["rint_mean_per_copy"])
    print("ok test_none_trains_end_to_end")


def test_visit_count_is_refused():
    """The PointMaze-only visit-count bonus fails at composition, with a plain message."""
    try:
        Runner(PPOConfig(**SMALL), bonus="gt_position_velocity_sqrt",
               env_cfg=MontezumaConfig())
    except ValueError as e:
        assert "not defined for this environment" in str(e), e
        print("ok test_visit_count_is_refused")
        return
    raise AssertionError("composing visit-count with Montezuma did not raise")


if __name__ == "__main__":
    test_rnd_trains_end_to_end()
    test_none_trains_end_to_end()
    test_visit_count_is_refused()
