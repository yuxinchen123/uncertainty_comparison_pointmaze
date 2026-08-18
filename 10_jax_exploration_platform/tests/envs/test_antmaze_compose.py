"""AntMaze through the whole fused trainer: compose, prime, iterate, and refuse what must fail.

Checks that the platform's one-program iteration — rollout, bonus, advantages, update — compiles
and runs with the MJX environment inside it, for the bonuses that are defined on it, and that
the visit-count family (a PointMaze-only bonus) is refused at composition time rather than
producing wrong numbers.

Run: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <jax python> test_antmaze_compose.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
import exploration_platform  # noqa: E402,F401
import numpy as np  # noqa: E402
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.envs.antmaze.am_common import preset  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

SMALL = dict(n_copies=2, n_envs=2, num_steps=8, prime_iterations=1, track_coverage=True)


def test_rnd_trains_end_to_end():
    """Two full iterations with the distillation bonus: finite loss, finite rewards, coverage."""
    runner = Runner(PPOConfig(**SMALL), bonus="rnd_next_state", env_cfg=preset("umaze"))
    state, stats = runner.train(2, log_every_seconds=0, history_every=1)
    assert np.isfinite(stats["history"][-1]["reward_ext_sum_per_copy"]).all()
    assert all(np.isfinite(r["rint_mean_per_copy"]).all() for r in stats["history"])
    cov = np.asarray(stats["history"][-1]["coverage_per_copy"])
    # 2 iterations x 8 steps never leave the start cell's neighbourhood, but the start cell
    # itself must be counted
    assert (cov > 0).all(), cov
    print("ok test_rnd_trains_end_to_end")


def test_none_trains_end_to_end():
    """The no-bonus arm composes and runs on the MJX environment too."""
    runner = Runner(PPOConfig(**SMALL), bonus="none", env_cfg=preset("umaze"))
    state, stats = runner.train(2, log_every_seconds=0, history_every=2)
    assert np.isfinite(stats["history"][-1]["reward_ext_sum_per_copy"]).all()
    assert all(v == 0.0 for v in stats["history"][-1]["rint_mean_per_copy"])
    print("ok test_none_trains_end_to_end")


def test_visit_count_is_refused():
    """The PointMaze-only visit-count bonus fails at composition, with a plain message."""
    try:
        Runner(PPOConfig(**SMALL), bonus="gt_position_velocity_sqrt", env_cfg=preset("umaze"))
    except ValueError as e:
        assert "not defined for this environment" in str(e), e
        print("ok test_visit_count_is_refused")
        return
    raise AssertionError("composing visit-count with AntMaze did not raise")


if __name__ == "__main__":
    test_rnd_trains_end_to_end()
    test_none_trains_end_to_end()
    test_visit_count_is_refused()
