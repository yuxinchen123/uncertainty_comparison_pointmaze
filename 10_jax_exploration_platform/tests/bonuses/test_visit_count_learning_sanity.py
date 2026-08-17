"""The visit-count bonus has to behave like a visit-count bonus during a real run.

Three things, on a short run of the composed program rather than on the family in isolation:

  1. the bonus a heavily visited state earns is below the bonus an unvisited state earns,
  2. the mean intrinsic reward falls as the table fills,
  3. the copies keep reaching cells they had not reached, so the table is being used by a run that
     is actually moving rather than by one standing still.

Run: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <jax python> test_visit_count_learning_sanity.py
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

ITERATIONS = 20
CONFIG = dict(n_copies=4, n_envs=4, num_steps=128, prime_iterations=1,
              update_style="full_batch", track_coverage=True)


def short_run(preset: str):
    """A few iterations of the composed program; returns the runner, the state and the history."""
    runner = Runner(PPOConfig(**CONFIG), bonus=preset)
    state = runner.prime(runner.init_state(run_seed=2))
    intrinsic, coverage = [], []
    for iteration in range(1, ITERATIONS + 1):
        state, metrics = runner.iterate(state, runner.lr_argument(iteration, ITERATIONS))
        intrinsic.append(float(np.asarray(metrics["rint_mean"]).mean()))
        coverage.append(float(runner.coverage(state).mean()))
    return runner, state, intrinsic, coverage


def test_the_bonus_falls_where_the_visits_are():
    """Score the most visited state and a never-visited one, from the same finished table."""
    runner, state, _intrinsic, _coverage = short_run("gt_position_velocity_sqrt")
    counts = np.asarray(state.bonus_state["counts"])[0]
    busiest = int(counts.max())

    # the two observations to score: the start cell at rest, which every episode begins from and
    # which is therefore heavily counted, and a far corner of the maze at a velocity no copy
    # reached in twenty iterations
    # before: observations (x, y, vx, vy); after: their bonuses, read from the table the run built
    probe = jnp.asarray([[[-4.5, -3.0, 0.0, 0.0], [4.5, 3.0, 4.9, 4.9]]] * CONFIG["n_copies"],
                        dtype=jnp.float32)
    _state, bonus, _extra = runner.bonus.post_rollout({}, state.bonus_state, probe, None)
    visited, unvisited = float(np.asarray(bonus)[0, 0]), float(np.asarray(bonus)[0, 1])
    print(f"busiest table entry after {ITERATIONS} iterations: {busiest} visits; "
          f"bonus at the start cell {visited:.4f}, at an unreached state {unvisited:.4f}")
    assert busiest > 10, "no state was visited often enough for this comparison to mean anything"
    assert visited < unvisited, "a heavily visited state did not score below an unvisited one"
    assert unvisited == 1.0, "an unvisited state must score the maximum bonus"
    print("ok test_the_bonus_falls_where_the_visits_are")


def test_the_intrinsic_reward_falls_and_coverage_grows():
    """Both presets: the mean bonus goes down over the run, and the copies keep finding cells."""
    for preset in ("gt_position_velocity_sqrt", "gt_position_velocity_linear"):
        _runner, _state, intrinsic, coverage = short_run(preset)
        print(f"{preset:28} intrinsic reward {intrinsic[0]:.4f} -> {intrinsic[-1]:.4f}, "
              f"maze coverage {coverage[0]:.3f} -> {coverage[-1]:.3f}")
        assert intrinsic[-1] < intrinsic[0], "the bonus did not fall as the table filled"
        assert coverage[-1] > coverage[0], "the copies stopped reaching new cells"
    print("ok test_the_intrinsic_reward_falls_and_coverage_grows")


if __name__ == "__main__":
    test_the_bonus_falls_where_the_visits_are()
    test_the_intrinsic_reward_falls_and_coverage_grows()
