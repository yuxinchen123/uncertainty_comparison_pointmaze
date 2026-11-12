"""Check the per-copy episode log (2026-11-12): its reductions against the existing episode
tracker, its first-arrival step, and that turning it on changes nothing in training.

Run on a compute node: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> -m pytest <file>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.envs.pointmaze.pm_common import EnvConfig  # noqa: E402
from exploration_platform.training import diagnostics as diag  # noqa: E402
from exploration_platform.training import episode_log  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

SMALL = dict(n_copies=4, n_envs=2, num_steps=16, prime_iterations=2)
# episodes of at most 6 steps, so every 16-step rollout ends several of them
SHORT_EPISODES = EnvConfig(continuing_task=False, max_episode_steps=6)


def state_arrays(state) -> dict:
    """Every array of a training state except the episode log itself, by its path in the tree."""
    kept = state._replace(agent_state={name: value for name, value in state.agent_state.items()
                                       if name != "episode_log"})
    flat, _ = jax.tree_util.tree_flatten_with_path(kept)
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}


def run(cfg, iterations=3):
    """Train a few iterations on the short-episode maze; return the final state."""
    runner = Runner(cfg, env_cfg=SHORT_EPISODES)
    state = runner.prime(runner.init_state(0))
    for k in range(1, iterations + 1):
        state, _ = runner.iterate(state, runner.lr_argument(k, iterations))
    return state


def test_reductions_equal_the_tracker_on_random_buffers():
    """The log's masked sums against the tracker's step-by-step loop, carried over two rollouts.

    The tracker counts an episode as having reached the goal when any of its steps was inside
    it, and records the first such step. With arrivals that end the episode, as in the study,
    that is the log's arrival count and its steps to the goal.
    """
    rng = np.random.default_rng(0)
    T, C, N = 40, 3, 2
    tracker = diag.tracker_init(C, N)
    running = jnp.zeros((C, N), jnp.float32)
    length = np.zeros((C, N), np.int32)
    for _ in range(2):
        term = rng.random((T, C, N)) < 0.05
        done = term | (rng.random((T, C, N)) < 0.1)
        reward = (term * 1.0 + rng.normal(size=(T, C, N)) * 0.1).astype(np.float32)
        tracker, expected = diag.track_episodes(tracker, jnp.asarray(term), jnp.asarray(reward),
                                                jnp.asarray(done))

        # what the rollout scan stores per step: the return so far and the step's number
        return_buf, length_buf = [], []
        for t in range(T):
            running, so_far = episode_log.extend_running_return(running, jnp.asarray(reward[t]),
                                                                jnp.asarray(done[t]))
            length = length + 1
            return_buf.append(np.asarray(so_far))
            length_buf.append(length.copy())
            length = np.where(done[t], 0, length)
        sums = episode_log.rollout_sums(jnp.asarray(term, jnp.float32),
                                        jnp.asarray(done, jnp.float32),
                                        jnp.asarray(np.stack(return_buf)),
                                        jnp.asarray(np.stack(length_buf)))
        for name, tracked in (("episodes", "episodes"), ("arrivals", "reached"),
                              ("length_sum", "length"), ("steps_to_goal_sum", "first_hit")):
            np.testing.assert_array_equal(np.asarray(sums[name]), np.asarray(expected[tracked]),
                                          err_msg=name)
        np.testing.assert_allclose(np.asarray(sums["return_sum"]), np.asarray(expected["ret"]),
                                   rtol=1e-5, atol=1e-5)
        assert sums["episodes"].dtype == jnp.int64 and sums["return_sum"].dtype == jnp.float64
    print("PASS the log's reductions equal the episode tracker over two carried rollouts")


def test_first_arrival_step_and_window_helpers():
    """The first-arrival step of a hand-made rollout, kept once set; reading and clearing."""
    # T = 4, N = 2: copy 0 arrives at rollout index 1, copy 1 never, copy 2 at index 0 and 3
    term = np.zeros((4, 3, 2), np.float32)
    term[1, 0, 1] = term[0, 2, 0] = term[3, 2, 1] = 1
    arrived, step = episode_log.first_arrival_step(jnp.asarray(term), jnp.asarray(3, jnp.int32), 2)
    np.testing.assert_array_equal(np.asarray(arrived), [True, False, True])
    # iteration 3 starts after 2 x 4 x 2 = 16 steps; index 1 ends at 16 + 2 x 2 = 20
    assert int(step[0]) == 20 and int(step[2]) == 18 and step.dtype == jnp.int64

    log = episode_log.log_init(3, 2)
    assert (np.asarray(log["first_goal_step"]) == -1).all()
    zeros = jnp.zeros((4, 3, 2), jnp.float32)
    lengths = jnp.ones((4, 3, 2), jnp.int32)
    log = episode_log.add_rollout(log, jnp.zeros((3, 2), jnp.float32), jnp.asarray(term),
                                  jnp.asarray(term), zeros, lengths, jnp.asarray(3, jnp.int32), 2)
    np.testing.assert_array_equal(np.asarray(log["first_goal_step"]), [20, -1, 18])
    # a later arrival does not move a step that is already set, and sets one that is not
    later = np.zeros((4, 3, 2), np.float32)
    later[0, :, 0] = 1
    log = episode_log.add_rollout(log, jnp.zeros((3, 2), jnp.float32), jnp.asarray(later),
                                  jnp.asarray(later), zeros, lengths, jnp.asarray(4, jnp.int32), 2)
    np.testing.assert_array_equal(np.asarray(log["first_goal_step"]), [20, 26, 18])
    np.testing.assert_array_equal(np.asarray(log["arrivals"]), [2, 1, 3])

    # the driver reads the window and clears its sums; first arrivals stay
    window = episode_log.read_window(log)
    assert set(window) == {*episode_log.SUM_FIELDS, "first_goal_step"}
    cleared = episode_log.clear_window(log)
    assert all((np.asarray(cleared[name]) == 0).all() for name in episode_log.SUM_FIELDS)
    np.testing.assert_array_equal(np.asarray(cleared["first_goal_step"]), [20, 26, 18])
    print("PASS first-arrival step, kept once set; window read and cleared")


def test_log_does_not_change_training():
    """Episode log on: the same training state as off, bit for bit, in both update styles."""
    for style in ("full_batch", "epoch_minibatch"):
        off = state_arrays(run(PPOConfig(update_style=style, **SMALL)))
        logged = run(PPOConfig(update_style=style, episode_log=True, **SMALL))
        on = state_arrays(logged)
        assert off.keys() == on.keys()
        for name in off:
            assert off[name].tobytes() == on[name].tobytes(), f"{style}: {name} differs"
        episodes = np.asarray(logged.agent_state["episode_log"]["episodes"])
        assert (episodes > 0).all(), "no episode ended, so the log was not exercised"
        print(f"PASS {style}: the episode log leaves training bit-identical "
              f"({int(episodes.sum())} episodes logged)")
    # without the key the agent state holds exactly the two original entries
    assert set(Runner(PPOConfig(**SMALL)).init_state(0).agent_state) == {"int_filter", "int_rms"}


def test_log_equals_the_tracker_in_a_real_run():
    """In one program with both on, the log's sums equal the diagnostics tracker's, per copy."""
    cfg = PPOConfig(update_style="full_batch", episode_log=True, diagnostics=True, **SMALL)
    runner = Runner(cfg, env_cfg=SHORT_EPISODES)
    state = runner.prime(runner.init_state(0))
    tracked = {name: np.zeros(cfg.n_copies) for name in ("episodes", "reached", "ret", "length")}
    for k in range(1, 5):
        state, metrics = runner.iterate(state, runner.lr_argument(k, 4))
        for name in tracked:
            tracked[name] += np.asarray(metrics["diagnostics"][f"episode_{name}"])
    log = episode_log.read_window(state.agent_state["episode_log"])
    np.testing.assert_array_equal(log["episodes"], tracked["episodes"])
    np.testing.assert_array_equal(log["arrivals"], tracked["reached"])
    np.testing.assert_array_equal(log["length_sum"], tracked["length"])
    np.testing.assert_allclose(log["return_sum"], tracked["ret"], rtol=1e-6, atol=1e-6)
    # 4 iterations x 16 steps x 2 environments with episodes of at most 6 steps
    assert (log["episodes"] >= 4 * 16 * 2 // 6 - 2).all(), log["episodes"]
    print(f"PASS the log equals the tracker in a run: episodes {log['episodes'].tolist()}")


def test_log_needs_the_hoisted_rollout():
    """The log is built into the hoisted rollout only; the older form refuses it."""
    try:
        Runner(PPOConfig(episode_log=True, hoist_rollout=False, **SMALL))
    except ValueError as error:
        assert "hoisted rollout" in str(error)
        print(f"PASS refused: {error}")
        return
    raise AssertionError("episode_log with hoist_rollout=False was accepted")


if __name__ == "__main__":
    test_reductions_equal_the_tracker_on_random_buffers()
    test_first_arrival_step_and_window_helpers()
    test_log_does_not_change_training()
    test_log_equals_the_tracker_in_a_real_run()
    test_log_needs_the_hoisted_rollout()
