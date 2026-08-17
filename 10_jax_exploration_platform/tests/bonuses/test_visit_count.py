"""The fused visit-count bonus must compute what the plain-python reference computes.

Supplied randomness: the trajectories are drawn here and handed to both forms, so the two are
compared on identical inputs and nothing depends on either of them drawing the same numbers.

The count table is compared for exact equality — it is the whole semantics, and it is integer
arithmetic, so anything but exact equality is a real disagreement. The bonus is compared to 1e-6,
because `n**decay` is a transcendental function whose last bit belongs to whichever library
evaluated it.

Run: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <jax python> test_visit_count.py
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.bonuses.registry import make_bonus  # noqa: E402
from exploration_platform.bonuses.visit_count import reference  # noqa: E402
from exploration_platform.envs.pointmaze.pm_common import MAPS, EnvConfig  # noqa: E402

ENV = EnvConfig()
WALL = np.asarray(MAPS[ENV.map_name])
ROWS, COLS = WALL.shape
DECAYS = {"gt_position_velocity_sqrt": -0.5, "gt_position_velocity_linear": -1.0}


def build(preset: str, n_copies: int):
    """The family, bound to a small run."""
    cfg = PPOConfig(n_copies=n_copies, n_envs=2, num_steps=4)
    return make_bonus(preset)(cfg, ENV, n_copies, cfg.base_seed, list(range(n_copies)))


def random_rollout(rng, n_copies: int, n_rows: int):
    """Observations spread over the whole world, with velocities that reach past the clip.

    before: nothing; after: [C, M, 4] float32 with x in [-7, 7], y in [-5.5, 5.5] (so some rows
    fall outside the maze and have to be clipped to its edge cells) and each velocity component
    in [-7, 7] (so some fall outside the +-5 the environment itself clips to)
    """
    x = rng.uniform(-7.0, 7.0, (n_copies, n_rows))
    y = rng.uniform(-5.5, 5.5, (n_copies, n_rows))
    v = rng.uniform(-7.0, 7.0, (n_copies, n_rows, 2))
    return np.stack([x, y, v[..., 0], v[..., 1]], axis=-1).astype(np.float32)


def compare(preset: str, next_obs, counts_before):
    """Run both forms on the same input and return their worst disagreements."""
    bonus = build(preset, next_obs.shape[0])
    state = {"counts": jnp.asarray(counts_before, dtype=jnp.int32)}
    state_after, reward, extra = bonus.post_rollout({}, state, jnp.asarray(next_obs), None)
    ref_counts, ref_bonus = reference.post_rollout(counts_before, next_obs, WALL,
                                                   DECAYS[preset])
    assert extra == {}, "this family needs no extra fields in the update batch"
    counts_differ = int(np.abs(np.asarray(state_after["counts"]) - ref_counts).max())
    bonus_differ = float(np.abs(np.asarray(reward) - ref_bonus).max())
    return counts_differ, bonus_differ, np.asarray(state_after["counts"]), np.asarray(reward)


def test_first_rollout_matches_the_reference():
    """One rollout into an empty table: counts exactly equal, bonuses equal to 1e-6."""
    rng = np.random.default_rng(11)
    for preset in DECAYS:
        obs = random_rollout(rng, 4, 40)
        empty = np.zeros((4, ROWS * COLS * 100), dtype=np.int32)
        counts_differ, bonus_differ, counts, bonus = compare(preset, obs, empty)
        print(f"{preset:28} first rollout: counts differ by {counts_differ}, "
              f"bonus by {bonus_differ:.2e}; {int(counts.sum())} visits counted, "
              f"bonus in [{bonus.min():.3f}, {bonus.max():.3f}]")
        assert counts_differ == 0, "the two forms disagree on the count table"
        assert bonus_differ <= 1e-6, "the two forms disagree on the bonus"
    print("ok test_first_rollout_matches_the_reference")


def test_repeated_rollouts_match_the_reference():
    """Three rollouts in a row, so entries accumulate and the bonus actually falls."""
    rng = np.random.default_rng(12)
    for preset in DECAYS:
        counts = np.zeros((3, ROWS * COLS * 100), dtype=np.int32)
        means = []
        for _ in range(3):
            # a narrow patch of the maze, so the same entries are hit again and again
            obs = np.stack([rng.uniform(-5.0, -4.0, (3, 30)),
                            rng.uniform(-3.5, -2.5, (3, 30)),
                            rng.uniform(-1.0, 1.0, (3, 30)),
                            rng.uniform(-1.0, 1.0, (3, 30))], axis=-1).astype(np.float32)
            counts_differ, bonus_differ, counts, bonus = compare(preset, obs, counts)
            assert counts_differ == 0, "the two forms disagree on the count table"
            assert bonus_differ <= 1e-6, "the two forms disagree on the bonus"
            means.append(float(bonus.mean()))
        print(f"{preset:28} three rollouts on one patch: mean bonus {means}")
        assert means[-1] < means[0], "revisiting a patch did not lower its bonus"
    print("ok test_repeated_rollouts_match_the_reference")


def test_wall_and_clip_cases():
    """The three cases a random draw is unlikely to place exactly: wall, edge, past the clip."""
    # a wall cell of the large maze: row 0 is entirely wall, so any y above 3.5 with any x
    # (2.5, 4.0) sits over one; and the world's own corner is outside the maze entirely
    cases = np.asarray([
        [0.0, 4.0, 0.0, 0.0],        # inside the top wall row -> bonus 1.0, never counted
        [-4.5, -3.0, 0.0, 0.0],      # the start cell (row 7, column 1), open
        [-99.0, 99.0, 0.0, 0.0],     # far outside the world -> clipped onto a corner wall cell
        [0.5, 0.5, 40.0, -40.0],     # velocity far past the +-5 the environment clips to
        [0.5, 0.5, 5.0, -5.0],       # exactly at the clip, which must land in the same end bins
    ], dtype=np.float32)[None]       # one copy, five rows
    for preset in DECAYS:
        counts_differ, bonus_differ, counts, bonus = compare(
            preset, cases, np.zeros((1, ROWS * COLS * 100), dtype=np.int32))
        print(f"{preset:28} edge cases: bonus {np.round(bonus[0], 4).tolist()}, "
              f"{int(counts.sum())} visits counted of 5 rows")
        assert counts_differ == 0 and bonus_differ <= 1e-6
        assert bonus[0, 0] == 1.0, "a wall cell must score the maximum bonus"
        assert bonus[0, 2] == 1.0, "a position outside the world lands on a wall cell"
        # three of the five rows are on open cells; the last two share one entry, because a
        # velocity past the clip lands in the same end bin as one exactly at it
        assert int(counts.sum()) == 3, "the open-cell rows were not all counted"
        assert bonus[0, 3] == bonus[0, 4], "clipping did not put the two velocities in one bin"
        assert bonus[0, 3] < 1.0, "a state reached twice in one rollout must score below 1"
    print("ok test_wall_and_clip_cases")


def test_the_two_presets_differ():
    """The sqrt and linear presets must actually score differently once a count exceeds one."""
    obs = np.tile(np.asarray([[-4.5, -3.0, 0.0, 0.0]], dtype=np.float32), (1, 4, 1))
    empty = np.zeros((1, ROWS * COLS * 100), dtype=np.int32)
    scores = {p: compare(p, obs, empty)[3][0, 0] for p in DECAYS}
    print(f"one state reached four times in a rollout: {dict(scores)}")
    assert abs(scores["gt_position_velocity_sqrt"] - 0.5) <= 1e-6, "1/sqrt(4) is 0.5"
    assert abs(scores["gt_position_velocity_linear"] - 0.25) <= 1e-6, "1/4 is 0.25"
    print("ok test_the_two_presets_differ")


if __name__ == "__main__":
    test_first_rollout_matches_the_reference()
    test_repeated_rollouts_match_the_reference()
    test_wall_and_clip_cases()
    test_the_two_presets_differ()
