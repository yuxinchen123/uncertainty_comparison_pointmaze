"""Tests of the PointMaze fresh-seed comparison on small hand-made records (numpy: run through Slurm)."""
import json

import numpy as np

import pointmaze_fresh_seeds as fs


def write_unit(path, returns, episodes, first_goal, seeds):
    """Write one unit file: a start record and 256 windows with the given per-policy values."""
    start = {"record": "unit_start", "rollout_steps": 128, "envs_per_copy": 4, "window_iterations": 2,
             "copy_seed_index": seeds}
    with open(path, "w") as f:
        f.write(json.dumps(start) + "\n")
        for t in range(256):
            f.write(json.dumps({"return_sum_per_copy": returns(t), "episodes_per_copy": episodes(t),
                                "first_goal_step_per_copy": first_goal}) + "\n")


def test_per_seed_whole_curve_final_level_and_capped_first_goal(tmp_path):
    """W carries the last value through empty windows, L is the last tenth, M is capped at the budget."""
    path = tmp_path / "u.jsonl"
    # policy 0 returns 1 per episode from window 128 on (none before); policy 1 never finishes an episode
    write_unit(path, lambda t: [1.0 if t >= 128 else 0.0, 0.0], lambda t: [1 if t >= 128 else 0, 0],
               [5000, -1], [30000, 30001])
    out = fs.per_seed([path])
    assert abs(out[30000][0] - 0.5) < 1e-12 and out[30000][1] == 1.0 and out[30000][2] == 5.0
    budget = 128 * 4 * 2 * 256
    assert out[30001] == (0.0, 0.0, budget / 1e3)


def test_per_seed_refuses_a_short_unit(tmp_path):
    """A unit with fewer than 256 windows is an error, not a smaller reading."""
    path = tmp_path / "u.jsonl"
    write_unit(path, lambda t: [0.0], lambda t: [0], [-1], [30000])
    lines = open(path).read().splitlines()[:100]
    open(path, "w").write("\n".join(lines) + "\n")
    try:
        fs.per_seed([path])
    except ValueError:
        return
    raise AssertionError("a short unit was accepted")


def test_interval_and_verdicts():
    """A constant positive difference gives a point interval above zero; the verdict rules combine as stated."""
    method = {s: (0.6, 0.9, 10.0) for s in range(100)}
    rnd = {s: (0.5, 0.9, 12.0) for s in range(100)}
    w = fs.interval(method, rnd, 0, "test")
    assert np.isclose(w[0], 0.1) and np.isclose(w[1], 0.1) and np.isclose(w[2], 0.1) and w[3] == 100
    assert fs.verdict_w(w, fs.interval(method, rnd, 1, "test")) == "ahead"
    assert fs.verdict_m(fs.interval(method, rnd, 2, "test")) == "ahead"
    assert fs.combined(["ahead", "level"]) == "level" and fs.combined(["ahead", "behind"]) == "behind"
    assert fs.words("x_cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1") \
        == "learning rate 0.006, entropy 0.01, weight 1/512"
