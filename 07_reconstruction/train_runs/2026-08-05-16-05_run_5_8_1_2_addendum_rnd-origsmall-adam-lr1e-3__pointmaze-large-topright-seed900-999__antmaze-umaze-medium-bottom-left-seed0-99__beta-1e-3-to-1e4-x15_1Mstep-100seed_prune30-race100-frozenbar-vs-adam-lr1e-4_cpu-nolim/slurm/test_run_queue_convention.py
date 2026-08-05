#!/usr/bin/env python
"""Pin the queue convention of this run: the configuration list, the seed-outermost id order, the
per-environment seed mapping, the config keys and the one changed knob.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python -m pytest slurm/test_run_queue_convention.py -q
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_queue as bq  # noqa: E402
import score_rules        # noqa: E402


def test_config_count_is_three_environments_times_fifteen_bonus_weights():
    """45 configurations: 3 environments x the 15 bonus weights of train run 1.2."""
    assert len(bq.ENV_SETUPS) == 3
    assert len(bq.BETAS) == 15
    assert len(bq.CONFIGS) == 45
    assert bq.RUN_TOTAL == 45 * 100 == 4500


def test_bonus_weight_grid_is_train_run_12s_grid():
    """The swept range is exactly the train-run-1.2 grid, ascending from 1e-3 to 1e4."""
    assert bq.BETAS == ["0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1", "3", "10", "30",
                        "100", "300", "1000", "3000", "10000"]
    assert [float(b) for b in bq.BETAS] == sorted(float(b) for b in bq.BETAS)


def test_the_only_changed_knob_is_the_adam_learning_rate():
    """Every configuration is the train-run-5 original-small stack with rnd_lr 1e-3 and nothing
    else moved. The reference dict below is train run 5's ORIGSMALL_PARAMS verbatim."""
    run5_origsmall = {
        "rnd_optimizer": "adam", "rnd_bonus_readout": "mse_mean", "rnd_lr": "0.0001",
        "rnd_update_proportion": "1.0", "rnd_activation": "leaky_relu",
        "rnd_predictor_extra_layers": "1", "rnd_obs_warmup_mode": "env_steps",
        "rnd_obs_warmup_steps": "6400", "rnd_reward_norm": "True", "rnd_reward_norm_gamma": "0.99",
        "rnd_bias_init": "zero", "rnd_weight_init": "orthogonal",
    }
    ours = bq.ORIGSMALL_ADAM_LR1E3_PARAMS
    assert set(ours) == set(run5_origsmall)
    differing = {k for k in ours if ours[k] != run5_origsmall[k]}
    assert differing == {"rnd_lr"}
    assert ours["rnd_lr"] == "0.001"
    for cfg in bq.CONFIGS:
        assert cfg["params"] == ours
        assert cfg["algorithm"] == "rnd_next_state"


def test_seed_index_is_outermost_and_each_index_owns_a_contiguous_id_block():
    """Seed index i owns run ids [45*i .. 45*i+44] in the fixed CONFIGS order."""
    ids_by_index = {}
    run_id = 0
    for seed_index in bq.SEED_INDICES:
        for _ in bq.CONFIGS:
            ids_by_index.setdefault(seed_index, []).append(run_id)
            run_id += 1
    assert run_id == bq.RUN_TOTAL
    assert ids_by_index[0] == list(range(0, 45))
    assert ids_by_index[1] == list(range(45, 90))
    assert ids_by_index[99] == list(range(4455, 4500))


def test_seed_mapping_is_fresh_on_pointmaze_and_zero_based_on_antmaze():
    """PointMaze runs seeds 900-999 (disjoint from train run 5's 600-899); AntMaze runs 0-99."""
    assert bq.a_seed_of("initial_single_large_pointmaze_max_400", 0) == 900
    assert bq.a_seed_of("initial_single_large_pointmaze_max_400", 99) == 999
    assert bq.a_seed_of("AntMaze_UMaze-v5_start_bottom_left", 0) == 0
    assert bq.a_seed_of("AntMaze_Medium-v5_start_bottom_left", 99) == 99


def test_config_key_shape_matches_train_run_12():
    """config_key is env_setup|arm|lr<lr>|b<beta>, and key_from_record rebuilds it from a record."""
    cfg = bq.CONFIGS[0]
    key = bq.config_key(cfg)
    assert key == "initial_single_large_pointmaze_max_400|origsmall|lr0.001|b0.001"
    assert len(key.split("|")) == 4
    record = {"env_setup": cfg["env_setup"], "rnd_lr": 0.001, "beta": 0.001}
    assert bq.key_from_record(record) == key


def test_every_configuration_key_is_unique():
    """No two configurations share a key, or the controller would pool their seeds."""
    keys = [bq.config_key(c) for c in bq.CONFIGS]
    assert len(set(keys)) == len(keys)


def test_every_environment_has_a_registered_score_rule():
    """A configuration whose environment has no score rule would crash the controller mid-sweep."""
    for env in bq.ENV_SETUPS:
        assert score_rules.rule_for(env) in ("final_reward", "whole_run_mean")


def test_marker_label_is_filesystem_safe_and_identifies_the_configuration():
    """The queue filename tag carries the environment, arm, learning rate and bonus weight."""
    for cfg in bq.CONFIGS:
        tag = bq.label(cfg)
        assert "/" not in tag and " " not in tag
        assert f"b{cfg['beta']}" in tag and "lr0.001" in tag
    tags = {bq.label(c) for c in bq.CONFIGS}
    assert len(tags) == len(bq.CONFIGS)


def test_fixed_args_carry_one_million_steps_and_no_environment_knob():
    """The environment comes only from --env_setup, so no env-side knob may sit in the fixed args."""
    assert bq.FIXED_COMMON["total_timesteps"] == 1000000
    for forbidden in ("env_name", "discount_factor", "env_max_episode", "goal_position",
                      "reward_shift", "position_noise_range", "continuing_task"):
        assert forbidden not in bq.FIXED_COMMON


def test_queue_build_writes_one_marker_per_run(tmp_path, monkeypatch):
    """End to end on a temporary run dir: 4,500 markers, gap-free ids, correct seeds inside."""
    monkeypatch.setattr(bq, "RUN_DIR", str(tmp_path))
    monkeypatch.setattr(bq.sys, "argv", ["build_queue.py", "--sweep_id", "test-sweep"])
    monkeypatch.setattr(bq.getpass, "getuser", lambda: "sl5nw")
    bq.main()
    pending = tmp_path / "queue" / "test-sweep" / "pending"
    names = sorted(os.listdir(pending))
    assert len(names) == bq.RUN_TOTAL
    ids = sorted(int(n.split("_")[0]) for n in names)
    assert ids == list(range(bq.RUN_TOTAL))
    # marker 0 is seed index 0 of the first configuration -> PointMaze, seed 900, weight 0.001
    first = json.loads((pending / names[0]).read_text())
    assert first["env_setup"] == "initial_single_large_pointmaze_max_400"
    assert first["a_seed"] == 900 and first["seed_index"] == 0 and first["beta"] == "0.001"
    # marker 15 is the first AntMaze UMaze configuration of the same seed index -> seed 0
    at_15 = json.loads((pending / names[15]).read_text())
    assert at_15["env_setup"] == "AntMaze_UMaze-v5_start_bottom_left" and at_15["a_seed"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
