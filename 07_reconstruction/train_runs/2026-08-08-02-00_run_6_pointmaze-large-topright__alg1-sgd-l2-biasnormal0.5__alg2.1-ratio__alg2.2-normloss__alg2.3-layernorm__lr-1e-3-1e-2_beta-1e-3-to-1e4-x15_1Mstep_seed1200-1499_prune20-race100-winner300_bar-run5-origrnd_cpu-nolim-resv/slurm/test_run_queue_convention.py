#!/usr/bin/env python
"""Pin train run 6's queue convention: the configuration list, the seed-outermost id order, the
seed mapping, the config keys, the ARM definitions' byte-equality with train run 8.1.2, and the
frozen bar's provenance.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python -m pytest slurm/test_run_queue_convention.py -q
"""
import glob
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_queue as bq  # noqa: E402
import score_rules        # noqa: E402


def load_run812_build_queue():
    """Import train run 8.1.2's build_queue.py under a private name (the arm source of truth)."""
    hits = glob.glob(os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                  "*run_8_1_2_antmaze-umaze-medium*", "slurm", "build_queue.py"))
    assert len(hits) == 1, hits
    spec = importlib.util.spec_from_file_location("bq812", hits[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_config_count_is_four_arms_times_two_rates_times_fifteen_weights():
    """120 configurations, 300 seeds, 36,000 runs."""
    assert bq.ARMS == ["alg1", "alg2.1", "alg2.2", "alg2.3"]
    assert len(bq.LRS) == 2 and len(bq.BETAS) == 15
    assert len(bq.CONFIGS) == 120
    assert len(bq.SEED_INDICES) == 300
    assert bq.RUN_TOTAL == 120 * 300 == 36000


def test_arm_definitions_are_verbatim_from_run_812():
    """The four arms are the train-run-8.1.2 arms: ALG1_PARAMS, ARM_EXTRAS, LRS and BETAS all
    equal that run's build_queue.py values, so this run tests the arms it claims to test."""
    src = load_run812_build_queue()
    assert bq.ALG1_PARAMS == src.ALG1_PARAMS
    assert bq.ARM_EXTRAS == src.ARM_EXTRAS
    assert bq.ARMS == src.ARMS
    assert bq.LRS == src.LRS
    assert bq.BETAS == src.BETAS


def test_environment_is_train_run_5s_task_and_only_that():
    """One environment: train run 5's PointMaze task, with a registered score rule."""
    assert bq.ENV_SETUP == "initial_single_large_pointmaze_max_400"
    assert {c["env_setup"] for c in bq.CONFIGS} == {bq.ENV_SETUP}
    assert score_rules.rule_for(bq.ENV_SETUP) == "final_reward"


def test_seed_mapping_is_fresh():
    """Seeds 1200-1499: disjoint from run 5's 600-899, the addendum's 900-1199, everything <=599."""
    assert bq.a_seed_of(0) == 1200
    assert bq.a_seed_of(299) == 1499


def test_seed_index_is_outermost_and_each_index_owns_a_contiguous_id_block():
    """Seed index i owns run ids [120*i .. 120*i+119] in the fixed CONFIGS order."""
    run_id = 0
    for seed_index in bq.SEED_INDICES:
        for cfg_index, _ in enumerate(bq.CONFIGS):
            assert run_id == seed_index * len(bq.CONFIGS) + cfg_index
            run_id += 1
    assert run_id == bq.RUN_TOTAL


def test_config_key_shape_and_key_from_record_roundtrip():
    """config_key is env|arm|lr<lr>|b<beta>; key_from_record rebuilds it from record flag fields."""
    cfg = bq.CONFIGS[0]
    assert bq.config_key(cfg) == "initial_single_large_pointmaze_max_400|alg1|lr0.001|b0.001"
    # one record-shaped dict per arm, carrying exactly the flags that identify it
    for cfg in [bq.CONFIGS[0], bq.CONFIGS[30], bq.CONFIGS[60], bq.CONFIGS[90]]:
        rec = {"env_setup": cfg["env_setup"], "rnd_lr": float(cfg["lr"]),
               "beta": float(cfg["beta"]), **{k: v for k, v in cfg["params"].items()
                                              if k in ("rnd_readout_norm_init",
                                                       "rnd_predictor_loss", "rnd_layer_norm")}}
        assert bq.key_from_record(rec) == bq.config_key(cfg), cfg["arm"]


def test_every_configuration_key_is_unique():
    """No two configurations share a key, or the controller would pool their seeds."""
    keys = [bq.config_key(c) for c in bq.CONFIGS]
    assert len(set(keys)) == len(keys)


def test_the_frozen_bar_is_run5s_final_number():
    """FROZEN_BARS.json holds the run-5 original RND best configuration: 38.6412 at bonus weight
    1000 over its full 300 seeds."""
    doc = json.load(open(os.path.join(HERE, "FROZEN_BARS.json")))
    bar = doc["bars"][bq.ENV_SETUP]
    assert bar["beta"] == "1000"
    assert bar["n"] == 300
    assert abs(bar["mean"] - 38.6412) < 5e-4
    assert bar["score_rule"] == "final_reward"


def test_fixed_args_carry_one_million_steps_and_no_environment_knob():
    """The environment comes only from --env_setup; no env-side knob may sit in the fixed args."""
    assert bq.FIXED_COMMON["total_timesteps"] == 1000000
    for banned in ("discount_factor", "max_episode_steps", "reward_shift", "goal_cell"):
        assert banned not in bq.FIXED_COMMON


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
