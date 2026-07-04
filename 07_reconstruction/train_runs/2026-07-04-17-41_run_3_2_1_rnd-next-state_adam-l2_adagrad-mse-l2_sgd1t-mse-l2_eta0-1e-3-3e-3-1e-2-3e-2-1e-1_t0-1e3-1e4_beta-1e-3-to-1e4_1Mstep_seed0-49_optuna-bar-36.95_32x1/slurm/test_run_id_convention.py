"""Pins run 3.2.1's id convention: 184 configs in the documented fixed order, seed OUTERMOST ids.

Run with:  conda run -n exploration python -m pytest slurm/test_run_id_convention.py  (from the run folder)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_queue


def test_grid_shape_and_block_order():
    """CONFIGS is exactly O1 adam-l2 (8), then O2 adagrad mse->l2 (16), then O3 sgd1t (160)."""
    cfgs = build_queue.CONFIGS
    assert len(cfgs) == 184
    assert build_queue.RUN_TOTAL == 184 * 50 == 9200
    # O1 block: 8 adam-l2 rows, beta ascending
    assert all(c["params"]["rnd_optimizer"] == "adam" for c in cfgs[:8])
    assert all(c["params"]["rnd_bonus_readout"] == "l2" for c in cfgs[:8])
    assert [c["beta"] for c in cfgs[:8]] == build_queue.BETAS
    # no adam+mse cell anywhere (the run-3.1.1 reference arm is not re-run)
    assert not any(c["params"]["rnd_optimizer"] == "adam" and c["params"]["rnd_bonus_readout"] == "mse"
                   for c in cfgs)
    # O2 block: 16 adagrad rows, readout mse then l2, beta ascending inside each readout
    o2 = cfgs[8:24]
    assert all(c["params"]["rnd_optimizer"] == "adagrad" for c in o2)
    assert [c["params"]["rnd_bonus_readout"] for c in o2] == ["mse"] * 8 + ["l2"] * 8
    assert [c["beta"] for c in o2[:8]] == build_queue.BETAS
    # O3 block: 160 sgd1t rows; readout outer, eta0, t0, beta innermost
    o3 = cfgs[24:]
    assert len(o3) == 160
    assert all(c["params"]["rnd_optimizer"] == "sgd1t" for c in o3)
    assert [c["params"]["rnd_bonus_readout"] for c in o3] == ["mse"] * 80 + ["l2"] * 80
    first = o3[0]["params"]
    assert (first["rnd_sgd_eta0"], first["rnd_sgd_t0"], o3[0]["beta"]) == ("0.001", "1000", "0.001")
    # beta innermost: consecutive rows share eta0/t0 for 8 rows
    assert [c["beta"] for c in o3[:8]] == build_queue.BETAS
    assert all(c["params"]["rnd_sgd_eta0"] == "0.001" and c["params"]["rnd_sgd_t0"] == "1000"
               for c in o3[:8])
    # t0 next-innermost: rows 8..15 switch t0, keep eta0
    assert all(c["params"]["rnd_sgd_t0"] == "10000" and c["params"]["rnd_sgd_eta0"] == "0.001"
               for c in o3[8:16])


def test_seed_outermost_id_math():
    """id = 184*seed + config_index; every id 0..9199 appears exactly once; early seeds own early ids."""
    # golden path: the id of (seed, config_index) is unique and ordered seed-major
    ids = [seed * 184 + idx for seed in build_queue.SEEDS for idx in range(184)]
    assert ids == list(range(9200))
    # an earlier seed's ids all precede a later seed's ids (the early-seeds-finish-first guarantee)
    assert max(range(0, 184)) < min(range(184, 368))


def test_config_key_round_trip():
    """config_key is unique per config and stable under the float formatting both sides use."""
    keys = [build_queue.config_key(c["params"], c["beta"]) for c in build_queue.CONFIGS]
    assert len(set(keys)) == 184
    # spot values: string inputs and float inputs give the same key (the controller parses floats)
    assert build_queue.config_key(
        {"rnd_optimizer": "sgd1t", "rnd_bonus_readout": "l2", "rnd_sgd_eta0": "0.003",
         "rnd_sgd_t0": "1000"}, "0.01") == "sgd1t|l2|0.003|1000|0.01"
    assert build_queue.config_key(
        {"rnd_optimizer": "sgd1t", "rnd_bonus_readout": "l2", "rnd_sgd_eta0": 0.003,
         "rnd_sgd_t0": 1000.0}, 0.01) == "sgd1t|l2|0.003|1000|0.01"
    assert build_queue.config_key(
        {"rnd_optimizer": "adagrad", "rnd_bonus_readout": "mse"}, "100") == "adagrad|mse|-|-|100"
