"""Unit tests for the revised run-id / seed-outermost sweep convention.

Covers build_queue.py (seed-outermost integer ids, run_total stamping) and train.py `_run_name`
(id-based filename for sweep runs, descriptive name for standalone runs). Run: `pytest test_run_id_convention.py`.
"""
import os
import sys
import json
import glob
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = "/p/rlprojects/RND/07_reconstruction"


def test_build_queue_seed_outermost_ids(tmp_path):
    """build_queue emits 600 configs with seed-outermost ids: seed s -> a contiguous block, algos in order."""
    # run the queue builder into a throwaway RUN_DIR so the real queue is untouched
    env = dict(os.environ, RUN_DIR=str(tmp_path))
    subprocess.check_call([sys.executable, os.path.join(HERE, "build_queue.py")], env=env)
    # load every config, sorted by id. before: 600 JSON files; after: list[dict] ordered by run_id
    cfgs = sorted((json.load(open(f)) for f in glob.glob(str(tmp_path / "queue" / "pending" / "*.json"))),
                  key=lambda c: c["run_id"])
    # 600 configs total, each stamped with run_total=600
    assert len(cfgs) == 600 and all(c["run_total"] == 600 for c in cfgs)
    # seed 0 occupies ids 0,1,2 (the three algorithms in ALGO_BETA order)
    assert [(c["run_id"], c["a_seed"], c["g_algo_beta"].split("|")[0]) for c in cfgs[:3]] == \
        [(0, 0, "gt_position_velocity"), (1, 0, "rnd_elliptical"), (2, 0, "rnd_state")]
    # the next seed starts the next id block -> an earlier seed always has smaller ids
    assert cfgs[3]["run_id"] == 3 and cfgs[3]["a_seed"] == 1
    assert cfgs[-1]["run_id"] == 599 and cfgs[-1]["a_seed"] == 199
    # ids are exactly 0..599 with no gaps or duplicates
    assert [c["run_id"] for c in cfgs] == list(range(600))


def test_run_name_id_based_for_sweep_runs():
    """_run_name returns the zero-padded 'NNN_of_TOTAL' id for sweep runs, descriptive name when run_total=0."""
    sys.path.insert(0, PROJ)
    from train import Config, _run_name
    # sweep run: id zero-padded to run_total's width (600 -> 3 digits)
    assert _run_name(Config(run_id=42, run_total=600)) == "042_of_600"
    assert _run_name(Config(run_id=0, run_total=600)) == "000_of_600"
    # standalone run (no sweep id) keeps the descriptive pipe-delimited name
    assert _run_name(Config(run_total=0)).startswith("algorithm=")
