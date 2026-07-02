"""Unit tests for Train-run-3.1.1's sweep-id + seed-outermost run-id convention over the 91-config inner grid
(.claude/rules/run-id-and-logging.md). Run: `pytest test_run_id_convention.py`.
"""
import os
import sys
import json
import glob
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(pending):
    """Load all queue configs sorted by run_id."""
    return sorted((json.load(open(f)) for f in glob.glob(os.path.join(pending, "*.json"))), key=lambda c: c["run_id"])


def test_build_queue_91_configs_seed_outermost(tmp_path):
    """build_queue --sweep_id S emits 9100 configs (91/seed x 100 seeds) under queue/S/pending/, seed-outermost
    ids, each stamped with sweep_id=S, its per-config params, and the fixed args (eval OFF, 1e6 steps)."""
    # copy build_queue into a throwaway run dir so the real queue is untouched; it derives RUN_DIR from its path
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir()
    shutil.copy(os.path.join(HERE, "build_queue.py"), str(slurm_dir / "build_queue.py"))
    subprocess.check_call([sys.executable, str(slurm_dir / "build_queue.py"), "--sweep_id", "test-sweep"])
    # all FOUR queue subdirs must exist (workers rename pending->running->done/failed)
    for sub in ("pending", "running", "done", "failed"):
        assert (tmp_path / "queue" / "test-sweep" / sub).is_dir(), f"missing queue subdir {sub}"
    pending = tmp_path / "queue" / "test-sweep" / "pending"
    cfgs = _load(str(pending))
    # 9100 = 91 configs/seed x 100 seeds; gap-free ids 0..9099; every config tagged with the sweep id
    assert len(cfgs) == 9100 and all(c["run_total"] == 9100 for c in cfgs)
    assert [c["run_id"] for c in cfgs] == list(range(9100))
    assert all(c["sweep_id"] == "test-sweep" for c in cfgs)
    # fixed args: standalone eval OFF, 1e6 steps, distance OFF, top_right goal
    assert all(c["fixed"]["eval_standalone"] == "False" for c in cfgs)
    assert all(c["fixed"]["total_timesteps"] == 1000000 for c in cfgs)
    assert all(c["fixed"]["log_distance"] == "False" and c["fixed"]["goal_position"] == "top_right" for c in cfgs)

    # --- the 91-config inner order for seed 0 (ids 0..90) ---
    # id 0: A1 batch elliptical, state_action input, ridge 1, beta 0.001, sample timing, unit norm
    c0 = cfgs[0]
    assert (c0["run_id"], c0["a_seed"], c0["algorithm"], c0["beta"]) == (0, 0, "rnd_elliptical", "0.001")
    assert c0["params"] == {
        "elliptical_update_timing": "sample", "elliptical_feature_input": "state_action",
        "elliptical_regularization": "1", "elliptical_feature_normalization": "unit",
    }
    # the 84 elliptical configs come first, then the 7 RND configs; id 84 is the first RND (beta 0.001)
    seed0 = cfgs[:91]
    assert sum(c["algorithm"].startswith("rnd_elliptical") for c in seed0) == 84
    assert sum(c["algorithm"] == "rnd_next_state" for c in seed0) == 7
    assert (cfgs[84]["run_id"], cfgs[84]["algorithm"], cfgs[84]["beta"], cfgs[84]["params"]) == (84, "rnd_next_state", "0.001", {})
    assert (cfgs[90]["algorithm"], cfgs[90]["beta"]) == ("rnd_next_state", "1000")

    # --- seed outermost: seed 1 begins at id 91 ---
    assert cfgs[91]["run_id"] == 91 and cfgs[91]["a_seed"] == 1 and cfgs[91]["algorithm"] == "rnd_elliptical"
    # last id: seed 99, the final RND config (beta 1000)
    assert cfgs[-1]["run_id"] == 9099 and cfgs[-1]["a_seed"] == 99
    assert (cfgs[-1]["algorithm"], cfgs[-1]["beta"]) == ("rnd_next_state", "1000")

    # the sweep is catalogued in the manifest
    manifest = (tmp_path / "data" / "SWEEPS.md").read_text()
    assert "test-sweep" in manifest and "| 9100 |" in manifest


def test_both_global_variants_distinguished_by_timing(tmp_path):
    """A2 and A3 share algorithm rnd_elliptical_global; they are told apart only by elliptical_update_timing,
    so both timings must be present (28 configs each) for that algorithm."""
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir()
    shutil.copy(os.path.join(HERE, "build_queue.py"), str(slurm_dir / "build_queue.py"))
    subprocess.check_call([sys.executable, str(slurm_dir / "build_queue.py"), "--sweep_id", "s"])
    seed0 = _load(str(tmp_path / "queue" / "s" / "pending"))[:91]
    g = [c for c in seed0 if c["algorithm"] == "rnd_elliptical_global"]
    assert sum(c["params"]["elliptical_update_timing"] == "sample" for c in g) == 28
    assert sum(c["params"]["elliptical_update_timing"] == "add" for c in g) == 28
    # every elliptical config is unit-norm and covers both inputs and both ridges
    assert all(c["params"]["elliptical_feature_normalization"] == "unit" for c in seed0 if c["params"])
    assert {c["params"]["elliptical_feature_input"] for c in seed0 if c["params"]} == {"state_action", "next_state"}
    assert {c["params"]["elliptical_regularization"] for c in seed0 if c["params"]} == {"1", "0.01"}


def test_per_run_json_name_id_based():
    """train.py names the JSON 'NNNN_of_9100.json' (id zero-padded to TOTAL's width = 4)."""
    for run_id, expected in [(0, "0000_of_9100.json"), (84, "0084_of_9100.json"), (9099, "9099_of_9100.json")]:
        assert f"{run_id:0{len(str(9100))}d}_of_9100.json" == expected
