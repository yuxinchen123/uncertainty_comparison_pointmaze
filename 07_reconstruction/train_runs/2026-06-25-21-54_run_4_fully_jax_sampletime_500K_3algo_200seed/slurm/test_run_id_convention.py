"""Unit tests for Train-run-4's sweep-id + seed-outermost run-id convention
(.claude/rules/run-id-and-logging.md). Run: `pytest test_run_id_convention.py`.
"""
import os
import sys
import json
import glob
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def test_build_queue_sweep_scoped_seed_outermost_ids(tmp_path):
    """build_queue --sweep_id S emits 600 configs under queue/S/pending/, seed-outermost ids, each stamped
    with sweep_id=S and the fixed args (log_distance off)."""
    # copy build_queue into a throwaway run dir so the real queue is untouched; it derives RUN_DIR from its path
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir()
    shutil.copy(os.path.join(HERE, "build_queue.py"), str(slurm_dir / "build_queue.py"))
    subprocess.check_call([sys.executable, str(slurm_dir / "build_queue.py"), "--sweep_id", "test-sweep"])
    # all FOUR queue subdirs must exist: workers rename pending->running->done/failed, so a missing dir would
    # make every claim's os.rename fail and the queue look empty (the bug that stalled the first launch)
    for sub in ("pending", "running", "done", "failed"):
        assert (tmp_path / "queue" / "test-sweep" / sub).is_dir(), f"missing queue subdir {sub}"
    # configs live ONLY under the sweep-scoped pending dir
    pending = tmp_path / "queue" / "test-sweep" / "pending"
    cfgs = sorted((json.load(open(f)) for f in glob.glob(str(pending / "*.json"))), key=lambda c: c["run_id"])
    assert len(cfgs) == 600 and all(c["run_total"] == 600 for c in cfgs)
    # every config is tagged with its sweep_id and carries the fixed args, distance OFF
    assert all(c["sweep_id"] == "test-sweep" for c in cfgs)
    assert all(c["fixed"]["total_timesteps"] == 500000 and c["fixed"]["log_distance"] == 0 for c in cfgs)
    assert all(c["fixed"]["eval_freq"] == 50000 and c["fixed"]["n_eval_episodes"] == 100 for c in cfgs)
    # seed 0 -> ids 0,1,2 (the three algorithms in ALGO_BETA order), with their betas
    assert [(c["run_id"], c["a_seed"], c["algorithm"], c["beta"]) for c in cfgs[:3]] == \
        [(0, 0, "gt_position_velocity", 1.0), (1, 0, "rnd_elliptical", 0.01), (2, 0, "rnd_state", 100.0)]
    assert cfgs[3]["run_id"] == 3 and cfgs[3]["a_seed"] == 1
    assert cfgs[-1]["run_id"] == 599 and cfgs[-1]["a_seed"] == 199
    assert [c["run_id"] for c in cfgs] == list(range(600))
    # the sweep is catalogued in the manifest
    manifest = (tmp_path / "data" / "SWEEPS.md").read_text()
    assert "test-sweep" in manifest and "| 600 |" in manifest


def test_two_sweeps_isolated(tmp_path):
    """Two sweeps in the same run folder land in separate queue subtrees (legacy data never collides)."""
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir()
    shutil.copy(os.path.join(HERE, "build_queue.py"), str(slurm_dir / "build_queue.py"))
    for sid in ("sweep-a", "sweep-b"):
        subprocess.check_call([sys.executable, str(slurm_dir / "build_queue.py"), "--sweep_id", sid])
    a = glob.glob(str(tmp_path / "queue" / "sweep-a" / "pending" / "*.json"))
    b = glob.glob(str(tmp_path / "queue" / "sweep-b" / "pending" / "*.json"))
    assert len(a) == 600 and len(b) == 600  # independent queues, no overlap
    assert (tmp_path / "data" / "SWEEPS.md").read_text().count("| 600 |") == 2  # both catalogued


def test_per_run_json_name_id_based():
    """run4_train.py names the JSON 'NNN_of_TOTAL.json' (id zero-padded to TOTAL's width)."""
    for run_id, run_total, expected in [(42, 600, "042_of_600.json"), (0, 600, "000_of_600.json"),
                                        (599, 600, "599_of_600.json")]:
        assert f"{run_id:0{len(str(run_total))}d}_of_{run_total}.json" == expected
