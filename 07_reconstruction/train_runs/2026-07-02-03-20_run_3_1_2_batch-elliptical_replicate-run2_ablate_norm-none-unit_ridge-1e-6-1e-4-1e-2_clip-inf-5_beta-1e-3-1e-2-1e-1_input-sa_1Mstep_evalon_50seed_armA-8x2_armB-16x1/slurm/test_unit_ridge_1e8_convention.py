"""Unit tests for the 2026-07-03 follow-up sweep (unit norm, ridge 1e-8, no clip, beta 5 decades,
50 seeds) and for the ordered-window worker claim required for post-3.1.2 sweeps
(.claude/rules/run-id-and-logging.md). Run: `pytest test_unit_ridge_1e8_convention.py`."""
import os
import sys
import json
import glob
import random
import shutil
import subprocess
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))


def _build(tmp_path, sweep_id):
    """Run build_queue_unit_ridge_1e8.py in a throwaway run dir and return its configs sorted by run_id."""
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "SWEEPS.md").write_text("| sweep_id | arm (shape / seeds) | runs | configs x seeds x steps | eval | status |\n|---|---|---|---|---|---|\n")
    shutil.copy(os.path.join(HERE, "build_queue_unit_ridge_1e8.py"), str(slurm_dir / "build_queue_unit_ridge_1e8.py"))
    subprocess.check_call([sys.executable, str(slurm_dir / "build_queue_unit_ridge_1e8.py"), "--sweep_id", sweep_id])
    pending = tmp_path / "queue" / sweep_id / "pending"
    return sorted((json.load(open(f)) for f in glob.glob(str(pending / "*.json"))), key=lambda c: c["run_id"])


def test_grid_seed_outermost_and_axes(tmp_path):
    """250 configs = 5 betas/seed x 50 seeds; ids gap-free; the single cell's axes and beta order pinned."""
    cfgs = _build(tmp_path, "sweep-followup")
    assert len(cfgs) == 250 and all(c["run_total"] == 250 for c in cfgs)
    assert [c["run_id"] for c in cfgs] == list(range(250))
    assert all(c["sweep_id"] == "sweep-followup" for c in cfgs)
    # all 50 seeds, seed-outermost: first 5 ids are seed 0, next 5 are seed 1
    assert sorted({c["a_seed"] for c in cfgs}) == list(range(50))
    assert all(c["a_seed"] == 0 for c in cfgs[:5]) and cfgs[5]["a_seed"] == 1
    # the cell is fixed: unit normalization, ridge 1e-8, no clip, sample timing, (s,a) input
    for c in cfgs:
        assert c["algorithm"] == "rnd_elliptical"
        assert c["params"] == {"elliptical_feature_normalization": "unit",
                               "elliptical_regularization": "1e-08",
                               "elliptical_bonus_clip": "inf",
                               "elliptical_update_timing": "sample",
                               "elliptical_feature_input": "state_action"}
    # beta inner and ascending within each seed block
    assert [c["beta"] for c in cfgs[:5]] == ["0.0001", "0.001", "0.01", "0.1", "1.0"]
    # eval is ON for every config (numbers compare to run-2's 44.49 / the replica's 52.76)
    assert all(c["fixed"]["eval_standalone"] == "True" and c["fixed"]["total_timesteps"] == 1000000 for c in cfgs)
    # the sweep is catalogued in the manifest
    manifest = (tmp_path / "data" / "SWEEPS.md").read_text()
    assert "sweep-followup" in manifest and "| 250 |" in manifest


def _import_worker(run_dir, sweep_id):
    """Import worker.py fresh with RUN_DIR/SWEEP_ID pointing at a throwaway queue."""
    os.environ["RUN_DIR"] = str(run_dir)
    os.environ["SWEEP_ID"] = sweep_id
    spec = importlib.util.spec_from_file_location("worker_under_test", os.path.join(HERE, "worker.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_claim_stays_in_earliest_32_window(tmp_path):
    """claim() only ever takes a file from the first 32 pending names in sorted (= run-id) order."""
    sweep_id = "claim-test"
    for sub in ("pending", "running", "done", "failed"):
        (tmp_path / "queue" / sweep_id / sub).mkdir(parents=True)
    pending = tmp_path / "queue" / sweep_id / "pending"
    # before: 100 pending files 000..099; after: claims drain them approximately in id order
    names = [f"{i:03d}_of_250_rnd_elliptical_seed{i // 5}.json" for i in range(100)]
    for n in names:
        (pending / n).write_text("{}")
    worker = _import_worker(tmp_path, sweep_id)
    random.seed(0)
    for _ in range(100):
        remaining_window = sorted(os.listdir(pending))[:32]
        name, path = worker.claim()
        assert name in remaining_window, f"claimed {name} outside the earliest-32 window"
        assert os.path.exists(path) and not (pending / name).exists()
    assert worker.claim() == (None, None)  # queue drained


def test_claim_survives_race_loss(tmp_path):
    """If a chosen file vanishes (another worker won the rename), claim() retries and still succeeds."""
    sweep_id = "race-test"
    for sub in ("pending", "running", "done", "failed"):
        (tmp_path / "queue" / sweep_id / sub).mkdir(parents=True)
    pending = tmp_path / "queue" / sweep_id / "pending"
    for i in range(3):
        (pending / f"{i:03d}_of_250_rnd_elliptical_seed0.json").write_text("{}")
    worker = _import_worker(tmp_path, sweep_id)
    # simulate a lost race: remove one file after listing by stealing it before the claim loop runs
    stolen = pending / "000_of_250_rnd_elliptical_seed0.json"
    real_listdir = os.listdir

    def listdir_then_steal(path):
        names = real_listdir(path)
        if stolen.exists():
            stolen.rename(tmp_path / "queue" / sweep_id / "running" / "stolen.json")
        return names

    os.listdir, _saved = listdir_then_steal, os.listdir
    try:
        name, path = worker.claim()
    finally:
        os.listdir = _saved
    assert name is not None and name != "000_of_250_rnd_elliptical_seed0.json"
