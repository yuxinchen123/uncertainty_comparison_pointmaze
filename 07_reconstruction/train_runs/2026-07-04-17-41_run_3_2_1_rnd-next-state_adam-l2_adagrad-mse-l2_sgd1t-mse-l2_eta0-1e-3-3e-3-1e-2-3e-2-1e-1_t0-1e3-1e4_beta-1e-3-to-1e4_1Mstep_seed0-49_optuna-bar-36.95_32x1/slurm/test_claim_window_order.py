"""Pins worker.py's ordered claim window: every claim comes from the FIRST 32 pending names in
sorted (= run-id, = seed) order, so early seeds run first even with a random pick inside the window.

Run with:  conda run -n exploration python -m pytest slurm/test_claim_window_order.py  (from the run folder)
"""
import importlib
import os
import sys


def _load_worker(tmp_path):
    """Import worker.py against a temp RUN_DIR/SWEEP_ID queue (worker reads env at import time)."""
    os.environ["RUN_DIR"] = str(tmp_path)
    os.environ["SWEEP_ID"] = "testsweep"
    for sub in ("pending", "running", "done", "failed"):
        os.makedirs(tmp_path / "queue" / "testsweep" / sub, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import worker
    return importlib.reload(worker)  # reload so a prior test's env does not stick


def test_claim_stays_in_first_32_sorted_names(tmp_path):
    """60 pending entries; 20 claims each land within the first 32 of the sorted remaining set."""
    worker = _load_worker(tmp_path)
    pending = tmp_path / "queue" / "testsweep" / "pending"
    # zero-padded ids so lexical order == numeric order (the build_queue naming convention)
    names = [f"{i:04d}_of_9200_x_seed{i // 184}.json" for i in range(60)]
    for name in names:
        (pending / name).write_text("{}")
    remaining = sorted(names)
    for _ in range(20):
        name, path = worker.claim()
        assert name is not None
        # the claim window: always one of the first 32 remaining names in sorted order
        assert name in remaining[:32], f"{name} claimed outside the first-32 window"
        remaining.remove(name)
        os.unlink(path)  # tidy running/ so the claim's rename target set stays clean


def test_claim_empty_queue_returns_none(tmp_path):
    """An empty pending folder yields (None, None) so the worker exits cleanly."""
    worker = _load_worker(tmp_path)
    assert worker.claim() == (None, None)
