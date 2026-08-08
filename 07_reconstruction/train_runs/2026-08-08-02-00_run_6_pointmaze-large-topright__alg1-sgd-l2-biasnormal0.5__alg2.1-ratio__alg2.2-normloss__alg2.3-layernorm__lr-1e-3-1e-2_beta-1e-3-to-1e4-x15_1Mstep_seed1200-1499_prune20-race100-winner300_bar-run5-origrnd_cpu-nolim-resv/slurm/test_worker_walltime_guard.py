#!/usr/bin/env python
"""Tests for the worker's walltime guard — the rule that stops a worker claiming a run its job
cannot finish.

The guard exists because this run writes no model checkpoints: a run killed at the job's walltime
restarts from zero somewhere else, so starting a 16-hour run with 3 hours left is guaranteed waste.
Covered here: the golden path (plenty of time -> claim), the edge the guard is for (too little time
-> stop), the exact boundary, and the fail-open behaviour when the job's end time cannot be read —
which matters most, because a guard that failed closed on a scontrol hiccup would idle the fleet.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python -m pytest slurm/test_worker_walltime_guard.py -q
"""
import importlib
import os
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)


def load_worker(monkeypatch, job_end_offset_hours=None, required_hours=None, job_id="123"):
    """Import a fresh copy of worker.py with the job's end time set `job_end_offset_hours` from now.

    worker.py resolves the job end ONCE at import (it is fixed for a job's life), so each case needs
    its own import rather than a mutated module.
    """
    monkeypatch.setenv("RUN_DIR", RUN_DIR)
    monkeypatch.setenv("SWEEP_ID", "test-sweep")
    monkeypatch.setenv("SLURM_JOB_ID", job_id)
    if required_hours is not None:
        monkeypatch.setenv("WORKER_REQUIRED_HOURS", str(required_hours))
    else:
        monkeypatch.delenv("WORKER_REQUIRED_HOURS", raising=False)
    if job_end_offset_hours is None:
        monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_END_TIME",
                           str(int(time.time() + job_end_offset_hours * 3600)))
    sys.path.insert(0, HERE)
    if "worker" in sys.modules:
        del sys.modules["worker"]
    return importlib.import_module("worker")


def test_claims_when_the_job_has_plenty_of_time(monkeypatch):
    """A fresh 4-day job is far above the 20 h a run needs."""
    w = load_worker(monkeypatch, job_end_offset_hours=96)
    assert w.enough_walltime() is True


def test_stops_claiming_when_the_job_is_running_out(monkeypatch):
    """3 h left against a ~15.8 h run: the guard's whole reason for existing."""
    w = load_worker(monkeypatch, job_end_offset_hours=3)
    assert w.enough_walltime() is False


def test_the_boundary_is_the_required_hours(monkeypatch):
    """Just above the threshold claims; just below does not."""
    w = load_worker(monkeypatch, job_end_offset_hours=20.5, required_hours=20)
    assert w.enough_walltime() is True
    w = load_worker(monkeypatch, job_end_offset_hours=19.5, required_hours=20)
    assert w.enough_walltime() is False


def test_the_required_hours_are_configurable(monkeypatch):
    """A run length change is an env var, not a code edit: 10 h left passes a 6 h requirement."""
    w = load_worker(monkeypatch, job_end_offset_hours=10, required_hours=6)
    assert w.enough_walltime() is True
    w = load_worker(monkeypatch, job_end_offset_hours=10, required_hours=24)
    assert w.enough_walltime() is False


def test_fails_open_when_the_job_end_time_is_unknown(monkeypatch):
    """No SLURM_JOB_END_TIME and no job id: claim anyway.

    The guard is an optimization, never a correctness rule. Failing closed here would idle every
    worker on the first scontrol hiccup, which is far worse than the waste the guard prevents.
    """
    monkeypatch.setenv("RUN_DIR", RUN_DIR)
    monkeypatch.setenv("SWEEP_ID", "test-sweep")
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    sys.path.insert(0, HERE)
    if "worker" in sys.modules:
        del sys.modules["worker"]
    w = importlib.import_module("worker")
    assert w.JOB_END is None
    assert w.enough_walltime() is True


def test_the_default_requirement_covers_the_measured_run_length(monkeypatch):
    """The default must exceed the observed median run (15.8 h on 2026-08-06) with margin."""
    w = load_worker(monkeypatch, job_end_offset_hours=96)
    assert w.REQUIRED_SECONDS / 3600 >= 18


def test_a_20_day_nolim_job_is_never_blocked(monkeypatch):
    """The nolim partition's 20-day jobs must never stop claiming for walltime reasons."""
    w = load_worker(monkeypatch, job_end_offset_hours=20 * 24)
    assert w.enough_walltime() is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
