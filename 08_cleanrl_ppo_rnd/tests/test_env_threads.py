"""Unit tests for resolve_env_threads: the per-run share must win over the job's total.

This is the setting a packed worker job gets wrong by default. A job holding 5 slots on one GPU is
allocated 40 cpus, so SLURM_CPUS_PER_TASK is 40 for every one of its 5 runs — each would start 40
envpool worker threads on 8 cores' worth of allocation. The sweep's worker manager exports
GPU_SWEEP_CPUS_PER_RUN with this run's real share, which is what must be used.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from ppo_rnd_envpool_shuze import resolve_env_threads  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Run each test with both cpu-count variables unset."""
    monkeypatch.delenv("GPU_SWEEP_CPUS_PER_RUN", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)


def test_explicit_request_wins(monkeypatch):
    """Golden path: an explicit --opt_env_threads beats both environment variables."""
    monkeypatch.setenv("GPU_SWEEP_CPUS_PER_RUN", "8")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "40")
    assert resolve_env_threads(12) == 12


def test_per_run_share_beats_the_jobs_total(monkeypatch):
    """The packed-job case: 5 runs in a 40-cpu job must each take 8, not 40."""
    # before: SLURM_CPUS_PER_TASK=40 for all 5 runs -> 5 x 40 = 200 threads on 40 cores
    # after:  GPU_SWEEP_CPUS_PER_RUN=8 per run      -> 5 x  8 =  40 threads on 40 cores
    monkeypatch.setenv("GPU_SWEEP_CPUS_PER_RUN", "8")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "40")
    assert resolve_env_threads(0) == 8


def test_falls_back_to_the_job_total_when_unpacked(monkeypatch):
    """An unpacked run outside the sweep harness still reads the job's cpu count."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    assert resolve_env_threads(0) == 16


def test_falls_back_to_the_machine_off_slurm():
    """Edge case: off Slurm entirely, fall back to the machine's cpu count."""
    assert resolve_env_threads(0) == (os.cpu_count() or 1)
