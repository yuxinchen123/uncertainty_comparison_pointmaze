"""Unit tests for run_record.RunRecord: the golden path, the episode cap, and resume."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from run_record import RunRecord, record_filename  # noqa: E402


def test_flush_writes_config_and_history(tmp_path):
    """Golden path: a flushed record carries the config, the history lists and completed=False."""
    path = str(tmp_path / "0_of_30.json")
    rec = RunRecord(path, {"run_id": 0, "run_total": 30, "a_seed": 7, "algorithm": "ppo_rnd"})
    rec.add_episode({"step": 128, "train/extrinsic_reward": 0.0, "train/episode_length": 64})
    rec.add_update({"step": 128, "train/mean_extrinsic_reward": 0.0}, {"step": 128, "charts/steps_per_second": 900.0})
    rec.flush(completed=False)

    on_disk = json.load(open(path))
    assert on_disk["run_id"] == 0 and on_disk["a_seed"] == 7
    assert on_disk["completed"] is False
    assert len(on_disk["train_episode_history"]) == 1
    assert len(on_disk["train_history"]) == 1 and len(on_disk["eval_history"]) == 1
    assert on_disk["runtime_seconds"] >= 0.0
    # The record is group-readable, which is what lets a collaborator's analysis open it.
    assert os.stat(path).st_mode & 0o060 == 0o060


def test_flush_completed_flag(tmp_path):
    """A run that reaches its end flushes completed=True."""
    path = str(tmp_path / "1_of_30.json")
    rec = RunRecord(path, {"run_id": 1, "run_total": 30})
    rec.flush(completed=True)
    assert json.load(open(path))["completed"] is True


def test_episode_cap_keeps_every_stride_past_the_cap(tmp_path):
    """Edge case: past the cap only every stride-th episode is kept, and the drops are counted."""
    path = str(tmp_path / "2_of_30.json")
    rec = RunRecord(path, {"run_id": 2, "run_total": 30}, episode_history_cap=10, episode_stride=5)
    # before: 30 episodes reported, cap=10, stride=5
    # after:  10 kept outright, then episodes 15, 20, 25, 30 kept -> 14 rows, 16 dropped
    for i in range(30):
        rec.add_episode({"step": i})
    assert rec.episodes_seen == 30
    assert len(rec.train_episode_history) == 14
    assert rec.episodes_dropped == 16
    rec.flush()
    on_disk = json.load(open(path))
    assert on_disk["episodes_seen"] == 30
    assert on_disk["episodes_dropped_from_history"] == 16


def test_restore_from_disk_appends_instead_of_restarting(tmp_path):
    """Resume: a second RunRecord over the same path continues the history and the runtime."""
    path = str(tmp_path / "3_of_30.json")
    first = RunRecord(path, {"run_id": 3, "run_total": 30})
    for i in range(4):
        first.add_episode({"step": i})
    first.flush()
    first_runtime = json.load(open(path))["runtime_seconds"]

    second = RunRecord(path, {"run_id": 3, "run_total": 30})
    assert second.restore_from_disk() is True
    assert second.episodes_seen == 4
    second.add_episode({"step": 4})
    second.flush()

    on_disk = json.load(open(path))
    assert len(on_disk["train_episode_history"]) == 5
    assert [e["step"] for e in on_disk["train_episode_history"]] == [0, 1, 2, 3, 4]
    # The resumed record's runtime includes the first segment's, so it never goes backwards.
    assert on_disk["runtime_seconds"] >= first_runtime


def test_restore_from_disk_on_a_fresh_run(tmp_path):
    """Edge case: with no file on disk, restore reports False and leaves the record empty."""
    rec = RunRecord(str(tmp_path / "4_of_30.json"), {"run_id": 4, "run_total": 30})
    assert rec.restore_from_disk() is False
    assert rec.train_episode_history == [] and rec.episodes_seen == 0


def test_record_filename():
    """The filename pattern matches the project's analysis loader."""
    assert record_filename(7, 30) == "7_of_30.json"
