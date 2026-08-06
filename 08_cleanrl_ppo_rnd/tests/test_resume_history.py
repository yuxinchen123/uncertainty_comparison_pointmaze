"""The four checkpoint-and-record pairings a resume can encounter, and the signal handler.

A resumed run must end up with a history that stops exactly where the training state stops. Four
situations arise in practice:

| checkpoint | record on disk | what must happen |
|---|---|---|
| version 2 | absent (the sweep's requeue moved it) | history comes from the checkpoint |
| version 2 | present and AHEAD of the checkpoint | history comes from the checkpoint; the record's extra rows are discarded |
| version 1 | present and ahead | history comes from the record, TRUNCATED to the checkpoint |
| version 1 | absent | no history; the run logs that and continues |

Version 1 is not hypothetical: the 30 runs already training were started before the history was
carried, so every checkpoint they write for the rest of their lives is version 1.
"""

import json
import os
import signal
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from checkpointing import load_checkpoint, save_checkpoint  # noqa: E402
from run_record import RunRecord  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "..", "src", "ppo_rnd_envpool_shuze.py")
PYTHON = "/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python"


class TinyRMS:
    """Stand-in for gym's RunningMeanStd."""

    def __init__(self):
        import numpy as np
        self.mean, self.var, self.count = np.zeros(1), np.ones(1), 1e-4


class TinyFilter:
    """Stand-in for RewardForwardFilter."""

    def __init__(self):
        self.rewems = None


def pieces():
    """Build the objects save_checkpoint needs."""
    agent = nn.Linear(4, 2)
    rnd = nn.ModuleDict({"predictor": nn.Linear(4, 3), "target": nn.Linear(4, 3)})
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    return agent, rnd, opt, TinyRMS(), TinyRMS(), TinyFilter()


def build_record(path, n_updates, n_episodes):
    """A record holding n_updates logged updates and n_episodes episodes."""
    rec = RunRecord(path, {"run_id": 0, "run_total": 1})
    for i in range(1, n_updates + 1):
        rec.add_update({"step": i * 100, "update": i}, {"step": i * 100, "update": i})
    for i in range(n_episodes):
        rec.add_episode({"step": i * 10})
    return rec


def save(path, record, update, step):
    """Write a checkpoint at (update, step), carrying record's history when record is given."""
    agent, rnd, opt, o, r, f = pieces()
    return save_checkpoint(path, agent=agent, rnd_model=rnd, optimizer=opt, obs_rms=o,
                           reward_rms=r, discounted_reward=f, global_step=step, update=update,
                           avg_returns=[], device=torch.device("cpu"), record=record)


def load(path):
    """Read a checkpoint back."""
    agent, rnd, opt, o, r, f = pieces()
    return load_checkpoint(path, agent=agent, rnd_model=rnd, optimizer=opt, obs_rms=o,
                           reward_rms=r, discounted_reward=f, device=torch.device("cpu"))


def test_v2_checkpoint_with_record_moved_away(tmp_path):
    """The sweep's requeue case: the record is gone, so the checkpoint's history is all there is."""
    rp, cp = str(tmp_path / "0_of_1.json"), str(tmp_path / "0_of_1.checkpoint.pt")
    rec = build_record(rp, n_updates=6, n_episodes=40)
    save(cp, rec, update=6, step=600)
    os.remove(rp) if os.path.exists(rp) else None      # the requeue moved it

    state = load(cp)
    assert state["record_history"] is not None
    fresh = RunRecord(rp, {"run_id": 0, "run_total": 1})
    note = fresh.restore_for_resume(state["record_history"], state["update"], state["global_step"])
    assert len(fresh.train_history) == 6
    assert fresh.episodes_seen == 40
    assert "from the checkpoint" in note


def test_v2_checkpoint_beats_a_record_that_ran_ahead(tmp_path):
    """The record is fresher than the checkpoint; the checkpoint must still win.

    The rows the record has and the checkpoint does not describe updates the resumed run is about
    to redo. Keeping them would duplicate every row in the overlap.
    """
    rp, cp = str(tmp_path / "0_of_1.json"), str(tmp_path / "0_of_1.checkpoint.pt")
    rec = build_record(rp, n_updates=6, n_episodes=40)
    save(cp, rec, update=6, step=600)          # checkpoint at update 6
    for i in range(7, 11):                     # the run kept going and logged four more
        rec.add_update({"step": i * 100, "update": i}, {"step": i * 100, "update": i})
    rec.flush()
    assert len(json.load(open(rp))["train_history"]) == 10

    state = load(cp)
    fresh = RunRecord(rp, {"run_id": 0, "run_total": 1})
    fresh.restore_for_resume(state["record_history"], state["update"], state["global_step"])
    # before: the record on disk holds updates 1..10, the checkpoint stopped at 6
    # after:  the resumed history holds 1..6, so updates 7..10 are redone once, not twice
    assert [r["update"] for r in fresh.train_history] == [1, 2, 3, 4, 5, 6]


def test_v1_checkpoint_falls_back_to_the_record_truncated(tmp_path):
    """A checkpoint from a process that predates the history: use the record, cut at the checkpoint."""
    rp, cp = str(tmp_path / "0_of_1.json"), str(tmp_path / "0_of_1.checkpoint.pt")
    rec = build_record(rp, n_updates=10, n_episodes=80)
    rec.flush()
    save(cp, None, update=6, step=600)                 # record=None -> format 1
    payload = torch.load(cp, weights_only=False)
    assert payload["format_version"] == 1 and "record_history" not in payload

    state = load(cp)
    assert state["record_history"] is None
    fresh = RunRecord(rp, {"run_id": 0, "run_total": 1})
    note = fresh.restore_for_resume(None, state["update"], state["global_step"])
    assert [r["update"] for r in fresh.train_history] == [1, 2, 3, 4, 5, 6]
    side = rp[: -len(".json")] + ".episodes.jsonl"
    if os.path.exists(side):
        assert all(json.loads(l)["step"] <= 600 for l in open(side) if l.strip())
    assert "truncated to update 6" in note


def test_v1_checkpoint_with_no_record(tmp_path):
    """Edge case: nothing to restore, and the run says so rather than failing."""
    rp, cp = str(tmp_path / "0_of_1.json"), str(tmp_path / "0_of_1.checkpoint.pt")
    save(cp, None, update=6, step=600)
    state = load(cp)
    fresh = RunRecord(rp, {"run_id": 0, "run_total": 1})
    note = fresh.restore_for_resume(None, state["update"], state["global_step"])
    assert fresh.train_history == [] and fresh.episodes_seen == 0
    assert "no history to restore" in note


def test_a_v1_checkpoint_is_still_readable(tmp_path):
    """The 30 live runs write version-1 checkpoints; refusing them would restart all of them."""
    cp = str(tmp_path / "c.pt")
    save(cp, None, update=3, step=300)
    state = load(cp)
    assert state is not None and state["update"] == 3, "a version-1 checkpoint must still load"


def test_history_in_the_checkpoint_costs_little(tmp_path):
    """The carried history must not bloat the checkpoint out of proportion."""
    cp_small, cp_big = str(tmp_path / "s.pt"), str(tmp_path / "b.pt")
    rec = build_record(str(tmp_path / "r.json"), n_updates=611, n_episodes=50000)
    small = save(cp_small, None, update=1, step=1)["bytes"]
    big = save(cp_big, rec, update=1, step=1)["bytes"]
    added_mb = (big - small) / 1024 / 1024
    # The episode rows are in the sidecar, so only the interval history rides in the checkpoint.
    assert added_mb < 5, f"history added {added_mb:.1f} MB, more than expected"
    print(f"\n  history adds {added_mb:.1f} MB at the run's final size")


def test_sigterm_checkpoints_and_exits_cleanly(tmp_path):
    """A clean kill must leave a checkpoint, not lose everything since the last one."""
    out = str(tmp_path / "sig")
    os.makedirs(out, exist_ok=True)
    cmd = [PYTHON, "-u", TRAINER, "--env_id", "MontezumaRevenge-v5", "--num_envs", "8",
           "--num_steps", "16", "--total_timesteps", "163840", "--num_iterations_obs_norm_init", "1",
           "--num_minibatches", "2", "--update_epochs", "1", "--log_every_updates", "1",
           "--output_dir", out, "--run_id", "0", "--run_total", "1", "--seed", "1", "--no-cuda",
           "--opt_env_threads", "2", "--minibatch_drop_allowance", "8",
           # Far in the future, so ONLY the signal can produce a checkpoint.
           "--checkpoint_every_seconds", "1e9", "--no-resume"]
    env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    # Let it get past start-up and into the training loop, then ask it to stop.
    deadline, started = None, False
    import time as _t
    t0 = _t.time()
    while _t.time() - t0 < 300:
        if os.path.exists(os.path.join(out, "0_of_1.json")):
            started = True
            break
        if proc.poll() is not None:
            break
        _t.sleep(2)
    assert started, "the run never reached its first logged update"
    proc.send_signal(signal.SIGTERM)
    stdout, _ = proc.communicate(timeout=300)

    assert proc.returncode == 0, f"expected a clean exit after SIGTERM, got {proc.returncode}\n{stdout[-2000:]}"
    assert "termination requested" in stdout, stdout[-2000:]
    ckpt = os.path.join(out, "0_of_1.checkpoint.pt")
    assert os.path.exists(ckpt), "SIGTERM did not leave a checkpoint"
    payload = torch.load(ckpt, weights_only=False)
    assert payload["format_version"] == 2, "the signal checkpoint should carry the history"
    assert payload["global_step"] > 0
