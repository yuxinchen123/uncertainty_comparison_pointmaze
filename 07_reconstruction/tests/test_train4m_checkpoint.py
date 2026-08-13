#!/usr/bin/env python
"""Tests for train4m.py: checkpoint state roundtrips, deterministic suspend/resume across real
subprocess kills, auto-delete of older checkpoints, and record continuity to completion.

The integration test drives train4m.py exactly as the ext4m worker does (same argv shape), with a
tiny budget: 4,000 steps, eval every 500, checkpoint every 1,000. `--suspend_end_epoch=1` (an
epoch in 1970) forces the walltime suspend at the FIRST checkpoint boundary deterministically.

Run: pytest tests/test_train4m_checkpoint.py   (a few minutes: three short real SAC trainings)
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
import train4m
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym

SUSPEND = train4m.SUSPEND_EXIT_CODE


def test_buffer_tail_roundtrip_partial():
    """Golden path: an unfull buffer's newest-n slice restores with pos advanced past it."""
    buf = ReplayBuffer(50, gym.spaces.Box(-1, 1, (3,)), gym.spaces.Box(-1, 1, (2,)))
    for i in range(20):
        buf.add(np.full((1, 3), i, dtype=np.float32), np.full((1, 3), i + 1, dtype=np.float32),
                np.zeros((1, 2), dtype=np.float32), np.array([float(i)]), np.array([False]),
                [{}])
    state = train4m.buffer_tail_state(buf, 8)
    assert state["n"] == 8
    assert state["rewards"][0] == 12.0 and state["rewards"][-1] == 19.0   # newest 8, oldest first
    buf2 = ReplayBuffer(50, gym.spaces.Box(-1, 1, (3,)), gym.spaces.Box(-1, 1, (2,)))
    train4m.buffer_tail_restore(buf2, state)
    assert buf2.pos == 8 and not buf2.full
    assert buf2.rewards[7, 0] == 19.0


def test_buffer_tail_roundtrip_wrapped():
    """Edge case: a FULL ring buffer's tail crosses the wrap point and stays chronological."""
    buf = ReplayBuffer(10, gym.spaces.Box(-1, 1, (1,)), gym.spaces.Box(-1, 1, (1,)))
    for i in range(23):    # pos ends at 3, full=True; newest entries are steps 13..22
        buf.add(np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32), np.array([float(i)]), np.array([False]),
                [{}])
    state = train4m.buffer_tail_state(buf, 6)
    assert list(state["rewards"][:, 0]) == [17.0, 18.0, 19.0, 20.0, 21.0, 22.0]


def test_rms_roundtrip():
    """RunningMeanStd state survives a save/restore with exact values."""
    from gymnasium.wrappers.utils import RunningMeanStd
    a = RunningMeanStd(shape=(4,))
    a.update(np.random.default_rng(0).normal(size=(100, 4)))
    state = train4m.rms_state(a)
    b = RunningMeanStd(shape=(4,))
    train4m.rms_restore(b, state)
    assert np.allclose(a.mean, b.mean) and np.allclose(a.var, b.var) and a.count == b.count


def _argv(tmp, suspend_end_epoch):
    """The worker-shaped train4m argv for one tiny alg2.3 run."""
    return [
        sys.executable, os.path.join(PROJ, "train4m.py"),
        f"--ckpt_dir={tmp}/ckpt", "--ckpt_every=1000",
        f"--suspend_end_epoch={suspend_end_epoch}", "--buffer_tail=800",
        "--algorithm=rnd_next_state", "--beta=30", "--a_seed=7",
        "--env_setup=initial_single_large_pointmaze_max_400",
        "--z_logging_mode=local", "--use_wandb=False",
        f"--local_log_dir={tmp}/data", "--run_id=0", "--run_total=1",
        "--rnd_optimizer=sgd", "--rnd_bonus_readout=l2", "--rnd_lr=0.01",
        "--rnd_update_proportion=1.0", "--rnd_activation=leaky_relu",
        "--rnd_predictor_extra_layers=1", "--rnd_obs_warmup_mode=env_steps",
        "--rnd_obs_warmup_steps=800", "--rnd_reward_norm=False",
        "--rnd_bias_init=normal_0.5", "--rnd_weight_init=orthogonal",
        "--rnd_readout_norm_init=True", "--rnd_layer_norm=True",
        "--eval_freq=500", "--n_eval_episodes=2", "--eval_standalone=False",
        "--log_distance=False", "--device=cpu", "--rnd_obs_norm=True", "--rnd_distance=mse",
        "--rnd_output_dim=128", "--n_predictors=1", "--total_timesteps=4000",
    ]


def _run(tmp, suspend_end_epoch):
    """Run train4m.py once; returns (returncode, combined output)."""
    env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1")
    out = subprocess.run(_argv(tmp, suspend_end_epoch), cwd=PROJ, env=env,
                         capture_output=True, text=True, timeout=1200)
    return out.returncode, out.stdout + out.stderr


def _record(tmp):
    """The run's JSON record (train.py naming: 0_of_1.json under <local_log_dir>/local/)."""
    paths = glob.glob(f"{tmp}/data/local/*.json")
    assert len(paths) == 1, paths
    with open(paths[0]) as fh:
        return json.load(fh)


@pytest.mark.slow
def test_suspend_resume_autodelete_complete(tmp_path):
    """The whole lifecycle: suspend at 1000, resume + suspend at 2000 (older checkpoint deleted),
    resume to 4000 (complete, record continuous, checkpoints gone)."""
    tmp = str(tmp_path)

    # phase 1: a 1970 end-epoch suspends at the FIRST checkpoint boundary (step 1000)
    rc, out = _run(tmp, suspend_end_epoch=1)
    assert rc == SUSPEND, out
    assert [os.path.basename(p) for p in glob.glob(f"{tmp}/ckpt/state_step*.pt")] == \
        ["state_step1000.pt"]
    rec = _record(tmp)
    assert rec["completed"] is False
    assert [row["step"] for row in rec["eval_history"]] == [500, 1000]

    # phase 2: resume, run one more chunk, suspend at 2000; the step-1000 file must be deleted
    rc, out = _run(tmp, suspend_end_epoch=1)
    assert rc == SUSPEND, out
    assert "resumed from state_step1000.pt at step 1000" in out
    assert [os.path.basename(p) for p in glob.glob(f"{tmp}/ckpt/state_step*.pt")] == \
        ["state_step2000.pt"]

    # phase 3: resume with the suspend disabled; the run completes and cleans up
    rc, out = _run(tmp, suspend_end_epoch=0)
    assert rc == 0, out
    assert "resumed from state_step2000.pt at step 2000" in out
    rec = _record(tmp)
    assert rec["completed"] is True
    # continuity: every eval boundary exactly once, no replays, no gaps
    assert [row["step"] for row in rec["eval_history"]] == [500, 1000, 1500, 2000, 2500,
                                                           3000, 3500, 4000]
    steps = [row["step"] for row in rec["train_episode_history"]]
    assert steps == sorted(steps)
    assert glob.glob(f"{tmp}/ckpt/state_step*.pt") == []
