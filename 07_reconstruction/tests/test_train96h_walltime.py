#!/usr/bin/env python
"""Tests for train96h.py: the walltime end stops training at an eval boundary and marks the
record COMPLETE with `ended_by: "walltime"`; without a wall the run ends at the step cap with
`ended_by: "step_cap"`. Real short trainings (a few minutes).

Run: pytest tests/test_train96h_walltime.py
"""
import glob
import json
import os
import subprocess
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _argv(tmp, walltime_end_epoch, total):
    """The worker-shaped train96h argv for one tiny alg2.3 run (adopted switches included)."""
    return [
        sys.executable, os.path.join(PROJ, "train96h.py"),
        f"--walltime_end_epoch={walltime_end_epoch}",
        "--opt_polyak_foreach=True", "--opt_torch_reward=True",
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
        "--rnd_output_dim=128", "--n_predictors=1", f"--total_timesteps={total}",
    ]


def _run(tmp, walltime_end_epoch, total):
    env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1")
    out = subprocess.run(_argv(tmp, walltime_end_epoch, total), cwd=PROJ, env=env,
                         capture_output=True, text=True, timeout=1200)
    return out.returncode, out.stdout + out.stderr


def _record(tmp):
    paths = glob.glob(f"{tmp}/data/local/*.json")
    assert len(paths) == 1, paths
    with open(paths[0]) as fh:
        return json.load(fh)


def test_walltime_end_marks_complete(tmp_path):
    """A wall a few boundary-lengths away stops the run early, COMPLETE, ended_by=walltime."""
    tmp = str(tmp_path)
    # the wall lands mid-run: startup+warmup ~30-60 s, each 500-step boundary a few seconds,
    # so 90 s of budget ends the run several boundaries in, far short of the 100k cap
    rc, out = _run(tmp, walltime_end_epoch=time.time() + 90, total=100000)
    assert rc == 0, out
    rec = _record(tmp)
    assert rec["completed"] is True
    assert rec["ended_by"] == "walltime"
    steps = [row["step"] for row in rec["eval_history"]]
    assert steps and steps[-1] < 100000            # ended early, at a logged boundary
    assert steps == sorted(steps)
    assert "walltime end at step" in out


def test_step_cap_end(tmp_path):
    """Without a wall the run trains to total_timesteps and records ended_by=step_cap."""
    tmp = str(tmp_path)
    rc, out = _run(tmp, walltime_end_epoch=0, total=2000)
    assert rc == 0, out
    rec = _record(tmp)
    assert rec["completed"] is True
    assert rec["ended_by"] == "step_cap"
    assert [row["step"] for row in rec["eval_history"]] == [500, 1000, 1500, 2000]
