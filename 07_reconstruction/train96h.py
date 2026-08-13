#!/usr/bin/env python
"""96-hour trainer for the run-6 extension sweep (ext96h): the 1M sweep's pipeline, merged with
the throughput research's adopted optimizations, running until the JOB's walltime.

The user's final design (2026-08-13, replacing the checkpoint/resume 4M variant): each run is a
FRESH single attempt with a 10,000,000-step upper bound that no 96-hour job actually reaches;
the run ends at the last logged milestone before the job's walltime and its record is COMPLETE
there (`ended_by: "walltime"`). No model checkpoints, no resume — exactly train.py's machinery
plus:

- `torch.set_num_interop_threads(1)` (adopted optimization, bit-exact);
- the two Config switches arrive via the worker's argv (`--opt_polyak_foreach=True`,
  `--opt_torch_reward=True`), which train.py already parses;
- a WalltimeEndCallback that stops training cleanly at the last 50k-step boundary the remaining
  walltime can fit, so the final record write always happens inside the job.

Extra args on top of train.py's:
  --walltime_end_epoch <unix>   job end time; 0 disables the walltime stop (default 0)

Exit code 0 always on a clean end (walltime or the 10M cap); the worker marks the run done.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics

from stable_baselines3.common.callbacks import BaseCallback

import train as t
from rnd_exploration.callbacks import LocalLogCheckpointCallback


def parse_ext_args():
    """Split the ext96h-only args out of sys.argv; the remainder feeds train.parse_config()."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--walltime_end_epoch", type=float, default=0.0)
    ext, rest = p.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    return ext


class WalltimeEndCallback(BaseCallback):
    """Stop training at the last eval boundary the remaining walltime can fit, measuring the
    per-boundary wall time live so the margin tracks the node's actual speed."""

    def __init__(self, eval_freq, end_epoch):
        super().__init__()
        self.eval_freq = eval_freq
        self.end_epoch = end_epoch
        self.ended_by_walltime = False
        self._t_last = time.time()
        self._per_boundary = None   # seconds per eval_freq steps, measured after the first

    def _on_step(self):
        if self.end_epoch <= 0 or self.eval_freq <= 0 or \
                self.num_timesteps % self.eval_freq != 0:
            return True
        now = time.time()
        self._per_boundary = now - self._t_last
        self._t_last = now
        # stop when the NEXT 50k-step stretch (plus a write margin) cannot finish before the wall
        if self.end_epoch - now < self._per_boundary * 1.5 + 300:
            print(f"[train96h] walltime end at step {self.num_timesteps}: "
                  f"{(self.end_epoch - now) / 60:.0f} min left < next boundary "
                  f"{self._per_boundary / 60:.0f} min + margin", flush=True)
            self.ended_by_walltime = True
            return False
        return True


def run96h(cfg, ext):
    """One fresh 96-hour run: train.py's construction, learn to the wall or the 10M cap, one
    complete final record."""
    torch.set_num_interop_threads(1)
    seed = cfg.a_seed
    t_start = time.time()
    if os.environ.get("OMP_NUM_THREADS"):
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

    # identical initial seeding to train.run()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gym.register_envs(gymnasium_robotics)

    # construction: the exact sequence of train.run() (its helpers, not a copy of its logic)
    base_env = t.make_base_env(cfg)
    goal_cell, start_cell = t.select_cells(cfg, base_env, seed)
    print(f"Fixed goal cell: {goal_cell}, start cell: {start_cell}")
    is_antmaze = cfg.env_name.startswith("AntMaze")
    train_flat, train_position, train_second = t.build_env_stack(cfg, base_env, start_cell, goal_cell)
    full_obs_dim = int(np.prod(train_flat.observation_space.shape))
    action_dim = int(np.prod(train_flat.action_space.shape))

    intrinsic_model = None
    if cfg.beta > 0:
        warmup_env = None
        if getattr(cfg, "rnd_obs_warmup_mode", "space_sample") == "env_steps":
            warmup_base = t.make_base_env(cfg)
            warmup_env, _, _ = t.build_env_stack(cfg, warmup_base, start_cell, goal_cell)
        from rnd_exploration.methods import EnvContext, build_intrinsic_model
        ctx = EnvContext(
            obs_shape=(full_obs_dim,), action_dim=action_dim,
            observation_space=train_flat.observation_space, action_space=train_flat.action_space,
            position_wrapper=train_position,
            position_velocity_wrapper=None if is_antmaze else train_second,
            position_1m_wrapper=train_second if is_antmaze else train_position,
            env=warmup_env,
        )
        intrinsic_model = build_intrinsic_model(cfg.algorithm, cfg, ctx)
        if warmup_env is not None:
            warmup_env.close()

    train_vec = t.wrap_for_rollout(train_flat, cfg, intrinsic_model, seed)
    model = t.build_sac(cfg, train_vec, intrinsic_model, seed)

    eval_base = t.make_base_env(cfg)
    eval_flat, _, _ = t.build_env_stack(
        cfg, eval_base, start_cell, goal_cell,
        count_map_refs=(train_position.visit_counts, train_second.visit_counts),
        update_counts=False)
    eval_vec = t.wrap_for_rollout(eval_flat, cfg, intrinsic_model, seed)

    callbacks = t.build_callbacks(cfg, train_vec, eval_vec, train_position, train_second,
                                  intrinsic_model, goal_cell, start_cell, t._run_name(cfg), False)

    def diagnostics():
        """The intrinsic model's diagnostics dict, or None (mirrors train.run)."""
        if intrinsic_model is not None and hasattr(intrinsic_model, "diagnostics"):
            return intrinsic_model.diagnostics()
        return None

    def flush_record(completed):
        """Atomically rewrite this run's JSON with everything accumulated so far."""
        t._write_local_log(cfg, time.time() - t_start, callbacks[0].history, callbacks[2].history,
                           callbacks[1].history, callbacks[1].episode_history,
                           intrinsic_diagnostics=diagnostics(), completed=completed)

    wall_cb = WalltimeEndCallback(cfg.eval_freq, ext.walltime_end_epoch)
    callbacks.append(wall_cb)
    callbacks.append(LocalLogCheckpointCallback(eval_freq=cfg.eval_freq,
                                                flush=lambda: flush_record(False)))

    model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks)

    # the run's natural end — the walltime boundary or the step cap — is COMPLETE by design
    flush_record(completed=True)
    # record how the run ended (read-modify-write keeps _write_local_log's format untouched)
    path = os.path.join(cfg.local_log_dir, cfg.z_logging_mode, t._run_name(cfg) + ".json")
    with open(path) as fh:
        rec = json.load(fh)
    rec["ended_by"] = "walltime" if wall_cb.ended_by_walltime else "step_cap"
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, path)
    print(f"runtime_seconds={time.time() - t_start:.2f} mode={cfg.z_logging_mode} "
          f"ended_by={rec['ended_by']} final_step={model.num_timesteps}")
    print("-------------Program Finished-------------")


def main():
    """Parse the ext args + train.py config and run one fresh 96-hour job."""
    ext = parse_ext_args()
    cfg = t.parse_config()
    run96h(cfg, ext)


if __name__ == "__main__":
    main()
