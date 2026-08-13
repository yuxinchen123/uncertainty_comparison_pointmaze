#!/usr/bin/env python
"""Checkpointing trainer for the run-6 4M-step extension sweep (ext4m).

A thin resumable layer over train.py (which is untouched — the live 1M sweep still runs it):
same construction path, plus a model checkpoint every --ckpt_every steps at the eval boundary
(only the newest kept), restore-on-start when a checkpoint exists, and a walltime suspend that
exits with code 3 so the worker re-pends the marker for the next claimer to resume.

Checkpoint contents and the accepted resume infidelities are documented in the run folder's
slurm/EXT4M_DESIGN.md. The replay buffer is NOT saved except its most recent --buffer_tail
transitions (the user's storage decision).

Extra args on top of train.py's:
  --ckpt_dir <dir>            checkpoint directory for THIS run (required)
  --ckpt_every <steps>        checkpoint cadence, must be a multiple of eval_freq (default 500000)
  --suspend_end_epoch <unix>  job end time; 0 disables the walltime suspend (default 0)
  --buffer_tail <n>           replay-buffer tail size saved per checkpoint (default 100000)

Exit codes: 0 = run complete; 3 = suspended at a checkpoint (resume me); anything else = failure.
"""
import argparse
import glob
import hashlib
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics

from stable_baselines3.common.callbacks import BaseCallback

import train as t
from rnd_exploration.methods.rnd import RND
from rnd_exploration.callbacks import LocalLogCheckpointCallback

SUSPEND_EXIT_CODE = 3


def parse_ext_args():
    """Split the ext4m-only args out of sys.argv; the remainder feeds train.parse_config()."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--ckpt_every", type=int, default=500000)
    p.add_argument("--suspend_end_epoch", type=float, default=0.0)
    p.add_argument("--buffer_tail", type=int, default=100000)
    ext, rest = p.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    return ext


def substream_seed(*parts):
    """Keyed 32-bit seed per the rng-seeding rule, so resume reseeding never replays a stream.
    before: parts = (1500, 'resume', 2000000); after: int in [0, 2^32) from sha256('1500::resume::2000000')."""
    key = "::".join(str(p) for p in parts)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF


def rms_state(rms):
    """The three arrays of a gymnasium RunningMeanStd, or None for a missing one."""
    if rms is None:
        return None
    return {"mean": np.asarray(rms.mean).copy(), "var": np.asarray(rms.var).copy(),
            "count": float(rms.count)}


def rms_restore(rms, state):
    """Write a saved RunningMeanStd state back into a live instance (same shapes by construction)."""
    rms.mean = np.asarray(state["mean"]).copy()
    rms.var = np.asarray(state["var"]).copy()
    rms.count = state["count"]


def buffer_tail_state(buf, n_tail):
    """The newest min(n_tail, n_valid) transitions of an SB3 replay buffer, oldest first.

    before: buf.observations shape (1000000, 1, obs_dim), pos=734210, full=False
    after:  {"n": 100000, "observations": (100000, 1, obs_dim) slice of steps 634210..734209, ...}
    """
    valid = buf.buffer_size if buf.full else buf.pos
    n = min(n_tail, valid)
    # chronological index of the newest n entries, handling the ring's wrap when full
    end = buf.pos
    idx = (np.arange(end - n, end) % buf.buffer_size) if buf.full else np.arange(end - n, end)
    return {"n": int(n),
            "observations": buf.observations[idx].copy(),
            "next_observations": buf.next_observations[idx].copy(),
            "actions": buf.actions[idx].copy(),
            "rewards": buf.rewards[idx].copy(),
            "dones": buf.dones[idx].copy(),
            "timeouts": buf.timeouts[idx].copy()}


def buffer_tail_restore(buf, state):
    """Refill an EMPTY replay buffer's front with a saved tail; pos advances past it."""
    n = state["n"]
    for name in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
        getattr(buf, name)[:n] = state[name]
    buf.pos = n
    buf.full = False


def sac_state(model):
    """Every mutable SAC piece a resume needs (nets, optimizers, entropy coef, counters)."""
    return {"policy": model.policy.state_dict(),
            "actor_opt": model.actor.optimizer.state_dict(),
            "critic_opt": model.critic.optimizer.state_dict(),
            "log_ent_coef": model.log_ent_coef.detach().clone(),
            "ent_coef_opt": model.ent_coef_optimizer.state_dict(),
            "num_timesteps": int(model.num_timesteps),
            "n_updates": int(model._n_updates),
            "episode_num": int(model._episode_num)}


def sac_restore(model, state):
    """Load a saved SAC state into a freshly built model."""
    model.policy.load_state_dict(state["policy"])
    model.actor.optimizer.load_state_dict(state["actor_opt"])
    model.critic.optimizer.load_state_dict(state["critic_opt"])
    with torch.no_grad():
        model.log_ent_coef.copy_(state["log_ent_coef"])
    model.ent_coef_optimizer.load_state_dict(state["ent_coef_opt"])
    model.num_timesteps = state["num_timesteps"]
    model._n_updates = state["n_updates"]
    model._episode_num = state["episode_num"]


def rnd_state(m):
    """The RND model's mutable state, or None for a bonus with none (gt_* visit counts)."""
    if not isinstance(m, RND):
        return None
    return {"predictor": m.predictor.state_dict(),
            "target": m.target.state_dict(),
            "init_predictor": m.init_predictor.state_dict() if m.init_predictor is not None else None,
            "opt": m.opt.state_dict(),
            "obs_rms": rms_state(m.obs_rms),
            "reward_rms": rms_state(m.reward_rms),
            "rff_return": None if m._rff_return is None else np.asarray(m._rff_return).copy()}


def rnd_restore(m, state):
    """Load a saved RND state into a freshly built RND model."""
    m.predictor.load_state_dict(state["predictor"])
    m.target.load_state_dict(state["target"])
    if state["init_predictor"] is not None:
        m.init_predictor.load_state_dict(state["init_predictor"])
    m.opt.load_state_dict(state["opt"])
    if state["obs_rms"] is not None:
        rms_restore(m.obs_rms, state["obs_rms"])
    if state["reward_rms"] is not None:
        rms_restore(m.reward_rms, state["reward_rms"])
    m._rff_return = None if state["rff_return"] is None else np.asarray(state["rff_return"]).copy()


class CheckpointSuspendCallback(BaseCallback):
    """Write the resume checkpoint every ckpt_every steps (newest only), then suspend the run
    when the job's remaining walltime cannot fit the next 0.5M-step chunk."""

    def __init__(self, ckpt_every, write_fn, suspend_end_epoch):
        super().__init__()
        self.ckpt_every = ckpt_every
        self.write_fn = write_fn
        self.suspend_end_epoch = suspend_end_epoch
        self.suspended = False
        self._chunk_t0 = time.time()
        self._next = None   # set on first step from the restored num_timesteps

    def _on_step(self):
        # first call fixes the next boundary from wherever the (possibly restored) run stands
        if self._next is None:
            self._next = (self.num_timesteps // self.ckpt_every + 1) * self.ckpt_every
        if self.num_timesteps < self._next:
            return True
        chunk_seconds = time.time() - self._chunk_t0
        self.write_fn(self.num_timesteps)
        self._chunk_t0 = time.time()
        self._next += self.ckpt_every
        # suspend when the remaining walltime cannot fit another chunk like the last one
        if self.suspend_end_epoch:
            remaining = self.suspend_end_epoch - time.time()
            if remaining < chunk_seconds * 1.15 + 900:
                print(f"[train4m] suspending at step {self.num_timesteps}: {remaining/3600:.1f} h "
                      f"walltime left < last chunk {chunk_seconds/3600:.1f} h + margin", flush=True)
                self.suspended = True
                return False
        return True


def newest_checkpoint(ckpt_dir):
    """Path of the newest state_step<N>.pt in ckpt_dir, or None."""
    paths = glob.glob(os.path.join(ckpt_dir, "state_step*.pt"))
    return max(paths, key=lambda p: int(p.split("state_step")[1].split(".pt")[0])) if paths else None


def record_path(cfg):
    """This run's JSON record path (train.py's naming: <local_log_dir>/<mode>/<run_name>.json)."""
    return os.path.join(cfg.local_log_dir, cfg.z_logging_mode, t._run_name(cfg) + ".json")


def load_histories(cfg, upto_step):
    """The record's four history lists truncated to step <= upto_step, for callback pre-seeding.

    before: eval_history=[{'step':50000,...} .. {'step':2050000,...}] on disk, upto_step=2000000
    after:  ([{'step':50000,...} .. {'step':2000000,...}], ..., ...) with every list truncated
    """
    with open(record_path(cfg)) as fh:
        d = json.load(fh)
    def cut(rows):
        return [r for r in rows if r.get("step", 0) <= upto_step]
    return (cut(d.get("eval_history", [])), cut(d.get("train_history", [])),
            cut(d.get("train_episode_history", [])), cut(d.get("distance_history", [])))


def run4m(cfg, ext):
    """One resumable 4M run: train.py's construction, checkpoint restore, learn, suspend/finish."""
    if ext.ckpt_every % cfg.eval_freq != 0:
        sys.exit(f"ckpt_every {ext.ckpt_every} must be a multiple of eval_freq {cfg.eval_freq}")
    seed = cfg.a_seed
    t_start = time.time()
    if os.environ.get("OMP_NUM_THREADS"):
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

    # identical initial seeding to train.run(), so a fresh ext4m run is a faithful fresh run
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

    # restore from the newest checkpoint, if this run was suspended or killed before
    ckpt_path = newest_checkpoint(ext.ckpt_dir)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        step = state["meta"]["step"]
        sac_restore(model, state["sac"])
        if state["rnd"] is not None:
            rnd_restore(intrinsic_model, state["rnd"])
        # in-place so the eval stack's count_map_refs (same arrays by reference) see the restore
        train_position.visit_counts[:] = state["counts"]["position"]
        train_second.visit_counts[:] = state["counts"]["second"]
        buffer_tail_restore(model.replay_buffer, state["buffer"])
        ev, tr, ep, di = load_histories(cfg, step)
        callbacks[0].history = ev
        callbacks[1].history = tr
        callbacks[1].episode_history = ep
        callbacks[2].history = di
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        # fresh keyed env seeds: never replay the original seed's episode stream after a resume
        train_vec.seed(substream_seed(seed, "resume-train", step))
        eval_vec.seed(substream_seed(seed, "resume-eval", step))
        model._last_obs = None   # forces learn() to reset the train env once
        print(f"[train4m] resumed from {os.path.basename(ckpt_path)} at step {step} "
              f"(buffer tail {state['buffer']['n']} transitions, "
              f"histories {len(ev)}/{len(tr)}/{len(ep)}/{len(di)} rows)", flush=True)

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

    def write_checkpoint(step):
        """Write state_step<step>.pt atomically, then delete every older checkpoint."""
        os.makedirs(ext.ckpt_dir, exist_ok=True)
        state = {
            "meta": {"step": int(step), "run_id": cfg.run_id, "wrote_at": time.time(),
                     "a_seed": seed, "algorithm": cfg.algorithm},
            "sac": sac_state(model),
            "rnd": rnd_state(intrinsic_model),
            "counts": {"position": np.asarray(train_position.visit_counts).copy(),
                       "second": np.asarray(train_second.visit_counts).copy()},
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state()},
            "buffer": buffer_tail_state(model.replay_buffer, ext.buffer_tail),
        }
        final = os.path.join(ext.ckpt_dir, f"state_step{step}.pt")
        tmp = final + ".tmp"
        torch.save(state, tmp, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, final)
        for old in glob.glob(os.path.join(ext.ckpt_dir, "state_step*.pt")):
            if old != final:
                os.remove(old)
        flush_record(completed=False)
        print(f"[train4m] checkpoint at step {step} "
              f"({os.path.getsize(final) / 1e6:.1f} MB, older deleted)", flush=True)

    ckpt_cb = CheckpointSuspendCallback(ext.ckpt_every, write_checkpoint, ext.suspend_end_epoch)
    callbacks.append(ckpt_cb)
    # the record flush at every eval cadence (train.run's checkpointed-record behavior)
    callbacks.append(LocalLogCheckpointCallback(eval_freq=cfg.eval_freq,
                                                flush=lambda: flush_record(False)))

    remaining = cfg.total_timesteps - model.num_timesteps
    model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=False)

    if ckpt_cb.suspended:
        flush_record(completed=False)
        print(f"[train4m] suspended cleanly at step {model.num_timesteps}; exit {SUSPEND_EXIT_CODE}",
              flush=True)
        sys.exit(SUSPEND_EXIT_CODE)

    # complete: final record, then the checkpoint directory is deleted (the record is the artifact)
    flush_record(completed=True)
    for p in glob.glob(os.path.join(ext.ckpt_dir, "state_step*.pt")):
        os.remove(p)
    if os.path.isdir(ext.ckpt_dir) and not os.listdir(ext.ckpt_dir):
        os.rmdir(ext.ckpt_dir)
    print(f"runtime_seconds={time.time() - t_start:.2f} mode={cfg.z_logging_mode}")
    print("-------------Program Finished-------------")


def main():
    """Parse the ext args + train.py config and run one resumable job."""
    # single-thread workers have nothing for the inter-op pool to schedule (throughput research
    # 2026-08-13, gated bit-exact on all three configurations; must run before any torch work)
    torch.set_num_interop_threads(1)
    ext = parse_ext_args()
    cfg = t.parse_config()
    run4m(cfg, ext)


if __name__ == "__main__":
    main()
