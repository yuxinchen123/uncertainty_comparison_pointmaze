#!/usr/bin/env python3
"""
fast_sac_rnd.py -- a lean, single-file SAC + intrinsic-bonus trainer (SEPARATE codebase, bold rewrite).

Why: the SB3 path spends ~20-30 ms/step of its ~50 ms/step in plumbing (DummyVecEnv + Monitor + 3
wrappers + 3 callbacks + collect_rollouts<->train ping-pong + polyak zip_strict + the buffer's
numpy<->torch round-trip), on top of the ~28 ms compute core. This rewrite keeps the SAC math and the
intrinsic methods IDENTICAL (it reuses the project's exact env stack + intrinsic model from train.py) but
replaces SB3's training loop with a hand-written CleanRL-style loop: one process, one env, a preallocated
torch replay buffer, intrinsic reward folded into the batched update, and a torch._foreach_ soft target
update. Goal: ~1.5-2x throughput, env-agnostic so it survives heavier envs later.

It is NOT yet validated to convergence against the SB3 runs -- it is presented for a merge decision.
Validate with --env Pendulum-v1 (SAC core learns fast) and benchmark fps with --env pointmaze.

Usage:
  python fast_sac_rnd.py --env pointmaze --algorithm rnd_state --beta 100 --steps 8000 --warmup 1000 --out fps.json
  python fast_sac_rnd.py --env Pendulum-v1 --steps 12000 --validate            # SAC-core sanity (reward goes up)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJ = "/p/rlprojects/RND/07_reconstruction"
import sys
sys.path.insert(0, PROJ)

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Actor(nn.Module):
    """Tanh-squashed Gaussian policy (CleanRL SAC actor): 256x256, outputs mean+logstd, samples + logprob."""

    def __init__(self, obs_dim: int, act_dim: int, act_low, act_high):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, act_dim)
        self.logstd = nn.Linear(256, act_dim)
        # action rescaling buffers (env action bounds)
        self.register_buffer("act_scale", torch.tensor((act_high - act_low) / 2.0, dtype=torch.float32))
        self.register_buffer("act_bias", torch.tensor((act_high + act_low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        # shared trunk -> mean + squashed logstd in [LOG_STD_MIN, LOG_STD_MAX]
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mean = self.mean(h)
        logstd = torch.tanh(self.logstd(h))
        logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logstd + 1.0)
        return mean, logstd

    def sample(self, x):
        # reparameterized sample with the tanh-correction log-prob (standard SAC)
        mean, logstd = self(x)
        std = logstd.exp()
        normal = torch.distributions.Normal(mean, std)
        xt = normal.rsample()
        yt = torch.tanh(xt)
        action = yt * self.act_scale + self.act_bias
        logprob = normal.log_prob(xt) - torch.log(self.act_scale * (1 - yt.pow(2)) + 1e-6)
        logprob = logprob.sum(1, keepdim=True)
        return action, logprob


class QNet(nn.Module):
    """Q(s,a) critic: concat(obs,act) -> 256x256 -> scalar."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        h = F.relu(self.fc1(torch.cat([x, a], 1)))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class TorchReplayBuffer:
    """Preallocated CPU-tensor replay buffer (no numpy<->torch round-trip on sample)."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        # before: nothing; after: fixed (capacity, dim) float32 tensors for obs/next_obs/act + (capacity,1) r/done
        self.cap = capacity
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.act = torch.zeros((capacity, act_dim), dtype=torch.float32)
        self.rew = torch.zeros((capacity, 1), dtype=torch.float32)
        self.done = torch.zeros((capacity, 1), dtype=torch.float32)
        self.pos = 0
        self.full = False

    def add(self, o, no, a, r, d):
        # write one transition at the ring position
        i = self.pos
        self.obs[i] = torch.as_tensor(o, dtype=torch.float32)
        self.next_obs[i] = torch.as_tensor(no, dtype=torch.float32)
        self.act[i] = torch.as_tensor(a, dtype=torch.float32)
        self.rew[i, 0] = float(r)
        self.done[i, 0] = float(d)
        self.pos = (self.pos + 1) % self.cap
        self.full = self.full or self.pos == 0

    def sample(self, n: int):
        # uniform indices over the filled region; returns tensor views (no copy to numpy)
        hi = self.cap if self.full else self.pos
        idx = torch.randint(0, hi, (n,))
        return self.obs[idx], self.act[idx], self.next_obs[idx], self.rew[idx], self.done[idx]


def soft_update_foreach(src_params, tgt_params, tau: float):
    """Bit-exact torch._foreach_ soft target update (same as the main repo's sb3_patches)."""
    with torch.no_grad():
        t = list(tgt_params)
        s = list(src_params)
        torch._foreach_mul_(t, 1.0 - tau)
        torch._foreach_add_(t, s, alpha=tau)


def build_env(args):
    """Build the env + (obs_dim, act_dim, bounds) + optional intrinsic model. Reuses train.py for PointMaze."""
    if args.env == "pointmaze":
        # reuse the project's EXACT env stack + intrinsic model so the comparison vs SB3 is apples-to-apples
        import random
        import gymnasium as gym
        import gymnasium_robotics
        import train as T
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        gym.register_envs(gymnasium_robotics)
        cfg = T.Config(
            env_name="PointMaze_Large-v3", a_seed=args.seed, total_timesteps=args.steps,
            device="cpu", beta=args.beta, algorithm=args.algorithm, discount_factor=0.999,
            env_max_episode=400, goal_position="top_right", rnd_obs_norm=True, rnd_distance="mse",
            rnd_output_dim=128, n_predictors=1, apply_termination_wrapper=False, use_wandb=False,
        )
        if not T.REGISTRY[cfg.algorithm].builds_model:
            cfg.beta = 0
        base_env = T.make_base_env(cfg)
        goal_cell, start_cell = T.select_cells(cfg, base_env, args.seed)
        env, pos_w, posvel_w = T.build_env_stack(cfg, base_env, start_cell, goal_cell)  # flat, extrinsic reward
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        intrinsic = None
        if cfg.beta > 0:
            ctx = T.EnvContext(obs_shape=(obs_dim,), action_dim=act_dim,
                               observation_space=env.observation_space, action_space=env.action_space,
                               position_wrapper=pos_w, position_velocity_wrapper=posvel_w)
            intrinsic = T.build_intrinsic_model(cfg.algorithm, cfg, ctx)
        return env, obs_dim, act_dim, env.action_space.low, env.action_space.high, intrinsic, cfg.beta
    else:
        # standard gym env (e.g. Pendulum-v1) for SAC-core validation -- no intrinsic
        import gymnasium as gym
        env = gym.make(args.env)
        env.reset(seed=args.seed)
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        return env, obs_dim, act_dim, env.action_space.low, env.action_space.high, None, 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="pointmaze")
    p.add_argument("--algorithm", default="rnd_state")
    p.add_argument("--beta", type=float, default=100.0)
    p.add_argument("--steps", type=int, default=8000, help="timed steps (after warmup)")
    p.add_argument("--warmup", type=int, default=1000, help="unprofiled steps to fill the buffer past learning_starts")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--learning_starts", type=int, default=100)
    p.add_argument("--validate", action="store_true", help="print rolling episodic return (SAC sanity)")
    p.add_argument("--out", default="")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    env, obs_dim, act_dim, a_low, a_high, intrinsic, beta = build_env(args)

    # SAC modules + optimizers (CleanRL hyperparameters)
    actor = Actor(obs_dim, act_dim, a_low, a_high)
    q1, q2 = QNet(obs_dim, act_dim), QNet(obs_dim, act_dim)
    q1t, q2t = QNet(obs_dim, act_dim), QNet(obs_dim, act_dim)
    q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)
    a_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    target_entropy = -float(act_dim)
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr)

    rb = TorchReplayBuffer(200_000, obs_dim, act_dim)
    obs, _ = env.reset(seed=args.seed)
    ep_ret, ep_rets = 0.0, []

    def policy_action(o):
        # sample an action from the current policy (no grad) as a numpy array for env.step
        with torch.no_grad():
            a, _ = actor.sample(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))
        return a.squeeze(0).numpy()

    def update():
        # one SAC gradient update on a sampled batch, with the intrinsic reward folded in (torch-only)
        bobs, bact, bnobs, brew, bdone = rb.sample(args.batch_size)
        if intrinsic is not None and beta > 0:
            samples = {"observations": bobs, "next_observations": bnobs, "actions": bact}
            ir = intrinsic.compute(samples)
            intrinsic.update(samples)
            if not torch.is_tensor(ir):
                ir = torch.as_tensor(ir, dtype=torch.float32)
            brew = brew + beta * ir.reshape(-1, 1).to(torch.float32)
        # critic target: r + gamma*(1-done)*(min Q_t(s',a') - alpha*logp(a'|s'))
        with torch.no_grad():
            na, nlogp = actor.sample(bnobs)
            qt = torch.min(q1t(bnobs, na), q2t(bnobs, na)) - log_alpha.exp() * nlogp
            target = brew + args.gamma * (1.0 - bdone) * qt
        qloss = F.mse_loss(q1(bobs, bact), target) + F.mse_loss(q2(bobs, bact), target)
        q_opt.zero_grad(set_to_none=True); qloss.backward(); q_opt.step()
        # actor + temperature update
        pi, logp = actor.sample(bobs)
        aloss = (log_alpha.exp().detach() * logp - torch.min(q1(bobs, pi), q2(bobs, pi))).mean()
        a_opt.zero_grad(set_to_none=True); aloss.backward(); a_opt.step()
        alpha_loss = (-log_alpha.exp() * (logp.detach() + target_entropy)).mean()
        alpha_opt.zero_grad(set_to_none=True); alpha_loss.backward(); alpha_opt.step()
        # soft target update (foreach)
        soft_update_foreach(q1.parameters(), q1t.parameters(), args.tau)
        soft_update_foreach(q2.parameters(), q2t.parameters(), args.tau)

    def run_steps(n, do_time):
        # roll the env for n steps, storing transitions and updating after learning_starts; returns wall time
        nonlocal obs, ep_ret
        t0 = time.time() if do_time else None
        for step in range(n):
            a = env.action_space.sample() if (not rb.full and rb.pos < args.learning_starts) else policy_action(obs)
            nobs, r, term, trunc, _ = env.step(a)
            rb.add(obs, nobs, a, r, term)
            ep_ret += r
            obs = nobs
            if term or trunc:
                ep_rets.append(ep_ret); ep_ret = 0.0
                obs, _ = env.reset()
            if (rb.full or rb.pos >= args.learning_starts):
                update()
        return (time.time() - t0) if do_time else 0.0

    # warmup (fill buffer + reach the hot update path), then timed segment
    run_steps(args.warmup, do_time=False)
    if args.validate and ep_rets:
        print(f"[validate] mean episodic return over warmup: {np.mean(ep_rets[-20:]):.2f} ({len(ep_rets)} eps)")
        ep_rets.clear()
    wall = run_steps(args.steps, do_time=True)
    fps = round(args.steps / wall, 2)

    out = {
        "trainer": "fast_sac_rnd", "node": os.environ.get("SLURMD_NODENAME", os.uname().nodename),
        "env": args.env, "algorithm": args.algorithm, "beta": beta, "threads": args.threads,
        "steps": args.steps, "wall_s": round(wall, 3), "fps": fps,
    }
    if args.validate and ep_rets:
        out["mean_return_timed"] = round(float(np.mean(ep_rets)), 3)
        out["n_eps_timed"] = len(ep_rets)
        print(f"[validate] mean episodic return over timed segment: {out['mean_return_timed']} ({out['n_eps_timed']} eps)")
    print(f"[fast_sac_rnd] env={args.env} algo={args.algorithm} fps={fps} wall={out['wall_s']}s")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
