#!/usr/bin/env python3
"""
Train run 4 driver: fully-JAX SAC (sbx) on the EXACT run-2 PointMaze task, with the intrinsic recomputed at
SAMPLE time (== run-2's VectorIntrinsicReplayBuffer), so the algorithm is identical to run-2 -- only the SAC
backend (JAX) and the RND backend (JAX, for rnd_state) differ. Logic parity vs run-2:
  - SAC hyperparameters = SB3/sbx defaults + gamma=0.999 (lr 3e-4, buffer 1e6, learning_starts 100,
    batch 256, tau 0.005, train_freq 1, gradient_steps 1, ent_coef auto, policy_delay 1, net 256x256).
  - Reward stored in the buffer = EXTRINSIC; reward = extrinsic + beta*intrinsic is recomputed on the
    SAMPLED batch every gradient step (SACWithIntrinsic.train), and the predictor is trained then -- exactly
    like run-2. The env wrapper only computes the step-time intrinsic for the info (train/intrinsic logging).
  - rnd_state uses the fully-JAX RND (jax_rnd.JaxRND, feature=observations); gt_position_velocity uses the
    project's visit-count (counts updated by the env-stack wrapper), rnd_elliptical the torch elliptical.
Logging = run-4 convention (run-id-and-logging.md): per-eval eval results + every training episode
(episode-level); no final-eval special case. One JSON per run at <local_log_dir>/local/<id>_of_<total>.json.
"""
from __future__ import annotations

import os

# Cap per-process CPU threads BEFORE importing jax/sbx/numpy/torch. In the sweep each worker is pinned to 2
# CPUs (srun cpus-per-task=2) and 8 workers share a node; without this, XLA:CPU sizes its Eigen thread pool to
# the whole node and 8x oversubscribes. setdefault so worker.slurm's exported values win when present.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time

import numpy as np
import gymnasium as gym

PROJ = "/p/rlprojects/RND/07_reconstruction"
sys.path.insert(0, PROJ)
sys.path.insert(0, PROJ + "/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class InfoIntrinsicWrapper(gym.Wrapper):
    """Compute the step-time intrinsic for the info (== run-2's env wrapper) -- for train/intrinsic logging
    ONLY. The buffer stores EXTRINSIC reward; training uses the sample-time intrinsic (SACWithIntrinsic).

    Builds samples {observations=state acted from, next_observations, actions} (== run-2 buffer fields) so any
    intrinsic's compute() works; never updates the model (the predictor is trained at sample time)."""

    def __init__(self, env, compute_fn, beta):
        super().__init__(env)
        self.compute_fn = compute_fn
        self.beta = beta
        self.last_obs = None

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.last_obs = np.asarray(obs, np.float32)
        return obs, info

    def step(self, action):
        next_obs, r, term, trunc, info = self.env.step(action)
        samples = {"observations": self.last_obs[None], "next_observations": np.asarray(next_obs, np.float32)[None],
                   "actions": np.asarray(action, np.float32)[None]}
        intr = float(np.asarray(self.compute_fn(samples)).reshape(-1)[0])
        info = dict(info)
        info["extrinsic_reward"] = float(r)   # buffer stores extrinsic; reward returned is extrinsic
        info["intrinsic_reward"] = intr
        self.last_obs = np.asarray(next_obs, np.float32)
        return next_obs, r, term, trunc, info


def make_sac_with_intrinsic():
    """Build the SACWithIntrinsic subclass (sbx.SAC that recomputes reward on the sampled batch each step)."""
    from sbx import SAC as _SBXSAC
    from sbx.common.type_aliases import ReplayBufferSamplesNp

    class SACWithIntrinsic(_SBXSAC):
        """sbx SAC, but reward = extrinsic + beta*intrinsic is recomputed on the SAMPLED batch each gradient
        step and the predictor trained then (identical to run-2's VectorIntrinsicReplayBuffer.sample)."""

        def configure_intrinsic(self, compute_fn, update_fn, beta):
            """Attach the intrinsic compute/update + beta used in the sample-time reward recompute."""
            self._compute_fn, self._update_fn, self._beta = compute_fn, update_fn, beta

        def train(self, gradient_steps: int, batch_size: int) -> None:
            """sbx.SAC.train with the sample-time intrinsic inserted at the numpy hook (before the jit _train)."""
            assert self.replay_buffer is not None
            # sample the whole (batch*grad_steps) block at once, exactly as sbx does
            data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)
            self._update_learning_rate(self.policy.actor_state.opt_state,
                                       learning_rate=self.lr_schedule(self._current_progress_remaining),
                                       name="learning_rate_actor")
            self._update_learning_rate(self.policy.qf_state.opt_state,
                                       learning_rate=self.initial_qf_learning_rate or self.lr_schedule(self._current_progress_remaining),
                                       name="learning_rate_critic")
            self._maybe_reset_params()
            # to numpy (PointMaze obs is a Box, not a dict)
            obs, next_obs, actions = data.observations.numpy(), data.next_observations.numpy(), data.actions.numpy()
            discounts = (np.full((batch_size * gradient_steps,), self.gamma, np.float32)
                         if data.discounts is None else data.discounts.numpy().flatten())
            # ===== SAMPLE-TIME INTRINSIC (== run-2 VectorIntrinsicReplayBuffer.sample) =====
            # before: data.rewards = extrinsic (stored by the env). after: reward = extrinsic + beta*intrinsic,
            # with the intrinsic recomputed on THIS sampled batch and the predictor trained on it.
            samples = {"observations": obs, "next_observations": next_obs, "actions": actions}
            intrinsic = np.asarray(self._compute_fn(samples)).reshape(-1).astype(np.float32)
            self._update_fn(samples)
            rewards = data.rewards.numpy().flatten() + self._beta * intrinsic
            # ==============================================================================
            data = ReplayBufferSamplesNp(obs, actions, next_obs, data.dones.numpy().flatten(), rewards, discounts)
            (self.policy.qf_state, self.policy.actor_state, self.ent_coef_state, self.key, _) = self._train(
                self.tau, self.target_entropy, gradient_steps, data, self.policy_delay,
                (self._n_updates + 1) % self.policy_delay, self.policy.qf_state, self.policy.actor_state,
                self.ent_coef_state, self.key)
            self._n_updates += gradient_steps

    return SACWithIntrinsic


def build_intrinsic_fns(algorithm, cfg, ctx, obs_dim, seed):
    """(intrinsic_model, compute_fn, update_fn) per algorithm: JAX RND for rnd_state, torch for gt/elliptical."""
    import train as T
    if algorithm == "rnd_state":
        from jax_rnd import JaxRND
        m = JaxRND(input_dim=obs_dim, seed=seed)
        m.warmup_obs_rms(ctx.observation_space)  # 200-sample obs_rms warm-up == run-2 _warmup_obs_rms (parity)
        return m, m.compute, m.update
    m = T.build_intrinsic_model(algorithm, cfg, ctx)
    # gt visit-count: counts are updated by the env-stack wrapper -> the model's update is a no-op
    return (m, m.compute, (lambda s: None)) if algorithm == "gt_position_velocity" else (m, m.compute, m.update)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--a_seed", type=int, default=0)
    p.add_argument("--total_timesteps", type=int, default=500000)
    p.add_argument("--eval_freq", type=int, default=50000)
    p.add_argument("--n_eval_episodes", type=int, default=100)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--run_id", type=int, default=-1)
    p.add_argument("--run_total", type=int, default=0)
    p.add_argument("--local_log_dir", default="")
    # distance-to-GT (distance_to_gt/*) is OFF by default: it grids the whole maze through the intrinsic model
    # at every eval (overhead), and run-4's deliverable is the reward comparison. Pass --log_distance=1 to enable
    # (then logged only for ALGORITHMS_NO_ACTION). Run-2's data already carries the distance field.
    p.add_argument("--log_distance", type=int, default=0)
    args = p.parse_args()

    import random
    import torch
    import gymnasium_robotics
    import train as T
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rnd_exploration.callbacks.wandb_eval_logging import WandbEvalLoggingCallback
    from rnd_exploration.callbacks.train_episode_stats import TrainEpisodeStatsCallback
    from rnd_exploration.callbacks.distance_logging import DistanceLoggingCallback
    from rnd_exploration.methods import ALGORITHMS_NO_ACTION

    torch.set_num_threads(args.threads)
    seed = args.a_seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    gym.register_envs(gymnasium_robotics)
    t0 = time.time()

    cfg = T.Config(env_name="PointMaze_Large-v3", a_seed=seed, total_timesteps=args.total_timesteps,
                   eval_freq=args.eval_freq, n_eval_episodes=args.n_eval_episodes, device="cpu",
                   beta=args.beta, algorithm=args.algorithm, discount_factor=0.999, env_max_episode=400,
                   goal_position="top_right", rnd_obs_norm=True, rnd_distance="mse", rnd_output_dim=128,
                   n_predictors=1, apply_termination_wrapper=False, use_wandb=False)
    base = T.make_base_env(cfg)
    goal_cell, start_cell = T.select_cells(cfg, base, seed)
    train_flat, train_pos, train_posvel = T.build_env_stack(cfg, base, start_cell, goal_cell)
    obs_dim = int(np.prod(train_flat.observation_space.shape))
    act_dim = int(np.prod(train_flat.action_space.shape))
    ctx = T.EnvContext(obs_shape=(obs_dim,), action_dim=act_dim, observation_space=train_flat.observation_space,
                       action_space=train_flat.action_space, position_wrapper=train_pos,
                       position_velocity_wrapper=train_posvel)
    intrinsic, compute_fn, update_fn = build_intrinsic_fns(args.algorithm, cfg, ctx, obs_dim, seed)

    eval_base = T.make_base_env(cfg)
    eval_flat, _, _ = T.build_env_stack(cfg, eval_base, start_cell, goal_cell,
                                        count_map_refs=(train_pos.visit_counts, train_posvel.visit_counts),
                                        update_counts=False)
    train_env = InfoIntrinsicWrapper(train_flat, compute_fn, args.beta)
    eval_env = DummyVecEnv([lambda: InfoIntrinsicWrapper(eval_flat, compute_fn, args.beta)])

    # SAC hyperparameters == run-2 (SB3/sbx defaults + gamma=0.999): learning_starts 100, buffer 1e6
    SAC = make_sac_with_intrinsic()
    model = SAC("MlpPolicy", train_env, learning_starts=100, batch_size=256, train_freq=1, gradient_steps=1,
                buffer_size=1_000_000, learning_rate=3e-4, gamma=cfg.discount_factor, tau=0.005, verbose=0, seed=seed)
    model.configure_intrinsic(compute_fn, update_fn, args.beta)

    eval_cb = WandbEvalLoggingCallback(eval_env=eval_env, eval_freq=args.eval_freq, n_eval_episodes=args.n_eval_episodes,
                                       use_wandb=False, visit_count_env=train_pos, goal_cell=goal_cell, start_cell=start_cell,
                                       run_name=f"run4_{args.algorithm}_s{seed}", beta=args.beta,
                                       total_timesteps=args.total_timesteps, log_to_wandb=False, verbose=0)
    train_cb = TrainEpisodeStatsCallback(train_env=model.get_env(), eval_freq=args.eval_freq,
                                         n_eval_episodes=args.n_eval_episodes, use_wandb=False, beta=args.beta,
                                         log_to_wandb=False, verbose=0)
    callbacks = [eval_cb, train_cb]
    # distance-to-GT (distance_to_gt/*) is OFF by default (--log_distance=0). When enabled, it is logged only for
    # state-only bonus fields (ALGORITHMS_NO_ACTION): gt_position_velocity + rnd_state, NOT rnd_elliptical.
    # JaxRND/VisitCount expose compute(samples), so the metric grids the maze exactly as run-2
    # (compute_intrinsic_vector_distance -> intrinsic_reward_model.compute over the cell grid).
    dist_cb = None
    if args.log_distance and args.algorithm in ALGORITHMS_NO_ACTION:
        dist_cb = DistanceLoggingCallback(algorithm=args.algorithm, intrinsic_reward_model=intrinsic,
                                          visit_count_env_position_velocity=train_posvel, eval_freq=args.eval_freq,
                                          use_wandb=False, log_to_wandb=False, verbose=0)
        callbacks.append(dist_cb)
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    record = {
        "run_id": args.run_id, "run_total": args.run_total, "trainer": "run4_jax_sbx_sampletime",
        "algorithm": args.algorithm, "beta": args.beta, "a_seed": seed,
        "total_timesteps": args.total_timesteps, "eval_freq": args.eval_freq,
        "runtime_seconds": round(time.time() - t0, 2),
        "eval_history": eval_cb.history, "train_history": train_cb.history,
        "train_episode_history": train_cb.episode_history,
        "distance_history": dist_cb.history if dist_cb is not None else [],
    }
    if args.local_log_dir:
        out_dir = os.path.join(args.local_log_dir, "local")
        os.makedirs(out_dir, exist_ok=True)
        name = (f"{args.run_id:0{len(str(args.run_total))}d}_of_{args.run_total}.json"
                if args.run_total > 0 else f"run4_{args.algorithm}_s{seed}.json")
        json.dump(record, open(os.path.join(out_dir, name), "w"))
    fe = eval_cb.history[-1]["eval/mean_extrinsic_reward"] if eval_cb.history else None
    print(f"[run4_train] {args.algorithm} seed={seed} {record['runtime_seconds']}s | final eval extrinsic={fe} "
          f"| {len(train_cb.episode_history)} train episodes")


if __name__ == "__main__":
    main()
