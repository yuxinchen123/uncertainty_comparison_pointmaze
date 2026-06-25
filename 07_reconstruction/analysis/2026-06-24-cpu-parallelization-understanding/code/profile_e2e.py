"""
End-to-end CPU profiler for the real training loop.

Faithful snapshot of the env+model construction in
07_reconstruction/04_many_exploration_method.py (commit 3b212d7), with the
reusable profiling module (cpu_profiler.py) attached. It reuses the project's
real classes (SAC config, RND / VisitCount / EllipticalBonus, the PointMaze
wrappers, VectorIntrinsicReplayBuffer) — only the ~40 lines of wiring are
replicated so we can insert the StepRateProfiler callback and a CPU sampler.

It measures STEADY-STATE training throughput (intermediate eval is disabled, so
the number reflects the train step itself), under a fixed thread cap.

    OMP_NUM_THREADS=N MKL_NUM_THREADS=N OPENBLAS_NUM_THREADS=N RND_PROFILE=1 \
        python profile_e2e.py --algorithm rnd_linear_next_state --device cpu \
            --n_threads N --total_timesteps 12000 --warmup_steps 3000 \
            --out logs/e2e_rnd_cpu_N.json

Output: one JSON object with steady_steps_per_sec, steady_effective_cores,
CPU% distribution, and how many OS threads actually accumulated CPU time.
"""
import argparse
import json
import os
import sys
import time

# Make the project root (07_reconstruction) importable, and this dir for cpu_profiler.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # 07_reconstruction
sys.path.insert(0, _HERE)
sys.path.insert(0, _PROJ)

from cpu_profiler import (
    apply_thread_limit, CpuSampler, StepRateProfiler, per_thread_cpu_active,
)


def _algorithm_to_config(algorithm):
    if algorithm == "no_exploration":
        return ("none", None, False)
    if algorithm in ("gt_position", "gt_position_velocity"):
        return ("visit_count", None, False)
    if algorithm == "rnd_next_state":
        return ("rnd", "rnd_next_state", False)
    if algorithm == "rnd_next_state_position_only":
        return ("rnd", "rnd_next_state_position_only", False)
    if algorithm == "rnd_state":
        return ("rnd", "rnd_state", False)
    if algorithm == "rnd_state_action":
        return ("rnd", "rnd_state_action", False)
    if algorithm == "rnd_state_action_next_state":
        return ("rnd", "rnd_state_action_next_state", False)
    if algorithm == "rnd_linear_next_state":
        return ("rnd", "rnd_next_state", True)
    if algorithm == "rnd_elliptical":
        return ("elliptical", None, False)
    raise ValueError(f"unknown algorithm {algorithm!r}")


def build_and_profile(args):
    import numpy as np
    import random
    import torch
    import gymnasium as gym
    import gymnasium_robotics
    from gymnasium.wrappers import FlattenObservation
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from env_wrapper.point_maze_utils import (
        select_fixed_goal_bottom_left, select_fixed_goal_top_right,
    )
    from env_wrapper.point_maze_wrappers import (
        FixedGoalWrapper, FixedStartWrapper, RemoveGoalWrapper,
        PositionVisitCountWrapper, PositionVelocityVisitCountWrapper,
        ComputeIntrinsicRewardWrapper,
    )
    from intrinsic.vector_intrinsic_replay_buffer import VectorIntrinsicReplayBuffer
    from intrinsic.intrinsic_method import RND, VisitCount, EllipticalBonus

    # n_threads <= 0 means "auto": do NOT cap; use whatever torch defaults to from
    # the process CPU affinity (this measures the real default under srun binding).
    if args.n_threads and args.n_threads > 0:
        apply_thread_limit(args.n_threads)
    affinity_ncpus = len(os.sched_getaffinity(0))
    torch_threads_effective = int(torch.get_num_threads())

    seed = args.a_seed
    method, rnd_feature, linear_rnd = _algorithm_to_config(args.algorithm)
    beta = 0.0 if args.algorithm == "no_exploration" else args.beta

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    gym.register_envs(gymnasium_robotics)
    base_env = gym.make(args.env_name, continuing_task=True, reset_target=False,
                        max_episode_steps=args.env_max_episode)
    fixed_goal_cell = select_fixed_goal_bottom_left(base_env)
    fixed_start_cell = select_fixed_goal_top_right(base_env)

    start_env = FixedStartWrapper(base_env, fixed_start_cell)
    goal_env = FixedGoalWrapper(start_env, fixed_goal_cell)
    no_goal_env = RemoveGoalWrapper(goal_env)
    vc_pos = PositionVisitCountWrapper(no_goal_env)
    vc_posvel = PositionVelocityVisitCountWrapper(vc_pos)
    flat_env = FlattenObservation(vc_posvel)

    full_obs_dim = int(np.prod(flat_env.observation_space.shape))
    obs_shape = (full_obs_dim,)
    replay_buffer_class = VectorIntrinsicReplayBuffer if beta > 0 else None
    intrinsic_model = None
    if beta > 0:
        if method == "rnd":
            action_dim = int(np.prod(flat_env.action_space.shape))
            intrinsic_model = RND(
                obs_shape=obs_shape, output_dim=args.rnd_output_dim, lr=0.001,
                batch_size=256, device=device, use_obs_norm=True, distance="mse",
                n_predictors=1, beta_std=0.0, linear_rnd=linear_rnd,
                feature=rnd_feature, action_dim=action_dim,
            )
            if getattr(intrinsic_model, "obs_rms", None) is not None:
                obs_buf, next_buf, act_buf = [], [], []
                for _ in range(200):
                    obs_buf.append(np.asarray(flat_env.observation_space.sample(), dtype=np.float32))
                    next_buf.append(np.asarray(flat_env.observation_space.sample(), dtype=np.float32))
                    act_buf.append(np.asarray(flat_env.action_space.sample(), dtype=np.float32))
                samples = {"observations": np.stack(obs_buf), "next_observations": np.stack(next_buf),
                           "actions": np.stack(act_buf)}
                x = intrinsic_model._get_feature_tensor(samples)
                intrinsic_model.obs_rms.update(x.detach().cpu().numpy())
        elif method == "visit_count":
            intrinsic_model = VisitCount(
                vc_posvel if args.algorithm == "gt_position_velocity" else vc_pos)
        elif method == "elliptical":
            action_dim = int(np.prod(flat_env.action_space.shape))
            intrinsic_model = EllipticalBonus(obs_shape=obs_shape, action_dim=action_dim,
                                              feature_dim=128, device=device, regularization=1e-6)
        replay_buffer_kwargs = {"beta": beta, "intrinsic_reward_model": intrinsic_model}
    else:
        replay_buffer_kwargs = None

    intrinsic_env = ComputeIntrinsicRewardWrapper(flat_env, beta=beta, intrinsic_reward_model=intrinsic_model)
    monitored_env = Monitor(intrinsic_env, filename=None)
    env = DummyVecEnv([lambda: monitored_env])
    env.seed(seed); env.reset()

    setup_t0 = time.perf_counter()
    model = SAC("MlpPolicy", env, verbose=0, seed=seed, device=device,
                gamma=args.discount_factor, replay_buffer_class=replay_buffer_class,
                replay_buffer_kwargs=replay_buffer_kwargs)

    prof = StepRateProfiler(every=50, warmup_steps=args.warmup_steps)
    sampler = CpuSampler(interval=0.1).start()
    learn_t0 = time.perf_counter()
    model.learn(total_timesteps=args.total_timesteps, callback=[prof])
    learn_wall = time.perf_counter() - learn_t0
    sampler.stop()

    summ = prof.summary()
    # steady-state wall window bounds for the CPU sampler distribution stats
    win = {}
    if prof.records:
        w = [r for r in prof.records if r[1] >= args.warmup_steps] or prof.records
        win = sampler.window_summary(w[0][0], w[-1][0])

    out = {
        "algorithm": args.algorithm, "device": device, "n_threads": args.n_threads,
        "intrinsic_method": method, "beta": beta,
        "total_timesteps": args.total_timesteps,
        "torch_default_threads": int(__import__("torch").get_num_threads()),
        "torch_threads_effective": torch_threads_effective,
        "affinity_ncpus": affinity_ncpus,
        "slurm_procid": os.environ.get("SLURM_PROCID"),
        "slurm_ntasks": os.environ.get("SLURM_NTASKS"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "peak_rss_mb": (win or {}).get("peak_rss_mb"),
        "full_learn_wall_s": round(learn_wall, 3),
        **summ,
        "cpu_window": win,
        "thread_activity": per_thread_cpu_active(),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", type=str, default="rnd_linear_next_state")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--n_threads", type=int, default=8)
    ap.add_argument("--total_timesteps", type=int, default=12000)
    ap.add_argument("--warmup_steps", type=int, default=3000)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--a_seed", type=int, default=1)
    ap.add_argument("--env_name", type=str, default="PointMaze_Large-v3")
    ap.add_argument("--env_max_episode", type=int, default=400)
    ap.add_argument("--discount_factor", type=float, default=0.99)
    ap.add_argument("--rnd_output_dim", type=int, default=128)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    out = build_and_profile(args)
    print(json.dumps(out, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
