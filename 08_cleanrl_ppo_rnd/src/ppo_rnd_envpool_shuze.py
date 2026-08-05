"""PPO + Random Network Distillation on Atari, from CleanRL, with the envpool bug fixed.

Starts from `cleanrl/cleanrl/ppo_rnd_envpool.py` at upstream commit fe8d8a0 and changes four things.
Every change is behind a flag so the profiling can measure it, and every flag's default is the
setting used for the real 30-seed run.

1. **The envpool auto-reset bug is fixed** (`--fix_envpool_autoreset`, default on). CleanRL's own
   documentation records it: envpool spends one extra `step()` call auto-resetting after an episode
   ends, and the action on that call is discarded. The row it writes into the rollout buffer is a
   transition that never happened. See `docs/ENVPOOL_AUTORESET_FIX.md` next to this file.
2. **Weights and Biases and tensorboard are gone.** The run writes one JSON record in the shape
   train run 8.1.2 uses, carrying both the extrinsic and the intrinsic reward. See `run_record.py`.
3. **The run checkpoints and resumes** (`--checkpoint_every_seconds`, default 8 hours), because the
   GPU partitions cap a job at 4 days and a full run is far longer. See `checkpointing.py`.
4. **Throughput options**, each behind its own `--opt_*` flag, split into two groups: options that
   leave the computation unchanged (on by default) and options that change the numbers (off by
   default). The split is the point — a faithful run may take the first group and must not take the
   second without saying so.

Nothing else about the algorithm is changed: the hyperparameters, the network shapes, the loss and
the environment options are CleanRL's.
"""

import os
import random
import time
from collections import deque
from dataclasses import dataclass

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from gym.wrappers.normalize import RunningMeanStd
from torch.distributions.categorical import Categorical

from checkpointing import load_checkpoint, save_checkpoint
from run_record import RunRecord


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Algorithm specific arguments — every one of these is CleanRL's value, unchanged.
    env_id: str = "MontezumaRevenge-v5"
    """the id of the environment"""
    total_timesteps: int = 2000000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # RND arguments — CleanRL's values, unchanged.
    update_proportion: float = 0.25
    """proportion of exp used for predictor update"""
    int_coef: float = 1.0
    """coefficient of extrinsic reward"""
    ext_coef: float = 2.0
    """coefficient of intrinsic reward"""
    int_gamma: float = 0.99
    """Intrinsic reward discount rate"""
    num_iterations_obs_norm_init: int = 50
    """number of iterations to initialize the observations normalization parameters"""

    # The bug fix.
    fix_envpool_autoreset: bool = True
    """drop the rows envpool burns auto-resetting, and stop them contaminating the intrinsic advantage"""

    # Run identity and output. These replace wandb: the run is identified by a run id inside a sweep.
    run_id: int = 0
    """this run's index inside the sweep"""
    run_total: int = 1
    """how many runs the sweep holds"""
    output_dir: str = "."
    """directory the JSON record and the checkpoint are written to"""
    log_every_updates: int = 25
    """how often, in policy updates, to append a row to train_history and eval_history"""
    episode_history_cap: int = 50000
    """keep every episode up to this many, then keep every episode_history_stride-th"""
    episode_history_stride: int = 100
    """the stride applied past episode_history_cap"""

    # Checkpointing.
    checkpoint_every_seconds: float = 28800.0
    """how often to write a checkpoint; 28800 = 8 hours"""
    resume: bool = True
    """load the checkpoint next to the record if one is there"""

    # Throughput options that leave the computation unchanged. On by default.
    opt_env_threads: int = 0
    """envpool worker threads; 0 means read SLURM_CPUS_PER_TASK, which envpool cannot see itself"""
    opt_fused_policy_pass: bool = True
    """run the policy trunk once per rollout step instead of twice (bit-identical)"""
    opt_rnd_no_grad: bool = True
    """compute the rollout RND bonus under no_grad (bit-identical)"""
    opt_uint8_obs: bool = True
    """hold the observation buffer as uint8 instead of float32 (bit-identical through the networks)"""
    opt_gpu_obs_rms: bool = True
    """compute the observation-normalisation batch moments on the GPU"""
    opt_gpu_norm_stats: bool = True
    """keep the observation-normalisation statistics on the GPU in float32"""
    opt_no_sync_update: bool = True
    """remove the forced host-device synchronisations from the update loop (bit-identical)"""
    opt_fast_obs_norm_init: bool = True
    """fill the normalisation-init buffer directly instead of through Python lists (bit-identical)"""
    opt_cudnn_benchmark: bool = True
    """let cuDNN autotune its convolution algorithms"""
    opt_fixed_minibatch_shape: bool = True
    """keep the minibatch shape constant across updates by dropping a fixed row allowance"""
    minibatch_drop_allowance: int = 1024
    """rows held back so the minibatch shape is constant; must exceed the auto-reset rows dropped"""

    # Throughput options that change the numbers. Off by default.
    opt_amp_fp16: bool = False
    """run the update under float16 autocast with a gradient scaler"""
    opt_channels_last: bool = False
    """hold the convolution weights in channels-last layout; only pays off together with opt_amp_fp16"""
    opt_matmul_tf32: bool = False
    """allow TF32 for matrix multiplication (Ampere and newer)"""
    opt_torch_compile: bool = False
    """compile the three networks with torch.compile(mode='default')"""

    # Profiling.
    profile_iterations: int = 0
    """if greater than 0, stop after this many policy updates and report throughput"""
    profile_tag: str = ""
    """a label written into the record, naming the option set under test"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class RecordEpisodeStatistics(gym.Wrapper):
    """Track per-environment episode return and length across envpool's auto-resets."""

    def __init__(self, env, fix_autoreset=True):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None
        # envpool burns one step call auto-resetting after an episode ends. That call must not be
        # counted as a step of the new episode, or reported episode lengths drift upward by one per
        # life lost. `burned` marks the call that follows a done.
        self.fix_autoreset = fix_autoreset
        self.burned = None

    def reset(self, **kwargs):
        """Reset the environments and zero the per-environment episode counters."""
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.burned = np.zeros(self.num_envs, dtype=bool)
        return observations

    def step(self, action):
        """Step the environments and update the episode return and length counters."""
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        # A burned call carries reward 0, so the return is unaffected either way; the length is not.
        # before: every call adds 1, so a 120-step episode with 4 life losses reports 124
        # after:  the 4 burned calls add 0, so it reports 120
        if self.fix_autoreset:
            self.episode_lengths += (~self.burned).astype(np.int32)
        else:
            self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        self.burned = dones.astype(bool)
        return observations, rewards, dones, infos


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal-initialise a layer's weight and set its bias to a constant, as CleanRL does."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """The Nature-CNN policy with two value heads, one extrinsic and one intrinsic."""

    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action_and_value(self, x, action=None):
        """Sample an action (or score a given one) and return both value estimates."""
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_ext(features + hidden),
            self.critic_int(features + hidden),
        )

    def get_value(self, x):
        """Return both value estimates without touching the actor."""
        hidden = self.network(x / 255.0)
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)

    def get_action_and_value_rollout(self, x):
        """Sample an action and both values in ONE trunk pass, for the rollout.

        CleanRL calls get_value and then get_action_and_value on the same input, running the CNN
        trunk twice. This runs it once. `get_value` draws no random numbers and the entropy is not
        used in the rollout, so the returned values are bit-identical to the two-call form.
        """
        hidden = self.network(x / 255.0)
        features = self.extra_layer(hidden)
        value_ext, value_int = self.critic_ext(features + hidden), self.critic_int(features + hidden)
        probs = Categorical(logits=self.actor(hidden))
        action = probs.sample()
        return action, probs.log_prob(action), value_ext, value_int


class RNDModel(nn.Module):
    """The Random Network Distillation pair: a trained predictor and a frozen random target."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        feature_output = 7 * 7 * 64

        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )

        self.target = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
        )

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        """Return the predictor's and the target's features for the same input."""
        return self.predictor(next_obs), self.target(next_obs)


class RewardForwardFilter:
    """Accumulate a discounted sum of intrinsic rewards, never reset at episode boundaries."""

    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        """Advance the accumulator by one step and return it."""
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def resolve_env_threads(requested):
    """Pick envpool's worker-thread count, since envpool cannot see the job's cpu allocation.

    envpool computes its default as min(batch_size, std::thread::hardware_concurrency()), and
    hardware_concurrency() reports the machine's core count, not the cgroup's. On a 224-core node
    inside a 16-cpu allocation that default becomes 128 threads fighting over 16 cores.

    The order below matters when several runs share one job. A packed worker job holds the cores of
    ALL its slots, so SLURM_CPUS_PER_TASK is the job's total, not this run's share — five runs in a
    40-core job would each start 40 envpool threads on 8 cores' worth of cpu. The sweep's worker
    manager exports GPU_SWEEP_CPUS_PER_RUN with this run's actual share, so that is read first.
    """
    if requested > 0:
        return requested
    per_run = os.environ.get("GPU_SWEEP_CPUS_PER_RUN")
    if per_run:
        return int(per_run)
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def main():
    """Train PPO + RND on one Atari environment, writing one JSON record and one checkpoint."""
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Seeding, exactly as CleanRL does it.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    if args.opt_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if args.opt_matmul_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup. Every option here is CleanRL's; only num_threads is added, because
    # envpool's own default reads the machine's core count rather than the job's allocation.
    env_threads = resolve_env_threads(args.opt_env_threads)
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
        repeat_action_probability=0.25,
        num_threads=env_threads,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs, fix_autoreset=args.fix_envpool_autoreset)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    rnd_model = RNDModel(4, envs.single_action_space.n).to(device)
    if args.opt_channels_last:
        agent = agent.to(memory_format=torch.channels_last)
        rnd_model = rnd_model.to(memory_format=torch.channels_last)
    combined_parameters = list(agent.parameters()) + list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(combined_parameters, lr=args.learning_rate, eps=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=args.opt_amp_fp16)

    if args.opt_torch_compile:
        # Compile the three networks only. The training loop itself is full of graph breaks —
        # the envpool call, the numpy round trips, the per-episode Python branch — so compiling it
        # buys nothing and recompiles constantly.
        agent.network = torch.compile(agent.network)
        rnd_model.predictor = torch.compile(rnd_model.predictor)
        rnd_model.target = torch.compile(rnd_model.target)

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    discounted_reward = RewardForwardFilter(args.int_gamma)

    # Storage. The observation buffer dominates: float32 it is 1.85 GB, uint8 it is 0.46 GB, and the
    # networks divide by 255.0 either way so the arithmetic is unchanged.
    obs_dtype = torch.uint8 if args.opt_uint8_obs else torch.float32
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape,
                      dtype=obs_dtype).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # The run's durable record and its checkpoint sit side by side in the output directory.
    os.makedirs(args.output_dir, exist_ok=True)
    record_path = os.path.join(args.output_dir, f"{args.run_id}_of_{args.run_total}.json")
    checkpoint_path = os.path.join(args.output_dir, f"{args.run_id}_of_{args.run_total}.checkpoint.pt")
    record = RunRecord(
        record_path,
        config={
            "run_id": args.run_id,
            "run_total": args.run_total,
            "algorithm": "ppo_rnd",
            "implementation": "cleanrl ppo_rnd_envpool.py @ fe8d8a0, envpool auto-reset fixed",
            "env_id": args.env_id,
            "a_seed": args.seed,
            "profile_tag": args.profile_tag,
            "device": device.type,
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A",
            "env_threads": env_threads,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "hostname": os.environ.get("SLURMD_NODENAME", ""),
            **{k: v for k, v in vars(args).items() if k not in ("run_id", "run_total", "seed", "env_id")},
        },
        episode_history_cap=args.episode_history_cap,
        episode_stride=args.episode_history_stride,
    )

    global_step, start_update = 0, 1
    resumed = False
    if args.resume:
        record.restore_from_disk()
        state = load_checkpoint(
            checkpoint_path, agent=agent, rnd_model=rnd_model, optimizer=optimizer,
            obs_rms=obs_rms, reward_rms=reward_rms, discounted_reward=discounted_reward, device=device,
        )
        if state is not None:
            global_step, start_update = state["global_step"], state["update"] + 1
            avg_returns.extend(state["avg_returns"])
            resumed = True
            print(f"[resume] continuing at update {start_update} of {args.num_iterations}, "
                  f"global_step {global_step}", flush=True)

    next_obs_np = envs.reset()
    next_obs = torch.as_tensor(next_obs_np, device=device) if args.opt_uint8_obs \
        else torch.Tensor(next_obs_np).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.num_iterations

    # The observation normaliser is primed by a random agent. On a resume it is restored from the
    # checkpoint, so priming it again would both waste time and double-count the statistics.
    init_start = time.time()
    if not resumed:
        print("Start to initialize observation normalization parameter.....", flush=True)
        if args.opt_fast_obs_norm_init:
            # before: 6,400 calls to .tolist() building ~115M python objects, then np.stack on a
            #         list of 16,384 nested lists, 50 times over — about nine minutes of pure python
            # after:  a preallocated uint8 buffer filled by slice assignment — about one second
            buf = np.empty((args.num_steps * args.num_envs, 1, 84, 84), dtype=np.uint8)
            fill = 0
            for _ in range(args.num_steps * args.num_iterations_obs_norm_init):
                acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
                s, _, _, _ = envs.step(acs)
                buf[fill:fill + args.num_envs] = s[:, 3, :, :].reshape(-1, 1, 84, 84)
                fill += args.num_envs
                if fill == buf.shape[0]:
                    obs_rms.update(buf)
                    fill = 0
        else:
            next_ob = []
            for _ in range(args.num_steps * args.num_iterations_obs_norm_init):
                acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
                s, _, _, _ = envs.step(acs)
                next_ob += s[:, 3, :, :].reshape([-1, 1, 84, 84]).tolist()
                if len(next_ob) % (args.num_steps * args.num_envs) == 0:
                    obs_rms.update(np.stack(next_ob))
                    next_ob = []
        print("End to initialize...", flush=True)

    obs_norm_init_seconds = time.time() - init_start if not resumed else 0.0
    start_time = time.time()
    steps_at_start = global_step
    last_checkpoint_time = time.time()
    dropped_last_update = 0

    for update in range(start_update, num_updates + 1):
        iteration_start = time.time()
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # Refresh the device-resident copy of the observation-normalisation statistics once per
        # iteration, instead of uploading the float64 arrays on every one of the 128 rollout steps.
        if args.opt_gpu_norm_stats:
            obs_mean_g = torch.as_tensor(obs_rms.mean, dtype=torch.float32, device=device)
            obs_std_g = torch.sqrt(torch.as_tensor(obs_rms.var, dtype=torch.float32, device=device))

        rollout_start = time.time()
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                if args.opt_fused_policy_pass:
                    action, logprob, value_ext, value_int = agent.get_action_and_value_rollout(obs[step])
                else:
                    value_ext, value_int = agent.get_value(obs[step])
                    action, logprob, _, _, _ = agent.get_action_and_value(obs[step])
                ext_values[step], int_values[step] = value_ext.flatten(), value_int.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
            if args.opt_uint8_obs:
                next_obs = torch.as_tensor(next_obs_np, device=device)
            else:
                next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

            # The RND bonus is the squared error between the frozen target and the predictor on the
            # newest frame of the next observation, whitened by the running statistics and clipped.
            rnd_frame = next_obs[:, 3, :, :].reshape(args.num_envs, 1, 84, 84).float()
            if args.opt_gpu_norm_stats:
                rnd_next_obs = ((rnd_frame - obs_mean_g) / obs_std_g).clip(-5, 5)
            else:
                rnd_next_obs = ((rnd_frame - torch.from_numpy(obs_rms.mean).to(device))
                                / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))).clip(-5, 5).float()
            if args.opt_rnd_no_grad:
                # The predictor has trainable parameters, so without this the rollout builds and
                # throws away a full autograd graph 128 times per iteration.
                with torch.no_grad():
                    target_next_feature = rnd_model.target(rnd_next_obs)
                    predict_next_feature = rnd_model.predictor(rnd_next_obs)
                    curiosity_rewards[step] = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
            else:
                target_next_feature = rnd_model.target(rnd_next_obs)
                predict_next_feature = rnd_model.predictor(rnd_next_obs)
                curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data

            # One host copy per rollout step, rather than one per finished episode.
            step_curiosity = curiosity_rewards[step].detach().cpu().numpy()
            for idx, d in enumerate(done):
                if d and info["lives"][idx] == 0:
                    avg_returns.append(info["r"][idx])
                    record.add_episode({
                        "step": global_step,
                        "train/extrinsic_reward": float(info["r"][idx]),
                        "train/intrinsic_reward": float(step_curiosity[idx]),
                        "train/episode_length": int(info["l"][idx]),
                    })
        rollout_seconds = time.time() - rollout_start

        # Normalise the intrinsic reward by the standard deviation of its discounted return, which
        # is what the original RND does — not by the standard deviation of the reward itself.
        curiosity_reward_per_env = np.array(
            [discounted_reward.update(reward_per_step) for reward_per_step in curiosity_rewards.cpu().data.numpy().T]
        )
        mean, std, count = (np.mean(curiosity_reward_per_env), np.std(curiosity_reward_per_env),
                            len(curiosity_reward_per_env))
        reward_rms.update_from_moments(mean, std**2, count)
        raw_curiosity_mean = float(curiosity_rewards.mean().item())
        curiosity_rewards /= np.sqrt(reward_rms.var)

        update_start = time.time()
        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs)
            next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
            ext_advantages = torch.zeros_like(rewards, device=device)
            int_advantages = torch.zeros_like(curiosity_rewards, device=device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                # The intrinsic return is deliberately non-episodic, as in Burda et al., so it
                # carries no terminal mask.
                int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues - int_values[t]
                ext_advantages[t] = ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                int_advantages[t] = int_delta + args.int_gamma * args.gae_lambda * int_lastgaelam
                if args.fix_envpool_autoreset:
                    # dones[t] == 1 marks exactly the rows envpool burned auto-resetting. Pass the
                    # incoming carry straight through them so row t-1 chains to row t+1 as if the
                    # burned row were not there. Without this the fabricated row leaks backward
                    # through the whole rollout in the intrinsic stream, which has no terminal mask
                    # to stop it — measured at 50% amplitude ten rows back.
                    burned = dones[t]
                    ext_lastgaelam = burned * ext_lastgaelam + (1.0 - burned) * ext_advantages[t]
                    int_lastgaelam = burned * int_lastgaelam + (1.0 - burned) * int_advantages[t]
                else:
                    ext_lastgaelam = ext_advantages[t]
                    int_lastgaelam = int_advantages[t]
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)
        b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

        # Update the observation normaliser from this batch. The terminal observation is a genuine
        # observation, so every row belongs here, including the burned ones.
        v = b_obs[:, 3, :, :].reshape(-1, 1, 84, 84)
        if args.opt_gpu_obs_rms:
            # before: 462 MB copied device-to-host, then single-threaded numpy moments over 115.6M
            #         elements — 389 ms per iteration
            # after:  the same parallel-variance update, with the batch moments computed on device
            obs_rms.update_from_moments(
                v.float().mean(0).cpu().numpy().astype(np.float64),
                v.float().var(0, unbiased=False).cpu().numpy().astype(np.float64),
                v.shape[0],
            )
        else:
            obs_rms.update(v.cpu().numpy())

        if args.opt_gpu_norm_stats:
            obs_mean_g = torch.as_tensor(obs_rms.mean, dtype=torch.float32, device=device)
            obs_std_g = torch.sqrt(torch.as_tensor(obs_rms.var, dtype=torch.float32, device=device))
            rnd_next_obs = ((v.float() - obs_mean_g) / obs_std_g).clip(-5, 5)
        else:
            rnd_next_obs = (((v.float() - torch.from_numpy(obs_rms.mean).to(device))
                             / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))).clip(-5, 5)).float()

        # Drop the burned rows from the batch: envpool discarded their action, so the row describes
        # a transition that never happened.
        # before: b_inds = [0, 1, 2, 3, ...] — every row of the flattened batch
        # after:  b_inds = [0, 1, 3, ...] — the burned rows removed, about 1 to 4 per cent
        if args.fix_envpool_autoreset:
            b_inds = torch.nonzero(dones.reshape(-1) == 0, as_tuple=False).squeeze(-1).cpu().numpy()
        else:
            b_inds = np.arange(args.batch_size)
        dropped_last_update = args.batch_size - len(b_inds)
        # Keep the minibatch shape CONSTANT across updates. The number of rows the auto-reset fix
        # drops varies from iteration to iteration, so a minibatch sized as valid//num_minibatches
        # changes shape every update — and cuDNN's autotuner re-benchmarks every convolution on
        # every new shape. Measured cost of letting the shape drift, at 16 cpus: the update phase
        # went 1.28 s -> 5.32 s on a Quadro RTX 6000 and 2.22 s -> 10.02 s on a Quadro RTX 4000,
        # while the rollout was unchanged. Holding back a fixed allowance costs a few genuine rows
        # per update, chosen at random by the shuffle, and keeps one shape for the whole run.
        # before: valid=16139 -> minibatch 4034; next update valid=16146 -> minibatch 4036 (new shape)
        # after:  target=15360 -> minibatch 3840 every update, whatever the drop count
        if args.opt_fixed_minibatch_shape:
            target = ((args.batch_size - args.minibatch_drop_allowance) // args.num_minibatches
                      * args.num_minibatches)
            if len(b_inds) >= target:
                np.random.shuffle(b_inds)
                b_inds = b_inds[:target]
            # If the auto-reset dropped more rows than the allowance, keep every valid row for this
            # update and accept one odd shape rather than train on fabricated transitions.
        valid_batch_size = len(b_inds)
        minibatch_size = max(1, valid_batch_size // args.num_minibatches)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            # Walk exactly num_minibatches equal slices and drop the remainder, at most
            # num_minibatches-1 rows. Walking to valid_batch_size instead would leave a ragged final
            # slice; when that slice holds one row, mb_advantages.std() is NaN and the NaN reaches
            # the weights within one step. The shuffle above means the dropped rows differ per epoch.
            # before: valid=127, num_minibatches=2 -> slices of 63, 63, 1   <- the 1-row slice is fatal
            # after:  valid=127, num_minibatches=2 -> slices of 63, 63      <- 1 row unused this epoch
            for start in range(0, minibatch_size * args.num_minibatches, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                with torch.autocast("cuda", dtype=torch.float16, enabled=args.opt_amp_fp16):
                    predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[mb_inds])
                    forward_loss = F.mse_loss(
                        predict_next_state_feature, target_next_state_feature.detach(), reduction="none"
                    ).mean(-1)

                    if args.opt_no_sync_update:
                        # .type(torch.FloatTensor) is the CPU type, so the original round-trips this
                        # mask to the host and back on every minibatch.
                        mask = (torch.rand(len(forward_loss), device=device) < args.update_proportion).float()
                        forward_loss = (forward_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)
                    else:
                        mask = torch.rand(len(forward_loss), device=device)
                        mask = (mask < args.update_proportion).type(torch.FloatTensor).to(device)
                        forward_loss = (forward_loss * mask).sum() / torch.max(
                            mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                        )

                    _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                    if args.clip_vloss:
                        ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                        ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                            new_ext_values - b_ext_values[mb_inds], -args.clip_coef, args.clip_coef,
                        )
                        ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                        ext_v_loss = 0.5 * torch.max(ext_v_loss_unclipped, ext_v_loss_clipped).mean()
                    else:
                        ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                    int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                    v_loss = ext_v_loss + int_v_loss
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss

                optimizer.zero_grad()
                if args.opt_amp_fp16:
                    scaler.scale(loss).backward()
                    # The gradients must be unscaled before clipping, or max_grad_norm clips the
                    # scaled gradients and silently does the wrong thing.
                    scaler.unscale_(optimizer)
                    if args.max_grad_norm:
                        nn.utils.clip_grad_norm_(combined_parameters, args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.max_grad_norm:
                        nn.utils.clip_grad_norm_(combined_parameters, args.max_grad_norm)
                    optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        update_seconds = time.time() - update_start

        # Append one row per log_every_updates policy updates, and always on the last one.
        if update % args.log_every_updates == 0 or update == num_updates:
            elapsed = time.time() - start_time
            steps_per_second = (global_step - steps_at_start) / max(elapsed, 1e-9)
            record.add_update(
                train_entry={
                    "step": global_step,
                    "update": update,
                    "train/mean_extrinsic_reward": float(np.mean(avg_returns)) if avg_returns else None,
                    "train/mean_intrinsic_reward": raw_curiosity_mean,
                    "train/mean_normalized_intrinsic_reward": float(curiosity_rewards.mean().item()),
                    "train/n_episodes_averaged": len(avg_returns),
                },
                eval_entry={
                    "step": global_step,
                    "update": update,
                    "charts/steps_per_second": steps_per_second,
                    # The honest per-iteration cost: the rollout, the update, and everything
                    # between them (the discounted-return filter, GAE, the normaliser refresh).
                    # Steady-state throughput is batch_size / iteration_seconds.
                    "charts/iteration_seconds": time.time() - iteration_start,
                    "charts/rollout_seconds": rollout_seconds,
                    "charts/update_seconds": update_seconds,
                    "charts/obs_norm_init_seconds": obs_norm_init_seconds,
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "charts/burned_rows_dropped": dropped_last_update,
                    # Peak device memory decides how many runs fit on one GPU.
                    "charts/gpu_memory_peak_mb": (
                        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
                    ),
                    "charts/gpu_memory_reserved_mb": (
                        torch.cuda.max_memory_reserved() / 1024 / 1024 if device.type == "cuda" else 0.0
                    ),
                    "losses/value_loss": float(v_loss.item()),
                    "losses/policy_loss": float(pg_loss.item()),
                    "losses/entropy": float(entropy_loss.item()),
                    "losses/fwd_loss": float(forward_loss.item()),
                    "losses/approx_kl": float(approx_kl.item()),
                    "losses/old_approx_kl": float(old_approx_kl.item()),
                },
            )
            record.flush(completed=False)
            print(f"update={update}/{num_updates} global_step={global_step} "
                  f"steps_per_second={steps_per_second:.0f} dropped={dropped_last_update} "
                  f"mean_extrinsic_reward={np.mean(avg_returns) if avg_returns else float('nan'):.1f}",
                  flush=True)

        # Checkpoint on a wall-clock cadence, so the interval is the same however fast the node is.
        if time.time() - last_checkpoint_time >= args.checkpoint_every_seconds:
            info_ckpt = save_checkpoint(
                checkpoint_path, agent=agent, rnd_model=rnd_model, optimizer=optimizer,
                obs_rms=obs_rms, reward_rms=reward_rms, discounted_reward=discounted_reward,
                global_step=global_step, update=update, avg_returns=avg_returns, device=device,
            )
            record.flush(completed=False)
            last_checkpoint_time = time.time()
            print(f"[checkpoint] update={update} global_step={global_step} "
                  f"bytes={info_ckpt['bytes']:,}", flush=True)

        if args.profile_iterations and update - start_update + 1 >= args.profile_iterations:
            break

    # A profiling run stops early on purpose, so it is not a completed training run.
    completed = args.profile_iterations == 0
    save_checkpoint(
        checkpoint_path, agent=agent, rnd_model=rnd_model, optimizer=optimizer, obs_rms=obs_rms,
        reward_rms=reward_rms, discounted_reward=discounted_reward, global_step=global_step,
        update=update, avg_returns=avg_returns, device=device,
    )
    record.flush(completed=completed)
    envs.close()
    print(f"[done] completed={completed} global_step={global_step} record={record_path}", flush=True)


if __name__ == "__main__":
    main()
