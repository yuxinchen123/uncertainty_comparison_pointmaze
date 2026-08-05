"""One subsection per change that the 30-seed run actually uses.

Each entry carries: what the change is, the upstream code, the replacement, the measured gain, and
what it costs. The gain is not written here — it is computed from the ladder JSON, so the prose and
the numbers can never drift apart.

`ladder_key` names the ladder variant that switched this option on. Its gain is that variant's
throughput divided by the previous variant's, per node. Two entries have no ladder variant because
their effect is not steady-state throughput: the start-up rewrite (measured separately) and the
constant minibatch shape (measured by the four-way shape test).
"""

OPTIONS = [
    {
        "flag": "--opt_env_threads",
        "ladder_key": "01_env_threads",
        "title": "Tell the environment simulator how many cores it actually has",
        "what": """
envpool decides its own worker-thread count as `min(batch_size, hardware_concurrency())`, and
`hardware_concurrency()` reports the **machine's** core count, not the job's allocation. Inside a
16-cpu allocation on a 224-core node that default becomes 128 threads competing for 16 cores. The
run passes the number explicitly.

There is a second half to this that the canary caught rather than the profiling. In a packed job
Slurm sets `SLURM_CPUS_PER_TASK` to the **job's** total, so five runs sharing a 40-cpu job would each
ask for 40 threads on eight cores' worth of allocation. The sweep's worker manager exports
`GPU_SWEEP_CPUS_PER_RUN` with each run's real share, and that is read first.
""",
        "before": '''envs = envpool.make(
    args.env_id,
    env_type="gym",
    num_envs=args.num_envs,
    episodic_life=True,
    reward_clip=True,
    seed=args.seed,
    repeat_action_probability=0.25,
)                       # num_threads not passed -> envpool reads the machine's core count''',
        "after": '''def resolve_env_threads(requested):
    """Pick envpool's worker-thread count, since envpool cannot see the job's cpu allocation."""
    if requested > 0:
        return requested
    per_run = os.environ.get("GPU_SWEEP_CPUS_PER_RUN")   # this run's share in a packed job
    if per_run:
        return int(per_run)
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

envs = envpool.make(
    args.env_id,
    env_type="gym",
    num_envs=args.num_envs,
    episodic_life=True,
    reward_clip=True,
    seed=args.seed,
    repeat_action_probability=0.25,
    num_threads=env_threads,
)''',
        "costs": """
**Nothing.** The environment stepping is identical; only the number of threads doing it changes, and
`seed` still controls the per-environment seeds.

The measured gain in the ladder is close to zero, because the ladder ran at 16 cpus on nodes where
envpool's own default happened to be workable. The reason to keep it is the packed case, where the
default is a five-fold oversubscription — and that case is the entire reservation node.
""",
    },
    {
        "flag": "--opt_gpu_obs_rms",
        "ladder_key": "02_opt_gpu_obs_rms",
        "title": "Compute the observation-normalizer's batch moments on the GPU",
        "what": """
Once per iteration the observation normalizer is updated from the whole 16,384-row batch. Upstream
copies that batch to the host and runs single-threaded numpy over 115.6 million elements. The
parallel-variance update itself is unchanged — only where the batch mean and variance are computed
moves.
""",
        "before": '''obs_rms.update(b_obs[:, 3, :, :].reshape(-1, 1, 84, 84).cpu().numpy())
# 462 MB device-to-host, then numpy mean/var over 115.6M elements on one core''',
        "after": '''v = b_obs[:, 3, :, :].reshape(-1, 1, 84, 84)
obs_rms.update_from_moments(
    v.float().mean(0).cpu().numpy().astype(np.float64),
    v.float().var(0, unbiased=False).cpu().numpy().astype(np.float64),
    v.shape[0],
)
# the same pooled-moments update; only the two moments are computed on device''',
        "costs": """
**A difference of about 1 part in 10 million, from summation order.** `unbiased=False` matches
numpy's population variance, so it is the same estimator — but the GPU sums in a different order, so
the result is not bit-identical. The mean is in fact exact either way: the values are integers no
larger than 255 and the batch sum stays below 2^24, where float32 is exact.

This is the only place in the run where a same-numbers change is accepted at 1e-7 rather than
bit-identical. It is worth it: this single line was about a fifth of all GPU-side time per iteration.
""",
    },
    {
        "flag": "--opt_fused_policy_pass",
        "ladder_key": "03_opt_fused_policy_pass",
        "title": "Run the policy trunk once per rollout step instead of twice",
        "what": """
Each rollout step calls `get_value` and then `get_action_and_value` on the **same** observation. Both
run the full convolutional trunk, so the trunk runs twice on identical input, 128 times per
iteration. One method returns everything from one pass.
""",
        "before": '''with torch.no_grad():
    value_ext, value_int = agent.get_value(obs[step])          # trunk pass 1
    ext_values[step], int_values[step] = (
        value_ext.flatten(),
        value_int.flatten(),
    )
    action, logprob, _, _, _ = agent.get_action_and_value(obs[step])   # trunk pass 2, same input''',
        "after": '''def get_action_and_value_rollout(self, x):
    """Sample an action and both values in ONE trunk pass, for the rollout."""
    hidden = self.network(x / 255.0)
    features = self.extra_layer(hidden)
    value_ext, value_int = self.critic_ext(features + hidden), self.critic_int(features + hidden)
    probs = Categorical(logits=self.actor(hidden))
    action = probs.sample()
    return action, probs.log_prob(action), value_ext, value_int

with torch.no_grad():
    action, logprob, value_ext, value_int = agent.get_action_and_value_rollout(obs[step])
    ext_values[step], int_values[step] = value_ext.flatten(), value_int.flatten()''',
        "costs": """
**Nothing — bit-identical.** `get_value` draws no random numbers, so removing it does not disturb the
random stream that `probs.sample()` consumes; the actions are the same actions. The entropy that
`get_action_and_value` also returns is unused in the rollout. Verified by comparing both forms on
device.
""",
    },
    {
        "flag": "--opt_rnd_no_grad",
        "ladder_key": "04_opt_rnd_no_grad",
        "title": "Compute the rollout's exploration bonus without building a gradient graph",
        "what": """
The bonus for each rollout step is a forward pass through the two Random Network Distillation towers.
Upstream computes it **outside** the surrounding `no_grad` block. The predictor has trainable
parameters, so autograd builds a full graph for a 128-row forward pass, 128 times per iteration, and
then throws all of it away at the trailing `.data`.
""",
        "before": '''target_next_feature = rnd_model.target(rnd_next_obs)
predict_next_feature = rnd_model.predictor(rnd_next_obs)
curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data
# builds and discards an autograd graph 128 times per iteration''',
        "after": '''with torch.no_grad():
    target_next_feature = rnd_model.target(rnd_next_obs)
    predict_next_feature = rnd_model.predictor(rnd_next_obs)
    curiosity_rewards[step] = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2''',
        "costs": """
**Nothing — bit-identical.** The value was already detached by `.data`, and the target tower already
has `requires_grad=False`. The graph was pure waste. It also removes 128 allocate-and-free cycles of
activation memory per iteration, which is why peak memory drops slightly.
""",
    },
    {
        "flag": "--opt_uint8_obs",
        "ladder_key": "05_opt_uint8_obs",
        "title": "Keep the observation buffer as bytes instead of floats",
        "what": """
The rollout observation buffer has shape 128 x 128 x 4 x 84 x 84. As `float32` that is **1.85 GB**;
as `uint8`, which is what the environment actually returns, it is **0.46 GB**. Worse than the memory:
`torch.Tensor(next_obs)` converts the array from `uint8` to `float32` **on the host** and then copies
the float version to the GPU, so every step moved four times the bytes it needed to.
""",
        "before": '''obs = torch.zeros((args.num_steps, args.num_envs)
                  + envs.single_observation_space.shape).to(device)   # float32: 1.85 GB
...
next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
# torch.Tensor() is FloatTensor: converts uint8 -> float32 on the CPU, then copies 14.5 MB''',
        "after": '''obs_dtype = torch.uint8 if args.opt_uint8_obs else torch.float32
obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape,
                  dtype=obs_dtype).to(device)                          # uint8: 0.46 GB
...
next_obs = torch.as_tensor(next_obs_np, device=device)                 # uint8, copies 3.6 MB
next_done = torch.as_tensor(done, dtype=torch.float32, device=device)''',
        "costs": """
**Nothing through the networks — bit-identical.** Both networks divide the input by 255.0 as their
first operation, and every `uint8` value from 0 to 255 is exactly representable in `float32`, so
`uint8/255.0` and `float32/255.0` give the same floats. Verified with an exact comparison on device.

One caveat that does not apply here: numpy promotes a `uint8` array to `float64` for `mean`/`var` but
keeps a `float32` array in `float32`, so a **host-side** normalizer update would shift by about 1e-7.
The normalizer moments moved to the GPU in the previous change, so the two interact harmlessly.

This is the largest single change in the ladder.
""",
    },
    {
        "flag": "--opt_gpu_norm_stats",
        "ladder_key": "06_opt_gpu_norm_stats",
        "title": "Keep the normalizer statistics on the GPU, in single precision",
        "what": """
The observation normalizer stores its mean and variance as `float64` numpy arrays. Upstream uploads
them from host to device on **every one of the 128 rollout steps**, and because they arrive as
`float64` every arithmetic step promotes to `float64` before the trailing `.float()`. On the whole
batch that means three transient `float64` tensors of 925 MB each. The run keeps one device-resident
`float32` copy, refreshed once per iteration.
""",
        "before": '''rnd_next_obs = (
    (
        (next_obs[:, 3, :, :].reshape(args.num_envs, 1, 84, 84)
         - torch.from_numpy(obs_rms.mean).to(device))      # float64, re-uploaded every step
        / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
    ).clip(-5, 5)
).float()''',
        "after": '''# refreshed once per iteration, not once per rollout step
obs_mean_g = torch.as_tensor(obs_rms.mean, dtype=torch.float32, device=device)
obs_std_g = torch.sqrt(torch.as_tensor(obs_rms.var, dtype=torch.float32, device=device))
...
rnd_frame = next_obs[:, 3, :, :].reshape(args.num_envs, 1, 84, 84).float()
rnd_next_obs = ((rnd_frame - obs_mean_g) / obs_std_g).clip(-5, 5)''',
        "costs": """
**One or two units in the last place of float32.** The formula is the same; the intermediates are
`float32` instead of `float64` before a cast to `float32` that was happening anyway. Statistically
irrelevant, but not bit-identical, so it is recorded as a change rather than a pure speedup.

The effect is largest on the consumer cards, which run `float64` at a thirty-second to a
sixty-fourth of their `float32` rate — that is most of this cluster.
""",
    },
    {
        "flag": "--opt_no_sync_update",
        "ladder_key": "07_opt_no_sync_update",
        "title": "Remove three forced waits from the innermost update loop",
        "what": """
Three lines inside the minibatch loop each force the GPU pipeline to drain, sixteen times per
iteration:

1. `clipfracs` appends a `.item()`, which waits for the GPU. The list is assigned, appended to, and
   **never read** — it is dead code whose only effect is the wait.
2. `torch.FloatTensor` is the **CPU** tensor type, so `.type(torch.FloatTensor).to(device)` copies the
   mask down to the host and straight back up, with a wait in between.
3. `torch.tensor([1], device=device)` allocates a new device tensor from a Python list every
   minibatch.
""",
        "before": '''clipfracs = []
...
mask = torch.rand(len(forward_loss), device=device)
mask = (mask < args.update_proportion).type(torch.FloatTensor).to(device)   # GPU -> CPU -> GPU
forward_loss = (forward_loss * mask).sum() / torch.max(
    mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)       # new tensor each time
)
...
clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()] # .item() waits; unused''',
        "after": '''mask = (torch.rand(len(forward_loss), device=device) < args.update_proportion).float()
forward_loss = (forward_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)
# the clipfracs line is deleted: it was never read''',
        "costs": """
**Nothing — bit-identical.** `.float()` and `.type(torch.FloatTensor).to(device)` produce the same
values; `torch.clamp(x, min=1.0)` equals `torch.max(x, tensor([1.0]))`; and `clipfracs` was unused, so
deleting it removes no information.

The only thing lost is a diagnostic that upstream computed and discarded. If the clipping fraction is
ever wanted, it should be accumulated on device and read once per iteration, not once per minibatch.
""",
    },
    {
        "flag": "--opt_cudnn_benchmark",
        "ladder_key": "08_opt_cudnn_benchmark",
        "title": "Let the convolution library pick its algorithm by measurement",
        "what": """
cuDNN can either choose a convolution algorithm from a built-in heuristic or try the candidates once
and keep the fastest. The shapes here are fixed — 128 in the rollout, and a constant minibatch in the
update — so the tuning cost is paid once and amortised over the whole run.
""",
        "before": '''torch.backends.cudnn.deterministic = args.torch_deterministic
# benchmark left at its default of False''',
        "after": '''torch.backends.cudnn.deterministic = args.torch_deterministic
if args.opt_cudnn_benchmark:
    torch.backends.cudnn.benchmark = True''',
        "costs": """
**Three things, all small, and one large trap.**

1. **Not bit-reproducible across runs.** The tuner may pick a different, still mathematically correct,
   algorithm on another machine or another day. Because the script also sets
   `cudnn.deterministic = True`, the tuner only chooses among deterministic algorithms, so a single
   run stays reproducible within itself.
2. **More workspace memory.** Peak GPU memory rose from about 4,770 MB to 6,212 MB on the two cards
   whose tuner picked a larger-workspace algorithm. Both still fit; it is sized for in the estimate.
3. **A few seconds of tuning** at the first occurrence of each shape.

**The trap:** this only works when the shape is constant. The auto-reset correction makes the batch
size differ every iteration, which re-triggers tuning on nearly every update and costs more than
everything above gains. That is what the next change exists to prevent.
""",
    },
    {
        "flag": "--opt_fast_obs_norm_init",
        "ladder_key": None,
        "title": "Fill the start-up buffer directly instead of through Python lists",
        "what": """
Before training, the observation normalizer is primed by stepping a random agent for 6,400 steps.
Upstream builds that batch by calling `.tolist()` on a `(128, 1, 84, 84)` `uint8` array on every
step — creating on the order of 115 million Python objects — and then `np.stack`s a list of 16,384
nested lists, fifty times over.

This does not affect steady-state throughput, so it does not appear in the ladder. It is pure
start-up, and start-up is paid again on **every resume**.
""",
        "before": '''next_ob = []
for step in range(args.num_steps * args.num_iterations_obs_norm_init):
    acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
    s, r, d, _ = envs.step(acs)
    next_ob += s[:, 3, :, :].reshape([-1, 1, 84, 84]).tolist()   # ~115M python objects
    if len(next_ob) % (args.num_steps * args.num_envs) == 0:
        next_ob = np.stack(next_ob)                              # stack 16,384 nested lists
        obs_rms.update(next_ob)
        next_ob = []''',
        "after": '''buf = np.empty((args.num_steps * args.num_envs, 1, 84, 84), dtype=np.uint8)
fill = 0
for _ in range(args.num_steps * args.num_iterations_obs_norm_init):
    acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
    s, _, _, _ = envs.step(acs)
    buf[fill:fill + args.num_envs] = s[:, 3, :, :].reshape(-1, 1, 84, 84)
    fill += args.num_envs
    if fill == buf.shape[0]:
        obs_rms.update(buf)
        fill = 0''',
        "costs": """
**Nothing — bit-identical.** numpy promotes both the stacked integer array and the `uint8` buffer to
`float64` when computing the moments, so `np.mean` and `np.var` return exactly the same values.
Verified by direct comparison.

It also removes a transient spike of about 2 GB of Python objects, which is why the host-memory
figure in the resource estimate is as low as it is.
""",
    },
    {
        "flag": "--opt_fixed_minibatch_shape",
        "ladder_key": None,
        "title": "Hold the minibatch shape constant so the autotuner tunes once",
        "what": """
This one exists only because two otherwise-sound changes are destructive together.

The auto-reset correction drops the rows envpool burned resetting. How many rows that is varies from
iteration to iteration — around 240 to 350 out of 16,384 — so a minibatch sized as
`valid // num_minibatches` changes shape on nearly every update. Convolution autotuning is **per
shape**, so it re-tuned almost every update.

The fix reserves a fixed row allowance and takes a constant number of rows, so the shape is decided
once for the whole run. The rows given up beyond the dropped ones are chosen at random by the shuffle
that was already there.
""",
        "before": '''b_inds = torch.nonzero(dones.reshape(-1) == 0, as_tuple=False).squeeze(-1).cpu().numpy()
valid_batch_size = len(b_inds)                       # 16,139 one update, 16,146 the next ...
minibatch_size = valid_batch_size // args.num_minibatches   # ... so the shape drifts''',
        "after": '''if args.opt_fixed_minibatch_shape:
    target = ((args.batch_size - args.minibatch_drop_allowance) // args.num_minibatches
              * args.num_minibatches)
    if len(b_inds) >= target:
        np.random.shuffle(b_inds)
        b_inds = b_inds[:target]                     # 15,360 rows, every update
    # if the correction dropped more than the allowance, keep every valid row and accept one
    # odd shape rather than train on fabricated transitions
valid_batch_size = len(b_inds)
minibatch_size = max(1, valid_batch_size // args.num_minibatches)''',
        "costs": """
**About 6% of the batch, given up per update.** With a 1,024-row allowance the update sees 15,360
rows instead of the roughly 16,100 that survive the correction. Those rows are real transitions, and
they are dropped at random and differently every update, so nothing is systematically excluded — but
they are dropped.

The alternative was to turn autotuning off, which is simpler and keeps every row. Measured, the
constant shape wins: on a Quadro RTX 6000, 4,765 steps/s with the fixed shape against 4,553 with
autotuning off. So the run pays 6% of rows to keep the tuner, and ends up **faster than the
uncorrected code** because the constant-shape batch is slightly smaller.

If a future change makes the drop count exceed the allowance, the code keeps every valid row for that
update rather than training on fabricated ones. The allowance is a throughput knob; correctness never
depends on it.
""",
    },
]
