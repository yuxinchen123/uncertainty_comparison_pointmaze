# CleanRL `ppo_rnd_envpool.py` — complete specification, and comparison against Burda et al. (2019)

## 0. Provenance of every number below

| Source | Location / identifier |
| --- | --- |
| CleanRL implementation | `/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl/cleanrl/ppo_rnd_envpool.py` (539 lines). Repo commit `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`, dated 2026-04-20. Byte-identical to `/p/rlprojects/RND/07_reconstruction/ppo_rnd_envpool.py` and `/p/rlprojects/cleanrl/cleanrl/ppo_rnd_envpool.py`. |
| CleanRL docs | `/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl/docs/rl-algorithms/ppo-rnd.md` |
| CleanRL benchmark script | `/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl/benchmark/rnd.sh` and `cleanrl_utils/benchmark.py` |
| envpool version pin | `requirements/requirements-envpool.txt` → `envpool==0.6.6`; `pyproject.toml` extra → `envpool>=0.6.4,<0.7` |
| envpool Atari C++ spec | `envpool/atari/atari_env.h` at tag `v0.6.6`, downloaded to `/tmp/claude-2618919/-p-rlprojects-RND/12f4bbb4-d71e-4c5d-8c09-aae9a5c45be7/scratchpad/atari_env_v066.h` |
| envpool Atari registration | `/u/sl5nw/.conda/envs/cleanrl/lib/python3.10/site-packages/envpool/atari/registration.py` (installed 0.6.6) |
| envpool live values | Instantiated `MontezumaRevenge-v5` with CleanRL's exact call, using `/u/sl5nw/.conda/envs/cleanrl/bin/python` |
| envpool web docs | `https://envpool.readthedocs.io/en/latest/env/atari.html` (documents a NEWER envpool than 0.6.6 — see §2.4) |
| RND paper | arXiv:1810.12894, Burda, Edwards, Storkey, Klimov, "Exploration by Random Network Distillation", ICLR 2019. Appendix text extracted from the ar5iv render. |
| Original RND code | `openai/random-network-distillation` at commit `f75c0f1efa473d5109d487062fd8ed49ddce6634` (the commit CleanRL's docs link to): `run_atari.py`, `ppo_agent.py`, `policies/cnn_policy_param_matched.py`, `cmd_util.py`, `atari_wrappers.py` |

Pinned library versions from `requirements/requirements-envpool.txt`: `envpool==0.6.6`, `gym==0.23.1`, `gymnasium==0.29.1`, `torch==2.4.1`, `numpy==1.24.4`, `tyro==0.5.10`, `opencv-python==4.7.0.72`.

---

## 1. Complete default hyperparameter set (the `Args` dataclass)

Every field, in source order. "Meaning" is the source docstring, quoted, except where marked. Note rows 24 and 25: **the two docstrings are swapped in the source** (`int_coef` is documented as "coefficient of extrinsic reward" and `ext_coef` as "coefficient of intrinsic reward"); the variable *names* and the *use sites* are correct, only the docstrings are crossed.

| # | Field | Default | Meaning |
| --- | --- | --- | --- |
| 1 | `exp_name` | `"ppo_rnd_envpool"` (the file's basename) | the name of this experiment |
| 2 | `seed` | `1` | seed of the experiment |
| 3 | `torch_deterministic` | `True` | if toggled, `torch.backends.cudnn.deterministic=False` (docstring is inverted relative to the code, which sets `cudnn.deterministic = args.torch_deterministic`) |
| 4 | `cuda` | `True` | if toggled, cuda will be enabled by default |
| 5 | `track` | `False` | if toggled, this experiment will be tracked with Weights and Biases |
| 6 | `wandb_project_name` | `"cleanRL"` | the wandb's project name |
| 7 | `wandb_entity` | `None` | the entity (team) of wandb's project |
| 8 | `capture_video` | `False` | whether to capture videos of the agent performances (declared but never read in this file) |
| 9 | `env_id` | `"MontezumaRevenge-v5"` | the id of the environment |
| 10 | `total_timesteps` | `2000000000` | total timesteps of the experiments |
| 11 | `learning_rate` | `1e-4` | the learning rate of the optimizer |
| 12 | `num_envs` | `128` | the number of parallel game environments |
| 13 | `num_steps` | `128` | the number of steps to run in each environment per policy rollout |
| 14 | `anneal_lr` | `True` | Toggle learning rate annealing for policy and value networks |
| 15 | `gamma` | `0.999` | the discount factor gamma (extrinsic stream) |
| 16 | `gae_lambda` | `0.95` | the lambda for the general advantage estimation (used for BOTH streams) |
| 17 | `num_minibatches` | `4` | the number of mini-batches |
| 18 | `update_epochs` | `4` | the K epochs to update the policy |
| 19 | `norm_adv` | `True` | Toggles advantages normalization (applied to the COMBINED advantage, per minibatch) |
| 20 | `clip_coef` | `0.1` | the surrogate clipping coefficient (also reused as the value-clip range) |
| 21 | `clip_vloss` | `True` | Toggles whether or not to use a clipped loss for the value function, as per the paper (extrinsic head only) |
| 22 | `ent_coef` | `0.001` | coefficient of the entropy |
| 23 | `vf_coef` | `0.5` | coefficient of the value function |
| 24 | `max_grad_norm` | `0.5` | the maximum norm for the gradient clipping (over agent + predictor parameters jointly) |
| 25 | `target_kl` | `None` | the target KL divergence threshold (no early stop at the default) |
| 26 | `update_proportion` | `0.25` | proportion of exp used for predictor update |
| 27 | `int_coef` | `1.0` | source docstring says "coefficient of extrinsic reward"; **actual role: multiplier on the INTRINSIC advantage** |
| 28 | `ext_coef` | `2.0` | source docstring says "coefficient of intrinsic reward"; **actual role: multiplier on the EXTRINSIC advantage** |
| 29 | `int_gamma` | `0.99` | Intrinsic reward discount rate (used both in the forward filter and in the intrinsic GAE) |
| 30 | `num_iterations_obs_norm_init` | `50` | number of iterations to initialize the observations normalization parameters |
| 31 | `batch_size` | `0`, filled at runtime to `16384` | the batch size (computed in runtime) = `num_envs * num_steps` |
| 32 | `minibatch_size` | `0`, filled at runtime to `4096` | the mini-batch size (computed in runtime) = `batch_size // num_minibatches` |
| 33 | `num_iterations` | `0`, filled at runtime to `122070` | the number of iterations (computed in runtime) = `total_timesteps // batch_size`. Note the training loop actually uses a second, identically-computed local variable `num_updates`, not this field. |

Derived quantities (arithmetic checked):

1. `batch_size` $= 128 \times 128 = 16{,}384$.
2. `minibatch_size` $= 16{,}384 / 4 = 4{,}096$.
3. `num_updates` $= 2{,}000{,}000{,}000 \;//\; 16{,}384 = 122{,}070$ policy updates (covering $1{,}999{,}994{,}880$ environment steps).
4. Optimizer steps $= 122{,}070 \times 4 \text{ epochs} \times 4 \text{ minibatches} = 1{,}953{,}120$.
5. `global_step` increments by `num_envs` per rollout step, so it counts **environment steps** (post-frame-skip agent steps summed over the 128 copies). At `frame_skip = 4` the configured budget is $8.0 \times 10^{9}$ emulator frames.
6. Rollout observation buffer: $128 \times 128 \times 4 \times 84 \times 84$ float32 $= 1.85$ GB on device (CleanRL stores the uint8 frames as float32).
7. Observation-normalization warm-up: $128 \times 50 = 6{,}400$ rollout steps $\times 128$ envs $= 819{,}200$ environment steps ($3{,}276{,}800$ frames), producing exactly 50 `obs_rms` updates of 16,384 frames each.

---

## 2. The exact benchmark command line

`benchmark/rnd.sh` in full (4 executable lines; line 1 is a commented-out `WANDB_ENTITY` export):

```bash
# export WANDB_ENTITY=openrlbenchmark

uv pip install ".[envpool]"
xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids MontezumaRevenge-v5 \
    --command "uv run python cleanrl/ppo_rnd_envpool.py --track" \
    --num-seeds 1 \
    --workers 1
```

`cleanrl_utils/benchmark.py` builds `commands` as `[" ".join([command, "--env-id", env_id, "--seed", str(start_seed + seed)])]` over `seed in range(0, num_seeds)` with `start_seed = 1` by default. With `--num-seeds 1` and one env id, that expands to exactly one run:

```bash
uv run python cleanrl/ppo_rnd_envpool.py --track --env-id MontezumaRevenge-v5 --seed 1
```

`--workers 1` means it is actually executed (a `ThreadPoolExecutor` with one worker); `auto_tag` is on by default, so the wandb run carries a `WANDB_TAGS` value derived from the git tag/commit plus the pull-request number. Every other hyperparameter takes the dataclass default in §1.

Reported result (CleanRL docs, `ppo-rnd.md`): `MontezumaRevengeNoFrameSkip-v4` / `MontezumaRevenge-v5` → **7100, one seed**, versus **8152, three seeds** attributed to Burda et al. 2019 Figure 7 at 2000M steps. The docs state the single seed is due to limited compute and a run time of about 250 hours. The committed learning curve `docs/rl-algorithms/ppo-rnd/MontezumaRevenge-v5.png` has an x-axis running to just under 2G steps, so the benchmark run did consume close to the full configured budget.

---

## 3. Environment specification

### 3.1 The exact construction call

```python
envs = envpool.make(
    args.env_id,                      # "MontezumaRevenge-v5"
    env_type="gym",
    num_envs=args.num_envs,           # 128
    episodic_life=True,
    reward_clip=True,
    seed=args.seed,                   # 1
    repeat_action_probability=0.25,
)
envs.num_envs = args.num_envs
envs.single_action_space = envs.action_space
envs.single_observation_space = envs.observation_space
envs = RecordEpisodeStatistics(envs)          # CleanRL's own wrapper, defined in the file
assert isinstance(envs.action_space, gym.spaces.Discrete)
```

No other wrapper is applied. All Atari preprocessing happens inside envpool's C++ code.

### 3.2 Every envpool config key in force

These are the 21 keys of `AtariEnvSpec` in envpool 0.6.6. "Origin" says where the value comes from: `passed` = an explicit keyword in CleanRL's `envpool.make` call, `registration` = set by envpool's own `atari/registration.py`, `default` = the C++ `AtariEnvFns::DefaultConfig()` value, `auto` = derived by envpool. All values were read back from the live `envs.config` of an env built with CleanRL's exact call.

| Key | Value in CleanRL | Origin | Notes |
| --- | --- | --- | --- |
| `task` | `montezuma_revenge` | auto, from the env id | ROM `envpool/atari/roms/montezuma_revenge.bin` |
| `num_envs` | `128` | passed | 128 parallel ALE copies |
| `batch_size` | `128` | auto | defaults to `num_envs`, so stepping is fully synchronous |
| `num_threads` | `0` | default | 0 means "use `batch_size` threads" |
| `max_num_players` | `1` | default | single-agent |
| `thread_affinity_offset` | `-1` | default | no pinning |
| `base_path` | envpool install dir | auto | ROM search root |
| `seed` | `1` | passed | env copy `i` gets `seed + i` internally |
| `gym_reset_return_info` | `False` | default | `reset()` returns only the observation |
| `max_episode_steps` | `108000` | registration | **agent steps**, not frames; empirically verified (see §3.5) |
| `stack_num` | `4` | default | 4-frame stack |
| `frame_skip` | `4` | default | action repeated 4 emulator frames |
| `noop_max` | `30` | default | 0 to 29 no-ops at reset (see §3.4) |
| `episodic_life` | `True` | passed | `done` is raised on loss of life |
| `zero_discount_on_life_loss` | `False` | default | affects the dm-env `discount` only |
| `reward_clip` | `True` | passed | reward mapped to its sign |
| `img_height` | `84` | default | |
| `img_width` | `84` | default | |
| `gray_scale` | `True` | default | ALE palette grayscale conversion |
| `use_inter_area_resize` | `True` | default | OpenCV `INTER_AREA` for the 210x160 to 84x84 resize |
| `repeat_action_probability` | `0.25` | passed | sticky actions, set on the ALE itself |

### 3.3 Spaces, dtypes, and step outputs (read from a live env)

| Item | Value |
| --- | --- |
| `observation_space` | `Box(0, 255, (4, 84, 84), uint8)` |
| Observation shape from `reset` / `step`, 128 envs | `(128, 4, 84, 84)`, dtype `uint8` |
| Channel layout | `[stack_num * (gray_scale ? 1 : 3), img_height, img_width]`, oldest frame first, so index 3 is the most recent frame |
| `action_space` | `Discrete(18)` |
| Action set | `ale::ALEInterface::getMinimalActionSet()`; for Montezuma's Revenge this equals the full 18-action set |
| Reward dtype / shape | `float32`, `(128,)` — clipped to the sign |
| Done dtype | `bool`, `(128,)` |
| Info keys | `TimeLimit.truncated` (bool), `elapsed_step` (int32), `env_id` (int32), `lives` (int32), `players` (dict), `reward` (float32, the RAW unclipped reward), `terminated` (int32, ALE `game_over()`) |
| Auto-reset | envpool resets a finished copy internally; the observation returned on the `done` step is the terminal state (see the "bug" admonition in `ppo-rnd.md` about the off-by-one relative to gym's vector env) |

### 3.4 Preprocessing pipeline as implemented in `atari_env.h` (v0.6.6)

1. **Reset.** Draw `noop` uniformly from $\{0,\dots,29\}$ (`dist_noop_(0, noop\_max - 1)` then `+1 - fire_reset_`, and `fire_reset_` is true because FIRE is in Montezuma's minimal action set). Execute that many NOOP frames, then execute one FIRE frame. With `episodic_life = True`, the underlying game is reset only if the ALE reports `game_over()` or the step limit was hit — a life-loss reset therefore runs the no-ops and the FIRE **mid-game**.
2. **Step.** Repeat the chosen action for `frame_skip = 4` emulator frames, breaking early on `game_over()`. Rewards over the skipped frames are summed.
3. **Max-pool.** The last two of the four frames are element-wise maxed.
4. **Grayscale.** ALE `colourPalette().applyPaletteGrayscale()` on the raw 210x160 screen.
5. **Resize.** 210x160 to 84x84 with `INTER_AREA`.
6. **Stack.** Push onto a 4-deep deque, oldest popped. At a full game reset the newest frame is copied into all four slots.
7. **Termination.** `done = game_over() OR (elapsed_step >= 108000) OR (episodic_life AND 0 < lives() < previous lives)`.
8. **Reward clipping.** `reward = sign(reward)`; the unclipped value is preserved in `info["reward"]`.
9. **Sticky actions.** `env_->setFloat("repeat_action_probability", 0.25)` — ALE's own per-frame sticky-action mechanism.

### 3.5 Two facts to be careful about in the environment table

1. **`max_episode_steps` is 108000 agent steps, not 27000 and not 108000 frames.** The envpool web documentation table lists a default of `27000`, but that is the raw C++ default for a *newer* envpool; in 0.6.6 the C++ default is `2147483647` (INT_MAX) and `atari/registration.py` overrides it to `108000` for every `*-v5` id. Verified empirically: an env built with `max_episode_steps=10` returned `done=True` with `elapsed_step=10` and `TimeLimit.truncated=True` after exactly 10 agent steps. At `frame_skip = 4` the effective cap is therefore $432{,}000$ emulator frames, i.e. about 2 hours of game time — in practice non-binding.
2. **`use_fire_reset` and `full_action_space` do not exist in envpool 0.6.6.** The current web documentation lists them (defaults `True` and `False`); the pinned 0.6.6 `DefaultConfig()` has neither. In 0.6.6 the FIRE-on-reset is unconditional whenever FIRE is in the minimal action set, and the action set is always the minimal one. For Montezuma's Revenge the resulting behaviour happens to coincide with `use_fire_reset=True, full_action_space=False`, since its minimal set is all 18 actions.

Also worth recording: envpool's own documentation states that its Atari environments follow gym's `*NoFrameskip-v4` ALE settings (with the openai/baselines wrappers) rather than the ALE v5 defaults, despite the `-v5` suffix in the id. The `-v5`-style sticky actions are only present here because CleanRL passes `repeat_action_probability=0.25` explicitly.

---

## 4. The RND design of the CleanRL implementation, knob by knob

Notation used below. Define $s_t$ as the 4-frame stacked observation at rollout index $t$, $f$ as the frozen target network, $\hat f$ as the trained predictor, and $d = 512$ as the shared output width of both. Define $\mu$ and $\sigma^2$ as the per-pixel running mean and variance held in `obs_rms` (a `gym.wrappers.normalize.RunningMeanStd` of shape `(1, 1, 84, 84)`, initialised to mean 0, variance 1, count $10^{-4}$).

### 4.1 What the bonus is computed on

1. **Which frame.** Only `next_obs[:, 3, :, :]`, reshaped to `(num_envs, 1, 84, 84)` — the single most recent frame of the 4-stack of the observation *after* the environment step. The policy sees all 4 stacked frames; the RND networks see 1.
2. **Normalisation.** $x \mapsto \mathrm{clip}\!\left((x-\mu)/\sigma,\,-5,\,5\right)$, computed in float64 from the numpy `obs_rms` buffers, then cast to float32. There is no epsilon added under the square root, and the raw uint8 values (0 to 255) are used — the division by 255 that the policy trunk applies is **not** applied here.
3. **`obs_rms` update schedule.** Once before training, from 6,400 random-agent rollout steps in 50 chunks of 16,384 frames; and thereafter once per policy iteration, from the whole rollout batch, `obs_rms.update(b_obs[:, 3, :, :].reshape(-1, 1, 84, 84))`. Note the ordering inside an iteration: the rollout bonuses are computed with the statistics from the *end of the previous* iteration; `obs_rms` is then updated; and the predictor's training batch is normalised with the *updated* statistics.

### 4.2 Architectures (`RNDModel`)

Both towers take a `(N, 1, 84, 84)` float32 input. The constructor arguments `RNDModel(4, envs.single_action_space.n)` are stored but never used; the conv input channel count is hard-coded to 1 and the output width to 512.

**Predictor** $\hat f$ (trainable, 9 modules):

| Layer | Spec | Activation |
| --- | --- | --- |
| conv 1 | `Conv2d(1, 32, kernel 8, stride 4)` → 20x20x32 | LeakyReLU |
| conv 2 | `Conv2d(32, 64, kernel 4, stride 2)` → 9x9x64 | LeakyReLU |
| conv 3 | `Conv2d(64, 64, kernel 3, stride 1)` → 7x7x64 | LeakyReLU |
| flatten | 3136 | — |
| fc 1 | `Linear(3136, 512)` | ReLU |
| fc 2 | `Linear(512, 512)` | ReLU |
| fc 3 | `Linear(512, 512)` | none (output) |

**Target** $f$ (frozen, `requires_grad = False` on every parameter):

| Layer | Spec | Activation |
| --- | --- | --- |
| conv 1 | `Conv2d(1, 32, kernel 8, stride 4)` | LeakyReLU |
| conv 2 | `Conv2d(32, 64, kernel 4, stride 2)` | LeakyReLU |
| conv 3 | `Conv2d(64, 64, kernel 3, stride 1)` | LeakyReLU |
| flatten | 3136 | — |
| fc | `Linear(3136, 512)` | none (output) |

**Initialisation.** Every layer of both towers goes through `layer_init(layer, std=np.sqrt(2), bias_const=0.0)`, i.e. `torch.nn.init.orthogonal_(weight, gain=sqrt(2))` and `constant_(bias, 0)`. No layer in `RNDModel` overrides the default gain. PyTorch's `nn.LeakyReLU` default negative slope is 0.01.

### 4.3 The policy and value network (`Agent`), for completeness

| Component | Spec | Orthogonal gain |
| --- | --- | --- |
| `network` conv 1 | `Conv2d(4, 32, 8, stride 4)` + ReLU | $\sqrt{2}$ |
| `network` conv 2 | `Conv2d(32, 64, 4, stride 2)` + ReLU | $\sqrt{2}$ |
| `network` conv 3 | `Conv2d(64, 64, 3, stride 1)` + ReLU | $\sqrt{2}$ |
| `network` fc 1 | `Linear(3136, 256)` + ReLU | $\sqrt{2}$ |
| `network` fc 2 | `Linear(256, 448)` + ReLU | $\sqrt{2}$ |
| `extra_layer` | `Linear(448, 448)` + ReLU | 0.1 |
| `actor` fc 1 | `Linear(448, 448)` + ReLU | 0.01 |
| `actor` fc 2 | `Linear(448, 18)` | 0.01 |
| `critic_ext` | `Linear(448, 1)` | 0.01 |
| `critic_int` | `Linear(448, 1)` | 0.01 |

Input scaling: `hidden = network(x / 255.0)`. Value heads are applied to `extra_layer(hidden) + hidden` (a residual). The actor is applied to `hidden` directly, with **no** residual.

### 4.4 Loss, reduction, and the update-proportion mask

Per minibatch (size 4096), with `mb` the minibatch index set:

1. `predict, target = rnd_model(rnd_next_obs[mb])`.
2. `forward_loss = F.mse_loss(predict, target.detach(), reduction="none").mean(-1)` — a per-sample mean over the 512 output dimensions, giving a vector of length 4096.
3. `mask = (torch.rand(4096) < update_proportion).float()` — a fresh Bernoulli(0.25) draw **per minibatch per epoch**.
4. `forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), tensor([1.0]))` — mean over the kept samples, with a guard against an all-zero mask.

The total loss is

`loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef + forward_loss`

with `v_loss = ext_v_loss + int_v_loss`. The distillation term carries **no coefficient of its own** and is **not** multiplied by `vf_coef`.

Optimizer: a single `Adam` over `list(agent.parameters()) + list(rnd_model.predictor.parameters())`, `lr = 1e-4`, `eps = 1e-5`. Linear annealing of the learning rate: `lr_now = (1 - (update - 1) / num_updates) * 1e-4`, applied to `param_groups[0]` (there is exactly one group). Gradient clipping: `nn.utils.clip_grad_norm_(combined_parameters, 0.5)` — one global norm over the policy and the predictor together.

### 4.5 The intrinsic reward and its normalisation

Rollout-time bonus, per environment copy, per step:

$r_t = \tfrac{1}{2}\,\lVert f(\tilde s_{t+1}) - \hat f(\tilde s_{t+1}) \rVert_2^2$

where $\tilde s$ is the normalised, clipped single frame of §4.1. In code: `((target - predict).pow(2).sum(1) / 2).data` — a **sum** over the 512 dimensions, halved. (The loss in §4.4 uses a **mean** over the same 512 dimensions, so the reward is $256\times$ the per-sample loss.)

Normalisation, once per policy iteration, in three steps:

1. **Forward filter.** `RewardForwardFilter(int_gamma=0.99)` holds a persistent accumulator `rewems` across iterations and applies `rewems = rewems * 0.99 + rews`.
2. **Running statistics.** `reward_rms` is a scalar-shaped `RunningMeanStd` (mean 0, variance 1, count $10^{-4}$ at start), updated via `update_from_moments(mean, std**2, count)` computed over the filter output.
3. **Division.** `curiosity_rewards /= np.sqrt(reward_rms.var)`.

Two points that matter for a faithful comparison:

- **It divides by the standard deviation of the discounted intrinsic return, not of the raw bonus, and it does not subtract a mean.** This is what the paper prescribes ("we normalized the intrinsic reward by dividing it by a running estimate of the standard deviations of the intrinsic returns").
- **The filter is applied along the wrong axis relative to the original code.** `curiosity_rewards` has shape `(num_steps, num_envs)` = (128, 128), and the code iterates `for reward_per_step in curiosity_rewards.cpu().data.numpy().T`, i.e. over the **environment** axis, calling `update` 128 times with length-128 vectors that index *time*. So the geometric accumulation runs across environment index, not across time. The original stores `buf_rews_int` as `(nenvs, nsteps)` and iterates `buf_rews_int.T`, which walks the **time** axis. I verified the difference numerically on a 4x3 example. Consequently the count fed to `update_from_moments` is also `num_envs = 128` rather than the $128 \times 128 = 16{,}384$ elements actually summarised (the original calls `rff_rms_int.update(rffs_int.ravel())`, which uses all 16,384). Because the array happens to be square at the default settings, no shape error surfaces. This is my reading of the code, not a documented CleanRL statement.

### 4.6 Two value heads, non-episodic intrinsic return, and advantage combination

1. Two separate scalar heads, `critic_ext` and `critic_int`, both on `extra_layer(hidden) + hidden`.
2. Extrinsic GAE uses `gamma = 0.999` and `ext_nextnonterminal = 1.0 - dones[t+1]` (and `1.0 - next_done` at the boundary).
3. Intrinsic GAE uses `int_gamma = 0.99` and **`int_nextnonterminal = 1.0` unconditionally**, in both branches of the `t == num_steps - 1` test. The intrinsic advantage accumulator `int_lastgaelam` is therefore never reset at an episode boundary — this is the non-episodic intrinsic return of the paper.
4. `gae_lambda = 0.95` is shared by both streams.
5. Returns: `ext_returns = ext_advantages + ext_values`, `int_returns = int_advantages + int_values`.
6. Combination: `b_advantages = b_int_advantages * int_coef + b_ext_advantages * ext_coef` with `int_coef = 1.0`, `ext_coef = 2.0`. This single combined advantage drives the clipped surrogate.
7. `norm_adv = True` standardises the **combined** advantage within each minibatch: `(a - a.mean()) / (a.std() + 1e-8)`.
8. Value losses: the extrinsic head uses the clipped form (`clip_vloss = True`, clip range `clip_coef = 0.1`), `ext_v_loss = 0.5 * max(unclipped, clipped).mean()`. The intrinsic head is **always** unclipped, `int_v_loss = 0.5 * ((new_int_values - b_int_returns) ** 2).mean()`. Both are then scaled by `vf_coef = 0.5`, so each contributes $0.25 \times$ mean squared error to the total loss.

### 4.7 Observation-normalisation warm-up

```python
for step in range(args.num_steps * args.num_iterations_obs_norm_init):   # 128 * 50 = 6400
    acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
    s, r, d, _ = envs.step(acs)
    next_ob += s[:, 3, :, :].reshape([-1, 1, 84, 84]).tolist()
    if len(next_ob) % (args.num_steps * args.num_envs) == 0:              # every 16384 frames
        obs_rms.update(np.stack(next_ob)); next_ob = []
```

Uniform random actions over the 18-action set; 6,400 steps across 128 envs; exactly 50 `obs_rms` updates. These steps are **not** counted in `global_step` and their rewards are discarded. The docs state the value 50 comes from `run_atari.py#L69` in the original repo, which is `agent.collect_random_statistics(num_timesteps=128*50)` — the same 6,400.

### 4.8 Smaller implementation facts worth pinning down

1. **The predictor's training batch is `b_obs`, not the next observations.** `rnd_next_obs` at update time is built from `b_obs[:, 3, :, :]`, i.e. $s_0,\dots,s_{T-1}$ from the rollout buffer, whereas the rollout bonus was computed on $s_1,\dots,s_T$. The variable name says "next_obs" but the contents are the current observations. Within one rollout, 127 of the 128 per-environment frames overlap, so the effect is small, but it is a genuine one-step offset relative to the original, which trains on `ph_ob[:, 1:]` — the same tensor the bonus is computed from.
2. **`next_obs` is stale when training starts.** `next_obs = torch.Tensor(envs.reset())` is taken *before* the 6,400-step warm-up loop and is never refreshed afterwards, while envpool's internal state has advanced. The first stored observation of the first rollout therefore does not correspond to the environment state that the first action is applied to.
3. **The rollout bonus is computed outside `torch.no_grad()`.** Lines computing `target_next_feature` / `predict_next_feature` sit after the `with torch.no_grad():` block; the result is detached with `.data`. Correct, but it builds and discards an autograd graph every step.
4. **`RecordEpisodeStatistics`** accumulates `infos["reward"]`, the **unclipped** reward, and resets the accumulator on `infos["terminated"]` (true ALE game over), not on life loss. Logging fires when `done and info["lives"][idx] == 0`. `charts/avg_episodic_return` is a mean over a `deque(maxlen=20)`.
5. **Minibatching** shuffles a flat index array over the whole $16{,}384$-element batch, so a minibatch mixes time steps and environment copies freely.

---

## 5. Where CleanRL deviates from Burda et al. (2019)

### 5.1 What the CleanRL documentation itself says

`docs/rl-algorithms/ppo-rnd.md` lists only two RND-specific implementation details, both framed as *agreements* with the original, not deviations:

1. "We initialize the normalization parameters by stepping a random agent in the environment by `args.num_steps * args.num_iterations_obs_norm_init`. `args.num_iterations_obs_norm_init=50` comes from the original implementation."
2. "We use sticky action from envpool to facilitate the exploration like done in the original implementation."

The only other caveat in the page is the envpool-versus-gym off-by-one on the terminal observation (envpool 0.6.4 and earlier return $s_\text{last}$ where gym returns $s_\text{new}$), which the docs describe as not appearing to affect performance. The docs otherwise say the file "has the same other implementation details as `ppo_atari.py`", i.e. the nine Atari details of *The 37 Implementation Details of PPO* — which is itself a statement of alignment with `openai/baselines`, not with the RND paper. **The documentation does not enumerate any deviation from Burda et al.** Everything in §5.2 and §5.3 is my own line-by-line comparison against the paper and the original code, not a CleanRL claim.

### 5.2 Environment and preprocessing deviations

| Knob | Burda et al. 2019 (paper Table 2 / original code) | CleanRL + envpool 0.6.6 |
| --- | --- | --- |
| Terminal on loss of life | **False** (Table 2). No `EpisodicLifeEnv` in `atari_wrappers.py`. | **True** (`episodic_life=True` is passed explicitly) |
| Max frames per episode | **18K frames** = 4,500 agent steps (`run_atari.py` default `max_episode_steps=4500`; `make_atari` sets `env._max_episode_steps = 4500*4`) | **432K frames** = 108,000 agent steps (envpool registration), effectively non-binding |
| Random starts (no-op starts) | **False** (Table 2). No `NoopResetEnv`. | 0 to 29 no-ops on every reset (`noop_max=30`), including on life-loss resets |
| FIRE on reset | Not used | Applied unconditionally after the no-ops, because FIRE is in Montezuma's minimal action set |
| Sticky action probability | 0.25, via a `StickyActionEnv` wrapper placed **before** the frame-skip wrapper | 0.25, via ALE's own `repeat_action_probability` |
| Max-and-skip | 4, max over the last two frames | 4, max over the last two frames — same |
| Grayscale and resize | 84x84, `cv2.INTER_AREA` | 84x84, `INTER_AREA` — same |
| Extrinsic reward clipping | $[-1,1]$ by sign | sign — same |
| Intrinsic reward clipping | False | none — same |
| Frames stacked, policy | 4 | 4 — same |
| Frames stacked, RND | 1 (the last channel) | 1 (channel index 3) — same |
| Policy observation normalisation | $x \mapsto x/255$ | same |
| RND observation normalisation | $x \mapsto \mathrm{clip}((x-\mu)/\sigma, [-5,5])$ | same |
| Base env id | `MontezumaRevengeNoFrameskip-v4` | `MontezumaRevenge-v5` in envpool, which envpool documents as following the `NoFrameskip-v4` ALE settings; the sticky actions come from the explicit argument, not from the `-v5` label |

### 5.3 Algorithm deviations

Values in the "Burda et al." column are from paper Table 5 where the paper states them, and otherwise from the original code at the linked commit (marked "code").

| Knob | Burda et al. 2019 | CleanRL | Same? |
| --- | --- | --- | --- |
| Rollout length | 128 | 128 | yes |
| Parallel environments | 128 | 128 | yes |
| Minibatches per epoch | 4 | 4 | yes |
| Optimization epochs | 4 | 4 | yes |
| Learning rate | 0.0001 | 0.0001 | yes |
| Learning-rate schedule | constant (code: `ph_lr` fed `self.lr` every step) | linear anneal to 0 over 122,070 updates | **no** |
| Optimizer | Adam | Adam | yes |
| Adam epsilon | $10^{-8}$ (code: TF `AdamOptimizer` default) | $10^{-5}$ | **no** |
| Clip range | $[0.9, 1.1]$, i.e. 0.1 | 0.1 | yes |
| Entropy coefficient | 0.001 | 0.001 | yes |
| GAE $\lambda$ | 0.95 (both streams) | 0.95 (both streams) | yes |
| Extrinsic discount | 0.999 | 0.999 | yes |
| Intrinsic discount | 0.99 | 0.99 | yes |
| Extrinsic advantage coefficient | 2 | 2 | yes |
| Intrinsic advantage coefficient | 1 | 1 | yes |
| Predictor keep probability | 0.25 at 128 envs (Table 5; the code's argparse default is 1.0 and 0.25 is passed for the 128-env runs) | 0.25 | yes |
| Non-episodic intrinsic return | yes (code: `use_news=0`, `nextnew = 0.0`) | yes (`int_nextnonterminal = 1.0`) | yes |
| Two value heads | yes | yes | yes |
| Advantage normalisation | **none** (code: `buf_advs` fed unnormalised) | per-minibatch standardisation of the combined advantage | **no** |
| Value-loss clipping | **none** (code: plain squared error on both heads) | clipped on the extrinsic head, unclipped on the intrinsic head | **no** |
| Value-loss weight | code: `vf_coef=1.0`, giving $0.5\times$ squared error per head | `vf_coef=0.5` on two $0.5\times$ terms, giving $0.25\times$ per head | **no** — half the original weight |
| Gradient clipping | **none in effect** (code: `max_grad_norm=0.0` from `run_atari.py`; and even when non-zero the clipped tensors are discarded rather than applied) | global norm 0.5 over policy and predictor jointly | **no** |
| Minibatch construction | contiguous blocks of **whole environments** (`envsperbatch = nenvs // nminibatches`), no shuffling, trajectories kept intact for the RNN | flat shuffle over all 16,384 (step, env) pairs | **no** |
| Distillation loss reduction | mean over 512 dims, then the same mask-and-renormalise | identical | yes |
| Intrinsic reward magnitude | mean over 512 dims | half the sum over 512 dims, i.e. $256\times$ larger | **no** in raw scale; largely cancelled by the running-standard-deviation division |
| Forward-filter axis | over time | over environment index (see §4.5) | **no** |
| Running-statistics sample count | all 16,384 filter values | 128 | **no** |
| RND target: conv stack | 32@8s4, 64@4s2, 64@3s1, all leaky ReLU, then one `fc` to 512 | identical | yes |
| RND predictor: head | 512, 512, 512 with ReLU between | identical | yes |
| RND layer initialisation | orthogonal, gain $\sqrt{2}$, on every layer | orthogonal, gain $\sqrt{2}$, on every layer | yes |
| Leaky-ReLU negative slope | 0.2 (TensorFlow `tf.nn.leaky_relu` default) | 0.01 (PyTorch `nn.LeakyReLU` default) | **no** |
| Predictor optimisation | same Adam step as PPO, `aux_loss` added to the total loss with coefficient 1 | same | yes |
| Predictor training input | the next observations, the same tensor the bonus came from | the rollout's own observations, one step earlier (§4.8 item 1) | **no** |
| Policy conv layer 3 | kernel **4**, stride 1, so 2304 flattened features (code, `apply_policy`) | kernel 3, stride 1, so 3136 flattened features | **no** |
| Policy trunk widths | 256 then 448 | 256 then 448 | yes |
| Extra 448-unit layer, value path | residual, `X + relu(fc)`, gain 0.1 | residual, `extra_layer(hidden) + hidden`, gain 0.1 | yes |
| Extra 448-unit layer, actor path | residual, `X + relu(fc)`, gain **0.1** | **non-residual**, gain **0.01** | **no** |
| Actor and critic output gains | 0.01 | 0.01 | yes |
| Policy family used for the headline result | **RNN (GRU)** for Figure 7 and Table 1 (the 8,152 figure); CNN results are in Figure 9 | **CNN only** | **no** |
| Training budget | 30K rollouts per environment, 1.97 billion frames (about 492M agent steps) | 2.0 billion agent steps, i.e. about 8.0 billion frames | **no** — roughly $4\times$ longer |

The last row deserves care in the writeup. The CleanRL comparison table pairs its own number against "(Burda et al., 2019, Figure 7) 2000M steps", but the paper's 1.97 billion is a **frame** count, and its Figure 7 / Table 1 number of 8,152 is the **RNN** policy. CleanRL's x-axis unit is environment steps, so its configured budget corresponds to about 8 billion frames with a CNN policy. Any statement of "CleanRL reaches 7100 versus the paper's 8152" should carry both qualifications.

---

## 6. The paper's own tables, reproduced for the writeup

**Table 2 — environment preprocessing (all experiments).** Grey-scaling True; observation downsampling (84,84); extrinsic reward clipping $[-1,1]$; intrinsic reward clipping False; max frames per episode 18K; terminal on loss of life False; max and skip frames 4; random starts False; sticky action probability 0.25.

**Table 3 — policy and value network preprocessing.** Frames stacked 4; observation normalization $x \mapsto x/255$.

**Table 4 — target and predictor network preprocessing.** Frames stacked 1; observation normalization $x \mapsto \mathrm{CLIP}((x-\mu)/\sigma, [-5,5])$.

**Table 5 — PPO and RND hyperparameters.** Rollout length 128; total number of rollouts per environment 30K; number of minibatches 4; number of optimization epochs 4; coefficient of extrinsic reward 2; coefficient of intrinsic reward 1; number of parallel environments 128; learning rate 0.0001; optimization algorithm Adam; $\lambda$ 0.95; entropy coefficient 0.001; proportion of experience used for training predictor 0.25; $\gamma_E$ 0.999; $\gamma_I$ 0.99; clip range $[0.9, 1.1]$; policy architecture CNN.

**Table 1 — final mean performance, Montezuma's Revenge.** RND 8,152; PPO 2,497; Dynamics 400; prior state of the art 3,700 (Bellemare et al. 2016); average human 4,753. The paper states these are RNN-policy results (Figure 7 caption; CNN results are in Figure 9).

**Paper abstract, verbatim.** "We introduce an exploration bonus for deep reinforcement learning methods that is easy to implement and adds minimal overhead to the computation performed. The bonus is the error of a neural network predicting features of the observations given by a fixed randomly initialized neural network. We also introduce a method to flexibly combine intrinsic and extrinsic rewards. We find that the random network distillation (RND) bonus combined with this increased flexibility enables significant progress on several hard exploration Atari games. In particular we establish state of the art performance on Montezuma's Revenge, a game famously difficult for deep reinforcement learning methods. To the best of our knowledge, this is the first method that achieves better than average human performance on this game without using demonstrations or having access to the underlying state of the game, and occasionally completes the first level."

Bibliographic record, for the `.bib` entry: Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov, "Exploration by Random Network Distillation", Seventh International Conference on Learning Representations (ICLR), 2019. (The arXiv posting is 1810.12894, submitted 30 October 2018. Per the project bib rule, the entry itself must be built from a single source of record — DBLP's ICLR 2019 record — and not from this summary.)

**Paper text supporting the design choices**, for quoting in the writeup:

1. Non-episodic intrinsic return: "In preliminary experiments that used only intrinsic rewards, treating the problem as non-episodic resulted in better exploration. In that setting the return is not truncated at 'game over'."
2. Two value heads: "the return is linear in the rewards and so can be decomposed as a sum $R = R_E + R_I$ ... Hence we can fit two value heads $V_E$ and $V_I$ separately using their respective returns, and combine them to give the value function $V = V_E + V_I$."
3. Intrinsic reward normalisation: "In order to keep the rewards on a consistent scale we normalized the intrinsic reward by dividing it by a running estimate of the standard deviations of the intrinsic returns."
4. Observation normalisation: "we whiten each dimension by subtracting the running mean and then dividing by the running standard deviation. We then clip the normalized observations to be between -5 and 5. We initialize the normalization parameters by stepping a random agent in the environment for a small number of steps before beginning optimization. We use the same observation normalization for both predictor and target networks but not the policy network."
5. Why the keep probability is 0.25: "To avoid this situation when scaling up from 32 to 128 environments we kept the effective batch size for the predictor network the same by randomly dropping out elements of the batch with keep probability 0.25."

---

## 7. Files written during this investigation

All under `/tmp/claude-2618919/-p-rlprojects-RND/12f4bbb4-d71e-4c5d-8c09-aae9a5c45be7/scratchpad/`:

- `atari_env_v066.h` — envpool 0.6.6 Atari C++ environment, 305 lines, the authoritative source for §3.2 and §3.4.
- `rnd_ar5iv.html`, `rnd_text.txt` — the RND paper rendered and flattened to text; the appendix tables in §6 come from `rnd_text.txt` lines 480 to 555.
- `orig_ppo_agent.py`, `orig_run_atari.py`, `orig_policies_cnn_policy_param_matched.py`, `orig_cmd_util.py`, `orig_atari_wrappers.py` — the original RND implementation at commit `f75c0f1`, the source for every "code" entry in §5.3.
- `envpool_cfg.txt` — the live `envs.config`, spaces, dtypes, and info keys dumped from an env built with CleanRL's exact call.