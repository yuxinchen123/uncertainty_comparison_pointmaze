# Intrinsic Reward, Environment Integration, and Wrappers

This document describes what the intrinsic reward is, how it is added to the environments, and what wrappers are applied in `06rl_integration`.

---

## 1. What Is the Intrinsic Reward?

The **intrinsic reward** is an exploration bonus that encourages the agent to visit less-visited states. It is produced by an **uncertainty method** and added to the environment’s **extrinsic reward** (e.g. goal-reaching) so that the policy is trained on **total reward** = extrinsic + β × intrinsic.

### 1.1 Ground-truth intrinsic (GT baseline)

When `use_gt_baseline=True`, the intrinsic reward uses **ground-truth uncertainty**:

- **Formula:** \( r_{\text{intrinsic}} = 1 / \sqrt{N(i)} \), where \( N(i) \) is the **visit count** of the grid cell \( i \) (for the current goal in multi-goal mode).
- **Meaning:** Unvisited cells get maximum bonus (capped at 1.0); frequently visited cells get a smaller bonus.
- **Implementation:** `uncertainty/gt_intrinsic.py` — `GTIntrinsicReward` with `get_uncertainty(coordinates)` and `update_visit_counts(visit_counts)`.

So in this setup, “intrinsic reward” is exactly this count-based exploration bonus.

### 1.2 Other uncertainty methods (RND, ensembles, etc.)

When `use_gt_baseline=False`, the intrinsic reward comes from a configurable **uncertainty method** (e.g. RND, ensemble RND, scalar ensemble methods from folders 01–05). Those methods are wrapped by `UncertaintyMethodAdapter` in `uncertainty/integration.py` and expose:

- **`get_uncertainty(state)`** — returns uncertainty (used as intrinsic reward) for given state(s).
- **`update_with_batch(states)`** (optional) — online updates from visited states.

The wrapper treats the method’s **uncertainty output as the intrinsic reward** (after mapping state to grid/position as required).

---

## 2. How Is the Intrinsic Reward Added to the Environment?

The intrinsic reward is **not** added inside the base PointMaze env. It is added by a **wrapper** that sits around the base env and modifies the reward in `step()`.

### 2.1 Where it happens

- **File:** `wrappers/intrinsic_reward_wrapper.py`
- **Class:** `IntrinsicRewardWrapper(gym.Wrapper)`

In **`step(action)`** the wrapper:

1. Calls **`self.env.step(action)`** and gets `obs`, **`reward_extrinsic`**, `terminated`, `truncated`, `info`.
2. Maps the **new** observation to a grid cell and updates **visit counts** for that cell (and optionally feeds state to the uncertainty method for online updates).
3. Computes **intrinsic reward** by calling **`_get_uncertainty(obs)`**, which:
   - Syncs the uncertainty method’s visit counts with the wrapper’s (for GT or visit-count–based methods).
   - Maps `obs` to position (e.g. `achieved_goal` or array) and calls **`uncertainty_method.get_uncertainty(position_array)`**.
   - Returns a scalar; NaN/inf are turned into 0.
4. Combines rewards:
   - **`reward_total = reward_extrinsic + self.beta * intrinsic_reward`**
5. Fills **`info`** with `'intrinsic_reward'`, `'extrinsic_reward'`, `'total_reward'`.
6. Returns **`(obs, reward_total, terminated, truncated, info)`**.

So the **agent and the replay buffer** always see **total reward**; the wrapper is the only place that adds intrinsic to extrinsic.

### 2.2 Replay buffer and intrinsic reward

For **off-policy algorithms (e.g. SAC)**:

- Transitions are stored with the **total reward** and **info** from the wrapper.
- **`IntrinsicReplayBuffer`** / **`DictIntrinsicReplayBuffer`** store **extrinsic** (from `info['extrinsic_reward']`) and, when **sampling**, **recompute intrinsic reward** via a callback that uses the **current** visit counts (or current uncertainty model). So training uses up-to-date intrinsic rewards even though past transitions were collected with older counts/model.

---

## 3. What Wrappers Are Applied on the Environments?

Wrappers are applied in a fixed order. “Inner” means closer to the base env, “outer” means closer to the RL code (SB3).

### 3.1 Training environment

Built in **`train_rl.py`** in **`train(config)`**:

1. **Base env:** PointMaze via **`make_pointmaze_env(config.env_name, ...)`** (e.g. `PointMaze_Large-v3`), with `continuing_task=False` so the episode ends when the goal is reached.

2. **GoalWrapper** (`wrappers/goal_wrapper.py`):
   - **Single-goal:** fixed goal cell; observation space is modified to **remove** `desired_goal`.
   - **Multi-goal:** list of goal cells; at reset a goal is sampled and `desired_goal` stays in the observation.
   - Ensures the same goal(s) and observation format as used in the rest of the pipeline.

3. **IntrinsicRewardWrapper** (`wrappers/intrinsic_reward_wrapper.py`):
   - **Input:** env already wrapped by **GoalWrapper**.
   - Holds **uncertainty_method** (e.g. `GTIntrinsicReward` or an adapter from `uncertainty/integration.py`), **beta**, grid size, maze map, goal mode.
   - In **step()** adds intrinsic reward and returns **total reward**; updates visit counts and optionally the uncertainty model.
   - **Output:** env that returns **total reward** and **info** with extrinsic/intrinsic/total.

4. **Monitor** (SB3 `stable_baselines3.common.monitor.Monitor`):
   - Wraps the single env for logging (e.g. episode returns, lengths) to a log directory.
   - Applied in `train_rl.py`: **`train_env = Monitor(train_env, config.log_dir)`**.

5. **DummyVecEnv** (SB3 `stable_baselines3.common.vec_env.DummyVecEnv`):
   - Turns the single (wrapped) env into a **VecEnv** of one environment so SB3’s API works.
   - Applied: **`train_env = DummyVecEnv([lambda: train_env])`**.

So the **training** stack (outer → inner) is:

```text
DummyVecEnv → Monitor → IntrinsicRewardWrapper → GoalWrapper → PointMaze (base)
```

Comment in code: *“DummyVecEnv -> Monitor -> IntrinsicRewardWrapper -> env”* (where the innermost “env” is GoalWrapper → base).

### 3.2 Evaluation environment

Built in **`create_eval_env(config, continuing_task=..., track_visit_counts=...)`** in `train_rl.py`:

1. **Base env:** PointMaze via **`make_pointmaze_env(...)`** (e.g. with `continuing_task=True` for “continuation” eval).

2. **GoalWrapper:** Same goal mode and goal(s) as training (same seed/config).

3. **IntrinsicRewardWrapper** (optional):
   - Only if **`track_visit_counts=True`** (e.g. for visit-count heatmaps).
   - Uses a **GT method with `beta=0`**, so **no intrinsic reward** is added to the eval reward; only visit counts are tracked for visualization.

4. **Monitor** and **DummyVecEnv** are then applied in **`train()`** to this eval env:  
   **`eval_env = Monitor(eval_env, os.path.join(config.log_dir, 'eval'))`**, then **`eval_env = DummyVecEnv([lambda: eval_env])`**.

So the **evaluation** stack (when visit counts are tracked) is:

```text
DummyVecEnv → Monitor → [IntrinsicRewardWrapper (beta=0)] → GoalWrapper → PointMaze (base)
```

If **`track_visit_counts=False`**, there is no **IntrinsicRewardWrapper** in the eval env; rewards are purely extrinsic (plus whatever the base env returns).

### 3.3 Summary table

| Layer (outer → inner) | Training | Eval (default) | Eval (track_visit_counts=True) |
|------------------------|----------|----------------|---------------------------------|
| DummyVecEnv            | ✓        | ✓              | ✓                               |
| Monitor                | ✓        | ✓              | ✓                               |
| IntrinsicRewardWrapper | ✓ (beta from config) | ✗   | ✓ (beta=0, visit counts only)   |
| GoalWrapper            | ✓        | ✓              | ✓                               |
| PointMaze (base)       | ✓        | ✓              | ✓                               |

---

## 4. How the Monitor captures reward (and how intrinsic is captured for logging)

The **Monitor** (SB3) does **not** separately record intrinsic vs extrinsic reward. It only sees and stores the **single scalar reward** returned by the inner env each step — which in our setup is **total reward** from `IntrinsicRewardWrapper`. Intrinsic reward is “captured” for logging by the **wrapper** (info + episode stats) and the **WandBLoggingCallback** reading from the wrapper. Below is the relevant code.

### 4.1 Where Monitor is applied

Monitor wraps the training (and eval) env **after** `IntrinsicRewardWrapper` and **before** `DummyVecEnv`. So when the vec env calls `step()`, the flow is: **DummyVecEnv** → **Monitor** → **IntrinsicRewardWrapper** → … → base env. Monitor’s inner env is the one that returns `reward_total`.

```python
# 06rl_integration/train_rl.py (inside train())

# train_env at this point: IntrinsicRewardWrapper → GoalWrapper → PointMaze
train_env = Monitor(train_env, config.log_dir)
train_env = DummyVecEnv([lambda: train_env])
```

### 4.2 What Monitor receives and stores (SB3 behavior)

SB3’s `Monitor` wraps an env and in `step()` does roughly:

- Call `obs, reward, terminated, truncated, info = self.env.step(action)`.
- Accumulate `reward` into the current episode return; increment current episode length.
- When `terminated or truncated`, append the episode return to `episode_returns` and length to `episode_lengths`, then reset accumulators.

So the **reward** Monitor sees each step is the **single scalar** returned by the inner env — here, **`reward_total`** from `IntrinsicRewardWrapper`. Monitor therefore “captures” **total reward** (extrinsic + β × intrinsic), not intrinsic by itself. It exposes:

- `get_episode_rewards()` / `episode_returns`: list of **total** episode returns (sum of `reward_total` per episode).
- `get_episode_lengths()` / `episode_lengths`: list of episode lengths.

### 4.3 IntrinsicRewardWrapper: returning total and exposing intrinsic

The wrapper returns **total** reward and puts **intrinsic** (and extrinsic) in `info` and in its own episode statistics. That is how “intrinsic” is available for logging.

**Step return (total reward + info):**

```python
# 06rl_integration/wrappers/intrinsic_reward_wrapper.py

def step(self, action):
    obs, reward_extrinsic, terminated, truncated, info = self.env.step(action)
    # ... grid update, visit counts, intrinsic_reward = _get_uncertainty(obs) ...
    reward_total = reward_extrinsic + self.beta * intrinsic_reward
    # ...
    info['intrinsic_reward'] = intrinsic_reward
    info['extrinsic_reward'] = reward_extrinsic
    info['total_reward'] = reward_total
    return obs, reward_total, terminated, truncated, info
```

**Per-episode stats (and last-episode after reset):**

```python
# 06rl_integration/wrappers/intrinsic_reward_wrapper.py

def reset(self, seed=None, options=None, **kwargs):
    # Store final episode rewards BEFORE clearing (for callbacks after reset)
    self.last_episode_extrinsic_reward = self.episode_extrinsic_reward
    self.last_episode_intrinsic_reward = self.episode_intrinsic_reward
    obs, info = self.env.reset(**kwargs)
    self.episode_extrinsic_reward = 0.0
    self.episode_intrinsic_reward = 0.0
    return obs, info

def get_episode_statistics(self):
    """Current (or last, if just reset) episode extrinsic/intrinsic."""
    if self.episode_extrinsic_reward == 0.0 and self.episode_intrinsic_reward == 0.0:
        return {
            'episode_extrinsic_reward': self.last_episode_extrinsic_reward,
            'episode_intrinsic_reward': self.last_episode_intrinsic_reward,
        }
    return {
        'episode_extrinsic_reward': self.episode_extrinsic_reward,
        'episode_intrinsic_reward': self.episode_intrinsic_reward,
    }
```

So: **Monitor** only ever sees and stores **total reward**. **Intrinsic** is captured by the wrapper (in `info` and in episode/last-episode stats).

### 4.4 Callback: finding Monitor and IntrinsicRewardWrapper, and reading intrinsic

The logging callback locates both wrappers in the stack and uses **Monitor** for episode count and total return (and length), and **IntrinsicRewardWrapper** for intrinsic (and extrinsic) breakdown.

**Finding Monitor and IntrinsicRewardWrapper:**

```python
# 06rl_integration/train_rl.py — WandBLoggingCallback.__init__

self.monitor = None
self.intrinsic_wrapper = None
if hasattr(train_env, 'envs') and len(train_env.envs) > 0:
    env = train_env.envs[0]
    while hasattr(env, 'env'):
        if isinstance(env, Monitor):
            self.monitor = env
        if isinstance(env, IntrinsicRewardWrapper):
            self.intrinsic_wrapper = env
        env = env.env
# Stack: DummyVecEnv -> Monitor -> IntrinsicRewardWrapper -> ...
```

**Each step: track last-seen episode intrinsic/extrinsic from the wrapper**

(Because when an episode ends, the wrapper is reset before the callback runs, the callback must have already stored the final episode extrinsic/intrinsic.)

```python
# 06rl_integration/train_rl.py — WandBLoggingCallback._on_step()

if self.intrinsic_wrapper is not None:
    episode_stats_temp = self.intrinsic_wrapper.get_episode_statistics()
    current_ep_extrinsic = episode_stats_temp.get('episode_extrinsic_reward', 0.0)
    current_ep_intrinsic = episode_stats_temp.get('episode_intrinsic_reward', 0.0)
    if current_ep_extrinsic > 0.0 or current_ep_intrinsic > 0.0:
        self.last_seen_episode_extrinsic = current_ep_extrinsic
        self.last_seen_episode_intrinsic = current_ep_intrinsic
```

**When a new episode is detected (Monitor’s count increased): use Monitor for total/length, wrapper-derived values for intrinsic/extrinsic**

```python
# 06rl_integration/train_rl.py — WandBLoggingCallback._get_monitor_stats()

episode_rewards = self.monitor.get_episode_rewards()  # total return per episode
episode_lengths = self.monitor.get_episode_lengths()
episode_count = len(episode_rewards)
if episode_count > self.last_episode_count:
    latest_length = episode_lengths[-1]
    # Intrinsic/extrinsic from wrapper (tracked before reset)
    latest_extrinsic_reward = self.last_seen_episode_extrinsic
    latest_intrinsic_reward = self.last_seen_episode_intrinsic
    latest_total_from_monitor = episode_rewards[-1]  # total from Monitor
    # Fallback: if extrinsic was 0 but Monitor has total, derive extrinsic
    if latest_extrinsic_reward == 0.0 and latest_total_from_monitor > 0.0 and latest_intrinsic_reward > 0.0:
        computed_extrinsic = latest_total_from_monitor - self.beta * latest_intrinsic_reward
        if computed_extrinsic > 0.0:
            latest_extrinsic_reward = computed_extrinsic
    latest_total_reward = latest_extrinsic_reward + self.beta * latest_intrinsic_reward
    return {
        'train/extrinsic_reward': latest_extrinsic_reward,
        'train/intrinsic_reward': latest_intrinsic_reward,
        'train/total_reward': latest_total_reward,
        'train/episode_length': latest_length,
        'train/episode_number': episode_count,
    }
```

So: **Monitor** is used to know *when* an episode ended and to get **total** return and length; **intrinsic** (and extrinsic) for logging come from **IntrinsicRewardWrapper** via `get_episode_statistics()` and the callback’s `last_seen_*` tracking.

---

## 5. Quick reference

- **Intrinsic reward:** Exploration bonus from an uncertainty method (e.g. GT: \( 1/\sqrt{N(i)} \), or RND/ensemble methods). Used as **r_intrinsic** in **total = r_extrinsic + β × r_intrinsic**.
- **Where it’s added:** Only in **`IntrinsicRewardWrapper.step()`**; the base env is unchanged.
- **Wrappers (training):** **DummyVecEnv → Monitor → IntrinsicRewardWrapper → GoalWrapper → PointMaze**.
- **Wrappers (eval):** Same, but **IntrinsicRewardWrapper** is optional and, when present for heatmaps, uses **beta=0** so eval metrics remain extrinsic-only.
