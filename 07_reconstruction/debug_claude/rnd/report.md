# RND Debug Report: Why RND Stops Working Between a558f7c and 2ff924b

## Summary

Three issues were found between the working commit (`a558f7c`, `03_rnd_my_implementation.py` + `03_wandb_sweep.yaml`) and the broken commit (`2ff924b`, `04_many_exploration_method.py` + `04_wandb_sweep.yaml`). **Issue #1 is critical and almost certainly the primary cause of failure.**

---

## Issue #1: `TerminateOnTimeLimitWrapper` Breaks SB3 Timeout Handling (CRITICAL)

### What changed

The new code adds `TerminateOnTimeLimitWrapper` to both training and eval environments:

**Old code (`03_rnd_my_implementation.py`) — NO wrapper:**
```python
base_env = gym.make(
    args.env_name,
    continuing_task=True,
    reset_target=False,
    max_episode_steps=args.env_max_episode,
)
# (no TerminateOnTimeLimitWrapper)
if args.goal_position == "top_left":
    fixed_goal_cell = select_fixed_goal_top_left(base_env)
```

**New code (`04_many_exploration_method.py`) — wrapper ADDED:**
```python
base_env = gym.make(
    args.env_name,
    continuing_task=True,
    reset_target=False,
    max_episode_steps=args.env_max_episode,
)
base_env = TerminateOnTimeLimitWrapper(base_env)   # <--- THIS BREAKS SAC
if args.goal_position == "top_left":
    fixed_goal_cell = select_fixed_goal_top_left(base_env)
```

Same for eval:
```python
eval_base = gym.make(...)
eval_base = TerminateOnTimeLimitWrapper(eval_base)  # <--- ALSO ADDED HERE
```

### Why it breaks

The wrapper is defined as:
```python
class TerminateOnTimeLimitWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if truncated and not terminated:
            return obs, reward, True, False, info  # terminated=True, truncated=False
        return obs, reward, terminated, truncated, info
```

With `continuing_task=True`, the env **never** sets `terminated=True` — every episode ends only via the TimeLimit wrapper (`truncated=True`). The wrapper converts ALL of these to `terminated=True, truncated=False`.

SB3's `DummyVecEnv.step_wait()` handles timeout detection:
```python
# Inside DummyVecEnv.step_wait():
if truncated and not terminated:
    self.buf_infos[env_idx]["TimeLimit.truncated"] = True
```

With `TerminateOnTimeLimitWrapper`, the DummyVecEnv sees `terminated=True, truncated=False`, so it **never** sets `info["TimeLimit.truncated"] = True`.

SB3's `ReplayBuffer._get_samples()` uses this for done correction:
```python
dones = self.dones[batch_inds] * (1 - self.timeouts[batch_inds])
```

Without the timeout flag, `dones=1` for ALL episode ends. SAC's critic target becomes:
```
target_Q = reward + gamma * (1 - done) * V(s')
         = reward + 0.999 * (1 - 1) * V(s')
         = reward                              # NO BOOTSTRAPPING
```

### Impact

- With `gamma=0.999` and `max_episode_steps=400`, `gamma^400 ≈ 0.67` — **33% of future value is lost**
- But more importantly, SAC thinks every episode truly ends, making the value function unable to plan beyond the 400-step horizon
- This is especially harmful for intrinsic rewards (RND), where exploration value extends across episodes
- The Q-function is systematically biased downward for states near the end of episodes

### Fix

Remove `TerminateOnTimeLimitWrapper` from both training and eval environments in `04_many_exploration_method.py`:

```python
# DELETE these two lines:
# base_env = TerminateOnTimeLimitWrapper(base_env)     # line ~131
# eval_base = TerminateOnTimeLimitWrapper(eval_base)    # line ~207
```

---

## Issue #2: RND `update()` No Longer Uses DataLoader Mini-Batching (MEDIUM)

### What changed

**Old code (`intrinsic/my_rnd/rnd.py`) MyRND.update() — DataLoader loop:**
```python
def update(self, samples):
    obs = samples["observations"].to(self.device).float()
    obs = self._slice_obs(obs)
    if self.use_obs_norm and self.obs_rms is not None:
        self.obs_rms.update(obs.detach().cpu().numpy())
    obs = self._normalize_obs(obs)
    dataset = TensorDataset(obs)
    loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
    for batch in loader:
        o = batch[0]
        self.opt.zero_grad()
        src = self.predictor(o)
        with torch.no_grad():
            tgt = self.target(o)
        distances = self._dist_ensemble(src, tgt)
        loss = distances.mean()
        loss.backward()
        self.opt.step()
```

**New code (`intrinsic/intrinsic_method/rnd.py`) RND.update() — single step:**
```python
def update(self, samples):
    x = self._get_feature_tensor(samples)
    if self.use_obs_norm and self.obs_rms is not None:
        self.obs_rms.update(x.detach().cpu().numpy())
    x = self._normalize_obs(x)
    self.opt.zero_grad()
    src = self.predictor(x)
    if self.n_predictors == 1:
        src = src.unsqueeze(1)
    with torch.no_grad():
        tgt = self.target(x)
    loss = self._dist_ensemble(src, tgt).mean()
    loss.backward()
    self.opt.step()
```

### Impact

With default settings (SB3 `batch_size=256`, RND `batch_size=256`), both do **1 gradient step** per `sample()` call, so this is equivalent. Only matters if the internal batch sizes differ.

**This is likely NOT the cause of failure at default settings.**

---

## Issue #3: Old RND Trains on `observations`, New Trains on `next_observations` (LOW)

### What changed

**Old MyRND.update():**
```python
obs = samples["observations"]     # trains predictor on CURRENT states
```

**New RND.update() with feature="rnd_next_state":**
```python
x = self._get_feature_tensor(samples)
# _get_feature_tensor for "rnd_next_state" returns samples["next_observations"]
```

### Impact

Both `observations` and `next_observations` come from the same distribution in the replay buffer (they're consecutive states). The predictor learns the state distribution in either case. **This is unlikely to cause failure.**

---

## Sweep YAML Parameter Differences

| Parameter | Old (`03_wandb_sweep.yaml`) | New (`04_wandb_sweep.yaml`) | Impact |
|-----------|---------------------------|---------------------------|--------|
| `device` | `cuda` | `cpu` | Speed only, not correctness |
| `beta` | `[0.0, 0.1, 1, 10, 100]` | `[0.1, 1, 10, 100]` | No beta=0 baseline |
| `rnd_distance` | `["mse", "abs"]` | `["mse"]` | Fewer configs |
| `rnd_input` / equivalent | `["position", "all"]` | via `algorithm` | Covered by algorithm list |
| `rnd_output_dim` | `[128, 640]` | `[128]` | Fewer configs |
| `n_predictors` | `[1, 10]` | `1` | Fixed at 1 |
| `beta_std` | `[1.0, 5.0, 10.0]` | not set (default 0.0) | No ensemble std bonus |

None of these parameter changes should cause RND to completely stop working. The `device: cpu` change only affects speed.

---

## Conclusion

**Root cause: `TerminateOnTimeLimitWrapper`** (Issue #1).

This wrapper silently breaks SB3's timeout handling, causing SAC to treat every episode timeout as a true termination. The agent's value function can no longer plan beyond the episode boundary, which is especially damaging for exploration with intrinsic rewards.

### Recommended fix priority:
1. **Remove `TerminateOnTimeLimitWrapper`** from `04_many_exploration_method.py` (both train and eval envs)
2. (Optional) Restore DataLoader mini-batching in `RND.update()` if you want exact parity with old behavior
3. No action needed for the `observations` vs `next_observations` training data difference

### Test scripts
- `debug_claude/rnd/code/test_terminate_wrapper_impact.py` — demonstrates the broken timeout handling
- `debug_claude/rnd/code/test_rnd_compute_equivalence.py` — verifies forward-pass equivalence
- `debug_claude/rnd/code/test_update_training_data.py` — documents the training data difference
- `debug_claude/rnd/code/fix_04_many_exploration_method.py` — shows exact code changes needed
