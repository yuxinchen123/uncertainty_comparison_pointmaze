# Daniel's Coding Note

---

## State

Observation data format at each stage. Example: **Dict with `observation` and `desired_goal`** (e.g. PointMaze before RemoveGoalWrapper). Shapes: state `(4,)`, goal `(2,)`; `n_envs=1`, `batch_size=256`.

### Observation space types (Gymnasium / SB3)

**Box (vector)** — Single continuous array. SB3: **MlpPolicy**.

```python
from gymnasium import spaces
observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
# obs from env: np.ndarray shape (4,) or (n_envs, 4)
obs = np.array([0.1, -0.2, 0.0, 0.5], dtype=np.float32)

# FlattenObservation: x.flatten() -> same values, shape (4,). Already 1D so no change for (4,).
from gymnasium.wrappers import FlattenObservation
env = FlattenObservation(env)  # observation_space -> Box(shape=(4,)); obs shape (4,)
```

**Dict** — Named components (each key can be Box, Discrete, etc.). SB3: **MultiInputPolicy**, **DictReplayBuffer**.

```python
observation_space = spaces.Dict({
    "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
    "desired_goal": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
})
# obs from env: dict of arrays
obs = {"observation": np.array([0.1, -0.2, 0.0, 0.5]), "desired_goal": np.array([0.5, 0.5])}

# FlattenObservation: concat flattened values in key order -> one vector, key structure lost
env = FlattenObservation(env)  # observation_space -> Box(shape=(6,)); obs shape (6,) e.g. [0.1,-0.2,0,0.5,0.5,0.5]
```

**Discrete** — Single integer in {0, ..., n-1}. SB3: MlpPolicy (often one-hot in preprocessing).

```python
observation_space = spaces.Discrete(10)
# obs from env: int or np.int64, e.g. 3
obs = 3

# FlattenObservation: one-hot of length n (1 at index obs, 0 elsewhere)
env = FlattenObservation(env)  # observation_space -> Box(shape=(10,), low=0, high=1); obs e.g. [0,0,0,1,0,0,0,0,0,0]
```

**MultiDiscrete** — Cartesian product of several Discrete spaces (each dimension has its own number of values). SB3: MLP after preprocessing (one-hot).

```python
# nvec[i] = number of values for dimension i; default start=0 per dimension
observation_space = spaces.MultiDiscrete([5, 3, 2])  # dim0 in {0..4}, dim1 in {0..2}, dim2 in {0..1}
# obs from env: np.ndarray of ints, shape (3,), e.g. x = [2, 1, 0]
obs = np.array([2, 1, 0], dtype=np.int64)

# FlattenObservation (MultiDiscrete): one-hot encode each dimension, then concatenate.
# flatdim = sum(nvec) = 5+3+2 = 10. Offsets = cumsum([5,3,2]) -> [0, 5, 8, 10].
# Dim0: one-hot of length 5, 1 at index x[0]=2 -> [0,0,1,0,0]
# Dim1: one-hot of length 3, 1 at index x[1]=1 -> [0,1,0]
# Dim2: one-hot of length 2, 1 at index x[2]=0 -> [1,0]
# Result: concat -> shape (10,), e.g. [0,0,1,0,0, 0,1,0, 1,0]; observation_space -> Box(shape=(10,), low=0, high=1)
from gymnasium.wrappers import FlattenObservation
env = FlattenObservation(env)
```

**MultiBinary** — Multiple binary (0/1) dimensions. SB3: MLP. FlattenObservation just flattens (same as Box).

```python
observation_space = spaces.MultiBinary(4)
obs = np.array([0, 1, 1, 0], dtype=np.int8)
# FlattenObservation: np.asarray(x).flatten() -> shape (4,); no one-hot
env = FlattenObservation(env)  # observation_space -> Box(shape=(4,), ...)
```

**Image (Box as image)** — Box with 2D/3D shape, uint8 0–255. SB3: **CnnPolicy** (NatureCNN) or one key in **MultiInputPolicy**.

```python
# channels-last (H, W, C) or channels-first (C, H, W)
observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
# obs from env: np.ndarray shape (84, 84, 3) or (n_envs, 84, 84, 3)
obs = np.zeros((84, 84, 3), dtype=np.uint8)

# FlattenObservation: x.flatten() -> shape (84*84*3,) = (21168,); spatial structure lost, CnnPolicy no longer applies
env = FlattenObservation(env)  # observation_space -> Box(shape=(21168,), ...); obs shape (21168,)
```


---

### 1. Single env from Gym (gym.Env)

One environment; no batch dimension. Each key has the shape of its subspace for a **single** step.

```python
# base_env = gym.make("PointMaze_...");  obs, info = base_env.reset()
# or  obs, r, term, trunc, info = base_env.step(action)

obs = {
    "observation":   np.ndarray,   # shape (4,)   — state e.g. (x, y, vx, vy)
    "desired_goal": np.ndarray,   # shape (2,)   — target (x, y)
    "achieved_goal": np.ndarray,   # shape (2,)   — current (x, y)
}
```

### 2. DummyVecEnv (VecEnv)

Same dict; each value has an **extra leading dimension** `(n_envs, ...)`.

```python
# env = DummyVecEnv([lambda: base_env]);  obs = env.reset()
# or  obs, rews, dones, infos = env.step(actions)

obs = {
    "observation":   np.ndarray,   # shape (n_envs, 4), e.g. (1, 4)
    "desired_goal": np.ndarray,   # shape (n_envs, 2), e.g. (1, 2)
    "achieved_goal": np.ndarray,   # shape (n_envs, 2), e.g. (1, 2)
}
```

### 3. DictReplayBuffer and IntrinsicReplayBuffer

**add(obs, next_obs, action, reward, done, infos)**  
Same dict as DummyVecEnv; stored **per key**. No conversion to a single vector.

```python
# Called by SAC with obs = _last_original_obs, next_obs from env.step()

obs = {
    "observation":   np.ndarray,   # (n_envs, 4), e.g. (1, 4)
    "desired_goal": np.ndarray,   # (n_envs, 2), e.g. (1, 2)
    "achieved_goal": np.ndarray,   # (n_envs, 2), e.g. (1, 2)
}
next_obs = { ... }   # same structure

# Stored per key, e.g.:
#   self.observations["observation"][self.pos]   -> (buffer_size, n_envs, 4)
#   self.observations["desired_goal"][self.pos]   -> (buffer_size, n_envs, 2)
#   self.observations["achieved_goal"][self.pos] -> (buffer_size, n_envs, 2)
```

**sample(batch_size)**  
Returns a **dict of tensors** with batch dimension first. Buffer does **not** flatten the dict.

```python
# batch = replay_buffer.sample(256)  -> DictReplayBufferSamples

batch.observations = {
    "observation":   Tensor,   # (batch_size, 4),  e.g. (256, 4)
    "desired_goal": Tensor,   # (batch_size, 2),  e.g. (256, 2)
    "achieved_goal": Tensor,   # (batch_size, 2),  e.g. (256, 2)
}
batch.next_observations = { ... }   # same structure
# Still a dict; the policy receives this dict.
```

### 4. How the dict is converted for the neural network

The **buffer does not convert** the dict. The **policy**’s **CombinedExtractor** does:

- Receives a **dict of tensors** (same keys as above, shapes `(batch_size, *subspace_shape)`).
- For each key: run a submodule (e.g. `nn.Flatten()` for vector), then **concatenate** on `dim=1` → one feature vector per batch item.

```python
# CombinedExtractor.forward(observations)
# Input:
observations = {
    "observation":   Tensor,   # (256, 4)
    "desired_goal": Tensor,   # (256, 2)
    "achieved_goal": Tensor,   # (256, 2)
}

# Per key:  encoded_tensor_list.append(extractor(observations[key]))
#          e.g. Flatten keeps (256, 4), (256, 2), (256, 2)
# Output:  th.cat(encoded_tensor_list, dim=1)  -> (256, 4+2+2) = (256, 8)
# Actor and critic MLPs take this single tensor; they never see the raw dict.
```

### 5. How the dict in the replay buffer is used for training

1. **SAC** calls `replay_buffer.sample(batch_size)` and gets a batch where `observations` and `next_observations` are still **dicts of tensors** (one tensor per key, shape `(batch_size, ...)`).

2. That batch is passed into the **actor** and **critic**: e.g. `self.actor(replay_data.observations)`, `self.critic(replay_data.observations, replay_data.actions)`. So the networks receive the **dict**, not a single vector.  
   **Why `.observations` not `["observations"]`?** `sample()` returns a **NamedTuple** (`DictReplayBufferSamples`), not a dict. NamedTuples have fixed field names, so you use attribute access (`.observations`, `.actions`, …). You could use `replay_data[0]` for the first field, but the code uses names for clarity.

3. The **policy** (MultiInputPolicy) uses a **CombinedExtractor**. Inside the forward pass it:
   - Takes the dict (e.g. `{"observation": (256, 4), "desired_goal": (256, 2), ...}`).
   - Runs each value through a small submodule (e.g. `Flatten` for vectors).
   - **Concatenates** all results on the feature dimension → one tensor of shape `(batch_size, total_features)`.
   - That single tensor is what the actor/critic MLPs see; they do not see the raw dict.

4. Loss and backprop use that **single feature tensor**. So: the dict is only used inside the policy’s feature extractor to build one vector per sample; the rest of training (Q-values, policy loss, gradients) uses that vector.

---

## Replay buffer

Data structures for **add** and **sample**. Two cases: **DictReplayBuffer** (Dict observation, MultiInputPolicy) and **ReplayBuffer** (Box observation, MlpPolicy). Example shapes: `n_envs=1`, `state_dim=4`, `action_dim=2`, `batch_size=256`.

### DictReplayBuffer

**add(obs, next_obs, action, reward, done, infos)**

```python
# obs, next_obs: dict of np.ndarray, each value shape (n_envs, *subspace_shape)
obs = {
    "observation": np.ndarray,   # (1, 4), float32
}
next_obs = { "observation": np.ndarray }   # (1, 4)
action = np.ndarray   # (1, 2), float32
reward = np.ndarray   # (1,), float32
done = np.ndarray     # (1,), float32 or bool
infos = [dict]        # length n_envs
```

**sample(batch_size)** → `DictReplayBufferSamples` (NamedTuple)

```python
batch = replay_buffer.sample(256)
# batch.observations, batch.next_observations: dict of torch.Tensor, batch dim first
batch.observations = {
    "observation": torch.Tensor,   # (256, 4)
}
batch.next_observations = { "observation": torch.Tensor }   # (256, 4)
batch.actions    # Tensor (256, 2)
batch.dones      # Tensor (256, 1) or (256,)
batch.rewards    # Tensor (256, 1)
```

### ReplayBuffer

**add(obs, next_obs, action, reward, done, infos)**

```python
# obs, next_obs: single np.ndarray, shape (n_envs, obs_dim)
obs = np.ndarray       # (1, 4), float32
next_obs = np.ndarray  # (1, 4)
action = np.ndarray    # (1, 2), float32
reward = np.ndarray    # (1,), float32
done = np.ndarray      # (1,), float32 or bool
infos = [dict]         # length n_envs
```

**sample(batch_size)** → `ReplayBufferSamples` (NamedTuple)

```python
batch = replay_buffer.sample(256)
# batch.observations, batch.next_observations: single Tensor, shape (batch_size, obs_dim)
batch.observations      # Tensor (256, 4)
batch.next_observations  # Tensor (256, 4)
batch.actions   # Tensor (256, 2)
batch.dones     # Tensor (256, 1) or (256,)
batch.rewards   # Tensor (256, 1)
```

**How `n_envs` is handled (ReplayBuffer / DictReplayBuffer):**

- The buffer does **not** flatten `n_envs` when you call `add()`. Inputs have shape `(n_envs, *subspace_shape)`; the buffer stores them as one row at index `self.pos`, so internal storage is `(N, n_envs, *subspace_shape)` where `N` is the internal first dimension (see below). Each `add()` stores one such row and advances `pos` by 1.
- **`buffer_size` you pass vs internal size:** You pass a **total transition capacity** (e.g. 100_000). The implementation does `self.buffer_size = max(buffer_size // n_envs, 1)`. So the first dimension of the arrays is “number of steps” (rows), not total transitions. Each row holds `n_envs` transitions (one per env). That way:
  - **Total transitions stored** = (internal first dim) × n_envs = `(buffer_size // n_envs) × n_envs` ≈ the `buffer_size` you asked for (exactly if divisible).
  - Example: `buffer_size=100_000`, `n_envs=4` → internal first dim = 25_000. Arrays are `(25000, 4, *subspace_shape)`. You call `add()` once per env step (with 4 transitions); after 25_000 such steps the buffer is full and you have 100_000 transitions. So “buffer_size” in the API means “how many transitions I want,” and the buffer uses fewer rows by packing n_envs per row.
- When you **sample**, the buffer “opens” the `n_envs` dimension: for each sampled row index it draws a random env index and returns `data[batch_inds, env_indices, :]`, so you get `batch_size` transitions (one per sampled row, env chosen at random per row). The `n_envs` dimension is resolved at sample time, not at add time.

### Exploration with RND: sample → intrinsic reward → update RND → SAC

For exploration, use the **same batch** for intrinsic reward, RND update, and policy update. You do **not** update the buffer's stored rewards—only return a **copy** of the batch with `rewards` set to the new total (extrinsic + β × intrinsic). In `add()`, pass **extrinsic** reward to the parent so the buffer stores it: `super().add(..., reward=extrinsic_reward, ...)`.


```python
def sample(self, batch_size, env=None):
    batch = super().sample(batch_size, env)   # batch.rewards = extrinsic (stored in buffer)
    # this batch is in (batch_size, observation_size)
    if self.rnd_module is None or self.beta <= 0:
        return batch

    samples = build_rnd_samples_from_batch(batch, device)
    intrinsic_rewards = self.rnd_module.compute(samples, sync=True)
    self.rnd_module.update(samples)

    total_rewards = batch.rewards + self.beta * intrinsic_rewards
    
    return ReplayBufferSamples(
        observations=batch.observations,
        actions=batch.actions,
        next_observations=batch.next_observations,
        dones=batch.dones,
        rewards=total_rewards,
    )
    ### If DictReplayBufferSamples
    # return DictReplayBufferSamples(
    #     observations=batch.observations,
    #     actions=batch.actions,
    #     next_observations=batch.next_observations,
    #     dones=batch.dones,
    #     rewards=total_rewards,
    # )
# SAC then uses this batch for its gradient step.
```




Do **not** update RND in `add()` when using this flow.

---

# Exploration

## Paper:
EXPLORE: Accelerating Exploration with Unlabeled Prior Data, NeurIPS 2023


RLEXPLORE: Accelerating Research in Intrinsically-Motivated Reinforcement Learning, TMLR 2025

## Update proportion
The impact is marginal and mixed, sometimes positive and sometimes negative according to "RLEXPLORE".

## Initialization
EXPLORE: All Dense layers Xavier uniform   
CleanRL: Orthogonal. CleanRL thinks orthogonal is better for RL.

```python
self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
```
RLEXPLORE: The default uniform and the Orthogonal are basically the same.





## Obs Normalization
RL explore Normalization

```python
#RL explore normlaization
    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Normalize the observations data, especially useful for images-based observations."""
        if self.obs_norm:
            x = (
                ((x - self.obs_norm.mean.to(self.device)))
                / th.sqrt(self.obs_norm.var.to(self.device))
            ).clip(-5, 5)
        else:
            x = x / 255.0 if len(self.obs_shape) > 2 else x
        return x
```
Clean RL
```python
from gym.wrappers.normalize import RunningMeanStd
    print("Start to initialize observation normalization parameter.....")
    next_ob = []
    for step in range(args.num_steps * args.num_iterations_obs_norm_init):
        acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
        s, r, d, _ = envs.step(acs)
        next_ob += s[:, 3, :, :].reshape([-1, 1, 84, 84]).tolist()

        if len(next_ob) % (args.num_steps * args.num_envs) == 0:
            next_ob = np.stack(next_ob)
            obs_rms.update(next_ob)
            next_ob = []
    print("End to initialize...")
```
```
rnd_next_obs = (
    (
        (next_obs[:, 3, :, :].reshape(args.num_envs, 1, 84, 84) - torch.from_numpy(obs_rms.mean).to(device))
        / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
    ).clip(-5, 5)
).float()
```

## Neural Network Architecture
EXPLORE:
[obs; action] → Dense(256) → ReLU → Dense(256) → ReLU → Dense(256) → ReLU → Dense(256) → feature
RND always uses current observation and action

ClEANRL:   
RND reward is computed based on next observation


## Reward Normalization   
Don't use reward normalization   
RLEXPLORE shows reward normalization has on mixed effect for RMS and min-max.    Vanilla reward outperforms RMS most of the time and is the most stable one.
Sometimes, using min-max normalization makes the algorithms stop working.


## Output
L1 norm (take average)
L2 norm (take average)

<!-- RL explore Normalization -->


## Plasticity
reset the network


```python

```
Clean RL
```python
        mean, std, count = (
            np.mean(curiosity_reward_per_env),
            np.std(curiosity_reward_per_env),
            len(curiosity_reward_per_env),
        )
        reward_rms.update_from_moments(mean, std**2, count)

        curiosity_rewards /= np.sqrt(reward_rms.var)
```



---

## Snippet 2
*description: example — quick env test*

```python
from env_wrapper import make_env
env = make_env("PointMaze-v0")
obs, _ = env.reset()
print(obs.shape)
```

---

## Snippet 3
*description: example — wandb log one value*

```python
import wandb
wandb.log({"loss": 0.42, "step": 100})
```
