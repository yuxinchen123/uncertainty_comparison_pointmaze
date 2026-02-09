"""
Convert a single env transition (SB3/add format) into the samples dict expected by rllte RND.update().
"""
import torch
from typing import Any, Union


def build_rnd_samples_from_transition(
    obs: Any,
    next_obs: Any,
    action: Any,
    reward: Any,
    done: Any,
    device: Union[str, torch.device],
    n_envs: int = 1,
) -> dict:
    """
    Input format:
        - obs, next_obs: Either dict with key "observation" (array shape (n_envs, state_dim) or (state_dim,))
                          or raw array. This is the format from SB3 when replay_buffer.add() is called
                          (one transition from the environment).
        - action: np.ndarray or tensor, shape (n_envs, action_dim) or (action_dim,).
        - reward: array or scalar, shape (n_envs,) or scalar.
        - done: array or scalar, shape (n_envs,) or scalar.
        - device: torch device for output tensors.
        - n_envs: number of envs in the transition (for shaping).

    Output format:
        - Dict with keys: "observations", "next_observations", "actions", "rewards",
          "terminateds", "truncateds". All values are torch tensors on `device` with shape
          (1, n_envs, *feature_shape), i.e. (n_steps=1, n_envs, ...). This is the format
          required by rllte.xplore.reward.rnd.RND.update(samples).

    Why we need it:
        RND.update() expects a batch in rllte convention: (n_steps, n_envs, *obs_shape).
        The replay buffer receives one transition per add() in SB3 convention (dict obs,
        raw arrays). We need to reshape and convert to tensors so we can call rnd.update()
        every time we get new state from the env and add it to the buffer.
    """
    obs_vec = obs.get("observation", obs) if isinstance(obs, dict) else obs
    next_vec = next_obs.get("observation", next_obs) if isinstance(next_obs, dict) else next_obs
    obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=device)
    next_t = torch.as_tensor(next_vec, dtype=torch.float32, device=device)
    if obs_t.dim() == 1:
        obs_t = obs_t.unsqueeze(0).unsqueeze(0)
        next_t = next_t.unsqueeze(0).unsqueeze(0)
    else:
        obs_t = obs_t.unsqueeze(0)
        next_t = next_t.unsqueeze(0)
    action_t = torch.as_tensor(action, dtype=torch.float32, device=device)
    if action_t.dim() == 1:
        action_t = action_t.unsqueeze(0).unsqueeze(0)
    else:
        action_t = action_t.unsqueeze(0)
    reward_t = torch.as_tensor(reward, dtype=torch.float32, device=device)
    if reward_t.dim() == 0:
        reward_t = reward_t.unsqueeze(0).unsqueeze(0)
    else:
        reward_t = reward_t.unsqueeze(0)
    done_t = torch.as_tensor(done, dtype=torch.float32, device=device)
    if done_t.dim() == 0:
        done_t = done_t.unsqueeze(0).unsqueeze(0)
    else:
        done_t = done_t.unsqueeze(0)
    return {
        "observations": obs_t,
        "next_observations": next_t,
        "actions": action_t,
        "rewards": reward_t,
        "terminateds": done_t,
        "truncateds": torch.zeros_like(done_t, device=device, dtype=torch.float32),
    }
