"""
Convert a sampled batch (SB3 replay buffer / DictReplayBufferSamples style) into the samples dict
expected by rllte RND.compute(), and optionally compute intrinsic rewards.
"""
import torch
from typing import Any, Dict, Optional, Union


def build_rnd_samples_from_batch(
    batch_dict: Dict[str, Any],
    device: Union[str, torch.device],
) -> dict:
    """
    Input format:
        - batch_dict: Dict with keys "observations", "next_observations", "actions", "dones".
          These come from IntrinsicReplayBuffer.sample() / DictReplayBufferSamples:
          - "observations" and "next_observations" are dicts with key "observation" (tensors
            shape (batch_size, state_dim)) or raw tensors (batch_size, state_dim).
          - "actions": tensor (batch_size, action_dim).
          - "dones": tensor (batch_size,) or (batch_size, 1).
          Batch size is n_steps in rllte terms (number of transitions in the batch).

    Output format:
        - Dict with keys: "observations", "next_observations", "actions", "rewards",
          "terminateds", "truncateds". All values are torch tensors on `device` with shape
          (n_steps, n_envs, *feature_shape), i.e. (batch_size, 1, ...) since we use n_envs=1.
          This is the format required by rllte.xplore.reward.rnd.RND.compute(samples).

    Why we need it:
        RND.compute() expects (n_steps, n_envs, *obs_shape). The replay buffer returns a batch
        as dict of tensors with shape (batch_size, *shape) and possibly Dict observations.
        We extract the state vector ("observation" key), add the n_envs dimension, and
        build the full samples dict so we can call rnd.compute(samples, sync=True) and get
        intrinsic rewards for the batch.
    """
    obs_dict = batch_dict["observations"]
    next_obs_dict = batch_dict["next_observations"]
    obs_t = obs_dict["observation"] if isinstance(obs_dict, dict) else obs_dict
    next_obs_t = next_obs_dict["observation"] if isinstance(next_obs_dict, dict) else next_obs_dict
    if hasattr(obs_t, "to"):
        obs_t = obs_t.to(device)
        next_obs_t = next_obs_t.to(device)
    else:
        obs_t = torch.as_tensor(obs_t, dtype=torch.float32, device=device)
        next_obs_t = torch.as_tensor(next_obs_t, dtype=torch.float32, device=device)
    n_steps = obs_t.size(0)
    # rllte expects (n_steps, n_envs, *obs_shape)
    obs_t = obs_t.unsqueeze(1)
    next_obs_t = next_obs_t.unsqueeze(1)
    actions = batch_dict.get("actions")
    if actions is not None:
        if hasattr(actions, "to"):
            actions = actions.to(device)
        else:
            actions = torch.as_tensor(actions, device=device, dtype=torch.float32)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
    else:
        actions = torch.zeros(n_steps, 1, 1, device=device, dtype=torch.float32)
    dones = batch_dict.get("dones")
    if dones is not None and hasattr(dones, "to"):
        dones = dones.to(device).float()
    else:
        dones = torch.zeros(n_steps, 1, device=device, dtype=torch.float32)
    if dones.dim() == 1:
        dones = dones.unsqueeze(1)
    return {
        "observations": obs_t,
        "next_observations": next_obs_t,
        "actions": actions,
        "rewards": torch.zeros(n_steps, 1, device=device, dtype=torch.float32),
        "terminateds": dones,
        "truncateds": torch.zeros(n_steps, 1, device=device, dtype=torch.float32),
    }


def make_rnd_intrinsic_reward_fn(rnd_module: Any, device: Union[str, torch.device]) -> Any:
    """
    Returns a callable that takes a batch_dict (SB3-style) and returns a 1D numpy array of
    intrinsic rewards of length batch_size, by converting the batch to RND samples and
    calling rnd.compute(samples, sync=True).

    Input format (for the returned callable):
        - batch_dict: Same as build_rnd_samples_from_batch (observations, next_observations,
          actions, dones). If "observation" key is missing from observations/next_observations,
          the callable returns None so the buffer can fall back to per-sample mode.

    Output format (for the returned callable):
        - np.ndarray of shape (batch_size,) with intrinsic reward per transition, or
          None if batch_dict is not in the expected form (e.g. no "observation" key).

    Why we need it:
        The IntrinsicReplayBuffer expects intrinsic_reward_fn(batch_dict) to return either
        a batch of intrinsic rewards (array) or to be called per-sample. This wrapper
        performs the batch→RND samples conversion and calls rnd.compute(), so the buffer
        can use RND in batch mode without knowing rllte's format.
    """
    def intrinsic_reward_fn(batch_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        obs_dict = batch_dict.get("observations")
        next_obs_dict = batch_dict.get("next_observations")
        if isinstance(obs_dict, dict) and "observation" not in obs_dict:
            return None
        if isinstance(next_obs_dict, dict) and "observation" not in next_obs_dict:
            return None
        samples = build_rnd_samples_from_batch(batch_dict, device)
        ir = rnd_module.compute(samples, sync=True)
        return ir.cpu().numpy().ravel()
    return intrinsic_reward_fn
