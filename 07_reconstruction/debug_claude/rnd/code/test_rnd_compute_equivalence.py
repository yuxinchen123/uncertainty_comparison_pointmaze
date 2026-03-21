"""
Test that old MyRND and new RND produce equivalent intrinsic rewards
when configured to be functionally identical (rnd_next_state, n_predictors=1).

Key difference found: old MyRND ALWAYS uses EnsembleObservationEncoder (even n=1),
while new RND uses ObservationEncoder for n=1. Both should produce same results
if weights are identical, but the initialization differs (different random init).

This test forces shared weights to verify the forward-pass math is identical.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import torch

from intrinsic.my_rnd.rnd import MyRND, EnsembleObservationEncoder, ObservationEncoder as OldObservationEncoder
from intrinsic.intrinsic_method.rnd import RND, ObservationEncoder as NewObservationEncoder


def test_compute_equivalence():
    """Verify old MyRND and new RND give same rewards with same weights."""
    obs_shape = (4,)
    output_dim = 128
    device = "cpu"
    batch_size = 32

    torch.manual_seed(42)
    np.random.seed(42)

    old_rnd = MyRND(
        obs_shape=obs_shape,
        output_dim=output_dim,
        lr=0.001,
        batch_size=256,
        device=device,
        use_obs_norm=False,
        distance="mse",
        obs_slice=None,
        n_predictors=1,
        beta_std=0.0,
    )

    torch.manual_seed(42)
    new_rnd = RND(
        obs_shape=obs_shape,
        output_dim=output_dim,
        lr=0.001,
        batch_size=256,
        device=device,
        use_obs_norm=False,
        distance="mse",
        n_predictors=1,
        beta_std=0.0,
        feature="rnd_next_state",
        action_dim=2,
        linear_rnd=False,
    )

    # --- Copy old target weights into new target ---
    with torch.no_grad():
        new_rnd.target.network[0].weight.copy_(old_rnd.target.network[0].weight)
        new_rnd.target.network[0].bias.copy_(old_rnd.target.network[0].bias)
        new_rnd.target.network[2].weight.copy_(old_rnd.target.network[2].weight)
        new_rnd.target.network[2].bias.copy_(old_rnd.target.network[2].bias)

    # --- Copy old ensemble predictor weights into new single predictor ---
    # Old EnsembleObservationEncoder stores w1[0] = Linear.weight.T, so
    # w1[0].T = Linear.weight
    with torch.no_grad():
        new_rnd.predictor.network[0].weight.copy_(old_rnd.predictor.w1[0].T)
        new_rnd.predictor.network[0].bias.copy_(old_rnd.predictor.b1[0])
        new_rnd.predictor.network[2].weight.copy_(old_rnd.predictor.w2[0].T)
        new_rnd.predictor.network[2].bias.copy_(old_rnd.predictor.b2[0])

    obs_data = torch.randn(batch_size, 4)
    samples_old = {"next_observations": obs_data}
    samples_new = {"next_observations": obs_data, "observations": obs_data, "actions": torch.randn(batch_size, 2)}

    old_reward = old_rnd.compute(samples_old)
    new_reward = new_rnd.compute(samples_new)

    max_diff = (old_reward - new_reward).abs().max().item()
    mean_old = old_reward.mean().item()
    mean_new = new_reward.mean().item()

    print(f"Old MyRND mean reward: {mean_old:.6f}")
    print(f"New RND mean reward:   {mean_new:.6f}")
    print(f"Max absolute diff:     {max_diff:.10f}")
    assert max_diff < 1e-5, f"Rewards differ by {max_diff}"
    print("PASS: compute() outputs are equivalent with shared weights.")


def test_update_difference():
    """
    KEY FINDING: old update() trains on observations with DataLoader mini-batching,
    new update() trains on next_observations with a single gradient step.
    """
    obs_shape = (4,)
    output_dim = 128
    device = "cpu"

    torch.manual_seed(99)
    old_rnd = MyRND(
        obs_shape=obs_shape, output_dim=output_dim, lr=0.001, batch_size=64,
        device=device, use_obs_norm=False, distance="mse", obs_slice=None,
        n_predictors=1, beta_std=0.0,
    )

    torch.manual_seed(99)
    new_rnd = RND(
        obs_shape=obs_shape, output_dim=output_dim, lr=0.001, batch_size=256,
        device=device, use_obs_norm=False, distance="mse", n_predictors=1,
        beta_std=0.0, feature="rnd_next_state", action_dim=2, linear_rnd=False,
    )

    obs = torch.randn(256, 4)
    next_obs = torch.randn(256, 4)
    actions = torch.randn(256, 2)

    # Old update uses samples["observations"], new uses _get_feature_tensor (next_observations for rnd_next_state)
    old_samples = {"observations": obs, "next_observations": next_obs}
    new_samples = {"observations": obs, "next_observations": next_obs, "actions": actions}

    # Capture predictor params before update
    old_w_before = old_rnd.predictor.w1.clone()
    new_w_before = new_rnd.predictor.network[0].weight.clone()

    old_rnd.update(old_samples)
    new_rnd.update(new_samples)

    old_w_after = old_rnd.predictor.w1
    new_w_after = new_rnd.predictor.network[0].weight

    old_changed = (old_w_before - old_w_after).abs().sum().item()
    new_changed = (new_w_before - new_w_after).abs().sum().item()

    print(f"\nOld predictor weight change (sum abs): {old_changed:.6f}")
    print(f"New predictor weight change (sum abs): {new_changed:.6f}")
    print(f"Old trains on: observations (with DataLoader mini-batching, batch_size=64)")
    print(f"New trains on: next_observations (single gradient step)")
    print()

    # Count gradient steps: old uses DataLoader with batch_size=64 on 256 samples = 4 steps
    n_old_steps = max(1, 256 // 64)
    print(f"Old update: ~{n_old_steps} gradient steps per call (DataLoader)")
    print(f"New update: 1 gradient step per call")
    print("FINDING: Old code does MORE gradient updates per replay buffer sample.")


if __name__ == "__main__":
    test_compute_equivalence()
    print("\n" + "="*60 + "\n")
    test_update_difference()
