## Purpose

Measure actual memory, initialization time, and steady training throughput for one original reference policy on an ACES H100. This resource probe belongs to campaign 2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1; it is excluded from the thirty paired optimization comparisons.

## Key hyperparameters

One independent policy uses 128 environments, 128 rollout steps, four epochs, four minibatches, and fifty normalization warmup rollouts. Two compilation warmup updates precede seven rounds of ten measured updates. The seed is 1. The complete fixed CleanRL settings are in config_resolved.yaml. Neural calculations use float32 with highest matrix precision; running statistics use float64. Deterministic GPU operations are enabled.

## Code and config changes

The probe executes the exact committed source archive and original named configuration in the dedicated ACES environment with verified alternative sprites. It requests one H100, eight processors, 32 gigabytes of host memory, and one hour. It saves no checkpoints and does not start the final two-billion-step training.

## Git state

Commit: `d28e6735e79788f618acb0ebfde8609db1f776af`. The archive contains committed files only, SHA256 `49b9a77ba47fbb24018223d82e64088f1a0e760d791149293a0b67805141c6b7`. Other working-tree edits are excluded.
