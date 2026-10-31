## Purpose

Check that the committed pixel Montezuma PPO and random network distillation program runs on a DeltaAI GH200 compute node. This short check belongs to campaign 2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1.

## Key hyperparameters

One policy, two environments, two rollout steps, one epoch, one minibatch, and one normalization warmup rollout. Two compilation warmup updates precede two measured updates. Other settings retain the original CleanRL defaults in config_resolved.yaml. Each action repeats for four game frames; observations contain four grayscale 84 by 84 pixel frames.

## Code and config changes

The check uses an immutable git archive, the separate ARM environment, and verified alternative sprites. It saves no checkpoints and does not start full training.

## Git state

Commit: `dd2709f928cb22bccd23ad0395a1d60b97ff3884`. The archive contains only committed files and has SHA256 `4fda91313925f8cf37750dd561905be0fee0fa47101521a0f66aae916dc426f9`. Uncommitted files are excluded.
