## Purpose

Measure common pixel PPO plus Random Network Distillation throughput on approved serval02 card 1.

## Key hyperparameters

Seven alternating pairs of ten complete updates per arm; 128 environments, 128 rollout steps, four epochs, four minibatches and fifty normalization warmups per independent copy. Original two-billion-transition annealing horizon and seeds remain unchanged.

## Code and config changes

Only room background caching and final-two-frame rendering differ. Historical evidence uses an actual complete CPU snapshot and independent review. The controller holds no GPU standing descriptor.

## Git state

The exact numerical and helper commits and all actual input hashes appear in the immutable measurement plan. No numerical or GPU work occurs during metadata preparation.
