## Purpose

Replace a submission that stopped before a comparison claim. Reuse the unchanged fifth-attempt approval and frozen metadata helper; preserve every old failure and launch record.

## Key hyperparameters

Use two policies, 128 environments and 128 rollout steps per policy, four epochs, four minibatches, fifty normalization warmup rollouts, seeds one and two, seven alternating pairs with one full update per arm, and three latency calls. Request one RTX A4500, eight CPUs and 64 GiB host memory on jaguar03.

## Code and config changes

Before: resolve and export the selected GPU UUID before srun. After: record the step allocation, resolve and export the UUID inside srun, record the resulting allocation, and execute the same approved worker. Use cuda_async with fraction 0.85 and preallocation false; keep float32/highest neural arithmetic, float64 statistics, deterministic operations, the original checks and timing. Do not prepare a large array. No sixth attempt is permitted. The prior failed job did not capture its actual CUDA visibility or cgroup; this wrapper records both.

## Git state

Numerical source: d28e6735e79788f618acb0ebfde8609db1f776af. Frozen metadata helpers: b9a8a366d6efad3b305caf71366efd8414d0d26a. The original approval, prior failed submission, reference and candidate configurations, and all new wrapper files are bound by hashes in this folder. Preparation does not submit work or consume the approval.
