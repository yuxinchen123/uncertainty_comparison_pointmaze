## Purpose

Finish one fresh original gathered-minibatch comparison after the separately completed allocator diagnosis. This fifth attempt has its own approval; earlier case bytes, the original retry limit and all earlier attempts remain preserved.

## Key hyperparameters

Two independent policies, 128 pixel environments per policy, 128 rollout steps, four epochs, four minibatches, fifty normalization warmup rollouts, seeds one and two, and seven alternating one-update timing pairs. Networks use float32/highest and statistics float64, with deterministic operations.

## Code and config changes

Use the diagnosed CUDA asynchronous allocator equally for reference and candidate, retaining fraction 0.85 and disabled preallocation. Run the unchanged original numerical functions and every original correctness/timing check in a fresh process. A separate process/runtime/source/library receipt binds the raw original result. No sixth attempt is granted.

## Git state

Numerical source d28e673. Reviewed and pushed helper b9a8a36. The approval, completed diagnostic, exact script, resource and source hashes are in launch_plan.json. Persistent monitor 6778089 was updated before this launch.
