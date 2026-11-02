## Purpose

Check supplied-input and complete-update correctness for the combined room-background cache and final-two-frame rendering changes on H100 NVL. This is a separate correctness prerequisite, with no timing or study credit.

## Key hyperparameters

Each independent copy has 128 environments, 128 rollout steps, four epochs, four minibatches and fifty normalization warmup rollouts. Networks and Adam use float32; statistics use float64; matrix precision is highest and operations are deterministic. The annealing horizon is two billion transitions per copy. Copy seeds are [1, 2]. No timing rounds or checkpoints are requested.

## Code and config changes

The candidate changes only cache_room_backgrounds and render_final_two_only from false to true. The original numerical source, configuration, independent random streams and qualified package image remain fixed. The two copy counts run serially, after exact prior exits.

## Git state

Original numerical source: d28e6735e79788f618acb0ebfde8609db1f776af. Exact clean private helper source: 0a880984b4a03a2ff7335337cc18576628bc9fc1. No process, device binding, correctness success or admission has been observed for this planned unit.
