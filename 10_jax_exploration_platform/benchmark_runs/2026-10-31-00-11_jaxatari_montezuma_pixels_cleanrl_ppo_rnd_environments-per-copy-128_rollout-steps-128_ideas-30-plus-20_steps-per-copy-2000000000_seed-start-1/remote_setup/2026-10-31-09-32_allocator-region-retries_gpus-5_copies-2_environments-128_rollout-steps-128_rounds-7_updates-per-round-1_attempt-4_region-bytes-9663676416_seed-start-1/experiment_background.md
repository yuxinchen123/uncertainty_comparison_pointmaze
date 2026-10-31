## Purpose

Run one recorded fourth attempt for each of five original two-copy A4500 comparisons after a successful allocator diagnosis. The original numerical source, case files, checks and timing remain unchanged.

## Key hyperparameters

Each of two independent policies has 128 pixel environments and 128 rollout steps. Training uses four epochs, four minibatches and fifty normalization warmup rollouts. Seven alternating timing pairs contain one full update per arm. Seeds are one and two. Networks use float32 and statistics use float64, with highest matrix precision and deterministic operations.

## Code and config changes

Before learner and game initialization, each fresh process allocates, synchronizes and deletes one nine-GiB byte array. This changes allocator history only. Each card receives eight separate allocated CPU threads. The original three-attempt limit stays recorded; a separate approval permits only attempt four. All earlier files remain preserved.

## Git state

Numerical commit: d28e6735e79788f618acb0ebfde8609db1f776af. Helper commit: cd7d4161f7cc079c0124c919c995eb79315b14f6. Exact approval, source, script and resource identities are in launch_plan.json. No new optimization idea or final scaling result is granted.
