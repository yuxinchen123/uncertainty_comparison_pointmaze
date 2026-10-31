## Purpose

Run one recorded fourth attempt for each of five original two-copy A4500 comparisons after a successful allocator diagnosis. The original numerical source, case files, checks and timing remain unchanged.

## Key hyperparameters

Each of two independent policies has 128 pixel environments and 128 rollout steps. Training uses four epochs, four minibatches and fifty normalization warmup rollouts. Seven alternating timing pairs contain one full update per arm. Seeds are one and two. Networks use float32 and statistics use float64, with highest matrix precision and deterministic operations.

## Code and config changes

Before learner and game initialization, each fresh process allocates, synchronizes and deletes one nine-GiB byte array. This changes allocator history only. Each card receives eight separate allocated CPU threads. The original three-attempt limit stays recorded; a separate approval permits only attempt four. All earlier files remain preserved.

## Git state

Numerical commit: d28e6735e79788f618acb0ebfde8609db1f776af. Helper commit: 03b0f62b7756809c312011f8d7a9f3311ee0aac3. Exact approval, source, script and resource identities are in launch_plan.json. No new optimization idea or final scaling result is granted.

Slurm reports an exact resolved CPU mask for the requested thread binding. The new helper verifies that mask against actual affinity, then retains separate eight-thread assignments per GPU. The stopped launch and all old approvals remain unchanged; each new approval binds both the old approval and the preflight-failure receipt. No comparison attempt was consumed by the stopped launch.
