## Purpose

Prepare a fresh, zero-credit full-original A4500 target qualification with an explicit pytest collection root. The previous job 6782640 failed its root-directory validation after 48 passed tests and remains failed and unchanged; no earlier test work is reused as qualification.

## Key hyperparameters

One independent policy copy, 128 environments, 128 rollout steps, four epochs, four minibatches, fifty normalization warmup rollouts and one full reference update, seed 1. The original 2-billion-step annealing horizon, float32 networks/Adam, float64 statistics, highest matrix precision and deterministic operations are retained. Requested resources are one A4500, eight CPUs, 64 GiB host memory, three hours, gpu partition, sl5nw_base account, csresnolim quality of service and the verified owned reservation.

## Code and config changes

Before: the previous runner allowed pytest to choose an outer root, and the strict qualification reader rejected that run. After: the reviewed source passes the exact frozen numerical platform root using --rootdir and requires the same command/root and all 48 original cases with all 144 passed phases. The full 33-room/132-case/11-synthetic renderer gate and fifty-warmup/one-reference-update gate are unchanged. The configuration is fully original, with no combined optimization enabled. Actual executed controlled work is counted separately from unquantified test-fixture work; no whole-job transition total is claimed. This preparation grants no correctness, timing, case, idea, packing, training or worker-admission credit.

## Git state

Published helper source: e5afbf6b51a3d1666b46e71c0631a45d47080cd8. Numerical source: d28e6735e79788f618acb0ebfde8609db1f776af. Exact source/software/assets, full configuration and staged helper hashes are retained in the new packet and unit. Frozen job and physical GPU identities remain unresolved. No job is submitted, bound, released or cancelled by preparation.
