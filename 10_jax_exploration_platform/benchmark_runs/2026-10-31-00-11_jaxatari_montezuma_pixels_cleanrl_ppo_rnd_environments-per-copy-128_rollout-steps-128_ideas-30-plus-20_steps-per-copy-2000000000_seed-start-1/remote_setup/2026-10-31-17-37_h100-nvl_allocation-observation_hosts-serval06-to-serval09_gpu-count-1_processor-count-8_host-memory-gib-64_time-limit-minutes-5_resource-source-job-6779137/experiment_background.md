## Purpose

Record actual Slurm allocation identifiers and NVIDIA GPU hardware inventory. No GPU mapping, site compatibility or source admission is accepted by this collection.

## Key hyperparameters

One H100 NVL, eight CPUs, 64 GiB host memory and five minutes. No learner, policy, seed, rollout or numerical computation is configured.

## Code and config changes

A standalone standard-library helper records allowlisted allocation and hardware fields. The learner, failed site packets, queues, gates and policies remain unchanged.

## Git state

Detached helper commit: bba2278eb57e1f76fba26f561c448c0698301750. The executed helper file matches its committed bytes.
