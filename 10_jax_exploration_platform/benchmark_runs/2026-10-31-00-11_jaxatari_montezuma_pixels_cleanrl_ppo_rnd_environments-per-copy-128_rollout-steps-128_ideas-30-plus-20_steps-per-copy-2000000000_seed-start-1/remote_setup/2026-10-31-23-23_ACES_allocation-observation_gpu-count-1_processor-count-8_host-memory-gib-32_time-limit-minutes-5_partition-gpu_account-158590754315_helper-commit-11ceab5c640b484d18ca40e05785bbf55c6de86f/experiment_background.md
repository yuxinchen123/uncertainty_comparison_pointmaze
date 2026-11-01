## Purpose

Collect actual owned ACES scheduler, cgroup and NVIDIA hardware metadata. Collection does not select a physical card, qualify precision, admit a worker or start training.

## Key hyperparameters

One typed H100, eight CPUs, 32 GiB host memory and five minutes. No learner, policy, seed, rollout or numerical computation is configured.

## Code and config changes

The separate ACES guard reads only this campaign's ACES ledger and exact submission/release records. It retains all hardware CSV rows without equating CUDA visibility, scheduler numbers or device minors. XML is not collected because full NVIDIA XML includes process information. Missing configuration remains explicit.

## Git state

Exact helper commit: 11ceab5c640b484d18ca40e05785bbf55c6de86f. Both helper files are bound in the plan. The unchanged adjacent collector supplies only generic hardware and configured-isolation observations.
