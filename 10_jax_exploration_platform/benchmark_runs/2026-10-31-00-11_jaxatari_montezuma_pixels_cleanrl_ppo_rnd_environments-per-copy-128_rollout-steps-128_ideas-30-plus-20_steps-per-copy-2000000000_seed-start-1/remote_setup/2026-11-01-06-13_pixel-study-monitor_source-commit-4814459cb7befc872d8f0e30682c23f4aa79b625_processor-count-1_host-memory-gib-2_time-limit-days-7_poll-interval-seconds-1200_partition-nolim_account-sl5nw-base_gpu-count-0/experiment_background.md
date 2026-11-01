## Purpose

Replace the persistent 20-minute metadata monitor with the reviewed planned-diagnostic display correction. The existing owned CPU monitor 6780635 keeps running until root verifies the replacement first full report.

## Key hyperparameters

One requested CPU, two GiB host memory, zero GPUs, seven days on partition nolim with account sl5nw_base and normal quality of service. Poll every 1200 seconds. Submit held with automatic requeue disabled. One allocated logical CPU is accepted; two require actual two-thread-per-core node topology and consistent requested and allocated resources.

## Code and config changes

Before: planned diagnostic preparations could be shown as running by source f9a9c1672f788f52ba06836cee26f2dd6314cec7. After: source 4814459cb7befc872d8f0e30682c23f4aa79b625 preserves planned status and marks a preserved preparation finished only when no submission or execution evidence exists. Failure and held status keep precedence. The requested-versus-allocated CPU guard is unchanged.

Use only the exact committed src/scripts archive inside this packet. Preparation checks source visibility, hashes and syntax. A candidate full tick and the first full report on the actual new allocation remain required; this preparation runs neither.

## Git state

Committed monitor source: 4814459cb7befc872d8f0e30682c23f4aa79b625. Numerical source remains d28e6735e79788f618acb0ebfde8609db1f776af. All scientific, idea, case and final-training credit is zero. This packet admits no workers and submits or releases no job.
