## Purpose

Replace the persistent 20-minute metadata monitor with the separately reviewed ACES site-report reader. The existing owned CPU monitor 6780608 remains running until root verifies the replacement first full report.

## Key hyperparameters

One requested CPU, two GiB host memory, zero GPUs, seven days on partition nolim with account sl5nw_base and normal quality of service. Poll every 1200 seconds. Submit held with automatic requeue disabled. One allocated logical CPU is accepted; two require actual two-thread-per-core node topology and consistent requested and allocated resources.

## Code and config changes

Use the exact published src/scripts archive f9a9c1672f788f52ba06836cee26f2dd6314cec7. Both ACES portability and site-monitor helpers are included in its complete Git-blob roster. Keep the proven requested-versus-allocated CPU guard unchanged. The monitor reads mirrored operational evidence and retains existing science, admission and queue boundaries; it opens no SSH connection. A candidate full tick and the first full report on the actual new allocation remain required. This preparation performs neither.

## Git state

Committed monitor source: f9a9c1672f788f52ba06836cee26f2dd6314cec7. Numerical source remains d28e6735e79788f618acb0ebfde8609db1f776af. All scientific, idea, case and final-training credit is zero; this packet admits no workers and submits or releases no job.
