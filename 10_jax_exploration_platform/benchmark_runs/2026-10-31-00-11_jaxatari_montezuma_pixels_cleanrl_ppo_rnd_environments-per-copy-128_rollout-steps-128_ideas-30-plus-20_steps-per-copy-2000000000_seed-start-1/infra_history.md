# Infrastructure history

Hand-written session record. Times are Pacific.

- October 31, 2026, 00:11: created integration worktree; original checkout local changes preserved.
- Source reference checkout and isolated environment preparation started before any training or benchmark.
- ACES and DeltaAI existing SSH connections were verified from portal05. No jobs submitted there yet.

- First pixel jobs 6777084 and 6777085 failed before Python started because new NFS snapshot files were not visible on the compute host. Snapshot publication is now atomic and job scripts check host-side visibility before sourcing files. Failed job IDs remain in the ledger; replacement jobs are recorded separately.

- Pixel baseline numerical snapshot: 21 checks passed; one expected-array fixture was read-only and is corrected. Expanded snapshot: 23 passed; two custom kernel tests exposed pinned JAX Pallas API and RTX A4500 target-name metadata omissions, being corrected before acceptance.
- Small real-game checks passed on jaguar03 and DeltaAI; serval02 card1 small check also completed. These tiny checks are not representative throughput measurements.

## 2026-10-31 01:43 PT: deterministic checks and renderer constructor

The full A4500 learner suite passed 46 checks in job 6777336. Enabling deterministic XLA operations resolved exact-copy and checkpoint-continuation differences without changing test tolerances. Configuration-file job 6777337 passed. Both renderer checks found the optional upstream background-cache template missing prev_is_fire; the adapter now builds static room backgrounds from a complete reset state. The H100 learner check did not collect tests because its launcher omitted the source import path; the launcher now sets it explicitly. All failures remain in their original output records. Formal paired comparisons have not begun.

Preliminary one-copy resource probes completed at 9,114 steps per second on serval02 H100 and 3,712 steps per second on one jaguar03 RTX A4500. These used the earlier runtime configuration and are not accepted optimization comparisons. Two-copy probes will test memory and packing before choosing the common comparison grid.
