## Purpose

Restore the persistent20-minute metadata monitor after the previous owned monitor failed while constructing a filename for a252-byte auxiliary directory. The fixed monitor looks up existing specification and final-status filenames.

## Key hyperparameters

One processor,2GiB host memory,zero GPUs,seven days on partition nolim with account sl5nw_base and normal quality of service. Poll every1200seconds. Submit held with automatic requeue disabled. All scientific and final-training credit is zero.

## Code and config changes

The job requests one CPU task. An allocation of one logical CPU is accepted directly; two allocated logical CPUs require matching requested CPU, memory and node counts, matching NumCPUs and AllocTRES, and ThreadsPerCore equal to two on the actual job node.

Use committed src/scripts snapshot e6fa1040236f4b219d003061963a24f7c5ce82d2. The monitor retains exact-owned-ID scheduler queries,existing admission boundaries and all queue duties. No full tick runs during local preparation. Root will verify the first complete report on the actual released compute allocation.

## Git state

Committed monitor source: `e6fa1040236f4b219d003061963a24f7c5ce82d2`. Source paths match the committed tree. The earlier f988 packet and its failed job6780433 remain recorded unchanged.
