## Purpose

Prepare the exact one-shot held submission for the two reviewed common-candidate A4500 correctness packets. This utility never releases a job.

## Key hyperparameters

One A4500, eight processors, 64 GiB and three hours per independently submitted packet. The one-copy and two-copy requests are separate commands. Actual job identifiers exist only after a successful returned Slurm identifier.

## Code and config changes

No scientific configuration or producer source changes. This directory adds standard-library-only submission and durable ownership records.

## Git state

The numerical source remains d28e6735e79788f618acb0ebfde8609db1f776af and the staged helper source remains e5afbf6b51a3d1666b46e71c0631a45d47080cd8. Source and packet publication are required before any submission.
