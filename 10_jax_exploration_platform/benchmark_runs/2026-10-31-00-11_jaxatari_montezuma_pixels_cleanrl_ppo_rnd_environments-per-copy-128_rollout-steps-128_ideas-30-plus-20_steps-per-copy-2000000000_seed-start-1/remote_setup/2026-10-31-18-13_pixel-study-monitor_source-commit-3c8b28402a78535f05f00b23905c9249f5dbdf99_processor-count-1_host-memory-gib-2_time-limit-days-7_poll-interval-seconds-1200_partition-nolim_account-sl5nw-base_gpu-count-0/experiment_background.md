## Purpose

Prepare the reviewed persistent study monitor that shows allocation collection separately from qualification. Keep the current monitor running until a replacement is running and its first report is verified.

## Key hyperparameters

One CPU, 2 GiB host memory, zero GPUs, seven days on the nolim partition under sl5nw_base, and a 1,200-second interval. No learner or benchmark configuration changes.

## Code and config changes

Use only the committed src and scripts archive recorded in source_snapshot.json. The monitor reads only campaign-owned job identifiers. Allocation observations and site checks receive zero scientific credit and do not admit workers. The candidate check runs inside the existing owned CPU allocation.

## Git state

Committed source: 3c8b28402a78535f05f00b23905c9249f5dbdf99. The archive is verified against every Git blob and excludes all working-tree changes.
