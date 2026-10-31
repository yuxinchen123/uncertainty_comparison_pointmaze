## Purpose

Read Slurm CPU binding and actual CPU affinity after the retry parent stopped before claiming any comparison.

## Key hyperparameters

Forty CPU threads, no GPU, no numerical libraries and no training. Compare the default thread binding with one thread per core.

## Code and config changes

Only standard-library metadata is printed. No comparison files, runtime settings or attempt limits change.

## Git state

The failed parent used committed helper cd7d416. This record preserves the exact read-only commands.
