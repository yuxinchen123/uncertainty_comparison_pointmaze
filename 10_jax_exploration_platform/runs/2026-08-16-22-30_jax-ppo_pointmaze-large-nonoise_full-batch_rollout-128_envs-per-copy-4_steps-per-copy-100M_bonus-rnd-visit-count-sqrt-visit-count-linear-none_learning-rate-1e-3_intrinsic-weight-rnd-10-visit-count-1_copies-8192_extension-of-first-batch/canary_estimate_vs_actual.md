# Canary phase — estimate against actual

Four canary runs, one per (work unit, node class) pairing that the real submission uses, each at
the unit's full 8,192 copies and limited to 200 iterations (one phase-blocked episode window).
Their purpose is the skill's: prove the card holds the program, prove the compiled-program cache,
prove the resume, and correct the time estimate before the real jobs go out. The deviation from
Step 6's figure of at least 30 simultaneous canaries, and why four is the right number for a
four-unit two-class submission, is in `experiment_background.md`. Times are Pacific.

## Throughput

Seconds per iteration is the steady rate: the wall clock of the 200 iterations minus the first
iteration, which pays for compiling the program, divided by the remaining 199. Both rate columns
are the same measurement seen two ways — what the card does in total and what one of its 8,192
copies gets — and the column after them restates the per-copy rate in the unit a run is planned in.

TABLE PLACEHOLDER

## What the canaries changed

TO BE FILLED

## Resume

The resume was tested on a canary rather than argued for. Job 6538644 re-ran the exact command of
the already-finished unit-2 canary on the same node. Its log reads

```
resume: unit-2_gt-position-velocity-sqrt_... is already complete in
unit-2_gt-position-velocity-sqrt_....jsonl (4 records on disk); skipping it
```

and it exited 0 after one second. The shard held 4 records before the re-run and 4 after: nothing
was repeated, nothing was truncated, and the file was opened for appending rather than rewritten.
That is the whole resume contract of this platform — no model state is saved, so the resumable unit
is the unit, and a unit whose shard ends in a completion record makes a re-run a logged no-op.

## The missing-argument guards

Re-tested before the first submission; all three refuse rather than fall back to defaults. The
three cases and their exit codes are in `infra_history.md`.
