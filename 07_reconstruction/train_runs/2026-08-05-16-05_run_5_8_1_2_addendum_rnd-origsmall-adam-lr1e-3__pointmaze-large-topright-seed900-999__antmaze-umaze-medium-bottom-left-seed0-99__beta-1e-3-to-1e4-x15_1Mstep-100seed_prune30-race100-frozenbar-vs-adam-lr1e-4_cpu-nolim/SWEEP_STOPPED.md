# Sweep stopped 2026-08-08 01:53 (drain)

The user stopped this sweep on 2026-08-08 to free the CPU pools for train run 6 (the four
train-run-8.1.2 algorithms swept on the run-5 PointMaze task). Same procedure as the
train-run-1.2 stop: every unlaunched marker was archived, every in-flight run FINISHES and its
record is kept, and no job was cancelled — ours drain as their workers find the queue empty,
and the collaborator's drain the same way on their own.

## State at the stop (per learning rate and environment)

| lr | environment | done | running | pruned | unlaunched (archived) |
|---|---|---|---|---|---|
| 1e-3 | AntMaze Medium | 597 | 0 | 3121 | 782 |
| 1e-3 | AntMaze UMaze | 602 | 0 | 3118 | 780 |
| 1e-3 | PointMaze Large | 605 | 0 | 2594 | 1301 |
| 1e-2 | AntMaze Medium | 164 | 306 | 0 | 4030 |
| 1e-2 | AntMaze UMaze | 162 | 309 | 0 | 4029 |
| 1e-2 | PointMaze Large | 167 | 309 | 0 | 4024 |

Totals: 2,297 done, 924 running (all Adam 1e-2 — they finish over the next ~21 h and land in
done/), 8,833 pruned by the truncation controller (34 Adam 1e-3 configurations truncated
against the frozen bars), 14,946 archived to `queue/2026-08-05-16-05_lr1e3/stopped_unlaunched/`
(0 lost to claim races during the move).

## What this means for the data

- Adam 1e-3: final at roughly 40 completed seeds per surviving configuration (the truncated
  ones keep the n they had). No configuration reached a survivor verdict; 34 of 45 were
  truncated; the rest end undecided.
- Adam 1e-2: ends at roughly 10–13 completed seeds per configuration now, rising to ~17–20
  once the 924 in-flight runs land. All 45 of its configurations end undecided (its data never
  reached the 30-seed decision floor).
- The analysis/writeup tables regenerate one final time after the drain completes.

## Drain bookkeeping

- The monitor job 6534388 stays up during the drain (its top-up finds pending/ empty and
  submits nothing; the controller may still decide late-arriving 1e-2 configurations if any
  reaches 30 seeds — those verdicts are valid and kept). When running/ reaches 0: stop
  6534388 by its id from `slurm/submitted_jobids_2026-08-05-16-05_lr1e3_sl5nw.txt`, regenerate
  the tables/figures, and mark this file with the final counts.
- Collaborator notice: `for_collaborator/SWEEP_STOPPING_NOTICE.md`.
