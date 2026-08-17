# Infrastructure history

Times are Pacific.

## 2026-08-16 22:22 PT — the parent run's missing-argument guards were re-tested before launch

The parent run's `infra_history.md` records three jobs that started with the runner's defaults
instead of their unit's arguments, because `run_unit.sh` claimed a unit by renaming its queue entry
and only then tried to read that entry. This run copies the fixed scripts, and the fix was exercised
rather than assumed:

1. A job given a unit id with no queue entry anywhere printed `no queue entry for unit ...` and
   exited 2.
2. A job given a queue entry carrying two arguments printed `read no arguments from ...; refusing to
   start with the runner's defaults` and exited 3 — and the entry was still in `queue/pending/`
   afterwards, so no claim had happened.
3. `scripts/run_training.py` called without `--unit-id` exited 2 with `the following arguments are
   required: --unit-id`.

## 2026-08-16 22:47 PT — the run was respecified from 8,192 copies to 1,024, and four jobs were cancelled

The extension was first launched at 8,192 copies for 100 million steps per copy
(`runs/2026-08-16-22-30_..._copies-8192_extension-of-first-batch`). Its four canaries had finished
and its four science jobs were running when the specification changed to 1,024 copies for
1,000,038,400 steps per copy.

Jobs 6538647 (unit 1, serval07), 6538648 (unit 2, serval08), 6538649 (unit 3, serval09) and 6538651
(unit 4, cheetah01) were cancelled, each read from that run's own
`slurm/submitted_jobids.txt` and checked against it before `scancel` — never a blanket cancel. They
had run 8 minutes, 8 minutes, 8 minutes and 1 minute. Two of them had written window records, which
stay in that folder and are not used here: a run folder's name is an identifier and says 8,192
copies, so the 1,024-copy run got a folder of its own rather than reusing a folder whose name would
then be a lie.

Nothing else of the 8,192-copy attempt was discarded. Its canary rates at 8,192 copies are cited by
this run's submission planner as the fallback anchors, and its guards test is the entry above.

## 2026-08-16 23:08 PT — the rate probes changed the plan, twice

Two probe jobs (6538652 on serval06, 6538653 on cheetah01) measured every arm at 256, 512 and 1,024
copies before any science job was submitted. They were not a formality: the plan computed from the
shared survey alone would have been wrong in two ways, both of which would have cost hours.

1. The survey's trainer improves its per-copy rate 1.59 times when a unit is halved from 1,024 to
   512 copies. Measured here it is 1.43 for distillation and 1.25 for the other three arms. A plan
   built on 1.59 over-values splitting.
2. Worse, the first plan scaled the survey's per-class rates by each arm's cost relative to the
   survey's trainer as measured at 8,192 copies — 2.31 to 2.38 for the three non-distillation arms.
   At 1,024 copies the arms are within 1.7 of each other, so that factor credited an RTX 5080 with
   3.80 hours for a 512-copy chunk that an H100 measures at 3.64 — a slower card apparently matching
   the fastest one. The planner was changed to carry only the survey's ratio between two cards at
   the same copy count, keeping the arm and the copy count from its own measurement, and to record
   per chunk which source priced it.

The plan that went out is 5 chunks on 5 H100 cards with a computed makespan of 5.31 hours. Under the
first, wrong model the same combination read 4.20 hours, so the measurement also removed an estimate
that would have looked like a 25 per cent overrun later.

## 2026-08-17 03:44 PT — the queue drained, and the watchers had died before it

All five chunks completed on their first attempt. Jobs 6538662, 6538663 and 6538664 (the three whole
units) finished between 03:05 and 03:09 PT at 3 h 47 m to 3 h 50 m elapsed, and jobs 6538660 and
6538661 (the two distillation chunks) at 03:44 PT at 4 h 26 m. Nothing was requeued, nothing failed,
and `queue/done/` holds all five.

The session's own watchers — the 20-minute monitor and the drain waiter — stopped at about 00:25 PT
when the session hit its usage limit. **The jobs were unaffected**, which is the point of running
generation work as detached scheduler jobs: a watcher is not a job, and none of the five noticed.
Every window record between 00:25 and 03:44 PT is on disk. Nothing was restarted on account of the
watchers stopping, per the standing rule that a stopped watcher is never a reason to restart a job.

One measurement worth keeping: the makespan came in at 4 h 26 m against the plan's 5 h 19 m. The
plan's per-chunk estimates were built from 200-iteration canaries, whose steady rate still carries
the first iterations' transients; over 1.95 million iterations every chunk ran about 17 per cent
faster than its canary projected. The error is uniform across chunks, so the plan's ordering — which
is what the assignment actually rests on — was unaffected.
