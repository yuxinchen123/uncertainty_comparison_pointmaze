# Infrastructure history

Times are Pacific.

## 2026-08-16 21:01 PT — three real jobs started with the runner's defaults instead of the unit's arguments

**What happened.** Jobs 6538596 (unit-1, serval07), 6538597 (unit-2, serval07) and 6538598
(unit-3, serval08) each ran 128 copies for 200 iterations — the runner's defaults — instead of the
8,448 copies and 19,600 iterations their queue entry names. All three then failed in the aggregation
step, and one wrong shard, `data/unit_0000.jsonl`, was written under the runner's default unit name.

**Cause.** `code/run_unit.sh` resolved the unit's queue-entry path, then CLAIMED the unit by renaming
that entry from `queue/pending/` to `queue/running/`, and only then read the entry to get the unit's
arguments. The read therefore opened a path that no longer existed, `mapfile` got an empty list, and
`run_training.py` was called with nothing but `--run-dir`, so every knob fell back to its default.
The canaries had not caught it because a canary claims nothing and so never renames the entry.

**Fix.** Two changes, both in this run's own code:

1. `code/run_unit.sh` now reads the arguments BEFORE the claim, and refuses to start at all if it
   read fewer than four of them.
2. `scripts/run_training.py` makes `--unit-id` a required argument with no default, so a job that
   loses its arguments fails at once instead of writing a plausible-looking shard of defaults.

**Cleanup.** The three job ids were cancelled from this run's own `slurm/submitted_jobids.txt` (never
a blanket cancel), `data/unit_0000.jsonl` and its log were deleted, and all four unit entries were
put back in `queue/pending/`. No science record was affected: none had been written yet.

**Cost.** About four minutes, and the compiled-program cache on serval07 and serval08 was left warm
by the canaries, so the resubmission's build and prime fell from 350 seconds to 11.

## 2026-08-16 21:36 PT — the queue drained

Jobs 6538605, 6538606 and 6538607 completed on their first attempt, and the unscheduled `serval05`
unit finished at 21:07 PT. Nothing was requeued. The 20-minute monitoring ended here, with all four
units in `queue/done/` and every unit's completion record on disk.

One measurement worth keeping: the compiled-program cache in `/localtmp/$USER/platform_jax_cache`
turned the build-and-prime of the 8,448-copy programs from 241 to 350 seconds (the canary, cache
cold) into 101 to 175 seconds (the real job, cache warm), and the first iteration from 15 to 77
seconds into about one second. Pinning each real job to the node its canary ran on is what made that
possible, because the cache is node-local.
