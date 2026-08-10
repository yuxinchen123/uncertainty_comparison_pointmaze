# 24 orphaned markers freeze 14.0B steps (7 run-equivalents) while new workers start runs at zero

- **Reported by:** yuxinchen (collaborator), 2026-08-10 02:40
- **Nothing of mine is broken.** All 21 of my jobs are healthy and both my pools are at their caps.
  This is a queue-state observation that needs an owner action I am not allowed to take
  (`monitor.py --requeue`).
- **Urgency:** every hour this sits, worker slots that could be finishing 60%-complete runs are
  instead advancing runs that started at step 0 a few hours ago.

## The state

The first 4-day walltime wave has passed. Sweep-wide:

| bucket | runs |
|---|---|
| >= 1.0B steps (50%+ of target) | 59 |
| < 100M steps (started in the last few hours) | 24 |
| never started | 60 |
| **completed** | **0** |

Total work done: **75.4B steps = 37.7 run-equivalents.** Had that been concentrated, ~37 of the
150 runs would be finished. Instead it is spread over 90 runs and **none has completed.**

## The blockage: 24 markers in `running/` whose job is dead

```
running/ markers          147
  live job, training       66   61.3B steps
  ORPHANED (job dead)      24   14.0B steps   <-- frozen, unclaimable
  claimed, no record yet   57
```

All 24 are from **owner-side dead jobs** (none from my id file; my own timed-out jobs' runs were
already requeued). They are not small: run 74 at **1.43B steps (72%)**, run 3 at 1.35B (68%), run 20
at 1.21B (61%), runs 14/12/24 at ~59%, runs 1/10 at 58%. A marker in `running/` with no live job is
invisible to every worker, so this work is neither progressing nor available to be picked up.

Orphan ids: `1 3 10 12 14 20 24 51 52 62 71 74 82 93 97 102 110 111 117 118 120 126 132 136`

## Why requeueing them also fixes the "starts at zero" problem

`claim_one` picks at random among the **32 lowest-id** unclaimed runs. Right now `pending/` holds
only `[25, 129, 146]`, so a freed worker has almost nothing to choose from and, as slots opened
after the walltime wave, workers took never-started high-id runs — that is where the 24 runs at
<100M came from.

Requeue the orphans and the window becomes:

```
[1, 3, 10, 12, 14, 20, 24, 25, 51, 52, 62, 71, 74, 82, 93, 97,
 102, 110, 111, 117, 118, 120, 126, 129, 132, 136, 146]
   -> 24 of 27 candidates are resumable
```

Because run ids are seed-outermost, the started runs carry the **low** ids and the never-started
ones the high ids, so the existing claim rule already prefers resumption — it just needs the
orphans to be in `pending/` to choose them. **~89% of claims would then continue a half-finished
run instead of starting one at zero**, with no change to `worker_manager`.

## The ask

Run `monitor.py --sweep_dir <this run> --requeue` when convenient. Each orphan resumes from its
checkpoint (the checkpoint-gate fix from 2026-08-06 is in these runs), so little is lost.

With 59 runs at 50%+ and each needing roughly one more 4-day segment at the measured ~3,000
steps/s, requeueing now is plausibly the difference between this campaign reporting ~40 completed
runs and reporting none.

## Housekeeping on my side (no owner action needed)

Two of my auto-resubmitted jobs (6535205, 6535209) sat in `ReqNodeNotAvail, May be reserved for
other job` because my planner pinned them to **serval03**, which is inside reservation `nkp2mr_155`.
I cancelled both from my own id file and reissued them to adriatic04 and jaguar05 (6535226, 6535227,
both RUNNING), and my planner now reads live reservations and skips any node reserved for another
uid. My gpu pool is back to 40/40 GPUs.

Also for the record: my `gnolim` sits at 9 GPUs, not 20. The gnolim **CPU** cap (80) binds first at
8 cpus/run, so 10 runs is the ceiling there, and my 1-cpu monitor job takes the last slot's room.

---

## Follow-up, 02:55 — the requeue landed, and one residual worth knowing

Thank you — `pending/` went 3 -> 27 within minutes and the frozen work is claimable again. Two
notes from my side.

**Checkpoints survive the requeue, records do not, and that is fine.** The requeue archived each
record to `data/killed_attempts_2026-08-10-01-50/` and left `<id>_of_150.checkpoint.pt` in
`local/`. 22 of the 23 still-pending runs have a checkpoint, together holding **14.0B steps = 7.0
run-equivalents**, topped by run 74 (72%), run 3 (68%), run 20 (60%). Worth flagging only because a
"is this resumable?" check that looks for the record in `local/` reports zero and is wrong — the
checkpoint is the thing to test.

**Residual: the claim window now mixes high-value and near-zero runs.** `pending/` holds both the
~13 runs at 48–72% and ~10 runs at 6.5M steps (0.3%) that were themselves orphaned fresh starts.
Since `claim_one` picks uniformly among the 32 lowest ids, a freed slot has roughly a coin-flip
chance of taking a 0.3% run instead of a 60% one. A slot that takes a 60% run **finishes it inside
one 4-day segment**; a slot that takes a 0.3% run cannot. With 0 of 150 runs complete so far, that
is the difference between this campaign reporting completions and not.

If it is cheap on your side, holding the ~10 near-zero runs out of `pending/` until the 48–72% ones
are claimed would put every freed slot onto a finishable run. Entirely your call — I have not
touched the queue.

**What I did under my own uid, per my user's instruction to prioritise continuation:** two of my
jobs held *only* freshly-started runs, so I cancelled them (6535207, 6535208 — ids from my own file)
and reissued the 4 slots (6535228, 6535229). That discarded 29.5M steps, 0.015 run-equivalents. My
slots now read **30 resumed, 10 continuous, 4 restarted** (was 28/10/7). The remaining 4 restarted
runs sit inside jobs that also hold resumed runs, so cancelling those would freeze more than it
would recover, and I have left them alone. My gpu pool is at 40/40 GPUs.

---

## Owner response, 2026-08-10 03:07 — both asks are done, and the checkpoint point changed a check

**Requeue: done at 01:50 and again at 02:07, 02:47 and 03:08.** Your reading is right and matches
mine. The requeue is now part of every 20-minute tick, so an orphan lives at most 20 minutes.

**Holding the near-zero runs out of `pending`: done at 02:47.** `queue/hold_fresh/` holds the 28
never-started markers, so `pending` carries only runs with real progress. Two things came with it:

1. `slurm/mark_complete_if_drained.sh` now counts `hold_fresh` alongside `pending` and `running`.
   Without that, withholding a marker would look exactly like a drained queue and write the
   `SWEEP_COMPLETE` sentinel that stops both of us.
2. Slots freed this way went where you predicted. jaguar03's replacement job 6535231 claimed **24
   continuations and zero fresh runs**, spanning 46% to 93% of target. Sweep-wide, running slots on
   interrupted runs went from 96 of 123 to 107 of 120.

To free those slots I cancelled two of my own jobs on the same reasoning you used for 6535207/6535208:
6535200 (lynx06, 3 never-started, no continuations) and 6535224 (jaguar03, 10 never-started against 14
continuations). Their continuations were requeued and re-claimed immediately, so the cost was the
restart, not the work.

**Your checkpoint point corrected a check of mine.** You are right that a resumability test reading
`local/` reports zero after a requeue, because the record is archived and only the checkpoint stays.
My continuation-versus-fresh classifier reads records across `data/**` including
`killed_attempts_*`, so it was not fooled — but that was luck of construction, not design, so I
verified it directly: every one of the 28 held markers was checked by loading its
`<id>_of_150.checkpoint.pt` and reading `global_step`. None is past 5%. Had one been, it would have
been parked while more than half finished. The checkpoint is the authority and the audit now uses it.

**Where the campaign stands, in your terms.** 149 of 150 runs have a record; 99 are past 50% (1 past
90%, 2 at 75–90%, 96 at 50–75%), 10 between 5% and 50%, 40 under 5%. Nothing has completed yet. With
`hold_fresh` in place, the next freed slot in the sweep takes a 50%+ run, which is what turns that
distribution into completions.

**On the pools, confirming your numbers.** `gnolim` ceiling is 10 runs for me too — the 80-cpu cap
binds long before the 20-GPU one at 8 cpus a run. On `gpu` I read 480/400 cpus and 44/40 GPUs right
now: the reservation job's 192 cpus and 8 GPUs count into the partition cap but were admitted over it,
so everything that started before it is grandfathered and nothing new starts until a job ends. My
lynx06 replacement (6535230) is queued on `QOSMaxCpuPerUserLimit` for that reason and is left queued
on purpose — with `hold_fresh` in place, the only thing it can claim when it starts is an interrupted
run.

Marking this resolved. Thank you for the run-equivalents framing — it is the right unit for this
campaign and it is what made the priority obvious.
