# A worker slot died on a stale NFS handle in `claim_one`; 2 markers on lynx10 need requeue

- **Reported by:** yuxinchen (collaborator), 2026-08-06 02:05
- **Job ids:** 6534113 (lynx10, cancelled by me at 02:03) → resubmitted as 6534124
- **Log:** `for_collaborator/logs/vlmapiik3_6534113.out`

## What happened

About a minute after 6534113 started on lynx10, one of its three worker-slot threads died:

```
Exception in thread slot_g0_s0:
  File ".../submit-gpu-sweep/scripts/worker_manager.py", line 404, in worker_slot
    claimed = self.claim_one()
  File ".../submit-gpu-sweep/scripts/worker_manager.py", line 314, in claim_one
    cfg = json.load(f)
OSError: [Errno 116] Stale file handle
```

The job stayed RUNNING with 2 of 3 slots (`pgrep -c -f ppo_rnd_envpool_shuze.py` on lynx10 = 2,
against 3 on cheetah02 and 3 on ai09), so one of its GPUs would have idled for the whole campaign.

## No marker was leaked by this

Line 314 is in the **peek** loop, which reads candidate markers *before* the claiming
`os.rename`. So the run this slot was inspecting stayed in `pending/` — nothing moved to
`running/` without a process behind it. This is worth stating explicitly because the same
exception one block later (after the rename) *would* have leaked a marker.

## Cause: a too-narrow `except` in the shared worker_manager

The peek loop guards only the lost-race case:

```python
try:
    with open(p) as f:
        cfg = json.load(f)
except FileNotFoundError:
    continue
```

`FileNotFoundError` does not cover `OSError` errno 116, which this NFS-backed queue directory can
raise transiently when many slots list and read the same directory at once (my 49 slots all claimed
within a few seconds of each other). One transient read then kills the slot thread for good — the
thread is never restarted, so the loss is permanent for the job's lifetime.

Suggested fix in `/p/rlprojects/.claude/skills/submit-gpu-sweep/scripts/worker_manager.py`, since it
affects every sweep run from this skill, not just this one: catch `OSError` (and
`json.JSONDecodeError`, for a marker caught mid-write) in that peek block and `continue`, and
consider wrapping the slot body so a slot survives a transient claim error instead of dying.

## What needs the owner's monitor: 2 markers in `running/`

I cancelled 6534113 and resubmitted lynx10 as **6534124** (3 slots, all 3 trainers confirmed) to
recover the idle GPU. Per the worker_manager contract, SIGTERM leaves the 2 in-flight runs' markers
in `running/`, so they want `monitor.py --requeue` on the owner's next tick. Both runs were about
4 minutes old, so nothing meaningful was lost. I did not move, rename or requeue any marker myself.

Related: `20260806_0150_yuxinchen_smoke-test-probes-nonexistent-data-local.md`.

---

## Resolution (owner, 2026-08-06 02:10)

**Confirmed in every particular, including the part that was easy to get wrong.** You are right that
line 314 sits in the peek loop, *before* the claiming `os.rename`, so no marker was leaked — the run
that slot was inspecting stayed in `pending/`. Stating that explicitly saved a wasted hunt for a
missing marker, and it is the difference between a lost worker slot and lost work.

**Both fixes are in the shared `worker_manager.py`, since this affects every sweep run from the skill:**

1. The peek block now catches `(OSError, json.JSONDecodeError)` instead of `FileNotFoundError`. Three
   things land there and all mean "skip this candidate": a lost claim race, a transient NFS failure
   such as your errno 116, and a marker caught mid-write by a reader.
2. A slot no longer dies on one failed claim. `claim_one` is retried up to `CLAIM_RETRIES = 5` times
   with a widening backoff, and only then gives up with a printed reason. Losing a slot for a job's
   whole lifetime costs a GPU for days; a genuinely persistent fault still surfaces.

Tested by injecting an errno 116 stale file handle on the first marker read: `claim_one` skipped that
candidate, claimed a different run, and the slot kept working. Before the change the same injection
killed the thread.

**Your two markers are requeued.** Job 6534113 was terminal (CANCELLED), so
`monitor.py --requeue` moved run 0 (`arm1_original`, seed 1) and run 56 (`arm2_no_rnd_grad_clip`,
seed 12) back to `pending/`. Neither had written a record — they were about four minutes old and the
first row lands at update 200 — so nothing was archived and nothing was lost. Queue is now 46
pending, 104 running.

**A second defect your report led me to, on the owner's side.** `monitor.py` read only
`slurm/submitted_jobids.txt`, while this sweep follows the shared rlprojects convention of one file
per sweep, `submitted_jobids_<sweep_id>.txt`. So the **owner's** 11 jobs were invisible to the
monitor — it reported "21 running (yuxinchen 21)". That is worse than a cosmetic miscount: a marker
whose claiming job is never queried can never be seen as orphaned, so when one of the owner's jobs
hit its 4-day walltime its runs would have sat in `running/` forever and never been redone. Fixed:
the monitor now reads both spellings. It correctly reports 32 running (owner 11, yuxinchen 21).

**On what you did.** Cancelling your own job to recover an idle GPU, resubmitting it, confirming all
three trainers came up, and leaving the markers for the owner's requeue is exactly right. You touched
nothing shared, and the report named the two things the owner had to do.

**One correction to the packet, unrelated to your report but affecting you.** The launcher had a bug
that would have killed it on first use for anyone with no jobs yet (`grep -c` exits 1 when it matches
nothing, so the `|| echo 0` fallback made `live` the two-line string `0\n0` and the arithmetic
failed). It also had no node-to-script resolution and sized cpus and memory without the runs-per-GPU
factor. All fixed, and it now takes `NODES="node:G ..."` instead of asking you to edit it. Your own
submissions were unaffected — every one is one run per GPU and none is on the reserved node.
