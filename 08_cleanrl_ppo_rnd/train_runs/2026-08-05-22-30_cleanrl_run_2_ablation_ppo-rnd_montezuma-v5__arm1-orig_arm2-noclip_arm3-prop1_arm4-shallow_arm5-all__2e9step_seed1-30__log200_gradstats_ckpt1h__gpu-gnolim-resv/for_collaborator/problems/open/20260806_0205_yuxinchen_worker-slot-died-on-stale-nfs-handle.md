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
