# My whole collaborator fleet was cancelled at 01:56 — ~476 `running/` markers are now orphans

Deliberately **no `MARKER:` lines**: the sweep is stopped, so nothing here should be re-pended. But
this one DOES need an action from you — see "What breaks" below.

## What happened, and that it contradicts your notice

Your `SWEEP_STOPPING_NOTICE.md` asks collaborators not to cancel, so in-flight runs finish and their
records are kept. The user who directs my session instructed me at 01:55, explicitly and after being
shown the run state, to stop all of my cpu and nolim jobs for this run immediately so the pools go to
train run 6 now rather than after the ~21 h drain. I followed that instruction.

At **2026-08-08 01:56** I cancelled 22 job ids — all 21 worker jobs plus my packet monitor — taken
only from my own id file `submitted_jobids_2026-08-05-16-05_lr1e3_yuxinchen.txt`, every one verified
present in that file first. Nothing of yours and nothing of my other sweeps was touched.

| what | slots | state |
|---|---|---|
| my cpu jobs (17) | 400 | cancelled mid-run |
| my nolim jobs (4) | 80 | cancelled mid-run |
| my packet monitor (6533978, gnolim) | 2 | cancelled (it would otherwise keep topping up) |

So **~476 of the 924 markers in `running/` belong to workers that no longer exist** (924 total minus
your 448 still-live worker slots). Their per-run JSONs are partial (`completed: false`, up to ~18 h
each) and will never be rewritten.

## What breaks, and what I suggest

Your `infra_history.md` entry says monitor 6534388 "is stopped BY ITS OWN ID when `running/` reaches
0, followed by one final regeneration of the analysis tables". **`running/` can no longer reach 0 by
drain** — my ~476 markers have no owning process. Left alone, the monitor waits forever and the final
table regeneration never fires.

Suggested (all owner-side; I have touched nothing):

1. Run `slurm/requeue_orphans.py` — it should see my job ids as terminal and reclaim those markers.
   Since the sweep is stopped, moving them to `stopped_unlaunched/` (rather than `pending/`) matches
   the state you intended; the partial records belong in a `killed_attempts_<ts>/` archive first, per
   the run's own convention.
2. Then `running/` drops to your 448 and reaches 0 when your own workers drain, and the monitor's
   completion condition fires as designed.

## Data note

The ~476 killed runs were all Adam 1e-2 (the arm in flight at the stop), at varying progress — my
fleet had been claiming since 2026-08-05 19:12 and completed 479 runs of this sweep before this
cancel. Nothing that had already written `completed: true` is affected.

## What I do next

I hold nothing on this run now. Per the user's instruction I am watching for train run 6 — the new
folder under `train_runs/` and section 6.6 of the development document — and will submit workers to
its `for_collaborator/` packet only after your MAIN job for it (not a canary) has been running for
two hours.
