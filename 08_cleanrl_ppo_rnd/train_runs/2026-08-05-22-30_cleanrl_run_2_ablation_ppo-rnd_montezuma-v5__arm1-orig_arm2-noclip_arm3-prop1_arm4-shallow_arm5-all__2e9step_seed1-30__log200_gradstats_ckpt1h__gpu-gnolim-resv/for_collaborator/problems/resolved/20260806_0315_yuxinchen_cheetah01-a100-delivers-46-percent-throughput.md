# cheetah01's A100 delivered 46% of the same-arm median — slower than every 1080/P100 node

- **Reported by:** yuxinchen (collaborator), 2026-08-06 03:15
- **Job ids:** 6534101 (cheetah01, 1 slot — cancelled by me 03:12) → moved to 6534135 (affogato11)
- **Affected run:** run 79, `arm5_all`
- **Nothing is broken.** The run trained correctly the whole time; it was simply slow. Raising it
  because `cheetah01__ppo-rnd-atari.slurm` exists, so the owner's own top-ups can land there too.

## The measurement

Run 79 on cheetah01's A100-PCIE-40GB held **1,349 steps/s** — both its last-interval and its
cumulative rate, so this was its speed from the first update, not a dip.

| comparison | steps/s | run 79 vs it |
|---|---|---|
| run 79, cheetah01, A100 | 1,349 | — |
| same arm (`arm5_all`) elsewhere, n=21 | median 2,955 (2,415–4,109) | **46%** |
| my whole allocation, n=49 | median 2,771 | 49% |
| slowest other node class I hold (A16 / GTX 1080) | ~2,580 | 52% |

The arm is not the explanation: across all 105 runs the per-arm medians span only ~7%
(`arm1` 3,141 … `arm3` 2,816), and run 79 was compared against its own arm. At the same moment run
79 sat at 3,276,800 steps while most of my runs were at 9,830,400 — three times further on.

## It is not a CPU allocation fault

Checked on the node before moving anything:

- `Cpus_allowed_list` of the trainer was `11-14,27-30` — its full 8 threads, as asked.
- The trainer was burning **665% CPU** across 13 threads, i.e. saturating those 4 cores.
- The A100 meanwhile idled at **2% utilisation, 765 MHz, 36 W**, with 4,169 MiB resident.
- Node was not oversubscribed: `CPUAlloc=26 / CPUEfctv=28`, `CPULoad=10.28`, and the other three
  jobs there (user `pw7nc`, 6 CPUs each) accounted for only ~4 cores of load.

So the run was CPU-side bound and using everything it was given, yet produced half the steps a
GTX 1080 produces from the same 8 threads. That points at the node's memory/CPU behaviour rather
than at the allocation or the GPU — the same shape as the jaguar03 episode in the cluster notes,
where cores looked busy but memory latency was 4x normal. I did not run a latency probe; that would
mean putting a benchmark on someone else's node, which is not mine to do.

## What I did

Cancelled 6534101 (verified in my own id file first) and resubmitted the slot to affogato11
(RTX 2080 Ti, job 6534135, confirmed training). I added cheetah01 to the skip list in my own
placement script so my top-ups will not land there again.

**One marker needs the owner's monitor:** run 79's marker is in `running/` from the cancelled job, so
it wants `monitor.py --requeue`. It had ~79 minutes of progress and the run checkpoints hourly, so a
resume should pick up most of it. I did not touch the marker.

## Suggestion

Worth a look before the owner's next top-up: if cheetah01 reproduces this for the owner's uid too, it
is worth dropping from the node set for this campaign, since its A100 currently returns less than a
GTX 1080 for the same 8 CPU threads. If it is specific to the current co-tenants, it may clear on its
own — the co-tenant jobs there have been running 5–14 hours.

---

## Resolution (owner, 2026-08-06 03:50)

**Accepted, and cheetah01 is parked.** Its submission script has been moved to
`slurm/submission_script/parked/`, with the measurement and the reasoning recorded beside it. Both
launchers resolve a node to its script and refuse a node that has none, so the class is now
*enforced* out rather than merely advised against — neither your top-ups nor the owner's can land
there. `parked/README.md` says what to re-measure before unparking, since this may be the
co-tenants rather than the node.

The judgement is straightforward once the campaign's binding constraint is named: it is CPU threads,
not GPUs. A node returning 46% of the steps for the same 8 threads costs the campaign directly, and
an A100 that trails every GTX 1080 in the set is not worth a slot on those terms.

Your run 79 marker is requeued.

## The part that matters more than cheetah01: run 79 had no checkpoint

Chasing your report turned up a defect in the owner's trainer, and it would have cost the campaign
far more than one slow node.

Run 79's trainer log ends:

```
update=200/122070 global_step=3276800 steps_per_second=1349 ...
[signal] caught signal 15; will checkpoint at the next iteration boundary
```

and no checkpoint file exists. It ran 79 minutes and left nothing to resume from, so the requeue
restarts it at zero.

**Cause: the checkpoint was gated on a logging update.** The condition was

```python
due = time.time() - last_checkpoint_time >= args.checkpoint_every_seconds
if due and update % args.log_every_updates == 0:
```

so an "hourly" checkpoint could only actually be written at a multiple of 200 updates. On a fast node
a logging interval is about 16 minutes and the rounding is invisible. On cheetah01 at 1,349 steps/s a
logging interval is **40 minutes**, so the first checkpoint could not land before 80 minutes — and
the run was cancelled at 79. The two defects compounded exactly: the slow node made the checkpoint
gap twice as wide, and then the run was moved off the slow node before it closed.

**Fixed:** the checkpoint is now written at any iteration boundary once it is due. The alignment it
was protecting is not needed — a format-2 checkpoint carries its own copy of the history, so a
resume rebuilds from the checkpoint and truncates the record to that position whatever update it
sits at. 103 of the 106 runs that have passed the one-hour mark already have checkpoints, so the
mechanism itself was sound; only its gate was wrong.

**Also fixed:** your job's log was 85 bytes, holding only slurmstepd's cancellation line — the
manager's own account of the shutdown was block-buffered and lost at exactly the moment it was
wanted. Every submission script now runs `worker_manager.py` under `python3 -u`.

Both changes apply to runs started from now on; runs already executing carry the old code until they
are restarted.

Four reports, four real defects, and this one found a data-loss bug that would have fired on every
walltime kill of every slow run for the rest of the campaign. Thank you.
