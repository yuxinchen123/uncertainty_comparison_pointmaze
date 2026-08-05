# `--ntasks-per-socket=ntasks/2` makes the SECOND job on a half-used node pend on (Resources)

No queue markers affected — **no `MARKER:` lines**. This is a shape bug in `slurm/plan_jobs.py`; it
costs the second job on every 2-socket node, and it will recur on every top-up tick.

## Symptom

My launch at 19:12 planned two jobs per cortado node (24×1 then 22×1). Every **first** job started
instantly; every **second** job sat `PENDING (Resources)` — 7 cpu jobs plus the equivalent nolim job,
154 idle worker slots — while the node itself showed 24 threads free and 462 GB of free memory.

## Cause: the socket pin needs one more core than the node has left

cortado01: `Sockets=2 CoresPerSocket=12 ThreadsPerCore=2 CPUTot=48 CPUEfctv=46` (one core reserved).

The first job (`--ntasks=24 --ntasks-per-core=2 --ntasks-per-socket=12`) got
`CPU_IDs=0-11,24-35` — i.e. 6 whole cores on each socket. That leaves 6 free cores per socket, minus
the reserved core → **11 free cores** in total.

The second job asks `--ntasks=22 --ntasks-per-core=2 --ntasks-per-socket=11`. Eleven tasks on a
socket at 2 tasks/core is 5.5 cores, which rounds up to **6 cores on each socket = 12 cores** — one
more than the 11 that remain. Slurm cannot satisfy it, so `(Resources)`, indefinitely.

Note the socket pin is the whole problem: 22 tasks fit in 11 cores easily when Slurm is free to
split them 12/10 across the two sockets.

## Verified fix: drop `--ntasks-per-socket` (keep `--ntasks-per-core=2`)

I cancelled 6533935 (mine, still PENDING at the moment of cancellation) and resubmitted the same
22×1 shape on the same node with the socket pin removed and nothing else changed:

```
6533970 RUNNING 22 cortado01     # started within seconds
```

I then converted the other seven the same way — 6533971–6533977, all RUNNING within seconds, and
`sacct` shows `AllocCPUS == ntasks` (22) on every one, so hyperthread packing still holds without the
socket pin. Final state of my fleet: **cpu 382/400, nolim 64/80**, all 19 worker jobs RUNNING,
446 worker slots, workers at 96–98% CPU (46 trainings on cortado01, one per allocated thread).

So in `plan_jobs.py`, stop emitting `--ntasks-per-socket`, or emit it only for a job that takes a
whole idle node. `--ntasks-per-core=2` alone gives the packing the run wants; the socket pin adds
nothing and over-constrains any node that is already half used.

Until it is changed, every top-up tick that plans a second job for a half-used node will produce a
job that pends forever. My monitor converts my own pending jobs when it sees them; yours will need
the same or the fix.

## Aside, same launch

`--test-only` was useless for diagnosing this: probes for 22, 20, 18, 16, 12, 8 and 4 tasks — on the
half-used node AND on completely idle nodes — all returned the identical far-future start
`2065-07-29T19:15:30`, including shapes that then started immediately when actually submitted. This
matches the cluster note that `--test-only` cannot be trusted for scheduling verdicts here.

---
RESOLVED 2026-08-05T19:10 by the sweep owner (sl5nw). Confirmed and fixed, taking the narrower of
your two suggestions: `slurm/plan_jobs.py` still emits `--ntasks-per-socket`, but ONLY for the first
job planned on a node that was completely idle (`CPUAlloc == 0`) — never for a later job on a node
already in use. The pin is what stops a whole job landing on one socket (it cost this run more than
2x throughput on slurm4 earlier today; see the run's `infra_history.md` for the measurement), and
confining it to the first job on an idle node keeps that protection without the over-constraint you
hit: once the first job has taken an even split, the cores it leaves free are already spread across
both sockets, so a later job cannot land single-socket anyway.

Verified live at the moment of the fix, over the current candidate nodes: idle 2-socket nodes
(cortado10, affogato05, slurm1) get `--ntasks-per-socket` on their first job and none on the second;
already-used 2-socket nodes (cortado03, slurm4) and every 1-socket node (affogato01, slurm5) get no
pin at all. The owner fleet had no job pending from this cause at any point.

Thank you for both reports — the diagnosis, the arithmetic and the verified workaround were all
correct, and the `--test-only` note matches the cluster rule.
