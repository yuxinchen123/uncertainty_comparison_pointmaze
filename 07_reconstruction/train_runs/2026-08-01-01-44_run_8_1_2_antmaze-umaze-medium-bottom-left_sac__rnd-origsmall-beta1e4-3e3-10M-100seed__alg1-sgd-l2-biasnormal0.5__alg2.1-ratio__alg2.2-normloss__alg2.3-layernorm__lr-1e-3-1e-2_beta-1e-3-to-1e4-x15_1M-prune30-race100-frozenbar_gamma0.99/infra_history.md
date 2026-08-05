# Infra history — train run 8.1.2

## 2026-08-01 03:15 — launch complete

- Canaries (sweep 2026-08-01-02-34_run812-canary): cpu 30x1 (affogato04), gnolim 14x1 (titanx03),
  cuda 1x2 (cheetah02). AllocCPUS matched every ask. The first cuda canary job (6528755) exited
  empty because the gnolim workers (claim order pending_10m first) drained all 8 task-R canary
  markers within seconds; 4 extra task-R canary markers were added and the resubmitted cuda canary
  (6528759) completed two 60k-step baseline runs on cuda (~1.6 GB host RSS per worker at that
  scale, 478 MiB GPU memory for 2 runs).
- Bulk wave (sweep 2026-08-01-02-03_run812, launch commit 9dcb85a): 12x worker_30x1 + 1x
  worker_14x1 on cpu, 2x worker_30x1 on nolim, 39x worker_gpu_10m_1x2 (1 GPU, 2 cuda workers,
  16G each) on fast hosts, 2x worker_gnolim_14x1 on gnolim (10M-first claim order), LAST 2x
  worker_jaguar03_100x1 via reservation sl5nw_151 (--time=1-19:30:00; the reservation ends
  2026-08-04 00:00). Monitor loop job 6528827 (cpu partition); first tick stage1_check PASS.
- cheetah04 and cheetah08 refused new 4-day jobs with "ReqNodeNotAvail, UnavailableNodes" while
  showing mix state (cause not diagnosed; possibly a node-level reservation window). The 8 pending
  task-R jobs there (own ids) were cancelled after the 5-minute grace and resubmitted on the idle
  adriatic04/adriatic05 (3 each) and adriatic01/adriatic03 (1 each) — all 39 task-R jobs RUNNING.
- cpu13 (worker_14x1, job 6528781) pends on QOSMaxCpuPerUserLimit until the cpu canary job's 30
  CPUs free (expected within the hour). Every --time ends before the 2026-08-05 07:30 maintenance;
  all 10M cuda claims fit their jobs' walltime (>= 90 h at claim).

## 2026-08-01 03:56 — collaborator fleet joined; queue accounting verified

The running/ marker count (1628) briefly looked impossible against the owner's 728 workers; the
explanation is that yuxinchen's collaborator fleet went live at ~03:37 using the packet (60 cpu
jobs, prefix hxmxxwbg, ~900 workers on struct/cortado/lynx nodes, their own caps and id file).
728 owner claims + ~900 collaborator claims = 1628 exactly; sampled markers all have claim lines
in for_collaborator/logs/. No action needed. Total fleet ~1,630 worker slots; first stage-1 seed
waves complete correspondingly faster.

## 2026-08-01 14:20 — cuda memory growth measured; task-R gpu fleet reshaped W=2x16G -> W=1x84G

- The per-minute samplers show cuda worker host RSS growing LINEARLY with env steps: ~1.0 GB/h
  per worker (~6.7 KB/step), no saturation after the replay buffer fills (3.07 GB at t+1.5 h ->
  12.31 GB at t+10.7 h; uniform across nodes). cpu workers plateau at ~1.2 GB, so the growth is
  cuda-side (host allocations of the cuda path, mechanism not diagnosed). GPU memory constant at
  234 MiB/run.
- Projection: a 10M-step cuda run needs ~70 GB host RSS. The W=2 x 32G jobs would all have been
  OOM-killed near 2M steps (~17:30 today); no W=2 job could ever finish. This is the endurance
  risk the plan carried (nothing had ever run past 1M steps).
- Action (own ids only): the 40 W=2 task-R jobs were cancelled at ~13:45 (each ~10.7 h in, runs
  unfinishable) and replaced by worker_gpu_10m_1x1.slurm — 1 worker per GPU, --mem=84G (70 GB
  projection + 20%), --time=3-17:00:00 (ends 08-05 06:50, before maintenance),
  WORKER_REQUIRED_10M_HOURS=78 (measured ~67 h per 10M cuda run + margin; the default 90 h guard
  would refuse the pre-maintenance window). 37 of 40 started immediately and claimed 10M items;
  the 3 stragglers (cheetah09 full/unavailable, cheetah04 still refusing with ReqNodeNotAvail as
  at launch) were cancelled after the 5-minute grace — task R runs on 37 GPUs this round.
- Cost of the reshape: the 80 in-flight W=2 runs' ~10.7 h each are lost (restart-from-scratch is
  this run's design; their markers returned to pending_10m via requeue_orphans). Benefit: the
  37 W=1 runs project to FINISH (~67 h -> ~2026-08-04 09:00, before maintenance), where the old
  shape would have delivered zero completed 10M runs and looped OOM kills.
- gpu-partition memory ask now 37 x 84G + 2 x 200G (jaguar03) = 3.5 TB of the 4 TB per-user cap.
- Follow-up for the stage-2 plan: cpu 10M runs (~154 h on gnolim post-maintenance) are projected
  SAFE (cpu RSS plateaus ~1.2 GB) but are equally unverified past 1M — the first gnolim 10M runs
  get the same per-minute sampling before the fleet is widened.

## 2026-08-05 15:45 — sweep STOPPED by the owner (no new work claimed from here on)

- Owner decision (user instruction 2026-08-05): stop this sweep and freeze the writeup on the data
  finished so far, to free the cpu and nolim pools for the new Adam learning-rate 1e-3 run.
- How it was stopped, and what was deliberately NOT done:
  - Every remaining marker was moved OUT of the claim pools into
    `queue/2026-08-01-02-03_run812/stopped_2026-08-05/` — 15,313 from `pending_1m` and 125 from
    `pending_10m`, 15,438 in total. A worker (owner's or a collaborator's) that finishes its
    current run now finds nothing to claim and exits by itself, so the sweep enqueues no further
    work while nothing in flight is destroyed.
  - The two still-running owner jobs (6528781 on cpu, 6528782 on nolim, 44 worker slots) were NOT
    cancelled: their in-flight runs are 0-19 h in and each completed one adds a usable record.
    They release their CPUs on their own as the runs finish.
  - `yuxinchen`'s collaborator jobs cannot be cancelled from this uid and are not touched; they
    drain the same way (empty pools -> workers exit).
- State at the stop: 6,210 markers in `done/`, 0 in `failed/`, 2,552 left in `running/` (mostly
  stranded from earlier walltime kills — never requeued because the monitoring loop was not
  running), 8,764 per-run JSON records on disk.
- The stage-1 race never reached a verdict: no configuration had 30 completed seeds, so
  `stage1_decisions_2026-08-01-02-03_run812.jsonl` records no prune and no survivor. The section
  8.1.2 tables are therefore a stopped-state snapshot, not a finished race.
- To resume later: move the markers back from `stopped_2026-08-05/` into `pending_1m/` and
  `pending_10m/` (the pool each name came from is recorded in the marker's own `pool` field), then
  re-run `launch_queue.sh` with `SWEEP_ID=2026-08-01-02-03_run812`.
