# RND slurm submission convention (reservation-aware) — project rule

How to launch RND training sweeps (07_reconstruction) from Claude Code. The config source is a **local file
work queue, not a W&B sweep** (see `run-id-and-logging.md`): `slurm/launch_queue.sh` submits the jobs, each
running `srun slurm/worker.slurm` with **ntasks=8, cpus-per-task=2** (16 CPUs, **one node** `--nodes=1`), so
each job runs 8 work-queue workers in parallel; submit MANY jobs (64 jobs => 512 workers but only **64 squeue
IDs**, so an admin sees 64, not 512). Each worker atomically claims a config from `queue/pending/` and runs
`train.py` with `use_wandb=False`, logging to a per-run JSON under `data/local/`. Per-run slurm scripts live in
`train_runs/<run>/slurm/` and write output into `train_runs/<run>/logs/`. This supersedes the old `wandb
agent` fan-out in `slurm.md` (and the `wandb agent` description in CLAUDE.md's Slurm section). Shared cluster
facts (jaguar03 specs, the jaguar03/puma01 reservation, qos) live in the global `cluster-slurm.md`.

## 0. NEVER cancel jobs not submitted in THIS session (hard rule — see also cluster-slurm.md + global CLAUDE.md)
`scancel -u sl5nw` (or `-t PD`/`-t R`, or `-n <name>` since names like `tab-bench`/`meta-icl` can be reused by
another session) cancels **every** job under the uid — including other concurrent Claude sessions' jobs. This has
caused real damage. **Only cancel job IDs you submitted for this run, read from the run folder.** As you
submit, append each `sbatch` id to `train_runs/<run>/slurm/submitted_jobids.txt` (`id=$(sbatch ... | grep -oP
'[0-9]+$'); echo "$id" >> .../submitted_jobids.txt`). **After submitting, the monitoring loop (every ~10 min)
refreshes that id file** so it stays current; cancel only those ids. **Do NOT ask the user for job ids** (they
leave right after submitting — the id file is the source of truth) and never blanket-cancel.

## 1. Check for an active reservation first, then branch (dynamic — sometimes there is none)
Discover it at submit time; never hardcode:
```
scontrol show reservation -o | grep -i 'Users=.*sl5nw' | grep -oP 'ReservationName=\K\S+'
```
- **If a reservation is returned** (e.g. `sl5nw_151`): read its nodes and per-node CPUs
  (`scontrol show reservation <name>` → `Nodes=jaguar03,puma01`; `scontrol show node <node> | grep -oP
  'CPUTot=\K[0-9]+'`). Fill each reserved node: `jobs_per_node = floor(CPUTot / 16)`; submit that many with
  `--reservation=<name> --nodelist=<node> --nodes=1` (add `--qos=csresnolim` if required). Drop the reserved
  nodes from the normal allowlist (step 2/3).
- **If none is returned:** skip the reservation step; jaguar03 (our group node) is then not reserved, so it
  **joins the normal gpu allowlist** as an ordinary node (no `--reservation`).
Either way, proceed to step 2.

## 2. Open-partition jobs (always)
Same per-job shape (ntasks=8, cpus-per-task=2, `--nodes=1`):
- **10** jobs to `nolim`
- **20** jobs to `cpu`   (20 × 8 × 2 = 320 CPUs)
- **10** jobs to `gpu`   (`--gpus-per-node=0`; workload is CPU-bound and the driver falls back cuda→cpu).
  Submit **one allowlisted node per job** (`--nodelist=<node> --nodes=1`, round-robin across the allowlist in
  step 4) — an **include** list, never `--exclude`, so future high-end GPUs never auto-qualify. Never put a
  reserved node into a normal `--nodelist` (a multi-node `--nodelist` requires ALL nodes, so one busy/reserved
  sibling blocks the whole job — verified, see step 5).

## 3. Job names — at most TWO base names per launch; number the buckets (revised 2026-07-04)
Slurm always shows the owning uid (`squeue -u sl5nw` reveals every job regardless of name), so names are
cosmetic; still, keep the *job-name* column from reading as one project taking a lot of the cluster.
- **Pick at most 2 base codenames per sweep launch.** When more distinct labels are needed (one per
  partition bucket, per reserved node, controller, ...), distinguish them with a NUMBER SUFFIX on the
  same base — `sparse1`, `sparse2`, `sparse3`, ... — instead of inventing a new codename per bucket.
- Many jobs still share one name (a bucket's jobs are interchangeable); the suffix map is per launch
  and recorded in the launch script. Example map for a run-3.2.1-style launch (swap the bases freely,
  keep them distinct from names other sessions are currently running):

| bucket | job-name |
|---|---|
| partition cpu | `sparse1` |
| partition gpu | `sparse2` |
| partition gnolim | `sparse3` |
| partition nolim | `sparse4` |
| reserved node jaguar03 | `mixr1` |
| reserved node puma01 | `mixr2` |
| controller (1 CPU) | `mixr3` |

- The monitoring loop and the id-file refresh match jobs by these names; a base-plus-suffix scheme
  keeps that a single prefix match (`grep -E "sparse[0-9]|mixr[0-9]"`).

## 4. Curated gpu allowlist (include, not exclude)
Submit gpu jobs one node per job from this allowlist. It holds only clearly-lower-tier nodes; everything at
A4000-tier or newer is left off, so future high-end GPUs are excluded by default. Off-list tiers: A100, H100,
A40, A4500, A4000, Quadro RTX 6000, A16, RTX 4000 Ada, RTX 5080. jaguar03 (our group node, RTX A4500) is
**dynamic**: used via the reservation while reserved, else added to this allowlist as a normal node.

| node | GPU | CPUs | include? |
|---|---|---|---|
| adriatic01–06 | quadro_rtx_4000 | 32 | yes |
| affogato11 | rtx_2080_ti | 32 | yes |
| affogato13–15 | gtx_1080_ti | 32 | yes |
| ai01–04, ai06 | rtx_2080_ti | 32 | yes |
| cheetah03 | rtx_2080_ti | 72 | yes |
| jaguar05 | quadro_rtx_4000 | 16 | yes |
| lynx01 | titan_xp | 32 | yes |
| lynx02–04 | gtx_1080_ti | 32 | yes |
| lynx05–07 | tesla_p100 | 32 | yes |
| lynx10 | rtx_2080_ti | 32 | yes |
| jaguar03 | rtx_a4500 | 224 | group (dynamic, see step 1/5) |
| cheetah02 | rtx_4000_ada | 72 | no (Ada) |
| cheetah08–09 | rtx_a4000 | 40 | no (A4000) |
| jaguar01, jaguar06 | a40 | 64 / 48 | no (A40) |
| jaguar02 | a16 | 32 | no (A16) |
| lotus | quadro_rtx_6000 | 80 | no (RTX 6000) |
| nekomata01 | rtx_5080 | 24 | no (new) |
| cheetah01, cheetah04 | a100 | 32 / 256 | no (A100) |
| serval03, serval06–09 | h100_nvl | 128 / 64 | no (H100) |

Re-check node inventory periodically: `sinfo -p gpu -N -o "%n | cpus=%c | gres=%G"`.

## 5. Reserved-node behavior (verified with `sbatch --test-only`, non-submitting)
A reserved node (reservation flags `OVERLAP,IGNORE_JOBS,SPEC_NODES`) cannot be used by a normal job:
- normal job → free allowlist node (`-w lynx01`) → schedules. ✅
- normal job `-w jaguar03,lynx01` (no `--reservation`) → `allocation failure: Requested node configuration is not
  available`. ❌
- normal job `-w jaguar03` (no `--reservation`) → same failure. ❌
- job `-w jaguar03 --reservation=<name>` → schedules. ✅

So: while reserved, route jaguar03/puma01 only via `--reservation`, keep reserved nodes out of every normal
`--nodelist`, and submit normal gpu jobs one allowlisted node at a time. Re-run these `--test-only` probes at
launch (the reservation name/nodes change over time).

## 6. wandb rate limit — submit in batches
Do **not** submit all jobs at once. Submit at most **10 jobs, then sleep 30s**, then the next 10, until all are
in.

## 7. Single-cpu multi-task job shapes: always `--ntasks-per-core=2`, then verify AllocCPUS (added 2026-07-02)
All nodes have `ThreadsPerCore=2`. A job asking `--ntasks=16 --cpus-per-task=1` WITHOUT
`--ntasks-per-core=2` gets each task charged a full core (`AllocCPUS=32`, double the intent), can have two
tasks bound to ONE hardware thread (workers at ~50% CPU while allocated cores idle), and is rejected
outright on 16-core nodes (adriatic/affogato class). This bit run 3.1.2's arm-B fleet. Therefore:
- any job with `cpus-per-task=1` and many tasks MUST carry `--ntasks-per-core=2`;
- after submitting a NEW job shape, verify `sacct -j <id> --format=JobID,JobName,AllocCPUS,NodeList`
  shows the intended AllocCPUS before assuming capacity math holds; spot-check worker `%CPU` on the node
  (healthy = ~100 x cpus-per-task).
Also (verified 2026-07-02): **reservation jobs count toward the per-user partition caps** — puma01
reservation CPUs eat the cpu partition's 400/user, jaguar03's eat the gpu partition's 400/user — so
compute open-partition headroom as 400 minus BOTH normal and reservation usage in that partition. Read the
true counted usage with `scontrol show assoc_mgr qos=cspartcpu flags=qos` (format `limit(current)`).

## 8. Submission ORDER: open partitions FIRST, reserved nodes LAST (probe-verified 2026-07-02)
Reservation jobs (`--qos=csresnolim`, flag `OverPartQOS`) are ADMITTED even when the per-user partition
counter is already at 400/400 (verified: a probe started on puma01 at `cpu=400(400)`), but their usage
still fills that counter against later NORMAL jobs. Order therefore sets the ceiling:
- reserved-first (the old habit): reservation usage eats the 400s -> open ceiling shrinks accordingly.
- **open-first, reserved-last: normal jobs fill every open cap, reservation jobs ride ABOVE the caps.**
Per-user pools are INDEPENDENT per partition — count all five: cpu 400 + gpu 400 + nolim 80 + gnolim 80
(gnolim = old-tier gpu nodes ai05/07-10, jinx01-02, titanx03; own QOS `cspartgnolim`; usable for CPU-only
jobs with `--gpus-per-node=0`) + reserved jaguar03 224 + puma01 160 ≈ **1344 CPUs total ceiling**.
So: submit the open-partition buckets to their caps first, then fill the reserved nodes. (Run 3.1.2
launched reserved-first and paid the difference; step 1's reservation-first phrasing above is superseded
by this ordering for capacity purposes — the reservation checks in step 1 still apply, just last.)

## 9. Multi-task srun jobs MUST pass `srun --wait=0` (WaitTime=3600 kills siblings; verified 2026-07-04)
This cluster's `slurm.conf` sets `WaitTime=3600`: by default, srun TERMINATES a step's remaining tasks
3600 s after the FIRST task exits (job then shows FAILED 9:0, step CANCELLED, workers SIGKILLed with
their queue markers left in `running/`). Any job whose tasks can exit at different times — in particular
a work-queue job whose workers exit when `pending/` is empty — must run `srun --wait=0` (0 = never kill
remaining tasks). Verified 2026-07-04 on the run-3.1.2 follow-up sweep (single-wave queue, one run per
worker): 3 jobs whose SPARE workers found the queue empty died at exactly 1 h 00 m; 10 more died ~1 h
after their fastest worker finished (12–18 h in); 94 mid-flight runs were killed. The 3.1.2/3.1.1 sweeps
never hit this only because their queues kept every worker busy to the walltime. Both worker scripts
(`worker_16x1.slurm`, `worker_8x2.slurm`) now carry `srun --wait=0`; keep it in any new worker script.
Recovery for killed configs: archive partial JSONs to `data/<sweep_id>/killed_attempts_<ts>/`, requeue
markers, resubmit (see `run-id-and-logging.md`).
`--wait=0` semantics VERIFIED by controlled probe 2026-07-04 (jobs 6382110/6382111, logs in the run
folder's `logs/wait_probe/`; 4 tasks, task 0 exits at t=0, tasks 1-3 run 300 s): `--wait=30` killed the
survivors exactly 30 s after the first exit (FAILED 9:0 — the event's signature); `--wait=0` ran every
task to natural completion (COMPLETED 0:0). So 0 = unlimited wait / never kill — it does NOT mean "kill
immediately" (man srun: "A value of 0 indicates an unlimited wait").

## 10. Always leave 16 CPUs of headroom for the user's own jobs (added 2026-07-04)
The uid is shared: the user submits their own jobs as the same user (`sl5nw`), so sweep jobs compete
with the user's manual submissions for the same reserved nodes and the same per-user partition caps.
Therefore, at every sweep launch and every refill wave:
- **If a reservation is active:** leave at least 16 CPUs unallocated on the reserved node. Count other
  already-running jobs on the node first (`squeue -w <node>`); with 32-thread jobs on jaguar03
  (224 threads) that means at most `floor((224-16)/32)` = 6 sweep jobs.
- **If no reservation is active:** leave at least 16 CPUs of headroom under the gpu partition's
  per-user cap instead — fill sweep jobs only up to 384 of the 400 gpu-cap CPUs, counting every
  existing job under the uid (`scontrol show assoc_mgr qos=cspartgpu flags=qos` shows the counted
  usage as `cpu=400(<current>)`).

## 11. Task count must not exceed the node's PHYSICAL cores; pick the job shape per node class (probe-verified 2026-07-04)
**The fact:** sbatch validates `--ntasks` against a node's CORE count, not its hardware-thread count:
a node with 16 cores / 32 threads REJECTS `-n32 -c1` at submit ("Requested node configuration is not
available") in every variant — with `--ntasks-per-core=2`, with `--hint=multithread`, or bare — while
`-n16 -c1 --ntasks-per-core=2` schedules there (tasks pack two-per-core onto sibling threads fine,
but ONE JOB's task count may never exceed the node's core count). Verified with `sbatch --test-only`
probes on affogato13 (2 sockets x 8 cores x 2 threads) during the run-3.2.1 launch.

**Second fact — every node reserves one core for the system (verified 2026-07-04, cluster-wide):**
`scontrol show node` shows `CPUEfctv = CPUTot - 2` on every class checked (affogato/ai 30 of 32,
bigcat 62 of 64, cheetah03 70 of 72, jaguar03 222 of 224, puma01 158 of 160, heartpiece 38 of 40,
slurm2 46 of 48). Only CPUEfctv threads are allocatable — a set of jobs summing to CPUTot on one
node leaves the last job PENDING (Resources) forever. Size whole-node fills against CPUEfctv.

**The policy (how to fill each node class with single-cpu workers):**
- **Nodes with >= 32 physical cores: prioritize the 32x1 shape** (`--ntasks=32 --cpus-per-task=1
  --ntasks-per-core=2 --mem-per-cpu=2G` = 64G/job) — fewer squeue ids per worker. Verified-current
  32-core-plus nodes: bigcat01-06, cortado01-10, affogato02, cheetah03, heartpiece, slurm2-4,
  jaguar03, puma01.
- **Nodes that only allow up to 16 tasks (16-core / 32-thread class: adriatic, affogato, ai):
  submit 16x1 jobs — ONE 16x1 PLUS ONE 14x1 per node.** One 16x1 job with `--ntasks-per-core=2`
  occupies 16 of the 30 allocatable threads; a second 16x1 does NOT fit (16+16 = 32 > 30 = CPUEfctv
  — verified: it pends on Resources forever) — the second job must be 14x1 (16+14 = 30, fills the
  node exactly; both jobs started immediately at the run-3.2.1 launch after this correction).
- **Memory when pairing 16x1 + 14x1 on a 64G node (ai01-04):** 2G-per-cpu for both needs
  32768 + 28672 = 61440M <= 64000M (fits), but for safety margin the run-3.2.1 pairing used 2G for
  the 16x1 + 1900M for the 14x1 (32768 + 26600 = 59368M). 128G nodes (affogato13-15, ai08-10) take
  2G-per-cpu for both without thought.
- After submitting, verify per job: 32x1 -> AllocCPUS=32; 16x1 -> AllocCPUS=16; 14x1 ->
  AllocCPUS=14 (§7's check).

## 12. Capacity planning: target the five per-user pools, then fill fragments with odd-size jobs (added 2026-07-04)
**Plan every sweep launch backwards from the pool ceilings, not forwards from a job-shape count.**
The per-user pools are independent (§8): **cpu 400 + gpu 400 + gnolim 80 + nolim 80**, plus the
reserved nodes riding above the caps (jaguar03 and puma01 at their ALLOCATABLE size, CPUEfctv =
CPUTot - 2 per §11; puma01 additionally memory-capped), minus the §10 16-CPU user headroom on the
reserved node. The run-3.2.1 launch under-filled by sizing buckets as "N jobs of the preferred
shape" (nolim and gnolim got 2 x 32 = 64 of their 80s; no fragment was filled anywhere) — plan the
POOL TOTALS first, then choose shapes that sum to them.
- **Odd-size single-cpu jobs are fine and expected as gap fillers**: 30x1 on a 64-thread node's
  second half (62 allocatable - 32), 16x1 + 14x1 pairs on the 16-core class (§11), 22x1 to land a
  partition counter exactly on its cap, 14x1 to top nolim off at 80, 6x1 onto cheetah03's tail.
  Same flags as every worker job: `--ntasks-per-core=2`, `srun --wait=0`, 2G per cpu (1900M on the
  64G ai nodes); verify AllocCPUS == ntasks after each new shape (§7).
- **Displacement at top-up time:** a normal job is admitted only while
  (running usage + ask) <= the partition cap, and RUNNING reservation jobs count into that usage
  (§8). So when topping up, compute open-partition room as cap - (running open + running reserved)
  in that partition, and keep submitting open-partition fillers BEFORE adding more reservation
  load — e.g. do not add extra puma01 jobs while open cpu jobs are still trying to start, or the
  reservation usage eats their admission room.
