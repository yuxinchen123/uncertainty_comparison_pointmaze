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

## 3. Job names — make the buckets look like unrelated projects
Slurm always shows the owning uid (`squeue -u sl5nw` reveals every job regardless of name), so names are
cosmetic; still, give each bucket a distinct, unrelated-looking codename so the *job-name* column doesn't read as
one project taking a lot of the cluster. One name per bucket (per reserved node + per partition); many jobs share
a name, each still ntasks=8. So an admin sees at most (#partitions + #reserved-nodes) names. Suggested map (swap
freely):

| bucket | job-name |
|---|---|
| reserved node jaguar03 | `diff-prior` |
| reserved node puma01 | `seq2graph` |
| partition cpu | `tab-bench` |
| partition gpu | `meta-icl` |
| partition nolim | `causal-rep` |

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
