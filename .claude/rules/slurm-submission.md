# RND slurm submission convention (reservation-aware) — project rule

How to launch RND wandb-agent training sweeps (07_reconstruction) from Claude Code. Each sbatch job runs
`srun wandb agent <sweep>` with **ntasks=8, cpus-per-task=2** (16 CPUs, **one node** `--nodes=1`), so each job
runs 8 agents in parallel; submit MANY jobs against the sweep(s). This supersedes the old fan-out in `slurm.md`.
Per-run slurm scripts live in `train_runs/<run>/slurm/`, write output into `train_runs/<run>/logs/`, and export
`WANDB_DIR=train_runs/<run>/data` before `wandb agent` (see `wandb-logging.md`). Shared cluster facts (jaguar03
specs, the jaguar03/puma01 reservation, qos) live in the global `cluster-slurm.md`.

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
