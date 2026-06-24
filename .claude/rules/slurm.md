---
name: slurm
description: RND's own Slurm usage convention — numbered per-partition agent scripts fanned out at one wandb sweep by 00_batch_slurm.sh. Different from other projects' Slurm usage; keep it project-level.
---

# RND Slurm usage convention (project-specific)

RND submits Slurm jobs differently from the other projects on this cluster — this convention is
project-level on purpose; do not fold it into the global cluster rule.

- **Job scripts:** numbered `slurm/0X_run_<partition>.slurm`, one per partition
  (`gpu` / `gnolim` / `cpu` / `nolim`, plus reservation variants `04_run_reservation_gpu.slurm` and
  `05_run_reservation_cpu.slurm`). Each is minimal: `#SBATCH` headers + `srun wandb agent $1`.
- **Fan-out:** `slurm/00_batch_slurm.sh` submits many `sbatch` jobs (e.g. 15 gpu + 10 gnolim + 20 cpu
  + 20 nolim) **all pointed at one W&B sweep id**, then polls `squeue`. Cleanup helper:
  `cancel_jobs_nodes_ge3.sh`.
- **Env activation:** the `.slurm` scripts do NOT activate conda — the launching shell must have the
  `exploration` env active before `sbatch`, so each `wandb agent` worker runs under it.
- **Collaborator:** a parallel copy of the job scripts lives in `slurm_yuxin/` (owner `yuxinchen`).
- **Shared cluster facts** (jaguar03 specs, gpu-partition CPU sizing, network) live in the global
  `cluster-slurm.md` rule; this file is RND's *usage* pattern only.
