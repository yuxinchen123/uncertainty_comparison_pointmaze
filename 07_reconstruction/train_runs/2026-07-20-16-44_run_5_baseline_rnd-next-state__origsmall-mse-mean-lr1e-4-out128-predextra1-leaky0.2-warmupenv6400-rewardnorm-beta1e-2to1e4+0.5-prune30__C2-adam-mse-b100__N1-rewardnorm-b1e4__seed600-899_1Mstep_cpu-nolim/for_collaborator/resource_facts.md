# Train run 5 — resource facts (for collaborators)

- **SWEEP_ID**: `2026-07-20-16-55_set-baseline` (also in `packet_env.sh`).
- **Run total**: 3000 runs = 10 configs × 300 seeds (600–899). Each run = 1M SAC steps on CPU,
  roughly 9–25 h (median ~15 h).
- **Configs**: 8 "original-small" betas {1e-2…1e4} (these RACE — the owner's controller prunes
  losing betas after ≥30 seeds), plus C2 and N1 (never pruned, 300 seeds each). Full detail in the
  run folder's `experiment_background.md`.
- **Partitions you may use**: `cpu` (cap 400) and `nolim` (cap 80) ONLY. NO gpu partition, NO
  gnolim, NO reservation (user directive 2026-07-20 — the reservation nodes are being retired).
- **Job shape**: 32×1 (or 16×1 filler), `--ntasks-per-core=2`, `--mem-per-cpu=2G`, `srun --wait=0`.
  Verify `AllocCPUS == ntasks` (32 or 16) after your first submit.
- **Env**: `/p/rlprojects/RND/.venvs/exploration/bin/python` (shared canonical; Python 3.11).
- **Time limit**: cpu jobs `--time=4-00:00:00`; nolim jobs a long value finishing before the
  Aug-5 maintenance. `launch_workers_collaborator.sh` sets these for you.
- **Owner**: `sl5nw`. The owner runs the queue controller, requeue, and pruning; you only add
  workers and (if your env breaks) write a problem report.
- **Your caps are independent**: each user has their own cpu 400 / nolim 80. Adding your workers
  roughly doubles the throughput on top of the owner's ~416 slots.
