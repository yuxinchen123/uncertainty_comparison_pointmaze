# Sweeps in this run folder (Adam learning-rate addendum, 1e-3 and 1e-2)

90 configurations (2 predictor learning rates x 3 environments x 15 bonus weights) x up to 300 seeds at 1,000,000 env steps, raced against a per-environment bar frozen before launch from the matching Adam 1e-4 configuration (slurm/truncation_controller.py + slurm/FROZEN_BARS.json).

| sweep_id | runs | layout | status |
|---|---|---|---|
| 2026-08-05-16-05_lr1e3 | 4500 | 45cfg x 100seeds x 1000000 steps | grown in place 2026-08-06, see the row below |
| 2026-08-05-16-05_lr1e3 | 27000 | 90cfg x 300seeds x 1000000 steps | active (same sweep_id: extended live by slurm/extend_queue.py, nothing stopped) |
