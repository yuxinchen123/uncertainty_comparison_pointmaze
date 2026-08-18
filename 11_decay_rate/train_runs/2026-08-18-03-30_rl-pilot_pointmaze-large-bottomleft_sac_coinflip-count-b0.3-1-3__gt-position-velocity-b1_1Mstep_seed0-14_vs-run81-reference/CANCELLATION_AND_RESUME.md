# Cancellation record

Cancelled 2026-08-17 23:50 PT by user redirection: end-to-end reinforcement learning is out of
the campaign's scope. The campaign's evaluation stays convergence-rate-only (per-state decay
against min(1, m^-1/2)); generalization is evaluated on AntMaze states and Atari frames as
DISTILLATION point sets, not by running RL.

- Jobs 6539941 and 6539942 (this run's own id file) were cancelled ~50 minutes into training,
  before any run reached its first 50k-step checkpoint — zero per-run JSON records existed, so
  there was nothing to archive.
- The queue markers under `queue/2026-08-17-23-27_cfpilot/running/` are left as they were at
  cancellation (60 running, 0 done); the sweep is NOT to be resumed.
- The `coinflip_count` intrinsic model added to `rnd_exploration.methods` stays in the package
  (additive, unit-tested); only this sweep is withdrawn.
