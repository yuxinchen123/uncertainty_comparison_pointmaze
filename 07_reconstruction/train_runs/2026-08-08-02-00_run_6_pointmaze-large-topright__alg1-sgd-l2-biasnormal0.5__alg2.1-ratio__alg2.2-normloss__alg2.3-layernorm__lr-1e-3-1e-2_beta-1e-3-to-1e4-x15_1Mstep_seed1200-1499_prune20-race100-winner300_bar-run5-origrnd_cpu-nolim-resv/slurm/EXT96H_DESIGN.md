# 96-hour extension sweep (ext96h) — the FINAL design (2026-08-13, supersedes ext4m)

The user's revision after the ext4m launch: no run continues from a checkpoint (the replay
buffer was not saved, so a resumed run is not faithful), and the 1M sweep being finished, the
two designs are MERGED — the 1M sweep's plain pipeline is the default, keeping only the adopted
throughput optimizations (never the deferred ones).

## The design

- Three configurations x 300 fresh seeds (`a_seed` 1500–1799) = 900 runs, same parameters as
  ext4m (run-5 original RND Adam 1e-4 β=1000; algorithm 2.3 SGD lr 0.01 β=30;
  gt_position_velocity min(1, 1/√n) β=1), pinned by test_ext96h_queue_convention.py.
- Every run is ONE FRESH 96-HOUR ATTEMPT: `total_timesteps=10,000,000` is an upper bound no
  96-hour job reaches (~6.5M on the fastest class); the run ends at the last 50k-step boundary
  before its job's walltime, and its record is COMPLETE there with `ended_by: "walltime"`
  (`"step_cap"` if the bound were ever hit). No model checkpoints, no resume.
- Trainer `train96h.py`: train.py's construction and logging verbatim, plus the adopted
  optimizations (`opt_polyak_foreach`, `opt_torch_reward` via worker argv;
  `set_num_interop_threads(1)` at startup — all bit-exact, research folder APPLIED_CHANGES.md)
  and the WalltimeEndCallback. train.py itself stays untouched.
- Workers are effectively one-shot (claim guard 90 h): one run per worker slot per job; jobs
  request exactly `--time=4-00:00:00`. An early crash lets the slot claim a replacement.
- Orphans (node failure) re-pend and RESTART FRESH — acceptable by design, no checkpoint exists.
- Shares: owner 600 runs (two waves: ~450 now + ~150 ladder), collaborator 300, enforced by the
  ext96h slots ledger. cpu + nolim only, NO puma01, NO reservation.
- Results: Table 60's third block reports FIXED MILESTONES (2M / 4M / 6M) — one row per
  (configuration, milestone) once ≥5 seeds logged that step, ranked within each milestone;
  Figure 18 draws each configuration's mean curve to the step where ≥5 seeds still have data.
- Sentinel `SWEEP96H_COMPLETE` at done == 900.

## What happened to ext4m

Launched 02:38, restructured twice (one-shot 03:20, resumed-from-checkpoint 13:20), and retired
14:0x before any run completed. Its queue, data, and checkpoints were DELETED on the user's
order (fresh start); its scripts are removed from the working tree (git history keeps them:
launch commits a4a0da4/3476207, research commit ba65848); EXT4M_DESIGN.md stays as the record
of the superseded design. The seeds are reused — every ext96h run is a fresh attempt at the
same (configuration, seed) instances.
