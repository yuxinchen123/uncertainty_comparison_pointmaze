# Train run 3.1.1 — sweep status, cancellation, and resume

## Active sweep
- **SWEEP_ID:** `2026-06-26-05-45_4algo-betaridge-input`
- Queue: `queue/2026-06-26-05-45_4algo-betaridge-input/{pending,running,done,failed}/`
- Data: `data/2026-06-26-05-45_4algo-betaridge-input/local/<NNNN_of_9100>.json`
- Job ids (this run's only): `slurm/submitted_jobids_2026-06-26-05-45_4algo-betaridge-input.txt`
- Submitted 2026-06-26 ~05:45, 64 Slurm jobs (512 workers), `--time=3-23:00:00` (finishes before the
  2026-06-30T06:00 cluster maintenance).

## First launch was cancelled (float32 Cholesky bug) — superseded
- The first launch (`SWEEP_ID=2026-06-26-05-25_4algo-betaridge-input`) crashed every
  `rnd_elliptical_global` run: the global covariance accumulates `Λ=λI+Σφφ^T` and a **float32** Cholesky
  fails on it (`not positive-definite`) after ~3000 updates (worst at ridge `λ=1e-2`). The batch elliptical
  and RND were unaffected.
- Fix: `elliptical_bonus.py` now keeps the covariance + Cholesky/solve in **float64** (features stay
  float32, cast at the boundary; bonus cast back to float32). Verified: a 10k-step global smoke at `λ=1e-2`
  runs clean; the relaunched sweep has 0 failures past the crash point.
- The cancelled sweep's queue/data/jobids were removed; nothing had finished (no data lost).

## How to check progress
```
conda run -n exploration python slurm/progress.py 2026-06-26-05-45_4algo-betaridge-input
```
Prints finished distinct seeds per config (91 configs: 28 `rnd_elliptical`, 56 `rnd_elliptical_global`,
7 `rnd_next_state`) and `THRESHOLD_MET` once **every** config has `>=30` finished seeds. A live monitor
appends each 20-min snapshot to `logs/progress_2026-06-26-05-45_4algo-betaridge-input.log`.

## Trigger for the writeup update
When `progress.py` prints `THRESHOLD_MET` (all 91 configs `>=30` finished seeds), run the analysis
(`analysis/code/make_all.py`) and fill the run-3.1.1 results (the `\beta^\star` table + training-reward
curve) in `development_document/main.tex` (placeholder `\subsubsection{Train run 3.1.1 ...}`,
`\label{sec:train-run-3-1-1}`). The full 100-seed grid keeps running afterward to extend coverage.

## Resume / relaunch (e.g. after maintenance or if jobs end early)
The queue is durable: workers only claim from `pending/`, so re-submitting jobs continues where it left off.
```
# re-record orphaned running/ back to pending/ first if a launch was killed mid-run:
#   find queue/<SID>/running -name '*.json' -exec mv -t queue/<SID>/pending {} +
# then resubmit the SAME sweep's workers (reuses queue + data dir):
cd slurm && SWEEP_ID=2026-06-26-05-45_4algo-betaridge-input  # reuse; do NOT build a new queue
# (launch_queue.sh generates a NEW id + queue; to reuse, submit workers with --export=ALL,SWEEP_ID=<this id>
#  against the existing queue, mirroring launch_queue.sh's sbatch block.)
```

## Cancellation safety (HARD RULE)
Only ever `scancel` the ids in `slurm/submitted_jobids_2026-06-26-05-45_4algo-betaridge-input.txt`.
Never `scancel -u sl5nw` / `-t` / `-n` (shared uid with other sessions).
