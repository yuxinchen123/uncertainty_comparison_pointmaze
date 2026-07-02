# Train run 3.1.2 — analysis

Status: **launching**. Results are filled once every one of the 36 configs has `>=20` pooled finished
seeds (`slurm/progress.py <sweep_A> <sweep_B>` prints `THRESHOLD_MET`). Design, hypotheses, and the
pre-registered predictions: `../experiment_background.md` and writeup §5.3.5
(`\label{sec:train-run-3-1-2}`).

## What this analysis will produce

1. **12-cell table** (each cell at its best β): normalization, λ, clip, β⋆, final **eval** extrinsic
   reward $\bar{R} = \frac{1}{n}\sum_i R_i$ with $\mathrm{SE} = s/\sqrt{n}$, a **success rate** over
   seeds (fraction with final eval reward above a threshold chosen from the pooled bimodal gap; Wilson
   interval) with median/IQR — the per-seed outcome is bimodal (run-3.1.1 medians were 0), so mean ± SE
   alone is misleading — plus the training-reward secondary column. NaN-final runs counted per cell,
   never dropped silently.
2. **Reward-over-training figure**: the replica cell (raw, 1e-6, no clip), its single-factor swaps, the
   (unit, 1e-2, 5) run-3.1.1-like cell, and the run-2 `rnd_elliptical` reference — all EVAL reward (no
   metric caveat).
3. **Measured-ρ diagnostics table** per cell from `intrinsic_diagnostics`: mean ‖z‖², mean ‖φ‖²,
   effective ridge ratio $\rho = \lambda d / \overline{\lVert\varphi\rVert^2}$ (measured vs. assumed),
   Λ eigenvalue range, clip-hit fraction — the direct check of the ridge-ratio hypothesis vs. the
   corner-bias hypothesis.
4. **Slurm-shape comparison (arm A = 8 tasks × 2 CPUs vs arm B = 16 tasks × 1 CPU)**: per-run duration
   distributions paired per config on reservation silicon only (jaguar03 / puma01 co-located pairs),
   per-node throughput (runs/node/day), and infrastructure error rate = `failed/` + rc=124 timeouts +
   out-of-memory kills + stale `running/` markers older than 24 h (job-walltime losses classified
   separately from unexpected crashes). Worker logs (claim lines carry jobid + config + duration + rc)
   joined with sacct give per-run node attribution.

## Conventions

- Only records with `completed=true` count as finished (checkpoint records with `completed=false` are
  partial: counted separately, usable for truncated curves — new logging convention, see
  `run-id-and-logging.md`).
- Science pools BOTH arms' sweeps (identical 36-config grids, seeds split even/odd) to n=50 per config;
  the arm comparison never mixes arenas (reservation = evidence, open partitions = bulk drain).
- `min seeds = 20` gates table eligibility and curve truncation at this interim analysis; the grid keeps
  filling to 50.
