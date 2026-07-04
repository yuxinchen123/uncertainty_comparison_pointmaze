## Purpose

Train run 3.1.2 answers two questions at once.

**Science:** why is run-3.1.1's batch elliptical (best 15.65 training reward; 9.03 at the run-2-matched
configuration) so far below run-2's `rnd_elliptical` (44.49 eval reward)? The implementation changed in
exactly three ways between those runs — feature normalization raw→unit, ridge 1e-6→{1e-2, 1}, bonus clip
none→5 — so this run (a) replicates run-2's exact configuration under the current code and (b) runs the
full factorial over those three factors to attribute the gap. Pre-registered hypothesis: the decisive
quantity is the effective ridge ratio ρ = λ·d / mean‖φ‖² (novel-vs-covered bonus contrast = sqrt(1+1/ρ));
unit-norm silently redefined λ so the 3.1.1 grid sat at ρ ≥ 1 (contrast ≤ 1.33×) while run 2 ran at
ρ ~ 1e-5–1e-3 (contrast 30–300×). Competing hypothesis: raw features' corner-seeking norm bias
(zero-bias ReLU encoder ⇒ ‖z(x)‖ ∝ ‖(s,a)‖, largest at the goal corner) is what actually helped run 2.
Each run logs `intrinsic_diagnostics` (mean raw/normalized feature squared norms → measured ρ, covariance
eigenvalue range, clip-hit fraction) so the hypotheses are checked against measured numbers. Literature
note: neither the RND paper (§2.4) nor CleanRL clips the intrinsic reward (they scale-normalize by a
running std of intrinsic returns); the clip is a non-standard knob whose cost this run measures.

**Infrastructure:** which Slurm job shape drains the queue better at equal CPUs — 8 tasks × 2 CPUs
(run-3.1.1 convention, arm A) or 16 tasks × 1 CPU (arm B)? Both arms present an identical job-level ask
(16 CPUs + 32G, `--mem-per-cpu=2G`; measured worker peak RSS ≈ 620 MB) so the task shape + thread budget
(OMP/torch threads 2 vs 1) is the only difference. Evidence arena: co-located pairs on the reservation
(jaguar03 7A+7B, puma01 3A+3B — identical silicon); open partitions (cpu 8A+8B, gpu allowlist 5A+5B) only
bulk-drain the science queue; nolim skipped (80-CPU per-user cap, DenyOnLimit, filled by run-3.1.1).
Collected per arm: per-run duration (paired per config on reservation silicon), per-node throughput,
failure rate (failed/ + rc=124 + OOM + stale running/ markers older than 24 h).

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| algorithm | `rnd_elliptical` (batch covariance, sample-time update), `(s,a)` input — all cells |
| env | `PointMaze_Large-v3`, `top_right` goal, `continuing_task=True`, `reset_target=False`, max 400 steps |
| agent | SAC (`MlpPolicy`, SB3 defaults), γ=0.999, device cpu, 1M steps |
| eval | **`eval_standalone=True`**, 100 episodes every 50000 steps (same protocol as run 2) |
| swept: feature normalization | {none (raw), unit} |
| swept: ridge λ (`elliptical_regularization`) | {1e-6, 1e-4, 1e-2} |
| swept: bonus clip (`elliptical_bonus_clip`) | {inf (no clip), 5} |
| swept: β | {0.001, 0.01, 0.1} |
| seeds | 0–49 (50): arm A = even, arm B = odd (25 each) |
| grid | 12 cells × 3 β = 36 configs × 50 seeds = 1800 runs (900/arm) |
| replica cell | (none, 1e-6, inf, β=0.01) = run-2's exact configuration |
| logging | local JSON, checkpointed atomically every eval cadence (`completed` flag); `intrinsic_diagnostics` |
| worker per-run stopwatch | 24 h (healthy worst case ≈ 19 h with eval on) |
| Slurm | arm A: 8×2cpu; arm B: 16×1cpu; both 16 CPUs + 32G/job, `--time=4-00:00:00` |

## Code and config changes

Relative to the run-3.1.1 state of `07_reconstruction`:
- `train.py`: new `elliptical_bonus_clip` in Config + argparse + the per-run JSON ("inf" parses to
  float('inf')); `torch.set_num_threads` from `OMP_NUM_THREADS`; `_write_local_log` now atomic
  (tmp + `os.replace`) with a `completed` flag and optional `intrinsic_diagnostics`.
- `callbacks/local_log_checkpoint.py` (new): flushes the per-run JSON at each eval cadence with
  `completed=false`, so walltime-killed runs keep their partial curves (488 run-3.1.1 runs left nothing
  under the old write-once convention). Rule updated: `.claude/rules/run-id-and-logging.md`.
- `methods/elliptical_bonus.py`: diagnostics counters (mean raw/normalized feature squared norms over the
  covariance input stream, clip-hit fraction) + `diagnostics()` (adds Λ eigenvalue range, effective ridge
  ratio ρ).
- Run-3.1.1's `progress.py` and analysis loader patched to skip `completed=false` checkpoint records (its
  11 leftover jobs now run this updated trainer).
- This folder's `slurm/`: two-arm `build_queue.py` (parity seed split), `worker.py` (24 h stopwatch),
  `worker_8x2.slurm` / `worker_16x1.slurm`, `launch_queue.sh` (evidence arena on the reservation, bulk on
  cpu/gpu, nolim skipped, interleaved A/B, per-arm id files), `progress.py`, `test_run_id_convention.py`.

## Git state

Commit: `8622c33d5dc10f52898192459494fe7d02ef3777` (branch `Use-RLexplore-RND`); working tree dirty — it
carries the run-3.1.1 changes (elliptical family refactor, float64 Cholesky, eval_standalone) plus the
run-3.1.2 changes listed above, all uncommitted. The run uses the live package at this state; do not edit
the elliptical/train code while the sweep is running. A code snapshot is in this folder's `code/`.

---

# Follow-up sweep (launched 2026-07-03): unit normalization at ridge 1e-8, no clip

Sweep id: `2026-07-03-21-22_unit_ridge-1e-8_clip-inf_beta-1e-4-1e-3-1e-2-1e-1-1e0_50seed_16x1`
(queue under `queue/<sweep_id>/`, data under `data/<sweep_id>/local/`, job ids in
`slurm/submitted_jobids_<sweep_id>.txt`; manifest row in `data/SWEEPS.md`).

## Purpose

Run-3.1.2 finding (iv) (writeup section 5.3.5): unit-normalized features could not reach the run-2
replica's effective ridge ratio inside the 3.1.1/3.1.2 grids — their floor is $\rho = \lambda d$
($1.28\times 10^{-4}$ at $\lambda = 10^{-6}$, $35\times$ the replica's measured $3.7\times 10^{-6}$) —
and matching run-2's bonus contrast under unit norm would need $\lambda \approx 3\times 10^{-8}$, below
anything swept so far. This sweep tests that regime directly with ONE cell below the old grid: unit
normalization at $\lambda = 10^{-8}$ (floor $\rho = \lambda d = 1.28\times 10^{-6}$, BELOW the replica's
measured $3.7\times 10^{-6}$), no clip. The questions:

1. Does unit normalization at a replica-level (or smaller) $\rho$ recover replica-level eval reward
   (about 52.8), i.e. was the ridge ratio the whole story (Hypothesis 1, now tested where 3.1.2 could
   not test it)?
2. Or does it stay below the raw-feature replica even at matched $\rho$, which would revive the
   raw-feature extreme-state-premium hypothesis (Hypothesis 2, dropped on parsimony in 3.1.2)?
3. Where does $\beta^\star$ sit for this large-contrast bonus? Finding (v) says $\beta^\star$ shifts
   inversely with bonus scale, hence the wide five-decade beta sweep including the new endpoints
   $10^{-4}$ and $10^{0}$.

## Key hyperparameters

Identical to the 3.1.2 replicate/ablate sweeps except the cell and the beta range:

| Parameter | Value |
|-----------|-------|
| algorithm | `rnd_elliptical` (batch covariance, sample-time update), `(s,a)` input |
| cell | unit normalization, ridge $\lambda$ = 1e-8, clip = inf (no clip) |
| swept: β | {0.0001, 0.001, 0.01, 0.1, 1.0} (ascending, inner axis) |
| seeds | 0–49 (50), OUTERMOST — seed s owns ids [5s .. 5s+4]; run_total = 250 |
| env / agent / eval | PointMaze_Large-v3 top_right, SAC γ=0.999 cpu, 1M steps, standalone eval ON, 100 episodes / 50k steps |
| logging | local JSON per run (`data/<sweep_id>/local/<id>_of_250.json`), checkpointed at eval cadence, distance logging off |
| Slurm | 16 jobs × (16 tasks × 1 cpu, `--ntasks-per-core=2`, 2G/cpu, `--time=4-00:00:00`) = 256 workers ≥ 250 runs (one wave) |
| placement | OPEN partitions only (cpu 8, gpu allowlist lynx01–05 5, gnolim 3); reservation jaguar03/puma01 deliberately left free; nolim skipped (its per-user memory pool was full) |

## Code and config changes

Relative to the 3.1.2 sweeps in this folder (trainer/`train.py` untouched — same code state):

- `slurm/build_queue_unit_ridge_1e8.py` (new): builds this sweep's 250-config queue (single cell,
  beta inner ascending, seed outermost) and appends the `data/SWEEPS.md` row.
- `slurm/worker.py` `claim()`: now sorts pending names by id and picks randomly among only the FIRST 32
  (ordered-window claim) so early seeds finish first — the rule made explicit for post-3.1.2 sweeps in
  `.claude/rules/run-id-and-logging.md`; the old fully-shuffled claim executed runs in random order.
- `slurm/progress_unit_ridge_1e8.py` (new): single-sweep progress (per-beta completed-seed coverage).
- `slurm/launch_queue_unit_ridge_1e8.sh` (new): the 16-job open-partition submission described above.
- `slurm/test_unit_ridge_1e8_convention.py` (new): pins the 250-config grid and the ordered-window claim
  (all tests pass together with the existing `test_run_id_convention.py`).
- `slurm/worker_16x1.slurm` reused unchanged.

## Git state

Commit: `a85b1821de31d19b4a518c7a44e6bb8a4df59793` (branch `Use-RLexplore-RND`); working tree dirty —
uncommitted at launch: the new/edited `slurm/` files above, this file, `analysis/analysis.md` +
`analysis/code/` + `analysis/plots/` (run-3.1.2 analysis), `development_document/main.tex`, and the
project rule `.claude/rules/run-id-and-logging.md`. Trainer code (`train.py`, `methods/`, `callbacks/`)
is byte-identical to the state the 3.1.2 sweeps ran (snapshot in this folder's `code/`).
