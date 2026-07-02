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
