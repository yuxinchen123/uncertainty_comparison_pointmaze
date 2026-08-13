# Experiment background — train run 6: the four train-run-8.1.2 algorithms on the train-run-5 PointMaze task

## Purpose

Train run 8.1.2 built four algorithm arms (1, 2.1, 2.2, 2.3) and raced them on two AntMaze
environments, where nothing converges (see the writeup's "Why the AntMaze runs do not converge").
Train run 6 asks the same comparison question on the task where the original RND demonstrably
works: train run 5's PointMaze Large top-right task. One question:

> Do any of the four algorithm variations beat the run-5 original RND (Adam 10⁻⁴, best bonus
> weight 10³, final reward 38.6412 over 300 seeds) on its own task, under its own score rule?

The original RND is NOT re-run — its 300-seed final data from train run 5 is the frozen reference.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| environment | `initial_single_large_pointmaze_max_400` — train run 5's task verbatim: PointMaze Large, top-right goal, episode cap 400, discount 0.999, raw sparse reward |
| base RL algorithm | SB3 SAC `MlpPolicy`, defaults (Adam 3e-4, batch 256, tau 0.005, buffer 1e6, auto entropy) |
| the four arms | train run 8.1.2's, parameter dictionaries copied VERBATIM and pinned by unit test: alg1 = plain constant-rate SGD predictor, l2 readout, bias normal 0.5, reward norm OFF; alg2.1 = alg1 + frozen-init predictor copy, ratio bonus; alg2.2 = 2.1 + init-normalized loss; alg2.3 = 2.1 + LayerNorm |
| predictor learning rate | swept {10⁻³, 10⁻²} (plain SGD, constant) |
| bonus weight | swept over the 15 values 10⁻³, 3×10⁻³, 10⁻², 3×10⁻², 10⁻¹, 3×10⁻¹, 1, 3, 10, 30, 10², 3×10², 10³, 3×10³, 10⁴ |
| configurations | 4 arms × 2 learning rates × 15 bonus weights = **120** |
| steps per run | 1,000,000 |
| seeds | `a_seed` 1200–1499 (300 fresh; disjoint from run 5's 600–899 and the addendum's 900–1199) |
| score rule | train run 5's: final reward = last `train_history` row's `train/mean_extrinsic_reward` |
| frozen bar | 38.6412 (run-5 original RND, Adam 10⁻⁴, β=10³, n=300) — `slurm/FROZEN_BARS.json`, sha256 recorded on every decision |
| truncation phase 1 | from 20 completed seeds: truncate when mean + 2.576·s/√n < 38.6412; survivors race to 100 |
| truncation phase 2 | at 100 seeds, once every sibling of an arm is resolved: the arm's best-mean configuration CONTINUES to 300 seeds; the other candidates stop (verdict `stopped_at_100`) |
| logging | one JSON per run under `data/<sweep_id>/local/`, checkpointed at each 50k-step eval with a `completed` flag; wandb off |

## Code and config changes

- New run folder; worker/queue/monitor machinery copied from the proven learning-rate addendum
  run (walltime guard, plan_jobs with the pool-sizing and socket-pinning fixes, requeue_orphans).
- New two-phase `slurm/truncation_controller.py` + rewritten
  `20_mins_monitoring/truncation_check.py` (7 invariants incl. per-arm winner uniqueness and
  winner-mean dominance) + per-arm `monitoring_report.py`.
- Launch gates: 24 unit tests (queue convention, arm byte-equality against train run 8.1.2's
  build_queue.py, two-phase controller, walltime guard) and `slurm/simulate_truncation.py --full`
  at full scale — 120 synthetic configurations: 108 truncated at the 20-seed floor, 8 stopped at
  100, 4 winners complete at 300, 0 invariant violations at every wave.
- Submission: cpu → nolim → reservation (the user's order for this run; no gpu/gnolim), per the
  shared submit-cpu-sweep skill. Collaborator packet in `for_collaborator/`.

## Git state

Commit at folder creation: `2cde90d7813969af1241a61117b6880315cf0924`; the launch commit (with
this folder's code) is recorded below when the queue is built.

Launch commit: `5df4463` (queue built 2026-08-08 02:46, sweep_id 2026-08-08-02-46_run6)

## 4M extension sweep (ext4m, added 2026-08-13)

A second sweep in this same folder (nothing about the 1M sweep changed): three configurations x
300 fresh seeds (`a_seed` 1500–1799) x **4,000,000 steps** = 900 runs, no truncation race, every
run checkpointed every 0.5M steps and resumable across job walltimes. Full design (checkpoint
contents, the replay-buffer-tail storage decision, suspend/resume protocol, submission buckets):
`slurm/EXT4M_DESIGN.md`. The three configurations:

1. run-5 original RND (Adam 10⁻⁴, mse-mean readout, reward norm ON, β=1000) — re-run at 4M;
2. algorithm 2.3 at its best run-6 configuration (plain SGD lr 0.01, β=30);
3. the best ground-truth bonus: `gt_position_velocity`, bonus min(1, 1/√n), β=1 (the 66.24 ±
   1.30 oracle of the writeup's run-5 oracle table).

Trainer `train4m.py` (new file beside train.py), worker/plan/monitor under `slurm/ext4m_*`.
Results join Table 60 as extra rows and Figure 18 as extra lines (no separate section), generated
by the same `analysis/code/make_run6_results.py`. Submission cpu → nolim → reservation puma01
only (no jaguar03), puma01 at 2 CPUs per worker. Collaborator packet:
`for_collaborator/README_ext4m.md`.

ext4m sweep id: `2026-08-13-02-36_run6ext4m` (queue built 2026-08-13 02:36, 900 markers); the
launch commit is the commit carrying this line. Launch gates passed before submission: 6 queue-
convention tests (parameters verbatim against the run-5 and run-3.2.2 markers on disk and
run-6's build_queue.py) and 4 checkpoint tests including the full suspend → resume →
auto-delete → complete lifecycle on real trainings (tests/test_train4m_checkpoint.py).

## 96-hour extension sweep (ext96h, 2026-08-13 — REPLACES the ext4m section above)

The ext4m checkpoint/resume sweep was retired before any run completed (the buffer-tail resume
was judged not faithful enough); its queue, data, and checkpoints were deleted. The replacement,
sweep `2026-08-13-14-00_run6ext96h`, merges the finished 1M sweep's plain pipeline with the
throughput research's ADOPTED optimizations only: same three configurations and seeds
(1500–1799), each run ONE FRESH 96-HOUR ATTEMPT under a 10,000,000-step cap no job reaches; the
run ends complete at its job's walltime (`ended_by: "walltime"`). Trainer `train96h.py`
(train.py untouched); no checkpoints, no resume; jobs at exactly `--time=4-00:00:00`; owner 600
runs in two waves + collaborator 300 via slots ledgers; cpu + nolim, no puma01, no reservation.
Table 60 reports fixed milestones (2M / 4M / 6M, ranked within each); Figure 18 draws mean
curves to the last step with >= 5 seeds. Design: `slurm/EXT96H_DESIGN.md`. Launch gates: 7
queue-convention tests + 2 trainer tests (walltime end marks the record complete at a logged
boundary; step-cap path) — all passed.
