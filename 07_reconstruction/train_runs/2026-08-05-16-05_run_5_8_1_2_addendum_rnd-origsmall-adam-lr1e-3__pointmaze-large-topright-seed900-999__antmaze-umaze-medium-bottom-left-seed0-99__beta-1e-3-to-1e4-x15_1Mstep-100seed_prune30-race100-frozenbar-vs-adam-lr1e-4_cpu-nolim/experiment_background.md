# Experiment background — original RND at Adam learning rate 1e-3 (addendum to train run 5 and train run 1.2)

## Purpose

Train run 5 (writeup subsection "Train run 5") and train run 1.2 (writeup subsubsection "Train run
1.2") both run the *original* RND recipe — the "original-small" stack: Adam, the
mean-squared-error-over-dimensions readout, LeakyReLU 0.2, a predictor one block deeper than the
target, a 6400 env-step state-normalization warm-up, reward normalization on — at the original
paper's predictor learning rate of **1e-4**. This run asks one question:

> On the same three environments, does the same stack do better or worse at a predictor learning
> rate of **1e-3**?

Nothing else changes. The queue's configuration parameters are train run 5's `ORIGSMALL_PARAMS`
copied field for field with `rnd_lr` set to `0.001`, and a unit test
(`slurm/test_run_queue_convention.py`) asserts that `rnd_lr` is the only field that differs. So each
environment's result is a single new column beside the Adam 1e-4 column already in the writeup, not a
new arm.

The three environments are the three the two sections measure, each kept exactly as its own section
defines it:

| environment (`--env_setup`) | from | discount | episode cap | reward | reset noise |
|---|---|---|---|---|---|
| `initial_single_large_pointmaze_max_400` | train run 5 | 0.999 | 400 | raw sparse, no shift | ±0.25 cell |
| `AntMaze_UMaze-v5_start_bottom_left` | train run 1.2 | 0.99 | 700 (registered default) | −1 per step, 0 at the goal | 0 (exact cell centers) |
| `AntMaze_Medium-v5_start_bottom_left` | train run 1.2 | 0.99 | 1000 (registered default) | −1 per step, 0 at the goal | 0 (exact cell centers) |

Every environment knob comes from the named `EnvSetup`, so no run can silently disagree with its
environment's definition. `initial_single_large_pointmaze_max_400` is the registry's verbatim record
of the setup every run up to train run 5 used.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| intrinsic bonus | `rnd_next_state` — the train-run-5 original-small stack |
| **the one changed knob** | predictor optimizer Adam at learning rate **1e-3** (train run 5 and train run 1.2 used 1e-4) |
| unchanged RND knobs | `mse_mean` readout, update proportion 1.0, LeakyReLU slope 0.2, predictor +1 block of width 128, env-steps warm-up 6400, reward normalization on with discount 0.99, zero bias, orthogonal √2 weights, output dimension 128, state normalization by running RMS with clip ±5, next-state input |
| bonus weight | swept over the 15 values train run 1.2 uses: 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 1e2, 3e2, 1e3, 3e3, 1e4 |
| base RL algorithm | SB3 SAC `MlpPolicy`, Adam 3e-4, batch 256, tau 0.005, buffer 1e6, nets 2×256 ReLU, auto entropy target −dim(A), train freq 1/1, learning starts 100 |
| total steps per run | 1,000,000 (all three environments) |
| configurations | 3 environments × 15 bonus weights = **45** |
| seeds | 100 per configuration. PointMaze uses `a_seed` 900–999 (fresh, disjoint from train run 5's 600–899 and every earlier PointMaze run); AntMaze uses 0–99 (the seed space train runs 1.1 and 1.2 used there) |
| total runs | 45 × 100 = **4,500** |
| truncation floor / target | decisions from 30 completed seeds; 100 completed seeds without a truncation = survivor |
| truncation rule | truncate configuration *c* at the first controller tick where mean + 2.576·s/√n < BAR(environment) |
| eval / logging cadence | `eval_freq` 50000, `n_eval_episodes` 100, `eval_standalone` False, `log_distance` False, wandb off |
| device / partitions | cpu only, on the **cpu and nolim** partitions. No gpu, no gnolim, no reservation (user instruction for this run) |
| checkpoints | none — every run completes inside one Slurm job; a run killed mid-flight is re-queued and re-run from the start |
| env / python | `/p/rlprojects/RND/.venvs/exploration/bin/python` with `PYTHONNOUSERSITE=1` (see "Code and config changes") |

## The truncation bars (frozen before launch)

BAR(environment) is the mean of the **best Adam 1e-4 configuration on that environment**, computed
from the executed run that measured it, under **that environment's own score rule** — the user's
instruction that each environment be judged against its own section's number. The values live in
`slurm/FROZEN_BARS.json`, committed and never regenerated; every decision line records the file's
sha256 and `20_mins_monitoring/truncation_check.py` verifies it each tick.

| environment | score rule | source | best Adam 1e-4 bonus weight | BAR | n |
|---|---|---|---|---|---|
| PointMaze Large (top-right) | final reward — the last `train_history` row's `train/mean_extrinsic_reward`, i.e. the mean extrinsic return over the last 100 training episodes at the 1M-step evaluation (train run 5's own metric) | train run 5, sweep `2026-07-20-16-55_set-baseline` | 1000 | **38.6412** | 300 |
| AntMaze UMaze (bottom-left) | whole-run mean — the mean of `train/extrinsic_reward` over every `train_episode_history` row (train run 1.1/1.2's metric) | train run 1.1, sweep `2026-07-23-02-05_pm-am-run1` | 10000 | **−689.7404** | 84 |
| AntMaze Medium (bottom-left) | whole-run mean (as above) | train run 1.1, sweep `2026-07-23-02-05_pm-am-run1` | 3000 | **−998.4678** | 82 |

The two AntMaze bars are the same numbers train run 1.2 froze, recomputed here from the same records
and asserted to reproduce the same winning bonus weight. The PointMaze winner is bonus weight 1000
under both candidate score rules (final reward 38.6412; whole-run mean 17.1492), so the choice of
rule does not change which configuration defines the bar — only the scale it is measured on.

## Code and config changes

- **No trainer change.** Every knob this run uses already exists; `rnd_lr` is an existing train.py
  flag. The only new code is the run machinery in this folder's `slurm/` and `20_mins_monitoring/`.
- `slurm/score_rules.py` — the ONE definition of the per-environment score rule, imported by the bar
  computation, the controller, the checker and the report, so the bar and the runs raced against it
  can never be scored differently.
- `slurm/build_queue.py` — 45 configurations × 100 seeds, seed index outermost, per-environment seed
  offset (PointMaze +900, AntMaze +0).
- `slurm/compute_frozen_bars.py` → `slurm/FROZEN_BARS.json` — run once, committed, never regenerated.
- `slurm/truncation_controller.py` — the frozen-bar truncation; `20_mins_monitoring/truncation_check.py`
  re-derives every decision each tick (the sweep_prune skill requires verification whenever a run
  changes the default rule, and this run changes two things: a frozen external bar instead of the
  cell's own best mean, and a different score rule per environment).
- `slurm/plan_jobs.py` — sizes every worker job from the live cluster state instead of a fixed shape:
  `--ntasks` ≤ the node's physical cores, ≤ its free allocatable threads (`CPUEfctv − CPUAlloc`),
  memory per worker ≤ 92% of `RealMemory / CPUEfctv` capped at 2 GB, and the total under the
  per-user pool cap minus 16 threads of headroom. `--ntasks-per-core=2` only on nodes with two
  threads per core; reserved nodes excluded.
- `PYTHONNOUSERSITE=1` in every job script. This is load-bearing, not hygiene: the owner's
  `~/.local/lib/python3.11/site-packages` sits ahead of the shared env on `sys.path` and supplies
  torch 2.10.0+cu128, while a collaborator's uid has no such directory and gets the env's own torch
  2.5.1+cu124. Without it, the owner's workers and a collaborator's workers in the SAME sweep run
  different torch builds. **Known difference from the comparison runs:** train run 5, train run 1.1
  and train run 1.2 set no such variable, so their records are a mix of the two torch builds. This
  run is internally consistent (every worker on the env's 2.5.1+cu124) at the cost of not matching
  either comparison run exactly; the alternative — reproducing the mix — is not reproducible at all.

## Launch gates (all must pass before submission)

1. `slurm/test_run_queue_convention.py` — configuration list, id ordering, seed mapping, keys, and
   the assertion that `rnd_lr` is the only knob that differs from train run 5.
2. `slurm/test_truncation_controller.py` — the controller and the two score rules.
3. `20_mins_monitoring/test_truncation_check.py` — the invariant checker, on a correct decision log
   AND on deliberately wrong ones.
4. `slurm/simulate_truncation.py --full` — the whole machinery at sweep scale on synthetic data.
5. The PointMaze environment-equivalence check — a short run under `--env_setup=initial_single_large_pointmaze_max_400`
   must produce the same trajectory as one under train run 5's explicit environment flags at the
   same seed, proving the named setup reproduces train run 5's environment exactly.

## Git state

- Launch commit (commit-before-submit): recorded here at launch.
- Parent state at folder creation: `88b1e6d`.
