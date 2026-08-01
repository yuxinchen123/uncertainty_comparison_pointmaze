# Experiment background — train run 8.1.2 (section 8, train run 1.2)

## Purpose

Two independent tasks under one run folder, one queue system, one controller:

1. **Task R** — the RND baseline: the run-8.1 (train run 1.1) winning SAC + RND configuration per
   environment (AntMaze UMaze bonus weight 1e4, AntMaze Medium 3e3; run-5 "original-small" stack,
   reward normalization ON), run to 10,000,000 env steps, 100 seeds per environment, never pruned.
   It is the comparison curve, and its completed 10M records later define the 10M bar of a deferred
   second stage.
2. **Task S** — stage 1 of the new-algorithm sweep: every configuration of algorithms 1, 2.1, 2.2,
   2.3 (defined below) run to 1,000,000 env steps, racing to 100 seeds per configuration with
   frozen-bar pruning from 30 seeds: a configuration is pruned at the first controller tick where
   its 99% upper confidence bound (mean + 2.576·s/√n over completed seeds' whole-run mean training
   return) falls below the frozen bar — the run-8.1 RND winner's mean at 1M for that environment
   (recomputed from the run-8.1 records at launch and committed in `slurm/FROZEN_BARS.json`).
   Reaching 100 completed seeds unpruned = stage-1 survivor (final verdict in the decision log).

The research question: can any of the four new arms beat the best known RND configuration on the
two AntMaze environments where RND is the only method that ever reaches the goal?

The four algorithm arms (all share ONE stack — the Table-66 / run-5 original-small RND architecture
and state normalization — with reward normalization OFF):

1. **Algorithm 1**: `rnd_next_state`, plain constant-rate SGD predictor optimizer (swept), `l2`
   bonus readout, bias initialization `normal_0.5`, original training loss.
2. **Algorithm 2.1**: algorithm 1 + a frozen copy of the predictor taken at initialization; the
   bonus is the ratio ||e_theta||_2 / (||e_0||_2 + 1e-8), so every state's readout starts at 1.
3. **Algorithm 2.2**: algorithm 2.1 with the training loss also normalized by the frozen-
   initialization error: (1/b) Σ_i ||e_theta(s_i)||² / (||e_0(s_i)||² + 1e-8).
4. **Algorithm 2.3**: algorithm 2.1 with LayerNorm after each hidden Linear in both nets (the
   predictor's extra block carries its own LayerNorm; no output normalization); original loss.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| environments | `AntMaze_UMaze-v5_start_bottom_left` (start (3,1) → goal (1,1), episode limit 700), `AntMaze_Medium-v5_start_bottom_left` ((6,1) → (1,6), limit 1000) |
| reward convention | −1 per step, 0 on the goal step (reward shift −1); termination at 0.45 m; goal fixed; position noise 0 |
| base RL algorithm | SB3 SAC `MlpPolicy`, Adam 3e-4, batch 256, tau 0.005, buffer 1e6, nets 2×256 ReLU, auto entropy, train freq 1/1, learning starts 100, discount 0.99 |
| task R total steps / seeds / beta | 10,000,000 / 100 per env / fixed 1e4 (UMaze), 3e3 (Medium) |
| task R stack | run-8.1 `ORIGSMALL_PARAMS`: adam 1e-4, mse_mean readout, leaky_relu(0.2), predictor +1 block, env-steps warm-up 6400, reward norm ON (gamma 0.99), bias zero, orthogonal weights |
| task S total steps / seeds | 1,000,000 / race to 100 per config, prune checks from n ≥ 30 |
| task S swept knobs | SGD learning rate ∈ {1e-3, 1e-2}; bonus weight ∈ {1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 1e2, 3e2, 1e3, 3e3, 1e4} |
| task S fixed RND knobs | optimizer sgd (constant, momentum 0), l2 readout, bias normal_0.5, reward norm OFF, leaky_relu(0.2), predictor +1 block, env-steps warm-up 6400, obs norm RMS clip ±5, output dim 128, orthogonal weights |
| algorithm-2 knobs | `rnd_readout_norm_init True` (2.1/2.2/2.3), `rnd_predictor_loss mse_init_normalized` (2.2 only), `rnd_layer_norm True` (2.3 only), eps/delta 1e-8 |
| config count | 2 (baseline) + 4 arms × 2 env × 2 lr × 15 beta = 242 |
| frozen bars (1M, whole-run mean) | UMaze −689.7404 (n=84), Medium −998.4678 (n=82) — recomputed at launch from ALL completed run-8.1 RND records (`slurm/FROZEN_BARS.json`, committed; the writeup's interim tables show −689.85/−998.48 at the n=77 snapshot) |
| prune rule | at n ≥ 30 completed 1M records: prune iff mean + 2.576·s/√n < bar; re-checked every tick as n grows to 100 |
| eval/logging cadence | eval_freq 50000, n_eval_episodes 100, eval_standalone False, log_distance False, wandb off |
| devices / partitions | task R: gpu partition (cuda) + gnolim (cpu; its Pascal GPUs cannot run torch cu128); task S: cpu + nolim + reserved jaguar03, cpu only |
| gnolim claim order | task-R 10M items first, task-S 1M items as fallback (walltime guard decides claimability) |
| checkpoints | none — every run completes inside one Slurm job (10M cuda claims need ≥ 90 h of remaining job walltime; gnolim cpu claims ≥ 170 h) |
| env / python | `/p/rlprojects/RND/.venvs/exploration/bin/python` (Python 3.11, torch 2.10.0+cu128, SB3 2.7.1, gymnasium 1.2.3, gymnasium-robotics 1.3.1, mujoco 3.1.6) |

## Code and config changes

- `src/rnd_exploration/methods/rnd.py`: new switches `readout_norm_init` (+ frozen
  `init_predictor` deep copy, ratio bonus), `predictor_loss='mse_init_normalized'` (algorithm 2.2's
  loss), `layer_norm` (algorithm 2.3; keyed init streams skip LayerNorm positions so 2.3 draws the
  same Linear weights/biases as 2.1 at the same seed), and optimizer choice `sgd` (constant rate
  from `rnd_lr`). Unit tests in `tests/methods/rnd/test_run812_switches.py`.
- `train.py`: flags `--rnd_readout_norm_init`, `--rnd_readout_norm_eps`, `--rnd_predictor_loss`,
  `--rnd_layer_norm`; `--rnd_optimizer` gains `sgd`; all recorded in the per-run JSON.
- `src/rnd_exploration/methods/__init__.py`: duck-typed pass-through of the four new knobs.
- Run machinery in this folder's `slurm/` (adapted from run 8.1): two pending pools
  (`pending_10m`, `pending_1m`) with `WORKER_POOLS` claim order and a per-item walltime guard in
  `worker.py`; `stage1_controller.py` (frozen-bar pruning + survivor verdicts, replaces the
  cell-relative `prune_controller.py`); `stage1_check.py` invariant checker;
  `compute_frozen_bar.py` + committed `FROZEN_BARS.json`; `simulate_truncation.py` (the launch-gate
  simulation of the truncation logic).
- The deferred second stage (survivors to 10M, both-must-fail trigger vs a task-R-defined 10M bar,
  checkpoint/resume) is documented in the plan and NOT part of this run.

## Git state

Launch commit: recorded below at submit time (commit-before-submit).

- Code state at folder creation: branch `Use-RLexplore-RND`, parent commit
  `0d6b3f56bde60b1ae54fa984bcd8179cb94ad806` plus the working-tree changes listed above
  (rnd.py / train.py / methods/__init__.py / this run folder), committed before submission.
- Launch commit hash: (filled at submit time)
