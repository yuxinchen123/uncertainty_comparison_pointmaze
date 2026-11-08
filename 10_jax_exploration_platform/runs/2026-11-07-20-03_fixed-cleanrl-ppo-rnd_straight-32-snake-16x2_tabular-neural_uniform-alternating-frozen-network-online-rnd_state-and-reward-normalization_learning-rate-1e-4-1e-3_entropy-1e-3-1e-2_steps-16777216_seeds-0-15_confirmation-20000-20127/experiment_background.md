# Fixed CleanRL PPO-RND settings on the straight and snake grids

Times are Pacific. Launched 2026-11-07 (run folder created 20:03 PT). Plan:
`~/.claude/plans/2026-11-07-19-27_gitignore-cleanup-and-fixed-cleanrl-ppo-rnd-grid-study.md`.

## 1. Purpose

Test, with PPO fixed at CleanRL PPO-RND's scalar settings, how the starting bonus field and the
way the bonus decays affect exploration on two deterministic grids. The methods are: no bonus,
uniform field, alternating one-high-one-low field, frozen random-network field (all with the
count rule, bonus = field ÷ √visits), and online RND (the same networks, with the predictor
trained during PPO, so the bonus decays as it learns). Each method runs with CleanRL's state
normalization (RND inputs) and reward normalization off and on, where they apply. Configurations are
chosen by the mean capped first-goal time (project rule `.claude/rules/exploration-metrics.md`);
discovery rate, coverage and the final-policy goal rate are reported beside it.

## 2. Key hyperparameters

| setting | value |
|---|---|
| maps | straight grid, 32 cells; snake grid, width 16, 2 turns (50 cells) |
| policies | tabular (4 action logits and 2 values per cell); neural (shared 2→64→64 ReLU, actor 64→64→4, residual critic 64→64 with a 64→2 output) |
| PPO | rollout 128, 4 epochs, 4 minibatches, extrinsic discount 0.999, intrinsic discount 0.99, GAE 0.95, clip 0.1, value coefficient 0.5, maximum gradient norm 0.5, Adam epsilon 1e-5, extrinsic advantage weight 2, linear learning-rate annealing, one environment per policy |
| swept | learning rate {1e-4, 1e-3} × entropy {0.001, 0.01} × intrinsic weight 2^-6 … 2^6 (13 values); 52 configurations per condition, 4 for no bonus |
| conditions per map and policy | 13: no bonus; uniform ×2 reward normalization; alternating (odd cells 2^-6 of even cells) ×2; frozen network field ×2 state ×2 reward normalization; online RND ×2 ×2 |
| RND networks | target 2→32→64→64→512, predictor 2→32→64→64→512→512→512 (leaky ReLU 0.01 on the first three layers, ReLU after, linear output); orthogonal gain √2, biases zero; weights seeded by field seed, identical to the frozen field's |
| online RND training | mean squared feature error, each sample kept with probability 0.25, one Adam over policy and predictor with a joint gradient clip of 0.5; bonus = raw half summed squared error, not scaled |
| state normalization | running mean and variance of the RND input coordinates, started from 50 random-action rollouts of 128 steps (seeded by field seed, not counted as training), clipped to [-5, 5] |
| reward normalization | CleanRL's discounted-sum filter and running variance (one environment per policy) |
| budget | 16,777,216 transitions per policy, development and confirmation |
| seeds | development 0-15 (field seeds 0-3, four policies per field); confirmation 20000-20127 (field seeds 5000-5031) |
| confirmation | each condition's chosen configuration, and the best intrinsic weight at learning rate 1e-4 and entropy 0.001 |
| evaluation | 32 sampled episodes and 1 greedy episode per policy from the start cell, up to the training episode limit |

## 3. Code and config changes

Relative to the earlier initial-bonus study (commit 0d14eb3):

- `grid_initialization/config.py`: defaults learning rate 2^-8 → 1e-4 and entropy 2^-10 → 1e-3; new
  switches `intrinsic_source` (count / online_rnd), `normalize_rnd_inputs`,
  `predictor_update_proportion`.
- `grid_initialization/online_rnd.py` (new): processor-generated network weights, warm-up
  statistics, frozen field in float64, per-copy JAX forward pass at full float32 precision.
- `grid_initialization/training.py`: online bonus (all-cell errors once per rollout), input
  statistics update after GAE, masked predictor loss added to the PPO loss, final-policy evaluation.
  Count runs are unchanged: the check `check_count_path_unchanged.py` compares final states with
  the earlier code on the processor backend.
- `scripts/grid_initialization_campaign/`: `fixed_settings_campaign.py`, `fixed_settings_benchmark.py`,
  `fixed_settings_report.py`, `run_fixed_checks.py`, `check_fixed_unit.py`; `train_unit.py` builds
  this study's fields and networks and records count and RND-error histories and evaluations; the
  shared queue, submission and monitor scripts select this study's controller and ledgers from
  `source.json`.

## 4. Git state

Filled in at launch below.
