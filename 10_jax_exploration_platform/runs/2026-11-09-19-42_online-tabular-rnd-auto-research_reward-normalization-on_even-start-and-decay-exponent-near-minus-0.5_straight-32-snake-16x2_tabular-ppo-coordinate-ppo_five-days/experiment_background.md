# Online tabular-RND auto research: an even start and a decay exponent near −0.5

Times are Pacific. Run folder created 2026-11-09 19:42 PT; deadline 2026-11-14 20:00 PT. Plan:
`~/.claude/plans/2026-11-09-19-03_tabular-rnd-extension-and-online-tabular-rnd-auto-research.md`. The fixed protocol
and success rule are in `program.md`; the log and idea list are in `STATE.md`.

## 1. Purpose

Develop a new variant of online tabular-RND (RND whose networks read a one-hot cell vector) that beats the CleanRL
form with reward normalization on, in four cases: the straight and snake grids, each with tabular-PPO and
coordinate-PPO. The gain must come from two properties together: every cell starts with about the same bonus, and each
cell's bonus decays like $N^{-1/2}$ in its own visits $N$. The bonus may not use visit counts. The method is meant for
a paper, so it may not be a re-implementation of a published RND variant.

The baseline is the condition "online tabular-RND, reward normalization on" of the extension run
`runs/2026-11-09-19-40_fixed-cleanrl-ppo-rnd-extension_…`, confirmed there on 1,024 seeds.

## 2. Key hyperparameters

| setting | value |
|---|---|
| maps | straight grid, 32 cells; snake grid, width 16, 2 turns (50 cells) |
| policies | tabular-PPO (4 action logits and 2 values per cell); coordinate-PPO (shared 2→64→64 ReLU, actor 64→64→4, residual critic) |
| PPO | CleanRL PPO-RND scalars of the fixed-settings study (rollout 128, 4 epochs, 4 minibatches, clip 0.1, discounts 0.999 and 0.99, …), one environment per policy |
| budget | $2^{24}$ = 16,777,216 transitions per policy; screen units stop at $2^{22}$ on the same learning-rate schedule |
| control | online tabular-RND as in the extension: target S→32→64→64→512, predictor S→32→64→64→512→512→512, mean squared feature error, each sample kept with probability 0.25, one Adam with the policy and a joint clip of 0.5, bonus = half the summed squared error, CleanRL reward normalization on |
| screen settings per idea and case | learning rate 1e-3 × entropy {1e-3, 1e-2} × intrinsic weight {2^-6, 2^-4, 2^-2, 2^0, 2^2, 2^4, 2^6}: 14 settings |
| screen seeds | 0-31 (field seeds 0-7, four policies per field) |
| finalist protocol | 52 configurations (learning rate {1e-4, 1e-3} × entropy {1e-3, 1e-2} × 13 weights) on seeds 0-15 at $2^{24}$, then 1,024 confirmation seeds 20000-21023 |
| evaluation | 32 sampled episodes and 1 greedy episode per policy from the start cell |

Idea settings (each one behind a `GridPPOConfig` or `NetworkOptions` key whose default is the control):

| key | values used |
|---|---|
| `rnd_readout` | half squared error (control), length, either one divided by the untrained value |
| `rnd_unit_length_target`, `zero_predictor_output` | off (control); both on = "unit start" |
| `rnd_loss` | CleanRL (control); power law $(2/q)E^{q/2}$ of the summed squared error $E$; shrink target |
| `rnd_optimizer`, `rnd_learning_rate` | shared Adam (control); own SGD, AdaGrad or Adam with its own learning rate |
| `rnd_trainable` | all layers (control); input weights only |
| `bias_std`, `first_width`, `predictor_trunk` | 0, 32, own (control); 0.5; 64; target copy |

Each batch's exact ideas are in `batches/NN_…/rl_ideas.json`.

## 3. Code and config changes

All changes are in commit `9cd855bb` (trainer options, unit runner, bench) and `97f8a3f9` (unit builder, profile
script, report labels); see the extension run's `experiment_background.md`, section 3. The control is unchanged
CleanRL online RND on one-hot inputs: with every new key at its default, the trainer is bit-identical to the
fixed-settings study's source `5abf6b2` (check `check_online_path_unchanged.py`, job 6858052).

Scripts of this study, in `scripts/grid_initialization_campaign/`:
- `rnd_decay_bench.py`: the distillation bench (predictor only, the RL update shape, uniform and start-heavy visits);
- `auto_research_units.py`: writes and queues one batch's units (idea × case × entropy);
- `profile_online_tabular.py`: seconds per iteration at several copy counts on one card;
- `auto_research_report.py`: `analysis/analysis.md` from the finished units.

Workers and the persistent monitor come from the fixed-settings study (`worker.py`, `submit.py`, `monitor.py`); the
campaign phase is `auto-research`, which has no automatic phases: units are queued batch by batch.

## 4. Git state

Repository `/p/rlprojects/RND`, branch `main`. Batch 1 runs the frozen copy
`code/source_9cd855bb4d2c2da5399974396a9183c58fba890a/` (commit `9cd855bb`, pushed 2026-11-09 20:10 PT; recorded in
`source.json`). Later batches record their commit in `STATE.md` and `source.json`.

## 5. Outcome

(filled in when the study ends; interim results are in `analysis/analysis.md` and `STATE.md`)
