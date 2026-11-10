# Extension of the fixed CleanRL PPO-RND grid study: uniform decay exponents and tabular-RND

Times are Pacific. Run folder created 2026-11-09 19:40 PT. Plan:
`~/.claude/plans/2026-11-09-19-03_tabular-rnd-extension-and-online-tabular-rnd-auto-research.md`. The original study:
`runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_…` (complete 2026-11-09). Its report is regenerated with these rows.

## 1. Purpose

Ten conditions are added to every case of the original study, with the same maps, PPO settings, swept settings and
seeds, so the new rows can sit in the original tables:

1. **The uniform field with other decay exponents.** The bonus is $1 \cdot N^{-a}$, where $N$ is the cell's visits,
   for $a \in \{2, 1, 0.25\}$, each with reward normalization off and on. The original study's uniform field is $a = 0.5$.
2. **The frozen tabular-RND field.** The RND networks read a one-hot cell vector instead of the 2-d coordinates. The
   untrained error field is scaled to mean 1 and divided by $\sqrt{N}$. Reward normalization off and on.
3. **Online tabular-RND.** The same one-hot networks, with the predictor trained during PPO as in CleanRL. The bonus
   is the raw error. Reward normalization off and on.

State normalization does not apply to one-hot inputs (user decision 2026-11-09). With reward normalization on, the
chosen configuration of online tabular-RND is confirmed on 1,024 seeds. It is the baseline of the auto research in
`runs/2026-11-09-19-42_online-tabular-rnd-auto-research_…`.

Names used in every table: tabular-RND (one-hot input), coordinate-RND (2-d input), tabular-PPO (the table policy),
coordinate-PPO (the neural policy, which reads the coordinates).

## 2. Key hyperparameters

| setting | value |
|---|---|
| maps | straight grid, 32 cells; snake grid, width 16, 2 turns (50 cells) |
| policies | tabular-PPO (4 action logits and 2 values per cell); coordinate-PPO (shared 2→64→64 ReLU, actor 64→64→4, residual critic) |
| PPO | CleanRL PPO-RND scalars, as in the original study (rollout 128, 4 epochs, 4 minibatches, clip 0.1, ...) |
| swept | learning rate {1e-4, 1e-3} × entropy {1e-3, 1e-2} × intrinsic weight 2^-6 … 2^6; 52 configurations per condition |
| conditions per map and policy | 10: uniform field with decay exponent 2, 1, 0.25 × reward normalization off/on; frozen tabular-RND field × 2; online tabular-RND × 2 |
| tabular-RND networks | target S→32→64→64→512, predictor S→32→64→64→512→512→512 (S = 32 or 50 cells). The input layer has its own seed name; the deeper layers are exactly the coordinate-RND layers of the same field seed |
| online RND training | as in the original study: mean squared feature error, each sample kept with probability 0.25, one Adam with the policy and a joint clip of 0.5; bonus = raw half summed squared error |
| budget | 16,777,216 transitions per policy |
| seeds | development 0-15 (field seeds 0-3); confirmation 20000-20127 (field seeds 5000-5031); online tabular-RND with reward normalization on, chosen configuration: 20000-21023 (field seeds 5000-5255) |
| evaluation | 32 sampled episodes and 1 greedy episode per policy |

## 3. Code and config changes

The changes are relative to the original study's frozen source `5abf6b2`. Each one is behind a new key whose default
keeps the old numbers.
- **`config.py`:**
  - `count_decay_exponent` (default 0.5; 0.5 keeps the exact division by the square root);
  - `rnd_input` ("coordinates" or "one_hot");
  - online RND options for the auto research: readout, unit-length target, loss, own optimizer, trainable layers.
- **`training.py`:**
  - the count bonus `field × visits^(−exponent)`;
  - a one-hot input table for the RND networks only, while the policy keeps the coordinates;
  - the auto research's predictor options.
- **`online_rnd.py`:** `NetworkOptions`, `network_layers` (the one-hot input layer, wider first layer, random biases,
  zero predictor output, target-copy trunk), `input_table`, and the frozen field for any options.
- **Campaign scripts:**
  - `fixed_settings_campaign.py` gains the extension condition set (from `study.json`), with no measurement phase;
    confirmation units of 128 seeds;
  - `train_unit.py` reads the new keys and an optional early stop;
  - `check_fixed_unit.py` covers the run's condition set;
  - new: `check_online_path_unchanged.py`.
- **Gates before launch:**
  - 73 tests passed on puma01, job 6858030;
  - the count path is bit-identical to `0d14eb3`, and the online coordinate path to `5abf6b2`, on the processor
    backend (job 6858052).

## 4. Git state

(filled in at commit time)

## 5. Outcome

(filled in when the study ends)
