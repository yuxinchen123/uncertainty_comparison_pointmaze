# Stage 2: the chosen configurations against RND on seeds that chose nothing (pixel Montezuma's Revenge)

## Purpose

Stage 2 of the plan `/u/sl5nw/.claude/plans/2026-11-17-19-00_montezuma-fast-last-layer-against-rnd.md` (decision 16).
Stage 1 (`/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-18-12-00_pixel-montezuma_fast-last-layer-against-rnd_stage-1-grid_step-sizes-0.01-0.03-0.1-0.3_intrinsic-weights-0.25-0.5-1-2-4-8_sharing-groups-128-16-4_rnd-same-weights_seeds-13-to-20_steps-per-policy-200000000_schedule-horizon-2000000000`) trained every configuration for 200 million steps on seeds 13 to 20 and chose, by the mean score
over the run $S(B)$ (mean ± standard error over the 8 seeds):

1. the fast last layer, sharing group 4, step size 0.01, weight 1: 1,889 ± 278 (the method's best of 72);
2. the fast last layer, sharing group 16, step size 0.01, weight 2: 1,779 ± 326 (its runner-up, within one standard
   error, so it is checked too);
3. RND at weight 2: 1,379 ± 209 (RND's best of 6). RND's best is not the archived baseline's weight 1, so RND is
   trained again here instead of reusing the archived runs.

A pick looks better on the seeds that chose it, so the comparison is made here, on seeds 1 to 12. Each fast-last-layer
configuration is **ahead** of RND when the 95% interval over seeds of the difference in $S(B)$ (method minus RND,
paired by seed, intervals from resampling seeds) lies entirely above zero and that of the final level
$S_{\text{last}}(B)$ (the mean over the last 10% of the windows) does not lie entirely below zero (rule
`/p/rlprojects/RND/.claude/rules/exploration-metrics-atari-montezuma.md`). Stage 3 continues the stage-2 winner and
RND in this folder to 2 billion steps (`build_units.py --continue-units`).

## Key hyperparameters

| setting | value |
|---|---|
| trainer and settings | the same as stage 1 (`config_resolved.yaml`): CleanRL PPO with RND at its defaults, learning rate 1e-4 annealed over a 2,000,000,000-step horizon, entropy 0.001, clip 0.1, 128 environments per policy, 128-step rollouts (16,384 steps per update), 4 epochs of 4 minibatches; the fast last layer in 64-bit arithmetic |
| budget per policy | 61,035 updates = 999,997,440 steps (1 billion); the schedule keeps its 2-billion-step horizon, so every run is the first half of a full run |
| configurations | the three above; the same seed numbers for every configuration, so the comparison is paired by seed |
| seeds | 1 to 12 (12 per configuration); stage 1 used 13 to 20 |
| runs (units) | 36 single units, claimed seed by seed (the three configurations at seed 1, then seed 2, ...) |
| records | `record_game_events` and `record_bonus_profile` on in every unit |
| execution profile | "one copy" for every unit |
| checkpoints | every 1,200 s and at the end of each job (needed to continue a unit in a later job and for stage 3) |

Measured speed of one copy on each UVA card type, read from stage 1's finished one-copy units (median over their
attempts; `infrastructure/card_seconds_per_update.json`, written by `submit.py --card-speeds-from`). Define $s$
as the seconds per update; the hours per stage-2 unit are $\dfrac{61035 \times s}{3600}$.

| card type | seconds per update | hours per unit |
|---|---|---|
| A100 40 GB | 1.360 | 23.1 |
| A40 | 1.376 | 23.3 |
| RTX A4500 | 1.573 | 26.7 |
| Quadro RTX 6000 | 1.739 | 29.5 |
| RTX 2080 Ti | 1.878 | 31.8 |
| RTX A4000 | 2.084 | 35.3 |
| RTX 4000 Ada | 3.014 | 51.1 |

## Code and config changes

- The trainer is the one of stage 1. Changes to the sweep tooling since stage 1 began (all in the frozen copy):
  1. the planner plans cards free now against each account's running jobs only (f4cbedb1) and, for a job pinned to
     cards free now, no longer limits it by the room per card type (9545cedf);
  2. the planner asks for card types in the order of this run's measured seconds per update, fastest first, when the
     run has a speed table (7175b820); stage 1's catalogue order had put the RTX 4000 Ada before faster cards;
  3. files copied back from a remote server get group write, so a worker of the other UVA account can resume a run
     (afeffc19);
  4. the monitor passes every 10 minutes (050843cf).
- Placement: UVA only, both accounts (`uva-sl5nw`, `uva-yuxinchen`) and the owner's reservation `sl5nw_170` asked
  for last, on the listed card types (RTX 2080 Ti, RTX A4000, RTX A4500, Quadro RTX 6000, A40, A100, H100 NVL, RTX 4000
  Ada; RTX 5080 left out), cards free now first and jobs that wait in the queue for the units without a worker, 8-hour
  worker jobs, 4 processors and 10 GiB per card. With about 70 listed UVA cards free at launch every unit starts at
  once; the remote servers are not used, since a four-run H100 pack takes about 38 hours (2.24 s per update), longer
  than one run on a UVA card, and ACES's credits stay for stage 3 if needed.
- The persistent monitor job tops up workers with the same options at every pass and writes `SWEEP_COMPLETE` when
  every unit is done.

## Git state

Commit: `7a69c78a9c9cca0c1367293bb17616422d8ea4f2` (frozen copy 1 in `code/`, `source.json`). The code folders
were clean; modified run-data files of other run folders are left out of the frozen copy.
