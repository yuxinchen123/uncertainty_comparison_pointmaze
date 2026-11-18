# Stage 1: the full grid of the fast last layer against RND on pixel Montezuma's Revenge

## Purpose

Stage 1 of the plan `/u/sl5nw/.claude/plans/2026-11-17-19-00_montezuma-fast-last-layer-against-rnd.md`: every
configuration of the fast last layer and of RND, 8 seeds each, for 200 million steps per policy, so that each side's
best configuration can be chosen for stage 2 (seeds 1 to 12, 1 billion steps).

- **Choice of each side's configuration:** the largest mean score over the run, $S(B)$ of rule
  `/p/rlprojects/RND/.claude/rules/exploration-metrics-atari-montezuma.md` (the mean of the 61-update window values up
  to the budget $B$, over the 8 seeds; a window's value is the mean raw score of the training games that finished in
  it). Ties go to the larger mean over the last 10% of the windows.
- **Reported beside it, never used to choose:** the capped first exit from the start room, the rooms reached, the
  first-time steps of the key, the door, every room, every item and the first enemy kill, the per-window shares of
  games with the key and out of the start room, lives lost by cause, and the score levels 400, 1,000 and 2,500.
- **The live report** `early_signals.md` is rebuilt from the runs' record files at every monitoring pass. No run is
  stopped, held or pruned on it.

## Key hyperparameters

| setting | value |
|---|---|
| game | JAXtari Montezuma's Revenge, pixels, corrected room geometry (asset pack `jaxtari_pixel_study_assets_montezuma_room_geometry_from_png`, record hash `6e523c96...`), at most 27,000 agent steps per game |
| learner | CleanRL PPO with RND at its defaults: learning rate 1e-4 annealed over a 2,000,000,000-step horizon, entropy 0.001, clip 0.1, 128 environments per policy, 128-step rollouts (16,384 steps per update), 4 epochs of 4 minibatches, predictor trained on 25% of each minibatch, extrinsic weight 2, extrinsic discount 0.999, intrinsic discount 0.99 |
| budget per policy | 12,207 updates = 199,999,488 steps; the learning-rate schedule keeps its 2-billion-step horizon, so every run is a prefix of a full run |
| fast last layer, step size | 0.01, 0.03, 0.1, 0.3 |
| fast last layer, sharing group (environments of the 128 that share one temporary layer) | 128, 16, 4 |
| intrinsic weight | 0.25, 0.5, 1, 2, 4, 8 (both methods) |
| configurations | 72 of the fast last layer, 6 of RND |
| seeds | 13 to 20 (8 per configuration); the same seed numbers for every configuration |
| runs (units) | 624 single units: 576 of the fast last layer, 48 of RND; claimed seed by seed (all 78 configurations at seed 13 first, then seed 14, ...) |
| fast-layer arithmetic | 64-bit throughout (`fast_last_layer_float32_products` = "none", user decision 2026-11-18); accepted speed forms of the gates folder: plain errors stored, chunks of 128 rows for groups of up to 16 (the whole rollout for group 4), one step at a time for group 128, the written-out factorization for groups of up to 16 |
| records | `record_game_events` and `record_bonus_profile` on in every unit (RND too); the environment's `report_game_events` on |
| execution profile | "one copy" for every unit: the learner and environment settings of the archived RTX A4500 runs (`convolution_batching` unrolled, `cache_target_features`, `predictor_selected_rows` 1280, `reset_lane_budget` 8); no four-copy packs were made |
| checkpoints | every 1,200 s and at the end of each job (`checkpoint.pkl` beside the records; needed to continue a unit in a later job and for stage 3) |

Measured speed of one copy on the card types measured so far. Define $s$ as the seconds per iteration (one update,
16,384 steps of each copy) and $r = \dfrac{16384}{s}$ as the steps per second of one copy; the hours per million steps
per copy are $\dfrac{10^{6}}{3600 \times r}$, and the hours per unit are $\dfrac{12207 \times s}{3600}$.

| card type | setting (source) | copies | seconds per<br>iteration | total steps<br>per second<br>(millions) | steps per<br>second per<br>copy<br>(thousands) | hours per<br>million steps<br>per copy | hours<br>per unit |
|---|---|---|---|---|---|---|---|
| RTX A4500 | default program (gate 6) | 1 | 1.668 | 0.0098 | 9.82 | 0.0283 | 5.7 |
| RTX A4500 | fast last layer, group 16,<br>both records (gate 6) | 1 | 1.798 | 0.0091 | 9.11 | 0.0305 | 6.1 |
| RTX A4500 | the same, final code (gate 5) | 1 | 1.824 | 0.0090 | 8.98 | 0.0309 | 6.2 |
| RTX 2080 Ti | default program (gate 6) | 1 | 1.987 | 0.0082 | 8.25 | 0.0337 | 6.7 |
| RTX 2080 Ti | fast last layer, group 16,<br>both records (gate 6) | 1 | 2.059 | 0.0080 | 7.96 | 0.0349 | 7.0 |
| RTX 2080 Ti | the same, real unit (gate 7) | 1 | 2.174 | 0.0075 | 7.54 | 0.0369 | 7.4 |
| RTX A4000 | default program (gate 6) | 1 | 2.038 | 0.0080 | 8.04 | 0.0346 | 6.9 |
| RTX A4000 | fast last layer, group 16,<br>both records (gate 6) | 1 | 2.147 | 0.0076 | 7.63 | 0.0364 | 7.3 |

The A40, RTX 5080 and A100 (40 GB) were not measured in gate 6 (no free card then); they are allowed by the user's
decision and their speed is read from the units' own progress records (report section "Speed by card type").

## Code and config changes

- Code: the fast last layer, the game-event records and the bonus profile in the pixel trainer, and the sweep tooling
  `scripts/pixel_sweep_campaign/`, as tested in the gates folder
  `runs/2026-11-17-21-26_pixel-montezuma_fast-last-layer-against-rnd_gates_unit-tests_archive-identity_resume_speed-ideas-10_card-speed-checks/`
  (gates 1 to 7 passed; gate 5 again with the final code on 2026-11-18, gate 7 on 2026-11-18).
- Changes made for this stage (commits 17ffd5b0 and 08b7c4c8):
  1. `worker.py` gives the trainer `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` on an RTX 2080 Ti (the default update ran
     out of card memory at 0.85, gate-6 job 6955822); every other card keeps the job's 0.85, the archive's setting.
  2. `submit.py`: the RTX 4000 Ada is out of the default card types (3.039 s per update in gate 6, below the RTX
     A4000's 2.038 s). Allowed: `nvidia_geforce_rtx_2080_ti`, `nvidia_rtx_a4000`, `nvidia_rtx_a4500`, `nvidia_a40`,
     `a40`, `nvidia_geforce_rtx_5080`, `a100`, `nvidia_a100-pcie-40gb`, `nvidia_h100_nvl`.
  3. `submit.py --cancel-waiting-after S`: a job of a submission that has not started S seconds after it was sent is
     cancelled by its id from this folder's own ledger (`cancel_ledger.py --job-id`); its units stay pending.
  4. `submit.py --memory-gib-per-card G`: the host memory each card asks for (default 16 GiB). One-copy jobs held 5.0
     to 6.4 GiB (sacct MaxRSS of the gate jobs 6955816 to 6956643; the gate-7 unit's worker and trainer together
     6.35 GiB), and 16 GiB let only one card start on a node with 30 GiB unallocated.
- Placement: UVA only, both accounts, only nodes with a free card now (`--only-cards-free-now`), 4 cards that are
  free now left out of every plan for the AntMaze study (`--leave-cards 4`), each card asking for 4 processors and
  10 GiB (about 1.6 times the largest measured MaxRSS), worker jobs of 24 hours (`--worker-hours 24`), and every job still waiting 120 s after its submission
  cancelled (`--cancel-waiting-after 120`). No remote server takes units yet (their environments were being built
  by another worker).
- The persistent monitor job (`monitor.py --loop`, one processor on `nolim`) passes every 20 minutes, returns units
  of ended jobs to pending, tops up workers with the same options, rebuilds `early_signals.md`, and writes
  `SWEEP_COMPLETE` when every unit is done or failed.

## Git state

Commit: `08b7c4c89c8d885704bb559f5b623ca58e36ec2a` (branch `main`, pushed), frozen into
`code/source_08b7c4c89c8d885704bb559f5b623ca58e36ec2a/` (copy number 2) with `scripts/maze_campaign/freeze_code.py
--leave-out-uncommitted`; every job of this folder runs that copy. The units were built from copy number 1 (commit
2523c876, frozen 15 minutes earlier); the two commits differ only in `submit.py`, its README and its test (the
memory option), and no job ever ran copy number 1.

The code folders were clean except for one file left out of the copy:

```
 M 10_jax_exploration_platform/scripts/pixel_sweep_campaign/server_table.json
```

That edit belongs to another worker, who was building the remote servers' environments at the time; UVA workers
never read the file, and the copy holds its committed version. The rest of the working tree held other studies'
uncommitted record files (AntMaze and PointMaze), which this run does not use.
