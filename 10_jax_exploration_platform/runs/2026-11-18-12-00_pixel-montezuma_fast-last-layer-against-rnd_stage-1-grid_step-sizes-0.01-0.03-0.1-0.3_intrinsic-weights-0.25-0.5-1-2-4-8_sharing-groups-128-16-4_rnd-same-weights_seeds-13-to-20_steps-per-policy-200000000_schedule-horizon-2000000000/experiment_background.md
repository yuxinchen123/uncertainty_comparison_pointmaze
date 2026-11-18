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
  cancelled (`--cancel-waiting-after 120`). Remote servers since 2026-11-18 13:43 PT: section "Remote servers".
- The persistent monitor job (`monitor.py --loop`, one processor on `nolim`) passes every 20 minutes, returns units
  of ended jobs to pending, tops up workers with the same options, rebuilds `early_signals.md`, and writes
  `SWEEP_COMPLETE` when every unit is done or failed.

## Remote servers

Decisions of 2026-11-18 by the session, under the plan (the user was away and had said not to ask for permission).

**The smoke check's first-update comparison.** The smoke check exists to catch a broken environment, and the plan
accepts that bits differ between card types (Risks: "recorded and reported; bit-for-bit claims only for runs on one
card type"). Define $a$ as a number of the archived baseline's first update and $n$ as the same number in a smoke
job's first default update. A decimal number passes when $|n - a| \le 10^{-3} |a| + 10^{-6}$; whole numbers and
true/false values must be equal. Before, every number had to be within a relative difference of $10^{-3}$; the
policy loss of the first update (values near 1.5e-5) differed by 3.3e-8 to 5.1e-8 on the A30, H100 PCIe and GH200,
which is single-precision rounding. Code: `scripts/pixel_sweep_campaign/smoke_check.py`, commit da22476a. The
trainer copy the workers run (copy 2, commit 08b7c4c8) was not frozen again: its smoke job still runs the old
comparison, and `remote_fleet.py smoke-result` (the UVA working tree) decides that check again from the job's own
20 default metric lines on the server.

**Where units go.**

| server | pool | units | limits | why |
|---|---|---|---|---|
| Purdue Gautschi | `ai-h100`, `-q normal`,<br>1 H100 and 14 cores | four-copy packs only | at most 4 jobs of 10 hours,<br>8 packs held at once | account `dsimchil` holds 4 H100 at once, shared with long635;<br>no job of either at 13:23 PT, so all 4 used |
| Purdue Gilbreth | `standby-a30`, 4 hours | one-copy units, once<br>its smoke job passes | at most 20 jobs,<br>20 units held at once | 24 standby cards for the account,<br>4 left for the AntMaze study |
| Purdue Gilbreth | `standby-a100-80gb`,<br>`standby-a10` | none yet | held back in the table | card type not checked yet |
| TAMU ACES | `gpu-h100` | none | not ready | credits kept for stages 2 and 3 (plan, section D) |
| NCSA DeltaAI | `ghx4-h100` | none | not ready | last in the order of servers (plan, section D) |

- **Gautschi's jobs last 10 hours**, not the table's 24: a pack at the measured 2.30 s per update needs 7.8 hours,
  so a card goes back to the shared account within 10 hours when long635 waits for one. The hand-over loop lowers
  Gautschi's limit by the cards long635's running and waiting `ai` jobs ask for, before every tick (at most
  $4 - c$ jobs and $2(4 - c)$ packs held, where $c$ is those cards). Our waiting Gautschi jobs are not cancelled
  when long635 submits (the ledger cancel tool cancels a server's waiting jobs only as a whole, after `RUN/STOP`).
- **Packs** are made from never-started singles in claim order (`build_units.py --pack-pending`): the first 8, at
  13:43 PT, were seed 13 of the fast last layer at sharing group 16 (weights 0.5, 1, 2, 4, 8) and sharing group 4
  (weights 0.25, 0.5, 1). The loop makes new packs so that the packs pending on UVA and held by Gautschi reach its
  limit. A pack runs the "four copies" execution profile (the archived H100 run's settings), so these runs differ
  in their bits from one-copy runs of the same configuration; the report marks the card of every run.
- **Gilbreth is not ready yet** (2026-11-18 14:00 PT). Its A30 smoke job of this folder, 12434104 on gilbreth-d004
  (a D node, 16 cores), passed every check of the environment and of the numbers (9 default updates written, 0
  numbers of the first outside the rule), but the trainer needed about 7 minutes to its first update, so only 9 of
  20 default updates and 1 fast-layer update fit in the smoke job's 420 s per training; it did not pass. The gates
  folder's A30 job 12432866 on a B node wrote all 20 updates of both trainings. A second smoke job, 12434389, pinned
  to the B nodes (`--constraint=B`, for this one job only; workers may use every A30 node) waited for an idle card
  (estimated start 18:14 PT). The hand-over loop reads its verdict once it has ended; Gilbreth becomes ready only if
  it passed. A worker job on a D node loses about 7 minutes of its 4 hours to the start.
- **Gilbreth's A100 80 GB and A10** are held back: the gates-folder smoke jobs 12432865 and 12432867 were still
  waiting for idle cards (estimated starts 2026-11-18 21:17 PT and 2026-11-19 01:32 PT). Adding one needs its
  gates-folder verdict, the speed bar, and a smoke job of this folder; the A100 80 GB would take packs (four copies).
- **Speed bar for a card type**: its seconds per iteration (one update of every copy in the process) at most the
  RTX A4000's 2.31 s of this sweep, so no run on it takes longer than on an RTX A4000. The Gautschi H100 (2.30 s,
  four copies) and the A30 (1.94 s, one copy) pass.

| card type (server) | setting (smoke job) | copies | seconds per<br>iteration | total steps<br>per second<br>(millions) | steps per<br>second per<br>copy<br>(thousands) | hours per<br>million steps<br>per copy | hours<br>per unit |
|---|---|---|---|---|---|---|---|
| H100 80GB HBM3 (Gautschi) | default (18909289) | 4 | 2.30 | 0.0285 | 7.12 | 0.0390 | 7.8 |
| H100 80GB HBM3 (Gautschi) | fast last layer, group 16,<br>both records (18909289) | 4 | 2.30 | 0.0285 | 7.12 | 0.0390 | 7.8 |
| A30 (Gilbreth) | default (gates folder, 12432866) | 1 | 1.94 | 0.0084 | 8.45 | 0.0329 | 6.6 |
| A30 (Gilbreth) | fast last layer, group 16,<br>both records (12432866) | 1 | 1.95 | 0.0084 | 8.40 | 0.0331 | 6.6 |

**The hand-over loop** (`infrastructure/hand_over_loop/`, a copy of
`/tmp/claude-2618919/-p-rlprojects-RND/9ca0cb45-e2aa-43c6-9353-3a3d17cddb5f/scratchpad/montezuma_hand_over_loop.sh`
and its `montezuma_hand_over_steps.py`): every 5 minutes it checks the shared connections to Gautschi and Gilbreth
(a closed one is opened once with the key, counted for all sessions; three failed logins in a row end the loop),
sets Gautschi's share, reads Gilbreth's smoke verdict once job 12434389 has ended, makes packs, runs
`remote_fleet.py hand-over-tick`, and checks that every unit stands in one queue folder only. It ends when no unit is pending on UVA or held by a server, at `RUN/STOP` or `RUN/SWEEP_COMPLETE`,
or after 4 days, and logs every pass with a Pacific time.

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
