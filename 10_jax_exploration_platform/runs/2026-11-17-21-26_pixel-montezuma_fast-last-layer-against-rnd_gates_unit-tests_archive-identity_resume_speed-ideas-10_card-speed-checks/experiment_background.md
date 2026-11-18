# Gates and speed work for the fast last layer on pixel Montezuma

## Purpose

Checks that must pass before the 624-run sweep of the plan
`/u/sl5nw/.claude/plans/2026-11-17-19-00_montezuma-fast-last-layer-against-rnd.md` starts: unit tests of the fast
last layer and the game-event records, the default trainer against the archived 2-billion-step RND baseline, resume
bit for bit with the new options, ten speed ideas for the new parts (each profiled in under 10 minutes on serval05 or
serval02 card 1), and a speed and memory check per UVA card type (at most 15 minutes each). No result of this folder
is a scientific result; it decides whether and how the sweep runs.

## Code and data

- Code: commit 3d593552 on `main` (the fast last layer, the game-event records and the sweep tooling); the speed work
  adds options behind defaults that keep the RND program unchanged.
- Reference records: the archived baseline under
  `/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_*/data/`.
- Environment: `/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python`, the corrected asset pack through
  `XDG_DATA_HOME`, `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true` (the environment block in
  `implementation_design.md`).

## Results so far

| gate | result (2026-11-17 PT) |
|---|---|
| 1, unit tests | fast layer 18 passed; game events 13 passed; the default options trace to the same program as commit 19d50152 (8 programs) |
| 2, standard-library tests | 24 passed (fold, resume, archive lines, identity, metric comparison); sweep tooling 85 passed |
| 3, default path against the archive, RTX A4500, 200 updates of seed 5 | 3,000 archived fields equal with logging off and with logging on; final states of the two runs equal (194 leaves) |
| 5, resume with step 0.1, group 16, both record flags | 40 updates against 20 + 20: states, metrics and both record files equal |
| 4, default path against the archive, four copies, serval02 card 1 | 1,500 archived fields of 100 updates equal (`speed_ideas.md`) |
| 6, speed and memory per card type | `speed_ideas.md`; the RTX 4000 Ada is below the RTX A4000's speed and is left out |
| 5 again with the final code (accepted fast-layer forms as defaults), 2026-11-18 11:48 to 11:56 PT | RTX A4500, jaguar03; group 16 (jobs 6956638, 6956639) and group 4 (jobs 6956640, 6956641): 194 of 194 state leaves equal, 40 of 40 metric lines equal, both record files equal byte for byte, in both groups. Code: commit e0b6ab74 (pixel code as at 5aecde3b), clean. 1.824 s per update in both uninterrupted runs (median of updates 21 to 40) |
| 7, one real grid unit through the whole path, 2026-11-18 11:58 to 12:19 PT | PASSED. Folder `data/gate-7_whole-path_one-grid-unit_fast-last-layer_step-size-0.1_sharing-group-16_intrinsic-weight-1_seed-13_updates-300_rtx-2080-ti/`, code frozen at 17ffd5b0. Unit `gate-7_fast-last-layer_step-size-0.1_sharing-group-16_intrinsic-weight-1_seed-13`, 300 updates, both record flags. `build_units.py` made 1 unit; `submit.py --max-new-cards 1 --only-cards-free-now --leave-cards 4 --gpu-types nvidia_geforce_rtx_2080_ti --worker-hours 1 --cancel-waiting-after 120` sent job 6956643 under yuxinchen, pinned to ai02, running within the 120 s (not cancelled). The worker gave the trainer the memory fraction 0.95 (`worker_start` line), the trainer completed 300 of 300 updates at 2.17 s per update (card memory read 10,559 of 11,264 MB), the unit moved to `queue/done` with its attempt and result, and the worker ended because 1,505 s were left (under 1,800). Written: `metrics.jsonl` (300 lines), `game_first_times.json`, `game_windows.jsonl` (4 windows), `checkpoint.pkl`, `progress.json` (completed). One `monitor.py` pass by hand wrote `status.json`, `status.txt`, the monitoring report and `early_signals.md` (all nine sections; standing checks all 0). Nothing had to be fixed. One game ended in update 2 with the key although no key pickup was recorded; it had most likely picked up the key in the observation warm-up, whose random steps are not recorded (a known limit stated in `game_events.py`) |

Speed on an RTX A4500 before the speed work (one copy, rough, other jobs on the node): default 1.730 s per update;
game events +1.5%; fast layer group 128 +5.1%, group 16 with logging +10.4%, group 4 +22.5%.

## Remote servers' smoke jobs

**Decision of 2026-11-18 (the session, by the plan; the user was away and had said not to ask for permission).**
The smoke check exists to catch a broken environment, and the plan accepts that bits differ between card types
(Risks: "recorded and reported; bit-for-bit claims only for runs on one card type"). Define $a$ as a number of the
archived baseline's first update and $n$ as the same number in the smoke job's first default update. A decimal
number passes when $|n - a| \le 10^{-3} |a| + 10^{-6}$; whole numbers and true/false values must be equal. Before,
every number had to be within a relative difference of $10^{-3}$.

- Why: on the Gilbreth A30 (job 12432866), the ACES H100 PCIe (job 2379474) and the DeltaAI GH200 (job 3626431)
  the only field above the old limit was the policy loss of the first update, with values near 1.5e-5 and absolute
  differences of 3.3e-8 to 5.1e-8 (relative 2.1e-3 to 3.3e-3), the size of single-precision rounding. Every other
  field was within 7.5e-4 and every whole-number field was equal.
- Code: `scripts/pixel_sweep_campaign/smoke_check.py` (`outside_first_line_rule`, `compare_with_archive`), commit
  da22476a. A smoke job runs the comparison of its run folder's frozen code copy, so `remote_fleet.py smoke-result`
  decides the check again on UVA from the job's own 20 default metric lines and the reference on the server.
- Gilbreth A30, job 12432866, read again 2026-11-18 13:29 PT: **passed** under the rule (largest relative
  difference 3.33e-3, 0 numbers outside the rule; every other check passed); 1.94 s per update with one copy,
  peak card memory 12,777 MB.
- Gilbreth A100 80 GB (job 12432865) and A10 (job 12432867) were still waiting for idle cards at 14:00 PT
  (estimated starts 2026-11-18 21:17 PT and 2026-11-19 01:32 PT); they stay in this folder's Gilbreth ledger.
  Read them with `remote_fleet.py --run-dir <this folder> smoke-result --server gilbreth` once they have ended.
- ACES and DeltaAI were not read again: ACES's connection is on portal05 (no login to another portal node), and
  neither server is used in stage 1.

## Layout

`implementation_design.md` (the design), `speed_ideas.md` (the speed work), `slurm/jobs/<name>/` (job scripts and
outputs), `slurm/submitted_jobids_*.txt` (ledgers), `data/<check>/` (run folders of the checks), `logs/`.
