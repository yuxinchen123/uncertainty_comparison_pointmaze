# Infrastructure history

Times are Pacific.

- 2026-11-12 00:17 PT: folder created. serval05's card is in use by another account (pyt9dp), so profiles go to serval02 card 1 until it is free.
- 2026-11-12 00:22 PT: test jobs 6873325, 6873326 and 6873328 failed within a second: `platform_jax` has no pytest.
  The test job keeps the `platform_jax` interpreter and adds a folder of links to pytest's own pure-Python packages
  from `atari_pixels_jax` to its path, so JAX, NumPy and MuJoCo still come from `platform_jax`. The job script also
  names the `cpu` partition now.
- 2026-11-12 00:23 PT: environment build job 6873341 failed within a second: the conda path of the older build
  script does not exist on puma01. The script now uses
  `/sw/ubuntu2204/current/software/common/core/miniforge/26.7.2/bin/conda`.
- 2026-11-12 00:24 PT: environment build job 6873344 failed within a second: that conda starts with
  `#!/usr/bin/env python` and puma01 has no `python` on its path. The script now puts miniforge's own `bin` first
  on the path.
- 2026-11-12 01:10 to 01:25 PT: the third ant profile unit waits for serval02 card 1. The card is held by an
  evaluation of another session of this account (`evaluate_single_pool.py play`, RLforOR project, process 1052761,
  started about 01:07 PT, no holder file). serval05's card is held by another account. The rule allows stopping that
  evaluation; I let it finish, because evaluations on this card are limited to 30 minutes, so the wait is short and
  stopping it would lose about 17 minutes of its work. If it is still running after 30 minutes of its own time, it
  is stopped under `serval-profiling-priority.md`, with its command line and folder recorded first.
- 2026-11-12 01:40 PT: PointMaze profile jobs start (ids in `slurm/submitted_jobids_profiles_sl5nw.txt`): 15-minute
  jobs on one RTX A4500 of jaguar03 (through the reservation) or one RTX 5080 of nekomata01, at most three at a time,
  submitted by `infrastructure/submit_profile.sh` and `infrastructure/submit_idea.sh`.
- 2026-11-12 01:50 PT: serval05's card is still used by another account (pyt9dp, six processes, 25.8 GB), so no
  PointMaze profile is queued there.
- 2026-11-12 01:53 PT: profile job 6873896 (idea 25) failed after 3 seconds: an arm's name held a colon, which the
  comparison script reads as the start of the arm's keys. Resubmitted as job 6873901 with names without a colon.
  No other job was affected.
- 2026-11-12 01:50 PT: test job 6873876: one new test failed because it required the packed-key permutation scheme
  to give other row orders than the default scheme. The two schemes read the same random bits, so with few rows the
  orders are the same. The test was wrong, not the trainer; it was corrected and run again.
- 2026-11-12 02:08 PT: the coordinator set the production shape for PointMaze to 4 environments per policy (the
  rollout-shape check gave whole-curve means of 0.89, 0.70 and 0.24 at 4, 16 and 128). From here every idea is
  measured at 4 environments per policy first (512 policies; the card's total steps per second is also read at
  1,024 and 2,048), with 16 environments per policy as the second shape; 128 is dropped. Jobs already running at
  16 and 128 (6873929, 6873946) were left to finish.
- 2026-11-12 02:14 PT: serval02 card 1 shows no process and 0 MiB, but its lock is held by a unit of another
  session of this account. The PointMaze H100 NVL units (`profiles/pointmaze/h100_units/`) are started on serval02
  under `nohup`; each waits for the card and the lock (a try every 30 seconds, at most 25 minutes) and stops nothing.
  Runner log: `logs/h100-nvl_units_runner.log`; the runner's process id: `logs/h100-nvl_units.pid`.
- 2026-11-12 02:33 PT: profile job 6874089 (idea 20 at 256 and 1,024 policies, pinned to nekomata01) was cancelled
  by its id while pending: both RTX 5080 cards of nekomata01 had been taken by the study's first-pass sweep workers
  (account yuxinchen) and Slurm estimated its start for 2026-11-16. It had not started; nothing was lost.
- 2026-11-12 02:34 to 03:45 PT: the first H100 NVL runner on serval02 ran two of its four units: the
  policies-per-process reading at 4 environments (512, 1,024 and 2,048 policies measured; 4,096 was cut off by
  the unit's 15-minute limit, exit code 124) and the time breakdown at 512 policies. The units for idea 20 and
  idea 25 were not run: card 1 stayed busy with units of other sessions for 25 minutes each.
- 2026-11-12 10:31 PT (work continued after a pause of the session): all 8 RTX A4500 cards of jaguar03 are held by
  job 6874312 of another session of this account (`network_distillation_original_reference`, 8 cards, running since
  about 03:10 PT, limit 4 days). No other node has this card, so the ideas after idea 14 have no RTX A4500 column.
  The job is not ours to touch.
- 2026-11-12 10:35 PT: the RTX 5080 cards are free, but another user's processor jobs leave 2 idle processors and
  1.2 GB on nekomata01 and 6 idle processors on nekomata02. The profile jobs therefore run on nekomata02, two at a
  time, with 2 processors each (`PROFILE_CPUS=2` in `infrastructure/submit_profile.sh`); compilation takes longer,
  the iteration time is not affected.
- 2026-11-12 10:34 PT: serval02 card 1 is held by a profile of another session
  (`network_distillation_replication`, started 10:27 PT). A profile is never stopped for another profile, so the
  second list of H100 NVL units (`profiles/pointmaze/h100_units/run_second_unit_list_on_serval02_card_1.sh`,
  started 10:43 PT under `nohup`) waits: a try every 30 seconds, at most 90 minutes of waiting in all. Runner log:
  `logs/h100-nvl_units_second_list_runner.log`; its process id: `logs/h100-nvl_units_second_list.pid`.
- 2026-11-12 10:50 PT: profile jobs 6884200 (idea 17) and 6884203 (idea 15) were cancelled by their ids after 7
  minutes: with 2 processors each, building one program took 312 seconds (60 to 130 seconds with 8), so neither
  could finish its arms inside 15 minutes. The measurements are resubmitted one job at a time with the 6 idle
  processors of nekomata02.
- 2026-11-12 11:06 PT: profile job 6884620 (idea 28, 6 processors, pinned to nekomata02) was cancelled by its id
  while pending: another user's jobs had taken 4 of the 6 idle processors and Slurm estimated its start for
  19:55 PT. From here both RTX 5080 lanes ask for 2 processors per job, with at most three arms at one shape.
  Idea 27 (two processes per card) cannot be built in 15 minutes with 2 processors, so it is measured on the
  H100 NVL only, or on the RTX 5080 when processors are free again.
- 2026-11-12 11:12 PT: I replaced `profiles/pointmaze/h100_units/h100_unit.sh` (so that the processes a unit starts
  itself count as its own) while the idea 20 unit was running on serval02. The unit's measurement finished and
  its record is complete, but the unit's shell then ended with exit code 2 before its last lines (the check for
  another process on the card and the closing log line): a script replaced over the network file system is lost
  to a shell on another machine that is still reading it. No other process was noted on card 1 during that unit
  (no `.other_processes` file). Later units start with the new script. A script in use is not replaced again.
- 2026-11-12 12:26 to 14:20 PT, training iteration with the Warp AntMaze physics (`profiles/antmaze_warp_training/`,
  session 9ca0cb45; the work that put the Warp ant behind `--physics warp`):
  1. serval05 was not used: processes of another account (pyt9dp, 13 to 23 GB) were on its card at 12:28 and at
     13:11 PT.
  2. serval02 card 1 was shared with two other sessions of this account. From about 12:42 to 13:12 PT its lock
     was held by one evaluation unit of the cloud-service session (`flock -w 3600 ...
     serval02_card1_unit_2118328.sh`, 1,806 s, started without the holder file), before and after it by profiles
     of the network distillation session. The starter of the earlier Warp work
     (`profiles/antmaze_warp/units/start_units_when_card_1_is_free.sh`) looks every 30 seconds; a unit that waits
     on the lock itself passed it at 13:12 PT. From 13:13 PT the units of this work wait in line with a blocking
     `flock` (`profiles/antmaze_warp_training/units/start_units_waiting_in_line_for_card_1.sh`). I stopped only my
     own two starters, each while it held no unit; no process of another session was stopped.
  3. On serval02 one measuring process needs 72 to 80 s to load the libraries, and building and priming took
     about 0.115 s per policy (136 s for 1,024 policies, 469 s for 4,096). Unit 1 (three cells) therefore measured
     one cell: the 4,096-policy cell was stopped at the unit's limit and the third cell did not start. The later
     units hold two cells, or one; 8,192 policies cannot be built inside a 15-minute unit.
  4. RTX 5080: job 6886965 (4 processors, 12 GB, nekomata02, submitted 12:43 PT) waits on `Priority`. Both RTX
     5080 nodes had 22 of 24 processors in use by jobs of two other users, and three 2-processor profile jobs of
     another session (6885759, 6886016, 6886953) have waited there since 11:50 PT. RTX A4500: not measured, all 8
     cards of jaguar03 were held by job 6874312 of this account at 12:28, 12:43 and 13:02 PT.
  5. Test jobs on puma01 (ids in `slurm/submitted_jobids_tests_sl5nw.txt`): 6886702, 6886860, 6887246, 6888341
     (Warp interpreter), 6886713, 6886714, 6886926, 6887247 (default interpreter), 6886848, 6887049, 6887832,
     6887841 (single scripts). 6886702 failed on two faults of the new tests, corrected before 6886860; 6887832
     failed because its output path existed only on the login node; 6886926 failed in one test of another
     session's new file (`tests/bonuses/test_rnd_bonus_options.py::test_separate_predictor_step_size`).
  6. The working tree was shared with another session, whose commit 2b84cab1 (13:32 PT) took in this work's
     files as they stood then.
  7. What ran on serval02 card 1 for this work, all without another process on the card: unit 1 (13:26 PT, one
     cell measured), unit 5 (13:54 PT, two cells), unit 8 (14:02 PT, ants with random torques, Warp and JAX
     physics), unit 4 (14:07 PT, the example command stopped and continued), unit 3 (14:25 PT, three cells). The
     32,768-environment cells (unit 6) and the 4,096-policy cell (unit 7) were not run: from 14:37 PT the card
     was held by units of other sessions, an evaluation unit first, and I stopped my waiting starter at 14:47 PT.
- 2026-11-12 14:52 to 15:51 PT, the ceiling in the AntMaze model and the fault record (session 9ca0cb45; records
  and document: `profiles/antmaze_warp_training/`, section "After the ceiling" of `antmaze_warp_training.md`):
  1. serval05 was not used: three processes of another account (pyt9dp, 13,415 MiB together) were on its card at
     14:57 and at 15:26 PT.
  2. serval02 card 1 was held from before 15:12 PT by an evaluation unit of the cloud-service session (started
     without the holder file), and the starter of the network distillation session waited for the lock as well.
     This work's starter (`units/start_units_waiting_in_line_for_card_1.sh`, started 15:12 PT, log
     `logs/start_h100_units_after_the_ceiling.log`) waited in line; its first try at 15:22 PT found the card
     busy. Nothing was stopped and `interrupt_for_profile.sh` was not used.
  3. What ran on serval02 card 1, each without another process on the card: unit 9 (15:28 to 15:32 PT, 16,384
     ants with random torques, Warp then JAX physics, 224 s), unit 10 (15:35 to 15:39 PT, 1,024 policies x 16
     environments, 221 s), unit 11 (15:40 to 15:45 PT, 2,048 policies x 16 environments, 307 s).
  4. Processor jobs on puma01 through the reservation (ids in `slurm/submitted_jobids_tests_sl5nw.txt`, all
     COMPLETED): 6889957 (15:05 PT, periodic leg torques in C MuJoCo), 6890020 (15:08 to 15:12 PT, the new
     ceiling tests with the Warp interpreter, 14 passed), 6890149 (15:16 to 15:25 PT, the card units' scripts at
     a tiny size and the nine ants' record), 6890150 (15:16 to 15:23 PT, the environment's checks against C
     MuJoCo with the ceiling), 6890636 (15:35 to 15:38 PT, the nine ants' record again with 1,024 ants in its C
     MuJoCo random-torque part), and the suites on the changed source: 6890164 (`tests/envs`, 44 passed, 6
     skipped for want of warp-lang), 6890165 (the Warp interpreter's files, 42 passed), 6890166 (`tests/agents`
     and `tests/golden_09`, 80 passed, 2 deselected), 6890167 (`tests/integration`, 23 passed), 6890196 (the two
     JAX ant profile scripts' tests, 16 passed), 6890950 (15:43 to 15:51 PT, the ceiling tests, `test_antmaze_mjx.py`
     and the training-iteration script's test once more after the last source edit, 26 passed).
  5. `units/run_cells.sh` now takes the records folder from `RECORDS_FOLDER` (default `records`); the cells
     measured again went to `records_after_the_ceiling/`. A record of the training-iteration script made from
     now on holds `model_has_ceiling`. Job 6886965 (RTX 5080, queued by the earlier work) was not touched; when
     it starts it measures the model with the ceiling and writes to `records/`.
  6. The working tree is shared: commit d7054945 of another session (15:26 PT) took in the 8 lines this work
     inserted into `scripts/run_training.py` (the fault record in the window record). Without the environment
     files of this work, which were not committed, those lines do nothing.
- 2026-11-12 17:53 PT: the AntMaze Warp training profile for the RTX 5080 (job 6886965, pinned to nekomata02,
  4 processors, 12 GB, 15 minutes) has waited since 12:43 PT with reason `Priority`. Both RTX 5080 nodes
  (nekomata01, nekomata02) show both cards free but 22 of 24 processors allocated to other users' processor-only
  jobs, the longest with about 2 days left; a node with exactly 2 idle processors takes no further job. The job
  holds nothing while it waits and stays queued. Until it runs, the RTX 5080's copies per process for the ant
  come from the calculated estimate, not a measurement.
