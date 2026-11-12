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
