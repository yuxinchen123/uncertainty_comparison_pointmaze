# Infrastructure history: RND baseline on AntMaze

Times are Pacific, read from the clock or from scheduler records.

- 2026-11-12 15:57 PT: folder created, code frozen at commit 30b2c609, 8 units built (rollout-shape check).
  Placement: MIT cannot start the work (other sessions of this account hold 4 of 4 cards on `mit_preemptable` and
  have 5 jobs waiting on `mit_normal_gpu`); Gautschi's 2 H100 cards are in use by another session; so the units go
  to Gilbreth's standby cards with at least 24 GB (A100 80 GB, A30; not charged, 4-hour jobs, units continue from
  their checkpoints). ACES is next if Gilbreth is too slow.
- 2026-11-12 15:57 PT: the automatic first-unit check compares a server's first unit with a unit trained on UVA,
  and no AntMaze unit can run on UVA's cards today. The check was therefore set to passed by hand in the server
  table, with one unit at a time: the unit with 128 environments per policy and cells 41 to 48 went to Gilbreth
  (job 12346318). Its records are compared by hand with the UVA example run of the same trainer before more units
  are handed over.
- 2026-11-12 15:57 PT: monitor job 6891245 (UVA, no worker submissions).
- 2026-11-12 16:17 PT: first unit read by hand at window 36 of 256 (job 12346318 on an NVIDIA A30, started
  16:03 PT): the Warp kernels compiled on the node; record kinds and per-policy fields are complete, with no
  non-finite value; the contact pool is 8.2% used; one iteration at 16,384 environments takes 3.85 s, which is 0.545
  million steps per second (an H100 NVL: 1.67 million). The other 7 units were handed over (8 at once now).
  At 2.4 million steps per policy some policies of this unit already reach the goal. Its policies spend 0.7% to
  8.2% of their steps within 0.5 m of the ceiling (a random policy: about 0.33%), and their largest torso speeds
  are 7.8 to 12.8 m/s: policies trained with the bonus leave the floor far more often than random ones.
- 2026-11-12 16:18 to 16:31 PT: TAMU ACES set up for this run's units. No job was submitted there, `prepare` and
  `hand-over-tick` were not run for it, and the server table was not changed. ACES was reached only through the
  open connection on portal05 (checked 16:18:28 PT; no new login).
  1. **State of the account, read 16:18:34 PT.** `myproject`: 145,000 SUs allocated, 73,501.10 used or pending,
     71,498.90 left, which is 525.7 H100-hours at 136 SUs per hour. Jobs of this account: 4 waiting on `gpu`, none
     running (2 with names starting `cs`, 2 with names starting `cmc1f128`; they belong to other sessions). H100
     nodes of `gpu`: 26 cards on 5 nodes, 22 in use; the 4 cards not in use are all on ac041, whose state `mix-`
     says the scheduler holds it for a planned job; no node is idle. 114 waiting jobs of all users on `gpu` wait on
     priority.
  2. **Environment** `/scratch/group/p.cis261532.000/envs/rnd_platform_jax_mujoco_warp` (Python 3.12.13), built on
     the login node alogin1 from 16:25:47 to 16:29:04 PT with the commands of the Purdue scripts and the same
     requirement and constraint files (same checksums on ACES), JAXAtari at commit 035fc46b included. Cost: 0 SUs
     (no job). It is in the project's folder, not under `/scratch/user/u.sl359075/envs/`, because the user's scratch
     folder held 202,620 of the 250,000 files it may hold and the environment has 36,678 files; the project's
     folder held 106,482 of 500,000 before and 147,122 after. 79 packages, the same list as on Gilbreth. Against
     UVA's `platform_jax`: `pip` 26.2.2 (UVA 26.2.1), `setuptools` 84.2.1 (UVA 84.0.0), `warp-lang` 1.14.0 added
     (the version of UVA's `platform_jax_warp`), `PyMuPDF` absent. JAXAtari's file record has 296 lines; 294 are the
     same as on UVA, and the 2 others are the `install-sprites` command (its first line names the environment) and
     the `WHEEL` note. Build script, package list and build log:
     `infrastructure/remote_fleet/setup/build_rnd_platform_jax_mujoco_warp_env_aces.sh`,
     `infrastructure/remote_fleet/setup/pip_freeze_aces_rnd_platform_jax_mujoco_warp.txt`, and on ACES
     `/scratch/user/u.sl359075/rnd_maze/setup/`.
  3. **Check without numerical work, 16:29:58 PT, login node.** Python finds `jax`, `jaxlib`, the CUDA 12 plugin,
     `mujoco`, `mujoco.mjx`, `warp`, `flax`, `optax`, `numpy`, `scipy` and `jaxatari`; the package records give
     jax 0.11.0, numpy 2.5.2, mujoco and mujoco-mjx 3.11.0, warp-lang 1.14.0, flax 0.12.8, optax 0.2.8; 43 NVIDIA
     library files and Warp's two binaries are present. No JAX, MuJoCo or Warp code was run, so whether the CUDA
     libraries and Warp's kernel compiler work on an ACES card is not known until the first unit runs. The grid
     study's environment on ACES has the same JAX and NVIDIA package versions, and JAX used the card in 20 of its job
     logs there (the latest: job 2336764 on ac055, started 2026-11-10 15:44 PT).
  4. **Server-table entry**, in its own file `infrastructure/remote_fleet_servers_aces_entry.json`, to be merged
     with `infrastructure/remote_fleet/setup/merge_aces_entry_into_server_table.py` (tried on a copy of the
     table): route through portal05 with the connection check, pool `gpu-h100` (account 158590754315, partition
     `gpu`, 1 node, 1 task, 8 cores, 39 GB, 1 H100, 24 hours, at most 2 jobs; 136 SUs per hour, 3,264 SUs for a
     job that runs 24 hours), `units_at_once` 0, `match` antmaze, the balance read with `myproject`.
  5. **Job script.** `scripts/maze_campaign/remote_fleet.py` takes two new fields of a server's entry, both
     defaulting to nothing: `job_login_shell` (`#!/bin/bash -l`) and `job_setup_lines`. For ACES the lines set
     `WARP_CACHE_PATH` to the job's own folder on the node (`/tmp/job.<job id>`, set by the scheduler as TMPDIR)
     and print TMPDIR, the library path and the loaded modules into the job's log. A unit downloads nothing
     (Warp compiles its kernels on the node; JAXAtari downloads only in its `install-sprites` command), so the
     module `WebProxy/0000` is not loaded. The login shell on ACES loads no module and sets no library path
     (read on the login node); the script clears `LD_LIBRARY_PATH` after these lines in any case. The script the
     changed code writes for Gilbreth is identical to the one sent today.
  6. **`units_at_once` 0 now means no unit in every state of the first-unit check.** Before this change a server
     whose first unit was not yet verified had a limit of 1 whatever `units_at_once` said.
  7. **Tests.** `tests/maze_campaign`: 92 passed (91 before, 1 new, 1 extended), UVA job 6891860 on puma01,
     started 16:28:47 PT.
  8. **Found, not changed.** `remote_fleet.py` reads the server table before it takes the fleet's lock, so a tick
     that started before a merge and writes the table afterwards would write its older copy.
- 2026-11-12 16:45 PT: ACES is prepared but not in use. Its entry is kept in
  `infrastructure/remote_fleet_servers_aces_entry.json` and is NOT merged into the live server table, so that no
  fleet call touches the connection on portal05 before units are meant to go there. Reasons for waiting: no H100
  node is idle on ACES (22 of 26 cards in use, 114 jobs of all users waiting, 4 of them of other sessions of this
  account), and the 8 units are already on Gilbreth. ACES is merged in when the development sweeps of the three
  maps are ready, or when a unit has waited about 2 hours on Gilbreth. Balance read 16:18 PT: 71,498.90 SUs,
  which is 525.7 H100-hours at 136 SUs per hour.
- 2026-11-12 16:22 PT: **the first unit stopped itself at window 64 of 256** (job 12346318): one ant needed 161
  constraint rows and the limit was 160 (`warp_constraint_rows_per_environment`), so MuJoCo Warp would have
  dropped part of the physics and the trainer's check ended the run. The limit had been sized from random
  torques, which needed 49 rows; trained policies press the ant into walls, floor and ceiling at once.
  Correction (commit 5c7407bb): the limit is 512 in the default and in the recorded configuration files. The
  8 units were taken back from Gilbreth, the code was frozen again, and the units were rebuilt with 512. The two
  units that had started with 160 (the stopped one, and the unit with 4 environments per policy, whose job
  12346363 had started at 16:38 PT and was cancelled by its ledger id after 77 s) were built anew; their earlier
  queue records are in `replaced_2026-11-12_started-with-160-constraint-rows/` here, and their shards and logs in
  a folder of the same name in the run folder on Gilbreth. Nothing of the stopped unit is used.
- 2026-11-12 16:37 PT: the waiting ant jobs had planned starts 4 to 7 hours away on Gilbreth's A100 and A30
  cards. They may now also take A10 cards (24 GB), and the PointMaze folders stopped handing units to Gilbreth
  (their waiting jobs there were cancelled by ledger id), so that this account's standby places go to the ant.
- 2026-11-12 16:57 PT: 8 rebuilt units handed to Gilbreth again, 8 jobs submitted.
- 2026-11-12 17:01 to 17:20 PT: **two or three jobs ran one unit at once; everything of that period is discarded.**
  Cause: the take-back of 16:39 PT left a `withdrawn` marker per unit on the server. After the units were handed
  to the same server again (16:57 PT), every fleet step read the old marker as "withdrawn but not copied back",
  took six units back and handed them over again (17:01, 17:06, 17:11, 17:16 and 17:20 PT in `fleet_log.txt`).
  Each repeat put the unit into the server's waiting queue again while a job was running it, and the next free
  job started it a second time (the unit logs show the same window twice with different numbers; the unit with
  16 environments per policy was started three times). The two units built anew at 16:55 PT had no marker and
  were not affected, but they had not started.
  Found at the 17:20 PT tick from the remote listing (every unit "pending" with windows written).
  What was done: hand-overs stopped and the server marked not ready (17:22 PT); the 8 running jobs 12346505 to
  12346512 cancelled by ledger id; on Gilbreth the run folder's `queue`, `data`, `checkpoints` and `logs` moved to
  `replaced_2026-11-12_two-jobs-on-one-unit/`; on UVA the queue records, specifications, unit configurations and
  the fleet's assignment records moved to a folder of the same name here; the 8 units built fresh.
  Correction in the fleet (commit 89cd0c5b, with a test): a unit is taken back for a `withdrawn` marker only when
  the server holds it in no other queue folder.
- 2026-11-12 17:24 PT: 8 fresh units handed to Gilbreth, 8 jobs submitted. The fleet step run again at 17:26 PT
  took nothing back and handed nothing over; the server held 8 waiting units and no `withdrawn` marker.
- 2026-11-12 17:27 to 17:41 PT: two of the 8 fresh units failed three times each at their start (the unit with 4
  environments per policy, jobs 12346600, 12346603, 12346606; the unit with 128 environments per policy and cells
  41 to 48, jobs 12346625, 12346649, 12346665): the worker refuses a unit whose stored configuration file differs
  from its specification, and the server still held those two units' configuration files with the 160-row limit
  from the first hand-over (the folders set aside at 17:22 PT were `queue`, `data`, `checkpoints` and `logs`, not
  `slurm/unit_configurations` and `specifications`). The other six units started once each and run.
  Correction: the two stale configuration files, their specifications and failed records moved to
  `replaced_2026-11-12_stale-unit-configurations/` on the server; the two failed queue records moved to a folder
  of the same name here; the two units built again and handed over at 17:41 PT (jobs 12346670, 12346671). At
  17:44 PT all 8 units ran, one job each, and the fleet step handed over nothing.
- 2026-11-12 17:44 PT: speed with the 512-row limit, read from windows written: units with 128 environments per
  policy on A10 cards about 5.2 s per iteration (16,384 environments), so about 1.5 hours per unit; the unit with
  16 environments per policy on an A30 about 4.7 s per iteration (12,288 environments), so about 10 hours. With
  the 160-row limit an A30 took 3.85 s per iteration at 16,384 environments, so the higher limit costs time; the
  windows record the largest number of rows in use, and the limit for later sweeps is set from that.
- 2026-11-12, between 18:00 and 18:04 PT (the tick's start and the commit of this note; the copy keeps the server's file times): **the physics diverges for trained ants; the running check cannot be used for a decision.**
  Read from the partial records of all 8 units (copied from Gilbreth, standard-library reading):
  1. untrained ants: largest torso speed 9 to 10 m/s; 0.4% of steps with the torso above 1.5 m;
  2. once the policies learn (from window 45 to 110 of 256 with 128 environments per policy), 7% to 11% of steps
     have the torso above 1.5 m, and in most configurations 2 to 9 of 16 policies show a largest torso speed
     above 10,000 m/s at some window (from 1e6 up to 2.5e15 m/s). The two smallest intrinsic weights at learning
     rate 1e-4 show none;
  3. arrivals: 1% to 2% of episodes, after 550 to 580 of 700 steps on average, with and without such speeds;
  4. no unit stopped on its constraint-row or contact limit, so the limits are not the cause.
  The units run Warp with 4 solver iterations and 8 line-search iterations (MuJoCo's defaults: 100 and 50); that
  is the first suspect, not a finding. C MuJoCo resets a simulation whose state is not finite or beyond 1e10;
  this environment respawns on non-finite values only, so a diverged ant keeps values like 1e12 in its
  observations until the episode ends, and the RND input statistics keep them for the rest of the run.
  Decision: the 8 units keep running on Gilbreth's idle cards (no charge) for their records and checkpoints, which
  the diagnosis uses; nothing is chosen from them. A separate diagnosis was started at 18:04 PT in
  the speed study folder (`profiles/antmaze_warp_blow_up/`): reproduce, find the mechanism, measure corrections
  with their cost, add a guard that respawns a state beyond a bound.
- 2026-11-12 18:38 and 18:40 PT: the fleet step could not copy finished units back (`rsync ... exit 11`, "mkdir
  .../data/research failed: No such file or directory"): the folders `data/` and `checkpoints/` did not exist in
  this run folder on UVA after the clean-up of 17:22 PT. Created `data/research/` and `checkpoints/`; the step of
  18:41 PT copied, verified and moved 5 units to `queue/done` (the units with 128 environments per policy, cells
  1 to 40; 3,764 to 4,049 s each on A10 and A30 cards). Nothing was lost: the records stayed on the server until
  the copy succeeded. These units are records for the physics diagnosis only (entry of 18:00 to 18:04 PT).
- 2026-11-12 19:15 PT: **the diagnosis of the diverging physics is done** (speed study folder,
  `profiles/antmaze_warp_blow_up/notes.md`, commit ccd16d50; I checked its key numbers in the records).
  1. Cause: the integrator, not the solver (my first suspect is refuted). `implicitfast` at a 0.01 s time step
     adds rotational energy to a fast-turning ant in the air: the torso's turning rate grows over about 24 steps
     (32, 55, 94, 138, 188, 326 rad/s) and the speed follows in the last few steps.
  2. Replay of 33 recorded runaways in C MuJoCo: `implicitfast` and Euler repeat 20 to 32 of them at every
     solver setting; RK4 (the reference model's integrator) repeats none; `implicitfast` at 0.005 s repeats 1
     or 2 when started 8 or more steps before.
  3. On the H100 NVL with the same trained policies: training physics, 33 runaways in 362 million steps (largest
     5.7e13 m/s); RK4, none in 104 million (largest 14.6 m/s); `implicitfast` at 0.005 s with 10 physics steps,
     none in 127 million (largest 14.3 m/s). Seconds per training iteration at 16,384 environments: 1.374, 4.927
     and 2.707.
  4. A guard was added, off by default: an environment whose torso passes a speed or turning-rate bound, or
     leaves the maze's volume, is respawned without an arrival and counted
     (`physics_fault_respawns_per_copy`).
  **Decision (mine, reported to the user, reversible):** the AntMaze baseline runs `implicitfast` at 0.005 s
  with 10 physics steps per environment step, and the guard at 30 m/s and 100 rad/s as a second line of defence.
  With the guard alone the turning below the bounds stays inflated (0.53% of steps at 20 to 40 rad/s against
  0.28% with RK4 and 0.29% at 0.005 s). Cost: 1.97 times the time per iteration; RK4 would cost 3.59 times.
  Also seen: under the stable physics the same policies still spend about 7% of their steps with the torso above
  1.5 m, so launching itself is something this strong ant really does, not a fault of the integration.
- 2026-11-12 19:16 PT: the two old check units still running on Gilbreth (16 and 4 environments per policy) were
  stopped: hand-overs from this folder set to 0 units at once, jobs 12346602 and 12346671 cancelled by their ids
  from this folder's Gilbreth ledger (the fleet's take-back moves waiting units only). The 6 finished units stay
  as records; no result of this check is used. The check starts again under a new phase name once the time step
  is a recorded configuration field with its gates (work started 19:17 PT) and the code is frozen anew.
- 2026-11-12 19:20 PT: **the fleet step started the two stopped units again.** After their jobs were cancelled
  (19:16 PT) the units stood in the server's waiting queue, and the step of 19:20 PT submitted two jobs for them
  (12346977, 12346978): a hand-over limit of 0 stops new hand-overs, it does not stop jobs for units the server
  already holds. The take-back of 19:21 PT moved nothing because the new jobs had claimed the units.
  What was done (19:21 PT): Gilbreth marked not ready for this folder (the step then does nothing there); both
  jobs cancelled by their ids from this folder's Gilbreth ledger; the two units' queue files set aside on the
  server and their queue records and assignment records set aside here
  (`set_aside_2026-11-12_old-physics-check-units/`). The step of 19:22 PT reports "not part of this tick". The
  two jobs ran about a minute; nothing of theirs is used.
  To stop a unit that runs on a server: mark the server not ready first, then cancel its job, then set the unit
  aside. A take-back only moves units no job holds.
- 2026-11-12 19:42 PT: one more job for the stopped old units was still running on Gilbreth, found at the 19:40 PT
  tick: the five-minute hand-over loop's step of 19:18:45 PT, two minutes after the first cancel, had submitted
  jobs 12346964 and 12346965 for the two units (the fleet log shows it). 12346964 ended by itself at 19:21 PT when
  its unit's queue file was set aside; 12346965 had claimed the unit with 4 environments per policy and ran it for
  23 minutes. Cancelled by its id from this folder's Gilbreth ledger. No maze fleet job is left on Gilbreth, and
  nothing of these runs is used. In all, 6 jobs were started for the two stopped units between 19:18 and 19:21 PT
  by two steps running five minutes apart; marking the server not ready BEFORE cancelling would have prevented
  all of them.
- 2026-11-12 19:58 PT: the stable physics is a recorded configuration (commit f490e70d; report of the background
  work, whose test and gate numbers are in the speed study folder's `profiles/antmaze_warp_blow_up/notes.md`):
  1. new field `physics_time_step`; the baseline's file is
     `configs/experiments/antmaze_umaze_warp-physics_time-step-0.005_frame-skip-10_fault-guard-30-metres-and-100-radians-per-second_…toml`;
  2. gates at 0.005 s against C MuJoCo pass in both implementations (one step without contact 9.3e-7 m against a
     bound of 1e-4 m; with contact 8.4e-3 m against 5e-2 m);
  3. gate with trained policies through the real code path (serval02 card 1, 19:29 to 19:36 PT): 228.6 million
     environment steps, 0 physics-fault respawns, 0 non-finite respawns, largest torso speed 14.8 m/s, largest
     turning rate 61.9 rad/s, 6.9% of steps with the torso above 1.5 m, at most 179 of 512 constraint rows;
  4. throughput on an H100 NVL at 16,384 environments (serval02 card 1, 19:37 to 19:43 PT): 2.72 s per training
     iteration with 128 environments per policy and 2.74 s with 16; 0.77 million steps per second in total; peak
     card memory 17.8 and 18.3 GiB.
- 2026-11-12 20:00 PT: **the rollout-shape check starts again with the stable physics.** Code frozen at f490e70d;
  study description `infrastructure/descriptions/antmaze_umaze_rollout_shape_check_stable_physics.json`, phase
  `rollout-shape-check-stable-physics`: 4, 16 and 128 environments per policy; 48 configurations (learning rates
  3e-4, 1e-3, 3e-3, 1e-2; entropy 0.001 and 0.01; intrinsic weights 0.0625 to 64); 16 seeds; 16,777,216 steps per
  policy; 18 units of 8 configurations each, so every count runs as 6 units side by side. The learning rates start
  one step higher than in the stopped check, whose records (not used for any decision) had 1e-4 last and 3e-3,
  the top of its grid, first. The units' speed records go under workload names that carry the time step.
  The 48 configurations of the chosen count also serve as RND's first development sweep on the U-maze.
- 2026-11-12 20:01 PT: Gilbreth ready again for this folder with a limit of ONE unit; the unit with 4 environments
  per policy and cells 41 to 48 was handed over (job 12347248). Its records are read by hand before the limit is
  raised to 18. (With the first-unit state "waiting" the fleet hands over nothing, because it wants a UVA unit to
  compare with; as before, the state is "passed" by hand and the limit does the gating.)
- 2026-11-12 20:28 PT: **first unit with the stable physics read by hand; the other 17 released.** Job 12347248
  waited on priority from 20:01 to 20:15 PT, started on an A30 (gilbreth-b010), compiled the Warp kernels (no
  cache on that node) and began iterating at about 20:19 PT. Read at window 2 of 256 (unit with 4 environments per
  policy, cells 41 to 48): settings as intended in the start record (time step 0.005 s, frame skip 10, guard 30
  m/s and 100 rad/s, solver 4 / line search 8, 512 rows); record kinds job, unit_start, episode_window; 128 values
  per policy field, all finite; 0 fault respawns; largest torso speed 13.0 m/s; largest turning rate 40.0 rad/s;
  1.3% and 2.8% of steps with the torso above 1.5 m; contact pool use 9.1%; at most 41 of 512 constraint rows.
  Speed: 1.087 s per iteration at 512 environments, so a unit with 4 environments per policy (32,768 iterations)
  takes about 9.9 hours, in three jobs of at most 4 hours. Already in window 2 (131 thousand steps per policy),
  20 of 128 policies had reached the goal, arrivals taking 470 to 620 steps.
  At 20:28 PT the limit was raised to 18: 17 units handed over, 17 jobs submitted (all waiting at 20:28 PT). The
  five-minute hand-over loop runs again (`hand_over_loop.sh` in the session's scratchpad, background task of the
  session, at most 12 hours).
  A stale claim file of the old unit with 4 environments per policy, left in the server's running folder by the
  job cancelled at 19:41 PT, was set aside with the other old files.
- 2026-11-12 21:02 PT: all 18 units run on Gilbreth (15 on A100 80 GB cards, 3 on A30 cards), one job each, no
  unit started twice. Fault guard, read over all units so far: 2 respawns, both in units with 16 environments per
  policy (largest turning rate seen 120.4 rad/s, above the bound of 100), in about 1.2 billion environment steps of
  those units; none in the units with 4 and with 128 environments per policy. Largest torso speed anywhere: 15.0
  m/s. The guard's count per unit goes into every later reading.
- 2026-11-12 23:06 PT: the six units with 128 environments per policy are done, copied back and verified (five by
  22:00 PT on A100 80 GB cards, the one on an A30 by 23:06 PT). Fault guard over these six: 32 respawns in 12.9
  billion environment steps; largest torso speed 16.9 m/s; largest turning rate 145.5 rad/s (above the bound of
  100, so the guard did act). The 12 other units run, one job each. The units with 4 environments per policy are
  at 67 to 82 of 256 windows after 2.5 hours; their jobs reach the 4-hour limit at about 00:28 PT and continue in
  new jobs from their checkpoints.
- 2026-11-12 23:26 PT: the six units with 16 environments per policy are done, copied back and verified (about 3
  hours each on A100 80 GB cards). Fault guard over them: 43 respawns in 12.9 billion environment steps; largest
  torso speed 17.0 m/s. The six units with 4 environments per policy run (78 to 92 of 256 windows), one job each.
- 2026-11-12 23:28 PT: two units of the phase `budget-probe` built and handed to Gilbreth (study description
  `infrastructure/descriptions/antmaze_umaze_budget_probe_stable_physics.json`): RND's leading configurations for
  134,217,728 steps per policy with 128 environments per policy and for 67,108,864 with 16, learning rates up to
  3e-2 (the best one of the check was at the top of its grid). Each needs several 4-hour jobs.
- 2026-11-13 00:06 PT: the first unit with 4 environments per policy (job 12347248, started 20:15 PT) reached the
  4-hour limit and stopped with its state saved at window 112 of 256; the fleet step submitted its next job,
  which was waiting on priority at 00:06 PT. The other five units of that count reach their limit at about
  00:28 PT. The two units of the longer run are at 4 and 6 of 256 windows on A30 cards.
- 2026-11-13 02:57 PT: **this folder's code was frozen anew at commit a2df3c12** (before: f490e70d), so that it
  holds the option `reward_divisor=held_after_first_arrival`. Between the two commits the trainer changed only by
  that option, which adds nothing to the saved state, so a unit started under the old copy continues from its
  checkpoint under the new one with the same results. Units of the baseline still running at that time: the six
  check units with 4 environments per policy and the two longer-run units.
- 2026-11-13 02:58 PT: **six units of the auto research folder's batch 10 run here under the phase
  `bonus-arms-screen`** (the ant's server settings live in this folder): on the U-maze, the divisor held from the
  first arrival, the fast last layer, and both together, each with 128 and with 16 environments per policy, 8
  configurations and 16 development seeds per unit, 16,777,216 steps. Description and predictions:
  `batches/10_fast-last-layer-with-the-first-arrival-hold_pointmaze_and-ant-umaze-screen-of-the-bonus-arms/` of the
  auto research folder 2026-11-12-00-22. Handed to Gilbreth at 02:58 PT with jobs 12352864 to 12352869; three
  started within a minute. These units are not part of the RND baseline and its reports read only their own
  phases.
- 2026-11-13 05:28 PT: **the rollout-shape check on the U-maze is complete** (stable physics, 48 configurations and
  16 development seeds per environment count, 16,777,216 steps per policy). Best whole-curve mean per count:
  128 environments per policy 0.0487 ± 0.0014 (learning rate 1e-2, entropy 0.001, weight 16; final level 0.065;
  first-goal time 2.1 million steps); 16 environments 0.0487 ± 0.0027 (3e-4, 0.001, weight 0.0625; final level
  0.063; 1.3 million steps); 4 environments 0.0341 ± 0.0008 (3e-4, 0.001, weight 64; final level 0.036; 0.34
  million steps). Every policy reaches the goal at least once in all three. Fault-guard respawns in the six units
  of each count: 32, 43 and 16. **4 environments per policy is out; 128 and 16 tie** on the whole-curve mean and
  on the final level. Not decided yet: the longer run (phase `budget-probe`) compares the two at equal steps
  (67,108,864) when both units reach that point; until then ant screens run at both counts. Both best
  configurations sit at an end of the grid (the highest learning rate with 128, the lowest learning rate and
  weight with 16), so the edge rule applies before any sweep is called final.
- 2026-11-13 06:27 PT: four more units of the auto research folder run here under the phase `bonus-arms-screen`
  (its batch 14): the fast last layer at steps 0.03 and 0.01 with the first-arrival hold, with 128 and with 16
  environments per policy, the same eight configurations and seeds as the earlier screen units. The options exist
  in this folder's frozen copy (a2df3c12), so it was not frozen anew. Handed to Gilbreth at 06:27 PT with four
  jobs.
- 2026-11-13 06:31 PT: anomaly, harmless: about three minutes after the hand-over the 5-minute fleet step submitted
  four more jobs for the same four units (12353497, 12353499, 12353500, 12353501), although the first four jobs
  (12353491 to 12353494) were already running; they had started but not yet claimed their units when the step
  read the server, so the units still counted as having no job. All four units run on the first four jobs.
  12353497 started, found nothing and ended after 8 seconds; the other three wait and will do the same, or will
  continue the two screen units with 16 environments per policy if those reach their jobs' 4-hour limit at 06:58
  PT before finishing (they were at 240 and 243 of 256 windows at 06:26 PT). Not cancelled for that reason.
- 2026-11-13 07:08 PT: **a partial look at the longer run (phase `budget-probe`; records copied from Gilbreth at
  07:07 PT; 16 development seeds per configuration), and a concern about what the trained ants do.**
  1. At equal steps (the first 40 million per policy) the best configuration with 16 environments per policy has
     a whole-curve mean of 0.0779 ± 0.0035 (learning rate 1e-4, entropy 0.001, weight 0.0625; level over the last
     tenth 0.098) and the best with 128 has 0.0503 ± 0.0009 (1e-2, 0.001, weight 16; level 0.062). The 16-environment
     best sits at the lowest learning rate and the lowest weight of its grid, so the edge rule applies.
  2. **The fault guard fires far more often late in long runs, and mostly in the best configurations.** With 128
     environments per policy (158 windows, 83 million steps per policy): 54 respawns in the first half and 4,938
     in the second; 3,797 of them at learning rate 1e-2 with weight 64 (one per 0.3 million steps), 500 to 557 at
     weights 1 and 4, 4 to 6 in each configuration with learning rate 3e-2. With 16 environments per policy (154
     windows, 40 million steps): 790 in all, 724 of them in the best configuration (one per 0.9 million steps).
     In the 16.8-million-step check the six units of a count had 16 to 43.
  3. **Late in these runs the ants do not walk.** Over the last 20 windows, per configuration: 9 to 18% of all
     steps are in the zone above the walls, most in the best configurations (0.178 for the best with 16
     environments, 0.14 with 128); the largest torso speed seen is 14 to 26 m/s and the largest turning speed 63
     to 164 rad/s (the guard's limits are 30 m/s and 100 rad/s); an episode that arrives takes 510 to 545 of its
     700 steps; 3 to 10% of episodes arrive.
  This holds for RND and would hold for any method alike, but it may mean that the return is earned by motion a
  reference simulator would not allow. A check with late-training policies against reference MuJoCo was started
  at 07:08 PT (background worker; results to be added here).
- 2026-11-13 10:53 PT: **the motion check is done** (`diagnosis/2026-11-13_what-the-trained-ants-do_against-reference-mujoco/notes.md`;
  measured 08:46 to 10:35 PT by a background worker that stopped at a usage limit before writing its notes; the
  notes were written from its records at 10:52 PT). The hopping (about 80% of states with no contact, 33 to 50%
  above 1 m) and the arrival rates are reproduced by reference MuJoCo (RK4, 0.01 s, 100 solver iterations): the
  arrival rate under the reference settings is within −0.017 and +0.021 of ours in all 16 configurations of the
  two `budget-probe` checkpoints. The guard's runaways come from the implicitfast integrator at 4 solver
  iterations (in C MuJoCo too; none with RK4 or the implicit integrator at the same step), at most 9 of 1,024
  episodes, mostly with 128 environments per policy and learning rate 1e-2. The 07:08 PT caveat is withdrawn for
  the motion; a change of integrator would change the environment for every method and is left to the user.
- 2026-11-13 10:53 PT: no ticks from 07:26 to 10:47 PT (the session's model reached its usage limit); the 5-minute
  fleet loop ended at 08:48 PT after its 12-hour limit and was restarted at 10:53 PT. Nothing failed in the gap.
- 2026-11-13 11:29 PT: **AntMaze Medium and Large are ready for units** (background worker, 10:59 to 11:28 PT; notes
  `diagnosis/2026-11-13_medium-and-large-readiness/notes.md`, commits e54ab8e8 and b9c320e8; job ids in
  `slurm/submitted_jobids_medium-large-readiness_sl5nw.txt`, all ended). Both maps pass every environment gate
  against C MuJoCo at the stable physics (start, goal termination at 0.45 m, the 1,000-step limit as truncation,
  walls, ceiling, contact-free and in-contact steps, the fault guard); nothing map-specific was wrong, and the
  environment code is unchanged (one docstring count corrected). New configuration files
  `configs/experiments/antmaze_{medium,large}_warp-physics_time-step-0.005_frame-skip-10_fault-guard-30-metres-and-100-radians-per-second_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml`
  (126 and 30 tests pass). A unit costs the same as on the U-maze: 1.17 s per iteration for 128 policies of 16
  environments on an A100 40 GB (0.224 million steps per second); about 2.8 hours per 16.8 million steps per
  policy on an A100 and 3.6 on an A30. 128 policies of 16 environments need 7,000 MB of card memory (20,000 used
  so far is safe). No contact-pool or constraint-row overflow with untrained ants (at most 42 rows of 512; a
  trained U-maze unit once passed 160, so the first units' records are to be read). A study description for
  these maps must name the workload with the time step and give `policies_per_unit` and
  `minimum_card_memory_mb` as numbers. Noted for the user, not changed: the policy and value networks see the
  torso position unscaled, a range that grows from 10 m (U-maze) to 24 m (Large).
- 2026-11-13 11:31 PT: **the budget rule on the longer run (records of 11:30 PT; scratchpad script
  `ant_budget_rule.py`).** Define R(s) as the mean return of a configuration's 16 policies in the window containing
  step s, smoothed over 5 windows; L as its mean over the last 10% of the run; s95 as the first step where R(s)
  reaches 0.95 L. The plan's budget is B = 2 x max(s95 of the three configurations with the highest whole-curve
  mean), rounded up to a power of two.
  1. 16 environments per policy (65.5 of 67.1 million steps done): s95 = 35.9, 9.7 and 15.7 million for the best
     three, so B = 134,217,728 steps per policy. Best configuration: learning rate 1e-4, entropy 0.001, weight
     0.0625, whole-curve mean 0.0862, final level 0.103 (at the lowest learning rate and weight of its grid).
  2. 128 environments per policy (122.2 of 134.2 million done): two of the best three have not levelled off
     inside the run (s95 in its last tenth), so the rule gives at least 268,435,456. Best: learning rate 1e-2,
     weight 16, whole-curve mean 0.0722, final level 0.100.
  **Why this does not settle the environment count:** the two runs had different budgets (67 and 134 million
  steps) and the learning rate falls to zero over each run's own budget, so at 65 million steps the
  16-environment run has finished its schedule and the 128-environment run is halfway. At equal budgets (16.8
  million, the check) the two counts tie. The plan's rule compares at the same number of steps; the next RND sweep
  on the U-maze at B = 134 million with 16 environments per policy gives that comparison against the
  128-environment run of the same length, and its result decides the count. Until then, ant work uses 16
  environments per policy (the cheaper choice to correct: its sweep is needed for the comparison anyway).
- 2026-11-13 12:28 PT: **MIT and Gautschi `ai` are set up for ant units and left switched off** (background worker,
  12:00 to 12:27 PT; notes `infrastructure/h100-h200-servers_setup_notes.md`, commit 0457ba73). MIT: a MuJoCo Warp
  environment built on `mit_quicktest` (job 28985995) at `/orcd/scratch/orcd/008/shuzel/envs/rnd_platform_jax_mujoco_warp`,
  the same 79 packages as Gautschi's; checked on one Advanced H100 (job 28987093: the card guard found 1 card,
  jax and Warp see an H100 80 GB); entry `mit` with four pools (Advanced H100/H200 on `mit_normal_gpu`,
  preemptable H100/H200 leaving out the 79 nodes shared with `mit_normal_gpu`), 4 cores, 24 GB, card guard with
  requeue. Gautschi: the existing Warp environment (imports checked, job 18521082 on `cpu` standby); entry
  `gautschi`, pool `ai-h100` (`-p ai -q normal`, 14 cores, 2 jobs). Both `ready: false`. At setup time both
  servers' caps were full with another session's jobs (MIT: 4 of 4 preemptable H100s and 8 Advanced jobs waiting;
  Gautschi: both H100s until about 20:07 PT). Before a unit goes there: freeze this folder anew, set `first_unit`
  by hand (no UVA ant unit exists for the automatic check), `prepare`, then `ready: true`.
- 2026-11-13 12:28 PT: the setup worker found that both fleet scripts read the server table before taking the fleet
  lock, so a step that waited on the lock could write back an older table and drop entries added meanwhile.
  Fixed (the table is now read under the lock) with a test in each script; 144 tests pass (job 6909084).
- 2026-11-13 13:08 PT: **this folder's code was frozen anew at commit ef844a20, its first numbered code copy (copy
  number 1)**; it holds the RK4 configuration files (the user's decision of 2026-11-13: MuJoCo Warp with RK4 at
  0.01 s x 5), the Medium and Large files and the copy-number rule of the queue. No job of an older copy was
  alive (all 36 earlier units done).
- 2026-11-13 13:10 PT: **17 RK4 units built** (minimum copy number 1; interpreter `platform_jax_warp`), all at
  134,217,728 steps per policy, 16 development seeds, 256 windows:
  1. RND sweep on the U-maze, 16 environments per policy, 28 configurations per reward setting (learning rates
     3e-5, 1e-4, 3e-4, 1e-2 x weights 1/256 to 16 by factors of 4, entropy 0.001): 8 units of 112 policies (phase
     `rnd-sweep-rk4`);
  2. one RND unit with 128 environments per policy, reward 1 on arrival, learning rate 1e-2 x weights 1, 4, 16, 64
     (64 policies, phase `environment-count-check-rk4`), for the equal-steps rule;
  3. RND probes on Medium and Large, reward 1 on arrival, 8 configurations (128 policies each, phase
     `rnd-probe-rk4`);
  4. the auto research folder's batch 15: the method on the U-maze at steps 0, 0.03 and 0.1 with the first-arrival
     hold, 9 configurations each, both reward settings (6 units of 144 policies, phase `method-sweep-rk4`).
- 2026-11-13 13:11 PT: 13 units started on UVA cards free at once (6909227 and 6909228 on two RTX 5080 each,
  6909229 on an A40, 6909232 on eight RTX A4500 on jaguar03 through the reservation `sl5nw_168`); two jobs on RTX
  4000 Ada cards (6909230, 6909231) still waited after two minutes and were cancelled by their ledger ids. The
  other four units (the 128-environment unit and three method units with the step cost) went to Gilbreth at 13:13
  PT (jobs submitted by the fleet; the copy number 1 was sent there by `prepare`). MIT and Gautschi stay
  switched off until their caps have room. Measured speed on an A100 for 128 policies under RK4: 2.17 s per
  iteration, so one unit of about 112 to 144 policies needs about 1.5 to 2 days of card time and continues from
  its checkpoints across the 2-hour UVA jobs and 4-hour Gilbreth jobs. The fleet loop was restarted at 13:14 PT.
- 2026-11-13 13:51 PT: speeds of the RK4 units after 39 minutes (windows of 256; each window is 256 iterations): RTX
  5080 5 to 6 windows (about 6.5 minutes per window, about 28 hours per unit), RTX A4500 3 to 4 (10 to 12 minutes,
  about 45 hours), Gilbreth A10 2 windows after 20 minutes, the 128-environment unit on an A30 6. No fault-guard
  respawn in any unit so far. This folder's monitor job (6891245) starts no UVA workers, and the fleet loop would
  hand every unit whose 2-hour UVA job ends (about 15:11 PT) to Gilbreth, whose idle cards are mostly taken by
  other users; so the Gilbreth limit was pinned to the 4 units already there, and a one-time session step at 15:14
  PT starts new UVA workers on free cards and then hands what is left to Gilbreth.
- 2026-11-13 14:25 PT: **where an RK4 ant iteration spends its time** (background worker, serval05 H100 NVL, 16
  environments per policy; notes `benchmark_runs/2026-11-12-00-17_*/profiles/rk4_ant_iteration/notes.md`, commits
  7f32e6cd and 3a633b79): 1.855 s per iteration at 128 policies and 1.514 s at 32, so about 1.40 s is fixed and
  3.5 ms per policy; the bare physics is 98% of it (one RK4 physics step is 422 MuJoCo Warp kernel launches, about
  4 microseconds each), the update under 1%. Consequence: a unit's wall time hardly depends on its number of
  policies; at least about 1.4 s x 65,536 iterations, about 26 hours on this card, for 134 million steps per
  policy. No idea that keeps results identical reaches 5% (best: the solver loop as a plain loop, +4.3%); 2
  solver iterations would be 17% faster but change the physics (reported only). Two start-up options were added,
  off by default (`--put-state-on-card`, `--compilation-cache-dir`): a resumed unit starts in about 35 s
  instead of 68 s on the H100 (87 to 106 s for the first iteration on today's training nodes); worth about 1% of a
  2-hour UVA job, so not switched on for the running units. No A100 row: every card on cheetah01 and cheetah04 was
  taken by other users.
- 2026-11-13 15:10 PT: the four UVA worker jobs ended on their own after 1 h 50 min (they stop before the 2-hour
  limit); the 13 units went back to pending with their checkpoints. Five new UVA jobs on cards free at once took
  all 13 within two minutes (6909545 and 6909546 on two RTX 5080 each, 6909547 on an A40, 6909548 on an RTX 4000
  Ada, 6909549 on seven RTX A4500 on jaguar03 through the reservation). The one-time step planned for 15:14 PT
  was not needed and was removed. This restart is due every 1 h 50 min; each monitoring tick submits workers on
  free cards for any unit left pending.
- 2026-11-13 17:10 PT: second UVA restart: the five UVA jobs ended after 1 h 50 min; eight new jobs on free cards took
  all 13 units at once (6909963 to 6909970: four RTX 5080, one A40, six RTX 4000 Ada, two RTX A4500). On Gilbreth the
  four jobs of 13:13 PT reached their 4-hour limit; one unit continues in job 12361511 (A10), and three wait for
  jobs 12361584 to 12361586 (waiting on priority at 17:11 PT). If they still wait at about 19:10 PT they go to the
  next place (UVA free cards or MIT).
- 2026-11-13 19:10 PT: third UVA restart: the eight UVA jobs ended after 1 h 50 min; seven new jobs on free cards
  (6910417 to 6910423) started at once for the 13 units (the tick's script `ant_tick.sh` of the session's scratchpad
  does the restart, cancels by ledger id what still waits after two minutes, and hands what is left to Gilbreth).
- 2026-11-13 21:10 PT: fourth UVA restart: five new jobs on free cards (6910745 to 6910749: four RTX 5080, nine RTX
  4000 Ada) took the 13 units at once. On Gilbreth the four jobs of about 17:05 to 17:20 PT ended at their limit
  (20:57 and 21:11 PT, accounting times converted from Eastern); job 12365054 waits for the first of the four units
  (planned start 21:42 PT), and the fleet loop submits the other three. Jobs 12361859, 12361867 and 12361868 of
  17:21 PT found no unit and ended in 8 seconds, as generic workers do.
- 2026-11-13 23:10 PT: fifth UVA restart: five new jobs on free cards (6913986 to 6913990: four RTX 5080, one A40,
  eight RTX 4000 Ada) took the 13 units at once. The four Gilbreth units run in jobs 12365054, 12365300, 12365309
  and 12365400 (A10 cards).
- 2026-11-14 01:10 PT: sixth UVA restart: seven new jobs on free cards (6917566 to 6917572) took the 13 units at
  once.
- 2026-11-14 03:10 PT: seventh UVA restart: six new jobs on free cards (6919262 to 6919267) took the 13 units at once.
  The four Gilbreth units run in jobs 12368970, 12369082, 12369288 and 12369289, started 01:25, 01:31, 01:49 and 01:51 PT (accounting times converted from Eastern).
- 2026-11-14 05:10 PT: the 128-environment RK4 unit is complete (256 of 256 windows, finished 04:56 PT on Gilbreth,
  records copied back and verified; 0 fault-guard respawns over 134,217,728 steps per policy). Eighth UVA restart:
  six jobs submitted on free cards (6919311 to 6919316); 6919311 still waited after two minutes and was cancelled by
  its ledger id; the other five (four RTX 5080, four A40, two RTX 4000 Ada cards) took 10 units. The other 6 units
  (all six method units of batch 15) are on Gilbreth: the fleet loop handed one over when the 128-environment unit
  freed its place, and the tick handed two more with the limit raised to 6; two new Gilbreth jobs were submitted.
- 2026-11-14 07:10 PT: ninth UVA restart: four new jobs on free cards (6919414 to 6919417: three A100 40 GB, four RTX
  5080, three A40) took the 10 UVA units at once. The six method units stay on Gilbreth.
- 2026-11-14 08:53 PT: **the faster cards go to the units with the most windows left.** Windows per hour over each
  unit's last eight windows (read 08:50 PT): RTX 5080 11.2 to 12.4, A100 40 GB 6.7, A40 7.1 to 7.2; on Gilbreth
  (A10, A30) about 6. The three step-cost method units on Gilbreth had 150 to 156 windows left, about 25 hours there,
  against 13 hours on an RTX 5080; the UVA units have at most about 120 left. Steps:
  1. Gilbreth marked not ready for this folder (08:53 PT, reason in the table), so no new Gilbreth job starts; its
     six jobs end at their 4-hour limit between 09:05 and 09:40 PT (no waiting Gilbreth job of this folder);
  2. each method unit is taken back to UVA when its job has ended (`remote_fleet.py take-back`), with its
     checkpoint;
  3. before each UVA restart, every waiting unit except the four with the most windows left (the three step-cost
     method units and the reward-on-arrival unit with the first-arrival hold alone) gets a card-memory floor of
     17,000 MiB, which keeps it off the RTX 5080 cards (16,303 MiB); the floor changes only where a unit may run,
     not what it computes (session scratchpad script `steer.py`);
  4. Gilbreth is set ready again once the six are back, for units no free UVA card can take.
- 2026-11-14 09:10 PT: three method units (reward on arrival) were taken back from Gilbreth after their jobs ended,
  each with its checkpoint and record file. The 17,000 MiB floor was set on every waiting unit except the reward-on-
  arrival unit with the first-arrival hold alone (floor 14,000) and the three step-cost method units (still on
  Gilbreth). Tenth UVA restart: jobs 6919817 to 6919821 (three A100 40 GB, six A40, four RTX 5080); the unit with
  the first-arrival hold alone took an RTX 5080 (job 6919818). The free-card planner counts cards, not which unit
  fits which card, so three RTX 5080 cards found no unit their floor allowed: job 6919819 (nekomata02) ended after 4
  seconds, and the second card of 6919818 stays idle until that job ends. Three units were left waiting; workers
  on three free RTX 4000 Ada cards (6919827, 6919828, planner run with the RTX 5080 left out) took them at 09:12 PT.
  All 13 UVA units run; the three step-cost method units come back from Gilbreth after 09:30 PT and get RTX 5080
  workers (planner run with that card type only).
- 2026-11-14 09:31 PT: two step-cost method units (fast step 0.03 with the hold, and the first-arrival hold alone)
  were taken back from Gilbreth with their checkpoints and took both RTX 5080 cards of nekomata02 (job 6919852,
  planner run with that card type only). The third (fast step 0.1 with the hold) comes back after its Gilbreth job
  ends at 09:40 PT. The tick's restart step now submits in two passes: RTX 5080 workers first, for units whose floor
  allows that card, then the other card types.
- 2026-11-14 09:50 PT: the last step-cost method unit (fast step 0.1 with the hold) was taken back from Gilbreth with
  its checkpoint. No RTX 5080 was free and the one job submitted for another card type (6919962) still waited after
  110 seconds, so it was cancelled by its ledger id. The unit waits on UVA for an RTX 5080 (the 11:10 PT restart
  frees two on nekomata01) rather than going back to Gilbreth: on Gilbreth it would hold an A10 at about 6 windows
  per hour until about 14:00 PT. Gilbreth stays not ready for this folder until that restart; each tick still
  tries free UVA cards for it, RTX 5080 first.
- 2026-11-14 09:50 PT: **first fault-guard respawn under the RK4 physics**: RND sweep, reward on arrival, cells 1 to 7,
  window 209 (about 110 million steps per policy), one policy (learning rate 3e-5, intrinsic weight 1/256, seed
  index 2), torso angular speed 106.2 rad/s against the guard's 100 (torso speed 14.0 m/s). Over the unit's 211
  windows the largest torso speed of any policy is 17.8 m/s; this is the guard's angular bound catching a fast spin,
  not a runaway of the kind the implicitfast physics produced (up to 2.45 x 10^7 m/s).
- 2026-11-14 10:10 PT: the waiting step-cost unit (fast step 0.1 with the hold) got an RTX 4000 Ada card (job 6920059,
  lynx01). Windows per hour read 10:12 PT: RTX 5080 10.7 to 11.4, A40 6.7 to 7.1, A100 40 GB 6.7 to 6.8, RTX 4000
  Ada 5.7 to 7.4. At 5.7 per hour its 147 windows would take about 26 hours, so at the 11:10 PT restart (two RTX 5080
  cards of nekomata01 free) its job is to be cancelled by its ledger id and the unit restarted on an RTX 5080 from its
  checkpoint. The restart step's RTX 5080 pass now asks for only as many cards as there are waiting units allowed
  on them (`--max-new-cards`), so no RTX 5080 job starts without a unit.
- 2026-11-14 11:14 PT: eleventh UVA restart. Job 6920059 (the step-cost unit with fast step 0.1 on an RTX 4000 Ada)
  was cancelled by its ledger id at 11:10 PT; its worker gave the unit back within 10 seconds. The restart's RTX 5080
  pass asked for 2 cards and job 6922507 (nekomata01) took the two units allowed there, so all four units with the
  most windows left now run on RTX 5080 cards (6922507 and 6919852). The other pass placed the eight RND sweep units
  (6922509 to 6922511: three A100 40 GB, five A40); jobs 6922512 and 6922513 still waited after 110 seconds and were
  cancelled by their ledger ids. Gilbreth was set ready again for this folder and took the four units no free UVA
  card could take (two reward-on-arrival method units and the two RND probes); four Gilbreth jobs were submitted, and
  the session's 5-minute fleet loop was started again.
- 2026-11-14 11:55 PT: six edge-extension units of batch 15 (phase `method-edge-rk4`, auto research folder's batch 15
  `ideas.md`; claim order 63 to 68, after the six method units, so those keep the RTX 5080 cards) were built and
  placed: four on free UVA cards (jobs 6922949 and 6922950; 6922951 still waited after 110 seconds and was
  cancelled by its ledger id), two on Gilbreth (two jobs submitted). They move to RTX 5080 cards when the current
  units finish, about 01:00 to 02:00 PT on 15 November.
- 2026-11-14 13:11 PT: twelfth UVA restart, with the RTX 5080 pass first: job 6923400 (nekomata01, two RTX 5080)
  took the two waiting units allowed there (reward on arrival with the first-arrival hold alone; step cost with
  fast step 0.1 and the hold); jobs 6923407 to 6923409 (three A100 40 GB, two A40, three RTX 4000 Ada) took the eight
  RND sweep units. Nothing waited.
- 2026-11-14 14:10 PT: second fault-guard respawn under RK4: RND sweep, reward on arrival, cells 8 to 14, window 229
  (about 120 million steps per policy), one policy (learning rate 1e-4, weight 1/64), torso angular speed 100.9
  rad/s against the guard's 100 (torso speed 15.7 m/s). Like the first (09:50 PT entry), a fast spin just over the
  angular bound late in training, not a runaway. Edge-extension units on UVA restarted at 13:49 PT after their
  2-hour jobs (6923647 to 6923649: two A40, two RTX 4000 Ada).
- 2026-11-14 15:10 PT: thirteenth UVA restart (jobs 6926066 to 6926070: one A100 40 GB, two A40, seven RTX 4000 Ada;
  nothing waited). The RTX 5080 pass found no card: nekomata01's two RTX 5080 are free, but 22 of its 24 processors
  are held by another user's processor-only array jobs (2-day limits) that started after our job 6923400 ended at
  15:00 PT, and the last 2 are kept for the system, so a card job cannot start there. The reward-on-arrival unit
  with the first-arrival hold alone went to an RTX 4000 Ada, the step-cost unit with step 0.1 to an A100 40 GB
  (both about 6.5 to 7 windows per hour). nekomata02's two RTX 5080 stay with job 6923539 until about 15:30 PT;
  the same may happen there. A 2-hour job cannot be lengthened by its user, so the cards are kept only while the
  other user's jobs leave processors free.
- 2026-11-14 16:50 PT: the first RK4 sweep unit of 16 environments per policy is complete (RND, reward on arrival,
  cells 1 to 7; finished 16:36 PT; 256 windows, 134,217,728 steps per policy; 1 fault-guard respawn over the unit).
  Third respawn under RK4: RND, step cost, cells 1 to 7, window 187, one policy (learning rate 3e-5, weight 1/64),
  torso angular speed 115.0 rad/s (torso speed 15.7 m/s). All three respawns so far are in RND units and all three
  are the angular bound (100 rad/s), none the speed bound.
- 2026-11-14 17:10 to 17:32 PT: the RTX 5080 cards are no longer reachable: on nekomata01 and nekomata02 another
  user's processor-only jobs hold 22 of the 24 processors whenever one of our card jobs ends. The four method units
  that had used them went to Gilbreth (17:12 and 17:32 PT; one UVA job each time still waited after 110 seconds and
  was cancelled by its ledger id: 6929169, 6929607). Gilbreth now holds 10 units (the six method units, two edge-
  extension units, the two probes), UVA 11 (seven RND sweep units, four edge-extension units). The method units have
  69 to 93 windows left, about 12 to 16 hours on Gilbreth; the method's validation block waits for the edge-extension
  units in any case.
- 2026-11-14 19:52 PT: Gilbreth slowed: the four jobs submitted 19:13 to 19:34 PT for units between jobs (two
  reward-on-arrival method units with the fast last layer, the two probes) had planned starts 01:30 to 02:10 PT. With
  the Gilbreth limit lowered first to its six running units, the four were taken back to UVA with their checkpoints
  and started at once on free RTX 4000 Ada cards (jobs 6930386, 6930387). The four waiting Gilbreth jobs
  (12385466, 12385467, 12385727, 12385809) are generic workers; one that starts with no unit there ends in seconds.
  UVA now holds 14 units, Gilbreth 6.
- 2026-11-14 20:10 PT: the two step-cost edge-extension units on Gilbreth were between jobs after their 4-hour limit
  (Gilbreth's planned starts still about 01:30 PT); with the limit lowered to the four units still running there,
  both were taken back with their checkpoints and started at once on two RTX 4000 Ada cards (job 6930614). UVA 16
  units, Gilbreth 4 (the four step-cost and reward-on-arrival method units still in their jobs).
- 2026-11-14 21:13 PT: fourteenth UVA restart (the six RND sweep units: jobs 6931258 to 6931260, A100 40 GB and RTX
  4000 Ada). Two Gilbreth units between jobs (reward on arrival with the first-arrival hold alone; step cost with
  fast step 0.1) were taken back with the limit lowered first, and the RTX 5080 pass found two cards free again
  (jobs 6931264, 6931265). UVA 18 units, Gilbreth 2 (the step-cost units with step 0.03 and with the hold alone,
  in jobs that end about 21:30 PT).
- 2026-11-14 21:30 PT: the last two Gilbreth units (step cost, step 0.03 with the hold, and the hold alone) were taken
  back after their jobs ended (limit set to 0 first) and run on two RTX 4000 Ada cards (job 6931503). All 20 units
  are on UVA; nothing is on a remote server, so the session's 5-minute fleet loop ended by itself. The tick's restart
  step still hands any unit no free UVA card takes to Gilbreth (it raises the limit and runs the fleet's step itself);
  the fleet loop is started again when a unit is there.
- 2026-11-14 23:11 PT: fifteenth UVA restart: the RTX 5080 pass kept both cards for the two units allowed there (job
  6932337); the six RND sweep units restarted on A100 40 GB, A40 and RTX 4000 Ada cards (6932357, 6932358,
  6932368). All 20 units on UVA, nothing waiting.
- 2026-11-14 23:45 PT: notices from two other sessions of this account. (1) All remote connections are open: MIT on
  portal11, DeltaAI on portal04 (closes 2026-11-21 20:37 PT), ACES on portal05 (certificate expired, the open
  connection still answers), Purdue on portal04; at 23:40 PT the account held 4 running and 21 waiting jobs on MIT
  `mit_normal_gpu` and 4 running and 15 waiting on `mit_preemptable`. (2) The tinker talk-revision study uses the
  account's MIT H200 caps from 2026-11-15 PT and asked that no new MIT GPU jobs compete for the 4-card caps per pool.
  This folder's MIT entry stays not ready (it was never prepared); ant units do not go to MIT while that study holds
  the caps. Purdue and the ACCESS servers are not affected.
- 2026-11-15 00:52 PT: the RTX 5080 cards now go to the edge-extension units: they have about 170 windows left each
  (about 27 hours at RTX 4000 Ada speed, and the method's validation block waits for them), against at most 43 for
  the method sweep units and 16 for the RND sweep units. Before each restart the tick gives every waiting unit except
  the edge-extension units the 17,000 MiB floor, so only those may take an RTX 5080.
- 2026-11-15 02:10 PT: five of the eight RND sweep units are complete (the latest: reward on arrival, cells 15 to 21).
  Its worker (job 6935087, A40) then took a waiting edge-extension unit (step cost, step 0.03 with the hold), as
  workers do when a unit ends, so job 6935456 found nothing and ended at once. Fourth fault-guard respawn under RK4:
  RND, step cost, cells 1 to 7, window 249, one policy (learning rate 3e-5, weight 1/64), torso angular speed 112.0
  rad/s (torso speed 14.5 m/s). All four respawns are RND units and the angular bound; two are in this unit, both
  at learning rate 3e-5 and weight 1/64.
- 2026-11-15 03:14 PT: **first ant validation block, RND with the step cost, started**: four units (phase
  `validation-rk4`, one configuration each, seeds 10000 to 10127, 134,217,728 steps per policy; description
  `infrastructure/descriptions/antmaze_umaze_rk4-physics_validation_rnd_reward-step-cost_seeds-10000-to-10127_budget-134217728.json`)
  on free UVA cards (jobs 6936013 on an A40, 6936014 on three RTX 4000 Ada). Unit ids drop the environment count
  (every validation unit has 16 per policy) to stay under 200 characters. The step-cost RND sweep was complete; the
  reward-on-arrival half follows when its last RND unit (cells 22 to 28) ends.
- 2026-11-15 03:51 PT: **the RK4 RND sweep is complete** (all eight units, both reward settings). RND's reward-on-
  arrival validation block started: four units (seeds 10000 to 10127; description
  `infrastructure/descriptions/antmaze_umaze_rk4-physics_validation_rnd_reward-arrival-plus-one_seeds-10000-to-10127_budget-134217728.json`)
  on RTX 4000 Ada cards (jobs 6936193, 6936196), with seven restarted units (6936191, 6936194, 6936195). All 20
  running units are on UVA: 8 RND validation, 6 edge-extension, 4 method sweep, 2 probes; nothing waiting.
- 2026-11-15 05:12 PT: the RND probes on Medium and Large are complete (50 units done). At the restart six units
  were waiting; free UVA cards took three (6936491; 6936492 still waited after 110 seconds and was cancelled by its
  ledger id), and three RND step-cost validation units went to Gilbreth (jobs 12388381 to 12388383, waiting on
  priority with no planned start yet). The session's fleet loop was started again. If they still wait at about 07:10
  PT, they come back to free UVA cards. The four waiting Gilbreth workers of 19:13 to 19:34 PT started about 01:30 to
  02:10 PT, found no unit and ended.
- 2026-11-15 07:30 PT: fifth fault-guard respawn under RK4, the first in a method unit: edge extension, step cost,
  first-arrival hold alone, window 132, one policy (learning rate 3e-5, weight 1/64), torso angular speed 102.7
  rad/s (torso speed 15.0 m/s). All five respawns are the angular bound; four of the five are at learning rate 3e-5
  with weight 1/64 (RND's best configuration in both reward settings, and here the method at the same setting).
  At 07:11 PT the RTX 5080 rule gave its first card to an edge-extension unit (job 6937290).

- 2026-11-15 13:07 PT: the user decided the comparison on Medium and Large (plan "Change 4"): 8 validation units
  (seeds 10000 to 10127, reward 1 on arrival; RND and fast last layer step 0.03 with the hold, learning rate 1e-2,
  weights 1 and 16) built and started on free UVA cards: jobs 6937965 (A40), 6937966, 6937967, 6937968 (RTX 4000
  Ada), two units per job. The same restart put two edge-extension units on nekomata01's RTX 5080 cards (job
  6937960) and one U-maze RND validation unit on an A100 (job 6937964). Nothing waits; nothing on Gilbreth beyond
  its 4 units.

- 2026-11-15 13:10 PT: sixth fault-guard respawn under RK4, the second in a method unit: edge extension, reward 1 on
  arrival, first-arrival hold alone, window 181, one policy (learning rate 3e-5, weight 1/4), torso angular speed
  101.0 rad/s (torso speed 15.0 m/s). All six respawns under RK4 are the angular bound (100.9 to 115.0 rad/s; torso
  speed 14.0 to 15.7 m/s, under its 30 m/s bound), read from the records at 13:15 PT: five at learning rate 3e-5
  (weights 1/256, 1/64 three times, 1/4) and one at 1e-4 (weight 1/64); four in RND units, two in units of the hold
  alone, none with the fast last layer. Correction to the 07:30 entry: three of its five, not four, were at learning
  rate 3e-5 with weight 1/64.

- 2026-11-15 19:56 PT: no monitoring tick ran from 17:30 to 19:55 PT, and the UVA workers that ended in that time
  were not restarted: jobs 6938288, 6938290, 6938291 ended at 17:40 to 17:41 PT, 6938325 at 18:00 PT, and 6938468
  to 6938474 at 19:00 to 19:02 PT (sacct). All 18 UVA units then waited until 19:56 PT, so they lost 55 minutes
  to 2 hours 16 minutes each. Among them were the three reward-on-arrival edge-extension units with the fast last
  layer or the hold alone (stopped at 218, 219 and 245 windows) and the slowest unit (step cost, step 0.1, at 181).
  The 5-minute fleet loop kept running the whole time; Gilbreth's four units were not affected. All 18 restarted
  at 19:56 PT in seven jobs (6938639 to 6938645), four of them on RTX 5080 cards. From 19:58 PT the UVA restart step
  (the session's `ant_tick.sh`) also runs in a background loop every 10 minutes (`ant_restart_loop.sh`, at most
  12 hours, guarded so that it and a session tick never run at once), so a gap in the ticks no longer leaves units
  waiting.

- 2026-11-15 20:21 PT (window record): seventh fault-guard respawn under RK4, the first in a unit with the fast last
  layer: edge extension, reward 1 on arrival, step 0.1 with the hold, window 223, one policy (learning rate 3e-5,
  weight 1/64), torso angular speed 101.3 rad/s (torso speed 16.1 m/s). All seven are the angular bound; six of the
  seven are at learning rate 3e-5, four of those at weight 1/64.

- 2026-11-15 23:05 PT (window record): eighth fault-guard respawn under RK4: edge extension, step cost, the hold
  alone, window 233, one policy (learning rate 3e-5, weight 1/256), torso angular speed 101.5 rad/s (torso speed
  14.6 m/s). All eight are the angular bound; seven of the eight are at learning rate 3e-5.

- 2026-11-15 23:35 PT: the four method validation units with reward 1 on arrival were built at 23:31 PT. From this
  tick `ant_tick.sh` also leaves the method's validation units at the 14,000 MiB card-memory floor, so they may run
  on the RTX 5080 cards (the edge-extension units still claim first, as they were built first). Placement: job
  6938870 on nekomata01's two RTX 5080 cards (step 0.03 with the hold at weight 1/64, and the hold alone), job
  6938873 on an RTX 4000 Ada (step 0.03 at weight 1/16), and the unit at learning rate 1e-2 to Gilbreth (limit 5).

- 2026-11-16 00:00 PT (window record): ninth fault-guard respawn under RK4, the first in a validation unit: RND with
  the step cost at learning rate 3e-5, weight 1/64 (RND's chosen configuration), window 119, seed 10047, torso
  angular speed 100.5 rad/s (torso speed 14.5 m/s). All nine are the angular bound; eight of the nine are at
  learning rate 3e-5.

- 2026-11-16 05:52 PT: the session's monitoring cron moved from every 20 minutes to hourly (job 180c6ba9, minute 7;
  it replaces f476702d and expires about 2026-11-23 05:50 PT): stable since 04:51 PT, all 24 ant validation units
  running, restarts done by the background loop `ant_restart_loop.sh` every 10 minutes. Back to every 20 minutes
  when a block completes, a unit fails, or work is submitted.

- 2026-11-16 07:15 PT (Gilbreth's squeue): three RND U-maze validation units wait on Gilbreth after their jobs
  12395714, 12395715 and 12395730 reached the 4-hour limit. Their new jobs 12396350, 12396351 and 12396352 were
  submitted at 06:50 PT and wait on `Priority`; Gilbreth's estimate is 16:00 to 16:24 PT. The units are reward 1
  on arrival at learning rate 1e-2, weight 4 (153 windows of 256) and the step cost at 1e-4, weight 1/64 and at
  1e-2, weight 16 (147 and 145 windows). If they still wait at about 08:50 PT (2 hours), they are also placed on
  the next server. The other two Gilbreth units run on A10 cards (jobs 12395886 and 12396119). On UVA, three units
  whose 2-hour workers ended got new workers at 06:44 PT (jobs 6940583 to 6940586) and at 07:12 PT (jobs 6940672
  and 6940673, RTX 5080 cards). The fault guard still counts nine respawns.

- 2026-11-16 08:10 PT (Gilbreth's squeue and the fleet loop's log): the three waiting RND U-maze units started on
  A10 cards at 07:56:14 PT (jobs 12396350 to 12396352), after 66 minutes in the queue. At 07:57:17 PT the fleet
  loop counted the same three units as pending with no waiting job, because the new workers had not claimed them
  yet, and sent three more jobs (12397361 to 12397363, waiting on `Priority`; Gilbreth's estimate 20:10 to
  20:26 PT). A waiting job holds no unit, so no unit has two jobs. These jobs will take the units whose jobs reach
  the 4-hour limit next (12395886 at about 08:22 PT, 12396119 at about 09:44 PT, by their start times); one that
  starts with no unit left exits. They are not cancelled.

- 2026-11-16 10:05 PT (window record): tenth fault-guard respawn under RK4, the second in a validation unit and the
  first in a method validation unit: the first-arrival hold alone with the step cost at learning rate 1e-4, weight
  1/16 (the method's choice ranked first by the whole-curve mean), window 56, seed 10038, torso angular speed
  105.8 rad/s (torso speed 15.4 m/s). All ten are the angular bound; eight of the ten are at learning rate 3e-5.

- 2026-11-16 12:20 PT (Gilbreth's sacct and squeue, the take-back log, the worker logs): the 4-hour jobs 12396350
  to 12396352 of the three RND U-maze validation units ended at 11:46:57 PT, and the fleet loop sent new jobs
  12400875 to 12400877 at 11:47 PT, waiting on `Priority`. Five more units (Medium and Large: RND at learning rate
  1e-2, weight 16 on Large, and the method's four units) were handed to Gilbreth at 11:23 PT when no UVA card was
  free; their jobs 12400318 to 12400322 also wait on `Priority`. Gautschi `ai` cannot start the work: the account's
  four H100 cards are held by another session's jobs, with 16 more of them waiting (`AssocGrpGRES`); MIT stays
  left to the tinker study. At 12:16 PT three RTX 5080 cards were free on UVA (two on nekomata01, one on
  nekomata02), so the three RND U-maze units, which decide when RND's fresh seeds start, were taken back (the
  Gilbreth limit lowered to 7 first; records and checkpoints copied back: 177, 171 and 168 windows), given the
  14,000 MiB floor, and resumed from their checkpoints on the RTX 5080 cards at 12:19 PT (jobs 6941558 with two
  cards and 6941559). From now on ant_tick.sh keeps every U-maze unit at its lower floor; only Medium and Large
  units get the 17,000 MiB floor that keeps them off the RTX 5080 cards. Eight Gilbreth jobs now wait for five
  units; the three without a unit exit within seconds when they start, as 12397363 did, and are not cancelled.
- 2026-11-16 12:16 PT (window records copied back from Gilbreth): eleventh fault-guard respawn under RK4, the third
  in a validation unit: RND with the step cost at learning rate 3e-5, weight 1/16, window 152, seed 10079, torso
  angular speed 101.7 rad/s (torso speed 14.3 m/s). It happened on Gilbreth and shows only now because that
  unit's records are copied back when it leaves the server. All eleven are the angular bound; nine of the eleven
  are at learning rate 3e-5.
- 2026-11-16 12:32 PT (sacct, worker logs, the fleet log): the UVA worker 6941442 (four RTX A4500 cards on jaguar03)
  reached its 2-hour limit at 12:27:14 PT; its four units, the method's step-cost validation units (47 to 72
  windows of 256, the longest chain left), restarted at 12:29 PT on four RTX A4500 cards on jaguar03 (job
  6941633). The Gilbreth job 12397361 of the method's U-maze unit with reward 1 on arrival (fast step 0.1 with the
  hold, learning rate 1e-2, weight 16; 68 windows) reached its 4-hour limit at 12:25:28 PT. It was taken back at
  12:31 PT to start on lynx02, whose four RTX 4000 Ada cards are free; the planner sent no worker, because the node
  has only 2.1 GB of memory actually free. Handed back to Gilbreth at 12:32 PT, where 7 jobs wait for 5 waiting
  units. Gilbreth job 12400318 started at 12:23:58 PT (the RND Large unit, 144 windows). Measured peak memory of
  finished workers: at most 3.8 GiB per card (sacct MaxRSS), against the 8 GiB per card they ask for.
- 2026-11-16 12:53 PT (worker logs, sacct): one A100 40 GB card was free on cheetah01 at 12:49 PT, so the method's
  U-maze unit with reward 1 on arrival (fast step 0.1 with the hold, learning rate 1e-2, weight 16; 68 windows),
  waiting on Gilbreth since 12:32 PT, was taken back (Gilbreth limit lowered to 6 first). At the same moment the
  UVA worker 6941476 (three RTX A4500 cards on jaguar03) reached its 2-hour limit (12:49:16 PT). The four units
  waiting then started at 12:49 PT on the A100 (job 6941775) and at 12:51 PT on three RTX A4500 cards on jaguar03
  (job 6941794). UVA 18 units, Gilbreth 6 (two running on A10 cards; the method's two Medium and two Large units
  wait there since 11:23 PT, with 7 waiting jobs).
- 2026-11-16 13:14 PT (worker logs, squeue): one A40 card was free on jaguar01 at 13:10 PT, so the method's Large
  unit at weight 1 (fast step 0.03 with the hold, learning rate 1e-2), waiting on Gilbreth since 11:23 PT, was
  taken back (Gilbreth limit lowered to 5 first) and started there (job 6941869). At the same moment the four UVA
  workers started at 11:20 PT reached their 2-hour limit; their seven units restarted at 13:12 PT (jobs 6941900 to
  6941903: two A100 40 GB cards on cheetah01, two RTX 4000 Ada on cheetah02, two RTX 4000 Ada on lynx01, one RTX
  A4500 on jaguar03). UVA 19 units, Gilbreth 5 (two RND units running on A10 cards; the method's Large unit at
  weight 16 and its two Medium units wait, with 7 waiting jobs).
- 2026-11-16 13:28 PT (Gilbreth's and Gautschi's squeue): the method's Large unit at weight 16 and its two Medium
  units have waited on Gilbreth for two hours (since 11:23 PT; 7 of our jobs wait there on `Priority`). No next
  server can start them: Gautschi `ai` still holds the account's four H100 cards for another session, with 15 of
  its jobs waiting; MIT stays left to the tinker study; no ACCESS server is prepared for ant units. They stay on
  Gilbreth and are taken back to UVA cards as those come free (only lynx02's four RTX 4000 Ada cards are free now,
  and that machine has too little memory left for a worker).
- 2026-11-16 12:59 PT (window record written then; read at 13:48 PT): twelfth and thirteenth fault-guard respawns
  under RK4, both in one window (203) of the RND step-cost validation unit at learning rate 3e-5, weight 1/64 (on
  UVA): seed 10086, torso angular speed 101.3 rad/s (torso speed 14.9 m/s), and seed 10105, 113.8 rad/s
  (15.7 m/s). This unit now holds three of the thirteen (its first was window 119, seed 10047). All thirteen are
  the angular bound; eleven of the thirteen are at learning rate 3e-5.
- 2026-11-16 14:12 PT (sacct on both servers, worker logs): the RTX 5080 workers 6941558 and 6941559 reached their
  2-hour limit at about 14:07 PT; their three RND U-maze units restarted at 14:10 PT on the same three RTX 5080
  cards (jobs 6942407 with two cards, 6942408). The Gilbreth job 12397362 of the fourth RND U-maze unit on that
  server (step cost, learning rate 1e-4, weight 1/64; 187 windows) reached its 4-hour limit at 13:50 PT; the unit
  was taken back at 14:12 PT (Gilbreth limit lowered to 4 first) and resumed on an RTX 4000 Ada card on lynx02 (job
  6942460). All eight RND U-maze units now run on UVA. Gilbreth holds 4 units: the RND Large unit running on an
  A10 (job 12400318), and the method's Large unit at weight 16 and its two Medium units waiting, with 7 waiting
  jobs.
- 2026-11-16 14:29 PT (worker logs, squeue): the method's four step-cost U-maze units restarted at 14:24 PT after
  their 2-hour worker on jaguar03 ended (jobs 6942583 to 6942585: an A40 on jaguar06, RTX 4000 Ada cards on
  cheetah02 and lynx02). At 14:29 PT four RTX A4500 cards were free on jaguar03 and one RTX 4000 Ada on lynx02, so
  the method's three units waiting on Gilbreth since 11:23 PT (Large at weight 16, Medium at weights 1 and 16) were
  taken back (Gilbreth limit lowered to 1 first) and started at once (jobs 6942602 on lynx02 and 6942603 with two
  RTX A4500 cards). UVA 23 units; Gilbreth 1 (the RND Large unit, running on an A10, job 12400318). The 7 Gilbreth
  jobs still waiting hold no unit; they take units handed over later, or exit within seconds when they start.
- 2026-11-16 13:57 PT (window record written then; read at 14:48 PT): fourteenth fault-guard respawn under RK4: RND
  with the step cost at learning rate 3e-5, weight 1/16 (the unit of the eleventh), window 188, seed 10056, torso
  angular speed 102.3 rad/s (torso speed 14.8 m/s). All fourteen are the angular bound; twelve of the fourteen are
  at learning rate 3e-5. At 14:46 PT the restart loop restarted the four units of the workers started at 12:49 and
  12:51 PT (jobs 6942622 and 6942623); UVA 23 units, Gilbreth 1, nothing waiting.
- 2026-11-16 15:10 PT (Gilbreth's sacct, the restart loop's log): five of the seven Gilbreth jobs that held no unit
  started between 15:04:57 and 15:09:01 PT and exited after 8 seconds each, as expected (12400319 to 12400322 and
  12400875); 12400876 and 12400877 still wait. At 15:08 PT the restart loop restarted the units of the workers
  started at 13:10 and 13:12 PT (jobs 6942661 to 6942665). UVA 23 units, Gilbreth 1 (the RND Large unit, 160
  windows), nothing waiting.
- 2026-11-16 15:50 PT: monitoring moves back to an hourly tick (no change since 14:29 PT apart from the restart
  loop's routine 2-hour restarts). It returns to every 20 minutes when a unit fails, work is submitted, or the
  slowest RND U-maze unit (step cost, learning rate 1e-4, weight 1/64; 198 of 256 windows at 15:48 PT) nears 256,
  when RND's fresh seeds are built and placed.
- 2026-11-16 16:45 PT (the restart loop's log, Gilbreth's sacct and squeue): the slowest RND U-maze unit (step cost,
  learning rate 1e-4, weight 1/64; 200 windows) went to Gilbreth at 16:03 PT, when its 2-hour worker on lynx02 ended
  and no UVA card was free; Gilbreth job 12403171 waits for it (`Priority`; Gilbreth's estimate 18:53 PT). The RND
  Large unit's job 12400318 reached its 4-hour limit at about 16:14 PT; job 12403224 waits for it (estimate
  22:34 PT). No UVA card is free now. From now on the session's restart step (ant_tick.sh, every 10 minutes) takes
  U-maze units waiting on Gilbreth back whenever the UVA planner finds cards free for them, after the units already
  waiting on UVA (a new helper reads the scheduler as submit.py does and submits nothing; the Gilbreth limit is
  lowered first). At 16:24 PT the restart loop restarted five UVA workers (jobs 6942766 to 6942770). UVA 22 units,
  Gilbreth 2 (both waiting).
- 2026-11-16 16:50 PT (the restart step's output, the unit markers): the workers 6942622 and 6942623 ended at about
  16:37 PT; three of their four units restarted at 16:39 PT on three RTX A4500 cards on jaguar03 (job 6942792), and
  the fourth, the method's U-maze unit with reward 1 on arrival (fast step 0.1 with the hold, learning rate 1e-2,
  weight 16; 90 windows), went to Gilbreth at 16:41 PT (one new job sent there). It was left out because waiting
  units claim cards in the builder's order: RND U-maze 69 to 76, Medium and Large 77 to 84, the method's U-maze
  units 85 to 92. The method's U-maze units are the longest chain left (166 to 183 windows of validation, then 256
  windows of fresh seeds; about 58 hours at 7.5 windows an hour), so from now on the restart step gives every
  waiting method U-maze validation unit an order 30 lower (55 to 62; they claim first) and every waiting U-maze unit
  the 14,000 MiB floor. Gilbreth holds three waiting units (the RND U-maze step-cost unit at 1e-4, weight 1/64; the
  method's unit above; the RND Large unit at weight 16), with three waiting jobs; the U-maze ones come back when
  UVA cards free up.
- 2026-11-16 16:45 PT (window record): fifteenth fault-guard respawn under RK4, the second in a method validation
  unit and the first with reward 1 on arrival in a validation block: the first-arrival hold alone at learning rate
  3e-5, weight 1/16, window 161, seed 10048, torso angular speed 103.1 rad/s (torso speed 15.3 m/s). All fifteen
  are the angular bound; thirteen of the fifteen are at learning rate 3e-5.
- 2026-11-16 17:10 PT (the restart loop's log, worker logs): at 17:06 PT eight units waited on UVA after their
  workers ended; seven started on seven free cards (jobs 6942847 to 6942850: A100 40 GB on cheetah01, RTX 4000 Ada
  on cheetah02 and lynx01, RTX A4500 on jaguar03), and the eighth, the method's Large unit at weight 16 (last in
  the claim order), went to Gilbreth at 17:08 PT with one new job. Gilbreth holds four waiting units (two U-maze,
  two Large) with four waiting jobs. The take-back step now counts only the U-maze units waiting on UVA ahead of
  the U-maze units on Gilbreth: Medium and Large units claim after every U-maze unit, so a card they would have
  taken goes to a U-maze unit taken back, and they go to Gilbreth instead (U-maze first).
- 2026-11-16 17:29 PT (the restart loop's log, the worker log): the new take-back step worked on its own: at
  17:28 PT it found one UVA card free for the two U-maze units waiting on Gilbreth, took back the method's U-maze
  unit with reward 1 on arrival (fast step 0.1 with the hold, learning rate 1e-2, weight 16; 90 windows; taken
  first by name), gave it order 58, and it started on an RTX 5080 on nekomata01 at 17:29 PT (job 6942857).
  Gilbreth holds three waiting units: the RND U-maze step-cost unit at 1e-4, weight 1/64 (200 windows, waiting since
  16:03 PT) and two Large units, with four waiting jobs.
- 2026-11-16 18:08 PT (the restart loop's log, Gautschi's squeue): the restart step took the RND U-maze step-cost
  unit at 1e-4, weight 1/64 (200 windows, on Gilbreth's queue since 16:03 PT) back at 18:04 PT; it resumed on an RTX
  5080 on nekomata01 (job 6942882). All eight RND U-maze units run on UVA again. Gilbreth holds the two Large units
  (RND and the method, both at weight 16), waiting with four of our jobs. Gautschi `ai` still holds the account's
  four H100 cards for another session (14 more of its jobs wait). The session's 5-minute fleet loop reached its
  12-hour limit and was started again at 18:07 PT (pid 106100) while units are on Gilbreth.
- 2026-11-16 18:42 PT (the restart loop's log): three of the method's U-maze units with reward 1 on arrival waited
  after their workers ended (they now claim first, orders 55 to 57). Of the two workers sent at 18:40 PT, job
  6942928 still waited after 110 seconds and was cancelled by its ledger id, as the restart step does; job 6942929
  took two units. The third (fast step 0.03 with the hold, learning rate 1e-4; 149 windows) went to Gilbreth at
  18:42 PT; the take-back step brings it back when a UVA card frees up. Gilbreth holds three waiting units (it and
  the two Large units at weight 16) with four waiting jobs.
- 2026-11-16 19:51 PT (the restart step's output): the three RND U-maze units on the RTX 5080 cards waited after
  their 2-hour workers ended; they restarted at 19:49 PT (jobs 6943078 on an RTX 5080, 6943084 and 6943085). Both
  Large units now run on Gilbreth A10 cards (jobs 12403171 and 12403362). The session's restart loop, near its
  12-hour limit (it would have ended at about 20:08 PT), was stopped and started again at 19:51 PT.
- 2026-11-16 20:20 PT (unit_complete records): the first two RND U-maze validation units are complete, all 256
  windows: reward 1 on arrival at learning rate 1e-4, weight 1/64 (20:20:16 PT), and the step cost at learning rate
  3e-5, weight 1/16 (20:18:17 PT; three fault-guard respawns in it). The other six are at 225 to 253 windows.
- 2026-11-16 20:50 PT (scontrol show reservation, the submit log): the owner's reservation sl5nw_168 (jaguar03 and
  puma01) ends at 21:00:01 PT tonight. Since about 20:45 PT the UVA planner still offers jaguar03's free RTX A4500
  cards inside it, and submit.py refuses such a job ("No submission window of at least fifteen minutes remains");
  the error is not caught, so it also ends the rest of that submission pass. At 20:47 PT the take-back step counted
  two such cards, took the method's U-maze unit (185 windows) back from Gilbreth, no worker could be sent, and it
  went back to Gilbreth with a second method U-maze unit (fast step 0.03 with the hold, learning rate 1e-4); one new
  Gilbreth job was sent. free_cards_now.py now leaves out cards in a reservation that ends within 15 minutes. Our
  two workers inside the reservation (6942958 with two cards and 6943085, the RND U-maze step-cost unit at 1e-2,
  weight 16) end at 20:59 PT with it; after 21:00 PT jaguar03 is an open node and the restart step uses it again.
  Third RND U-maze validation unit complete (reward 1 on arrival, learning rate 1e-4, weight 1/256).
- 2026-11-16 21:10 PT (the restart step's output, worker logs): the reservation sl5nw_168 ended at 21:00 PT; its two
  workers on jaguar03 ended at 20:59 PT and their units waited, so at 20:49 PT three units had gone to Gilbreth
  (six there then: two Large running, four U-maze waiting). Fourth RND U-maze validation unit complete (reward 1 on
  arrival, learning rate 3e-5, weight 1/64). With jaguar03 open again, the take-back step brought one U-maze unit
  back at 20:59 PT and the other four at 21:07 PT (RND step cost at 1e-2, weight 16 and at 3e-5, weight 1/64, 250
  and 253 windows; the method's units at 160 and 185 windows); they run on an A100 on cheetah01 (job 6943190) and
  three RTX A4500 on jaguar03 (job 6943191). Job 6943179 (three RTX A4500) took the Medium and Large RND units that
  waited at 21:00 PT. UVA 18 units, Gilbreth 2 (both Large units, running); five Gilbreth jobs wait with no unit.
- 2026-11-16 22:06 PT (unit_complete records; ant_fresh_description.py on all 256 windows): **RND's U-maze validation
  block with reward 1 on arrival is complete** (four units, 128 seeds each, seeds 10000 to 10127; no fault-guard
  respawn in any of them). Whole-curve mean W, final level L (last 10%), mean capped first-goal time M:
  learning rate 1e-4, weight 1/64: W 0.1061 ± 0.0015, L 0.1183, M 2.49 million steps; learning rate 3e-5, weight
  1/64: W 0.1055 ± 0.0018, L 0.1234, M 5.64; learning rate 1e-4, weight 1/256: W 0.1024 ± 0.0018, L 0.1170, M 3.69;
  learning rate 1e-2, weight 4: W 0.0617 ± 0.0006, L 0.0656, M 0.30. All 128 policies of every configuration
  arrived. The choice for W is learning rate 1e-4, weight 1/64; the choice for M is learning rate 1e-2, weight 4.
  Description `infrastructure/descriptions/antmaze_umaze_rk4-physics_fresh-seeds_rnd_reward-arrival-plus-one_seeds-30000-to-30255_budget-134217728.json`
  (phase fresh-seeds-rk4, seeds 30000 to 30255); build_units.py added four units of 128 seeds (each choice at seeds
  30000 to 30127 and 30128 to 30255). Their claim order is set to 63 to 66: after the method's U-maze validation
  units (55 to 62), before the RND validation units still running (69 to 76) and Medium and Large (77 to 84).
  They are placed like every ant unit: UVA cards free now, then Gilbreth.
- 2026-11-16 23:17 PT (unit_complete records; ant_fresh_description.py on all 256 windows): **RND's U-maze validation
  block with the step cost is complete**: learning rate 3e-5, weight 1/64: W −0.970784 ± 0.0005, L −0.9646, M 5.29
  million steps (3 fault-guard respawns); learning rate 1e-4, weight 1/64: W −0.970802 ± 0.0006, L −0.9652, M 1.89;
  learning rate 3e-5, weight 1/16: W −0.9713 ± 0.0005, L −0.9656, M 4.70 (2 respawns); learning rate 1e-2, weight
  16: W −0.9849 ± 0.0002, L −0.9839, M 0.31. All 128 policies arrived in every configuration. The choice for W is
  learning rate 3e-5, weight 1/64 (ahead of 1e-4, weight 1/64 by 0.00002, far inside one standard error; the rule
  takes the larger W); the choice for M is learning rate 1e-2, weight 16. Description
  `infrastructure/descriptions/antmaze_umaze_rk4-physics_fresh-seeds_rnd_reward-step-cost_seeds-30000-to-30255_budget-134217728.json`;
  four units of 128 seeds built, claim order 67 to 70 (after the reward-on-arrival fresh seeds, 63 to 66).
- 2026-11-17 00:47 PT (window record): sixteenth fault-guard respawn under RK4, the third in a method validation
  unit: the step cost with fast step 0.1 and the hold at learning rate 1e-4, weight 1/64 (the method's third choice
  by the whole-curve mean), window 126, seed 10117, torso angular speed 102.7 rad/s (torso speed 13.7 m/s). All
  sixteen are the angular bound; thirteen of the sixteen are at learning rate 3e-5, three at 1e-4.
- 2026-11-17 01:24 PT (window record): seventeenth fault-guard respawn under RK4: the method's step-cost unit with
  fast step 0.1 and the hold at learning rate 1e-4, weight 1/64 (the unit of the sixteenth), window 130, seed
  10067, torso angular speed 129.1 rad/s (torso speed 14.4 m/s). All seventeen are the angular bound; thirteen at
  learning rate 3e-5, four at 1e-4.
- 2026-11-17 01:30 PT (Gilbreth's sacct, Gautschi's squeue, the restart loop's log): all eight Medium and Large
  validation units had gone to Gilbreth when their UVA workers ended (U-maze units claim free cards first). Seven
  waited there, up to about 2 hours, behind our oldest waiting Gilbreth jobs (submitted 20:48 to 20:50 PT, still
  waiting on `Priority`). Gautschi `ai` again holds the account's four H100 cards for another session (seven more of
  its jobs wait); MIT stays left to the tinker study; no ACCESS server is prepared for ant units. The restart step
  now also takes Medium and Large units waiting on Gilbreth back to UVA, only to cards left over after every unit
  waiting on UVA (counted at their 17,000 MiB floor). Its first pass at 01:28 PT took five back; they run on UVA
  (job 6943615). Gilbreth holds three Medium units (one running on an A10, two waiting).
- 2026-11-17 02:50 PT: monitoring moves to an hourly tick (cron a6a9b6fa, created 02:48 PT, expires about
  2026-11-24 02:48 PT). The restart step (every 10 minutes) restarts UVA units after their 2-hour workers and takes
  units back from Gilbreth's queue to free UVA cards (U-maze first, then Medium and Large). UVA 23 units, Gilbreth 1
  (an RND Medium unit waiting, 208 windows). The tick returns to every 20 minutes when a validation block completes
  (the method's U-maze blocks: about 120 to 210 windows of 256 now), a unit fails, or new units are built.
- 2026-11-17 03:23 PT (window record): eighteenth fault-guard respawn under RK4: the method's step-cost unit with
  the first-arrival hold alone at learning rate 1e-4, weight 1/1024, window 151, seed 10092, torso angular speed
  102.5 rad/s (torso speed 14.6 m/s). All eighteen are the angular bound; thirteen at learning rate 3e-5, five at
  1e-4. The same pass took the last Medium unit back from Gilbreth (03:23 PT): all 24 ant units run on UVA, so the
  5-minute fleet loop ended by itself at 03:23:41 PT.
- 2026-11-17 03:45 PT (window record): nineteenth fault-guard respawn under RK4: the method's unit with reward 1 on
  arrival and the first-arrival hold alone at learning rate 3e-5, weight 1/16, window 227, seed 10080, torso angular
  speed 104.1 rad/s (torso speed 13.9 m/s). It is that unit's second. All nineteen are the angular bound; fourteen at
  learning rate 3e-5, five at 1e-4. At 04:27 PT all 24 ant units run on UVA, none waiting, none on Gilbreth.
- 2026-11-17 05:14 PT: the first of the method's U-maze validation units completed (reward 1 on arrival, fast step
  0.03 with the hold, learning rate 1e-4, weight 1/64; 256 windows; no respawn). Results are in the auto research
  folder's STATE.md. The restart step at 05:20 PT started the six units whose 2-hour workers had ended (jobs
  6943714 to 6943746); at 05:26 PT all 23 remaining ant units run on UVA, none waiting, none on Gilbreth. The
  monitoring tick returns to every 20 minutes (RND's Medium units are at 254 and 255 windows, one Large unit at 252).
- 2026-11-17 05:46 PT: RND's two Medium validation units completed (05:33 and 05:37 PT; no respawn). Built RND's
  Medium fresh seeds (choice for W: learning rate 1e-2, weight 16): two units of 128 policies, seeds 30000 to 30255,
  claim order 85 and 86, at least 17,000 MiB per card, code copy number 1; UVA jobs 6943753 and 6943754 started
  them at 05:47 PT.
- 2026-11-17 05:48 PT: six 2-hour UVA workers (10 cards, jobs 6943683 to 6943688) stopped at 05:48:47 to 05:48:53
  PT, about 1 min 50 s after the same pass had placed its UVA workers, so their 10 units were handed to Gilbreth's
  queue at 05:48:55 PT. The next pass (05:50 PT) took all 10 back (nine U-maze units, one Medium unit) and started
  them on UVA cards free now (jobs 6943757 to 6943762, all running at 05:52 PT). At 05:54 PT all 23 ant units run on
  UVA; ten Gilbreth jobs wait with no unit.
- 2026-11-17 06:00 PT: RND's Large validation unit at learning rate 1e-2, weight 1 completed (256 windows, no
  respawn): W 3.9e-6 ± 0.5e-6, L 6.6e-6, M 108.2 million steps, 49 of 128 policies arrived. RND's other Large unit
  (weight 16) was at 219 windows at 06:05 PT; the Large fresh seeds wait for it. At 06:05 PT all 22 ant units run on
  UVA, none waiting, none on Gilbreth.
- 2026-11-17 06:26 PT: the session's 10-minute restart loop was restarted between passes (its 12-hour run would have
  ended about 07:51 PT); the new run ends by about 18:26 PT. At 06:25 PT all 22 ant units run on UVA, none waiting,
  none on Gilbreth; 19 fault-guard respawns in all, none new since 03:45 PT.
- 2026-11-17 07:38 to 07:53 PT: the 2-hour UVA workers started about 05:20 to 05:52 PT stopped one after another.
  The 07:36 PT pass handed RND's two Medium fresh-seed units to Gilbreth (07:38 PT; 12 and 11 windows); ten more
  units waited on UVA by 07:45 PT. The 07:48 PT pass started the ten on UVA cards free now (jobs 6946363 to
  6946742, all running at 07:50 PT) but did not take the Medium units back: its count subtracts every unit waiting on
  UVA, including U-maze units that can use RTX 5080 cards, which the Medium units (17,000 MiB) cannot. Free cards at
  17,000 MiB were then 2 (RTX A4500 on jaguar03), so take_back_and_start.sh took both back at 07:52 PT and job
  6947589 started them there. At 07:53 PT all 22 ant units run on UVA, none waiting, none on Gilbreth.
- 2026-11-17 08:35 PT: the second of the method's U-maze validation units completed (reward 1 on arrival, the
  first-arrival hold alone, learning rate 3e-5, weight 1/16; 256 windows; its 2 fault-guard respawns are among the
  19). Results are in the auto research folder's STATE.md. At 08:45 PT all 21 remaining ant units run on UVA, none
  waiting, none on Gilbreth.
- 2026-11-17 09:09 PT: four Medium and Large units waited on UVA after their workers' 2-hour stops. The pass started
  three on lynx01 (job 6947983, three RTX 4000 Ada cards); its A40 job on jaguar06 (6947982) was still waiting
  after 110 s and was cancelled by its ledger id. The fourth unit (the method's Large unit at weight 16) went to
  Gilbreth's queue at 09:11 PT; the take-back step brings it back when a UVA card at 17,000 MiB is free now.
- 2026-11-17 10:01 PT: the third of the method's U-maze validation units completed (reward 1 on arrival, fast step
  0.1 with the hold, learning rate 1e-2, weight 16; 256 windows; no respawn). Results are in the auto research
  folder's STATE.md. At 10:05 PT all 20 remaining ant units run on UVA, none waiting, none on Gilbreth.
- 2026-11-17 11:46 PT: the method's two Medium validation units and RND's Large unit at weight 16 completed (no
  respawn). Built the method's Medium fresh seeds (weight 16; claim order 87 and 88) and RND's Large fresh seeds
  (weight 1; claim order 89 and 90): two units of 128 policies each, seeds 30000 to 30255, at least 17,000 MiB per
  card, code copy number 1. The method units' specifications match the validation units' except for the seeds and
  names (same bonus options). UVA jobs 6948214 to 6948216 started all four at 11:47 PT; at 11:49 PT all 21 ant units
  run on UVA, none waiting, none on Gilbreth.
- 2026-11-17 12:03 PT: the method's last U-maze validation unit with reward 1 on arrival completed (step 0.03 with
  the hold, learning rate 1e-4, weight 1/16; no respawn). Built the method's U-maze fresh seeds with reward 1 on
  arrival: the choice for W (first-arrival hold alone, learning rate 3e-5, weight 1/16) and the choice for M (step
  0.1 with the hold, learning rate 1e-2, weight 16), two units of 128 policies each, seeds 30000 to 30255, at least
  14,000 MiB per card, code copy number 1, claim order set to 71 to 74 (ahead of the Medium and Large fresh seeds).
  UVA jobs 6948249 to 6948251 started all four at 12:08 PT; at 12:10 PT all 24 ant units run on UVA.
- 2026-11-17 12:46 PT: two of RND's U-maze step-cost fresh units (learning rate 1e-2, weight 16; learning rate 3e-5,
  weight 1/64) became waiting during the 12:43 PT pass, after its UVA workers were placed, so they were handed to
  Gilbreth at 12:46:36 PT. take_back_and_start.sh took both back at 12:47 PT (2 UVA cards free now at 14,000 MiB)
  and jobs 6948519 (jaguar01) and 6948520 (jaguar06) run them. At 12:47 PT all 24 ant units run on UVA. The same
  timing happened at 05:48 and 07:38 PT; the 10-minute pass takes such units back on its own, so ant_tick.sh is left
  as it is.
