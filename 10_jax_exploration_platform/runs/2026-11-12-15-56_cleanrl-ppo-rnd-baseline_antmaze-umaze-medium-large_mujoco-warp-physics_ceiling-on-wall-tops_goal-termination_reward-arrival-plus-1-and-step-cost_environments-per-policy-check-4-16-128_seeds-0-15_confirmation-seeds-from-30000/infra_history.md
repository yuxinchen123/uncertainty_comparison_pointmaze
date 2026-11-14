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
