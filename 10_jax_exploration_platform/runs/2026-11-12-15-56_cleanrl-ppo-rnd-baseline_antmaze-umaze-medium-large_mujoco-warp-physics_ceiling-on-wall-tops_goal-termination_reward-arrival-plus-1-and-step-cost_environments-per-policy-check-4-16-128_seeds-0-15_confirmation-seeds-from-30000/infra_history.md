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
