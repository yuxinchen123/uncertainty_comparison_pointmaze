# Infrastructure history

Times are Pacific.

- 2026-11-12 02:27 to 04:47 PT: PointMaze first pass, 16 units on 16 cards under both accounts (RTX 5080, A40,
  RTX 4000 Ada, RTX A4000), none failed. Measured per worker card: about one busy processor and 2.6 GiB of host
  memory (sacct MaxRSS 2.5 GiB for a one-card job, 10.4 GiB for a four-card job) against the 8 processors and 16 GiB
  asked for.
- 2026-11-12 10:48 PT: PointMaze second pass, 16 units. The planner found no node with a free card and room for
  8 processors and 16 GiB: the nodes with free cards have their memory held by processor-only jobs of other
  sessions of these two accounts. It therefore sent 16 workers that ask for a card type on any node, 8 of them
  for H100 NVL cards and 4 for A100 cards, which may wait for days here.
- 2026-11-12 10:52 to 10:56 PT: the submission got options for the request per card, and the workers now ask for
  2 processors and 4 GiB per card (the measured need plus a margin). Waiting workers cancelled by their ledger ids,
  none had started: 6884319 to 6884327 and 6884333 to 6884335 (the 12 that asked for H100 NVL or A100 cards), then
  6884457 to 6884462 (6 that asked for A40 cards, all in use) and 6884318, 6884336, 6884337, 6884338 (4 with the
  large request that waited on priority). Sent instead: 6884454, 6884455, 6884463, 6884467, 6884468, 6884456, and
  6884493 to 6884496, on RTX 2080 Ti, RTX 4000 Ada, GTX 1080 Ti and RTX 5080 cards. All 16 units were running by
  10:57 PT (times read from the scheduler's records; an earlier version of this entry gave later times that I
  had estimated instead of reading).
- The planner's list of card classes is older than the hardware: lynx01, lynx02, lynx05 and affogato13 now hold
  RTX 4000 Ada cards, which it does not pin to. Workers that ask for that card type on any node are placed there by
  the scheduler.
- 2026-11-12 12:07 to 12:24 PT: the three units of "development at the budget" waited with no worker.
  The planner counted two cards of the four-card job 6884496 as idle workers, but their worker processes had ended
  when the queue was empty (the job goes on for its other two cards). The planner now counts only the cards of
  waiting jobs as workers without a unit. The pinned worker 6886277 (ai02) waits on priority; workers 6886680
  (ai04) and 6886681 (any RTX 5080) were added at 12:24 PT (times from the scheduler's records).
- 2026-11-12 12:24 to 12:27 PT: the three added workers did not start although cards, processors and memory were
  free on several nodes. The card partition had 3,334 waiting jobs (2,901 of one other user), and our accounts'
  priority is low (815 to 1,220), so the scheduler did not reach them. At 12:26:54 PT one worker was started on an
  idle card inside job 6884496 (lynx04, yuxinchen; two of its four cards had finished their units), through
  `srun --jobid=6884496 --overlap`; its log is `logs/worker_on_idle_card_of_job_6884496.log`. The three waiting
  workers stay in the queue; whichever starts finds the remaining units or exits.
- 2026-11-12 13:11 to 13:26 PT: units of this run can now go to Purdue Gautschi and Gilbreth, because UVA's
  scheduler does not start our new jobs today. Times are from the clusters' own records.
  1. **Environment** on both clusters, beside the grid study's `rnd_platform_jax` (unchanged):
     `/scratch/<cluster>/liu4869/envs/rnd_platform_jax_mujoco_warp`, Python 3.12.13, built on the login nodes with
     conda and pip from wheels only (no numerical library was imported there). First build 13:11:14 to 13:13:37 PT
     on Gautschi and to 13:15:57 PT on Gilbreth: `jax` 0.11.0 with its CUDA 12 libraries, `numpy` 2.5.2, `mujoco`
     and `mujoco-mjx` 3.11.0, `flax` 0.12.8, `optax` 0.2.8, `warp-lang` 1.14.0. JAXAtari was added 13:25:02 to
     13:25:41 PT (Gautschi) and to 13:26:10 PT (Gilbreth), after the first unit failed without it (item 4). Now 79
     packages on each cluster: 76 are in UVA's `platform_jax` at the same version, `pip` and `setuptools` are newer
     (26.2.2 and 84.2.1 against 26.2.1 and 84.0.0), `warp-lang` is UVA's `platform_jax_warp` version, and only
     `PyMuPDF` of UVA's list is absent. JAXAtari is the same source commit (035fc46b) and its 157 installed files
     have the same checksum as on UVA. Build scripts, the pins, UVA's list used as constraints and each cluster's
     package list: `infrastructure/remote_fleet/setup/`; on each cluster `/scratch/<cluster>/liu4869/rnd_maze/setup/`
     and `<environment>.pip_freeze.txt` beside the environment.
  2. **Run folder and code on each cluster:**
     `/scratch/<cluster>/liu4869/rnd_maze/runs/2026-11-12-02-24_cleanrl-ppo-rnd-baseline` and
     `/scratch/<cluster>/liu4869/rnd_maze/code/source_6ad2fa46…/10_jax_exploration_platform` (the frozen copy of
     commit 6ad2fa46, sent 13:20 PT). `run_training.py` records the git commit of the folder above the code copy and
     stops where git reads none, so that folder is a git repository there with one commit of the copy, tagged
     `copy-of-uva-commit-6ad2fa46…` (the commits are 048b685c on Gautschi and 565e031a on Gilbreth).
  3. **Server table** `infrastructure/remote_fleet_servers.json`: Gautschi `smallgpu` (L40S, `-q normal`, account
     `dsimchil`, 64 cores per card, 12 hours, at most 4 jobs) and Gilbreth `standby` (account `dsmichil`,
     `-p a100-80gb,a30,a10`, 2 cores, 8 GB, 4 hours, at most 4 jobs; the scheduler starts the job in the first of the
     three partitions that has a card). Job ids: `slurm/submitted_jobids_gautschi.txt` and
     `slurm/submitted_jobids_gilbreth.txt`. The names carry no phase, because UVA's monitor reads every
     `submitted_jobids_*_*.txt` as UVA job ids. One command does a whole tick:
     `remote_fleet.py --run-dir RUN hand-over-tick` (scripts/maze_campaign/README.md, step 6).
  4. **First unit on Gilbreth failed once, because of the environment.** Job 12344547 (submitted 13:20:42 PT)
     started 13:21:34 PT on gilbreth-h003 (A10). Its unit (PointMaze Medium, step cost, CleanRL PPO with RND at
     CleanRL's defaults, seeds 30512 to 30767) ended after 4 s: `ModuleNotFoundError: No module named 'jaxatari'`
     (`training/compose.py` imports the Montezuma environment for every environment family). The unit went back to
     the server's pending folder with one failure counted. After JAXAtari was installed, the worker job script was
     changed to import the trainer's own modules before it claims a unit, so a missing library ends the job
     without costing a unit a failure. Job 12344581 (submitted 13:26:25 PT) started 13:26:41 PT on gilbreth-d004
     (A30) and took the unit again.
  5. **Gautschi `smallgpu` will not start our job today.** Job 18433051 (submitted 13:20:39 PT) waits on priority
     behind about 30 jobs for 8 L40S cards (4 more cards are on reserved nodes); the scheduler's estimate at
     13:28 PT was 2026-11-14 09:45 PT. Its unit (the same kind, seeds 30768 to 31023) was taken back to UVA's
     `queue/pending` at 13:26:24 PT, unstarted. New rule in the tick for Gautschi (`start_within_seconds` 7200 in
     the table): units wait there only while a job of ours runs or is expected by the scheduler to start within 2
     hours. The waiting job stays in the queue; if it starts with nothing pending it ends within seconds.
- 2026-11-12 13:10 to 13:20 PT: why our workers were not started, and what was changed. Workers asked for the
  partition's longest time (4 days). With a low priority, the scheduler places a job on a free card only when the
  job ends before the higher-priority jobs ahead of it are due to start, so a 4-day worker was never placed.
  1. Worker 6887495 was submitted at 13:10:51 PT with a 2-hour limit, 2 processors and 4 GiB; it started at
     13:11:29 PT on lynx01 (RTX 4000 Ada).
  2. At 13:12 PT the time limit of the 21 waiting workers of this folder's ledgers was lowered to 2 hours with
     `scontrol update` (same job ids, so they keep their age). Two started at 13:13:04 PT: 6886680 on ai04 and
     6887003 on ai08.
  3. The others could not start at any limit: they asked for cards in use (A40, H100 NVL) or for 8 processors and
     16 GiB, which no node with a free card has unallocated. Fifteen of them were cancelled by id from the ledgers
     at 13:18 PT (6887473 to 6887480, 6887007 to 6887012, 6886277); none had started. Eight of these had been
     submitted at 13:10 PT by the monitor job's own top-up, which uses the default request.
  4. So the monitor job 6874083 (which topped up with the default request) was cancelled by its ledger id at
     13:18:34 PT, and monitor 6887598 started at 13:19:41 PT without a top-up; workers are now submitted at the
     session's ticks with `--cpus-per-card 2 --memory-gib-per-card 4 --worker-hours 2` and the card types that
     are free.
  5. Fifteen such workers (6887574 to 6887597, RTX 4000 Ada and RTX 2080 Ti, any node) were submitted at
     13:18:35 to 13:19:11 PT. At 13:30 PT they were still waiting: the scheduler's planned starts are 15:11 PT
     and later, because the unallocated memory on the nodes with free cards (5 to 10 GiB each) is planned for
     higher-priority jobs. Only the first three small workers fitted.
  `submit.py` and `monitor.py` got the options `--worker-hours` (and the monitor passes the request size, the time
  limit and the card types on to its top-ups).
