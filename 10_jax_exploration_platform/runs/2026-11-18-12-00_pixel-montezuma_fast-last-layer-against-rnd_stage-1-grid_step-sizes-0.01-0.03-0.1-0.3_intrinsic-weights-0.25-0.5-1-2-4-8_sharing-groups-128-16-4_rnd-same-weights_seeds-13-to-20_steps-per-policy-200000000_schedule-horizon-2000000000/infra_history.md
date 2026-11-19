# Infrastructure history of the stage-1 grid

- 2026-11-18 13:10 PT: the launch and the monitor's top-ups held 4 free cards for the AntMaze study
  (`--leave-cards 4`) although no AntMaze unit was waiting, so 3 RTX A4500 cards on jaguar03 stood idle at the
  monitor's passes; and the Quadro RTX 6000 (7 cards on lotus, a type the user's card rule lists) was not in the
  study's card list. Both break item 4 of `~/.claude/rules/pure-rl-gpu-placement.md`. Fixed by a top-up with
  `--leave-cards 0` and the Quadro RTX 6000 added: job 6956682 (sl5nw, 3 x RTX A4500, jaguar03) and job 6956683
  (yuxinchen, 7 x Quadro RTX 6000, lotus), both running within 2 minutes. The Quadro job is the type's speed
  check: it is kept if its seconds per update are not clearly above the RTX A4000's (2.31 s in this sweep).
  AntMaze units that find no free UVA card go to Gilbreth (their default), so no card is held for them.
- 2026-11-18 13:20 PT: the Quadro RTX 6000 runs (job 6956683, lotus) measured 2.02 to 2.10 s per update (median over
  updates 11 to about 85 of seven runs), faster than the RTX A4000's 2.31 s, so the type stays. The monitor job
  6956662 was cancelled by its ledger id at 13:20 PT (between passes) and replaced by job 6956692 (nolim,
  heartpiece) with `--leave-cards 0` and the Quadro RTX 6000 in `--gpu-types` (command.txt, item 5).
- 2026-11-18 13:27 PT: the first-update comparison of the smoke check changed (commit da22476a): a decimal number
  passes within $|n - a| \le 10^{-3} |a| + 10^{-6}$, whole numbers equal (decision recorded in
  `experiment_background.md`, section "Remote servers"). The frozen code copy 2 keeps the old comparison inside
  smoke jobs; `remote_fleet.py smoke-result` (UVA working tree) decides that check again from each job's own 20
  default metric lines. The trainer copy that UVA workers run is unchanged.
- 2026-11-18 13:29 PT: `remote_fleet.py init-table` wrote `infrastructure/remote_fleet_servers.json`; this folder's
  limits set in it: Gautschi `ai-h100` jobs of 10 hours, at most 4 (account `dsimchil` had no job at 13:23 PT);
  Gilbreth only the A30 pool (`standby-a30`, 20 jobs, 20 units at once), the A100 80 GB and A10 pools held back
  until their card type is checked; ACES and DeltaAI not ready (not used in stage 1).
- 2026-11-18 13:30 PT: smoke jobs 18909289 (Gautschi, H100 80GB HBM3, node h003) and 12434104 (Gilbreth A30, node
  gilbreth-d004) submitted; `prepare` sent code copy 2 (commit 08b7c4c8), the job scripts and the smoke files.
- 2026-11-18 13:42 PT: Gautschi smoke job 18909289 PASSED (11 min 29 s, MaxRSS 8.3 GB; first update largest relative
  difference 5.95e-4, 0 numbers outside the rule; 2.30 s per update with four copies, default and fast layer
  alike; peak card memory 70,013 MB). Gautschi marked ready in this folder.
- 2026-11-18 13:43 PT: 8 packs made from never-started seed-13 singles (`build_units.py --pack-pending --packs 8`:
  the fast last layer at sharing group 16 with weights 0.5, 1, 2, 4, 8 and at sharing group 4 with weights 0.25, 0.5,
  1, each pack the four step sizes); all 8 handed to Gautschi; worker jobs 18909476 to 18909479 (one H100 and 14
  cores each, 10 hours) running on h003, h007, h008 and h017 within 23 s, each holding the one card it asked for.
  At 14:01 PT the four packs trained at 2.31 s per update (about 320 updates each); the other four wait on Gautschi.
- 2026-11-18 13:51 PT: Gilbreth A30 smoke job 12434104 (gilbreth-d004, a D node of 16 cores) did NOT pass: the trainer
  took about 7 minutes to its first update, so only 9 of 20 default updates and 1 fast-layer update fit in the smoke
  job's 420 s per training (the 9 written updates: 0 numbers of the first outside the rule; 1.92 s per update;
  MaxRSS 6.5 GB). Every other check passed. Gilbreth stays not ready.
- 2026-11-18 13:52 PT: Gilbreth A30 smoke job 12434389 submitted, pinned to the B nodes (`--constraint=B` added to the
  pool's options for this one submission and removed after it), where the gates-folder job 12432866 wrote all 20
  updates of both trainings. Estimated start 18:14 PT at 14:00 PT. The hand-over loop reads its verdict when it ends.
- 2026-11-18 14:00 PT: the gates-folder Gilbreth smoke jobs 12432865 (A100 80 GB) and 12432867 (A10) still wait
  (estimated starts 21:17 PT and 2026-11-19 01:32 PT); they stay in the gates folder's ledger.
- 2026-11-18 14:05 PT: the hand-over loop written (`infrastructure/hand_over_loop/`, the copy of the scratchpad
  script the session starts); not started by this step.
- 2026-11-18 14:34 to 14:55 PT: Gilbreth became ready at 14:34 PT (A30 smoke job 12434389 passed under the
  corrected first-update rule; the job itself ended FAILED because its own older check from the frozen copy failed,
  as designed). 20 units were handed over and 20 standby jobs submitted (12434768 to 12434787). At 14:39 PT the tick
  found that no job of this run ran there and the scheduler expected none to start within 2 hours, so it sent the
  20 waiting units back. The copy back then failed at every pass (14:39, 14:44, 14:49 PT): rsync exit 23,
  "change_dir .../data/<unit> failed", because a unit that never started on the server has no data/<unit>/ folder
  and `--ignore-missing-args` does not cover a missing parent folder. The 20 units stayed held by Gilbreth for
  15 minutes. Fixed in scripts/pixel_sweep_campaign/remote_fleet.py `bring_back`: the copy uses rsync's relative
  form (`--relative`, `queue/withdrawn/./<unit>.json`, `data/./<unit>/run`), which skips a missing unit folder
  (checked on local folders first); one test changed and one added, 92 standard-library tests pass. At 14:54 PT all
  20 units were back in queue/pending on UVA; Gilbreth's own queue holds them in withdrawn/, and its 20 worker jobs
  wait there (a worker that starts finds no unit and ends). The one Gilbreth job running at 14:49 PT was the gates
  folder's A10 smoke job 12432867, not a worker. No unit stood in two queue folders at any pass.
- 2026-11-18 15:50 PT: 17 RTX 2080 Ti cards became startable (other users' jobs ended on affogato11, ai01 to ai04,
  ai06, lynx10 and cheetah03). Submitted by hand before the monitor's next pass, with the monitor's options:
  jobs 6956854, 6956855, 6956856, 6956857, 6956859 (sl5nw) and 6956858, 6956860, 6956863 (yuxinchen), all
  running by 15:52 PT. The session's free-card reading used the system's free
  memory until now; it now reads Slurm's configured minus allocated memory, which is what decides a start (the
  monitor's planner already used it: its passes from 13:41 to 15:21 PT found no startable card).
- 2026-11-18 15:58 to 16:01 PT, user rule ("next time, just submit jobs early to wait in the queue at first place"):
  16 worker jobs that wait in UVA's queue were sent for the units without a worker (6956892 to 6956909: 4 A100
  40 GB, 4 RTX 5080, 6 A40, 1 Quadro RTX 6000, 1 RTX 2080 Ti; 9 under sl5nw, 7 under yuxinchen; all waiting on
  Priority), 8-hour jobs. The monitor job 6956910 replaced 6956692 (cancelled by its ledger id between passes) and
  tops up both the jobs on cards free now and the waiting ones (command.txt, item 6). Rule
  `~/.claude/rules/pure-rl-gpu-placement.md` item 3 revised the same day.
- 2026-11-18 16:12 to 16:16 PT: two four-run packs made from never-started singles (seed 14, sharing group 16,
  weights 0.25 and another of the planner's order; build_units.py --pack-pending --packs 2) for H100 NVL cards, and
  two 8-hour jobs that wait in the queue sent for them: 6956914 (sl5nw) and 6956915 (yuxinchen), both waiting on
  Priority. The 2 free H100 NVL cards on serval08 that the tick's reading showed were in use a minute later
  (AllocTRES gres/gpu=2). The waiting RTX 2080 Ti job 6956909 got a 2-hour limit (`scontrol update`) because a
  2080 Ti on ai01 was free; it still waited on Priority at 16:16 PT.
- 2026-11-18 16:33 to 16:46 PT: the four waiting RTX 5080 jobs (6956896 to 6956899) started on nekomata01 and each
  ended FAILED at update 0: out of card memory, one allocation of 11.69 GiB in `jit_iteration` at memory fraction
  0.85 (13.6 of 16.3 GB). The same fast-last-layer runs fit on the 11 GB RTX 2080 Ti at fraction 0.95 (10.5 GB), so
  the program XLA builds for this card (Blackwell) differs. The RTX 5080 had no speed check for this study (gate 6
  measured the RTX A4500, RTX 2080 Ti, RTX A4000 and RTX 4000 Ada). The four units went back to queue/pending with
  no step lost. Taken out of the card list: monitor job 6956910 cancelled by its ledger id at 16:52 PT and replaced
  by 6957005 without `nvidia_geforce_rtx_5080`. A 15-minute check at fraction 0.95 runs as job 6957009
  (gates folder, slurm/jobs/gate-6_speed-and-memory_rtx-5080_...); the type comes back only if it passes.
- 2026-11-18 16:53 to 16:58 PT: the RTX 5080 check (job 6957009, nekomata01, fraction 0.95) FAILED in its first
  variant, the default RND program, before any timed update: the same single allocation of 11.69 GiB in
  `jit_iteration` ran out of card memory. On 2026-11-07 four archived baseline seeds ran on these cards with the same
  environment variables (fraction 0.85, the determinism flag) using 8.6 GB, so something differs from then (the
  code, the driver now 580.159.04, or how XLA plans this card); the cause is not established. Decision: the RTX 5080
  stays out of the stage-1 card list (it has 4 cards, mostly used by the AntMaze study); no unit was lost.
- 2026-11-18 17:32 to 17:36 PT: at 17:22 PT the AntMaze study's 18 units came off their 2-hour UVA workers together;
  6 restarted on UVA cards free then and 12 went to Gilbreth, that study's default server (12 jobs waiting there).
  The account runs at most 24 Gilbreth jobs at once and this study held 20 units there, so its share was lowered
  to 12 (units at once and jobs, server table, under the fleet's lock), 8 of its waiting Gilbreth units were taken
  back to queue/pending (`take-back --units 8 --from-end`: the six RND runs of seed 14 and two fast-last-layer
  runs), and 7 of its surplus waiting Gilbreth jobs were cancelled by their ledger ids (12434786, 12434787,
  12435014 to 12435017, 12435048). After it: this study 12 units, 12 waiting jobs and 1 running job on Gilbreth;
  AntMaze 12 waiting jobs.
- 2026-11-18 17:32 PT: the first unit finished: RND at weight 2, seed 13, on an RTX A4500 (jaguar03, job 6956649),
  12,207 of 12,207 updates in 5 hours 8 minutes (12:24 to 17:32 PT).
- 2026-11-18 19:14 PT, user instruction of about 19:00 PT ("MIT is free now, you can also send those to MIT
  requesting H200 (you might be able to pack more to MIT because they are H200)", then "send montezuma to mit not
  antmaze"): MIT added for this study (plan decision 14). Read through the shared connection held on portal04
  (open since 2026-11-17 11:55 PT; `ssh -O check mit`, no new login): account shuzel had no MIT job; home 68.9 of
  200 GB, scratch 908 of 1,024 GB. Everything of this study on MIT goes to home (backed up, room for the 7.1 GB
  environment and about 280 MB per finished pack).
- 2026-11-18 19:16 PT: the asset pack copied to `/home/shuzel/rnd_pixel/assets/` on MIT; `sha256sum -c` against the
  gates folder's `asset_pack_sha256.txt`: all 4,531 SHA-256 equal (the UVA pack was hashed again first and equals
  that list).
- 2026-11-18 19:18 to 19:38 PT: the environment `/home/shuzel/envs/atari_pixels_jax` built with the gates folder's
  `build_environment.sh`, unchanged. The first try downloaded from PyPI on the MIT login node at about 90 to 100 kB/s
  (conda-forge was fast), which would have taken hours for the 3.4 GB of wheels; it was stopped after 5 minutes (its
  own processes) and its partial environment removed. A `pip download` on UVA was as slow (about 200 kB/s) and was
  stopped too. The same 88 wheel files were then taken from Gautschi's build cache of 2026-11-18 (`pip download` on
  the Gautschi login node, 20 s), copied Gautschi to UVA to MIT (SHA-256 equal on UVA and MIT:
  `infrastructure/environment_setup_mit/wheelhouse_sha256.txt`), and the build ran again with `PIP_NO_INDEX=1
  PIP_FIND_LINKS=<that folder>` (conda packages from conda-forge as before; JAXAtari cloned from GitHub at its pinned
  commit; hatchling 1.33.0 for the two source builds, as on Gautschi). Result: Python 3.12.13; `pip check` clean;
  the 88 distributions equal Gautschi's file by file (88 of 88) and UVA's in 87 of 88: JAXAtari (not imported by the
  trainer) differs only in its `WHEEL` file (written by hatchling 1.33.0 there, 1.32.0 on UVA), as on Gautschi. 7.1
  GB, 49,740 files. Records in `infrastructure/environment_setup_mit/`. The download caches and the wheel copies on
  MIT, UVA and Gautschi were removed afterwards.
- 2026-11-18 19:30 PT: MIT entry added to `infrastructure/remote_fleet_servers.json` (not ready) and to the tooling's
  `scripts/pixel_sweep_campaign/server_table.json`. Pools `advanced-h200` (mit_normal_gpu with the Advanced account
  and QoS) and `preemptable-h200` (mit_preemptable, the nodes shared with mit_normal_gpu and node3401-c1 left out):
  `--gpus=h200:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --requeue --open-mode=append`, the card guard with
  `GUARD_ACTION=requeue` in every job script, 2-day jobs (both pools' limit; rule mit-cluster-slurm.md asks for the
  partition maximum), at most 4 jobs each (the account's caps). `--mem=24G` from the four-copy smoke jobs' MaxRSS
  (8.3 to 9.4 GB). The ssh command is `ssh -o BatchMode=yes -o ControlMaster=no -o ProxyCommand=false mit`: with the
  shared connection closed it fails on portal04 without contacting MIT (checked with a missing control socket:
  exit 255, nothing sent), so no command can become a new login; every call also checks `ssh -O check mit` first.
  Units held at once: 16 (2 per card), kept in the first pool's limit, because `hand_over` counts every four-copy
  unit a server holds against each pool's limit.
- 2026-11-18 19:40 PT: `remote_fleet.py smoke --server mit`: `prepare` sent code copy 2 (commit 08b7c4c8), the job
  scripts and the smoke files; smoke jobs 29603827 (advanced-h200) and 29603828 (preemptable-h200), 20 minutes each.
  Both waited (Priority); the scheduler's estimate for 29603827 was 21:18 PT at 19:44 PT (164 waiting H200 jobs on
  mit_normal_gpu and 280 on mit_preemptable then, from all users).
- 2026-11-18 19:41 PT: pack-size profile job 29603853 (advanced-h200, one H200, 65 minutes; scripts in
  `infrastructure/mit_pack_size_profile/`): four settings of at most 15 minutes each on one card, the default program
  with four copies (seeds 1 to 4) and eight copies (seeds 1 to 8, reset lane budget 32), and the fast last layer
  (sharing group 16, both records) with four and eight copies. Estimated start 22:00 PT at 19:44 PT. All three ids in
  `slurm/submitted_jobids_mit.txt`.
- 2026-11-18 20:06 to 20:16 PT: MIT smoke job 29603827 (advanced-h200, node3000, NVIDIA H200 143,771 MB, driver
  590.48.01) PASSED: AllocTRES equal to ReqTRES (`gres/gpu=1`, card index 3), card guard "ok: 1 card(s)"; asset
  pack 4,531 files equal; the 20 default updates equal the archived four-copy run (H100 NVL) in every field of every
  update, bit for bit; the fast last layer with the game events completed; 2.16 s per update with four copies
  (default) and 2.15 s (fast last layer, group 16); peak card memory 75,343 MB; MaxRSS 9.5 GB; 9 min 42 s.
- 2026-11-18 20:16 PT: pack-size profile job 29603853 started on node3000 (the same card, index 3); AllocTRES equal
  to ReqTRES (one H200), card guard ok.
- 2026-11-18 20:19 PT: the preemptable pool is held back (`pools_held_back` in the MIT entry, `units_at_once_when_back`
  0) until its own smoke job passes: 29603828 waited on Resources with no estimated start (every H200 node that
  only mit_preemptable has was allocated), and those nodes include H200 NVL cards, another part than the Advanced
  pool's H200. New in `remote_fleet.py`: `smoke-result --server S --pool P` reads one held-back pool's verdict and
  brings the pool back only when it passed (the server's ready state is not changed); 96 tests pass.
  `smoke-result --server mit` then made MIT READY (its one pool that takes units passed).
- 2026-11-18 20:19:41 PT: the running hand-over loop (pass 75, 22 s after MIT became ready, before the share below was
  set) made 14 packs (`build_units.py --pack-pending`, seed 15 mostly), handed 16 packs to MIT (the table's limit of
  16 then) and submitted 4 Advanced worker jobs: 29606825, 29606826, 29606828, 29606833 (2 days each, all waiting;
  one of them waits on the account's cap of 4 Advanced cards while the profile job holds one).
- 2026-11-18 20:20 PT: MIT's share set from the account's other jobs (`mit-share` of the new loop's helper): the
  profile job holds 1 Advanced card, so this run's Advanced pool takes at most 3 new jobs and the server holds at most
  6 packs (2 per card). 8 of the 16 waiting packs were taken back to queue/pending (`take-back --server mit --units 8
  --from-end`), leaving 8 on MIT for the 4 jobs (2 per card). queue-check: every unit in one queue folder.
- 2026-11-18 20:24 and 20:26 PT: MIT worker jobs 29606825 (node3201, card index 5) and 29606826 (node3301, card
  index 3) started; each read 28 s after its start: AllocTRES equal to ReqTRES (`gres/gpu:h200=1,gres/gpu=1`), card
  guard "ok: 1 card(s)". Each took a pack (sharing group 128, weight 8, seed 14; sharing group 16, weight 0.25, seed
  14). Jobs 29606828 and 29606833 still waited (Priority; the scheduler's estimates 2026-11-19 06:30 and 07:30 PT).
  The start of every job of the MIT ledger is read by
  `/tmp/claude-2618919/-p-rlprojects-RND/9ca0cb45-e2aa-43c6-9353-3a3d17cddb5f/scratchpad/mit_start_card_check.sh`
  (one command every 45 s through the shared connection, for at most 4 hours from 20:21 PT), which appends each
  reading to `mit_start_card_check.txt` beside it.
- 2026-11-18 20:16 to 20:45 PT: the pack-size profile (job 29603853, one H200, node3000, card index 3; MaxRSS 12.1 GB):

  | setting | copies | seconds per<br>iteration | total steps<br>per second<br>(millions) | steps per<br>second per<br>copy<br>(thousands) | hours per<br>million steps<br>per copy | peak card<br>memory (MB) |
  |---|---|---|---|---|---|---|
  | default, seeds 1 to 4 | 4 | 2.169 | 0.0302 | 7.55 | 0.0368 | 75,343 |
  | default, seeds 1 to 8 | 8 | 3.546 | 0.0370 | 4.62 | 0.0601 | 122,785 |
  | fast last layer, group 16,<br>both records, seeds 13 to 16 | 4 | 2.166 | 0.0303 | 7.57 | 0.0367 | 75,343 |
  | fast last layer, group 16,<br>both records, seeds 13 to 20 | 8 | 3.554 | 0.0369 | 4.61 | 0.0603 | 122,787 |

  Eight copies give 22.3% (default) and 21.9% (fast last layer) more total steps per second than four, each copy
  1.64 times slower (a stage-1 run 12.0 hours instead of 7.4). At eight copies the card's memory pool reached the
  0.85 share JAX may take (122.8 of 143.8 GB); nothing ran out of memory. Each copy against the archived baseline
  for its seed (`compare_pack_size_profile.py`, output in `infrastructure/mit_pack_size_profile/comparison_job-29603853.txt`):
  four copies: every field of all 20 archived updates equal, bit for bit, for seeds 1 to 4; eight copies: every
  copy's first update within the smoke rule (largest relative differences 8.9e-5 to 4.1e-3, 0 numbers outside the
  rule), but 166 to 173 differing fields in the 20 updates, and seeds 1 to 4 differ from the four-copy run on the
  same card in 524 to 531 fields of 60 updates. So a copy's numbers depend on the number of copies in the process at
  the level of rounding (the cause is not established; the reset lane budget was doubled with the copies, which by
  design changes only the speed). The fast-layer runs were finite and completed at both sizes.
- 2026-11-18 20:50 PT, decision: four-copy packs on MIT. Reasons: (1) plan decision 14 allows more than four only if
  every copy's numbers stay those of its seed; with four copies the H200 reproduces the archived four-copy run bit
  for bit, with eight it does not (the first-update rule passes, the bits do not), and the plan makes bit-for-bit
  claims for runs on one card type; (2) eight-copy packs need a new execution profile (card memory of 120,000 MB or
  more), a builder for packs over two settings (four step sizes times two weights), eight-wide smoke references
  from two archived runs, and a new frozen code copy on every server during the sweep; (3) its smoke job could not
  start while the account's Advanced cards hold this run's 2-day jobs. The 22% stays recorded for the user, who may
  choose eight copies, accepting rounding-level differences of the size seen between card types.
