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
- 2026-11-18, before 20:47 PT (the time of the commit that recorded it): two H100 NVL cards free (serval07, serval08) while the two H100 NVL pack jobs waited; both got
  a 2-hour limit (`scontrol update`: 6956914 under sl5nw, 6956915 under yuxinchen). About a minute later 6956914 still waited
  on Priority and 6956915 on its own account's card cap (yuxinchen holds 40 of 40). MIT: four-copy packs train on
  two Advanced-pool H200 cards (29606825, 29606826, one card each as asked); two more wait (29606828, 29606833).
- 2026-11-18 20:55 PT: after the profile job ended, MIT's share set again: this run's Advanced pool at most 4 jobs,
  8 packs held on the server (8 held: 2 training at 2.17 and 2.20 s per update, 6 waiting). The watcher of job
  starts runs detached for at most 16 hours (process 702046 on portal04, pid file beside the script) for the jobs
  not started yet: 29606828, 29606833 and the preemptable smoke job 29603828. The new loop
  (`montezuma_hand_over_loop_with_mit.sh` in the session's scratchpad) is written and not started; the loop that
  runs (process 53680) already serves MIT through `hand-over-tick`, without the MIT connection and share steps.
- 2026-11-18 20:49 PT: the hand-over loop without MIT (process 53680) was stopped between passes (after pass 80,
  20:45 PT) and the loop with MIT started (process 704949; script montezuma_hand_over_loop_with_mit.sh in the
  session's scratchpad, copy in infrastructure/hand_over_loop/); its first pass ended 20:49 PT with every unit in
  one queue folder. Correction: the entry above first gave 20:52 and 20:53 PT, times that were not read from the
  clock; the commit time is the reading.
- 2026-11-18 21:37 PT: the same `KeyError: 'EndTime'` fault (see the AntMaze folder's infra_history.md) was in the
  frozen copy 2 this run's monitor job submits with; its top-ups had planned no new job since the fault began, so
  none had crashed yet. The frozen copy's scripts/grid_initialization_campaign/submit.py was replaced by the fixed
  file (submission tooling only; no trainer or worker file changed): the difference is the skip of lines without a
  ReservationName and `reservation is not None` in the match.
- 2026-11-18 22:18 PT, user decision of about 21:50 PT ("8 montezuma per h200"; the rounding-level differences between
  eight-copy and four-copy runs accepted): the tooling (`scripts/pixel_sweep_campaign/`) gets an "eight copies"
  execution profile for cards of 140,000 MB or more (the H200): the four-copy settings with the reset lane budget
  32, as job 29603853 measured them; "four copies" now ends below 140,000 MB, so the ranges do not overlap. A
  profile's `continues` names profiles whose started units it also continues: an H200's worker continues a
  four-copy pack that started, never starts one. `build_units.py --pack-pending --pack-size 8` makes packs of
  eight never-started singles that share the method, the sharing group and the fast layer's state: the four step
  sizes at two weights of one group and seed, or RND's weights 0.25 to 2 at two seeds (a full cross product of the
  values in the name); an eight-copy pack's record needs the run's frozen code copy, a four-copy pack keeps its
  singles' number. Every run's record says its pack size: `copies` in the unit's record and in every
  `worker_start` line of its events.jsonl; report.py counts runs by it in the fleet section (its speed table
  already reads by card type and copies). `remote_fleet.py`: the archived reference of a profile joins several
  archived runs (eight copies: seeds 1 to 4 of the four-copy run, seeds 5 to 8 of the one-copy runs), so prepare
  and a smoke job work for eight copies; new commands `hand-over` and `submit` for one pool, also one the tick
  does not use; a pool may set its worker's least training time and margin (`worker.py --margin-seconds`, new).
  The tooling's server_table.json has MIT's pools at eight copies. 110 standard-library tests pass (login node,
  python3 3.10). The hand-over loop's helper was replaced between passes (pass 18 ended 22:16:55 PT): its pack
  step makes packs of each pool's own size (four for Gautschi, eight for pools of the eight-copy profile); with
  MIT's pools still at four copies it asks for the same packs as before (checked on a temporary run folder).
  Nothing changed in the trainer (src/ and scripts/atari_pixel_campaign/ equal between copy 2's commit 08b7c4c8
  and this one).
- 2026-11-18 22:20 PT (steps of 22:18 to 22:20 PT): code copy 3 frozen at the pushed commit 48019fc4 (`maze_campaign/freeze_code.py`;
  source.json copy number 3). Why a new copy and not only the submission tooling: the worker on the server picks
  the profile and its settings from the copy's own card_profiles.json, and copy 2's has no eight-copy profile; the
  trainer is unchanged. Only the eight-copy packs require it (minimum copy number 3); every four-copy pack and
  single keeps its number (1), so the copy-2 jobs that run or wait (UVA, Gautschi, Gilbreth, MIT) still take
  them. The hand-over loop's next tick sends copy 3 to Gautschi and Gilbreth (prepare); MIT got it by hand at
  22:19 PT. MIT check pool `eight-copy-check` added to the table, outside `hand_over.pools` (the tick leaves it
  alone): mit_preemptable (its own cap of 4 cards; the account's 4 Advanced cards are held by this run's two
  running and two waiting four-copy jobs, so a check job there could not start before them), every H200 node but
  node3401-c1 (the nodes only mit_preemptable has were all allocated; a 20-minute job may be preempted there and
  then requeues), one card, 4 cores, 24 GB, 20 minutes, the card guard, the worker's least training time and
  margin 240 s each. The check pack, built by `build_units.py --pack-pending --packs 1 --pack-size 8` (the first
  eight-copy pack of the claim order): `grid_fast-last-layer_eight-copies_step-sizes-0.01-0.03-0.1-0.3_sharing-group-4_intrinsic-weights-0.25-0.5_seed-15`
  (sharing group 4, the costliest form of the fast layer), minimum copy 3; queue-check: every unit in one folder.
  Handed to MIT with `remote_fleet.py hand-over --server mit --pool eight-copy-check --units 1`, and check job
  29615935 submitted with `remote_fleet.py submit --server mit --pool eight-copy-check --jobs 1` (id in
  slurm/submitted_jobids_mit.txt); waiting at 22:20 PT. Its start is read by a second copy of the session's start
  watcher (`mit_start_card_check_eight_copy.txt` in the session's scratchpad, at most 6 hours). The Advanced
  pool keeps its four-copy packs while this is checked.
- 2026-11-18 22:34 PT: check job 29615935 (mit_preemptable) still waited at 22:32 PT with no reason given yet (the scheduler had
  not tested it); nearly every H200 of mit_preemptable was allocated (sinfo: 2 free on node5001, 1 on node5100,
  held for a larger job). A second check pool `eight-copy-check-advanced` (the Advanced pool's options, 20
  minutes, outside `hand_over.pools`) and its job 29617243 submitted at 22:33 PT (id in the MIT ledger): the
  account holds 2 of its 4 Advanced cards, and a 20-minute job can start in a gap the waiting 2-day jobs cannot
  use. Whichever check job starts first takes the check pack; nothing was cancelled. The hand-over loop's pass 19
  (22:21:55 PT) ran the new helper (four-copy packs: 0 asked; eight-copy packs: 0, no eight-copy pool yet) and sent
  code copy 3 to Gautschi and Gilbreth. A third watcher reads every id of the MIT ledger, the later ones included,
  within a minute of its start (`mit_ledger_start_check.sh` in the session's scratchpad, process 896334, at most
  2 days).
- 2026-11-18 23:10 PT, the eight-copy check (steps of 22:52 to 23:08 PT): check job 29617243 (Advanced pool) started 22:52:02 PT on
  node3201, card index 7; read 29 s after its start: AllocTRES equal to ReqTRES (`gres/gpu:h200=1,gres/gpu=1`,
  4 cores, 24 GB), card guard "ok: 1 card(s)". The waiting preemptable check job 29615935 was cancelled at 22:52 PT
  by its ledger id (it was no longer needed). Through the worker path of code copy 3: the worker took the profile
  "eight copies" (NVIDIA H200, 143,771 MB), claimed the check pack (attempt 1), and started train_final.py with
  eight copies, seed 15 for each, the per-copy step sizes 0.01, 0.03, 0.1, 0.3 at weights 0.25 and 0.5 and the
  reset lane budget 32 (run.json says the same); its `worker_start` line records `copies` 8. In the trainer's
  953 s it wrote 200 updates (its "started" event 156 s and its first update 250 s after the worker's start), every number finite, the record files folded to
  update 200 with the game events; it saved at update 200 ("time limit") and ended with exit code 0; the unit is
  back in MIT's pending folder with its run folder (stopped with its state saved; it continues in a real job),
  minimum copy number 3. The tick's copy of its record files reached `slurm/remote_live/mit/`. Readings (records
  copied to `infrastructure/mit_eight_copy_check/job-29617243/` by `read_eight_copy_check.sh` beside them):

  | setting | copies | seconds per<br>iteration | total steps<br>per second<br>(millions) | steps per<br>second per<br>copy<br>(thousands) | hours per<br>million steps<br>per copy | card memory,<br>largest reading (MB) |
  |---|---|---|---|---|---|---|
  | fast last layer, group 4,<br>both records (29617243) | 8 | 3.546 | 0.0370 | 4.62 | 0.0601 | 122,785 |

  Seconds per iteration: the median of 197 gaps between metric lines (3.37 to 3.59 s), the first two updates left
  out; the worker's own median of its one-minute readings was 3.548 s. Host memory: MaxRSS 11.1 GB (sacct). The
  speed equals job 29603853's eight-copy readings (3.546 and 3.554 s at group 16). Verdict: the eight-copy path
  works; MIT switches to eight-copy packs.
- 2026-11-18 23:11 PT, MIT switched to eight-copy packs (steps of 23:09:35 to 23:10:12 PT), with
  `infrastructure/mit_eight_copy_switch/switch_mit_to_eight_copies.py` in one hold of the fleet lock, so no tick ran
  between its steps:
  1. table: `advanced-h200` and the held-back `preemptable-h200` run the profile "eight copies"; the check pools
     `eight-copy-check` and `eight-copy-check-advanced` removed.
  2. the 6 four-copy packs that waited on MIT (sharing group 128, seed 15, weights 0.25 to 8; no job had claimed
     them) taken back to UVA and taken apart into their 24 singles (`build_units.unpack_pending`, by exact name).
     The 2 packs that train on MIT (jobs 29606825 and 29606826, group 128 weight 8 seed 14 at update 4,481 and
     group 16 weight 0.25 seed 14 at 4,113 of 12,207) were not touched; they finish there with four copies.
  3. the two copy-2 Advanced worker jobs that still waited, 29606828 and 29606833, cancelled by their ledger ids
     (each checked: in the MIT ledger, this run's job name, PENDING). Reason: their worker (copy 2) has no
     eight-copy profile, so they could take no unit MIT now holds, and while they waited they filled the pool's
     4 job places, so no copy-3 job would have been submitted before they started (estimates 01:59 and 06:30 PT)
     and ended.
  Then by hand, the steps of one loop pass: mit-share (advanced-h200 at most 4 jobs, 8 units held); pack-top-up
  (four-copy packs: 0 asked; eight-copy packs: 8 wanted, 1 held, 7 made); hand-over-tick: the 7 eight-copy packs
  handed to MIT (sharing group 128 at seed 15 and at seed 16, weights 0.25 and 0.5, 1 and 2, 4 and 8; RND weights
  0.25 to 2 at seeds 16 and 17), so MIT holds 8 eight-copy packs (with the check pack) and the 2 four-copy packs
  that train; 2 copy-3 Advanced worker jobs submitted, 29619724 and 29619725 (2 days, in the MIT ledger; the pool's
  other 2 places are the running copy-2 jobs, whose workers end when their pack is done and find no four-copy
  unit). queue-check: every unit in one queue folder (done 61, packed 136, pending 362, remote_gautschi 8,
  remote_gilbreth 12, remote_mit 10, running 61). The ledger watcher reads 29619724 and 29619725 within a minute of
  their start. The preemptable pool stays held back until its own smoke job 29603828 (four copies, copy 2) has
  passed; it then comes back with the eight-copy profile.
- 2026-11-18 23:14 PT: MIT job 29606826 (copy 2, four-copy pack `..._sharing-group-16_intrinsic-weight-0.25_seed-14`, about update
  4,113) ended in NODE_FAIL on node3301 at 23:11:31 PT, not caused by this run, and the scheduler requeued it under
  the same id (Requeue=1, Restarts=1; waiting, BeginTime 23:13:44 PT). Its pack stays on MIT: the tick puts it back
  in MIT's pending folder, and either 29606826's restarted worker (copy 2) or a copy-3 worker (an H200's eight-copy
  worker continues a started four-copy pack, under its four-copy settings) continues it from its last checkpoint.
  The ledger watcher was replaced at 23:13 PT (process 989111; the first one, 896334, stopped by its process
  number) by a version that reads every start of a job, a requeued job's second start included, so the restart of
  29606826 is read too (`infrastructure/mit_eight_copy_switch/mit_ledger_start_check.sh`).
- 2026-11-18 23:17 PT: the monitor job 6957005 (copy 2) was cancelled by its ledger id between passes and replaced by job 6958072 on copy 3 (48019fc4), so its report has the pack-size table; same options as command.txt item 6.
- 2026-11-19 00:00 PT, user rule: "for RND's run on MIT, if there is another session that also submits to MIT, please hold or
  low priority the jobs of this RND runs on MIT (choose one that cumulate aging), after the other sessions run is
  finished, put them back to normal priority; you don't need to stop existing runs". MIT sets no ACCRUE_ALWAYS
  (`scontrol show config`: PriorityFlags empty), so a held job stops gaining age; a job with a higher nice stays
  eligible and keeps gaining it. Tested on our own waiting job 29603828: nice 3,000,000 gave priority 1 with its age
  term still counting (1,251 then 1,254), and nice 0 gave back 38,072. The hand-over loop's helper now does it at
  every pass (step mit-share, function mit_yield): while any job of account shuzel that is not in this run's MIT
  ledger is running or waiting on mit_normal_gpu or mit_preemptable, every waiting job of this run's MIT ledger
  gets nice 3,000,000 (above the largest priority gap between two jobs of the account there: age 50,000, partition
  600,000, QOS 2,000,000; fair share is the same); when none is left, the jobs it lowered get nice 0 again. Running
  jobs are left alone. State in slurm/mit_yield_state.json; copy of the helper in infrastructure/hand_over_loop/.
  First pass with it, 23:59 PT: no other job, nothing lowered.
- 2026-11-19 00:04 PT: a hold test on our own waiting MIT smoke job 29603828 (preemptable pool): `scontrol hold` set
  its AccrueTime to Unknown, and after `scontrol release` its EligibleTime started again at 03:04 Eastern; its age
  term fell from 1,301 to 0 (about 4.4 hours of age lost on that job). So on MIT a hold resets a waiting job's age,
  which confirms the choice of nice for yielding. The shared rules and skills that said MIT has ACCRUE_ALWAYS and that
  a hold keeps the age are being corrected (user: "correct").
- 2026-11-19 00:06 PT: every running Claude session (7 online; none found on the other portal nodes) was told what this study
  holds on MIT (4 Advanced-pool H200 jobs with 2-day limits, at most until 2026-11-20 20:23 and 23:31 PT; one waiting
  preemptable job), that its waiting MIT jobs step aside by nice for any other job of the account, and that a hold
  on MIT now resets a job's age (user: "tell other sessions so other sessions don't get frustrated by RND usage of MIT").
- 2026-11-19 00:25 PT, user decision on a limit of the nice (a lower priority does not stop our waiting jobs from starting on a
  card that another session's waiting multi-card job cannot use yet, as measured on 2026-09-06 in the
  multi-card-job-priority skill): "held for now, change the rule until such things appear". The nice stays; when a
  waiting job of another session needs two or more cards on MIT's card partitions, the rule changes to holds for this
  run's waiting MIT jobs (each hold costs that job its age). The loop helper now records such jobs
  (slurm/mit_yield_state.json, other_waiting_multi_card_jobs) and writes a WARNING line in its log.
- 2026-11-19 00:51 PT, user: "this low priority apply to Purdue Gautschi as well for RND session". Gautschi sets no PriorityFlags
  (read 00:46 PT; weights age 10,000, QOS 10,000, job size 500, partition 500). Tested on our waiting job 18931522:
  nice 100,000 gave priority 1 with the age term still counting (67 then 72), and nice 0 gave back 1,572. The loop
  helper's gautschi-share step now calls gautschi_yield at every pass: while any job on ai under the group account
  dsimchil that is not in this run's Gautschi ledger runs or waits (other sessions of liu4869, and the group's other
  member long635), every waiting job of the ledger gets nice 100,000; when none is left, the jobs it lowered get
  nice 0. Running jobs are left alone. State in slurm/gautschi_yield_state.json. First pass with it, 00:50 PT: no
  other job, nothing lowered. The first four Gautschi jobs (18909476 to 18909479) ended at their 10-hour limit;
  four new jobs wait (18931498, 18931499, 18931521, 18931522).
- 2026-11-19 01:01 to 01:13 PT : another session (the multi-echelon study, job names m5-setting_...) sent 24
  jobs to MIT (12 cards on mit_normal_gpu, 13 on mit_preemptable) and 4 to Gautschi ai (18931650 to 18931653).
  - Gautschi, pass of 01:06 PT: gautschi_yield lowered this run's 4 waiting jobs (18931498, 18931499, 18931521,
    18931522) to nice 100,000, as designed.
  - MIT, pass of 01:06 PT: the mit-share step stopped with "ValueError: not enough values to unpack (expected 7, got
    6)": the other session's job names are longer than the 150-character name column of the squeue reading, so a
    name ran into the next column. Fixed at about 01:10 PT: the name is now the last column and is read as the rest
    of the line (MIT_SHARE_COMMAND, mit_cards_of). Pass of 01:11 PT: 24 other jobs seen, this run's waiting job
    29603828 lowered to nice 3,000,000 (priority 1, age term still 331), and MIT's limit for new jobs went to 0.
  - The 4 running MIT jobs would have taken MIT's 6 waiting packs one after another until their 2-day limits, so the
    other session's Advanced-pool jobs (the account's 4 Advanced cards are all ours) would have waited up to 2 days.
    The 6 waiting packs were taken back to UVA at 01:12 PT (take-back --server mit --units 6): the 5 that never
    started were taken apart into their 40 single runs (build_units.py --unpack-pending --match eight-copies), and
    the started check pack (200 updates, group 4, weights 0.25 and 0.5, seed 15) waits on UVA with its run folder
    until MIT takes packs again. No running training was stopped: each running MIT job ends after its current pack
    (four-run packs at updates 7,915 and 6,961, eight-run packs at about 1,620, of 12,207). queue-check: every unit in
    one folder.
- 2026-11-19 05:21 PT: **free cards were left unused because this run's waiting jobs filled each account's room in the
  planner.** At 05:03 PT an A40 on jaguar01 was free while sl5nw held 40 of 40 cards (its waiting A40 job 6958359
  waits on QOSMaxGRESPerUser) and yuxinchen held 37 of 40; an RTX 2080 Ti on ai01 had been free since at least
  04:03 PT. The monitor planned no job for either: `scheduler_state` in `scripts/maze_campaign/submit.py`
  subtracted every waiting job of this run from its account's free cards, and since yuxinchen always has waiting
  jobs for busy card types, its budget for cards free now was 0. Slurm's cap counts running jobs only, so a pinned
  job on a free card starts whatever waits. The session's checks from 04:03 PT on also passed over the free RTX
  2080 Ti, which this study lists. By hand (yuxinchen, ledger `slurm/submitted_jobids_grid_yuxinchen.txt`): job
  6958380 pinned to the A40 on jaguar01 (running at once) and job 6958381 pinned to the RTX 2080 Ti on ai01, which
  waits on Priority (another user's job is planned there; Slurm's start estimate 2026-11-20 06:37 PT); it was
  given a 2-hour limit and is cancelled by its id and placed again if it still waits at the next check (ceiling 3).
  Fix: the planner now takes each account's room from its running jobs only; this run's waiting jobs still lower
  only the new waiting jobs (through the spare worker cards). Test
  `test_waiting_jobs_of_the_run_leave_the_room_for_cards_free_now`; 55 tests passed (job 6958383, gates ledger).
  The changed file was copied into frozen copy 3 (equal to the committed file before the change; the trainer is
  untouched) and the monitor was restarted between passes: job 6958072 cancelled by its ledger id at 05:20 PT
  after its 05:19 PT pass, replaced by job 6958392 with the same options (command.txt item 7).
- 2026-11-19 05:24 PT: job 6958381 (yuxinchen, pinned to the RTX 2080 Ti on ai01) still waited on Priority with the 2-hour limit (start estimate 2026-11-20 06:37 PT: the card is planned for another user's job), so it was cancelled by its ledger id (ceiling 3: a pinned job is not left waiting). No other listed card was free; the monitor's waiting jobs by card type cover the units.
- 2026-11-19 06:27 PT: **the live report's check "runs whose first exit came before their first key" read 62 (expected 0), all
  seed 14.** Read from the record files: all 61 seed-14 runs with event records opened a door at update 1, rollout
  step 73 (9,472 steps) and left the start room at rollout step 93 (12,032 steps), while their first counted key
  comes at update 4 (60,032 steps); seeds 13, 15 and 16 have no such run. A door opens only for a player holding a
  key, so one of seed 14's 128 environments picked up the key during the 50 random warm-up rollouts, whose events
  are not counted (definitions), and carried it into training. Update 1 is the same in every configuration of a
  seed (untrained policy, warm-up actions drawn from the seed), so seed 14's first door and first exit are about
  0.01 million steps in every configuration, RND's included. Not a logging or game error: exits with both doors
  locked are still 0. Consequence: in the capped first exit (reported beside the primary measure) seed 14 carries
  no information and is equal across configurations, so paired comparisons are not biased; the mean score over the
  run also includes that key's 100 points in one game of seed 14, in every configuration. The report's check now
  counts such runs apart (a door opened before the first counted key) and lists only the runs with no such door
  (expected 0); 16 report tests passed (job 6958406, gates ledger); report.py copied into frozen copy 3, which the
  monitor runs as a new process at each pass.
- 2026-11-19 07:09 PT: **development document refreshed at the 20 and 50 million marks** (every configuration has at least one
  run past 50 million steps; records read 2026-11-19 07:06 PT): section "Pixel Montezuma: the fast last layer
  against RND" of /p/rlprojects/RND/10_jax_exploration_platform/development_document/RND_JAX_exploration_development_document.pdf
  (generator make_all.py, PDF rebuilt, no line too wide, pages 62 to 65 looked at). The exit-before-key check now
  uses the live report's new function `exits_before_counted_key` (same split as the report; report.py copied into
  copy 3 again), and the prose states that every seed-14 run opens a door and leaves in its first update with a key
  carried from the random warm-up. 12 generator tests and 16 report tests passed (report job 6958440, gates ledger).
- 2026-11-19 09:45 PT: an H100 NVL card free on serval06 while this run's H100 NVL jobs waited on Priority (start estimate 2026-11-20 23:56 PT): jobs 6957716, 6957718, 6958486, 6958545 (sl5nw) and 6958488 (yuxinchen), from their ledgers, given a 2-hour limit (rule: a waiting job on Priority beside free listed cards). The Gautschi yield ended at the 09:40 PT pass (no other job of dsimchil on ai; nothing left lowered), and the 4 Gautschi jobs train with the 8 units handed back to them.
- 2026-11-19 12:14 PT: **MIT's preemptable H200 pool stayed held back on a wrong comparison.** Its smoke job 29603828 was read
  PREEMPTED at 05:33 PT; a preempted job on mit_preemptable is requeued under the same id, and it ran again from
  11:45 PT (node4802, 1 H200 as asked) and completed at 12:00 PT. Two gaps, both fixed:
  1. the hand-over helper read a held pool's verdict once, so the second run was never read. It now reads a verdict
     recorded as preempted again once the job has ended in any other state (scratchpad helper and its copy in
     infrastructure/hand_over_loop/); its 12:11 PT pass read the second run.
  2. that read compared the job's four-copy default updates with the eight-copy reference, since the pool was set
     to eight copies at 01:33 PT, after the job had been sent with four. Every other check passed, and the job's
     own comparison by its frozen code found every field equal to its four-copy reference. remote_fleet.py
     (read_smoke_results) now takes the reference of the profile the job's result names, and the pool's present
     profile only when the result names none; 2 new tests, 29 remote fleet tests passed (job 6958831, gates
     ledger). The marker of the 12:11 PT read was renamed
     (smoke_verdict_read_mit_pool-preemptable-h200_job-29603828_compared-with-the-eight-copy-reference.txt), so the
     next pass reads the verdict once more. While the other session holds the account's 4 preemptable cards, the
     loop's MIT share gives that pool no jobs anyway.
- 2026-11-19 12:17 PT: the second read passed at 12:16 PT (first update: 0 numbers outside the rule, every field equal), and MIT's preemptable pool takes units again, as eight-copy packs. The loop sent its jobs 29678001 and 29678003 (MIT ledger) at 12:16 PT; they were given nice 3,000,000 by hand at once, ahead of the yield step's next pass, since the other session's jobs are on MIT (its waiting preemptable job 29625540 keeps nice 0).
- 2026-11-19 12:28 PT: three pinned jobs that still waited were cancelled by their ledger ids (sl5nw), as the monitor's 12:21 PT
  list asked (ceiling 3): 6958810 (cheetah01, A100 40 GB, waiting on sl5nw's own card cap, which kept the planner
  from placing a yuxinchen job on that node), 6958457 (cheetah04, A100, 260 minutes on Priority) and 6958545
  (serval06, H100 NVL, 160 minutes on Priority). The study's top-up then sent 3 jobs that wait by card type
  (6958902 and 6958903, A100 40 GB; 6958904, A40); the two A100 40 GB jobs, waiting on Priority beside a free
  card on cheetah01 that Slurm holds for another user, got a 2-hour limit. yuxinchen had dropped from 37 to 25
  cards as ten 8-hour worker jobs ended after their units; the monitor's 12:21 PT pass had pinned 9 yuxinchen jobs
  on free cards. The monitor also lists 3 runs whose attempts ran under code copies 2 (08b7c4c8) and 3 (48019fc4):
  src/, train_final.py, scripts/atari_pixel_campaign/ and every training option of the one-copy and four-copy
  profiles are equal between the two copies (card_profiles.json differs only in how the archived reference run is
  named, which profiles may continue others' packs, and the card-memory bound for four copies), so those runs
  trained with the same code and options throughout.
- 2026-11-19 13:50 PT: **the RTX 4000 Ada is a card type of this run again** (user, about 13:45 PT: "i have a new uva
  reservation also include RTX 4000 Ada cards to the allowed list"). It runs one copy at 3.04 s per update (gate 6),
  slower than the RTX A4000 (2.04), which is why it had been left out. The worker and the profiles already accept
  it (one copy, under 60 GB); only the monitor's card list left it out. The monitor job 6958392 was cancelled by its
  ledger id at 13:45 PT between passes and replaced by job 6959048 on copy 3 with the same options plus
  nvidia_rtx_4000_ada_generation (command.txt item 8). Its first pass, 13:46 PT, pinned 9 yuxinchen jobs to free
  RTX 4000 Ada cards (cheetah02, lynx01, lynx02, lynx05; all started) and sent 11 that wait by type. The new
  reservation is sl5nw_170 (jaguar03 with 8 RTX A4500, and puma01 with no card; 2026-11-19 11:35 PT to
  2026-12-10 23:59 PT); the planner reads it live and asks for it last, and since its 12:21 PT pass all 8 RTX A4500
  cards there run this run's workers inside it (jobs 6958851 and 6959030).
- 2026-11-19 15:14 PT: **giving way to another session on MIT and on Gautschi, two gaps closed.** Another session's jobs
  (m5-forecast-inputs, account shuzel on MIT and liu4869 on Gautschi) appeared on both servers.
  1. MIT: the share step had lowered this run's limits at 15:03 PT (Advanced pool 0 jobs, preemptable 1, 2 units
     held), but 6 units were still held on the server: the running pack of job 29619724 (3,589 of 12,207 updates;
     its pack ends in about 8.5 hours, after which it would have taken a held unit and kept an Advanced card for
     another half day) and 5 never-started eight-copy packs. As at 01:12 PT, 4 were taken back at 15:07 PT
     (take-back --server mit --units 4 --from-end) and split into their 32 single runs (build_units.py
     --unpack-pending --match eight-copies); 2 units stay held, the share step's limit. No running job was touched.
  2. Gautschi: the share step counted as "other members" only users other than liu4869, so another session of
     liu4869 (job 18949792, 55 minutes, waiting on the group's 4-card cap) counted as this run's and the pool kept
     4 jobs and 8 units. The step now counts every ai job of the group outside this run's Gautschi ledger as
     another's (scratchpad helper and its copy in infrastructure/hand_over_loop/); its 15:13 PT pass read 1 card of
     others and lowered the pool to 3 jobs and 6 units. This run's 4 running Gautschi jobs end at their limits in
     about 4.5 hours (their packs, at 243 to 781 updates since 09:40 PT, do not finish before), and later jobs wait
     at nice 100,000 behind the other session's job.
  The decision of 13:50 PT that the 2-hour step covers taking packs back was wrong for a server where one of this
  run's jobs runs: that step acts only when none runs there. Plan decision 15 now says to take back by hand the
  never-started units above the share step's limit when that case occurs.
- 2026-11-19 15:51 PT: **development document refreshed at the 100 and 200 million marks** (every configuration has at least one
  seed at 200 million steps; records read 2026-11-19 15:50 PT). New: four generated score tables (RND, and the fast
  last layer with sharing groups 128, 16 and 4), each giving S(B) and the final level over the seeds at the budget
  (2 to 5 of 8 per configuration), sorted by S(B), with the archived baseline's first 200 million steps below a rule;
  the text says they choose nothing until all 8 seeds are in. Generator functions score_row and score_tabular, one
  new test (13 generator tests passed); PDF rebuilt, no line too wide, pages 63 to 67 looked at.
- 2026-11-19 16:08 PT: job 6959153 (pinned to affogato11, still waiting on Priority after 20 minutes) cancelled by its ledger id, as the monitor's 16:05 PT list asked (ceiling 3).
- 2026-11-19 17:29 PT: another session (m5-forecast-inputs) sent about 140 jobs to MIT (59 waiting on mit_normal_gpu, 86 on
  mit_preemptable), and the share step lowered this run's MIT limits to 0 jobs and 0 units on both pools. As plan
  decision 15 says, the one never-started pack held on MIT (sharing group 16, weights 0.25 and 0.5, seed 18) was
  taken back at 17:28 PT and split into its 8 single runs; MIT now holds only the pack job 29619724 trains. This
  run's waiting Advanced-pool job 29673418 (MIT ledger), which had no unit left to train, was cancelled by its id:
  the account had 61 of its 64 places on mit_normal_gpu filled, and the job would have taken one of the other
  session's remaining places. The two waiting preemptable jobs (29678001, 29678003) stay at nice 3,000,000
  (that pool's cap is 448 jobs).
- 2026-11-19 19:49 PT: Gautschi: the other session sent 3 more ai jobs (12 to 13 hours each; 18960498 to 18960500), which wait on
  the group's 4-card cap behind its running job 18949792 and this run's 3 running jobs (18959407, 18959409,
  18959546; the last started on a free card at nice 100,000). The share step lowered this run's pool to 0 jobs and
  0 units. The 5 waiting units held on Gautschi were taken back at 19:48 PT (take-back --server gautschi --units 5
  --from-end): the 3 never-started four-copy packs were split into their 12 single runs (build_units.py
  --unpack-pending --match four-copies), and the 2 started packs (group 4 weight 4 seed 14 at 7,176 updates; RND
  seed 15 at 4,633) wait on UVA with their copied-back checkpoints, to go back to Gautschi when its share allows or
  to a free card of at least 60 GB. Taking the started packs back as well goes beyond plan decision 15's
  "never-started" so that this run's 3 running jobs end after their current packs (7,600 to 8,756 of 12,207
  updates, about 2 to 3 hours) instead of holding 2 cards for another 3 to 5 hours while the other session waits.
  No running job was touched.
- 2026-11-19 20:09 PT: 7 more 8-hour worker jobs ended (59 runs training on UVA instead of 76) and 15 listed cards were free after the monitor's 20:05 PT pass (10 RTX 2080 Ti on ai02, ai03, ai04, ai06 and lynx10; 5 RTX A4500 on jaguar03 in the reservation sl5nw_170). The top-up was run by hand with the monitor's options: 15 cards pinned to them and 2 RTX A4000 jobs that wait.
- 2026-11-19 20:49 PT: more 8-hour worker jobs ended, and 8 listed cards were free after the monitor's 20:45 PT pass: 5 Quadro RTX 6000 on lotus (back in service) and 3 RTX A4500 on jaguar03 in the reservation. The top-up was run by hand with the monitor's options: 6959801 (yuxinchen, 5 cards on lotus) and 6959802 (sl5nw, 3 cards on jaguar03, reservation sl5nw_170) started at once; 6959803 (one Quadro RTX 6000) waits by type.
- 2026-11-19 21:07 PT: the monitor now passes every 10 minutes instead of 20 (monitor.py PASS_SECONDS 600; the hourly pass when
  nothing changes stays). Twice this evening 8-hour worker jobs ended in a batch and their freed listed cards sat
  until the next pass (15 cards after 20:05 PT, 8 after 20:45 PT; both topped up by hand). The test that checked
  the old interval now checks the constant; 21 monitor tests passed (job 6959833, gates ledger; the first run,
  6959806, waited on sl5nw's full cpu-partition cap and was cancelled by its id, then run on nolim). The changed
  file was copied into code copy 3, and the monitor was restarted between passes: job 6959048 cancelled by its
  ledger id at 21:06 PT after its 21:05 PT pass, replaced by job 6959838 with the same options (command.txt item 9).
- 2026-11-19 22:20 PT: a third way this run's waiting jobs kept free cards idle, found at the 22:07 PT check. Two A100 40 GB cards
  on cheetah01 were free while 156 units waited and yuxinchen held 34 of 40 cards, and the top-up sent nothing: the
  planner's room per card type (cards of the type on open nodes, less every job of this run that asks for the type)
  was 0, because sl5nw's two waiting A100 40 GB jobs (6959406, 6959790) counted as taking the two free cards, while
  they wait on sl5nw's own cap (QOSMaxGRESPerUser). Change in scripts/maze_campaign/submit.py plan_jobs: a job pinned
  to cards free now is no longer limited by that room (the node's free cards bound it, and the room still shrinks
  for the jobs that wait). New test test_free_cards_of_a_type_that_waiting_jobs_already_ask_for_still_get_jobs; 44
  planner and monitor tests passed (job 6959889, gates ledger; the first run, 6959888, failed on a mistake in the new
  test itself). The top-up by hand then sent 6959890 (yuxinchen, 2 A100 40 GB on cheetah01, running) and 6959891
  (yuxinchen, 1 A100 80 GB on cheetah04, four copies), which waited on Priority and got a 2-hour limit
  (scontrol update by its ledger id). The changed file was copied into code copy 3, and the monitor was restarted
  between passes: job 6959838 cancelled by its ledger id at 22:17 PT after its 22:17 PT pass, replaced by job
  6959895 with the same options (command.txt item 10); its first pass ran at 22:19 PT.
- 2026-11-19 22:28 PT: job 6959891 (yuxinchen, pinned to cheetah04, one A100 80 GB, four copies) still waited on Priority after its 2-hour limit and one check, and cheetah04 had no free card any more: cancelled by its ledger id (cancel_ledger.py, ceiling 3); it held no unit. Gautschi: the last of RND's three running packs ended and was verified at 22:28 PT (18959407, 18959409, 18959546 all COMPLETED); this run now holds no job and no unit on Gautschi while the other session of the group uses its cards.
- 2026-11-20 00:12 PT: this run's 4 waiting A100 jobs (6958489 and 6959406 and 6959790 under sl5nw, 6958490 under yuxinchen) were
  given nice 5000 (scontrol update by their ledger ids), so that the AntMaze study's 5 waiting A100 jobs (6959920 to
  6959924) start first on the next A100 cards of each account. Why: the plan gives the maze study first claim on UVA
  cards ("its units keep their UVA cards"); its 5 units came back from Gilbreth at 22:44 PT and asked for A100 cards
  (the most capable that fit their 17 GB), but this run's A100 jobs, sent earlier, stood ahead (priority 1,291 to
  4,668 against 777 to 1,182). Nice keeps a job's age on UVA. They go back to nice 0 once no AntMaze job waits for
  an A100 card (ids in the scratchpad file montezuma_a100_jobs_niced_for_antmaze.txt).
- 2026-11-20 04:08 PT: an A100 40 GB card on cheetah01 came free about 04:07 PT. The AntMaze waiting job 6959924 started on it (as intended by the nice of 00:07 PT), and this run's monitor had pinned job 6960094 (yuxinchen) to the same card seconds before; it then waited on Priority and was cancelled by its ledger id (ceiling 3); it held no unit.
- 2026-11-20 04:11 PT: job 6960089 (sl5nw, pinned to lynx10, 2 RTX 2080 Ti), sent by the monitor when sl5nw had room for 2 cards, waited on sl5nw's own cap after the AntMaze job 6959924 started under sl5nw and took that room; since the planner leaves out nodes where a job of this run waits, the 2 free cards on lynx10 got no other job. Cancelled by its ledger id (ceiling 3) and replaced by a top-up by hand: 6960098 (yuxinchen, lynx10, 2 cards).
- 2026-11-20 11:37 PT: no session checked between 04:28 and 11:31 PT. At 11:31 PT the queue held no pending unit (490 runs done;
  49 units training on UVA, 11 on Gilbreth, 2 packs held by the one running Gautschi job). Actions:
  1. the 14 waiting worker jobs on UVA (9 under sl5nw, 5 under yuxinchen; among them the two pinned ones the
     monitor listed, 6960153 on lotus after 410 minutes on sl5nw's own cap and 6960558 on lynx01 after 70 minutes,
     and the A100 jobs that carried nice 5000) were cancelled by their ledger ids, since no unclaimed unit was left
     for them (ceilings 2 and 3);
  2. the end of stage 1 was set by the never-started four-copy pack sharing group 4, weight 8, seed 20, queued on
     Gautschi behind RND's running pack (58%, about 3.2 hours left, then about 7.8 hours for the pack in a second
     job: about 11 hours). It was taken back (remote_fleet.py take-back --server gautschi --match
     sharing-group-4_intrinsic-weight-8_seed-20) and split into its four singles (build_units.py --unpack-pending);
     the top-up by hand started them at once: 3 on Quadro RTX 6000 cards on lotus (job 6960624, sl5nw; about 5.7
     hours at the 1.67 s per update read there) and 1 on an RTX 4000 Ada on lynx05 (job 6960623, yuxinchen; about
     10.3 hours at 3.04 s). Pack size 1 is recorded for these runs, as for every single.
  3. the slowest units left are on RTX 4000 Ada cards (this new single, and the run at update 1,747 with about 8.8
     hours left); when faster listed cards free up, moving them is weighed by memory note
     move-slow-card-units-to-fast-cards-near-the-end.
- 2026-11-20 11:38 PT: the 2 waiting MIT jobs 29678001 and 29678003 (mit_preemptable, nice 3,000,000, no unit held, share 0) were cancelled by their ledger ids with scancel over the open connection, since no unit is left for them (ceiling 2); cancel_ledger.py cancels a remote server's waiting jobs only together with a RUN/STOP file, which would stop the whole sweep.
- 2026-11-20 11:54 PT: correction to the 11:35 PT step. Cancelling every waiting worker job was early: 8-hour worker jobs that reach their limit give their unit back with its checkpoint, and 8 units came back by 11:52 PT. The monitor's 11:48 PT pass placed 7 (ai02, ai03, ai04 one RTX 2080 Ti each, jobs 6960662 to 6960664; jaguar03 4 RTX A4500 in the reservation, 6960665) and the eighth went by hand to the last free RTX A4500 there (6960669). A pinned job sent by hand a minute after that pass (6960666, ai02) found its card taken and was cancelled by its ledger id. Waiting jobs are surplus only once every running unit will finish inside its job.
