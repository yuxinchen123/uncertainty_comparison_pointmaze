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
