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
