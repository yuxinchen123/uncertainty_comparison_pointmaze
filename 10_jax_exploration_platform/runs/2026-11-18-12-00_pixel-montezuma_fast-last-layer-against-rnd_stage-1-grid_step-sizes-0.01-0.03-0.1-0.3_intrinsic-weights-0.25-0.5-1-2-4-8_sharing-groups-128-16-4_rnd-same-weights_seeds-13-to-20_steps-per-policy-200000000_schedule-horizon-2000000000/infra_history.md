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
