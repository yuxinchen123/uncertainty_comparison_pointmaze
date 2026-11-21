# Infrastructure history of stage 2

- 2026-11-20 18:17 PT: folder created; code frozen at 7a69c78a (copy 1); card-speed table written from stage 1 (submit.py --card-speeds-from); 36 units built from study_descriptions/stage-2_unused-seeds_....json.
- 2026-11-20 18:30 PT: submitted on UVA (command.txt item 4): 10 worker jobs, all started at once on cards free now, 36 cards for 36
  units: 1 A100 40 GB (cheetah01), 7 Quadro RTX 6000 (lotus), 28 RTX 2080 Ti (affogato11, ai01 to ai04, ai06, lynx10,
  cheetah03); sl5nw 16 cards in 5 jobs, yuxinchen 20 cards in 5 jobs (ids in slurm/submitted_jobids_unused-seeds_*.txt).
  The reservation's RTX A4500 cards were not reached (asked last, user rule 2026-11-19). All 36 runs past update 100
  within about 10 minutes, 2.0 to 2.2 s per update on the RTX 2080 Ti cards.
- 2026-11-20 18:30 PT: the monitor job 6961775 (command.txt item 5) waited on sl5nw's full nolim processor cap (79 of 80, another
  session's solver jobs); moved by its ledger id to the gpu partition, pinned to the card-free node affogato14
  (cpu_only_nodes.py allowlist, --gpus-per-node 0), time limit lowered to 4 days (the gpu partition's maximum; stage 2
  needs about 1.5 days). First pass 18:29 PT.
