# Infrastructure history

Times are Pacific.

- 2026-11-12: folder created with the program and the log; no job submitted from this folder yet.
- 2026-11-12 13:40 PT: worker 6887738 (2 cards pinned to ai02, 4 processors, 8 GiB) could not start after worker 6887747 took 4 GiB of that node's 7.5 GiB of unallocated memory; cancelled by its ledger id, never started. The monitor's top-up places its two cards again.
- 2026-11-12 13:42 PT: five screen workers whose planned start was days away (four asking for an RTX 5080,
  one RTX 2080 Ti under the second account) were cancelled by ledger id, never started, and seven workers were
  submitted with 2 processors, 3 GiB (measured 1.7 GiB after 7 minutes of a 192-policy unit) and 2 hours. Jobs
  under the sl5nw account on RTX 2080 Ti and RTX 4000 Ada nodes start within a minute; those under the second
  account wait. Four units ran by 13:44 PT.
- 2026-11-12 13:45 PT: Gilbreth prepared for this folder (server table `infrastructure/remote_fleet_servers.json`,
  code copy of commit 2b84cab1, the environment built for the baseline folder). Up to 6 units at once on `standby`
  cards (A100 80 GB, A30, A10). The first unit goes there once one UVA unit of this folder is done, because its
  record format is compared with that unit's; the rest follow after that check.
  The tick's listing "Jobs of this run on the servers" also shows the baseline folder's Gilbreth jobs (both folders
  give their jobs the same name prefix); decisions and cancellations read this folder's own ledger only.
- 2026-11-12 15:06 PT: fault in the remote fleet, found and corrected. Both run folders (the baseline and the auto
  research) named their Gilbreth jobs with the same first part, and a folder takes jobs with its name for its own.
  So each folder wrote the other's job ids into its Gilbreth ledger, and the auto research folder submitted no job
  for its three validation units handed over at 14:45 PT (it counted the baseline's eight jobs as its own); they
  waited 21 minutes. Nothing was cancelled or lost. Correction: the name now carries eight characters taken from
  the run folder's name (`job_prefix` in `scripts/maze_campaign/remote_fleet.py`, with a test); the ledgers were
  set right by the working folder of each job as the scheduler records it (12345010 belongs to the auto research
  folder, every other id to the baseline); the earlier files are kept beside them as
  `submitted_jobids_gilbreth.txt.before_ledger_correction_2026-11-12-15-08`. The auto research folder then
  submitted jobs 12346016 to 12346018 for its units.
- 2026-11-12 15:26 PT: four screen workers asking for an A100 (planned start 2026-11-13 or unknown) cancelled by ledger id, never started; workers for the validation units submitted on RTX 2080 Ti, RTX 4000 Ada and GTX 1080 Ti with a 2-hour limit.
- 2026-11-12 15:27 PT: the first submission of batch 2's UVA workers stopped after one job: the planner pinned a
  GTX 1080 Ti request to affogato13, which holds RTX 4000 Ada cards only, and the scheduler refused it ("Requested
  node configuration is not available"). The planner's table of card types per node is wrong for that node; the
  workers were submitted again with RTX 2080 Ti and RTX 4000 Ada only (6890372 to 6890472). To correct: the
  source of that table.
- 2026-11-12 16:36 PT: Gilbreth's standby queue slowed for this account (6 jobs running, 17 waiting with planned
  starts 2 to 6 hours away) while UVA starts workers under the sl5nw account within a minute. This folder stopped
  handing units to Gilbreth (`units_at_once` 0 in the server table), the units waiting there without a job were
  taken back to UVA's queue, this folder's waiting Gilbreth jobs were cancelled by exact id from its Gilbreth
  ledger, and UVA workers were submitted under the sl5nw account only (new option `--accounts` of `submit.py`).
  Units already running on Gilbreth finish there.
- 2026-11-12 18:15 PT: **the folder's monitor job (6887756) ended in state FAILED**, read from its log at the
  18:20 PT tick. At its pass it planned one worker under the yuxinchen account on lynx02 with card type GTX 1080 Ti;
  the scheduler refused the request ("Requested node configuration is not available"), because lynx02 now holds
  RTX 4000 Ada cards, and the uncaught error ended the job. The shared resource file
  (`/p/rlprojects/.claude/skills/uva-submit-gpu-sweep/school_compute_resource/server_introduction.json`) is out
  of date for four nodes whose cards were replaced: affogato13, lynx01, lynx02 and lynx05 all hold RTX 4000 Ada
  cards now. Effect: two discovery-sweep units, whose workers had reached their 2-hour limit, waited without a
  worker from about 18:15 to 18:22 PT; they continue from their checkpoints.
  What was done: workers 6894353 and 6894354 submitted at 18:22 PT (both started within three minutes); the
  planner corrected (commit 28c6681f, three tests, 97 pass): a node's card type is read from the scheduler, and a
  refused job is reported and skipped. A new monitor job (6894373) submitted at 18:25 PT. It runs this folder's
  frozen code copy, which does not have the correction, so its card types are limited to RTX 2080 Ti and RTX 4000
  Ada, whose node lists in the resource file are right. The folder's code is frozen anew, with the correction,
  when no unit of it is part-way through.
