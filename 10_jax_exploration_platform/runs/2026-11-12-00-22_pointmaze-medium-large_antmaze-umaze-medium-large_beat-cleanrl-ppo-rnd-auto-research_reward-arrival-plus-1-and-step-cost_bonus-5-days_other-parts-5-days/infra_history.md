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
