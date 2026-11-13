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
- 2026-11-12 18:28 PT: the new monitor job 6894373 (partition `nolim`, 1 processor) waited with reason `Priority`
  under its 20-day limit; its limit was lowered to 1 day (`scontrol update`, id from the monitor ledger). At
  18:29 PT it was still waiting. Until it starts, the session's 20-minute tick does its work (the tick submits
  workers for waiting units itself).
- 2026-11-12 18:47 PT: why the new monitor job 6894373 does not start: it asks for 16 GB of memory, the account's
  fair-share priority is low after today's use (1,215 of the job's 1,301 points), and the scheduler's planned
  start was 2026-11-13 20:39 PT. The earlier monitor job used 130 MB at most (`sacct` MaxRSS of 6887756). Changes
  to the waiting job (id from the monitor ledger): partitions `cpu,nolim` (18:41 PT), memory 2 GB (18:47 PT). The
  submission script now asks for 2 GiB for a monitor job. At 18:50 PT the job was still waiting.
- 2026-11-12 19:07 PT: six worker jobs of this folder waited (four since 18:45 PT, reason `Priority`): they asked
  for RTX 4000 Ada cards, and every RTX 4000 Ada card with free processors was in use (most by this folder's own
  workers), while RTX 2080 Ti cards were free on ai03, ai06 and lynx10. The planner had also pinned one job to
  affogato13, which had 2 free cards but only 2 idle processors (a full node). Changed with `scontrol update`
  (ids checked against this folder's ledgers; the jobs keep their age): 6894925 to 6894928 and 6895892 now ask for
  one RTX 2080 Ti on any node; 6895891 is pinned to lynx10 with an RTX 2080 Ti. Their submission records carry the
  change. At 19:09 PT they were still waiting on `Priority`. The account had 35 of its 40 cards of the `gpu`
  partition in use over all sessions, so at most 5 more can run at once.
  Two planner faults to correct: it pins jobs to nodes whose free processors are only the 2 that a full node
  shows, and its unpinned jobs name one card type instead of leaving the choice between the allowed types open.
- 2026-11-12 19:46 PT: of the 11 worker jobs submitted at 19:43 PT, 8 were unpinned and asked for RTX 4000 Ada
  cards while 16 RTX 2080 Ti cards were free; six of them were switched to RTX 2080 Ti with `scontrol update`
  (ids checked against the validation-block ledger; their submission records carry the change). At 19:48 PT 12
  worker jobs of this folder waited with reason `Priority` although cards were free, and 2 had started: the
  account's fair-share value is low after today's use (0.0012; the other account's is 0.0008, so submitting there
  would not help). The units do not depend on these jobs: running workers take the next waiting unit when theirs
  ends (waiting units went from 14 to 8 in four minutes).
  Correction of the note of 19:07 PT: the planner naming one card type for its unpinned jobs is not a fault by
  itself. It asks for a type at most as often as that type has cards outside reservations, counting this folder's
  own running and waiting jobs; when the RTX 2080 Ti count is used up by our own jobs it turns to the next type.
  The open fault is the other one: pinning to a node whose only idle processors are the 2 a full node shows.
- 2026-11-12 20:41 PT: **why UVA no longer starts this folder's worker jobs.** Three jobs pinned to nodes with
  free RTX 2080 Ti cards and idle processors (6897191 to 6897193, submitted 19:43 PT) had waited an hour with
  reason `Priority`. Read at 20:41 PT: 3,015 waiting jobs in the cluster, 1,324 of them with a higher priority
  than ours (one user alone has 2,560 waiting); our jobs' priority is 1,462, of which the fair-share part is
  about 1,215 after today's use. The scheduler tests 50 jobs in its main pass and at most 500 in a backfill pass,
  in priority order, so jobs this far down are tested rarely, whatever is free. Jobs of the reservation still
  start at once (the test jobs on puma01 do), but the reservation's cards on jaguar03 are all held by another
  session's job. The account had 9 cards in use over all sessions, far below its cap of 40.
  Effect: the unit of batch 4 on PointMaze Large with the step cost (slower predictor with held divisor) stopped
  with its state saved at window 198 of 256 when its worker reached the 2-hour limit, and no worker took it.
  What was done (20:42 PT): Gilbreth's hand-over limit for this folder raised from 0 to 2; the unit was handed
  over with its checkpoint (job 12347874). The other unit of that case continues there the same way if its UVA
  worker ends. The waiting UVA jobs stay queued and gain age.
- 2026-11-12 22:10 to 22:19 PT: **placement changed after two messages from the user.**
  1. The user pointed to free UVA cards (4 RTX 5080, 12 Quadro RTX 4000) and asked to use such cards or better
     ones. Read at 22:14 PT: those cards are free, but their nodes have no usable processor left
     (`CPUAlloc` equals `CPUEfctv` on nekomata01 and 02, adriatic02, 05 and 06, jaguar05): on the RTX 5080 nodes
     other users' processor-only jobs hold them; on adriatic05 a job of the yuxinchen account holds all 30, on
     adriatic06 jobs of sl5nw hold all 30, on adriatic02 sl5nw holds 20 of 30 (processor-only work of other
     sessions). Elsewhere memory is the limit: most nodes with free cards have under 3 GiB that a job may still
     take. With every suitable card type allowed, the planner found 2 cards a worker could start on at once
     (ai01, RTX 2080 Ti; lynx04, GTX 1080 Ti); both jobs started within 30 seconds (6898887, 6898888).
  2. The user then proposed Gilbreth by default and UVA only where something starts at once; agreed and applied.
  What was done: the worker type limit of two card types was dropped (every type from GTX 1080 Ti up, except
  H100 NVL, which these small units leave to work that needs it); 12 waiting units had been taken back from
  Gilbreth for UVA and went to Gilbreth again at 22:18 PT; all this folder's waiting UVA jobs were cancelled by
  their ledger ids (the stuck three-card job 6897193, three and then ten unpinned jobs of this evening; also the
  baseline folder's stuck three-card job 6891971 through that folder's ledger). The submission script has a new
  option `--only-cards-free-now` (commit below, 98 tests pass). During the first take-back the five-minute loop
  handed the units over again within a second, before the limit was lowered; with the corrected fleet that is
  harmless (no unit had two jobs), but the limit is now lowered BEFORE a take-back.
- 2026-11-12 22:21 PT: this folder's monitor job was replaced so that it no longer submits UVA workers by itself
  (its top-ups sent unpinned jobs that wait in UVA's queue, which the placement of 22:18 PT rules out): 6894373
  cancelled by its id from the monitor ledger, 6898944 submitted without a top-up phase (2 GiB, partitions
  `cpu,nolim`, limit 1 day). It was waiting on priority at 22:21 PT; until it runs, the session's 20-minute tick
  covers it. UVA workers are now submitted only by the tick, with `--only-cards-free-now`.
- 2026-11-12 22:50 PT: first use of the new placement for a batch (batch 7, 11 units). The planner found one UVA
  card that could start a worker (lynx04, GTX 1080 Ti); its job 6899067 was still waiting on priority after two
  minutes, so it was cancelled by its ledger id and its unit went to Gilbreth with the others. "Can start at
  once" is therefore checked after the fact: a pinned UVA job that has not started two minutes after submission
  is cancelled and its unit handed over.
- 2026-11-13 01:27 PT: **this folder's code was frozen anew at commit bed45842** (it holds the option
  `reward_divisor=held_after_first_arrival`, commit eab85eba, and the planner corrections of 2026-11-12). Earlier
  copies: 2b84cab1, d7054945. Five units of batch 9 (PointMaze Large with the step cost, discovery check) were
  running on Gilbreth under the copy d7054945 at that time; the hand-over limit for Gilbreth was set to 0 so that
  no unit of the new option can be taken there by a job of the old copy (a worker does not check which copy a
  unit needs). If one of those five is continued by a later job, it will have attempts under two copies; the
  trainer gives the same results for the old options under both (the golden and bonus tests pass on the new code).
- 2026-11-13 01:28 PT: the 12 units of the new arm (batch 8, the divisor held from the first arrival) were built
  and stay on UVA. `submit.py --only-cards-free-now` found 7 cards that could start a worker; 6 jobs started
  within two minutes (6900129 to 6900132, 6900134, 6900135: two RTX 4000 Ada, two RTX 2080 Ti, two GTX 1080 Ti);
  the seventh (6900133, a Quadro RTX 4000 on adriatic02) was still waiting after two minutes and was cancelled by
  its ledger id. The six workers took the six PointMaze Medium units first.
- 2026-11-13 02:58 PT: batch 10's eight PointMaze units (fast last layer with the first-arrival hold) were built and
  the Gilbreth hand-over limit, pinned to 0 since 01:28 PT, was opened again (9: the one unit there and the eight).
  One job of the old code copy d7054945 is left on Gilbreth (12352113, on an A10, started 01:11 PT, about 1.5
  hours of its own unit to go). Eight jobs were submitted (12352872 to 12352879); one started within a minute and
  the scheduler's planned starts of the other seven were 0.7 to 1.7 hours away, with the account at 13 running
  and 14 waiting idle-card jobs against its limit of 24. At 03:00 PT I lowered the limit to 5 and took four
  waiting units back to UVA, where three worker jobs on four cards that were free started within two minutes
  (6900951 on two A40 cards, 6900952 on an RTX 2080 Ti, 6900954 on a GTX 1080 Ti). **To watch:** a unit of the new
  option must not be waiting on Gilbreth when job 12352113 finishes its own unit (about 04:40 PT), because that
  job would take it and fail it; a unit still waiting there at about 04:20 PT is taken back to UVA
  (`remote_fleet.py take-back`, after lowering the limit).
- 2026-11-13 03:08 PT: three units of the second validation block on PointMaze Large with the step cost were built
  and handed to Gilbreth (limit 7), where seven jobs of this folder were already waiting and two were running;
  no new job was needed. Two of the three are units of old options, which the old-copy job 12352113 could also
  run; the third (first-arrival hold) is of the new option and falls under the watch item above.
- 2026-11-13 03:53 PT: **this folder's code was frozen anew at commit 2017e2b8** (it holds the option
  `fast_last_layer_until_first_arrival`, commit 3abfe847; before: bed45842). What the new copy changes for
  running units (from the worker that added the option, who tested both):
  1. new code continues a checkpoint written by an older copy, for units that leave the new option off (a bonus
     option missing from a saved record is read as its default);
  2. the reverse is refused: a job of an older copy that picks up a unit already checkpointed by the new copy
     stops at once with "the checkpoint belongs to a different unit: bonus_options" if the unit sets any bonus
     option, and leaves the checkpoint as it was. So jobs of older copies must not take over such units, and must
     not find units of the new option waiting.
  Jobs of older copies alive at 03:53 PT: on Gilbreth 12352113 (copy d7054945) and 12352874 to 12352876 (copy
  bed45842); on UVA 6900951, 6900952, 6900954 (copy bed45842, all their cards busy until about 04:48 PT at the
  earliest, limit 05:00 PT). The Gilbreth hand-over limit is pinned to 0 until the four Gilbreth jobs have ended.
- 2026-11-13 03:53 PT: batch 11's first seven units were built and stayed on UVA: `submit.py
  --only-cards-free-now` placed five jobs on seven cards (6901667 on an A100 40 GB, 6901668 and 6901669 on two RTX
  5080 each, 6901670 and 6901671 on an RTX 4000 Ada each); all five started within two minutes and took the seven
  units, so no unit of the new option is waiting anywhere.
- 2026-11-13 04:09 PT: seven more units of batch 11 (the arm with RND's running divisor and the arm with the divisor
  held after 1/16 of the run) were built and stayed on UVA: three jobs on seven free cards (6901863 on two RTX 5080,
  6901864 on one RTX 4000 Ada, 6901865 on four RTX 4000 Ada) all started within two minutes and took them. No unit
  is waiting. The Gilbreth limit stays at 0 (jobs 12352113, 12352875 and 12352876 of older code copies were still
  running there at 04:06 PT).
- 2026-11-13 04:46 PT: no job of an older code copy is left on Gilbreth for this folder (12352113, 12352875 and
  12352876 have ended with their units complete); the hand-over limit is open again at 8 units at once. Three UVA
  worker jobs of the copy bed45842 run until 05:00 PT at the latest (6900951, 6900952, 6900954); the units they
  hold use options that copy knows, and newer-copy jobs can continue them from their checkpoints.
- 2026-11-13 05:07 PT: batch 12's eight units were built; six went to Gilbreth (jobs 12353238 to 12353243; four
  started within two minutes, two were waiting) and two stayed on UVA, where two jobs on free cards started within
  two minutes (6902810 on an A100 40 GB, 6902811 on an RTX 5080). Batch 10's two units on PointMaze Large with the
  step cost left UVA when their workers reached the two-hour limit at 05:00 PT and continued on Gilbreth from
  their checkpoints (jobs 12353198, 12353199).
- 2026-11-13 05:42 PT: **this folder's code was frozen anew at commit 42b327e7, its first numbered code copy (copy
  number 1).** It holds the option `predictor_divisor_after_first_arrival` (commit 9aab504b) and the new rule of
  the queue (commit 42b327e7): every unit names the lowest copy number that may run it, a worker never takes a
  unit above its own number, and a claim raises the unit's number to the worker's (README of
  `scripts/maze_campaign`, section "Code copies"). Workers started before this freeze do not know the rule. Three
  such jobs were still running, each on a unit of batch 12 with 35 minutes done: 12353240 and 12353241 on
  Gilbreth and 6902810 on UVA. I cancelled the three by their ledger ids at 05:42 PT so that no worker without
  the rule is left; their units went back to the queue and three jobs of the new copy were submitted for them on
  Gilbreth at 05:43 PT (the new copy continues checkpoints of the older one). From now on a freeze, a build and a
  hand-over need no waiting for older jobs, as long as every live worker was started at or after this freeze.
- 2026-11-13 05:43 PT: batch 13's thirteen units were built (minimum copy number 1). Nine went to Gilbreth (limit
  12 with the three units already there; twelve jobs, of which seven were running and five waiting two minutes
  later) and four stayed on UVA, where three jobs on four free cards started within two minutes (6903416 on an
  A100 40 GB, 6903417 on two RTX 5080, 6903418 on one RTX 5080).
- 2026-11-13 05:46 PT: five units of batch 13 were still waiting on Gilbreth three minutes after the hand-over
  (five jobs waiting on priority). I lowered the limit to 7, took the five back, and three UVA jobs on five free
  cards started within two minutes and took them (6903458 on an RTX 5080, 6903459 on three A40, 6903460 on one
  A40). The five waiting Gilbreth jobs stay in the queue; each ends at once when it starts and finds no unit.
