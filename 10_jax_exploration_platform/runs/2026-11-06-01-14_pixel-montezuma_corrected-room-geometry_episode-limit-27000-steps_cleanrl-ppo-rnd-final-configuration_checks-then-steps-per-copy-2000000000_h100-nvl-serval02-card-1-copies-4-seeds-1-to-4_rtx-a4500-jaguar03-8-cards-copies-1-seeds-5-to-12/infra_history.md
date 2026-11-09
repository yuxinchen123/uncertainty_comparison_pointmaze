# Infrastructure history

Times are Pacific, converted from machines running Eastern.

## 2026-11-06 01:21 PT: confirmation units ran in profile mode (my generator)

`make_unit.py` has no `--timing-rounds` / `--updates-per-round` options, so both confirmation units
measured five single-update rounds after warm-up instead of eleven rounds of twenty updates. The
rounds agree closely (A4500 1.896 to 1.955 s; H100 four copies 2.651 to 2.686 s), so the result is
used as measured.

## 2026-11-06 01:31 PT: H100 four-copy paired screen out of memory (expected limit)

The idea-51 screen at four copies holds the reference and candidate programs together (about 40 GB
each) and asked for 37.3 GiB more, above the 85% allocator limit. Paired screens on the H100 stay at
one or two copies, as in the earlier campaign; four-copy speed is measured one configuration at a time.

## 2026-11-06 01:48 PT: runner start lines are written before the card-1 lock is taken

The serval02 runner logs `start <unit>` before `flock` returns, so its start time can be earlier than
the unit's own `start` line. The unit logs show the card-1 units ran one at a time.

## 2026-11-06 02:00 PT: profiler traces committed and removed

`git add` of the run folder took the two profile units' traces (about 230 MB, largest file 100 MB) into
commit 86608cf5, which was pushed. The commit was replaced within two minutes by f1c73e78 without
them (push with `--force-with-lease` against 86608cf5 only), and `.gitignore` now excludes
`**/trace/plugins/`, `*.xplane.pb` and `**/checkpoint.pkl`. The traces remain on disk.

## 2026-11-06 04:40 PT: three H100 copies explore few rooms; checked, not a packing defect

At update 3,087 the four packed H100 copies had visited 15, 2, 2 and 3 rooms (seeds 1 to 4). The
single-copy A4500 seeds had visited 4 to 16 rooms by the same update. Checked whether the packing
treats copies 2 to 4 differently:

- The final H100 run repeats its own learning check exactly: mean intrinsic reward at updates 1, 50,
  100 and 160 is identical for every copy.
- Each H100 copy equals the A4500 single-copy run of the same seed at update 1 (for example,
  seed 2: 79.4050 on both). After that they drift apart, as runs on different cards do. By update 160
  the gap is under 0.2 for seed 2 and under 1.6 for every seed.
- In the 300-update A4500 learning check, seeds 2 to 4 had also reached only 1 or 2 rooms.

So the low counts belong to the seeds, not to the packed run. No action taken.

## 2026-11-06 22:33 PT: hourly checks from 17:33 to 21:33 PT did not run; no interruption found
The monitoring session did not run its checks for five hours. Read afterwards: all eight A4500 jobs had run
throughout (20 h 32 min elapsed), each copy had written 18 checkpoints in the six hours since 16:33 PT (one every
20 minutes), and the H100 supervisor log shows no restart or card yield since 02:17 PT. Updates per hour over the
six hours: A4500 2,456 to 2,631 per copy, H100 1,650. No action taken.

## 2026-11-07 01:45 PT: H100 copies slowed by about 13% from 00:52 PT; slow writes to network storage from serval02
Updates per hour on the H100 fell from about 1,650 to 1,511 (00:33 to 01:33 PT). The time of a normal update did not
change (median 2.17 s before and after), and game ends per update went down, not up. The loss came from 32 pauses of
5 to 40 s in the 45 minutes from 00:52 PT (378 s in all): about one a minute, when the trainer rewrites
`progress.json` (every 60 s, blocking), and the longest at the checkpoint writes. Creating, renaming and removing one
small file in the run folder from serval02 took 19.5 s, 4.4 s and 2.0 s (network storage `corezfs02:/p/rlprojects`).
The A4500 seeds on jaguar03 had one such pause each in the same 45 minutes (7 s and 18 s for seeds 5 and 10), so the
storage was slow for both machines, much more so from serval02. No other process is on card 1; the other user's
processor jobs on serval02 started at 20:21 PT and earlier, before the slowdown. No action taken: the trainer cannot
move its writes without a restart and a code change. If it persists for several hours, writing progress and
checkpoints to serval02's local disk and copying them to the run folder will be considered.

## 2026-11-07 02:33 PT: the slow writes from serval02 ended at about 01:50 PT
From 01:48 PT the H100 copies ran at 2.09 to 2.17 s per update again, with 2 pauses over 5 s (21 s in all) in the 45
minutes to 02:33 PT, against 34 (425 s) in the 45 minutes before. Creating, renaming and removing one small file in
the run folder from serval02 took 0.05 to 0.07 s (19.5 s at the worst, at 01:39 PT). The episode of slow storage, about
00:52 to 01:50 PT, cost the H100 copies about 7 minutes. No action was needed.

## 2026-11-07 19:57 PT: seeds 7, 8, 10 and 12 paused to free four jaguar03 cards for the user
The user asked for four of the eight jaguar03 cards. The four seeds furthest behind (7, 8, 10, 12; updates 104,507 to
105,922 of 122,070) were stopped with `scancel --full --signal=TERM` on their own ids (6832262, 6832263, 6832265,
6832267). The signal also reached the job shell, the job ended at once and Slurm stopped the trainers before they could
write their stop checkpoint, so each seed resumes from its 19:48 PT interval checkpoint (updates 105,521, 104,406,
104,114, 104,209), losing about 400 updates (9 minutes) each; the trainer cuts `metrics.jsonl` back to the checkpoint
on resume. Next time, signal only the trainer process (or `--batch` without `--full` after checking the shell waits)
so it writes the stop checkpoint. No comparable card elsewhere was free (jaguar01 A40s: no free processors; jaguar06:
no free memory), so each paused seed is queued on jaguar03 to start when a running seed ends: seed 7 job 6837907 after
seed 11 (6832266), seed 8 job 6837908 after seed 9 (6832264), seed 10 job 6837909 after seed 6 (6832261), seed 12 job
6837910 after seed 5 (6832260). They continue on the same card type (RTX A4500).

## 2026-11-07 20:15 PT: the four paused seeds moved to RTX 5080s; a resume defect found and fixed
The user allowed any card that fits, so the queued jaguar03 jobs 6837907 to 6837910 were cancelled (own ids) and the
seeds were sent to the four free RTX 5080s on nekomata01 and nekomata02 (4 processors and 24 GB per job; one copy
needs 8.6 GB of card memory and 6 GB of host memory). The first jobs (6837911 to 6837914) ended at once with "this run
folder belongs to a different configuration" on all four attempts. Cause: `train_final.py` recorded as the game's
commit the HEAD of the RND repository enclosing the environment folder (`git rev-parse HEAD` inside the installed
`jaxtari` package), and an unrelated commit to main (f0ef45bc, another session) had moved it from cef30739. Every
resume of every unit, jaguar03 and serval02 included, would have been refused. Fixed in 759c04a7 (pushed to main): the
commit counts only when git tracks the package's files, and older records match when the package path and asset
pack agree and no installed package file changed after the folder was created (checked: none did). The integration
worktree was moved to main (f0ef45bc touched only ignore rules and grid-study figures) before the commit. Resubmitted
as jobs 6837934 (seed 7) and 6837935 (seed 8) on nekomata01, 6837936 (seed 10) and 6837937 (seed 12) on nekomata02;
all four logged "resumed" at their 19:48 PT checkpoints at 20:12 PT and run at 1.23 to 1.26 s per update (the A4500
took 1.37 to 1.43). These seeds' last 16,000 to 18,000 updates are on a different card class; the throughput table
keeps the A4500 and RTX 5080 segments in separate rows.

## 2026-11-08 02:30 PT: the eight one-copy seeds completed
Seeds 11, 9, 6, 5 (RTX A4500, jaguar03) finished at 01:06, 01:19, 01:38 and 01:40 PT; seeds 7, 8, 10, 12 (RTX 5080,
nekomata01 and 02) at 01:54, 02:19, 02:28 and 02:30 PT. Every job ended COMPLETED, every progress record says completed
at update 122,070, every last event is "completed" and every metric record has exactly 122,070 lines. The H100 copies
(seeds 1 to 4) continue on serval02 card 1, about 2026-11-09 02:30 PT. Results of the eight seeds:
`analysis/2026-11-08-02-36_final-training-results_rtx-a4500-and-rtx-5080-seeds-5-to-12/` (Slurm job 6839300 on
nolim; the same job queued on cpu, 6839298, was cancelled by its id when the nolim one started).

## 2026-11-09 02:34 PT: the four H100 copies completed; all 12 copies done
Seeds 1 to 4 (four copies packed in one process on serval02 card 1, H100 NVL) finished at 02:34 PT after 3 days 0 hours
of training with no restart: the supervisor logged "trainer exit 0" and "training complete", the progress record says
completed at update 122,070, the last event is "completed" (216 checkpoints in all) and the metric record has exactly
122,070 lines. Over the run the four copies took 2.01 to 2.13 s per update. Card 1 held only this trainer at every
check; the supervisor removed its pid files on exit and card 1 is free. Results of all 12 copies:
`analysis/2026-11-09-02-46_final-training-results_all-12-copies_h100-nvl-rtx-a4500-rtx-5080/` (Slurm job 6847935 on
nolim).
