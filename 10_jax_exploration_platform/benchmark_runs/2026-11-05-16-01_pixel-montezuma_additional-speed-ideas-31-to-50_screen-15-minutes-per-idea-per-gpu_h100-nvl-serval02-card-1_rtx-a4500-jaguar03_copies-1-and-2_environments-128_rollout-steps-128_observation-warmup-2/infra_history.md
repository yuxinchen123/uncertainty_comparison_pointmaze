# Infrastructure history

Times are Pacific, converted from machines running Eastern.

## 2026-11-05 16:04 PT: first launch failed on an interactive JAXtari question

All ten A4500 jobs (6830716 to 6830725) ended in 5 s: JAXtari asked whether to download a newer sprite pack and read
standard input. Earlier campaign jobs set `JAXTARI_SKIP_SPRITE_UPDATE=1` (keep the pinned sprites); the unit scripts now
do too, and the jobs were resubmitted as 6830730 to 6830739. On serval05 the queue's terminal kept the first unit waiting
on that question (process stopped on terminal input); its own Python process 1136283 was stopped by id so the queue
could continue. The first unit on each H100 (the reference profile) was requeued.

## 2026-11-05 16:09 PT: three A4500 units ran out of the card's 20 GB

The two-copy profile held the rollout, prepared batch and starting state at once while timing phases; the window
matrix products of ideas 31 and 32 copied all 4096 rows of a minibatch at once (6.7 GB). Fixes in commit fa84a6bf:
the profile keeps only the arrays one phase needs, and the window products use chunks of at most 1024 rows. The units
were rerun as `_attempt-2` folders (jobs 6830750 to 6830757).

## 2026-11-05 16:10 PT: the one-update check rejects every float32 summation-order change

Ideas 31, 33 and 34 failed the existing one-update check by 1.2e-5 to 4.0e-5 against its 1e-5 absolute limit, on
predictor dense and actor weights, after 16 Adam steps. Adam divides each step by the gradient's running size, so
elements with gradients near zero turn float32 rounding differences into parameter differences. For settings that
only reorder float32 sums inside the networks, a failed one-update check is now followed by a gradient check: the
gradients of the CleanRL minibatch loss of both variants on one identical supplied minibatch must agree at absolute
1e-5 and relative 1e-4. The one-update error is recorded with every result.

## 2026-11-05 16:13 PT: serval02 runner log shows rc=0 for every unit

The runner wrote `$?` after a command substitution, so its log line always shows 0. The unit's own `unit.log` holds
the true exit code. The second runner (pass 2) uses the corrected script.

## 2026-11-05 16:38 PT: superseded first attempts and several serval02 runners

Units whose first attempt ran out of memory or used the old code keep their folders with a `progress.json` but no
`result.json` (A4500 ideas 31, 32 at two copies; idea 32 at one copy; idea 35 at two copies; the A4500 two-copy
reference profile); their reruns are the `_attempt-2` / `_attempt-3_allocator-cuda-async-fraction-0.85` folders.
serval02 card 1 is served by up to four runner processes (passes 1 to 4, one list each); each takes the card-1 lock
for one unit at a time, so units on the card never overlap. Every finished screen ended in under 15 minutes
(longest 822 s).

## 2026-11-05 19:29 and 19:49 PT: H100 eight- and sixteen-copy packing runs out of memory (expected)

`packing-final-configuration` at eight copies (serval05) and sixteen copies (serval02 card 1) stopped with
RESOURCE_EXHAUSTED when the training iteration asked for 74.65 GiB and 88.26 GiB, above the 85% allocator limit. These
mark the memory limit of the packing measurement (four copies fit at 40.6 GB), not an infrastructure fault.

## 2026-11-05 20:05 PT: confirmation runs found no time for a round (my estimate)

Both confirmation runs (A4500 job 6831378, H100 serval02 pass 12) ended without a timed round: before the first round
the budget check assumed 60 s per update, so a 20-update round (1200 s) never fit. The check now uses the measured
warm-up update. Failed folders kept with the suffix `_failed-attempt-1_empty-round-estimate`; reruns use the original names.

## 2026-11-05 21:05 PT: all launched work finished; stale runner record removed

No Slurm job from `slurm/submitted_jobids.txt` is queued or running, no unit is pending or running in the serval05
queue, and no campaign process remains on serval02 or serval05 (serval05 card at 0 MiB). The first serval02 runner
(process 1143116, pass 1) finished its units at 17:02 PT but never removed `logs/serval02_runner.pid`: its script was
edited and restored while it ran (16:13 PT), so bash did not reach its last lines. The process no longer exists; the
stale file was removed. The serval05 queue runner (tmux session serval05_queue, shared by other sessions) stays up,
idle. The 20-minute session monitor was stopped: the final training waits on the user's decision about the
environment (plan section 197).

## 2026-11-06 01:14 PT: the stuck game was a broken sprite pack; work continues in a new run folder

The probes in `probes/` (`game_version_random_play.py`, `stuck_state_anatomy.py`, `conveyor_left_end.py`,
`render_frames.py`, and the frames in `render_frames_original_pack/` and `render_frames_corrected_assets/`)
found that the alternative sprite pack's `.npy` collision maps of five rooms are solid below row 47
(plan section 198). Checks on the corrected game and the final training run in
`runs/2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_...`. Job ids of the probes
(6831990, 6831992, 6832000, 6832008, 6832019, 6832046, 6832058, 6832059, 6832061) are in this
folder's `slurm/submitted_jobids.txt`; all finished.
