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
