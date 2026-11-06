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
