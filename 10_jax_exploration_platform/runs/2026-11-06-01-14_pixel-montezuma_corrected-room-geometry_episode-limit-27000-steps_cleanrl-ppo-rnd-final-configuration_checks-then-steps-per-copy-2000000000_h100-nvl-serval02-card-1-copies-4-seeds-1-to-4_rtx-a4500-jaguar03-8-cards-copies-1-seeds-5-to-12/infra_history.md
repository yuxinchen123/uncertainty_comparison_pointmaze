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
