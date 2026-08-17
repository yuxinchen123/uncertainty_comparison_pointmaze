# Canary phase — estimate against actual

Five canary runs, one per (chunk, node) pairing the submission actually uses, each at the chunk's
full copy count and limited to 200 iterations (one phase-blocked episode window). Their purpose is
the skill's: prove the card holds the program, prove the compiled-program cache, prove the resume,
and correct the time estimate before the real jobs go out. The deviation from Step 6's figure of at
least 30 simultaneous canaries, and why five is the right number for a five-chunk one-class
submission, is in `experiment_background.md`. Times are Pacific.

The canaries ran at 2026-08-16 23:17 PT and each finished in 36 to 54 seconds.

## Throughput

Seconds per iteration is the steady rate: the wall clock of the 200 iterations minus the first
iteration, which pays for compiling the program, divided by the remaining 199. Both rate columns are
the same measurement seen two ways — what the card does in total and what one of its copies gets —
and the column after them restates the per-copy rate in the unit a run is planned in. Every card is
an H100 NVL.

| chunk | arm | node | copies | copy indices | planned (h) | s/iteration | total steps/s (millions) | steps/s per copy | hours per million steps per copy | build and prime (s) | first iteration incl. compile (s) | projected hours for 1,953,200 iterations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unit-1 chunk 1 of 2 | `rnd_next_state` | serval06 | 512 | 0–511 | 5.31 | 0.00929 | 28.21 | 55,104 | 0.0050 | 21.0 | 1.9 | 5.04 |
| unit-1 chunk 2 of 2 | `rnd_next_state` | serval06 | 512 | 512–1023 | 5.31 | 0.00950 | 27.58 | 53,874 | 0.0052 | 20.9 | 19.4 | 5.16 |
| unit-2 chunk 1 of 1 | `gt_position_velocity_sqrt` | serval08 | 1,024 | 0–1023 | 4.58 | 0.00853 | 61.45 | 60,011 | 0.0046 | 20.4 | 18.8 | 4.63 |
| unit-3 chunk 1 of 1 | `gt_position_velocity_linear` | serval07 | 1,024 | 0–1023 | 4.60 | 0.00866 | 60.57 | 59,148 | 0.0047 | 19.8 | 18.1 | 4.70 |
| unit-4 chunk 1 of 1 | `none` | serval09 | 1,024 | 0–1023 | 4.54 | 0.00869 | 60.31 | 58,895 | 0.0047 | 20.5 | 18.6 | 4.72 |

**The plan held.** Every chunk came in within 4 per cent of its planned time, and the makespan the
canaries project — 5.16 hours, set by the second distillation chunk — is 3 per cent under the plan's
5.31. No chunk was reassigned.

One detail the table shows outright: the first canary on serval06 compiled its program in 1.9
seconds while the other four took 18 to 19. Both distillation chunks are the same compiled program
(same arm, same 512 copies), and the rate probe had already built it on serval06 an hour earlier, so
that chunk read the node-local cache; the other three nodes had never seen their program. That is
also why every science job is pinned to the node its canary ran on.

## The rate probes that preceded the canaries

Before the plan was computed, two probe jobs measured every arm at 256, 512 and 1,024 copies —
6538652 on serval06 (an H100 NVL) and 6538653 on cheetah01 (an A100-PCIE-40GB) — because the shared
throughput survey stops at 512 copies and prices no four-way split at all. The full table is
`code/measured_cells.json`; the two things it changed about the plan are in `infra_history.md`. The
headline is that halving a unit buys much less at these copy counts than the survey suggests: on an
H100 the per-copy rate improves 1.43 times between 1,024 and 512 copies for distillation and 1.25
times for the other three arms, against the survey trainer's 1.59.

## Resume

The resume was tested on a canary rather than argued for. Job 6538659 re-ran the exact command of
the already-finished `unit-4` canary on the same node. Its log reads

```
resume: unit-4_none_learning-rate-1e-3_no-intrinsic-weight_chunk-1-of-1_copies-1024_copy-index-0-1023
is already complete in ...jsonl (4 records on disk); skipping it
```

and it exited 0 after one second. The shard held 4 records before the re-run and 4 after: nothing was
repeated, nothing was truncated, and the file was opened for appending rather than rewritten. That is
the whole resume contract of this platform — no model state is saved, so the resumable unit is the
chunk, and a chunk whose shard ends in a completion record makes a re-run a logged no-op.

## The chunk partition

Two chunks of one unit hold disjoint slices of its copies, and the canaries' own records show it: the
`unit-1` chunks report copy indices 0–511 and 512–1023 in their `unit_start`. The property is pinned
by three tests rather than by inspection — `tests/agents/test_copy_seed_offset.py` for the seed
arithmetic (including that a chunk's initial weights are bitwise the whole run's own copies),
`code/test_plan_submission.py` for the planner's index ranges, and
`code/test_chunk_partition_end_to_end.py`, which trains two chunks of a 16-copy toy unit through the
real runner and checks the aggregate holds all 16 copies exactly once.

## The missing-argument guards

Re-tested before the first submission; all three refuse rather than fall back to defaults. The three
cases and their exit codes are in `infra_history.md`.
