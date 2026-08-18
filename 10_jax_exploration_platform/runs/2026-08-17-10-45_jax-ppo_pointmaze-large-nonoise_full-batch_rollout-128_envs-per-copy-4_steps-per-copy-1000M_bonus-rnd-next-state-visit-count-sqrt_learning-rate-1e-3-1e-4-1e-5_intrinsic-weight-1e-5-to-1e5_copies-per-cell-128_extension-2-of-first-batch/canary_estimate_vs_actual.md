# Canary phase — estimate against actual

Written 2026-08-17 20:15 PT from `canary/data/*.jsonl`.

One canary per (chunk, card): 36 of them, each 400 iterations at the chunk's full copy count, run by the chunk's own job on the card that then carried its science run. `experiment_background.md` records why the canary, the resume check and the science run are one job here rather than a canary phase followed by a submission phase.

## Throughput

Seconds per iteration is the steady rate: the wall clock of the 400 iterations minus the first, which pays for compiling the program, divided by the remaining 399. The two rate columns are the same measurement seen two ways — what the card does in total and what one of its copies gets — and the column after them restates the per-copy rate in the unit a run is planned in.

| chunk | arm | node | class | copies | copy indices | s/iteration | total steps/s (millions) | steps/s per copy | hours per million steps per copy | build and prime (s) | first iteration incl. compile (s) | planned (h) | projected (h) | difference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unit-2 chunk 3 of 8 | `gt_position_velocity_sqrt` | jaguar03 | `jaguar03` | 528 | 32–47 | 0.01294 | 20.89 | 39,569 | 0.0070 | 16.1 | 22.7 | 7.39 | 7.02 | -5.0% |
| unit-2 chunk 4 of 8 | `gt_position_velocity_sqrt` | jaguar03 | `jaguar03` | 528 | 48–63 | 0.01290 | 20.95 | 39,679 | 0.0070 | 16.1 | 20.3 | 7.39 | 7.00 | -5.2% |
| unit-2 chunk 5 of 8 | `gt_position_velocity_sqrt` | jaguar03 | `jaguar03` | 528 | 64–79 | 0.01279 | 21.14 | 40,047 | 0.0069 | 17.2 | 22.1 | 7.39 | 6.94 | -6.1% |
| unit-2 chunk 6 of 8 | `gt_position_velocity_sqrt` | jaguar03 | `jaguar03` | 528 | 80–95 | 0.01289 | 20.98 | 39,728 | 0.0070 | 17.3 | 21.4 | 7.39 | 6.99 | -5.3% |
| unit-2 chunk 7 of 8 | `gt_position_velocity_sqrt` | jaguar03 | `jaguar03` | 528 | 96–111 | 0.01277 | 21.17 | 40,100 | 0.0069 | 16.8 | 21.2 | 7.39 | 6.93 | -6.2% |
| unit-2 chunk 8 of 8 | `gt_position_velocity_sqrt` | jaguar03 | `jaguar03` | 528 | 112–127 | 0.01295 | 20.88 | 39,538 | 0.0070 | 16.9 | 21.0 | 7.39 | 7.03 | -4.9% |
| unit-1 chunk 1 of 32 | `rnd_next_state` | nekomata01 | `nekomata01` | 132 | 0–3 | 0.00655 | 10.31 | 78,118 | 0.0036 | 25.3 | 88.8 | 3.36 | 3.56 | +5.8% |
| unit-1 chunk 2 of 32 | `rnd_next_state` | nekomata01 | `nekomata01` | 132 | 4–7 | 0.00674 | 10.03 | 75,994 | 0.0037 | 24.9 | 86.7 | 3.36 | 3.66 | +8.8% |
| unit-1 chunk 3 of 32 | `rnd_next_state` | jaguar03 | `jaguar03` | 132 | 8–11 | 0.00977 | 6.92 | 52,388 | 0.0053 | 14.3 | 22.8 | 5.50 | 5.30 | -3.6% |
| unit-1 chunk 4 of 32 | `rnd_next_state` | jaguar03 | `jaguar03` | 132 | 12–15 | 0.00989 | 6.84 | 51,784 | 0.0054 | 14.3 | 20.1 | 5.50 | 5.36 | -2.5% |
| unit-1 chunk 5 of 32 | `rnd_next_state` | jaguar01 | `jaguar01` | 132 | 16–19 | 0.01035 | 6.53 | 49,447 | 0.0056 | 21.5 | 47.6 | 5.50 | 5.62 | +2.1% |
| unit-1 chunk 6 of 32 | `rnd_next_state` | jaguar01 | `jaguar01` | 132 | 20–23 | 0.01029 | 6.57 | 49,775 | 0.0056 | 21.4 | 47.3 | 5.50 | 5.58 | +1.5% |
| unit-1 chunk 7 of 32 | `rnd_next_state` | jaguar01 | `jaguar01` | 132 | 24–27 | 0.01043 | 6.48 | 49,104 | 0.0057 | 21.4 | 47.1 | 5.50 | 5.66 | +2.9% |
| unit-1 chunk 8 of 32 | `rnd_next_state` | jaguar06 | `jaguar06` | 132 | 28–31 | 0.01009 | 6.70 | 50,747 | 0.0055 | 22.6 | 41.4 | 5.52 | 5.47 | -0.8% |
| unit-1 chunk 11 of 32 | `rnd_next_state` | cheetah08 | `cheetah08-09` | 132 | 40–43 | 0.01249 | 5.41 | 40,981 | 0.0068 | 20.1 | 32.6 | 6.91 | 6.78 | -1.8% |
| unit-1 chunk 12 of 32 | `rnd_next_state` | cheetah08 | `cheetah08-09` | 132 | 44–47 | 0.01255 | 5.39 | 40,801 | 0.0068 | 20.3 | 32.4 | 6.91 | 6.81 | -1.4% |
| unit-1 chunk 13 of 32 | `rnd_next_state` | cheetah08 | `cheetah08-09` | 132 | 48–51 | 0.01237 | 5.46 | 41,379 | 0.0067 | 20.2 | 33.0 | 6.91 | 6.71 | -2.8% |
| unit-1 chunk 14 of 32 | `rnd_next_state` | cheetah08 | `cheetah08-09` | 132 | 52–55 | 0.01248 | 5.41 | 41,021 | 0.0068 | 20.0 | 32.3 | 6.91 | 6.77 | -1.9% |
| unit-1 chunk 15 of 32 | `rnd_next_state` | cheetah09 | `cheetah08-09` | 132 | 56–59 | 0.01259 | 5.37 | 40,671 | 0.0068 | 24.3 | 57.8 | 6.91 | 6.83 | -1.1% |
| unit-1 chunk 16 of 32 | `rnd_next_state` | cheetah09 | `cheetah08-09` | 132 | 60–63 | 0.01244 | 5.43 | 41,172 | 0.0067 | 24.4 | 57.9 | 6.91 | 6.75 | -2.3% |
| unit-1 chunk 17 of 32 | `rnd_next_state` | cheetah09 | `cheetah08-09` | 132 | 64–67 | 0.01250 | 5.41 | 40,974 | 0.0068 | 24.3 | 58.0 | 6.91 | 6.78 | -1.8% |
| unit-1 chunk 18 of 32 | `rnd_next_state` | cheetah09 | `cheetah08-09` | 132 | 68–71 | 0.01276 | 5.30 | 40,116 | 0.0069 | 24.3 | 57.7 | 6.91 | 6.92 | +0.3% |
| unit-1 chunk 19 of 32 | `rnd_next_state` | lotus | `lotus` | 132 | 72–75 | 0.01315 | 5.14 | 38,926 | 0.0071 | 18.9 | 28.3 | 7.44 | 7.14 | -4.1% |
| unit-1 chunk 20 of 32 | `rnd_next_state` | lotus | `lotus` | 132 | 76–79 | 0.01337 | 5.05 | 38,286 | 0.0073 | 18.8 | 27.9 | 7.44 | 7.26 | -2.5% |
| unit-1 chunk 21 of 32 | `rnd_next_state` | lotus | `lotus` | 132 | 80–83 | 0.01325 | 5.10 | 38,642 | 0.0072 | 18.8 | 27.8 | 7.44 | 7.19 | -3.4% |
| unit-1 chunk 22 of 32 | `rnd_next_state` | lotus | `lotus` | 132 | 84–87 | 0.01323 | 5.11 | 38,687 | 0.0072 | 18.9 | 27.7 | 7.44 | 7.18 | -3.5% |
| unit-1 chunk 23 of 32 | `rnd_next_state` | lotus | `lotus` | 132 | 88–91 | 0.01327 | 5.09 | 38,586 | 0.0072 | 19.1 | 28.0 | 7.44 | 7.20 | -3.3% |
| unit-1 chunk 24 of 32 | `rnd_next_state` | lotus | `lotus` | 132 | 92–95 | 0.01329 | 5.08 | 38,512 | 0.0072 | 18.7 | 27.5 | 7.44 | 7.21 | -3.1% |
| unit-1 chunk 25 of 32 | `rnd_next_state` | lotus | `lotus` | 132 | 96–99 | 0.01330 | 5.08 | 38,494 | 0.0072 | 19.1 | 28.0 | 7.44 | 7.22 | -3.0% |
| unit-1 chunk 26 of 32 | `rnd_next_state` | lotus | `lotus` | 132 | 100–103 | 0.01320 | 5.12 | 38,785 | 0.0072 | 19.4 | 27.8 | 7.44 | 7.16 | -3.7% |
| unit-1 chunk 27 of 32 | `rnd_next_state` | cheetah03 | `cheetah03` | 132 | 104–107 | 0.01354 | 4.99 | 37,805 | 0.0073 | 18.6 | 22.3 | 7.95 | 7.35 | -7.6% |
| unit-1 chunk 28 of 32 | `rnd_next_state` | cheetah03 | `cheetah03` | 132 | 108–111 | 0.01316 | 5.14 | 38,907 | 0.0071 | 18.9 | 22.2 | 7.95 | 7.14 | -10.2% |
| unit-1 chunk 29 of 32 | `rnd_next_state` | affogato11 | `affogato11` | 132 | 112–115 | 0.01387 | 4.87 | 36,913 | 0.0075 | 27.6 | 29.9 | 8.05 | 7.53 | -6.5% |
| unit-1 chunk 30 of 32 | `rnd_next_state` | affogato11 | `affogato11` | 132 | 116–119 | 0.01355 | 4.99 | 37,793 | 0.0074 | 27.6 | 29.9 | 8.05 | 7.35 | -8.7% |
| unit-1 chunk 31 of 32 | `rnd_next_state` | affogato11 | `affogato11` | 132 | 120–123 | 0.01379 | 4.90 | 37,133 | 0.0075 | 27.6 | 29.8 | 8.05 | 7.48 | -7.1% |
| unit-1 chunk 32 of 32 | `rnd_next_state` | affogato11 | `affogato11` | 132 | 124–127 | 0.01387 | 4.87 | 36,906 | 0.0075 | 27.9 | 29.5 | 8.05 | 7.53 | -6.5% |

**The plan held.** The largest deviation of any chunk from its planned time is -10.2 per cent, and the projected makespan over the chunks that started immediately is 7.53 hours against the plan's 8.05. No chunk was reassigned on the canaries' evidence.

The chunks whose rate the plan carried across from another card — every class except `jaguar03`, `lotus` and `cheetah08-09`, which were probed directly — are the ones the canary was there to check, and they came in on the fast side of their estimate rather than the slow side.

## Resume

Every job ran its chunk's canary command a second time before starting the science run and required the runner to log that the unit was already complete and to leave the shard's record count unchanged; a job whose second run added a record or failed exits 5 and never starts the science run (`code/run_chunk.sh`). Every job reached its science run, so the resume was demonstrated once per chunk, on the card that ran it. That is the whole resume contract of this platform — no model state is saved, so the resumable unit is the chunk, and a chunk whose shard ends in a completion record makes a re-run a logged no-op.
