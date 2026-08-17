# Canary phase — estimate against actual

Four canary runs, one per (work unit, node class) pairing that the real submission uses, each at
the unit's full copy count and limited to 200 iterations (one phase-blocked episode window).
Their purpose is the skill's: prove the card holds the program, prove the compiled-program cache,
prove the resume, and correct the time estimate before the real jobs go out. Times are Pacific.

## Throughput

| unit | bonus | card | copies | seconds per iteration (estimated) | seconds per iteration (measured) | total steps per second (millions) | steps per second per copy | hours per million steps per copy | build and prime (s) | first iteration incl. compile (s) | projected minutes for 19,600 iterations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unit-1 | `rnd_next_state` | serval07 H100 NVL | 8,448 | 0.0840 | 0.0870 | 49.70 | 5,883 | 0.0472 | 350 | 77 | 28.4 |
| unit-2 | `gt_position_velocity_sqrt` | serval07 H100 NVL | 8,448 | 0.0345 | 0.0358 | 120.79 | 14,298 | 0.0194 | 241 | 15 | 11.7 |
| unit-3 | `gt_position_velocity_linear` | serval08 H100 NVL | 8,448 | 0.0343 | 0.0350 | 123.65 | 14,637 | 0.0190 | 243 | 16 | 11.4 |
| unit-4 | `none` | serval05 H100 NVL | 768 | 0.0070 | 0.0089 | 44.24 | 57,601 | 0.0048 | 24 | 18 | 2.9 |
