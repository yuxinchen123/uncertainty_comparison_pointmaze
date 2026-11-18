# Speed work on the fast last layer and the game-event logging (pixel Montezuma's Revenge)

Measured 2026-11-17 22:36 PT to 2026-11-18 00:11 PT (times are Pacific, converted from machines running Eastern).
Every number below is read from a `profile.jsonl` under `data/speed-profiles/<unit>/` of this folder; the unit and job
names are in `logs/serval_units.txt` and `slurm/submitted_jobids_speed_sl5nw.txt`.

## How it was measured

- **H100 NVL, four copies:** serval02 card 1 (`GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3`), run through
  `serval02_card1_run.sh --kind profile`. serval05's card was held during the whole work by another account
  (`pyt9dp`, three processes, 100% use, without the queue lock), so the rule `uva-serval02-card1-fallback.md` sent the
  profiles to serval02 card 1. Seeds 1 to 4; the learner and environment settings are those of the archived
  four-copy run (`run.json` of `h100-serval02-card-1_final-training_seeds-1-to-4`).
- **RTX A4500, one copy:** jaguar03 through Slurm, seed 13 (seed 5 for the checks from the archived state); the
  settings of the archived one-copy run (`a4500-jaguar03_final-training_seed-5`). Up to eight of these jobs shared the
  node, each on its own card; their whole-update spreads (0.07 to 0.13 s) are larger than the H100's.
- **Whole update:** compiled ahead of time, two warm updates, then 6 to 10 timed updates in a row, each followed by
  the host work `train_final.py` does for one update (metrics to the host, the non-finite check, the JSON line, the
  game-record fold and the first-times file). The observation warm-up was 2 random rollouts instead of 50; it does not
  change the work of an update.
- **Fast-layer settings:** four copies with the step sizes 0.01, 0.03, 0.1 and 0.3 (the plan's packing); one copy
  with step size 0.1 (0.3 in the checks on real rollouts); the bonus profile and the game events on.
- **Parts:** the post-rollout pass alone, on stored features of a rollout's shape, 7 timed calls after 2 warm calls;
  the rollout's time loop alone (`PixelPPORND.collect_steps`), 7 timed calls after 2 warm calls.
- No other process was on the card before or after any timed block (read with `nvidia-smi` each time), so no
  measurement had to be repeated.

Let $r$ be the steps per second one copy gets, $r = 16384 / s$ for $s$ seconds per update (16,384 steps per update
and copy). The tables give the total steps per second (copies times $r$, in millions), $r$ (in thousands), and the
hours per million steps per copy, $\dfrac{10^{6}}{3600 \times r}$.

## Where the time goes (H100 NVL, four copies)

Whole updates, reference form of the fast last layer:

| setting | copies | seconds per update | total steps per second (millions) | steps per second per copy (thousands) | hours per million steps per copy | range of the timed updates (spread) | timed updates | host work (ms) |
|---|---|---|---|---|---|---|---|---|
| default (unit 1) | 4 | 2.450 | 0.0267 | 6.69 | 0.0415 | 2.406 to 2.467 (0.061) | 6 | 0.16 |
| default (unit 4) | 4 | 2.477 | 0.0265 | 6.61 | 0.0420 | 2.433 to 2.503 (0.070) | 8 | 0.21 |
| game events (unit 4) | 4 | 2.501 | 0.0262 | 6.55 | 0.0424 | 2.439 to 2.515 (0.075) | 8 | 0.30 |
| fast layer, group 128, events, bonus profile | 4 | 2.475 | 0.0265 | 6.62 | 0.0420 | 2.468 to 2.492 (0.024) | 6 | 0.46 |
| fast layer, group 16, events, bonus profile | 4 | 2.500 | 0.0262 | 6.55 | 0.0424 | 2.481 to 2.542 (0.061) | 6 | 0.58 |
| fast layer, group 4, events, bonus profile | 4 | 2.550 | 0.0257 | 6.42 | 0.0432 | 2.528 to 2.582 (0.054) | 6 | 0.34 |

Peak card memory was 38.0 GiB in every variant (JAX peak bytes in use). The spread between updates (0.02 to 0.08 s)
is larger than most parts below, so the split comes from the parts timed alone:

| part, timed alone | H100, four copies (ms) | RTX A4500, one copy (ms) |
|---|---|---|
| rollout loop, default | 1,247.8 (1,243.9 to 1,252.9) | 800.3 (799.2 to 804.3) |
| rollout loop, game events on | 1,250.4 (1,249.7 to 1,250.4) | 805.8 (804.0 to 808.9) |
| rollout loop, fast layer storing features and targets (group 16, events on) | 1,243.7 (1,242.8 to 1,246.8) | 800.1 (799.7 to 800.9) |
| rollout loop, fast layer storing features and plain errors (accepted form, group 16, events on) | 1,247.6 (1,233.6 to 1,249.7) | 804.5 (804.3 to 808.2) |
| post-rollout pass, reference form, group 128 | 12.8 (12.7 to 12.9) | 102.8 (102.7 to 103.0) |
| post-rollout pass, reference form, group 16 | 22.7 (22.6 to 22.9) | 143.3 (143.3 to 143.4) |
| post-rollout pass, reference form, group 4 | 71.8 (68.7 to 72.2) | 392.9 (392.8 to 393.1) |
| bonus profile: predictor's own bonus from the stored features | 1.8 (1.6 to 2.0) | 1.1 (1.1 to 1.2) |
| bonus profile: row alignment | 1.0 (0.8 to 1.1) | 1.5 (1.4 to 1.6) |
| host work per update (`train_final.py`) | 0.16 to 0.58 | 0.09 to 0.17 |

What this shows:

1. **The event logging costs almost nothing:** 2.6 ms per update in the rollout on the H100 (0.1%) and 5.5 ms on
   the RTX A4500 (0.3%), whose whole updates with and without events have the same median (1.668 s); the host work
   is under 0.6 ms.
2. **Storing what the fast layer needs in the rollout costs nothing measurable:** with the stored features (and
   targets or plain errors) the rollout loop takes 1,243.7 and 1,247.6 ms on the H100 and 800.1 and 804.5 ms on the
   RTX A4500, against 1,250.4 and 805.8 ms with the events alone (the fast layer skips the bonus inside the loop).
3. **The post-rollout pass is the expensive part, and it grows as the sharing group shrinks:** 13 to 72 ms on the
   H100 (0.5% to 2.9% of an update), 103 to 393 ms on the RTX A4500 (6% to 24%). The RTX A4500 does float64
   arithmetic at about 1/64 of its float32 rate, the H100 at about the same rate, which is why the same pass costs 5
   to 8 times more there with one copy than the H100 needs for four.
4. **The bonus profile costs about 3 ms** on both cards.

## The ten ideas

Four ideas of the plan's list (section E2) target parts that cost nothing or were already in the first
implementation, so they were replaced by ideas aimed at the post-rollout pass:

1. Plan idea 1 (take the predictor's output from the stored hidden features, so no convolution runs twice): the
   first implementation already does this. Replaced by: the predictor's own bonus of the bonus profile from the
   stored plain errors.
2. Plan idea 8 (one batched pass over copies and groups): the first implementation already runs every copy and group
   on one batch axis. Replaced by: groups of more than 16 environments solved one step at a time instead of in
   chunks.
3. Plan idea 9 (reduce the events inside the rollout step): already the first implementation, and the events cost
   2.6 ms per update. Replaced by: float32 only for each step's change of the temporary layer, accumulated and
   applied in float64.
4. Plan idea 10 (compact event fields, fold once per checkpoint): the host work it would remove is 0.1 to 0.6 ms per
   update. Replaced by: chunks of 128 rows instead of 512 for groups of 16.

The ideas rest on one identity (written out in `src/exploration_platform/atari_pixels/fast_layer_forms.py`): define
$X_t$ as the error of step $t$'s states divided by their row lengths and $e_t$ as the predictor's plain error; the
step-by-step temporary layer of the reference is forward substitution of one block lower triangular system,
$(I + \eta Q) X = e / L$, whose entries $Q_{ts} = R_t F_s^{\top}$ (for $s$ before $t$) are products of unit rows $R_t$
with fit rows $F_s$ of earlier steps. A chunk of steps can be solved with one triangular solve, and the temporary
layer is then needed only as its change between chunks; a whole rollout of 512 rows needs no layer at all.

**Numbers:** a float64 form must give the reference form's bonus to float64 rounding. A form that reads float32
numbers (the plain errors, float32 products) is kept only if, over every transition of 20 real rollouts, its bonus
differs from the reference form's by at most $10^{-4}$ relative; this tolerance was fixed before the checks on real
rollouts were run.

### Decisions

| # | idea | pass time against the reference (ms, H100 four copies; RTX A4500 one copy) | numbers | decision |
|---|---|---|---|---|
| 1 | predictor's own bonus of the bonus profile from the stored plain errors | 1.90 to 0.69; 1.08 to 0.26 | equal to the plain RND bonus formula; bit-equal to the product with the stored features on test features | accepted (part of idea 4); too small to see in a whole update |
| 2 | chunked solve from products of rows of different steps | group 4: 71.8 to 13.3; 392.9 to 114.1. Group 16: 22.7 to 12.3; 143.3 to 106.0. Group 128: slower, 13.3 to 29.2 against 12.8; 126.5 to 264.8 against 102.8 | float32 bonus bit-equal on 1,310,720 real transitions per group | accepted for groups of up to 16 |
| 3 | the whole rollout as one chunk (all row products in one product, no loop) | group 4: 10.1; 89.2. Group 16 (2,048 rows): 13.9; 187.5, slower than chunks | bit-equal | accepted when the whole rollout has at most 512 rows (group 4) |
| 4 | the rollout stores the predictor's float32 plain error instead of the target output | group 4: 10.1 to 8.8; 89.2 to 64.3. Group 16: 12.6 to 11.4; 131.4 to 107.7. Group 128: 13.3 to 12.2; 126.5 to 102.2 | at most 2.3e-6 (H100) and 4.0e-6 (RTX A4500, also at the end of training) relative | accepted |
| 5 | float32 for both products with the layer's change | fastest on the RTX A4500: group 128 102.8 to 28.8, group 16 143.3 to 18.4, group 4 392.9 to 41.6; no gain on the H100 for group 4 (26.8 against 6.3) | group 128: 0.32 (H100) and 2.0e-3 (RTX A4500); group 16 at one step per chunk: 1.2e-4 (H100) | not accepted (fails the tolerance); kept as option `fast_last_layer_float32_products="changes_and_applications"` |
| 6 | the fit rows of small groups by a factorization written out entry by entry | group 4: 10.1 to 7.6, with idea 4 8.8 to 6.3; 89.2 to 80.2, with idea 4 64.3 to 55.0. Group 16: 12.6 to 11.2; 131.4 to 126.5 | bit-equal | accepted for groups of up to 16 |
| 7 | unroll the loop over steps or chunks by 2 or 4 | H100: 2 of 6 cases faster (by 1.4 and 0.3 ms), the others equal or up to 7.9 ms slower; RTX A4500: 1 of 8 faster (by 7.0 ms, a form that was not accepted), 7 slower by 0.4 to 10.9 ms | bit-equal | not accepted: for the accepted group-16 chunking it was not faster (12.33 to 12.34 ms; 106.0 to 110.2 ms), group 4 has no loop; option removed |
| 8 | groups of more than 16 environments one step at a time instead of in chunks | group 128: one step 13.3 against 24.1 (4 steps) and 29.2 (16 steps); 126.5 against 160.8 and 264.8 | bit-equal | accepted |
| 9 | float32 only for each step's change of the layer, accumulated and applied in float64 | RTX A4500: group 128 106.8 to 71.4, group 16 103.7 to 78.9 | group 128: 5.6e-4 (fresh state), 3.9e-5 (end of training); group 16: at most 2.1e-5 | not accepted (fails the tolerance for group 128); kept as option `fast_last_layer_float32_products="changes"` |
| 10 | chunks of 128 rows instead of 512 for groups of 16 | 12.6 to 12.3 (float64, targets); 103.7 to 80.0 (plain errors, written-out factorization) | bit-equal | accepted |

**The accepted set,** now the default of the fast-layer path (`config.py`): plain errors (idea 4) with the
predictor's own bonus from them (idea 1); the whole rollout as one chunk when it has at most 512 rows (idea 3, group
4); chunks of at most 128 rows for other groups of up to 16 environments (ideas 2 and 10, group 16: 8 steps); one step
at a time for larger groups (idea 8, group 128); the written-out factorization for groups of up to 16 (idea 6);
float64 throughout. The reference form stays available with `fast_last_layer_chunk_rows=0`,
`fast_last_layer_plain_errors=false` and `fast_last_layer_small_solve="cholesky"`; the default RND program is unchanged
(`tests/atari_pixels/test_default_program_unchanged.py` passes). Every accepted idea has a unit test against the
reference form in `tests/atari_pixels/test_fast_layer_forms.py`.

### Screens of the post-rollout pass, H100 NVL, four copies

Seconds per update: the measured whole update of the reference form of that group (first table) minus the measured
saving of the pass. The pass was timed alone on test features of a rollout's shape, which cannot show the float32
problems of ideas 5 and 9 (those come from real rollouts, below).

| form | group | pass (ms) | range (ms) | seconds per update | total steps per second (millions) | steps per second per copy (thousands) | hours per million steps per copy |
|---|---|---|---|---|---|---|---|
| reference | 4 | 71.83 | 68.73 to 72.19 | 2.550 | 0.0257 | 6.43 | 0.0432 |
| idea 2: chunks of 128 rows (32 steps) | 4 | 13.34 | 13.22 to 13.55 | 2.492 | 0.0263 | 6.58 | 0.0422 |
| idea 3: the whole rollout as one chunk (512 rows) | 4 | 10.13 | 10.11 to 10.29 | 2.488 | 0.0263 | 6.58 | 0.0422 |
| idea 4: plain errors, whole rollout | 4 | 8.84 | 8.80 to 8.91 | 2.487 | 0.0264 | 6.59 | 0.0422 |
| idea 5: plain errors, float32 products, whole rollout | 4 | 8.95 | 8.91 to 9.02 | 2.487 | 0.0264 | 6.59 | 0.0422 |
| idea 5: plain errors, float32 products, one step per chunk | 4 | 26.81 | 26.11 to 27.55 | 2.505 | 0.0262 | 6.54 | 0.0425 |
| idea 6: written-out factorization, whole rollout | 4 | 7.58 | 7.57 to 7.62 | 2.486 | 0.0264 | 6.59 | 0.0421 |
| ideas 4 and 6: plain errors, written-out factorization, whole rollout (accepted for group 4) | 4 | 6.25 | 6.20 to 6.30 | 2.484 | 0.0264 | 6.59 | 0.0421 |
| idea 7: unrolled by 4, chunks of 128 rows | 4 | 11.91 | 11.81 to 12.10 | 2.490 | 0.0263 | 6.58 | 0.0422 |
| idea 7: unrolled by 4, plain errors, float32 products, one step per chunk | 4 | 34.68 | 33.02 to 35.27 | 2.513 | 0.0261 | 6.52 | 0.0426 |
| reference | 16 | 22.69 | 22.56 to 22.90 | 2.500 | 0.0262 | 6.55 | 0.0424 |
| idea 2: chunks of 128 rows (8 steps) | 16 | 12.33 | 12.26 to 12.62 | 2.490 | 0.0263 | 6.58 | 0.0422 |
| idea 2: chunks of 512 rows (32 steps) | 16 | 12.62 | 12.36 to 12.81 | 2.490 | 0.0263 | 6.58 | 0.0422 |
| idea 3: the whole rollout as one chunk (2,048 rows) | 16 | 13.86 | 13.52 to 14.30 | 2.491 | 0.0263 | 6.58 | 0.0422 |
| idea 4: plain errors, chunks of 512 rows | 16 | 11.39 | 11.15 to 11.46 | 2.489 | 0.0263 | 6.58 | 0.0422 |
| idea 5: plain errors, float32 products, chunks of 512 rows | 16 | 10.48 | 10.42 to 10.59 | 2.488 | 0.0263 | 6.59 | 0.0422 |
| idea 5: plain errors, float32 products, one step per chunk | 16 | 13.12 | 12.89 to 13.21 | 2.490 | 0.0263 | 6.58 | 0.0422 |
| idea 6: written-out factorization, chunks of 512 rows | 16 | 11.15 | 11.05 to 11.50 | 2.488 | 0.0263 | 6.58 | 0.0422 |
| idea 7: unrolled by 4, chunks of 128 rows | 16 | 12.34 | 12.31 to 12.36 | 2.490 | 0.0263 | 6.58 | 0.0422 |
| idea 7: unrolled by 4, plain errors, float32 products, one step per chunk | 16 | 14.30 | 14.16 to 14.42 | 2.492 | 0.0263 | 6.58 | 0.0422 |
| ideas 5 and 6: plain errors, float32 products, one step per chunk, written-out factorization | 16 | 11.92 | 11.77 to 12.07 | 2.489 | 0.0263 | 6.58 | 0.0422 |
| reference | 128 | 12.76 | 12.65 to 12.88 | 2.475 | 0.0265 | 6.62 | 0.0420 |
| idea 2: chunks of 128 rows (one step) | 128 | 13.33 | 13.30 to 13.44 | 2.476 | 0.0265 | 6.62 | 0.0420 |
| idea 2: chunks of 512 rows (4 steps) | 128 | 24.05 | 23.98 to 24.08 | 2.486 | 0.0264 | 6.59 | 0.0422 |
| idea 2: chunks of 2,048 rows (16 steps) | 128 | 29.21 | 29.10 to 29.37 | 2.491 | 0.0263 | 6.58 | 0.0422 |
| idea 4: plain errors, one step per chunk (accepted for group 128) | 128 | 12.24 | 12.18 to 12.27 | 2.474 | 0.0265 | 6.62 | 0.0420 |
| idea 4: plain errors, chunks of 512 rows | 128 | 23.04 | 22.97 to 23.18 | 2.485 | 0.0264 | 6.59 | 0.0421 |
| idea 5: plain errors, float32 products, one step per chunk | 128 | 11.90 | 11.86 to 11.94 | 2.474 | 0.0265 | 6.62 | 0.0419 |
| idea 5: plain errors, float32 products, chunks of 512 rows | 128 | 22.67 | 22.48 to 22.72 | 2.485 | 0.0264 | 6.59 | 0.0421 |
| idea 7: unrolled by 4, plain errors, float32 products, one step per chunk | 128 | 11.58 | 11.55 to 11.64 | 2.474 | 0.0265 | 6.62 | 0.0419 |
| idea 7: unrolled by 4, chunks of 512 rows | 128 | 24.51 | 24.44 to 24.67 | 2.487 | 0.0264 | 6.59 | 0.0422 |

### Screens of the post-rollout pass, RTX A4500, one copy

Seconds per update: the measured whole update of the reference form (1.805, 1.860 and 2.101 s for groups 128, 16 and
4) minus the measured saving of the pass. Screens 1 to 3 are three jobs; each has its own reference row.

| form | group | pass (ms) | range (ms) | seconds per update | total steps per second (millions) | steps per second per copy (thousands) | hours per million steps per copy |
|---|---|---|---|---|---|---|---|
| reference (screen 1) | 4 | 392.94 | 392.83 to 393.05 | 2.101 | 0.0078 | 7.80 | 0.0356 |
| idea 2: chunks of 128 rows (32 steps) | 4 | 114.07 | 114.04 to 114.34 | 1.822 | 0.0090 | 8.99 | 0.0309 |
| idea 3: the whole rollout as one chunk (512 rows) | 4 | 89.16 | 89.08 to 89.40 | 1.797 | 0.0091 | 9.12 | 0.0305 |
| idea 4: plain errors, whole rollout | 4 | 64.27 | 64.18 to 64.58 | 1.772 | 0.0092 | 9.24 | 0.0300 |
| idea 5: plain errors, float32 products, whole rollout | 4 | 64.70 | 64.64 to 65.83 | 1.773 | 0.0092 | 9.24 | 0.0301 |
| idea 5: plain errors, float32 products, one step per chunk | 4 | 51.52 | 51.38 to 51.93 | 1.760 | 0.0093 | 9.31 | 0.0298 |
| idea 6: written-out factorization, whole rollout | 4 | 80.15 | 80.05 to 80.23 | 1.788 | 0.0092 | 9.16 | 0.0303 |
| ideas 4 and 6: plain errors, written-out factorization, whole rollout (accepted for group 4) | 4 | 55.02 | 54.78 to 55.09 | 1.763 | 0.0093 | 9.29 | 0.0299 |
| idea 7: unrolled by 4, chunks of 128 rows | 4 | 107.06 | 107.04 to 107.23 | 1.815 | 0.0090 | 9.03 | 0.0308 |
| idea 7: unrolled by 4, plain errors, float32 products, one step per chunk | 4 | 62.41 | 62.12 to 62.52 | 1.770 | 0.0093 | 9.25 | 0.0300 |
| reference (screen 2) | 4 | 392.95 | 392.85 to 393.26 | 2.101 | 0.0078 | 7.80 | 0.0356 |
| ideas 4, 5 and 6: plain errors, float32 products, one step per chunk, written-out factorization | 4 | 42.97 | 41.25 to 44.05 | 1.751 | 0.0094 | 9.36 | 0.0297 |
| the same with idea 7, unrolled by 2 | 4 | 48.82 | 48.59 to 48.91 | 1.757 | 0.0093 | 9.33 | 0.0298 |
| reference (screen 3) | 4 | 395.57 | 395.38 to 395.71 | 2.101 | 0.0078 | 7.80 | 0.0356 |
| idea 9: float32 changes, whole rollout (no change is formed) | 4 | 55.09 | 54.79 to 55.16 | 1.761 | 0.0093 | 9.31 | 0.0298 |
| idea 9: float32 changes, one step per chunk | 4 | 274.42 | 274.40 to 274.45 | 1.980 | 0.0083 | 8.28 | 0.0336 |
| idea 5: float32 products, one step per chunk, written-out factorization | 4 | 41.63 | 41.31 to 43.31 | 1.747 | 0.0094 | 9.38 | 0.0296 |
| ideas 2 and 10: chunks of 128 rows, plain errors, written-out factorization | 4 | 81.21 | 81.13 to 81.47 | 1.787 | 0.0092 | 9.17 | 0.0303 |
| reference (screen 1) | 16 | 143.30 | 143.25 to 143.37 | 1.860 | 0.0088 | 8.81 | 0.0315 |
| idea 2: chunks of 128 rows (8 steps) | 16 | 106.02 | 105.98 to 106.29 | 1.823 | 0.0090 | 8.99 | 0.0309 |
| idea 2: chunks of 512 rows (32 steps) | 16 | 131.43 | 131.40 to 131.44 | 1.848 | 0.0089 | 8.87 | 0.0313 |
| idea 3: the whole rollout as one chunk (2,048 rows) | 16 | 187.47 | 187.45 to 188.72 | 1.904 | 0.0086 | 8.60 | 0.0323 |
| idea 4: plain errors, chunks of 512 rows | 16 | 107.72 | 107.65 to 109.52 | 1.824 | 0.0090 | 8.98 | 0.0309 |
| idea 5: plain errors, float32 products, chunks of 512 rows | 16 | 57.61 | 57.47 to 57.64 | 1.774 | 0.0092 | 9.23 | 0.0301 |
| idea 5: plain errors, float32 products, one step per chunk | 16 | 23.41 | 23.35 to 23.64 | 1.740 | 0.0094 | 9.42 | 0.0295 |
| idea 6: written-out factorization, chunks of 512 rows | 16 | 126.49 | 126.28 to 126.81 | 1.843 | 0.0089 | 8.89 | 0.0312 |
| idea 7: unrolled by 4, chunks of 128 rows | 16 | 110.17 | 106.60 to 110.67 | 1.827 | 0.0090 | 8.97 | 0.0310 |
| idea 7: unrolled by 4, plain errors, float32 products, one step per chunk | 16 | 26.27 | 26.20 to 26.73 | 1.743 | 0.0094 | 9.40 | 0.0296 |
| reference (screen 2) | 16 | 143.28 | 142.94 to 150.41 | 1.860 | 0.0088 | 8.81 | 0.0315 |
| ideas 4, 5 and 6: plain errors, float32 products, one step per chunk, written-out factorization | 16 | 18.35 | 16.87 to 18.46 | 1.735 | 0.0094 | 9.44 | 0.0294 |
| the same with idea 7, unrolled by 2 | 16 | 20.52 | 20.31 to 20.54 | 1.737 | 0.0094 | 9.43 | 0.0295 |
| ideas 4 and 6: plain errors, written-out factorization, chunks of 512 rows | 16 | 103.15 | 101.07 to 103.27 | 1.820 | 0.0090 | 9.00 | 0.0309 |
| reference (screen 3) | 16 | 143.13 | 142.92 to 143.32 | 1.860 | 0.0088 | 8.81 | 0.0315 |
| idea 9: float32 changes, chunks of 512 rows, plain errors, written-out factorization | 16 | 78.85 | 76.12 to 82.79 | 1.796 | 0.0091 | 9.12 | 0.0304 |
| idea 5: float32 products, chunks of 512 rows, plain errors, written-out factorization | 16 | 52.86 | 51.30 to 59.17 | 1.770 | 0.0093 | 9.26 | 0.0300 |
| idea 9: float32 changes, one step per chunk | 16 | 107.80 | 106.96 to 107.95 | 1.825 | 0.0090 | 8.98 | 0.0309 |
| idea 5: float32 products, one step per chunk, written-out factorization | 16 | 18.38 | 16.91 to 18.44 | 1.735 | 0.0094 | 9.44 | 0.0294 |
| idea 10: chunks of 128 rows, plain errors, written-out factorization (accepted for group 16) | 16 | 79.98 | 75.75 to 80.51 | 1.797 | 0.0091 | 9.12 | 0.0305 |
| reference (screen 1) | 128 | 102.82 | 102.72 to 102.98 | 1.805 | 0.0091 | 9.08 | 0.0306 |
| idea 2: chunks of 128 rows (one step) | 128 | 126.51 | 126.37 to 127.30 | 1.829 | 0.0090 | 8.96 | 0.0310 |
| idea 2: chunks of 512 rows (4 steps) | 128 | 160.75 | 160.65 to 161.29 | 1.863 | 0.0088 | 8.79 | 0.0316 |
| idea 2: chunks of 2,048 rows (16 steps) | 128 | 264.82 | 263.55 to 264.89 | 1.967 | 0.0083 | 8.33 | 0.0333 |
| idea 4: plain errors, one step per chunk (accepted for group 128) | 128 | 102.21 | 102.10 to 104.37 | 1.804 | 0.0091 | 9.08 | 0.0306 |
| idea 4: plain errors, chunks of 512 rows | 128 | 136.49 | 136.40 to 138.47 | 1.839 | 0.0089 | 8.91 | 0.0312 |
| idea 5: plain errors, float32 products, one step per chunk | 128 | 28.81 | 28.76 to 29.35 | 1.731 | 0.0095 | 9.47 | 0.0293 |
| idea 5: plain errors, float32 products, chunks of 512 rows | 128 | 75.67 | 75.61 to 75.96 | 1.778 | 0.0092 | 9.22 | 0.0301 |
| idea 7: unrolled by 4, plain errors, float32 products, one step per chunk | 128 | 29.17 | 29.11 to 29.33 | 1.731 | 0.0095 | 9.46 | 0.0294 |
| idea 7: unrolled by 4, chunks of 512 rows | 128 | 161.15 | 161.06 to 161.43 | 1.863 | 0.0088 | 8.79 | 0.0316 |
| reference (screen 3) | 128 | 118.52 | 102.30 to 119.64 | 1.805 | 0.0091 | 9.08 | 0.0306 |
| plain errors, one step per chunk (the accepted form) | 128 | 106.82 | 101.20 to 107.07 | 1.809 | 0.0091 | 9.06 | 0.0307 |
| idea 9: float32 changes, one step per chunk | 128 | 71.40 | 71.13 to 72.29 | 1.774 | 0.0092 | 9.24 | 0.0301 |
| idea 5: float32 products, one step per chunk | 128 | 28.55 | 28.51 to 28.59 | 1.731 | 0.0095 | 9.47 | 0.0293 |

In screen 3 the reference of group 128 took about 119 ms in 6 of its 7 calls (102.3 ms once), against 102.8 ms in
screen 1, while other jobs of this work started on the same node; the seconds per update of screen 3's group-128
forms therefore use screen 1's reference (102.82 ms).

### Numbers on real rollouts

Largest relative difference of the bonus between each form and the reference form, over every transition of 20 real
updates (each form is computed on the same stored features of every rollout; training continues with the reference
bonus). H100: four copies, step sizes 0.01 to 0.3, fresh state, 1,310,720 transitions per form. RTX A4500: one copy,
step size 0.3, 327,680 transitions per form, from a fresh state and from the archived state at 2 billion steps
(seed 5). The largest absolute differences are one float32 rounding step of the largest bonuses (1.3e-5 to 6.1e-5),
at most 2.4e-6 of the mean bonus. A float64 form whose float32 bonus is bit-equal on every transition shows 0.

In the H100 column the plain-error rows used the library factorization and, for group 16, chunks of 512 rows; the
RTX A4500 columns used exactly the accepted form.

| form | group | H100, fresh | RTX A4500, fresh | RTX A4500, end of training | within 1e-4 |
|---|---|---|---|---|---|
| chunks (128 or 512 rows), whole rollout, written-out factorization (float64, targets) | 4, 16, 128 | 0 | | | yes |
| the accepted form (plain errors) | 4 | 8.6e-7 | 8.4e-7 | 4.0e-6 | yes |
| the accepted form (plain errors) | 16 | 2.3e-6 | 9.9e-7 | 3.7e-6 | yes |
| the accepted form (plain errors) | 128 | 1.6e-6 | 1.1e-6 | 3.2e-6 | yes |
| idea 5, both products float32, one step per chunk | 4 | 1.87e-5 | | | yes |
| idea 5, both products float32, one step per chunk | 16 | 1.22e-4 | 3.8e-5 | 8.1e-5 | no |
| idea 5, both products float32, chunks | 16 | 5.9e-5 (512 rows) | 6.4e-5 (128 rows) | 2.9e-5 (128 rows) | yes |
| idea 5, both products float32 | 128 | 0.32 | 2.0e-3 | 1.4e-4 | no |
| idea 9, float32 changes, 128-row chunks | 16 | | 1.0e-5 | 2.1e-5 | yes |
| idea 9, float32 changes, one step per chunk | 128 | | 5.6e-4 | 3.9e-5 | no |

## The accepted set, whole updates re-timed

H100 NVL, four copies (units `task-3-final_h100_*`, same process for the rows of one unit):

| setting | copies | seconds per update | total steps per second (millions) | steps per second per copy (thousands) | hours per million steps per copy | range (spread) | over the default |
|---|---|---|---|---|---|---|---|
| default | 4 | 2.475 | 0.0265 | 6.62 | 0.0420 | 2.427 to 2.494 (0.067) | |
| fast layer, accepted, group 128, events, bonus profile | 4 | 2.517 | 0.0260 | 6.51 | 0.0427 | 2.474 to 2.545 (0.071) | +1.7% |
| fast layer, accepted, group 16, events, bonus profile | 4 | 2.497 | 0.0262 | 6.56 | 0.0423 | 2.471 to 2.538 (0.067) | +0.9% |
| game events (second unit) | 4 | 2.462 | 0.0266 | 6.65 | 0.0417 | 2.410 to 2.481 (0.071) | -0.5% |
| fast layer, accepted, group 4, events, bonus profile (second unit) | 4 | 2.482 | 0.0264 | 6.60 | 0.0421 | 2.455 to 2.526 (0.071) | +0.3% (+0.8% over the events of its own unit) |

The rows of the second unit are compared with the first unit's default; these whole-update differences (0.3% to
1.7%) are within the spread of one update (about 0.07 s, 2.8%). The parts of the final code, timed alone in the second
unit, give the cost more precisely:

| part, H100 NVL, four copies | reference form (ms) | accepted form (ms) |
|---|---|---|
| post-rollout pass, group 128 | 12.75 (12.71 to 12.93) | 12.40 (12.36 to 12.45) |
| post-rollout pass, group 16 | 22.83 (22.79 to 22.89) | 9.87 (9.85 to 9.98) |
| post-rollout pass, group 4 | 72.15 (68.68 to 72.97) | 6.40 (6.39 to 6.53) |
| bonus profile: the predictor's own bonus | 1.90 (1.89 to 1.92), from the stored features | 0.69 (0.67 to 1.40), from the stored plain errors, bit-equal |
| bonus profile: row alignment | 1.12 (0.98 to 1.17) | 1.12 |

RTX A4500, one copy (one job per row pair, 10 timed updates each; jaguar03 also ran the other jobs of this work):

| setting | copies | seconds per update | total steps per second (millions) | steps per second per copy (thousands) | hours per million steps per copy | range (spread) | over the default (1.668 s) |
|---|---|---|---|---|---|---|---|
| default | 1 | 1.668 | 0.0098 | 9.82 | 0.0283 | 1.618 to 1.700 (0.082) | |
| game events | 1 | 1.668 | 0.0098 | 9.83 | 0.0283 | 1.621 to 1.695 (0.074) | 0.0% |
| fast layer, reference form, group 128 | 1 | 1.805 | 0.0091 | 9.08 | 0.0306 | 1.755 to 1.850 (0.095) | +8.2% |
| fast layer, accepted, group 128 | 1 | 1.820 | 0.0090 | 9.00 | 0.0309 | 1.761 to 1.860 (0.099) | +9.1% |
| fast layer, reference form, group 16 | 1 | 1.860 | 0.0088 | 8.81 | 0.0315 | 1.787 to 1.904 (0.117) | +11.5% |
| fast layer, accepted, group 16 | 1 | 1.798 | 0.0091 | 9.11 | 0.0305 | 1.720 to 1.831 (0.111) | +7.8% |
| fast layer, reference form, group 4 | 1 | 2.101 | 0.0078 | 7.80 | 0.0356 | 2.017 to 2.147 (0.130) | +26.0% |
| fast layer, accepted, group 4 | 1 | 1.755 | 0.0093 | 9.34 | 0.0298 | 1.676 to 1.796 (0.120) | +5.2% |

**Overhead of the fast layer with the logging (events and bonus profile), accepted form, against the default:**

| card | group 128 | group 16 | group 4 | under 5% |
|---|---|---|---|---|
| H100 NVL, four copies, whole updates | +1.7% | +0.9% | +0.3% | yes |
| H100 NVL, four copies, sum of the parts (pass, profile, events) | 16.8 ms, 0.7% | 14.3 ms, 0.6% | 10.8 ms, 0.4% | yes |
| RTX A4500, one copy, whole updates | +9.1% | +7.8% | +5.2% | no |
| RTX A4500, one copy, sum of the parts (pass, profile, events) | 109 to 114 ms, 6.6% to 6.8% | 87 ms, 5.2% | 62 ms, 3.7% | group 4 only |

The H100 meets the aim with a wide margin. The RTX A4500 meets it only for group 4: the rest is the float64
arithmetic of the pass (102 to 107 ms for group 128, 80 ms for group 16), which the card does at 1/64 of its float32
rate. The float32 options would bring these to 28.6 to 71 ms and 18 to 53 ms (about 2% to 4% of an update) but
changed small bonuses of group 128 by more than the tolerance (5.6e-4 and 0.32 relative). On this shared node the
whole-update medians are uncertain by about 0.05 s, which is why they lie 30 to 50 ms above the sum of the parts; the
rollout loop shows no extra cost of the fast layer there.

## Gate 4: default path against the archive, four copies

PASSED on serval02 card 1, the card that produced the archive (evaluation unit
`gate-4_default-path_four-copies_seeds-1-to-4_updates-100_logging-off_h100-serval02-card-1`, 23:33 to 23:47 PT):
`train_final.py` with seeds 1 to 4, 100 updates, the archived H100 settings; `compare_metric_prefix.py` against the
archived four-copy `metrics.jsonl` compared 1,500 fields of 100 lines and found 0 differing
(`data/gate-4_*/comparison_with_archive.json`). Seconds per update in that run: 2.505 (median of updates 21 to
100). Since serval02 card 1 gave the archive's bits, the comparison of old and new code on serval05 was not needed.

## Gate 6: speed and memory per card type (one copy)

One Slurm job per card type, the archived one-copy settings, seed 13, at most 15 minutes: (a) default, (b) the fast
layer in the accepted form, group 16, game events and bonus profile. Peak card memory is JAX's peak bytes in use of
the process after the variant (the second variant's value covers both); host memory is the job's MaxRSS.

Free cards were read with `sinfo` at 23:44 PT, just before the jobs were submitted. A card type with no free card
was skipped. The RTX A4000's default speed (2.038 s per update) is the bar.

| card type (node, job) | free cards | setting | copies | seconds per update | total steps per second (millions) | steps per second per copy (thousands) | hours per million steps per copy | over the default | peak card memory (GiB) | host memory, MaxRSS (GiB) | slower than the RTX A4000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX A4000 (cheetah08, 6955823) | 8 | default | 1 | 2.038 | 0.0080 | 8.04 | 0.0346 | | 3.95 | 5.71 | the bar |
| RTX A4000 | | fast layer | 1 | 2.147 | 0.0076 | 7.63 | 0.0364 | +5.3% | 3.95 | | |
| RTX 4000 Ada (lynx05, 6955824) | 15 | default | 1 | 3.039 | 0.0054 | 5.39 | 0.0515 | | 3.95 | 5.70 | yes: flagged |
| RTX 4000 Ada | | fast layer | 1 | 3.127 | 0.0052 | 5.24 | 0.0530 | +2.9% | 3.95 | | yes: flagged |
| RTX 2080 Ti (affogato11, 6955829) | 29 | default | 1 | 1.987 | 0.0082 | 8.25 | 0.0337 | | 5.28 | 5.03 | no |
| RTX 2080 Ti | | fast layer | 1 | 2.059 | 0.0080 | 7.96 | 0.0349 | +3.6% | 5.38 | | no |
| RTX A4500 (jaguar03, 6955816 and 6955818) | 8 | default | 1 | 1.668 | 0.0098 | 9.82 | 0.0283 | | 3.95 | 5.05 | no |
| RTX A4500 | | fast layer | 1 | 1.798 | 0.0091 | 9.11 | 0.0305 | +7.8% | 3.96 | 5.52 | no |
| A40 (jaguar01, jaguar06) | 0 of 6 | skipped: no free card | | | | | | | | | |
| A100 40 GB (cheetah01) | 0 of 4 | skipped: no free card | | | | | | | | | |
| A100 80 GB (cheetah04) | 0 of 4 | skipped: no free card | | | | | | | | | |

- **RTX 2080 Ti:** the first job (6955822) ran out of card memory in the default update (one allocation of 5.10 GiB
  with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` of 11 GB). The second job (6955829), identical except for the fraction
  0.95, ran both settings; its program needs 5.3 to 5.4 GiB against 3.95 GiB on the other cards. Workers on this
  card type need the fraction 0.95.
- **RTX 4000 Ada:** 1.5 times the RTX A4000's time per update, so it is below the bar and is left out by the plan's
  rule (section E, gate 6).
- The fast layer here is the accepted form at group 16 with the game events and the bonus profile on.
