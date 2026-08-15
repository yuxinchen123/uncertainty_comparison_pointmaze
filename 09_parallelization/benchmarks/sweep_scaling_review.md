# Review of bench_sweep_scaling.py — the sweep scaling measurement

Reviewed: `benchmarks/bench_sweep_scaling.py` (three studies: `rates`, `copies`, `groups`), against
`ppo/torch_ppo/torch_ppo_rnd.py` (`sweep_config`, `_adam_step_per_copy`, `_clip_per_copy_and_step`,
`_build_iteration_graph`), `ppo/torch_ppo/SWEEP.md`, `progress_and_changes.md` rows 13-14, and the
report's copy-scaling and sweep sections.

## Findings

### 1. The number of groups cannot cost anything. The `groups` study needs a noise floor, or it proves nothing — blocker

Traced through the code, the group count enters the trainer in exactly three places, and none of
them is inside the timed iteration:

- `sweep_config` sums the per-rate counts into `n_copies`; from there on only the total is used.
- `PPORND.__init__` builds `lr_per_copy`, a `[C]` float vector. More groups means more distinct
  values in that vector, never a different shape.
- `copy_group` (a `[C]` long tensor) and `copy_seed_index` are built. `copy_group` is written at
  line 182 and read nowhere else; `copy_seed_index` only picks which seed each copy's weights and
  environments are drawn from, at construction time.

Everything in the timed path reduces over the copy axis and is blind to how the copies are labelled:

- `_adam_step_per_copy` is elementwise over `[C, ...]` tensors, with `lr_view` broadcast along the
  copy axis. Six operations per parameter tensor, the same six whether the vector holds one distinct
  value or sixty-four.
- `_clip_per_copy_and_step` reduces each gradient to a `[C]` norm and scales by a `[C]` factor.
- `_losses` reduces per copy and returns `loss_c.sum()`.
- `_build_iteration_graph` captures shapes fixed by `C`, `T`, `N` only.

So the cost is a function of the total copy count alone. The `groups` study should therefore be a
flat line, and its whole content is a null result — which the current protocol cannot support: ten
timed iterations, one median, each configuration measured once, no repeated point, no noise floor. A
flat-looking line with no noise floor is not evidence of flatness, and a 2% bend in it cannot be told
from drift.

**Change**: measure the `groups` points in interleaved repeat order (1, 2, 4, ..., 64, 64, ..., 4, 2,
1), report per-point spread as the noise floor the way `ab_compare.py` does, and state the verdict
against it: flat means max-minus-min across group counts is within the within-point spread. Peak
memory should be identical to the byte across the whole study; report it as a second check.

**If the line is not flat**, do not report the shape — explain it first:

- Monotone in run order: drift or clock throttling. The reversed half of the interleaved order
  separates this in one run.
- A step at one point: that process compiled or placed memory differently. Re-run that point alone
  in a fresh process and compare peak reserved memory and the compile log.
- Reproducible, non-monotone, order-independent: a real dependency. Nothing in the traced code can
  produce one, so it would be a defect worth finding, not a curve to publish.

### 2. `rates` and `copies` measure the same six configurations — should-fix

With the documented defaults, `rates` runs 1, 2, 4, 8, 16, 32 rates at 128 copies each, and `copies`
runs 8, 16, 32, 64, 128, 256 copies at 16 rates. Both give total copy counts of 128, 256, 512, 1024,
2048, 4096 — the same six numbers. By finding 1 the two studies are the same six measurements, and
both are the copy-count curve the report already carries (round 2, C = 8 to 512 measured, saturation
quoted to 16,384).

Neither study, as written, adds anything to that curve. What they can add:

- Run them as a deliberate replicate pair. Point by point they should agree, and their difference is
  a noise floor obtained for free — which is what finding 1 needs.
- State the headline plainly: adding a rate costs exactly what adding the same number of copies
  costs. The rate count is free; only the total copy count is paid for.

### 3. No uniform-rate anchor, so nothing here prices the sweep mechanism — should-fix

Every point in all three studies is a sweep configuration, including the single-rate ones (`is_sweep`
is true whenever `learning_rates` is non-empty, so the hand-written per-copy Adam is used even at one
rate). The `+1.0%` sweep overhead in ledger row 14 is a single measurement at 512 copies. The studies
never re-check it, so they cannot say whether the sweep path stays free at 4096 copies.

**Change**: at two or three copy counts (512 and 4096, say), also time `production_config(C)` —
uniform rate, torch's fused Adam — and report the ratio beside the sweep point. One extra process per
anchored point.

### 4. The separate-runs baseline is missing, and it is free to add — should-fix

The user's alternative today is one job per rate. `bench_sweep.py` prices that once, at 4 rates x 128
copies. In the `rates` study the same comparison needs no extra GPU time: running the groups one
after another costs `G x ms(copies_per_rate)`, and `ms(128)` is already the first row of the table.

**Change**: add a derived column "same rates run in turn = G x ms(copies per rate)" and the ratio to
the fused number. That is the column that answers "what do I gain by fusing the sweep", across the
range instead of at one point.

### 5. Per-environment throughput as computed is the clock in different units — should-fix

`env_steps_per_sec_per_env` is `T / sec`, and `env_steps_per_sec_per_copy` is `T * N / sec`. With `N`
fixed at 4, the per-env column is the per-copy column divided by 4 at every row, and the total column
is the per-copy column times `C`. Three of the four reported speeds are `sec` and `C` rescaled by
constants; none of them answers a question the ms/iter column does not.

It is also not the number a user asking about "per env" throughput wants. They want to know how long
their sweep takes and what one more rate adds to it.

**Change**: report instead

- hours to a fixed budget per copy, e.g. 20,000 iterations = 10.24M environment steps per copy
  (`iterations * sec`) — this is what decides how long the sweep runs;
- the marginal cost of one more rate, in minutes added to that fixed budget;
- GPU-seconds per copy per million copy-steps, `sec / (C * T * N) * 1e6`, as the per-copy price;
- keep ms/iter and total env-steps/s; drop per-env, or say once that `N` is fixed so it is the
  per-copy number divided by `N`.

### 6. Points are always run smallest to largest, so drift is aliased into the scaling curve — should-fix

Each study walks its points in one fixed increasing order, in one process each, over tens of minutes.
Any monotone drift over that window — clock throttling as the card heats, memory fragmentation across
the machine — lands entirely on the large configurations and makes the scaling look steeper than it
is. The `groups` study is the most exposed, because its whole signal is smaller than plausible drift.

**Change**: run the point list forward and then backward in one invocation and report both passes;
record `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw` alongside each point, or lock
clocks with `nvidia-smi -lgc` for the duration.

### 7. Ten timed iterations, when raising it costs nothing — should-fix

Each point pays a full trainer construction, `prime_obs_rms`, `torch.compile`, and a graph capture —
on the order of a minute. The timed window is 10 iterations, which at 20 ms is 0.2 s, and at 4096
copies about 2 s. Raising to `--iters 60 --warmup 10` (what `ab_compare.py` uses) adds a few seconds
per point against a minute of setup, and it is what makes a median plus a p10/p90 spread meaningful.

**Change**: default `--iters 60 --warmup 10`; record min, median, p10, p90 and the sample count in
every row, not the median alone.

### 8. Failed and skipped points vanish from the record — should-fix

`child_run` returns `None` on failure and `main` skips the row, so an out-of-memory point leaves the
curve simply ending, with nothing in the JSON to say a point was attempted. The `groups` study also
silently drops any rate count that does not divide the total. The report documents the round-2 ceiling
at 16,384 copies, so the largest configurations here (4096) are within it, but a study extended past
it would truncate silently.

**Change**: append a row with `"status": "failed"` (or `"skipped"`) and the tail of stderr, so the
JSON records where the ceiling is instead of implying the curve stopped by choice.

### 9. The script does not take the H100 lock its own docstring names — should-fix

`locks/README.md` requires every GPU-touching command to go through `bash locks/gpu_run.sh "..."`.
The docstring says "under the H100 lock" but nothing in the script takes it, and a second GPU process
sharing the card invalidates every number in all three studies. The subprocesses inherit the parent's
lock, so wrapping the parent invocation is enough.

**Change**: launch through `gpu_run.sh`, and record in the results JSON that it was.

### 10. Peak memory is under-reported — note

`torch.cuda.max_memory_allocated()` excludes what the caching allocator reserves, the CUDA-graph
private pool, and the compile workspace, which is exactly the difference that decides whether a
configuration fits. Add `torch.cuda.max_memory_reserved()` and `torch.cuda.mem_get_info()`. The peak
is read after the timing loop with no reset, so it covers the graph build too — that is the right
choice for a ceiling number and should stay.

### 11. Startup work per point is larger than the measurement — note

Every point runs `prime_obs_rms` (10 x 128 eager environment steps) and builds parameters in a python
loop over copies with a QR per layer per copy (`stack` / `_orthogonal`), which at 4096 copies is tens
of thousands of small CPU factorizations. Neither affects the timed iteration, whose cost is set by
shapes. For timing-only runs `obs_norm_init_iters=1` is a legitimate override and returns most of that
time to the card.

### 12. The rate range makes most groups diverge, and paired seeding makes the copies duplicates — note

`rates_for` spans 1e-5 to 1e-2, so from two groups upward one group always trains at 1e-2, which the
report's demonstration run shows is unstable. This does not change the timing (shapes are fixed, and
NaN arithmetic costs nothing here), but a timed configuration whose parameters have gone non-finite is
worth knowing about; check that parameters are still finite at the end of a point. Separately, the
default `sweep_seed_mode="paired"` gives copy k of every group the same seed, so the `rates` study's
4096-copy point is 128 distinct problems replicated 32 times, not 4096 independent runs. Timing is
unaffected; the report wording should not call them independent runs.

### 13. `--study` has no default and no `required=True` — note

`--study` defaults to `None`, which falls through the `if/elif` into the `groups` branch. A typo'd or
forgotten flag silently runs the wrong study. Set `required=True`.

## What the three curves should look like

Predicted from the traced implementation and the report's round-2 copy-scaling table, so the measured
output can be checked against them rather than read for the first time.

Reference points, round 2, style B, T=128, N=4 (report table; the 1024-and-up rows are the round-1
measurements divided by the 1.6-1.8x round-2 speedup measured at 128-512 copies, so they are estimates):

| total copies | ms/iter | source |
|---|---|---|
| 128 | 20.4 | measured |
| 256 | 28.0 | measured |
| 512 | 43.8 | measured |
| 1024 | 65-72 | estimate |
| 2048 | 115-130 | estimate |
| 4096 | 220-245 | estimate |

Sweep points should sit about 1% above these (ledger row 14).

**Study `rates` (128 copies per rate, 1 to 32 rates).** Not flat anywhere, because the first point is
already at 128 copies, past the knee. Roughly: 1 rate ~20.6 ms, 2 rates ~28 ms, 4 rates ~44 ms, then
close to a doubling per doubling of the rate count. Total environment steps per second rises and
flattens near the 7.8e6 saturation the report quotes; per-copy throughput falls about as 1/C. Peak
memory rises close to linearly in the copy count. The 16-rate point (2048 copies) should equal the
report's 2048-copy point.

**Study `copies` (16 rates, 8 to 256 copies per rate).** The same six total copy counts as `rates`,
so the prediction is the strongest one available: point by point equal to the `rates` study within
the noise floor, in both time and memory. If the two studies disagree by more than their spread, the
disagreement is the measurement's, not the trainer's, and the `groups` null result cannot be trusted
until it is explained.

**Study `groups` (2048 copies, 1 to 64 groups).** Flat. Every point equal to the 2048-copy point of
the other two studies, all seven within the noise floor of each other, with peak memory identical
across the study. This is the prediction the design exists to test, and it is the one that needs
finding 1's repeated points to mean anything.

**Cross-check available for free.** `env_steps_per_sec_per_env` should equal
`env_steps_per_sec_per_copy / 4` in every row of every study, and `env_steps_per_sec` should equal
`env_steps_per_sec_per_copy * total_copies`. If those identities hold exactly, the three throughput
columns carry one number between them, which is finding 5.

**Honest headline if the predictions hold.** Adding a learning rate costs exactly what adding the
same number of copies costs, and nothing extra for the rate itself: a 16-rate sweep at 128 copies per
rate costs what 2048 copies at one rate cost, plus about 1% for the per-copy optimizer. What the
sweep saves is against the alternative of running the rates one after another, which costs 16 times
the price of 128 copies.
