# Progress and changes — ext4m CPU/PyTorch efficiency auto-research

The experiment log, in the autoresearch style (`../reference_autoresearch/program.md`): one row per
attempt, kept or discarded on the measurement, failures included. Newest section at the bottom.

**Metric:** `steps_per_s` of the real ext4m workload (10,000 steps, production config from a real
queue marker, `OMP_NUM_THREADS=1`), always compared to the `baseline` condition measured in the
**same job on the same exclusive node**.
**Hard constraint:** the run's record JSON, minus `runtime_seconds`, must be **identical** to the
baseline's. Faster-but-different goes to `future_changes.md`, never to `applied_changes.md`.

## Setup (2026-08-13 03:30-03:41)

| step | what |
|---|---|
| read the ground | the repo already carries a 2026-06-25 profiling campaign: per-phase breakdown (SAC 76%, RND 16%, polyak 4%, env 3%), two **bit-exact** switches built into `train.py` (`opt_polyak_foreach`, `opt_torch_reward`, +6.1% combined) that **default to off and have never been used by a production sweep**, and measured dead ends (C++ env ≤1.4%, >2 threads hurts) |
| pick the node | `puma01` (asked for) sits in reservation `sl5nw_156`, `Users=sl5nw` — this account cannot place a job there. Substitute: **`cortado01`**, 48 cores, idle, same class as the production node `cortado09`, held `--exclusive`, one job at a time |
| protect the live sweep | 450 ext4m runs are training out of this repo. No candidate edits the repo: `analysis/code/patched_entry.py` applies each candidate inside the measurement process only. Queue markers were **copied**, never moved |
| build the harness | `analysis/code/run_conditions.py` — one subprocess per condition, reports steps/s **and** diffs the produced record against the baseline's, field by field |
| smoke test | 400 steps, 3 conditions, on the login node: harness end-to-end OK, records produced and compared, `optflags` already showing ≈+6% and `IDENTICAL` |

## Experiment 1 — screen every candidate (job 6536785, cortado01 exclusive, 10,000 steps)

Conditions: `baseline`, `optflags`, `adamforeach`, `normcache`, `gcfreeze`, `tcmalloc`, `combined`
× three production configs (`run5origrnd`, `alg23`, `gtposvel`).

| candidate | hypothesis |
|---|---|
| `optflags` | the two shipped bit-exact switches; expect ≈+6%, `IDENTICAL` by construction |
| `adamforeach` | torch's CPU default for Adam/SGD is the single-tensor Python loop; SAC steps three optimizers plus the RND predictor every step |
| `normcache` | `_normalize_obs` rebuilds mean/var tensors from numpy on every call (twice per gradient step, once per env step) though the statistics move only in `RND.update()` |
| `gcfreeze` | the env/model/graph objects are long-lived; every generational collection rescans them |
| `tcmalloc` | the step allocates heavily (tensor churn); glibc malloc is the default only because nothing set `LD_PRELOAD` |
| `combined` | all of the above, the configuration a 96-hour run would actually use |

_Results appended below as the job reports them._

### Results — config `alg23` (job 6536785, cortado01 exclusive, 10,000 steps, 1 thread)

Baseline: **32.16 steps/s** (316.9 s for 10,000 steps ≈ 31 ms/step — the 2026-06-25 campaign
measured 30.5 ms/step for the same workload on a clean node, so the harness reproduces the known
cost of a step). At this rate one clean, uncontended 4M run is ~34.5 h.

| condition | steps/s | vs baseline | record identical? | verdict |
|---|---|---|---|---|
| `baseline` | 32.160 | — | (reference) | — |
| `optflags` | 32.880 | **+2.2%** | IDENTICAL | keep |
| `adamforeach` | 32.471 | +1.0% | IDENTICAL | keep (confirm with repeats) |
| `normcache` | 32.355 | +0.6% | IDENTICAL | keep (confirm with repeats) |
| `gcfreeze` | 32.475 | +1.0% | IDENTICAL | keep (confirm with repeats) |
| `tcmalloc` | 34.124 | **+6.1%** | IDENTICAL | keep — the largest single win found |

**The finding of this experiment: `LD_PRELOAD=libtcmalloc_minimal.so.4` is worth more than every
code-level change put together (+6.1%), and it changes no code at all** — the step allocates and
frees hard (tensor churn in the SAC update, the buffer's per-sample tensors, the wrapper's
per-step numpy), and glibc's malloc was the default only because nothing had ever set `LD_PRELOAD`.
Being an allocator swap it cannot change a number, and the record check confirms it did not.

`optflags` landing at +2.2% rather than the +6.1% the 2026-06-25 campaign measured is expected: that
campaign ran 2 threads per worker, where the polyak `zip_strict` loop it removes is a larger share of
a shorter step. It is still free and still identical, so it stays.

The three ~1% candidates are inside single-measurement noise; job 2 repeats them before anything is
claimed for them.

### Results — configs `gtposvel` and `run5origrnd`, and `combined` (same job)

Every condition, on every one of the sweep's three configurations, produced a record **identical**
to that configuration's baseline. Not one logged number moved.

| config | baseline | `optflags` | `adamforeach` | `normcache` | `gcfreeze` | `tcmalloc` | **`combined`** |
|---|---|---|---|---|---|---|---|
| `alg23` | 32.160 | 32.880 (+2.2%) | 32.471 (+1.0%) | 32.355 (+0.6%) | 32.475 (+1.0%) | 34.124 (+6.1%) | **35.130 (+9.2%)** |
| `gtposvel` | 38.116 | 39.366 (+3.3%) | 38.235 (+0.3%) | 38.248 (+0.3%) | 38.314 (+0.5%) | 40.152 (+5.3%) | **41.331 (+8.4%)** |
| `run5origrnd` | 33.076 | 34.492 (+4.3%) | 33.644 (+1.7%) | 33.151 (+0.2%) | _pending_ | _pending_ | _pending_ |

(steps/s; percentages against that row's own baseline, measured in the same job on the same
exclusive node.)

Reading:

- **`combined` is worth ~+9%** and is record-identical on both configurations that have finished it.
  The parts do not fully add (9.2% vs 2.2+1.0+0.6+1.0+6.1 = 10.9%), which is expected — `gcfreeze`
  and `tcmalloc` both attack allocation/bookkeeping and partly overlap.
- **`tcmalloc` is the single biggest lever on every config** (+5.3% to +6.1%) and is the one change
  that touches no code.
- `optflags` grows as the RND work grows: +2.2% on `alg23`, +3.3% on `gtposvel`, +4.3% on
  `run5origrnd` (whose Adam predictor makes the per-step optimizer work largest).
- `adamforeach` / `normcache` / `gcfreeze` sit between +0.2% and +1.7% — individually at the edge of
  single-run noise, which is what the repeats in experiment 2 are for. They are free and identical,
  so the question is only whether to claim a number for them, not whether to keep them.

### Experiment 1 complete — job 6536785, COMPLETED in 1:41:25, 21/21 runs

| config | baseline steps/s | `optflags` | `adamforeach` | `normcache` | `gcfreeze` | `tcmalloc` | **`combined`** |
|---|---|---|---|---|---|---|---|
| `alg23` | 32.160 | ×1.0224 | ×1.0097 | ×1.0061 | ×1.0098 | ×1.0611 | **×1.0924** |
| `gtposvel` | 38.116 | ×1.0328 | ×1.0031 | ×1.0035 | ×1.0052 | ×1.0534 | **×1.0843** |
| `run5origrnd` | 33.076 | ×1.0428 | ×1.0172 | ×1.0023 | ×1.0122 | ×1.0694 | **×1.0970** |

**21 of 21 runs: `IDENTICAL`.** Every condition, on every configuration of the sweep, produced a
record equal field-for-field to its baseline — same eval rows, same per-episode rows, same visit
counts. The stack is worth **+8.4% to +9.7% (mean +9.1%)** and changes nothing that is measured.

What that buys the 4M sweep: a clean-node run drops from ~34.5 h to ~31.6 h (`alg23`); across 900
runs, ~9% of the sweep's total CPU time.

## Experiment 2 — repeats + four more candidates (job 6536795, cortado01 exclusive)

`--configs alg23 --conditions baseline,interop1,mkldnnoff,infmode,arena1,combined --repeat 2`

Two questions:
1. **Are the ~1% candidates real?** Repeating `baseline` and `combined` twice gives the run-to-run
   spread, which is the only honest way to say whether +0.6% means anything. (The repeat is also a
   determinism check: baseline#0 vs baseline#1 must be `IDENTICAL`, or the whole method is void.)
2. **Do four more knobs help?**
   - `interop1` — the inter-op thread pool is sized from the node's 46 cores even though the worker
     pins the intra-op pool to 1.
   - `mkldnnoff` — oneDNN's dispatch may cost more than it saves on 256×256 MLPs. The one candidate
     here NOT expected to be record-identical; the check decides.
   - `infmode` — `inference_mode` rather than `no_grad` for the bonus forwards (no version counters).
   - `arena1` — glibc's per-thread arenas collapsed to one, as a no-LD_PRELOAD alternative to tcmalloc.

### Correction — the live trainer changed at 05:06:59, mid-experiment-1

`train4m.py` was edited by the sweep owner (sl5nw) **while job 6536785 was running**: `main()` now
calls `torch.set_num_interop_threads(1)` ("throughput research 2026-08-13, gated bit-exact on all
three configurations"). The owner's session is evidently running the same efficiency task in
parallel and applies its findings directly to the live trainer; mine deliberately does not touch it.

Which of my measurements this touches (condition start times vs the 05:07 edit):

| config | conditions on the pre-edit trainer | conditions on the post-edit trainer |
|---|---|---|
| `alg23` | all seven (03:45-04:16) | — |
| `gtposvel` | all seven (04:21-04:47) | — |
| `run5origrnd` | baseline 04:52, `optflags` 04:57, `adamforeach` 05:02 | **`normcache` 05:07, `gcfreeze` 05:12, `tcmalloc` 05:17, `combined` 05:22** |

So the four bolded `run5origrnd` cells were measured against a baseline that lacked an optimization
they had — they are **too generous by about the value of that change**. Its size, from my own
baselines on the same node and config: `alg23` baseline 32.160 (pre-edit) vs 32.301 / 32.451
(post-edit, job 6536795) = **+0.4% to +0.9%**. So `run5origrnd`'s `tcmalloc` +6.9% is really ~+6.4%
and its `combined` +9.7% is really ~+9.2% — leaving the headline unchanged (`alg23` and `gtposvel`,
both fully clean, gave +9.2% and +8.4%), but the row is corrected in `applied_changes.md` and
re-measured from scratch in experiment 3.

Two further consequences:
- **`interop1` leaves my candidate list**: it is now in the trainer itself, and torch raises when it
  is set twice, which is exactly why both `interop1` runs failed with rc=1 after 5 s. The patch is
  now a tolerated no-op.
- **Every later measurement must run on one code version.** Experiment 3 re-measures `baseline`,
  `combined` and `combined2` on all three configs on the current trainer, so the final numbers come
  from one consistent build.

### Results — experiment 2 (job 6536795, alg23, 2 repeats each)

| condition | run 0 | run 1 | mean vs baseline | record |
|---|---|---|---|---|
| `baseline` | 32.301 | 32.451 | — (spread **0.46%**) | IDENTICAL to each other |
| `interop1` | rc=1 | rc=1 | — | failed: already applied upstream |
| `mkldnnoff` | 32.358 | 32.600 | +0.3% | IDENTICAL |
| `infmode` | 32.103 | 32.132 | **-0.8%** | IDENTICAL |
| `arena1` | 32.357 | 32.337 | -0.1% | IDENTICAL |
| `combined` | 35.115 | _pending_ | **+8.5%** | IDENTICAL |

Two things this settles:

- **The measurement is precise.** `combined` came out 35.130 in job 6536785 and 35.115 here — 0.04%
  apart, on different jobs hours apart. Baseline repeats spread 0.46%. So differences above ~0.5%
  are signal and the ~1% candidates (`adamforeach`, `gcfreeze`) are real, if small.
- **Three candidates are dead.** `arena1` (-0.1%) does not reproduce tcmalloc's win, so the gain is
  tcmalloc's allocator, not merely fewer arenas. `infmode` is **slower** (-0.8%): the `.clone()`
  needed to hand a normal tensor back to autograd-tracked code costs more than the version-counter
  bookkeeping it saves. `mkldnnoff` (+0.3%) is inside the baseline spread on this config — carried
  into experiment 3 as part of `combined2` to see whether it survives on all three.

Experiment 2 complete (job 6536795, COMPLETED 00:51:56). `combined` repeats: 35.115 / 35.023 =
**+8.6% mean** over the post-edit baseline (32.301 / 32.451), every run IDENTICAL.

## Experiment 3 — final numbers on one consistent build (job 6536801, cortado01 exclusive)

`--conditions baseline,combined,combined2 --repeat 2` on all three configs (18 runs). Purpose:
(1) replace the four `run5origrnd` cells that straddled the 05:07 trainer edit, (2) settle whether
`mkldnnoff` earns its place in the stack (`combined2` = `combined` + oneDNN off) on configs other
than `alg23`, and (3) give the deliverable a repeated measurement of the exact stack that would be
applied, on the exact trainer that is running.

### Experiment 3 interim (job 6536801) — and a noise lesson

| config | baseline | `combined` | `combined2` (+oneDNN off) |
|---|---|---|---|
| `alg23` | 32.236 / **29.003** | 34.866 / 35.111 | 35.089 / 35.070 |
| `gtposvel` | 38.125 / 38.051 | 41.046 / 41.109 | 40.834 / _pending_ |

**`alg23 baseline#1` = 29.003 steps/s is an outlier** — 10% below its own twin five minutes earlier,
on a node this job holds exclusively, while every other pair in the job agrees to ~0.5%. The node is
exclusive but the *filesystem* is not: 450 sweep runs are writing records to the same
`/p/rlprojects` mount. So the measurement floor is not perfectly clean, and a mean over two runs is
the wrong statistic — one hiccup drags it 5%.

**Rule adopted for the rest of this campaign: compare the FASTEST run of each condition.** Timing
noise is one-sided (interference only ever slows a run down), so best-of-N estimates the
un-perturbed cost and is not distorted by an outlier. On that basis:

| config | baseline (best) | `combined` (best) | gain | `combined2` (best) | gain |
|---|---|---|---|---|---|
| `alg23` | 32.236 | 35.111 | **+8.9%** | 35.089 | +8.8% |
| `gtposvel` | 38.125 | 41.109 | **+7.8%** | 40.834 | +7.1% |

**`mkldnnoff` is dropped.** `combined2` never beats `combined` — equal on `alg23`, 0.7 pp worse on
`gtposvel` — so oneDNN is earning its dispatch cost on these shapes after all, and the +0.3% seen
once on `alg23` in experiment 2 was inside that config's own spread. It leaves the stack; the applied
set stays the six items of `applied_changes.md`.

### Experiment 3 complete (job 6536801, COMPLETED 01:25:09) — the clean, final single-worker numbers

All 18 runs `IDENTICAL`. Best-of-2 per condition (the outlier rule above):

| config | baseline | `combined` | **gain** | `combined2` (+oneDNN off) |
|---|---|---|---|---|
| `alg23` | 32.236 | 35.111 | **+8.9%** | 35.089 (+8.8%) |
| `gtposvel` | 38.125 | 41.109 | **+7.8%** | 40.906 (+7.3%) |
| `run5origrnd` | 33.448 | 36.451 | **+9.0%** | 36.419 (+8.9%) |

`run5origrnd` re-measured clean gives **+9.0%**, against the +9.7% its contaminated experiment-1 row
claimed — within 0.2 pp of the +9.2% predicted when the correction was worked out, which is a decent
check on that reasoning.

**Final single-worker verdict: +7.8% to +9.0% (mean +8.6%), record-identical on every configuration,
measured with repeats on one consistent build.** `combined2` is dropped.

## Experiment 4 — the packed measurement (job 6536802, cortado01 exclusive, 46 workers)

Every number so far is a lone worker on an idle 24-core node. The sweep does not run that way: it
packs ~46 single-threaded workers per cortado node, two per physical core, all sharing the L3 and
the memory bus. An allocator change is precisely the kind of thing that can behave differently
there, so the stack's value to the sweep is whatever it measures HERE.

46 workers, 3,000 steps each, `baseline` then `combined`; worker i runs marker i%3 at seed
(base + i), so the packed job is a slice of the real sweep and worker i's record still has a
matching baseline to be compared against.

**Experiment 4 attempt 1 (job 6536802) is void — two of my own mistakes, recorded so the number in
its log is never quoted.**

1. Every worker exited at startup with `ckpt_every 100000000 must be a multiple of eval_freq 1500`
   (100,000,000/1,500 is not an integer). The earlier jobs survived only because 2,500 happens to
   divide it. `marker_args` now derives `ckpt_every = eval_freq * 1_000_000`, so the constraint holds
   for any cadence.
2. The job was then killed `OUT_OF_MEMORY`: `--ntasks=1` with no `--mem` on this partition means
   `DefMemPerCPU=256M`, i.e. 256 MB for what was supposed to be 46 torch processes. The packed job
   now asks for `--mem=200G`.

Its log prints `combined vs baseline: x1.1910 aggregate`. **That number is meaningless** — it is the
ratio of how fast 46 processes crashed, over 11.4 s and 9.6 s, with `identical=0/46` and no records
written. It is not evidence of anything and is superseded by job 6536803.

Smoke-tested the fix at 2 workers × 300 steps before resubmitting: `identical=2/2`, per-worker
37.2 steps/s.

### Experiment 4 — packed, the shape the sweep actually runs (job 6536803, COMPLETED 00:06:51)

46 single-threaded workers on one exclusive 24-core node, 3,000 steps each, distinct seeds.

| condition | wall for all 46 | **aggregate steps/s** | per-worker median | records identical |
|---|---|---|---|---|
| `baseline` | 213.7 s | 645.79 | 15.615 | **46/46** |
| `combined` | 196.1 s | 703.72 | 17.162 | **46/46** |
| | | **×1.0897 (+9.0%)** | +9.9% | |

**The gain survives production packing — it is if anything slightly larger there (+9.0% aggregate)
than on an idle node (+8.6%), and all 46 workers still produced records identical to their
baselines.** That was the open question: an allocator swap is exactly the sort of change that can
behave differently when 46 processes share the L3 and the memory bus, and the answer is that it does
not lose its edge.

Two other things this measurement pins down:

- **What packing costs a worker.** 15.6 steps/s packed versus 32-38 idle: each worker runs at ~46%
  of its idle speed, while the node delivers 646 steps/s instead of ~35. Packing is right, by a
  factor of ~18 in throughput per node.
- **What the stack is worth to the sweep, in the units that matter.** At the packed baseline rate a
  4M-step run takes 4,000,000 / 15.615 = **71.2 h**; with the stack, 4,000,000 / 17.162 = **64.7 h**.
  **~6.5 hours per run, ~5,850 worker-hours over the sweep's 900 runs** — and not one recorded
  number changes.
  (The 71.2 h baseline also confirms the sweep design's own 60-76 h estimate for a packed 4M run.)

## Experiment 5 — the same, four times the horizon (job 6536805)

46 workers × 12,000 steps. Everything measured so far covers the first 10,000 steps of a 4,000,000
step run, so this asks whether the gain is a startup artefact: 4x the horizon, still in the packed
production regime, still with the record check on all 46 workers.

What it cannot reach is the LATE-run regime (a full 1M-transition replay buffer, a large resident
set at step 2M). The applied changes are per-step overheads that do not depend on horizon — with the
one exception of the allocator, which is precisely the item this packed test exercises hardest —
so the expectation is a flat result; it is measured rather than assumed.

Code version pinned for experiments 3-5: `train4m.py` as of 05:06:59 (the owner's interop change),
`train.py` and `rnd.py` untouched since 2026-08-01. Verified before submitting.

### Experiment 5 — 4x the horizon, packed (job 6536805, COMPLETED 00:26:41)

46 workers × 12,000 steps.

| condition | aggregate steps/s | per-worker median | records identical |
|---|---|---|---|
| `baseline` | 668.52 | 15.543 | **46/46** |
| `combined` | 712.27 | 16.702 | **46/46** |
| | **×1.0654 (+6.5%)** | +7.5% | |

**The gain shrinks with horizon, and that has to be said plainly: +9.0% at 3,000 steps became +6.5%
at 12,000.** Validity is untouched — 46/46 identical again, now over 4x as many steps — but the
speed claim is not horizon-independent the way I expected when I wrote "these are per-step overheads".

The likely reason is that the early steps are the allocation-heavy ones: the replay buffer is still
growing, the observation statistics are still warming up, and `gc.freeze()` protects a small heap
that later grows past it. Whatever the mechanism, the honest reading is that **the measured gain
decreases as the run lengthens**, and every measurement so far covers at most 0.3% of a 4M run.

Revised, more conservative production estimate (from the 12,000-step packed numbers):
4,000,000 / 15.543 = **71.5 h** baseline; 4,000,000 / 16.702 = **66.5 h** with the stack ⇒ **~5.0 h
saved per run, ~4,500 worker-hours over 900 runs** (rather than the 6.5 h / 5,850 h the 3,000-step
measurement suggested).

## Experiment 6 — a third point on the horizon curve (job 6536808)

46 workers × 40,000 steps (≈3.3x experiment 5, ≈13x experiment 4). Purpose: decide whether the gain
is still falling or has flattened, because the difference between quoting "+9%" and "+6.5%" and
"+5%" for the 96-hour run is exactly this curve. ~50 min per condition.

### Experiment 6 — 40,000 steps, packed (job 6536808, COMPLETED 01:28:05) — and it overturns experiment 5's conclusion

| condition | aggregate steps/s | per-worker median | records identical |
|---|---|---|---|
| `baseline` | 667.11 | 15.366 | **46/46** |
| `combined` | 728.17 | 16.821 | **46/46** |
| | **×1.0915 (+9.2%)** | +9.5% | |

The horizon curve, all packed, 46 workers:

| steps per worker | aggregate gain | per-worker gain | baseline per-worker |
|---|---|---|---|
| 3,000 | +9.0% | +9.9% | 15.615 |
| 12,000 | +6.5% | +7.5% | 15.543 |
| **40,000** | **+9.2%** | **+9.5%** | 15.366 |

**Correction: the gain does not shrink with horizon.** Last tick's reading — "+9.0% at 3,000 became
+6.5% at 12,000, the gain decays" — was drawn from a two-point line, and the third point lands back
at +9.2% after a 13x longer run. The 12,000-step figure is a single-measurement dip, of exactly the
kind already documented in experiment 3 (one-sided interference from the shared filesystem; the
`combined` arm there measured 16.702 per worker against 17.162 and 16.821 either side of it). The
baseline arm is steady across all three horizons (15.615 / 15.543 / 15.366), which is what makes the
dip attributable to that one measurement rather than to the horizon.

Two lessons kept for the record: a trend needs three points, not two; and with this noise floor a
single packed pair is worth ~±3%, so only differences of that size or larger should be read as real.

**Final answer: +9% (≈+9.0 to +9.2% aggregate under production packing), record-identical in every
one of the 46+46+46+46 packed worker comparisons and all 39 single-worker comparisons.**

Production numbers from the longest, most production-like measurement (40,000 steps, packed):

| | per worker | one 4M run | 900 runs |
|---|---|---|---|
| baseline | 15.366 steps/s | **72.3 h** | — |
| with the applied stack | 16.821 steps/s | **66.1 h** | — |
| saved | +9.5% | **6.2 h per run** | **~5,600 worker-hours** |

---

## Sweep event, 2026-08-13 13:04-13:05 (observed, not caused by anything here)

The sweep owner cancelled the **entire** running worker fleet and submitted a new one:

- 450 orphan reclaims in the owner's ledger (279 at 13:04 + 171 at 13:05), i.e. every in-flight run
  went back to `pending/`;
- 22 new jobs (6536863-6536884) started at ~13:05 on affogato02, bigcat01-06, slurm2/3 and panther01
  (a node the previous waves did not use), and re-claimed all 450 within about two minutes;
- queue back to 450 running / 450 pending / 0 done / **0 failed**; cumulative owner job states
  39 CANCELLED / 18 RUNNING / 5 PENDING.

**The checkpoint design did exactly what it exists for.** Every interrupted run resumed from its most
recent 0.5M-step checkpoint (444 checkpoints were on disk at the time), so the loss is bounded by the
steps since that boundary — the runs were at roughly 0.5-0.6M steps after ~9.8 h, so on the order of
1 h per run rather than the whole 9.8 h.

**It was not a relaunch for the optimizations**, and that is the part worth recording: at 13:07
`slurm/ext4m_worker_cpu.slurm` is still the 02:28 file with no `LD_PRELOAD`, the pending markers
still carry no `opt_polyak_foreach`, and `train4m.py` is unchanged since 05:06:59. A fleet relaunch
is the one moment when both no-code items cost nothing to add — one export line in the worker script
and one pass of `apply_optflags.py` — so this relaunch passed up ~9% for the ~66 h that remains of
every one of those 450 runs.
