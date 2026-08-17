# Train run 1.1, extension — each arm's best configuration at 1,024 copies for a thousand million steps

## Purpose

Train run 1.1 compared four intrinsic-exploration arms over 102 configurations, 256 copies each, for
10 million environment steps per copy, and named one winning configuration per arm. This run takes
those four winners and re-runs them at **1,024 copies** for **1,000,038,400 environment steps per
copy** — four times the copies of one parent cell and about a hundred times its step budget — so the
ordering the parent found is tested where the parent could not test it:

1. **Does the ordering survive a hundred-fold budget?** The parent's ranking was read off curves
   still rising at 10 million steps. A budget of $10^{9}$ steps says whether the oracle
   $1/\sqrt{n}$ bonus keeps its lead over distillation, or whether distillation catches it once the
   predictor has had time to learn what the oracle was given for free.
2. **Does the arm with no bonus ever solve the maze on its own?** At 10 million steps it reached the
   goal in 0.4 per cent of its copies. A hundred-fold budget is the fair test of whether the sparse
   reward is findable without a bonus at all.
3. **Are the parent's gaps real once the standard errors are small?** 1,024 copies is four times
   256, so a standard error shrinks by a factor of two, and the run is long enough that the
   end-of-run numbers are not a snapshot of a curve mid-climb.

The parent run is
`runs/2026-08-16-20-39_jax-ppo_pointmaze-large-nonoise_full-batch_rollout-128_envs-per-copy-4_steps-per-copy-10M_bonus-rnd-visit-count-sqrt-visit-count-linear-none_learning-rate-1e-3-1e-4-1e-5_intrinsic-weight-1e-5-to-1e5_copies-per-cell-256_first-batch`,
and `manifest.yaml` here names it and every configuration being extended, from what to what.

This is an EXTENSION in the sense of the `rnd-experiment-tex-track` skill: it adds no new
subsection, no new table and no new figure to the development document. Its rows go below the double
rule of the existing results table, its curves replace the parent's in the existing figure, and the
existing what-is-swept table gains one spanning extension row.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| environment | `pointmaze_large_cont400_nonoise@1` — large map, start cell (7, 1), goal cell (1, 10), position noise 0, truncation at 400 steps, continuing task, goal radius 0.45 |
| agent | `ppo_full_batch@1` — one gradient step per rollout, no epochs, no minibatches |
| rollout length T | 128 steps |
| environments per copy N | 4, so one iteration collects 512 transitions per copy |
| iterations | 1,953,200 |
| recorded windows | 9,766 per chunk, one record per 200 iterations |
| environment steps per copy | 1,000,038,400 |
| algorithm arms | `rnd_next_state` at intrinsic weight 10; `gt_position_velocity_sqrt` at 1; `gt_position_velocity_linear` at 1; `none` |
| learning rate | 1e-3 for every arm — the rate every arm won with in the parent run |
| copies per arm | 1,024 |
| work units and chunks | 4 units; the distillation unit runs as 2 chunks of 512 copies, the other three as one chunk of 1,024 each (the submission plan's arithmetic, below) |
| seeding | base seed 0; a chunk's `--copy-seed-offset` is its first copy index and its `--run-seed` is its chunk number |
| discounts | extrinsic 0.999, intrinsic 0.99 |
| extrinsic advantage weight | 2.0 |
| checkpoints | none — this platform saves no model state |
| environment (software) | `/p/rlprojects/RND/.venvs/platform_jax` (python 3.12.13, `jax[cuda12]` 0.11.0), the canonical platform environment in `.venvs/ENVS.md` |

The four configurations, and what each was in the parent run:

| unit | bonus | learning rate | intrinsic weight | parent copies → here | parent steps per copy → here |
|---|---|---|---|---|---|
| 1 | `rnd_next_state` | 1e-3 | 10 | 256 → 1,024 | 10,035,200 → 1,000,038,400 |
| 2 | `gt_position_velocity_sqrt` | 1e-3 | 1 | 256 → 1,024 | 10,035,200 → 1,000,038,400 |
| 3 | `gt_position_velocity_linear` | 1e-3 | 1 | 256 → 1,024 | 10,035,200 → 1,000,038,400 |
| 4 | `none` | 1e-3 | none | 256 → 1,024 | 10,035,200 → 1,000,038,400 |

## Code and config changes

- **The platform gained one knob: `copy_seed_offset`.** A copy's seed index keys both its initial
  weights and the environment draws it meets, and it used to be `0 .. copies-1` always. It is now
  `offset .. offset + copies - 1`, which is what lets one logical run of 1,024 copies be cut into
  chunks that run on separate cards: chunk *j* of *k* passes `--copy-seed-offset j x (1024/k)` and
  holds exactly that slice of the whole run's copies. Offset 0 is the old behaviour, so no earlier
  run's draws move — pinned by `tests/agents/test_copy_seed_offset.py`, which also checks that a
  chunk's weights are bitwise the corresponding slice of the whole run's weights, and the existing
  `tests/agents/test_sweep_jax.py` still passes unchanged. `scripts/run_training.py` takes the
  argument and records `copy_seed_offset`, `copy_seed_index_first` and `copy_seed_index_last` in
  every `unit_start`, so a shard says which copies it holds.
- **A chunk's policy sampling stream is its own.** The action noise is drawn from one key per
  process, so two chunks with the same `--run-seed` would draw the same noise for their copy 0. Each
  chunk therefore passes its chunk number as `--run-seed`. The consequence, stated plainly: the
  1,024 copies of a unit have the initial weights and the environment draws copies 0..1,023 of a
  single 1,024-copy run would have had, and their action noise comes from *k* separate streams
  rather than one — no copy shares randomness with another copy, which is the property the pooling
  needs.
- **`code/plan_submission.py` is new: the submission strategy is computed, not judged.** See the
  next section.
- **`code/probe_rates.sh`, `code/submit_probe.sh`, `code/measured_cells.py` are new**: they measure
  this run's own arms at 256, 512 and 1,024 copies on the cards in question and write the table the
  planner prefers over the shared survey.
- **`code/build_queue.py` writes one queue entry per CHUNK** from the plan, refuses to write a queue
  whose chunks do not partition each unit's 1,024 copies, and writes `manifest.yaml`,
  `config_resolved.yaml` and `command.txt` (which also stops `scripts/run_training.py` from
  overwriting them with its own generic versions).
- **`code/aggregate.py` pools chunks into configurations.** A configuration's numbers are taken over
  its chunks' concatenated per-copy lists, so the mean and the standard error are over all 1,024
  copies rather than an average of two chunk averages; a configuration is scored only when every one
  of its chunks has finished; and the chunks' copy ranges are checked to partition the 1,024. Proved
  end to end on real shards by `code/test_chunk_partition_end_to_end.py`, which trains two chunks of
  a 16-copy toy unit through the real runner and checks the aggregate has all 16 copies once.
- **`code/{worker_env.sh, submit_one.sh, run_unit.sh, write_job_records.py, status.py}` are copied
  unchanged from the parent run**, including the fix its `infra_history.md` records: `run_unit.sh`
  reads a chunk's arguments BEFORE claiming it and refuses to start if it read fewer than four.
- **1,953,200 iterations rather than 1,953,125.** $10^{9} / 512 = 1{,}953{,}125$ is not a whole
  200-iteration window. A window must span a whole number of turns of the 25-iteration episode clock
  ($400 / \gcd(128, 400)$) or its reward sum carries the episode phase and cannot be scored, and 200
  iterations is 8 whole turns. 9,766 windows is 1,953,200 iterations and 1,000,038,400 steps per
  copy, which is $10^{9}$ to within 0.0038 per cent; 9,765 windows would be 0.0064 per cent short,
  so 9,766 is the nearer of the two. Recorded here as a deliberate deviation, following the parent's
  precedent.
- **Steps to goal is still not recorded.** The trainer never reads the goal step back to the host,
  so that column says "not recorded in this run" for the extension rows too.

## The submission plan, and how it was computed

The tail the reader waits for is the LAST chunk to finish, so the quantity minimised is the
makespan. `code/plan_submission.py` enumerates every combination of per-unit splits (1, 2, 4 or 8
chunks), list-schedules the chunks onto the cards that are actually free, and takes the combination
whose last chunk lands earliest; among combinations within five minutes of the best it takes the one
with fewest chunks, which is the proportionality clause of `rnd-jax-submission` §2 done by
arithmetic. 81 combinations were priced. The full record, including every alternative's computed
finish time, is `code/submission_plan.json`; the tests of the arithmetic are
`code/test_plan_submission.py`.

**The H100 allowance did not bind, and the fused case is why.** Every job of this run occupies its
whole card with one compiled program holding hundreds of copies, which is the FUSED case of the
allowance amended on 2026-08-17 (commit `20ee693` of the shared `.claude` repository): a fused run
may hold as many free H100 cards as the work benefits from, and the $\max(2, F-2)$ cap now applies
only to unfused sweeps of many small single-copy processes. The plan therefore takes all five free
H100 cards. It does not take a sixth card of a lesser class, because the arithmetic says that would
not move the finish.

**The rates were measured for this run rather than transferred.** Two probe jobs (6538652 on
serval06, 6538653 on cheetah01) ran every arm for 200 real iterations at 256, 512 and 1,024 copies,
because the shared throughput survey stops at 512 copies and could not price a four-way split at
all. The measurement changed the answer twice over:

1. **The gain from halving a unit is much smaller at these copy counts than the survey suggests.**
   On an H100 the survey's trainer improves its per-copy rate by 1.59 times between 1,024 and 512
   copies; measured here, distillation improves by 1.43 and the other three arms by 1.25. A card at
   these sizes is not yet saturated, so cutting a unit in half buys much less than proportionally.
2. **An arm factor measured at 8,192 copies must not be carried to 1,024 copies.** The first plan
   scaled the survey's per-class rates by each arm's cost relative to the survey's trainer, measured
   at 8,192 copies where the bonus dominates an iteration (2.31 to 2.38 times for the three
   non-distillation arms). At 1,024 copies the four arms come within 1.7 times of each other, so
   that factor credited slow cards with rates they cannot reach — the first plan put two chunks on
   an RTX 5080 at an estimated 3.80 hours each, which is faster than the measurement says an H100
   is. The planner now carries only the survey's RATIO BETWEEN CARDS at the same copy count and
   keeps the arm and the copy count from its own probe, and it records per chunk which of the three
   sources priced it.

The chosen plan, and what it rejected:

| unit | arm | chunks | copies per chunk | copy indices | card | estimate |
|---|---|---|---|---|---|---|
| 1 | `rnd_next_state` | 2 | 512 | 0–511 and 512–1023 | serval06, both its cards | 5.31 h each |
| 2 | `gt_position_velocity_sqrt` | 1 | 1,024 | 0–1023 | serval08 | 4.58 h |
| 3 | `gt_position_velocity_linear` | 1 | 1,024 | 0–1023 | serval07 | 4.60 h |
| 4 | `none` | 1 | 1,024 | 0–1023 | serval09 | 4.54 h |

**Makespan 5.31 hours**, set by the distillation unit, on five H100 NVL cards.

| alternative | chunks | computed makespan | why it lost |
|---|---|---|---|
| nothing split | 4 | 7.55 h | the distillation unit alone is 7.55 h at 1,024 copies |
| **the chosen plan** | **5** | **5.31 h** | — |
| distillation in 4 chunks | 7 | 8.45 h | four of the five H100 cards go to one unit, and the other three units then have only one fast card between them |
| every unit in 2 chunks | 8 | 7.03 h | eight chunks on five cards means three cards run two chunks one after the other, and splitting also raises the total card-hours |
| every unit in 4 chunks | 16 | 10.71 h | the same effect, worse |

Two consequences worth stating outright. **Splitting is not free even when cards are idle**: a
512-copy chunk does not run twice as fast as a 1,024-copy one, so two chunks cost more card-hours
than one, and a split pays only when it moves work off the critical path onto a card that would
otherwise be idle. And **the card supply, not the rate curve, is what stops the splitting here**:
with five H100 cards and four units, one unit can be doubled and no more.

A split finer than four chunks (128 copies) is not priced by anything and was not planned on; the
plan's own output records that, with a bound on what it could have been worth.

## Walltime

`--time=4-00:00:00` for every science job: the `gpu` partition's maximum. The only maintenance
reservation on the cluster, `cs_admin_maint`, covers `serval03` alone and none of this run's nodes,
and no reservation of ours is used, so the partition limit is the binding cap. The walltime is the
maximum that will actually run and is deliberately not sized to the 5.3-hour estimate — at these
durations an under-set walltime is the most common way a long run dies near the end.

## Git state

Code commit at launch: `1e7c4495c39fc229ceb82f3c23f5f017042b19bc` (`git describe`:
`jax-platform-v0.1.0-24-g1e7c449`), branch `Use-RLexplore-RND`, pushed before the first science
submission. The working tree of `10_jax_exploration_platform/` was clean at that commit apart from
the development document, which is edited after a run's numbers exist; other folders of the shared
repository carried unrelated uncommitted work from other sessions and were left untouched.

## Deviation from the canary rule, and why

`uva-submit-gpu-sweep` Step 6 asks for at least 30 simultaneous canary runs, a figure sized for a
campaign of 150 runs across many node classes and configurations where the canaries catch per-config
outliers. This run has five chunks on one node class, so the same purpose is served by canarying
every (chunk, node) pairing the submission actually uses — five canary runs, each at the chunk's
full copy count so the memory it proves is the memory the real chunk will use, each limited to 200
iterations, and each on the exact node its chunk will run on so the node-local compiled-program
cache is warm when the real job starts. One is re-run to prove the resume. Measurements:
`canary_estimate_vs_actual.md`.

## The 8,192-copy attempt that preceded this run

An earlier attempt at this extension used 8,192 copies and 100 million steps per copy; its folder is
`runs/2026-08-16-22-30_..._copies-8192_extension-of-first-batch`. The specification changed to 1,024
copies and $10^{9}$ steps before its science jobs had written any window, and its four science jobs
were cancelled from its own id file. Nothing of it is used as science here. Two things of it are used
as measurement, and both are named where they are used: its canary rates at 8,192 copies are the
`ARM_ANCHORS` of `code/plan_submission.py` (the fallback path, which the probes then displaced), and
its `infra_history.md` records the missing-argument guards this run re-tested.

## Closing summary (written when the queue drained)

The queue drained at **2026-08-17 03:44 PT**, 4 hours 26 minutes after the first job started at
2026-08-16 23:19 PT. All five chunks completed on their first attempt; nothing was requeued and
nothing failed. The run moved $4.096 \times 10^{12}$ environment steps over 4,096 copies and wrote
48,830 window records, every one of them phase-blocked, so no record had to be dropped from a score.

| chunk | card | copies | seconds per iteration | total steps per second (millions) | steps per second per copy | hours per million steps per copy | build and prime (s) | wall clock (h) | planned (h) |
|---|---|---|---|---|---|---|---|---|---|
| unit-1 `rnd_next_state` chunk 1 of 2 | serval06 | 512 | 0.00811 | 32.33 | 63,150 | 0.0044 | 21 | 4.40 | 5.31 |
| unit-1 chunk 2 of 2 | serval06 | 512 | 0.00811 | 32.34 | 63,156 | 0.0044 | 21 | 4.40 | 5.31 |
| unit-2 `gt_position_velocity_sqrt` | serval08 | 1,024 | 0.00696 | 75.32 | 73,555 | 0.0038 | 16 | 3.78 | 4.58 |
| unit-3 `gt_position_velocity_linear` | serval07 | 1,024 | 0.00702 | 74.71 | 72,961 | 0.0038 | 15 | 3.81 | 4.60 |
| unit-4 `none` | serval09 | 1,024 | 0.00692 | 75.77 | 73,996 | 0.0038 | 16 | 3.75 | 4.54 |

**The makespan came in at 4.43 hours against a planned 5.31**, 17 per cent early, and every chunk
beat its own estimate by the same proportion. The estimate was built from 200-iteration canaries,
which carry the first iterations' transients in their steady rate; over 1.95 million iterations the
rate settles a little faster. An estimate wrong in this direction costs nothing, and the ORDERING
the plan rests on was right: the two 512-copy distillation chunks set the finish, exactly as the
plan said they would, and the three whole units finished 35 to 39 minutes earlier.

### What the run found

| arm | whole-run reward | last-window reward | success rate | maze-cell coverage % |
|---|---|---|---|---|
| `rnd_next_state`, $\beta = 10$ | 1.795 ± 0.101 | 1.242 ± 0.286 | 0.999 | 99.58 ± 0.06 |
| `gt_position_velocity_sqrt`, $\beta = 1$ | 0.658 ± 0.024 | 0.185 ± 0.118 | 0.999 | 99.07 ± 0.06 |
| `gt_position_velocity_linear`, $\beta = 1$ | 0.331 ± 0.030 | 0.171 ± 0.095 | 0.804 | 94.50 ± 0.35 |
| `no_exploration` | 0.001 ± 0.001 | 0.000 ± 0.000 | 0.016 | 33.36 ± 0.53 |

Two things the hundred-fold budget shows that ten million steps could not.

1. **Every arm's reward is a transient, and the parent measured its peak.** Mean episode return
   rises to a maximum inside the first 10 to 15 million steps and then decays for the remaining 985
   million: the $1/\sqrt{n}$ oracle arm peaks at 31.4 at 10M and ends at 0.19; distillation peaks at
   25.8 at 15M and ends at 1.24; the $1/n$ oracle arm peaks at 7.2 at 11M and ends at 0.17. Maze
   coverage meanwhile stays at 99 per cent and the success rate at 0.999 — the copies still reach
   every part of the maze and nearly all of them have reached the goal at some point, so this is not
   forgetting where the goal is. It is the same effect the parent's fourth finding named across the
   intrinsic weight, now seen across time: the bonus keeps paying after the goal is known, and the
   policy keeps chasing novelty instead of the reward.
2. **The ordering between the two leading arms reverses, and the reversal is not marginal.** At 10
   million steps the $1/\sqrt{n}$ oracle bonus beats distillation, 31.4 against 21.5. The curves
   cross at about 20 million steps and distillation is ahead for the remaining 98 per cent of the
   run, ending 6.7 times higher on the whole-run mean (1.795 ± 0.101 against 0.658 ± 0.024, a gap of
   more than 10 standard errors). The parent's conclusion that the oracle bonus bounds distillation
   from above holds only at the budget the parent used.

Neither of the two arms that lost at 10 million steps catches up. The $1/n$ oracle arm is below
distillation at every point of the run. The arm with no bonus never leaves the floor: 1.6 per cent
of its copies ever reach the goal, against 99.9 per cent for both leading arms, and it covers a
third of the maze against 99 per cent — a hundred times the budget does not make the sparse reward
findable without a bonus.

### Where the artifacts are

- `data/*.jsonl` (5 shards, 1.4 GB), `metrics.jsonl` (48,830 window records) and `summary.json`,
  written by `code/aggregate.py` from the shards.
- `analysis/plots/pm_jax_run11_reward_curves.pdf` and its `.png`, written by
  `analysis/code/make_reward_curves.py`.
- `../../development_document/platform_development_document.tex` subsection 1.1, whose results table
  is written by
  `development_document/code/2026-08-16-22-27_pm-jax-run11-extension-tables/make_all.py`.
- `code/submission_plan.json` for the plan and every alternative it rejected,
  `code/measured_cells.json` for the rates that priced it, `canary_estimate_vs_actual.md` for the
  canary phase, `infra_history.md` for the incidents.
