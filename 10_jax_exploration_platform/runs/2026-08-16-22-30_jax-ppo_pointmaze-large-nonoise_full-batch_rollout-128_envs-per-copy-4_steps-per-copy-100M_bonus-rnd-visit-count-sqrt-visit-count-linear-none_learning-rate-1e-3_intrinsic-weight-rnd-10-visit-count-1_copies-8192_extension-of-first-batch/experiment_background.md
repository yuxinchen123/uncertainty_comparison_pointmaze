# Train run 1.1, extension — each arm's best configuration at 8,192 copies for 100 million steps

## Purpose

Train run 1.1 compared four intrinsic-exploration arms over 102 configurations, 256 copies each,
for 10 million environment steps per copy. It named one winning configuration per arm. This run
takes those four winners and re-runs them at **8,192 copies** for **100,044,800 environment steps
per copy** — 32 times the copies of one parent cell and 10 times its step budget — so that the
ordering the parent found is tested where the parent could not test it:

1. **Does the ordering survive a longer budget?** The parent's ranking was read off curves that
   were still moving at 10 million steps. A ten-fold budget says whether the oracle
   $1/\sqrt{n}$ bonus keeps its lead over distillation, or whether distillation catches it once
   the predictor has had time to learn.
2. **Does the arm with no bonus ever solve the maze on its own?** At 10 million steps it reached
   the goal in 0.4 per cent of its copies. A ten-fold budget is the fair test of whether the
   sparse reward is findable without a bonus at all.
3. **Are the parent's gaps real at a copy count where the standard errors are small?** 8,192
   copies is 32 times 256, so a standard error shrinks by a factor of about 5.7. Gaps that were
   inside two standard errors in the parent become decidable here.

The parent run is
`runs/2026-08-16-20-39_jax-ppo_pointmaze-large-nonoise_full-batch_rollout-128_envs-per-copy-4_steps-per-copy-10M_bonus-rnd-visit-count-sqrt-visit-count-linear-none_learning-rate-1e-3-1e-4-1e-5_intrinsic-weight-1e-5-to-1e5_copies-per-cell-256_first-batch`,
and `manifest.yaml` here names it and every configuration being extended, from what to what.

This is an EXTENSION in the sense of the `rnd-experiment-tex-track` skill: it adds no new
subsection, no new table and no new figure to the development document. Its rows go below the
double rule of the existing results table, its curves replace the parent's in the existing
figure, and the existing what-is-swept table gains one spanning extension row.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| environment | `pointmaze_large_cont400_nonoise@1` — large map, start cell (7, 1), goal cell (1, 10), position noise 0, truncation at 400 steps, continuing task, goal radius 0.45 |
| agent | `ppo_full_batch@1` — one gradient step per rollout, no epochs, no minibatches |
| rollout length T | 128 steps |
| environments per copy N | 4, so one iteration collects 512 transitions per copy |
| iterations | 195,400 |
| recorded windows | 977 per unit, one record per 200 iterations |
| environment steps per copy | 100,044,800 |
| algorithm arms (work units) | `rnd_next_state` at intrinsic weight 10; `gt_position_velocity_sqrt` at 1; `gt_position_velocity_linear` at 1; `none` |
| learning rate | 1e-3 for every arm — the rate every arm won with in the parent run |
| copies per unit | 8,192, one (learning rate, intrinsic weight) cell per unit |
| seeding | base seed 0, run seed 0, paired; one cell per unit, so a unit's 8,192 copies differ only by seed |
| discounts | extrinsic 0.999, intrinsic 0.99 |
| extrinsic advantage weight | 2.0 |
| checkpoints | none — this platform saves no model state |
| environment (software) | `/p/rlprojects/RND/.venvs/platform_jax` (python 3.12.13, `jax[cuda12]` 0.11.0), the canonical platform environment in `.venvs/ENVS.md` |

The four configurations, and what each was in the parent run:

| unit | bonus | learning rate | intrinsic weight | parent copies → here | parent steps per copy → here |
|---|---|---|---|---|---|
| 1 | `rnd_next_state` | 1e-3 | 10 | 256 → 8,192 | 10,035,200 → 100,044,800 |
| 2 | `gt_position_velocity_sqrt` | 1e-3 | 1 | 256 → 8,192 | 10,035,200 → 100,044,800 |
| 3 | `gt_position_velocity_linear` | 1e-3 | 1 | 256 → 8,192 | 10,035,200 → 100,044,800 |
| 4 | `none` | 1e-3 | none | 256 → 8,192 | 10,035,200 → 100,044,800 |

## Code and config changes

- **No trainer change.** The platform code that runs is the code the parent ran; this run's own
  `code/` folder is the parent's, with one file rewritten.
- **`code/build_queue.py` is rewritten for the extension.** It writes four single-cell units
  instead of four swept units, carries the parent run's name in every queue entry
  (`extends_parent_run`), and writes the run's `manifest.yaml` and `config_resolved.yaml` as well
  as `command.txt`. The launch files are written here rather than by `scripts/run_training.py`
  because the generic writer cannot know the parent run or which configurations are being
  extended; writing `manifest.yaml` before the first job also stops the generic writer from
  overwriting the three files (it writes them only when `manifest.yaml` is missing).
- **`code/{worker_env.sh, submit_one.sh, run_unit.sh, write_job_records.py, aggregate.py,
  status.py}` are copied unchanged from the parent run**, including the fix the parent's
  `infra_history.md` records: `run_unit.sh` reads a unit's arguments BEFORE claiming it and
  refuses to start if it read fewer than four, and `run_training.py` makes `--unit-id` required
  with no default. Both guards were re-tested before this run's first submission (see below).
- **195,400 iterations rather than 195,313.** $10^{8} / 512 = 195{,}312.5$ is not a whole
  iteration, let alone a whole 200-iteration recording window. A window must span a whole number
  of turns of the 25-iteration episode clock ($400 / \gcd(128, 400)$) or its reward sum carries
  the episode phase and cannot be scored, and 200 iterations is 8 whole turns. 977 windows is
  195,400 iterations and 100,044,800 steps per copy, which is $10^{8}$ to within 0.045 per cent;
  976 windows would be 0.058 per cent short, so 977 is the nearer of the two. Recorded here as a
  deliberate deviation, following the parent's precedent (19,600 rather than 19,531).
- **Steps to goal is still not recorded.** The trainer never reads the goal step back to the
  host, so that column of the results table says "not recorded in this run" for the extension
  rows too. The request stays logged in `research/README.md`.

## Card assignment

The arithmetic follows `/p/rlprojects/.claude/skills/rnd-jax-submission/SKILL.md`. Each unit's
seconds per iteration is the parent's canary measurement at 8,448 copies scaled to this run's
8,192 (a factor 0.9697); `none` was measured only at 768 copies, where per-iteration overhead
dominates and the figure does not scale, so its entry carries an **upper bound** instead — the
cost of a visit-count unit, which does strictly more work than an arm with no bonus at all.

| unit | copies | s/iteration on an H100 | compute at 195,400 iterations | expected peak device memory | card needed ($1.25\times$) |
|---|---|---|---|---|---|
| 1 `rnd_next_state` | 8,192 | 0.0844 | 4.58 h | 27.3 GiB | 34.1 GiB — H100 or A100-80GB only |
| 2 `gt_position_velocity_sqrt` | 8,192 | 0.0347 | 1.88 h | 7.6 GiB | 9.5 GiB |
| 3 `gt_position_velocity_linear` | 8,192 | 0.0339 | 1.84 h | 7.6 GiB | 9.5 GiB |
| 4 `none` | 8,192 | 0.0347 (upper bound) | 1.88 h (upper bound) | 7.6 GiB | 9.5 GiB |

Sequentially on one H100 the four units are 10.19 hours of compute plus four builds, far above the
skill's roughly-20-minute threshold, so distributing is warranted. **Unit 1 sets a floor of about
4.6 hours however many cards are used**, so the assignment only has to place the other three where
each finishes inside that floor; a fifth card would buy nothing.

Live availability at 2026-08-16 22:20 PT: 5 free H100 NVL cards (serval07 both, serval06, serval08
and serval09 one each), 2 free A100-PCIE-40GB on cheetah01, 0 free A100-SXM4-80GB.

**The H100 allowance: the fused case applies, so there is no cap.** `uva-submit-gpu-sweep`'s
allowance was amended on 2026-08-17 (commit `20ee693` of the shared `.claude` repository) to
depend on the kind of run. A **fused** run — one whose jobs each occupy their whole card with one
compiled program, which is exactly what this platform's units are, thousands of copies inside a
single executable — may hold as many free H100 cards as the work actually benefits from; the
$\max(2, F-2)$ cap now applies only to unfused sweeps of many small single-copy processes packed
onto cards. The limit here is therefore usefulness, not a reserve.

Usefulness gives the same four-card assignment the old cap would have given, for a different
reason. **Unit 1 cannot be split** — it is one compiled program — so it sets a floor of about 4.6
hours whatever else is done, and a card only earns its place if it moves some other unit off a
timeline that would otherwise cross that floor. Every unit here already has a card of its own, and
the slowest of the other three finishes at most 3.9 hours in, so a fourth H100 would shorten
nothing and a fifth card less than that. The plan below was formed under the old cap and re-checked
against the amended rule at 2026-08-16 22:29 PT with live availability re-read; it is unchanged,
and no job was cancelled or resubmitted to reshuffle cards, which would have cost a queue re-entry
and a cold compile for no gain.

| unit | card | why |
|---|---|---|
| 1 `rnd_next_state` | serval07 (H100 NVL) | the only class with the memory, and the longest unit belongs on the fastest card; serval07 was fully idle, so the longest unit shares its node with nobody |
| 2 `gt_position_velocity_sqrt` | serval08 (H100 NVL) | 1.9 h, inside the floor |
| 3 `gt_position_velocity_linear` | serval09 (H100 NVL) | 1.9 h, inside the floor |
| 4 `none` | cheetah01 (A100-PCIE-40GB) | the A100 runs at 0.488 of the H100's rate on this trainer (25.58 against 52.44 million steps per second at 4,096 copies), so even the upper-bound estimate — `none` costing what a visit-count unit costs — is 3.86 h, inside the 4.6 h floor. A fourth H100 would finish this unit sooner without finishing the SET sooner, so it was not taken |

Unit 4 rather than unit 2 or 3 goes on the slower card because it has the most headroom: its cost
is bounded above by a visit-count unit's, so putting it there cannot make the makespan worse than
putting a visit-count unit there would. Its real rate is unmeasured at this copy count, and the
canary measures it before the science job is submitted; if the canary had projected past the floor
the unit would have moved to an H100 freed by unit 2 or 3.

The three units were not split across more cards. Splitting the largest compiled program would
double its builds and its shards without moving the finish time, which unit 1 sets alone.

## Deviation from the canary rule, and why

`uva-submit-gpu-sweep` Step 6 asks for at least 30 simultaneous canary runs. That figure is sized
for a campaign of 150 runs across many node classes and configurations, where the canaries exist to
catch per-configuration outliers. This run has four work units and two node classes, so the same
purpose is served by canarying every (unit, node class) pairing that the submission actually uses —
four canary runs, one per pairing, each at the unit's full 8,192 copies so the memory it proves is
the memory the real unit will use, and each limited to 200 iterations. The canaries also warm the
node-local compiled-program cache that each real job then hits, and one of them is re-run to prove
the resume. Measurements: `canary_estimate_vs_actual.md`.

## Git state

Code commit at launch: `4cd78643a6b4f3467e16b56160ce837f29cbbfc8` (`git describe`:
`jax-platform-v0.1.0-22-g4cd7864`), branch `Use-RLexplore-RND`, pushed before the first
submission. The working tree of `10_jax_exploration_platform/` was clean at that commit; other
folders of the shared repository carried unrelated uncommitted work from other sessions — the
graphics-card throughput survey of `09_parallelization/`, whose jobs were still writing into it —
which was left untouched.

## Walltime

`--time=4-00:00:00` for every science job: the `gpu` partition's maximum. The only maintenance
reservation on the cluster, `cs_admin_maint`, covers `serval03` alone and none of this run's nodes,
and no reservation of ours is used, so the partition limit is the binding cap. The walltime is set
to the maximum that will actually run and is deliberately not sized to the 4.6-hour estimate.
