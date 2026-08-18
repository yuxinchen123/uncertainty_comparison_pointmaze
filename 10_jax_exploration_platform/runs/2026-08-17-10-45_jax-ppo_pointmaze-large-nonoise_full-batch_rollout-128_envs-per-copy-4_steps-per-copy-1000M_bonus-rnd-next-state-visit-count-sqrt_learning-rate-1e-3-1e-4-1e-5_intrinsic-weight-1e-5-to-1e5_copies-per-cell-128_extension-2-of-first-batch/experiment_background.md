# Train run 1.1, second extension — the two leading arms' whole grid at a thousand million steps

## Purpose

Train run 1.1 compared four intrinsic-exploration arms over 102 configurations, 256 copies each,
for 10 million environment steps per copy. Its first extension took one winning configuration per
arm to 1,000,038,400 steps per copy and found that random network distillation overtakes the oracle
$1/\sqrt{n}$ bonus once the budget passes about 20 million steps. That result rests on one
configuration per arm — the one each arm won with at the SHORT budget.

This run removes that dependence. It re-runs the **whole learning-rate by intrinsic-weight grid** of
the two arms that were still improving — `rnd_next_state` and `gt_position_velocity_sqrt` — at
**1,000,038,400 environment steps per copy**, 128 copies per configuration. Three questions:

1. **Is the crossing a property of the arms or of two configurations?** If distillation still leads
   when each arm is given its own best long-budget configuration rather than its best short-budget
   one, the crossing is about the bonus and not about a lucky pair of settings.
2. **Does the best configuration move with the budget?** The parent chose intrinsic weight 10 for
   distillation and 1 for the oracle bonus at 10 million steps. A hundred-fold budget may prefer a
   different weight, and the whole grid is the only way to see that rather than assume it.
3. **How flat is each arm's grid at the long budget?** A method whose result holds across three
   learning rates and eleven intrinsic weights is a different claim from one that holds at a point.

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
| algorithm arms | `rnd_next_state`; `gt_position_velocity_sqrt` |
| learning rates | 1e-3, 1e-4, 1e-5 |
| intrinsic weights | 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000 |
| configurations | 33 per arm, 66 in all |
| copies per configuration | 128 |
| copies per arm | 4,224 — one fused compiled program over all 33 configurations |
| work units and chunks | 2 units; the distillation unit runs as 32 chunks of 132 copies (4 per configuration), the oracle unit as 8 chunks of 528 copies (16 per configuration) — the submission plan's arithmetic, below |
| seeding | base seed 0, paired; a chunk's `--copy-seed-offset` is its first copy index and its `--run-seed` is its chunk number |
| discounts | extrinsic 0.999, intrinsic 0.99 |
| extrinsic advantage weight | 2.0 |
| checkpoints | none — this platform saves no model state |
| environment (software) | `/p/rlprojects/RND/.venvs/platform_jax` (python 3.12.13, `jax[cuda12]` 0.11.0), the canonical platform environment in `.venvs/ENVS.md` |

Total scale: 8,448 copies x 1,000,038,400 steps = about $8.4 \times 10^{12}$ environment steps,
roughly twice the first extension.

This is an EXTENSION in the sense of the `rnd-experiment-tex-track` skill: it adds no new
subsection, no new table and no new figure to the development document. Its rows form a second
extension block below the first extension's in the existing results table, the existing
what-is-swept table gains one more spanning extension row, and its curves go into the existing
figure.

## How a chunk stays a slice of the same run

A unit is one arm's 33 configurations fused into one compiled program, and it is cut by COPY INDEX,
not by configuration. Chunk *j* of *k* passes `--copy-seed-offset j x (128/k)` and holds copy
indices *j* x (128/k) onward **of every configuration**, so:

- the *k* chunks partition each configuration's copies 0..127, no index appears twice;
- copy *i* of a configuration in the union has the initial weights and the environment draws copy
  *i* of a single 128-copy run would have had, because the seed index keys both;
- the policy's own sampling stream is keyed by `--run-seed`, which is the chunk index, so no copy
  shares its action noise with a copy of another chunk;
- every chunk carries all 33 configurations, so a partial result is the whole grid at fewer seeds
  rather than part of the grid.

Three tests pin this rather than argument: `tests/agents/test_copy_seed_offset.py` for the seed
arithmetic, `code/test_plan_submission.py` for the planner's index ranges, and
`code/test_chunk_partition_end_to_end.py`, which trains two chunks of a four-configuration toy unit
through the real runner and checks that the aggregator recovers one row per configuration with
every copy counted exactly once.

## The submission plan, and how it was computed

`code/plan_submission.py` minimises the MAKESPAN — when the LAST chunk finishes — over every
combination of per-unit splits, list-scheduling the chunks onto the free cards by earliest finish.
The plan it chose and the alternatives it rejected are in `code/submission_plan.json`.

**Rates come from this run's own probes.** Seven probe jobs measured both arms for 400 real
iterations at the chunk sizes in question, in the same 33-configuration fused shape the science jobs
use: 6538829/6538830/6538831 at 528 / 1,056 / 2,112 / 4,224 copies on an H100 NVL, an A100-SXM4 and
an A100-PCIE, and 6539460/6539461/6539462/6539463 at 132 / 264 / 528 copies on an RTX A4500, a
Quadro RTX 6000 and an RTX A4000. `code/measured_cells.json` holds all 42 cells.

**Only card-to-card ratios are transferred.** The first extension's lesson was that an arm's cost
relative to the shared survey's trainer is not a constant of the arm but a function of the copy
count — 2.4 times at 8,192 copies, 1.7 at 1,024 — so a factor measured at one copy count credits a
slow card with an impossible rate at another. This planner therefore never scales a survey rate by
an arm factor. Where a class was not probed it takes the probed rate of the same arm at the same
copy count on another class and multiplies by the survey's ratio between the two classes. That
ratio holds up at the finer counts: measured here, the RTX A4500 leads the Quadro RTX 6000 by 1.31
to 1.36 times at 264 and 132 copies against the survey's 1.39 at 512 copies, and the RTX A4000 by
1.34 against 1.40.

The chosen plan, 40 chunks on 37 cards, makespan 8.05 hours:

| arm | node | class | chunks | copies per chunk | estimated hours per chunk | rate from |
|---|---|---|---|---|---|---|
| `rnd_next_state` | nekomata01 | `nekomata01` | 4 | 132 | 3.36 | probed on jaguar03, carried by the survey's card ratio at 512 copies |
| `rnd_next_state` | jaguar01 | `jaguar01` | 3 | 132 | 5.50 | probed on jaguar03, carried by the survey's card ratio at 512 copies |
| `rnd_next_state` | jaguar03 | `jaguar03` | 2 | 132 | 5.50 | probed on this card at this copy count |
| `rnd_next_state` | jaguar06 | `jaguar06` | 1 | 132 | 5.52 | probed on jaguar03, carried by the survey's card ratio at 512 copies |
| `rnd_next_state` | cheetah08 | `cheetah08-09` | 4 | 132 | 6.91 | probed on this card at this copy count |
| `rnd_next_state` | cheetah09 | `cheetah08-09` | 4 | 132 | 6.91 | probed on this card at this copy count |
| `rnd_next_state` | lotus | `lotus` | 8 | 132 | 7.44 | probed on this card at this copy count |
| `rnd_next_state` | cheetah03 | `cheetah03` | 2 | 132 | 7.95 | probed on jaguar03, carried by the survey's card ratio at 512 copies |
| `rnd_next_state` | affogato11 | `affogato11` | 4 | 132 | 8.05 | probed on jaguar03, carried by the survey's card ratio at 512 copies |
| `gt_position_velocity_sqrt` | serval03 | `serval03` | 2 | 528 | 3.50 | probed on serval06-09, carried by the survey's card ratio at 512 copies |
| `gt_position_velocity_sqrt` | jaguar03 | `jaguar03` | 6 | 528 | 7.39 | probed on this card at this copy count |

The rejected alternatives, each with the finish the same program computed for it, are in
`infra_history.md` under the launch entry.

**The H100 allowance does not bind.** Every job of this run occupies its whole card with one fused
compiled program, which is the FUSED case of the allowance amended on 2026-08-17: no cap, take as
many free H100 cards as shorten the makespan. In the event the cluster had exactly one free H100 NVL
card when the plan was computed, and the plan took it.

## Walltime

The `gpu` partition's limit is 4 days and no maintenance window is scheduled, so open-partition jobs
ask for the partition maximum, `4-00:00:00`. Jobs on jaguar03 run under our own reservation
`sl5nw_156`, which ends 2026-08-19T23:59:59 Eastern, so they ask for `2-00:00:00` — the largest
whole value that finishes inside the reservation. Every chunk's estimate is under 8.1 hours, so the
walltime is not expected to bind on either.

## Processor and memory asks

Six processors and 24 GB per job, below this project's usual eight and 48 GB. The trainer keeps its
arrays on the card and the host only dispatches — the shared survey measured the iteration time
varying 0.1 to 1.6 per cent between 8, 16 and 32 processors — and this run's own probe processes
peaked at 1.5 to 2.9 GB of host memory. The smaller asks are what let eight jobs sit on lotus (250
GB and 78 processors free) and four on affogato11 (125 GB and 30 processors free), which the larger
ones would not.

## Deviation from the canary rule, and why

`uva-submit-gpu-sweep` Step 6 runs a canary phase and then, once its results are read, the science
submissions. Here the canary, the resume check and the science run are **three phases of one job**
(`code/run_chunk.sh`), so a card is never released between them.

The reason is measured, not theoretical: between the availability read at 19:46 PT and the one at
20:02 PT, every H100 and A100 card on the cluster changed hands, and the computed makespan moved
from 18.39 hours to 11.83 hours and back. A canary that finishes releases its card into that. What
the canary phase exists for is kept: the canary runs first on the real card at the real copy count,
its shard is written under `canary/` for the 20-minute tick to compare against the plan, and a
canary that does not end in a completion record stops the job before any science runs (exit 4). The
resume check follows it — the identical command again, which must be a logged no-op that adds no
record (exit 5 if not) — so the resume is tested per chunk on the card that will run it, and only
then is the queue entry claimed.

## Git state

Code commit at launch: recorded in `manifest.yaml` under `git.commit` and in `command.txt`. The
repository was committed and pushed before the first science submission, per the shared
commit-before-submit rule.

## Where the artifacts are

| what | where |
|---|---|
| per-chunk science records | `data/<chunk id>.jsonl`, one line per 200-iteration window |
| per-chunk canary and resume records | `canary/data/<chunk id>.jsonl` |
| rate probes | `probe/data/*.jsonl`, distilled into `code/measured_cells.json` |
| the plan and its rejected alternatives | `code/submission_plan.json` |
| the availability read the plan was computed from | `code/availability/availability_at_plan.json` |
| per-submission record | `slurm/jobs/<job id>_<node>/` — the script as submitted, the unit, the hardware, the log |
| job ids of this run | `slurm/submitted_jobids.txt` — the only file a cancel may read from |
| run-level aggregate | `summary.json`, written by `code/aggregate.py` |
| what happened | `infra_history.md` |

## Closing summary (written when the queue drained)

The run finished at 2026-08-18 08:00 PT, 11 h 48 m after its first chunk started. All 44 chunks
completed on their own shard and wrote their completion records; `queue/failed/` is empty. 429,704
window records, every one phase-blocked, and both arms' chunks were checked to partition their 128
copies per configuration exactly once, so all 66 configurations are scored over the full 128.

The plan's makespan was 8.05 hours. The extra 3 h 45 m is jaguar03 failing at 23:03 PT with eight
chunks on it — 41 per cent of the run's copies — and the re-placement that followed, which
`infra_history.md` records in full. 248 card-hours over 14 nodes of 12 classes, from an H100 NVL
down to an RTX 2080 Ti.

### What the run found

1. **The configuration a 10M-step sweep selects is not the configuration that wins at 1000M
   steps, and this holds for both arms.** Given its whole grid at the long budget, random network
   distillation wins at learning rate $10^{-4}$ and intrinsic weight $10^{2}$, scoring
   52.500 ± 3.368; the oracle $1/\sqrt{n}$ bonus wins at $10^{-4}$ and $10^{1}$, scoring
   38.199 ± 3.216. The configurations the parent's 10M sweep chose — both at learning rate
   $10^{-3}$ — score 1.795 ± 0.101 and 0.658 ± 0.024 over the same 1000M steps, which is 29 and 58
   times worse than their own arm's best.
2. **Both arms move the same way**: one step down in learning rate, one step up in intrinsic
   weight. A change that is identical across two arms whose bonuses share no machinery is a
   property of the budget, not of either bonus.
3. **The ordering between the arms is unchanged and no longer rests on borrowed configurations.**
   Distillation leads the oracle bonus by about four standard errors when each is given its own
   best. Neither arm is separated on the other metrics: both reach the goal in every one of their
   128 copies and both cover 99.85 per cent of the maze.
4. **Not every configuration rises then falls.** 20 of 66 do; 16 never reach the goal, 14 are
   still rising at 1000M steps, 13 decline throughout and 3 rise then flatten. The label is
   unchanged under all nine threshold variants for 53 of the 66, and every one of the ten
   highest-scoring configurations is unanimous — the instability is confined to configurations
   scoring 10.4 or below.
5. **An aggregate's shape need not be its copies' shape.** Of the 128 copies behind each named
   configuration, 76 and 84 share the aggregate label of the two winners, but only 6 share the
   oracle arm's rise-then-plateau exception, where 90 of its copies rise and fall on their own.

### What would make the conclusions wrong

Finding 1 rests on 128 copies per configuration against the parent's 256 and the first extension's
1,024, so its standard errors are the widest of the three runs. The gap it reports is 15 to 58
times the standard error, so no plausible sampling error closes it, but a reader comparing a
1000M-step cell with a 10M-step one is comparing different copy counts as well as different
budgets. Finding 3's four-standard-error gap is the one number here that a larger sample could
move.
