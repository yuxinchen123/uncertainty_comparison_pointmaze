# Infrastructure history

Times are Pacific.

## 2026-08-17 13:36 PT — the first rate probes: both arms at four chunk sizes on three classes

Three probe jobs — 6538829 (serval06, H100 NVL), 6538830 (cheetah04, A100-SXM4-80GB) and 6538831
(cheetah01, A100-PCIE-40GB) — measured both arms for 400 real iterations at 528, 1,056, 2,112 and
4,224 copies, in the same 33-configuration fused shape the science jobs use. All three completed;
24 cells. The session that submitted them stopped before the plan was computed.

## 2026-08-17 19:46 PT — every fast card on the cluster was busy

The availability read at 19:46 PT showed **zero free H100 and zero free A100 cards**: all nine H100
NVL cards held by one user's 15-hour array jobs, all eight A100 cards by two other users. Slurm's
own projection for a one-card job, asked with `sbatch --test-only` per node, put the H100 nodes
about 20 to 21 hours out and the A100 nodes about 9 hours out.

The read at 20:02 PT — sixteen minutes later — showed one free H100 NVL, three free A100-SXM4 and
one free A100-PCIE, and the computed makespan fell from 18.39 hours to 11.83 hours. By 20:15 PT
those A100 cards were gone again. This churn is the reason for the canary decision below.

## 2026-08-17 20:00 PT — a second round of probes at finer chunk sizes, and what it was worth

The first probes stopped at 528 copies, which is a unit cut eight ways. With no fast card free, the
best eight-way plan finished in 18.39 hours, set by chunks of 528 distillation copies on RTX A4500
and A40 cards. Four more probe jobs measured 132, 264 and 528 copies on the three classes that were
free in quantity and not contested — 6539460 and 6539461 on jaguar03 (RTX A4500, under our own
reservation, so nobody else could take the cards), 6539462 on lotus (Quadro RTX 6000) and 6539463 on
cheetah08 (RTX A4000). They took 3 to 8 minutes each and added 18 cells.

They changed the plan outright. Measured on an RTX A4500, one distillation chunk of 1,000,038,400
steps per copy takes 15.6 hours at 528 copies, 9.1 at 264 and **5.5 at 132**; the per-copy rate
rises from 17,846 to 50,882 environment steps a second. The makespan of the best plan fell from
15.61 hours (eight-way splits, the finest the earlier probes could price) to **8.05 hours**
(distillation cut 32 ways, the oracle arm 8 ways). Twenty minutes of probing bought seven and a
half hours.

One number worth keeping for the next planner: the survey's card-to-card ratios, which are what this
planner transfers, hold at the finer copy counts. The RTX A4500 leads the Quadro RTX 6000 by 1.31
and 1.36 times as measured at 264 and 132 copies against the survey's 1.39 at 512, and the RTX A4000
by 1.34 against 1.40 — within 6 per cent either way.

## 2026-08-17 20:25 PT — the plan, and the alternatives it rejected

`code/plan_submission.py` costed 36 combinations of per-unit splits against the 49 free cards it was
allowed to hold (39 on `gpu` and 10 on `gnolim`, the smaller of each partition's card cap and what
its processor cap allows at six processors a job). The chosen plan is 40 chunks on 37 cards
finishing in 8.05 hours. The alternatives, each with the finish the same program computed:

| plan | chunks | cards | computed finish | why not |
|---|---|---|---|---|
| distillation 32 ways, oracle 8 ways | 40 | 37 | **8.05 h** | chosen |
| both arms 32 ways | 64 | 49 | 8.99 h | the extra 24 chunks land on cards slower than the ones already carrying the oracle arm |
| both arms 16 ways | 32 | 30 | 9.10 h | halves the number of jobs and costs an hour; below the 15-minute proportionality margin it would have won |
| distillation 16, oracle 32 | 48 | 39 | 9.10 h | same finish as 16/16 with 16 more chunks |
| distillation 32, oracle 16 | 48 | 39 | 9.81 h | the oracle chunks displace distillation chunks from the faster cards |
| both arms 8 ways | 16 | 15 | 15.61 h | the finest split the first probe round could price at all |
| both arms 4 ways | 8 | 7 | 19.12 h | |
| both arms uncut | 2 | 2 | 31.86 h | |
| wait for the four A100-SXM4 and four A100-PCIE cards | 32 | 8 | about 19.0 h | Slurm projects them free in 8.6 to 8.7 hours; four distillation chunks per card at 2.6 to 2.9 hours each puts the finish at about 19 hours, more than twice the plan that starts now |
| wait for the eight H100 NVL cards | 32 | 8 | about 25.9 h | projected free in 19.8 to 20.2 hours; four chunks of 1.5 hours each after that |

The two waiting rows were computed by adding the busy cards to the planner's card list with the
seconds until Slurm's own `--test-only` projection frees them
(`code/availability/busy_fast_cards.json`). Adding them changed nothing: the arithmetic rejects
waiting on its own, and `code/submission_plan_with_waiting.json` is the same 8.05-hour plan.

**Splitting further was not available and would not have helped.** 32 chunks of 132 copies is four
copies per configuration, and 64 would be two — but the finer count is unpriced (no probe covers it)
and the plan already spends 37 of the 39 cards the processor cap allows, so the extra chunks would
queue behind the ones running rather than run beside them.

**Unequal splits were considered and not implemented.** Giving a slow card a smaller slice than a
fast one would balance the finish times better than one chunk size per arm does. It was bounded
instead of built: the makespan is set by cards at 7.4 to 8.1 hours against the fastest card's 3.4,
so a perfect balance could save at most about half an hour on this plan, against a new code path in
the queue builder, the aggregator's partition check and its per-configuration pooling. The
proportionality clause of `rnd-jax-submission` section 2 says not to.

## 2026-08-17 20:25 PT — canary, resume check and science run were made one job

`uva-submit-gpu-sweep` Step 6 runs a canary phase, reads it, and only then submits the science jobs.
A canary that finishes releases its card, and the 19:46-to-20:15 PT churn above shows what that
costs on this cluster today. So `code/run_chunk.sh` runs three phases in one job on one card: the
canary at 400 iterations into `canary/`, then the identical command again as the resume check, then
the science run. A canary with no completion record exits 4 and the science run never starts; a
resume check that adds a record or fails exits 5 and the science run never starts. The queue entry
is claimed only after both pass.

What is given up is the chance to read every canary before any science job starts. What is kept is
everything the canary phase is for: the card proves it holds the program at this copy count, the
node-local compiled-program cache is written by the process that will then use it, the resume is
tested per chunk on its own card, and the canary's own rate is on disk for the 20-minute tick to
compare against the plan.

## 2026-08-17 20:35 PT — one chunk was moved off a card another user took first

Thirty-six of the forty jobs started within seconds. The four that did not are the second chunk on
a card one of this run's own chunks already holds — two on nekomata01 and, on serval03, a chunk
waiting behind our own — except that serval03's single H100 NVL card had been taken by another
user's two-hour array job between the availability read the plan was computed from (20:02 PT) and
the submission (20:12 PT). With one array task running and one queued behind it, that card was
about four hours from us, which would have put the oracle arm's last chunk at about 07:20 PT
against the rest of the run's 04:15.

A card of `cheetah04` (A100-SXM4-80GB) had come free meanwhile. Job 6539520 was cancelled — its id
read from this run's own `slurm/submitted_jobids.txt` and checked against it first, never a blanket
cancel — and its chunk resubmitted as job 6539578 on that card, where this run's own probe measures
the oracle arm at 60,447 environment steps a second per copy and the chunk at about 4.6 hours. The
other serval03 chunk stays where it is; it now has the card to itself once the other user's array
finishes, and its projected finish is inside the rest of the run's.

The per-user allowance was full at 40 of 40 graphics cards while both jobs were queued, which is why
the chunk was moved rather than added: cancelling first freed the slot the new job needed.

That chunk's queue entry was repriced to the card it actually holds — 4.60 hours from this run's
own probe of the oracle arm on an A100-SXM4 at 528 copies, with the serval03 plan kept beside it
under `reassigned_from` — so the 20-minute tick stops reporting it as a canary off plan every tick
for a card it never went to.

## 2026-08-17 21:50 PT — the tick moved to hourly, and a question was queued for the document pass

The run had been stable for over an hour — no job changed state, no id was added or removed, every
running chunk's shard kept growing, nothing was waiting to be submitted — so the primary tick moved
from twenty minutes to hourly under the cadence rule of the `sweep-monitoring` skill. It drops back
to twenty minutes on its own when a job fails, is preempted or requeued, when a shard goes flat
while its job is RUNNING, when anything is submitted, or when fewer than about six chunks remain;
the tail is where a tick has work to do. Each tick states which cadence it is on.

A question was also queued for the document pass and is specified in
`analysis/question_curve_shape.md`: aggregated over a configuration's copies, does every
configuration of this sweep rise and then fall? Its tooling — `analysis/code/curve_shape.py` and
`analysis/code/shape_report.py` — is written and was validated against the completed 10M-step
batch, which is where two of its defects were found. It reads every configuration through this
run's own `code/aggregate.curve_of`, so a shape and a table score can never disagree.

## 2026-08-17 23:03 PT — jaguar03 failed with eight chunks on it, and four of them had to be re-cut

`jaguar03` stopped responding at about 23:03 PT. Slurm marked it `down*` with reason "Not
responding", failed the eight of this run's chunks that were on it — job ids 6539522 to 6539529,
each `NODE_FAIL` after about 2 h 50 m — and requeued them. Because every one was pinned
`--nodelist=jaguar03`, they then sat `(ReqNodeNotAvail, UnavailableNodes:jaguar03)` and could never
start while the node stayed down. The node was down for three days in late July, so waiting was not
a plan.

Those eight chunks held 3,432 of the run's 8,448 copies — 41 per cent of it — and about 2 h 50 m of
work each, which is lost outright: this platform saves no model state, so a chunk resumes only by
being re-run whole.

**No shard needed repairing.** A re-run appends its windows from the first one again, and
`aggregate.live_lines` keeps the later line wherever a window appears twice; that is what the
function is for and it is tested. `curves_all` inherits it. Nothing was edited.

**Four chunks moved as they were.** The oracle arm's chunks 7 and 8 and the distillation arm's
chunks 3 and 4 went to `cheetah04` (A100-SXM4-80GB), where this run's own probe prices them at 7.39
and 2.58 hours. Jobs 6539899, 6539900, 6539903, 6539904.

**Four had to be cut finer, and this is why.** The oracle arm's chunks 3 to 6 hold 528 copies each,
and a 528-copy chunk of that arm needs an H100 or an A100 to finish in the time the rest of the run
had left. Every such card on the cluster was allocated or projected free more than a day out —
`sbatch --test-only` put serval06 and serval07 at 2026-08-19 04:26 ET and serval03 at 2026-08-23,
and a shorter walltime did not change those projections, so backfill was not the obstacle. On the
fastest card actually free, an RTX 2080 Ti, one of those chunks would take 18 hours. Cut into eight
chunks of 264 copies instead, the same 64 copies per configuration take 7.38 hours each and run
side by side: `code/recut_stranded_chunks.py` wrote the eight entries, jobs 6539930 to 6539937 on
ai01 to ai04. The re-cut saves about ten hours against re-running the four unchanged on the same
cards.

The four superseded shards were MOVED, not edited or deleted, to `data_superseded/`, and their
queue entries to `queue/superseded/`. They had to leave `data/` because the aggregator checks that
a configuration's chunks partition its copies exactly once and copy indices 32 to 95 would
otherwise be covered twice. The oracle arm is now twelve chunks: four of 16 copies per
configuration (indices 0–15, 16–31, 96–111, 112–127) and eight of 8 (32–95), together covering
0–127 exactly once. The re-cut chunks carry run seeds 8 to 15, outside the 0 to 7 the original
cutting used, so no chunk shares another's action noise.

Two smaller repairs came out of it. The first submission of the eight re-cut chunks put four on one
node, whose 62.5 GB of memory cannot hold four 24 GB jobs; they were cancelled and respread two per
node over ai01 to ai04, where all eight started at once. And the status table keyed jobs by their
NAME, which `submit_one.sh` derives from the unit id by a pattern that reads `recut-chunk-1-of-8`
as `chunk-1-of-8` — so a re-cut chunk and an original collided and a pending resubmission was
reported as its own cancelled predecessor. Submissions now append the id and the unit to
`slurm/unit_jobs.tsv` at submit time, and the table reads that.

## 2026-08-18 01:24 PT — the flat-shard check reported a restart as a stall

The tick reported `unit-1-chunk-3-of-32` on cheetah04 as a shard that had stopped growing while its
job ran. It had not: the job was writing window 1,682 of 9,766 and its log was current. The counter
reads the LAST record in the shard, and that chunk is one of the four re-placed after jaguar03
failed, so its shard holds the jaguar03 attempt's windows followed by the re-run's, which start
again from the first one. The last record's iteration therefore went backwards at the moment the
re-run began, and a check written as "did not increase" read that as a stall.

The check now reports only an EQUAL count as flat; a smaller one is a restart, which is what this
platform does when a chunk is re-run, since it saves no model state.

The same reading gives the re-placed chunk's real rate: 1.44 seconds per window, so 3.90 hours for
its 9,766, against the 2.58 the plan carried from an RTX A4500 measurement through the survey's
card ratio. The transfer was about half an hour per hour optimistic for this class at this copy
count. It changes nothing — the chunk still finishes before the re-cut oracle chunks do — and it is
the kind of error the tick exists to surface.

## 2026-08-18 04:50 PT — the tail: one chunk moved again, and what the re-cut actually cost

At 04:44 PT, 42 of the 44 chunks were complete. Two were left, and one of them —
`unit-1-chunk-4-of-32`, the second distillation chunk re-placed onto cheetah04 after jaguar03 failed
— had never started: all four of that node's cards were busy, two with this run's own oracle chunks,
so it sat `(Priority)` behind them and would not have begun until about 06:40 PT, finishing about
10:30. Two RTX 5080 cards on nekomata01 were idle. Job 6539904 was cancelled, its id read from this
run's own id file first, and the chunk resubmitted there as job 6540084, where the same arm at the
same copy count measured 3.14 hours earlier in this run. That moves the run's finish from about
10:30 PT to about 08:00.

**The re-cut was worth more than the arithmetic promised.** Its chunks were planned at 7.38 hours
each, from an RTX A4500 measurement at 264 copies carried onto an RTX 2080 Ti by the survey's card
ratio at 512 copies. They came in at 5.18 to 5.25 hours — about 30 per cent faster than the
transfer predicted, in the same direction the first canaries erred. Against the 18 hours one
uncut 528-copy chunk would have taken on the same card, the re-cut saved about 13 hours rather
than the 10 it was chosen for.
