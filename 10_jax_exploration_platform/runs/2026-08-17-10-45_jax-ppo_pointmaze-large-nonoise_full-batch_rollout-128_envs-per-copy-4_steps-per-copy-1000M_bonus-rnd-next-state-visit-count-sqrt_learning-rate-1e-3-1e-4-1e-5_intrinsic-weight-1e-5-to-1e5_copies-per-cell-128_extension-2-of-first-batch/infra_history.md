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
