# Infrastructure history

## 2026-08-06 — canary wave and first full submission

**Canary, 32 slots across 8 node classes, jobs 6534078–6534086.** No failures, no errors in any
worker log, all five arms covered. Throughput per run, measured at real concurrency:

| node | GPU | runs on the node | steps/s per run |
|---|---|---|---|
| cheetah03 | RTX 2080 Ti | 2 | 3,867 |
| lotus | Quadro RTX 6000 | 8 | 3,550 |
| cheetah02 | RTX 4000 Ada | 4 | 3,420 |
| cheetah08 | RTX A4000 | 4 | 3,340 |
| ai09 | GTX 1080 Ti (gnolim) | 3 | 2,765 |
| ai07 | GTX 1080 Ti (gnolim) | 3 | 2,722 |
| adriatic01 | Quadro RTX 4000 | 3 | 2,586 |
| jaguar02 | A16 | 3 | 2,465 |

**The Pascal cards work.** The GTX 1080 Ti nodes on `gnolim` reach 78% of lotus's rate, which is the
verification the whole partition depended on: torch 2.6.0+cu124 ships sm_50 through sm_90, and
`PYTHONNOUSERSITE=1` keeps the ~/.local torch 2.10+cu128 build — which has no Pascal kernels — from
shadowing it. Nothing in `data/` was touched by the canary; every canary record landed in
`canary/data/` as the repeated `--output_dir` is meant to ensure.

GPU memory reserved sits at 3,042–3,246 MB against the 4,900 MB estimate, so the packing bound is
not tight anywhere. W and c are unchanged after the canary.

**First full submission.** 12 worker slots on the open `gpu` partition (cheetah09, adriatic02-04)
plus the reserved node. jaguar03 went last, as the order requires, and started immediately with
`--reservation=sl5nw_156 --qos=csresnolim`: 8 GPUs at 3 runs each = **24 runs claimed at once**.

The four open-partition jobs are PENDING (Priority). That is expected rather than a fault: the `gpu`
per-user GPU cap is 40 and the canary already holds 28, so they wait for the canary's 40-minute jobs
to end. They are not cancelled — a pending job accrues age, and cancelling would give that up for
nothing.

`gnolim` is at 64 of its 80 CPU threads, so it takes no more until the canary releases.

## 2026-08-06, second tick — the canary released, 53 runs live

**The reservation's CPU usage counts into the open partition's cap, and it bit here.** After the
reserved node started, the `gpu` counters read 424 of 400 CPU threads and 40 of 40 GPUs, and the four
open-partition jobs sat on `QOSMaxCpuPerUserLimit`. The submission order was right — open first,
reserved last — but the open jobs had not yet *started*, because the canary already held all 40 GPUs,
so the reserved node's 192 threads took the CPU headroom they were waiting for. The order rule
protects the ceiling only when the open jobs have actually been admitted; with the cap already
saturated by something else, reserved-last still displaces them.

**The canary was cancelled once it had answered.** Its 32 runs had already covered all eight node
classes with no failures, and the second round would have measured nothing new while holding 28 GPUs
and 240 CPU threads. Cancelled ids 6534078–6534086, each checked against this run's own id file
first. All four pending open-partition jobs started within a minute.

**Verified on the reserved node before topping up**: 24 runs sampled, 3,284 to 4,008 MB of GPU memory
each, so three runs share an A4500's 20 GB with room to spare. No record files yet is expected rather
than a fault — the first row is written at update 200, which is 3,276,800 steps.

**Topped up to both caps.** `gpu`: lotus 8, cheetah03 2, adriatic05 1. `gnolim`: ai07 3, ai08 3,
jinx01 2. Both partitions left with about 16 CPU threads of owner headroom.

**53 of 150 runs are now executing; 97 pending.** No failures.

## 2026-08-06, third tick — the campaign is producing data

All eleven jobs running; 55 of 150 runs executing, 95 pending, none failed. The first logged rows
arrived (28 records).

**Three runs per GPU on the reserved node is holding up.** Its 24 packed runs average 3,149 steps/s
each — against 3,388 for one-run-per-GPU on cheetah09 and 3,828 on cheetah03. So the reserved A4500s
are delivering about 9,450 steps/s per GPU where a single run would give roughly 4,600. The
reservation is worth about as much as the rest of the open partition put together.

**Both pools are at the owner's headroom limit** — `gpu` 384 of 400 CPU threads, `gnolim` 64 of 80 —
so this tick submitted nothing. `gnolim` still shows 12 free GPUs, but its CPU cap of 80 allows only
10 slots in total and 8 are taken; the GPUs there cannot be reached without spending the headroom.

**Every invariant still holds at campaign scale**: across every logged row of every run, zero
violations of the arm definitions and zero non-finite gradient norms.

Arm coverage of the runs started so far is 7 / 6 / 4 / 6 / 5 across arms 1 to 5, over seeds 1-9, 12
and 13 — the seed-outermost ordering keeping the arms balanced as intended.
