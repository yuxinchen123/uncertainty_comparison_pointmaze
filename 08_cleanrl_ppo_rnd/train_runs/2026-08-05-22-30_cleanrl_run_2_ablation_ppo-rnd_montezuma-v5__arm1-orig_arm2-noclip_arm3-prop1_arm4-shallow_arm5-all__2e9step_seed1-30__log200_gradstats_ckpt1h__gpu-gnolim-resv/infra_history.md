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

## 2026-08-06, fourth tick — the collaborator packet was not safe, and is now fixed

An independent check of the packet against the collaborator-handbook skill found it **not safe to
hand over**. Three blocking defects, all fixed this tick:

1. **The launcher died on its first use.** `live=$( { squeue … | grep -c … ; } || echo 0)` — `grep -c`
   prints `0` *and* exits 1 when it matches nothing, so `|| echo 0` appended a second `0`. `live`
   became the two-line string `0\n0` and the next line's arithmetic failed. A collaborator with no
   jobs yet — which is everyone, the first time — could not submit at all. The guard also counted
   *jobs* by name rather than *run slots*, wrong by up to 24x on a packed node, and job names are not
   a safe key anyway. It now counts slots by exact job id from the collaborator's own id file.
2. **`DRY=1` printed no submission plan**, because the launcher had no submit calls: it told the
   collaborator to hand-edit a file the owner owns. It now takes `NODES="node:G …"` like the owner's
   launcher, and resolves the node to its submission script itself.
3. **The README's very first command pointed at a directory that does not exist** —
   `queue/<sweep id>/pending` instead of `queue/pending` — and both the README and the launcher
   depended on a `SWEEP_COMPLETE` sentinel **that nothing ever wrote**, so the collaborator's "am I
   done" signal could never appear. `slurm/mark_complete_if_drained.sh` now writes it, and the
   owner's tick runs it.

Also fixed: the launcher sized cpus and memory as `GPUs x 8` with no runs-per-GPU factor, so a packed
node would have been asked for a third of what it uses — cpu oversubscription and a likely memory
kill; it now reads the factor from the submission script. Nodes inside the owner's reservation are
refused up front rather than pending forever with no error. The free-GPU listing now carries the
partition column, since a node belongs to one partition and the wrong `-p` means the job never runs.
A `monitor_collaborator.sh` was added. The owner's own launcher gained a uid guard: it was
group-writable, so a collaborator running it would have submitted under their uid while appending to
the **owner's** id file.

**The collaborator got there first and it worked out.** yuxinchen submitted 21 worker jobs at 01:50
and 01:52, and the queue went from 55 to **106 of 150 runs executing**. 104 of the 106 running
markers have a usage sample from the last five minutes, so those workers are genuinely training. All
their jobs are one run per GPU and none is on the reserved node, so neither the packing nor the
reservation defect could bite. Their nodes run at 2,566 to 2,823 steps/s.

Across every logged row of all 53 records with data: **zero invariant violations, zero non-finite
gradient norms**. Arm coverage 13 / 9 / 10 / 10 / 11 over 16 seeds.
## 2026-08-06 02:08  (jobs: 21 running (yuxinchen 21), 0 pending | runs: 106 running, 44 pending, 0 done, 0 failed)

| node | gpu type | gpus ours/node | gpu memory used/total (every gpu) | gpu usage (every gpu) | cpu busy/asked (threads) | sys memory used/asked/node (GB) | running runs |
|---|---|---|---|---|---|---|---|
| <span style="color:gray">cheetah01 (yuxinchen)</span> | <span style="color:gray">a100</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.1/40.0 GB</span> | <span style="color:gray">GPU0*: 1%</span> | <span style="color:gray">6.6/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (079)</span> |
| <span style="color:gray">jaguar06 (yuxinchen)</span> | <span style="color:gray">a40</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.3/45.0 GB</span> | <span style="color:gray">GPU0*: 6%</span> | <span style="color:gray">5.2/8</span> | <span style="color:gray">1.5/5.9/123.0</span> | <span style="color:gray">1 (048)</span> |
| <span style="color:gray">cheetah02 (yuxinchen)</span> | <span style="color:gray">rtx_4000_ada</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.3/20.0 GB<br>GPU1*: 3.3/20.0 GB<br>GPU2*: 3.3/20.0 GB<br>GPU3*: 3.3/20.0 GB</span> | <span style="color:gray">GPU0*: 4%<br>GPU1*: 4%<br>GPU2*: 5%<br>GPU3*: 4%</span> | <span style="color:gray">18.7/32</span> | <span style="color:gray">6.1/23.4/1000.0</span> | <span style="color:gray">4 (066–067, 085–086)</span> |
| jaguar03 | rtx_a4500 | 8/8 | GPU0*: 10.7/20.0 GB<br>GPU1*: 10.0/20.0 GB<br>GPU2*: 10.1/20.0 GB<br>GPU3*: 9.7/20.0 GB<br>GPU4*: 10.0/20.0 GB<br>GPU5*: 9.9/20.0 GB<br>GPU6*: 10.0/20.0 GB<br>GPU7*: 10.0/20.0 GB | GPU0*: 23%<br>GPU1*: 100%<br>GPU2*: 20%<br>GPU3*: 100%<br>GPU4*: 100%<br>GPU5*: 95%<br>GPU6*: 24%<br>GPU7*: 98% | 91.4/192 | 36.5/175.8/1000.0 | 24 (001, 005, 007, 010–022, 024–026, 028, 030–031, 038, 044) |
| <span style="color:gray">cheetah08 (yuxinchen)</span> | <span style="color:gray">rtx_a4000</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 5%<br>GPU2*: 5%<br>GPU3*: 5%</span> | <span style="color:gray">18.7/32</span> | <span style="color:gray">6.0/23.4/500.0</span> | <span style="color:gray">4 (058, 061, 073, 078)</span> |
| cheetah09 | rtx_a4000 | 4/4 | GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB | GPU0*: 7%<br>GPU1*: 7%<br>GPU2*: 6%<br>GPU3*: 6% | 18.7/32 | 6.0/29.3/500.0 | 4 (003, 023, 034–035) |
| <span style="color:gray">jaguar02 (yuxinchen)</span> | <span style="color:gray">a16</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.2/15.0 GB<br>GPU1*: 3.1/15.0 GB<br>GPU2*: 3.2/15.0 GB</span> | <span style="color:gray">GPU0*: 95%<br>GPU1*: 96%<br>GPU2*: 96%</span> | <span style="color:gray">8.7/24</span> | <span style="color:gray">4.5/17.6/1000.0</span> | <span style="color:gray">3 (084, 101, 112)</span> |
| lotus | rtx_6000 | 8/8 | GPU0*: 3.4/24.0 GB<br>GPU1*: 3.4/24.0 GB<br>GPU2*: 3.4/24.0 GB<br>GPU3*: 3.4/24.0 GB<br>GPU4*: 3.4/24.0 GB<br>GPU5*: 3.4/24.0 GB<br>GPU6*: 3.4/24.0 GB<br>GPU7*: 3.4/24.0 GB | GPU0*: 7%<br>GPU1*: 92%<br>GPU2*: 7%<br>GPU3*: 8%<br>GPU4*: 6%<br>GPU5*: 6%<br>GPU6*: 8%<br>GPU7*: 6% | 36.6/64 | 11.7/58.6/250.0 | 8 (002, 008, 032, 039, 054, 072, 074–075) |
| <span style="color:gray">ai01 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 28%<br>GPU2*: 59%</span> | <span style="color:gray">14.1/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (040, 047, 069)</span> |
| <span style="color:gray">ai02 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 31%<br>GPU1*: 5%<br>GPU2*: 6%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (037, 045, 094)</span> |
| <span style="color:gray">ai03 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 4.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 42%<br>GPU1*: 87%<br>GPU2*: 5%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (059, 076, 081)</span> |
| <span style="color:gray">ai04 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 54%<br>GPU1*: 5%<br>GPU2*: 6%</span> | <span style="color:gray">14.1/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (083, 088, 097)</span> |
| <span style="color:gray">ai06 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 15%</span> | <span style="color:gray">5.0/8</span> | <span style="color:gray">2.0/5.9/62.5</span> | <span style="color:gray">1 (092)</span> |
| cheetah03 | rtx_2080_ti | 2/2 | GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB | GPU0*: 46%<br>GPU1*: 22% | 9.1/16 | 2.9/14.6/1000.0 | 2 (055, 064) |
| <span style="color:gray">lynx10 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 5%<br>GPU2*: 10%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">4.5/17.6/62.5</span> | <span style="color:gray">3 (113, 122, 124)</span> |
| <span style="color:gray">adriatic01 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.1/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.1/8.0 GB</span> | <span style="color:gray">GPU0*: 97%<br>GPU1*: 94%<br>GPU2*: 95%</span> | <span style="color:gray">13.2/24</span> | <span style="color:gray">4.4/17.6/1000.0</span> | <span style="color:gray">3 (090, 114, 123)</span> |
| adriatic02 | quadro_rtx_4000 | 3/3 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB | GPU0*: 14%<br>GPU1*: 96%<br>GPU2*: 47% | 12.5/24 | 4.4/17.6/1000.0 | 3 (027, 033, 050) |
| adriatic03 | quadro_rtx_4000 | 3/3 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.2/8.0 GB | GPU0*: 10%<br>GPU1*: 8%<br>GPU2*: 97% | 12.5/24 | 4.4/17.6/1000.0 | 3 (051, 060, 062) |
| <span style="color:gray">adriatic04 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">2/2</span> | <span style="color:gray">GPU0*: 4.4/8.0 GB<br>GPU1*: 3.2/8.0 GB</span> | <span style="color:gray">GPU0*: 9%<br>GPU1*: 96%</span> | <span style="color:gray">8.6/16</span> | <span style="color:gray">3.0/11.7/1000.0</span> | <span style="color:gray">2 (042, 046)</span> |
| <span style="color:gray">adriatic05 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 97%</span> | <span style="color:gray">4.7/8</span> | <span style="color:gray">1.5/5.9/1000.0</span> | <span style="color:gray">1 (071)</span> |
| <span style="color:gray">adriatic06 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 94%<br>GPU2*: 13%</span> | <span style="color:gray">12.9/24</span> | <span style="color:gray">5.0/17.6/1000.0</span> | <span style="color:gray">3 (103, 108, 116)</span> |
| <span style="color:gray">jaguar05 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.4/8.0 GB</span> | <span style="color:gray">GPU0*: 95%</span> | <span style="color:gray">4.6/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (089)</span> |
| <span style="color:gray">lynx01 (yuxinchen)</span> | <span style="color:gray">titan_xp</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/12.0 GB</span> | <span style="color:gray">GPU0*: 6%</span> | <span style="color:gray">4.9/8</span> | <span style="color:gray">1.9/5.9/62.5</span> | <span style="color:gray">1 (077)</span> |
| ai07 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 7%<br>GPU1*: 6%<br>GPU2*: 7% | 13.6/24 | 4.0/17.6/125.0 | 3 (009, 049, 063) |
| ai08 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 8%<br>GPU1*: 6%<br>GPU2*: 97% | 13.5/24 | 4.0/17.6/125.0 | 3 (053, 065, 070) |
| <span style="color:gray">ai09 (yuxinchen)</span> | <span style="color:gray">gtx_1080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 6%<br>GPU2*: 6%</span> | <span style="color:gray">13.7/24</span> | <span style="color:gray">4.0/17.6/109.4</span> | <span style="color:gray">3 (091, 096, 119)</span> |
| <span style="color:gray">ai05 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 11%<br>GPU1*: 98%<br>GPU2*: 10%</span> | <span style="color:gray">11.9/24</span> | <span style="color:gray">4.3/17.6/125.0</span> | <span style="color:gray">3 (041, 105, 107)</span> |
| <span style="color:gray">ai10 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 10%<br>GPU1*: 97%<br>GPU2*: 8%</span> | <span style="color:gray">12.1/24</span> | <span style="color:gray">4.6/17.6/125.0</span> | <span style="color:gray">3 (029, 036, 115)</span> |
| jinx01 | gtx_1080 | 2/2 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 11%<br>GPU1*: 10% | 7.5/16 | 2.7/14.6/214.8 | 2 (004, 068) |
| <span style="color:gray">0 (yuxinchen)</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded/2</span> | <span style="color:gray">not recorded/not recorded/not recorded</span> | <span style="color:gray">not recorded</span> |
| <span style="color:gray">1 (yuxinchen)</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded/48</span> | <span style="color:gray">not recorded/not recorded/not recorded</span> | <span style="color:gray">not recorded</span> |
| <span style="color:gray">2 (yuxinchen)</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded/16</span> | <span style="color:gray">not recorded/not recorded/not recorded</span> | <span style="color:gray">not recorded</span> |
| <span style="color:gray">3 (yuxinchen)</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded/264</span> | <span style="color:gray">not recorded/not recorded/not recorded</span> | <span style="color:gray">not recorded</span> |
| <span style="color:gray">4 (yuxinchen)</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded/64</span> | <span style="color:gray">not recorded/not recorded/not recorded</span> | <span style="color:gray">not recorded</span> |
| **TOTAL (owner)** | — | **36/36** | **173.4/567.9 GB** | **mean 34% (all listed GPUs)** | **215.2/416** | **76.8/363.3/5214.8** | **52** |
| <span style="color:gray">**TOTAL (collaborators)**</span> | <span style="color:gray">—</span> | <span style="color:gray">**49/49**</span> | <span style="color:gray">**166.0/622.9 GB**</span> | <span style="color:gray">**mean 34% (all listed GPUs)**</span> | <span style="color:gray">**220.3/786**</span> | <span style="color:gray">**77.0/287.1/7919.9**</span> | <span style="color:gray">**49**</span> |
| **TOTAL** | — | **85/85** | **339.4/1190.8 GB** | **mean 34% (all listed GPUs)** | **435.5/1202** | **153.8/650.4/13134.8** | **101** |

## 2026-08-06 02:09  (jobs: 32 running (owner 11, yuxinchen 21), 0 pending | runs: 104 running, 46 pending, 0 done, 0 failed)

| node | gpu type | gpus ours/node | gpu memory used/total (every gpu) | gpu usage (every gpu) | cpu busy/asked (threads) | sys memory used/asked/node (GB) | running runs |
|---|---|---|---|---|---|---|---|
| <span style="color:gray">cheetah01 (yuxinchen)</span> | <span style="color:gray">a100</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.1/40.0 GB</span> | <span style="color:gray">GPU0*: 1%</span> | <span style="color:gray">6.6/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (079)</span> |
| <span style="color:gray">jaguar06 (yuxinchen)</span> | <span style="color:gray">a40</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.3/45.0 GB</span> | <span style="color:gray">GPU0*: 6%</span> | <span style="color:gray">5.2/8</span> | <span style="color:gray">1.5/5.9/123.0</span> | <span style="color:gray">1 (048)</span> |
| <span style="color:gray">cheetah02 (yuxinchen)</span> | <span style="color:gray">rtx_4000_ada</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.3/20.0 GB<br>GPU1*: 3.3/20.0 GB<br>GPU2*: 3.3/20.0 GB<br>GPU3*: 3.3/20.0 GB</span> | <span style="color:gray">GPU0*: 4%<br>GPU1*: 4%<br>GPU2*: 5%<br>GPU3*: 4%</span> | <span style="color:gray">18.7/32</span> | <span style="color:gray">6.1/23.4/1000.0</span> | <span style="color:gray">4 (066–067, 085–086)</span> |
| jaguar03 | rtx_a4500 | 8/8 | GPU0*: 10.7/20.0 GB<br>GPU1*: 10.0/20.0 GB<br>GPU2*: 10.1/20.0 GB<br>GPU3*: 9.7/20.0 GB<br>GPU4*: 10.0/20.0 GB<br>GPU5*: 9.9/20.0 GB<br>GPU6*: 10.0/20.0 GB<br>GPU7*: 10.0/20.0 GB | GPU0*: 23%<br>GPU1*: 100%<br>GPU2*: 20%<br>GPU3*: 100%<br>GPU4*: 100%<br>GPU5*: 95%<br>GPU6*: 24%<br>GPU7*: 98% | 91.4/192 | 36.5/175.8/1000.0 | 24 (001, 005, 007, 010–022, 024–026, 028, 030–031, 038, 044) |
| <span style="color:gray">cheetah08 (yuxinchen)</span> | <span style="color:gray">rtx_a4000</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 5%<br>GPU2*: 5%<br>GPU3*: 5%</span> | <span style="color:gray">18.7/32</span> | <span style="color:gray">6.0/23.4/500.0</span> | <span style="color:gray">4 (058, 061, 073, 078)</span> |
| cheetah09 | rtx_a4000 | 4/4 | GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB | GPU0*: 7%<br>GPU1*: 7%<br>GPU2*: 6%<br>GPU3*: 6% | 18.7/32 | 6.0/29.3/500.0 | 4 (003, 023, 034–035) |
| <span style="color:gray">jaguar02 (yuxinchen)</span> | <span style="color:gray">a16</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.2/15.0 GB<br>GPU1*: 3.1/15.0 GB<br>GPU2*: 3.2/15.0 GB</span> | <span style="color:gray">GPU0*: 95%<br>GPU1*: 96%<br>GPU2*: 96%</span> | <span style="color:gray">8.7/24</span> | <span style="color:gray">4.5/17.6/1000.0</span> | <span style="color:gray">3 (084, 101, 112)</span> |
| lotus | rtx_6000 | 8/8 | GPU0*: 3.4/24.0 GB<br>GPU1*: 3.4/24.0 GB<br>GPU2*: 3.4/24.0 GB<br>GPU3*: 3.4/24.0 GB<br>GPU4*: 3.4/24.0 GB<br>GPU5*: 3.4/24.0 GB<br>GPU6*: 3.4/24.0 GB<br>GPU7*: 3.4/24.0 GB | GPU0*: 7%<br>GPU1*: 92%<br>GPU2*: 7%<br>GPU3*: 8%<br>GPU4*: 6%<br>GPU5*: 6%<br>GPU6*: 8%<br>GPU7*: 6% | 36.6/64 | 11.7/58.6/250.0 | 8 (002, 008, 032, 039, 054, 072, 074–075) |
| <span style="color:gray">ai01 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 28%<br>GPU2*: 59%</span> | <span style="color:gray">14.1/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (040, 047, 069)</span> |
| <span style="color:gray">ai02 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 31%<br>GPU1*: 5%<br>GPU2*: 6%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (037, 045, 094)</span> |
| <span style="color:gray">ai03 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 4.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 42%<br>GPU1*: 87%<br>GPU2*: 5%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (059, 076, 081)</span> |
| <span style="color:gray">ai04 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 54%<br>GPU1*: 5%<br>GPU2*: 6%</span> | <span style="color:gray">14.1/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (083, 088, 097)</span> |
| <span style="color:gray">ai06 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 15%</span> | <span style="color:gray">5.0/8</span> | <span style="color:gray">2.0/5.9/62.5</span> | <span style="color:gray">1 (092)</span> |
| cheetah03 | rtx_2080_ti | 2/2 | GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB | GPU0*: 46%<br>GPU1*: 22% | 9.1/16 | 2.9/14.6/1000.0 | 2 (055, 064) |
| <span style="color:gray">lynx10 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 5%<br>GPU2*: 10%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">4.5/17.6/62.5</span> | <span style="color:gray">3 (113, 122, 124)</span> |
| <span style="color:gray">adriatic01 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.1/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.1/8.0 GB</span> | <span style="color:gray">GPU0*: 97%<br>GPU1*: 94%<br>GPU2*: 95%</span> | <span style="color:gray">13.2/24</span> | <span style="color:gray">4.4/17.6/1000.0</span> | <span style="color:gray">3 (090, 114, 123)</span> |
| adriatic02 | quadro_rtx_4000 | 3/3 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB | GPU0*: 14%<br>GPU1*: 96%<br>GPU2*: 47% | 12.5/24 | 4.4/17.6/1000.0 | 3 (027, 033, 050) |
| adriatic03 | quadro_rtx_4000 | 3/3 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.2/8.0 GB | GPU0*: 10%<br>GPU1*: 8%<br>GPU2*: 97% | 12.5/24 | 4.4/17.6/1000.0 | 3 (051, 060, 062) |
| adriatic04 (owner, yuxinchen) | quadro_rtx_4000 | 2/2 | GPU0*: 4.4/8.0 GB<br>GPU1*: 3.2/8.0 GB | GPU0*: 9%<br>GPU1*: 96% | 8.6/16 | 3.0/11.7/1000.0 | 2 (042, 046) |
| adriatic05 (owner, yuxinchen) | quadro_rtx_4000 | 1/1 | GPU0*: 3.3/8.0 GB | GPU0*: 97% | 4.7/8 | 1.5/5.9/1000.0 | 1 (071) |
| <span style="color:gray">adriatic06 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 94%<br>GPU2*: 13%</span> | <span style="color:gray">12.9/24</span> | <span style="color:gray">5.0/17.6/1000.0</span> | <span style="color:gray">3 (103, 108, 116)</span> |
| <span style="color:gray">jaguar05 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.4/8.0 GB</span> | <span style="color:gray">GPU0*: 95%</span> | <span style="color:gray">4.6/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (089)</span> |
| <span style="color:gray">lynx01 (yuxinchen)</span> | <span style="color:gray">titan_xp</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/12.0 GB</span> | <span style="color:gray">GPU0*: 6%</span> | <span style="color:gray">4.9/8</span> | <span style="color:gray">1.9/5.9/62.5</span> | <span style="color:gray">1 (077)</span> |
| ai07 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 7%<br>GPU1*: 6%<br>GPU2*: 7% | 13.6/24 | 4.0/17.6/125.0 | 3 (009, 049, 063) |
| ai08 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 8%<br>GPU1*: 6%<br>GPU2*: 97% | 13.5/24 | 4.0/17.6/125.0 | 3 (053, 065, 070) |
| <span style="color:gray">ai09 (yuxinchen)</span> | <span style="color:gray">gtx_1080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 6%<br>GPU2*: 6%</span> | <span style="color:gray">13.7/24</span> | <span style="color:gray">4.0/17.6/109.4</span> | <span style="color:gray">3 (091, 096, 119)</span> |
| <span style="color:gray">ai05 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 11%<br>GPU1*: 98%<br>GPU2*: 10%</span> | <span style="color:gray">11.9/24</span> | <span style="color:gray">4.3/17.6/125.0</span> | <span style="color:gray">3 (041, 105, 107)</span> |
| <span style="color:gray">ai10 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 10%<br>GPU1*: 97%<br>GPU2*: 8%</span> | <span style="color:gray">12.1/24</span> | <span style="color:gray">4.6/17.6/125.0</span> | <span style="color:gray">3 (029, 036, 115)</span> |
| jinx01 | gtx_1080 | 2/2 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 11%<br>GPU1*: 10% | 7.5/16 | 2.7/14.6/214.8 | 2 (004, 068) |
| <span style="color:gray">0 (yuxinchen)</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded/2</span> | <span style="color:gray">not recorded/not recorded/not recorded</span> | <span style="color:gray">not recorded</span> |
| <span style="color:gray">1 (yuxinchen)</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded/48</span> | <span style="color:gray">not recorded/not recorded/not recorded</span> | <span style="color:gray">not recorded</span> |
| <span style="color:gray">2 (yuxinchen)</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded/16</span> | <span style="color:gray">not recorded/not recorded/not recorded</span> | <span style="color:gray">not recorded</span> |
| <span style="color:gray">3 (yuxinchen)</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded/264</span> | <span style="color:gray">not recorded/not recorded/not recorded</span> | <span style="color:gray">not recorded</span> |
| <span style="color:gray">4 (yuxinchen)</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded/64</span> | <span style="color:gray">not recorded/not recorded/not recorded</span> | <span style="color:gray">not recorded</span> |
| **TOTAL (owner)** | — | **39/39** | **184.3/591.9 GB** | **mean 36% (all listed GPUs)** | **228.5/440** | **81.2/380.9/7214.8** | **55** |
| <span style="color:gray">**TOTAL (collaborators)**</span> | <span style="color:gray">—</span> | <span style="color:gray">**46/46**</span> | <span style="color:gray">**155.1/598.9 GB**</span> | <span style="color:gray">**mean 32% (all listed GPUs)**</span> | <span style="color:gray">**207.0/762**</span> | <span style="color:gray">**72.6/269.5/5919.9**</span> | <span style="color:gray">**46**</span> |
| **TOTAL** | — | **85/85** | **339.4/1190.8 GB** | **mean 34% (all listed GPUs)** | **435.5/1202** | **153.8/650.4/13134.8** | **101** |


## 2026-08-06, fifth tick — a lost worker slot, and a monitor that could not see the owner's jobs

**Collaborator report, second one, and a good one.** A worker slot on lynx10 died about a minute into
job 6534113 with `OSError: [Errno 116] Stale file handle` raised inside `claim_one`'s peek loop. The
job stayed RUNNING with two of its three slots, so one GPU would have idled for the whole campaign.

The peek loop caught only `FileNotFoundError`, which does not cover errno 116 — a transient failure
this NFS-backed queue directory can raise when dozens of slots list and read it within a few seconds,
which is exactly what happened when the collaborator's 49 slots all claimed at once. An uncaught
exception there killed the slot thread permanently; nothing restarts it.

**No marker was leaked**, because line 314 is before the claiming `os.rename` — the run that slot was
inspecting stayed in `pending/`. The report said so explicitly, which saved a hunt for a missing
marker.

Two fixes in the shared `worker_manager.py`, since this affects every sweep run from that skill: the
peek block now catches `(OSError, json.JSONDecodeError)`, and a slot retries a failed claim five times
with a widening backoff before giving up with a printed reason. Tested by injecting an errno 116 on
the first marker read — `claim_one` skipped that candidate, claimed another run, and the slot kept
working; before the change the same injection killed the thread.

The collaborator's two orphaned markers (run 0 and run 56, from the job they cancelled) were requeued.
Neither had written a record, so nothing was archived and nothing was lost.

**A second defect the report led to, owner-side and quieter.** `monitor.py` read only
`slurm/submitted_jobids.txt`, while this sweep follows the shared rlprojects convention of one id file
per sweep — `submitted_jobids_<sweep_id>.txt`. The **owner's own 11 jobs were therefore invisible to
the monitor**, which reported "21 running (yuxinchen 21)". That is not a cosmetic miscount: a marker
whose claiming job is never queried can never be judged orphaned, so when one of the owner's jobs hit
its 4-day walltime its runs would have sat in `running/` forever and never been redone. Every run of
this campaign spans several segments, so this would have leaked runs steadily. Fixed; the monitor now
reads both spellings and reports 32 running (owner 11, yuxinchen 21).

**Estimates against observation**, from the monitor: GPU memory peak 4,872 MB against the 4,900
estimate (0.99), CPU threads busy 6.9 against 8 (0.86) — both right. Host memory peak 2,059 MB against
6,000 (0.34), so that estimate is about three times too generous. It costs nothing here, since CPU and
not memory is what binds the packing, and it is not worth changing mid-campaign.

No top-up: `gpu` 384 of 400 CPU threads and `gnolim` 64 of 80, both at the owner's headroom.
