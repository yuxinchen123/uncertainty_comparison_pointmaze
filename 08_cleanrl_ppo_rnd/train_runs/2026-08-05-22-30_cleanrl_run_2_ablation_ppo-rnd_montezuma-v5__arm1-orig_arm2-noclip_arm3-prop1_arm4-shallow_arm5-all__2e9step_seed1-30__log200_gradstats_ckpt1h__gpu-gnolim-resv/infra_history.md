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
## 2026-08-06 02:27  (jobs: 32 running (owner 11, yuxinchen 21), 0 pending | runs: 104 running, 46 pending, 0 done, 0 failed)

| node | gpu type | gpus ours/node | gpu memory used/total (every gpu) | gpu usage (every gpu) | cpu busy/asked (threads) | sys memory used/asked/node (GB) | running runs |
|---|---|---|---|---|---|---|---|
| <span style="color:gray">cheetah01 (yuxinchen)</span> | <span style="color:gray">a100</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.1/40.0 GB</span> | <span style="color:gray">GPU0*: 3%</span> | <span style="color:gray">6.7/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (079)</span> |
| <span style="color:gray">jaguar06 (yuxinchen)</span> | <span style="color:gray">a40</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.3/45.0 GB</span> | <span style="color:gray">GPU0*: 6%</span> | <span style="color:gray">5.3/8</span> | <span style="color:gray">1.5/5.9/123.0</span> | <span style="color:gray">1 (048)</span> |
| <span style="color:gray">cheetah02 (yuxinchen)</span> | <span style="color:gray">rtx_4000_ada</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.3/20.0 GB<br>GPU1*: 3.3/20.0 GB<br>GPU2*: 3.3/20.0 GB<br>GPU3*: 3.3/20.0 GB</span> | <span style="color:gray">GPU0*: 5%<br>GPU1*: 4%<br>GPU2*: 4%<br>GPU3*: 3%</span> | <span style="color:gray">18.8/32</span> | <span style="color:gray">6.1/23.4/1000.0</span> | <span style="color:gray">4 (066–067, 085–086)</span> |
| jaguar03 | rtx_a4500 | 8/8 | GPU0*: 10.7/20.0 GB<br>GPU1*: 10.0/20.0 GB<br>GPU2*: 10.1/20.0 GB<br>GPU3*: 9.7/20.0 GB<br>GPU4*: 10.0/20.0 GB<br>GPU5*: 9.9/20.0 GB<br>GPU6*: 10.0/20.0 GB<br>GPU7*: 10.0/20.0 GB | GPU0*: 99%<br>GPU1*: 22%<br>GPU2*: 82%<br>GPU3*: 100%<br>GPU4*: 97%<br>GPU5*: 99%<br>GPU6*: 100%<br>GPU7*: 25% | 92.1/192 | 36.5/175.8/1000.0 | 24 (001, 005, 007, 010–022, 024–026, 028, 030–031, 038, 044) |
| <span style="color:gray">cheetah08 (yuxinchen)</span> | <span style="color:gray">rtx_a4000</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 7%<br>GPU2*: 6%<br>GPU3*: 9%</span> | <span style="color:gray">18.7/32</span> | <span style="color:gray">6.1/23.4/500.0</span> | <span style="color:gray">4 (058, 061, 073, 078)</span> |
| cheetah09 | rtx_a4000 | 4/4 | GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB | GPU0*: 96%<br>GPU1*: 7%<br>GPU2*: 7%<br>GPU3*: 5% | 18.6/32 | 6.0/29.3/500.0 | 4 (003, 023, 034–035) |
| <span style="color:gray">jaguar02 (yuxinchen)</span> | <span style="color:gray">a16</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.2/15.0 GB<br>GPU1*: 3.1/15.0 GB<br>GPU2*: 3.2/15.0 GB</span> | <span style="color:gray">GPU0*: 21%<br>GPU1*: 21%<br>GPU2*: 100%</span> | <span style="color:gray">8.2/24</span> | <span style="color:gray">4.5/17.6/1000.0</span> | <span style="color:gray">3 (084, 101, 112)</span> |
| lotus | rtx_6000 | 8/8 | GPU0*: 3.4/24.0 GB<br>GPU1*: 3.4/24.0 GB<br>GPU2*: 3.4/24.0 GB<br>GPU3*: 3.4/24.0 GB<br>GPU4*: 3.4/24.0 GB<br>GPU5*: 3.4/24.0 GB<br>GPU6*: 3.4/24.0 GB<br>GPU7*: 3.4/24.0 GB | GPU0*: 8%<br>GPU1*: 7%<br>GPU2*: 7%<br>GPU3*: 96%<br>GPU4*: 86%<br>GPU5*: 8%<br>GPU6*: 7%<br>GPU7*: 90% | 36.8/64 | 11.7/58.6/250.0 | 8 (002, 008, 032, 039, 054, 072, 074–075) |
| <span style="color:gray">ai01 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 6%<br>GPU2*: 6%</span> | <span style="color:gray">14.4/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (040, 047, 069)</span> |
| <span style="color:gray">ai02 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 6%<br>GPU2*: 73%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (037, 045, 094)</span> |
| <span style="color:gray">ai03 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 4.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 47%<br>GPU1*: 5%<br>GPU2*: 6%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (059, 076, 081)</span> |
| <span style="color:gray">ai04 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 6%<br>GPU2*: 46%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">5.0/17.6/62.5</span> | <span style="color:gray">3 (083, 088, 097)</span> |
| <span style="color:gray">ai06 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 84%</span> | <span style="color:gray">5.0/8</span> | <span style="color:gray">2.0/5.9/62.5</span> | <span style="color:gray">1 (092)</span> |
| cheetah03 | rtx_2080_ti | 2/2 | GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB | GPU0*: 54%<br>GPU1*: 97% | 9.2/16 | 2.9/14.6/1000.0 | 2 (055, 064) |
| <span style="color:gray">lynx10 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 13%<br>GPU1*: 50%<br>GPU2*: 5%</span> | <span style="color:gray">14.1/24</span> | <span style="color:gray">4.6/17.6/62.5</span> | <span style="color:gray">3 (113, 122, 124)</span> |
| <span style="color:gray">adriatic01 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.1/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.1/8.0 GB</span> | <span style="color:gray">GPU0*: 8%<br>GPU1*: 14%<br>GPU2*: 98%</span> | <span style="color:gray">12.7/24</span> | <span style="color:gray">4.4/17.6/1000.0</span> | <span style="color:gray">3 (090, 114, 123)</span> |
| adriatic02 | quadro_rtx_4000 | 3/3 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB | GPU0*: 10%<br>GPU1*: 10%<br>GPU2*: 7% | 12.8/24 | 4.4/17.6/1000.0 | 3 (027, 033, 050) |
| adriatic03 | quadro_rtx_4000 | 3/3 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.2/8.0 GB | GPU0*: 8%<br>GPU1*: 8%<br>GPU2*: 22% | 12.6/24 | 4.4/17.6/1000.0 | 3 (051, 060, 062) |
| adriatic04 (owner, yuxinchen) | quadro_rtx_4000 | 2/2 | GPU0*: 4.4/8.0 GB<br>GPU1*: 3.2/8.0 GB | GPU0*: 10%<br>GPU1*: 8% | 8.6/16 | 3.0/11.7/1000.0 | 2 (042, 046) |
| adriatic05 (owner, yuxinchen) | quadro_rtx_4000 | 1/1 | GPU0*: 3.3/8.0 GB | GPU0*: 10% | 4.8/8 | 1.5/5.9/1000.0 | 1 (071) |
| <span style="color:gray">adriatic06 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 12%<br>GPU2*: 10%</span> | <span style="color:gray">12.8/24</span> | <span style="color:gray">5.0/17.6/1000.0</span> | <span style="color:gray">3 (103, 108, 116)</span> |
| <span style="color:gray">jaguar05 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.4/8.0 GB</span> | <span style="color:gray">GPU0*: 7%</span> | <span style="color:gray">4.6/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (089)</span> |
| <span style="color:gray">lynx01 (yuxinchen)</span> | <span style="color:gray">titan_xp</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/12.0 GB</span> | <span style="color:gray">GPU0*: 7%</span> | <span style="color:gray">4.8/8</span> | <span style="color:gray">1.9/5.9/62.5</span> | <span style="color:gray">1 (077)</span> |
| ai07 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 7%<br>GPU1*: 5%<br>GPU2*: 6% | 13.6/24 | 4.0/17.6/125.0 | 3 (009, 049, 063) |
| ai08 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 7%<br>GPU1*: 7%<br>GPU2*: 6% | 13.8/24 | 4.0/17.6/125.0 | 3 (053, 065, 070) |
| <span style="color:gray">ai09 (yuxinchen)</span> | <span style="color:gray">gtx_1080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 6%<br>GPU2*: 6%</span> | <span style="color:gray">13.7/24</span> | <span style="color:gray">4.0/17.6/109.4</span> | <span style="color:gray">3 (091, 096, 119)</span> |
| <span style="color:gray">ai05 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 97%<br>GPU1*: 14%<br>GPU2*: 96%</span> | <span style="color:gray">12.2/24</span> | <span style="color:gray">4.3/17.6/125.0</span> | <span style="color:gray">3 (041, 105, 107)</span> |
| <span style="color:gray">ai10 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 9%<br>GPU1*: 9%<br>GPU2*: 8%</span> | <span style="color:gray">12.6/24</span> | <span style="color:gray">4.6/17.6/125.0</span> | <span style="color:gray">3 (029, 036, 115)</span> |
| jinx01 | gtx_1080 | 2/2 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 96%<br>GPU1*: 97% | 7.4/16 | 2.7/14.6/214.8 | 2 (004, 068) |
| <span style="color:gray">0 (yuxinchen)</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded/2</span> | <span style="color:gray">not recorded/not recorded/not recorded</span> | <span style="color:gray">not recorded</span> |
| <span style="color:gray">1 (yuxinchen)</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded/48</span> | <span style="color:gray">not recorded/not recorded/not recorded</span> | <span style="color:gray">not recorded</span> |
| <span style="color:gray">2 (yuxinchen)</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded/16</span> | <span style="color:gray">not recorded/not recorded/not recorded</span> | <span style="color:gray">not recorded</span> |
| <span style="color:gray">3 (yuxinchen)</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded/264</span> | <span style="color:gray">not recorded/not recorded/not recorded</span> | <span style="color:gray">not recorded</span> |
| <span style="color:gray">4 (yuxinchen)</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded</span> | <span style="color:gray">not recorded/64</span> | <span style="color:gray">not recorded/not recorded/not recorded</span> | <span style="color:gray">not recorded</span> |
| **TOTAL (owner)** | — | **39/39** | **184.3/591.9 GB** | **mean 39% (all listed GPUs)** | **230.3/440** | **81.2/380.9/7214.8** | **55** |
| <span style="color:gray">**TOTAL (collaborators)**</span> | <span style="color:gray">—</span> | <span style="color:gray">**46/46**</span> | <span style="color:gray">**155.1/598.9 GB**</span> | <span style="color:gray">**mean 21% (all listed GPUs)**</span> | <span style="color:gray">**207.2/762**</span> | <span style="color:gray">**72.9/269.5/5919.9**</span> | <span style="color:gray">**46**</span> |
| **TOTAL** | — | **85/85** | **339.4/1190.8 GB** | **mean 29% (all listed GPUs)** | **437.5/1202** | **154.1/650.4/13134.8** | **101** |

## 2026-08-06 02:29  (jobs: 32 running (owner 11, yuxinchen 21), 0 pending | runs: 104 running, 46 pending, 0 done, 0 failed)

| node | gpu type | gpus ours/node | gpu memory used/total (every gpu) | gpu usage (every gpu) | cpu busy/asked (threads) | sys memory used/asked/node (GB) | running runs |
|---|---|---|---|---|---|---|---|
| <span style="color:gray">cheetah01 (yuxinchen)</span> | <span style="color:gray">a100</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.1/40.0 GB</span> | <span style="color:gray">GPU0*: 2%</span> | <span style="color:gray">6.6/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (079)</span> |
| <span style="color:gray">jaguar06 (yuxinchen)</span> | <span style="color:gray">a40</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.3/45.0 GB</span> | <span style="color:gray">GPU0*: 7%</span> | <span style="color:gray">5.3/8</span> | <span style="color:gray">1.5/5.9/123.0</span> | <span style="color:gray">1 (048)</span> |
| <span style="color:gray">cheetah02 (yuxinchen)</span> | <span style="color:gray">rtx_4000_ada</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.3/20.0 GB<br>GPU1*: 3.3/20.0 GB<br>GPU2*: 3.3/20.0 GB<br>GPU3*: 3.3/20.0 GB</span> | <span style="color:gray">GPU0*: 4%<br>GPU1*: 5%<br>GPU2*: 4%<br>GPU3*: 5%</span> | <span style="color:gray">18.7/32</span> | <span style="color:gray">6.1/23.4/1000.0</span> | <span style="color:gray">4 (066–067, 085–086)</span> |
| jaguar03 | rtx_a4500 | 8/8 | GPU0*: 10.7/20.0 GB<br>GPU1*: 10.0/20.0 GB<br>GPU2*: 10.1/20.0 GB<br>GPU3*: 9.7/20.0 GB<br>GPU4*: 10.0/20.0 GB<br>GPU5*: 9.9/20.0 GB<br>GPU6*: 10.0/20.0 GB<br>GPU7*: 10.0/20.0 GB | GPU0*: 24%<br>GPU1*: 28%<br>GPU2*: 25%<br>GPU3*: 17%<br>GPU4*: 19%<br>GPU5*: 99%<br>GPU6*: 97%<br>GPU7*: 78% | 92.8/192 | 36.5/175.8/1000.0 | 24 (001, 005, 007, 010–022, 024–026, 028, 030–031, 038, 044) |
| <span style="color:gray">cheetah08 (yuxinchen)</span> | <span style="color:gray">rtx_a4000</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB</span> | <span style="color:gray">GPU0*: 9%<br>GPU1*: 7%<br>GPU2*: 94%<br>GPU3*: 8%</span> | <span style="color:gray">18.6/32</span> | <span style="color:gray">6.1/23.4/500.0</span> | <span style="color:gray">4 (058, 061, 073, 078)</span> |
| cheetah09 | rtx_a4000 | 4/4 | GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB | GPU0*: 95%<br>GPU1*: 93%<br>GPU2*: 7%<br>GPU3*: 97% | 18.8/32 | 6.0/29.3/500.0 | 4 (003, 023, 034–035) |
| <span style="color:gray">jaguar02 (yuxinchen)</span> | <span style="color:gray">a16</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.2/15.0 GB<br>GPU1*: 3.1/15.0 GB<br>GPU2*: 3.2/15.0 GB</span> | <span style="color:gray">GPU0*: 100%<br>GPU1*: 21%<br>GPU2*: 95%</span> | <span style="color:gray">8.2/24</span> | <span style="color:gray">4.5/17.6/1000.0</span> | <span style="color:gray">3 (084, 101, 112)</span> |
| lotus | rtx_6000 | 8/8 | GPU0*: 3.4/24.0 GB<br>GPU1*: 3.4/24.0 GB<br>GPU2*: 3.4/24.0 GB<br>GPU3*: 3.4/24.0 GB<br>GPU4*: 3.4/24.0 GB<br>GPU5*: 3.4/24.0 GB<br>GPU6*: 3.4/24.0 GB<br>GPU7*: 3.4/24.0 GB | GPU0*: 97%<br>GPU1*: 8%<br>GPU2*: 5%<br>GPU3*: 8%<br>GPU4*: 7%<br>GPU5*: 9%<br>GPU6*: 7%<br>GPU7*: 66% | 36.8/64 | 11.7/58.6/250.0 | 8 (002, 008, 032, 039, 054, 072, 074–075) |
| <span style="color:gray">ai01 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 5%<br>GPU2*: 5%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (040, 047, 069)</span> |
| <span style="color:gray">ai02 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 92%<br>GPU1*: 15%<br>GPU2*: 6%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (037, 045, 094)</span> |
| <span style="color:gray">ai03 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 4.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 5%<br>GPU1*: 5%<br>GPU2*: 5%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (059, 076, 081)</span> |
| <span style="color:gray">ai04 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 17%<br>GPU1*: 81%<br>GPU2*: 67%</span> | <span style="color:gray">14.1/24</span> | <span style="color:gray">5.0/17.6/62.5</span> | <span style="color:gray">3 (083, 088, 097)</span> |
| <span style="color:gray">ai06 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 6%</span> | <span style="color:gray">5.1/8</span> | <span style="color:gray">2.0/5.9/62.5</span> | <span style="color:gray">1 (092)</span> |
| cheetah03 | rtx_2080_ti | 2/2 | GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB | GPU0*: 66%<br>GPU1*: 9% | 9.0/16 | 2.9/14.6/1000.0 | 2 (055, 064) |
| <span style="color:gray">lynx10 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 13%<br>GPU1*: 4%<br>GPU2*: 5%</span> | <span style="color:gray">14.4/24</span> | <span style="color:gray">4.6/17.6/62.5</span> | <span style="color:gray">3 (113, 122, 124)</span> |
| <span style="color:gray">adriatic01 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.1/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.1/8.0 GB</span> | <span style="color:gray">GPU0*: 94%<br>GPU1*: 7%<br>GPU2*: 13%</span> | <span style="color:gray">12.6/24</span> | <span style="color:gray">4.4/17.6/1000.0</span> | <span style="color:gray">3 (090, 114, 123)</span> |
| adriatic02 | quadro_rtx_4000 | 3/3 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB | GPU0*: 8%<br>GPU1*: 98%<br>GPU2*: 10% | 12.6/24 | 4.4/17.6/1000.0 | 3 (027, 033, 050) |
| adriatic03 | quadro_rtx_4000 | 3/3 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.2/8.0 GB | GPU0*: 97%<br>GPU1*: 8%<br>GPU2*: 9% | 12.5/24 | 4.4/17.6/1000.0 | 3 (051, 060, 062) |
| adriatic04 (owner, yuxinchen) | quadro_rtx_4000 | 2/2 | GPU0*: 4.4/8.0 GB<br>GPU1*: 3.2/8.0 GB | GPU0*: 10%<br>GPU1*: 8% | 8.5/24 | 3.0/11.7/1000.0 | 2 (042, 046) |
| adriatic05 (owner, yuxinchen) | quadro_rtx_4000 | 1/1 | GPU0*: 3.3/8.0 GB | GPU0*: 7% | 4.7/24 | 1.5/5.9/1000.0 | 1 (071) |
| <span style="color:gray">adriatic06 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 94%<br>GPU1*: 94%<br>GPU2*: 94%</span> | <span style="color:gray">12.9/24</span> | <span style="color:gray">5.0/17.6/1000.0</span> | <span style="color:gray">3 (103, 108, 116)</span> |
| <span style="color:gray">jaguar05 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.4/8.0 GB</span> | <span style="color:gray">GPU0*: 97%</span> | <span style="color:gray">4.6/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (089)</span> |
| <span style="color:gray">lynx01 (yuxinchen)</span> | <span style="color:gray">titan_xp</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/12.0 GB</span> | <span style="color:gray">GPU0*: 7%</span> | <span style="color:gray">4.9/8</span> | <span style="color:gray">1.9/5.9/62.5</span> | <span style="color:gray">1 (077)</span> |
| ai07 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 6%<br>GPU1*: 6%<br>GPU2*: 6% | 13.7/24 | 4.0/17.6/125.0 | 3 (009, 049, 063) |
| ai08 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 5%<br>GPU1*: 6%<br>GPU2*: 6% | 13.7/24 | 4.0/17.6/125.0 | 3 (053, 065, 070) |
| <span style="color:gray">ai09 (yuxinchen)</span> | <span style="color:gray">gtx_1080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB</span> | <span style="color:gray">GPU0*: 97%<br>GPU1*: 91%<br>GPU2*: 94%</span> | <span style="color:gray">13.8/24</span> | <span style="color:gray">4.0/17.6/109.4</span> | <span style="color:gray">3 (091, 096, 119)</span> |
| <span style="color:gray">ai05 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 9%<br>GPU1*: 9%<br>GPU2*: 8%</span> | <span style="color:gray">12.5/26</span> | <span style="color:gray">4.3/17.6/125.0</span> | <span style="color:gray">3 (041, 105, 107)</span> |
| <span style="color:gray">ai10 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 96%<br>GPU1*: 97%<br>GPU2*: 97%</span> | <span style="color:gray">12.5/24</span> | <span style="color:gray">4.6/17.6/125.0</span> | <span style="color:gray">3 (029, 036, 115)</span> |
| jinx01 | gtx_1080 | 2/2 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 10%<br>GPU1*: 100% | 7.5/16 | 2.7/14.6/214.8 | 2 (004, 068) |
| **TOTAL (owner)** | — | **39/39** | **184.3/591.9 GB** | **mean 35% (all listed GPUs)** | **230.6/464** | **81.2/380.9/7214.8** | **55** |
| <span style="color:gray">**TOTAL (collaborators)**</span> | <span style="color:gray">—</span> | <span style="color:gray">**46/46**</span> | <span style="color:gray">**155.1/598.9 GB**</span> | <span style="color:gray">**mean 39% (all listed GPUs)**</span> | <span style="color:gray">**207.4/370**</span> | <span style="color:gray">**72.9/269.5/5919.9**</span> | <span style="color:gray">**46**</span> |
| **TOTAL** | — | **85/85** | **339.4/1190.8 GB** | **mean 37% (all listed GPUs)** | **438.1/834** | **154.1/650.4/13134.8** | **101** |

## 2026-08-06 02:47  (jobs: 32 running (owner 11, yuxinchen 21), 0 pending | runs: 104 running, 46 pending, 0 done, 0 failed)

| node | gpu type | gpus ours/node | gpu memory used/total (every gpu) | gpu usage (every gpu) | cpu busy/asked (threads) | sys memory used/asked/node (GB) | running runs |
|---|---|---|---|---|---|---|---|
| <span style="color:gray">cheetah01 (yuxinchen)</span> | <span style="color:gray">a100</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.1/40.0 GB</span> | <span style="color:gray">GPU0*: 2%</span> | <span style="color:gray">6.7/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (079)</span> |
| <span style="color:gray">jaguar06 (yuxinchen)</span> | <span style="color:gray">a40</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.3/45.0 GB</span> | <span style="color:gray">GPU0*: 5%</span> | <span style="color:gray">5.3/8</span> | <span style="color:gray">1.5/5.9/123.0</span> | <span style="color:gray">1 (048)</span> |
| <span style="color:gray">cheetah02 (yuxinchen)</span> | <span style="color:gray">rtx_4000_ada</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.3/20.0 GB<br>GPU1*: 3.3/20.0 GB<br>GPU2*: 3.3/20.0 GB<br>GPU3*: 3.3/20.0 GB</span> | <span style="color:gray">GPU0*: 5%<br>GPU1*: 5%<br>GPU2*: 6%<br>GPU3*: 5%</span> | <span style="color:gray">18.8/32</span> | <span style="color:gray">6.1/23.4/1000.0</span> | <span style="color:gray">4 (066–067, 085–086)</span> |
| jaguar03 | rtx_a4500 | 8/8 | GPU0*: 10.7/20.0 GB<br>GPU1*: 10.0/20.0 GB<br>GPU2*: 10.1/20.0 GB<br>GPU3*: 9.7/20.0 GB<br>GPU4*: 10.0/20.0 GB<br>GPU5*: 9.9/20.0 GB<br>GPU6*: 10.0/20.0 GB<br>GPU7*: 10.0/20.0 GB | GPU0*: 99%<br>GPU1*: 96%<br>GPU2*: 100%<br>GPU3*: 100%<br>GPU4*: 22%<br>GPU5*: 23%<br>GPU6*: 74%<br>GPU7*: 98% | 91.7/192 | 37.8/175.8/1000.0 | 24 (001, 005, 007, 010–022, 024–026, 028, 030–031, 038, 044) |
| <span style="color:gray">cheetah08 (yuxinchen)</span> | <span style="color:gray">rtx_a4000</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB</span> | <span style="color:gray">GPU0*: 90%<br>GPU1*: 6%<br>GPU2*: 9%<br>GPU3*: 6%</span> | <span style="color:gray">18.7/32</span> | <span style="color:gray">6.1/23.4/500.0</span> | <span style="color:gray">4 (058, 061, 073, 078)</span> |
| cheetah09 | rtx_a4000 | 4/4 | GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB | GPU0*: 8%<br>GPU1*: 93%<br>GPU2*: 6%<br>GPU3*: 6% | 18.6/32 | 6.2/29.3/500.0 | 4 (003, 023, 034–035) |
| <span style="color:gray">jaguar02 (yuxinchen)</span> | <span style="color:gray">a16</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.2/15.0 GB<br>GPU1*: 3.1/15.0 GB<br>GPU2*: 3.2/15.0 GB</span> | <span style="color:gray">GPU0*: 30%<br>GPU1*: 18%<br>GPU2*: 100%</span> | <span style="color:gray">8.1/24</span> | <span style="color:gray">4.5/17.6/1000.0</span> | <span style="color:gray">3 (084, 101, 112)</span> |
| lotus | rtx_6000 | 8/8 | GPU0*: 3.4/24.0 GB<br>GPU1*: 3.4/24.0 GB<br>GPU2*: 3.4/24.0 GB<br>GPU3*: 3.4/24.0 GB<br>GPU4*: 3.4/24.0 GB<br>GPU5*: 3.4/24.0 GB<br>GPU6*: 3.4/24.0 GB<br>GPU7*: 3.4/24.0 GB | GPU0*: 84%<br>GPU1*: 48%<br>GPU2*: 8%<br>GPU3*: 6%<br>GPU4*: 94%<br>GPU5*: 0%<br>GPU6*: 7%<br>GPU7*: 6% | 36.7/64 | 11.8/58.6/250.0 | 8 (002, 008, 032, 039, 054, 072, 074–075) |
| <span style="color:gray">ai01 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 53%<br>GPU1*: 7%<br>GPU2*: 62%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (040, 047, 069)</span> |
| <span style="color:gray">ai02 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 62%<br>GPU1*: 8%<br>GPU2*: 62%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (037, 045, 094)</span> |
| <span style="color:gray">ai03 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 4.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 16%<br>GPU2*: 6%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">4.9/17.6/62.5</span> | <span style="color:gray">3 (059, 076, 081)</span> |
| <span style="color:gray">ai04 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 6%<br>GPU2*: 5%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">5.0/17.6/62.5</span> | <span style="color:gray">3 (083, 088, 097)</span> |
| <span style="color:gray">ai06 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 6%</span> | <span style="color:gray">5.0/8</span> | <span style="color:gray">2.0/5.9/62.5</span> | <span style="color:gray">1 (092)</span> |
| cheetah03 | rtx_2080_ti | 2/2 | GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB | GPU0*: 7%<br>GPU1*: 89% | 8.9/16 | 3.0/14.6/1000.0 | 2 (055, 064) |
| <span style="color:gray">lynx10 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 51%<br>GPU1*: 6%<br>GPU2*: 6%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">4.6/17.6/62.5</span> | <span style="color:gray">3 (113, 122, 124)</span> |
| <span style="color:gray">adriatic01 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.1/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.1/8.0 GB</span> | <span style="color:gray">GPU0*: 93%<br>GPU1*: 97%<br>GPU2*: 95%</span> | <span style="color:gray">13.0/24</span> | <span style="color:gray">4.4/17.6/1000.0</span> | <span style="color:gray">3 (090, 114, 123)</span> |
| adriatic02 | quadro_rtx_4000 | 3/3 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB | GPU0*: 93%<br>GPU1*: 64%<br>GPU2*: 8% | 12.6/24 | 4.6/17.6/1000.0 | 3 (027, 033, 050) |
| adriatic03 | quadro_rtx_4000 | 3/3 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.2/8.0 GB | GPU0*: 7%<br>GPU1*: 8%<br>GPU2*: 10% | 12.7/24 | 4.5/17.6/1000.0 | 3 (051, 060, 062) |
| adriatic04 (owner, yuxinchen) | quadro_rtx_4000 | 2/2 | GPU0*: 4.4/8.0 GB<br>GPU1*: 3.2/8.0 GB | GPU0*: 95%<br>GPU1*: 15% | 8.2/24 | 3.0/11.7/1000.0 | 2 (042, 046) |
| adriatic05 (owner, yuxinchen) | quadro_rtx_4000 | 1/1 | GPU0*: 3.3/8.0 GB | GPU0*: 97% | 4.7/24 | 1.5/5.9/1000.0 | 1 (071) |
| <span style="color:gray">adriatic06 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 8%<br>GPU1*: 12%<br>GPU2*: 83%</span> | <span style="color:gray">12.9/24</span> | <span style="color:gray">5.0/17.6/1000.0</span> | <span style="color:gray">3 (103, 108, 116)</span> |
| <span style="color:gray">jaguar05 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.4/8.0 GB</span> | <span style="color:gray">GPU0*: 10%</span> | <span style="color:gray">4.7/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (089)</span> |
| <span style="color:gray">lynx01 (yuxinchen)</span> | <span style="color:gray">titan_xp</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/12.0 GB</span> | <span style="color:gray">GPU0*: 94%</span> | <span style="color:gray">4.9/8</span> | <span style="color:gray">1.9/5.9/62.5</span> | <span style="color:gray">1 (077)</span> |
| ai07 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 7%<br>GPU1*: 6%<br>GPU2*: 6% | 13.6/24 | 4.0/17.6/125.0 | 3 (009, 049, 063) |
| ai08 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 6%<br>GPU1*: 7%<br>GPU2*: 7% | 13.9/24 | 4.0/17.6/125.0 | 3 (053, 065, 070) |
| <span style="color:gray">ai09 (yuxinchen)</span> | <span style="color:gray">gtx_1080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB</span> | <span style="color:gray">GPU0*: 9%<br>GPU1*: 95%<br>GPU2*: 97%</span> | <span style="color:gray">13.8/24</span> | <span style="color:gray">4.0/17.6/109.4</span> | <span style="color:gray">3 (091, 096, 119)</span> |
| <span style="color:gray">ai05 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 9%<br>GPU1*: 15%<br>GPU2*: 7%</span> | <span style="color:gray">12.2/26</span> | <span style="color:gray">4.3/17.6/125.0</span> | <span style="color:gray">3 (041, 105, 107)</span> |
| <span style="color:gray">ai10 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 97%<br>GPU1*: 97%<br>GPU2*: 31%</span> | <span style="color:gray">12.2/24</span> | <span style="color:gray">4.6/17.6/125.0</span> | <span style="color:gray">3 (029, 036, 115)</span> |
| jinx01 | gtx_1080 | 2/2 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 60%<br>GPU1*: 10% | 7.3/16 | 2.7/14.6/214.8 | 2 (004, 068) |
| **TOTAL (owner)** | — | **39/39** | **184.3/591.9 GB** | **mean 41% (all listed GPUs)** | **229.1/464** | **83.2/380.9/7214.8** | **55** |
| <span style="color:gray">**TOTAL (collaborators)**</span> | <span style="color:gray">—</span> | <span style="color:gray">**46/46**</span> | <span style="color:gray">**155.1/598.9 GB**</span> | <span style="color:gray">**mean 35% (all listed GPUs)**</span> | <span style="color:gray">**207.4/370**</span> | <span style="color:gray">**72.9/269.5/5919.9**</span> | <span style="color:gray">**46**</span> |
| **TOTAL** | — | **85/85** | **339.4/1190.8 GB** | **mean 37% (all listed GPUs)** | **436.4/834** | **156.1/650.4/13134.8** | **101** |


## 2026-08-06, sixth tick — every running run is logging; first cross-arm comparison

104 runs executing, 46 pending, none failed, none orphaned. All 104 now have logged rows.

**Campaign throughput: 315,000 steps/s aggregate** across 32 jobs. Mean 10,019,446 steps per run, so
0.50% of each run's 2e9 target and 0.35% of the whole 150-run budget.

**First comparison with a real sample, all arms read at exactly 3,276,800 steps** so a faster node
cannot masquerade as a better arm:

| arm | seeds | joint norm | predictor norm | share | predictor clip fires | scale on predictor | intrinsic | episode length |
|---|---|---|---|---|---|---|---|---|
| arm1_original | 20 | 0.5870 | 0.0331 | 0.58% | 0.345 | 0.859 | 19.87 | 397 |
| arm2_no_rnd_grad_clip | 21 | 0.5457 | 0.0311 | 0.68% | 0.000 | 1.000 | 20.60 | 397 |
| arm3_update_proportion_1 | 20 | 0.5550 | 0.0249 | 0.61% | 0.323 | 0.872 | 19.15 | 401 |
| arm4_shallower_predictor | 21 | 0.5381 | 0.0239 | 0.59% | 0.309 | 0.878 | 17.01 | 426 |
| arm5_all | 22 | 0.5205 | 0.0153 | 0.55% | 0.000 | 1.000 | 16.79 | 420 |

Two effects are already separated from seed noise, and both are mechanical rather than surprising.

The **predictor's gradient norm falls with each departure**, and the ranges barely overlap: arm 1
spans 0.0260 to 0.0471 across its seeds while arm 5 spans 0.0096 to 0.0194. Training the predictor on
the whole batch instead of a random quarter averages its gradient over four times as many samples,
and removing a 512-wide block removes parameters from the norm; arm 5 does both.

The **intrinsic reward is lower in the two shallower-predictor arms** (17.0 and 16.8 against 19.9 to
20.6), which is what a predictor that fits its target more easily should produce — the bonus is the
prediction error.

Extrinsic reward is 0.00 in every arm, as expected at 0.5% of the budget on a game whose first reward
needs a long scripted sequence. Nothing should be read into the arms' ordering yet.

Zero invariant violations and zero non-finite gradient norms across every logged row of all 104 runs.

Topped up by one slot on ai06 — `gpu` was at 376 of 400 threads, and the rule is not to stop early
with GPUs on the table even when the remaining room is small.
## 2026-08-06 03:07  (jobs: 33 running (owner 12, yuxinchen 21), 0 pending | runs: 105 running, 45 pending, 0 done, 0 failed)

| node | gpu type | gpus ours/node | gpu memory used/total (every gpu) | gpu usage (every gpu) | cpu busy/asked (threads) | sys memory used/asked/node (GB) | running runs |
|---|---|---|---|---|---|---|---|
| <span style="color:gray">cheetah01 (yuxinchen)</span> | <span style="color:gray">a100</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.1/40.0 GB</span> | <span style="color:gray">GPU0*: 2%</span> | <span style="color:gray">6.6/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (079)</span> |
| <span style="color:gray">jaguar06 (yuxinchen)</span> | <span style="color:gray">a40</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.3/45.0 GB</span> | <span style="color:gray">GPU0*: 90%</span> | <span style="color:gray">5.2/8</span> | <span style="color:gray">1.6/5.9/123.0</span> | <span style="color:gray">1 (048)</span> |
| <span style="color:gray">cheetah02 (yuxinchen)</span> | <span style="color:gray">rtx_4000_ada</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.3/20.0 GB<br>GPU1*: 3.3/20.0 GB<br>GPU2*: 3.3/20.0 GB<br>GPU3*: 3.3/20.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 91%<br>GPU2*: 5%<br>GPU3*: 52%</span> | <span style="color:gray">18.7/32</span> | <span style="color:gray">6.3/23.4/1000.0</span> | <span style="color:gray">4 (066–067, 085–086)</span> |
| jaguar03 | rtx_a4500 | 8/8 | GPU0*: 10.7/20.0 GB<br>GPU1*: 10.0/20.0 GB<br>GPU2*: 10.1/20.0 GB<br>GPU3*: 9.7/20.0 GB<br>GPU4*: 10.0/20.0 GB<br>GPU5*: 9.9/20.0 GB<br>GPU6*: 10.0/20.0 GB<br>GPU7*: 10.0/20.0 GB | GPU0*: 100%<br>GPU1*: 57%<br>GPU2*: 100%<br>GPU3*: 23%<br>GPU4*: 97%<br>GPU5*: 100%<br>GPU6*: 100%<br>GPU7*: 59% | 92.2/192 | 37.8/175.8/1000.0 | 24 (001, 005, 007, 010–022, 024–026, 028, 030–031, 038, 044) |
| <span style="color:gray">cheetah08 (yuxinchen)</span> | <span style="color:gray">rtx_a4000</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 5%<br>GPU2*: 7%<br>GPU3*: 7%</span> | <span style="color:gray">18.8/32</span> | <span style="color:gray">6.3/23.4/500.0</span> | <span style="color:gray">4 (058, 061, 073, 078)</span> |
| cheetah09 | rtx_a4000 | 4/4 | GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB | GPU0*: 8%<br>GPU1*: 93%<br>GPU2*: 5%<br>GPU3*: 7% | 18.6/32 | 6.2/29.3/500.0 | 4 (003, 023, 034–035) |
| <span style="color:gray">jaguar02 (yuxinchen)</span> | <span style="color:gray">a16</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.2/15.0 GB<br>GPU1*: 3.1/15.0 GB<br>GPU2*: 3.2/15.0 GB</span> | <span style="color:gray">GPU0*: 95%<br>GPU1*: 100%<br>GPU2*: 100%</span> | <span style="color:gray">8.5/24</span> | <span style="color:gray">4.6/17.6/1000.0</span> | <span style="color:gray">3 (084, 101, 112)</span> |
| lotus | rtx_6000 | 8/8 | GPU0*: 3.4/24.0 GB<br>GPU1*: 3.4/24.0 GB<br>GPU2*: 3.4/24.0 GB<br>GPU3*: 3.4/24.0 GB<br>GPU4*: 3.4/24.0 GB<br>GPU5*: 3.4/24.0 GB<br>GPU6*: 3.4/24.0 GB<br>GPU7*: 3.4/24.0 GB | GPU0*: 95%<br>GPU1*: 6%<br>GPU2*: 6%<br>GPU3*: 47%<br>GPU4*: 8%<br>GPU5*: 14%<br>GPU6*: 95%<br>GPU7*: 6% | 36.4/64 | 12.1/58.6/250.0 | 8 (002, 008, 032, 039, 054, 072, 074–075) |
| <span style="color:gray">ai01 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 79%<br>GPU1*: 6%<br>GPU2*: 5%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">5.0/17.6/62.5</span> | <span style="color:gray">3 (040, 047, 069)</span> |
| <span style="color:gray">ai02 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 5%<br>GPU2*: 33%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">5.0/17.6/62.5</span> | <span style="color:gray">3 (037, 045, 094)</span> |
| <span style="color:gray">ai03 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 4.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 79%<br>GPU1*: 6%<br>GPU2*: 7%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">5.0/17.6/62.5</span> | <span style="color:gray">3 (059, 076, 081)</span> |
| <span style="color:gray">ai04 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 5%<br>GPU2*: 61%</span> | <span style="color:gray">14.4/24</span> | <span style="color:gray">5.0/17.6/62.5</span> | <span style="color:gray">3 (083, 088, 097)</span> |
| ai06 (owner, yuxinchen) | rtx_2080_ti | 1/1 | GPU0*: 3.4/11.0 GB | GPU0*: 6% | 5.0/16 | 1.5/7.3/62.5 | 1 (098) |
| cheetah03 | rtx_2080_ti | 2/2 | GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB | GPU0*: 97%<br>GPU1*: 64% | 9.1/16 | 3.0/14.6/1000.0 | 2 (055, 064) |
| <span style="color:gray">lynx10 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 58%<br>GPU2*: 33%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">4.6/17.6/62.5</span> | <span style="color:gray">3 (113, 122, 124)</span> |
| <span style="color:gray">adriatic01 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.1/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.1/8.0 GB</span> | <span style="color:gray">GPU0*: 8%<br>GPU1*: 94%<br>GPU2*: 9%</span> | <span style="color:gray">12.7/24</span> | <span style="color:gray">4.5/17.6/1000.0</span> | <span style="color:gray">3 (090, 114, 123)</span> |
| adriatic02 | quadro_rtx_4000 | 3/3 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB | GPU0*: 21%<br>GPU1*: 7%<br>GPU2*: 7% | 12.9/24 | 4.6/17.6/1000.0 | 3 (027, 033, 050) |
| adriatic03 | quadro_rtx_4000 | 3/3 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.2/8.0 GB | GPU0*: 9%<br>GPU1*: 96%<br>GPU2*: 11% | 12.8/24 | 4.6/17.6/1000.0 | 3 (051, 060, 062) |
| adriatic04 (owner, yuxinchen) | quadro_rtx_4000 | 2/2 | GPU0*: 4.4/8.0 GB<br>GPU1*: 3.2/8.0 GB | GPU0*: 75%<br>GPU1*: 97% | 8.7/24 | 3.1/11.7/1000.0 | 2 (042, 046) |
| adriatic05 (owner, yuxinchen) | quadro_rtx_4000 | 2/2 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 96%<br>GPU1*: 6% | 8.5/24 | 3.1/11.7/1000.0 | 2 (100, 106) |
| <span style="color:gray">adriatic06 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 11%<br>GPU1*: 9%<br>GPU2*: 96%</span> | <span style="color:gray">12.6/24</span> | <span style="color:gray">5.1/17.6/1000.0</span> | <span style="color:gray">3 (103, 108, 116)</span> |
| <span style="color:gray">jaguar05 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.4/8.0 GB</span> | <span style="color:gray">GPU0*: 10%</span> | <span style="color:gray">4.7/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (089)</span> |
| <span style="color:gray">lynx01 (yuxinchen)</span> | <span style="color:gray">titan_xp</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/12.0 GB</span> | <span style="color:gray">GPU0*: 6%</span> | <span style="color:gray">4.9/8</span> | <span style="color:gray">2.0/5.9/62.5</span> | <span style="color:gray">1 (077)</span> |
| ai07 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 96%<br>GPU1*: 96%<br>GPU2*: 10% | 13.6/24 | 4.2/17.6/125.0 | 3 (009, 049, 063) |
| ai08 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 6%<br>GPU1*: 6%<br>GPU2*: 8% | 13.6/24 | 4.2/17.6/125.0 | 3 (053, 065, 070) |
| <span style="color:gray">ai09 (yuxinchen)</span> | <span style="color:gray">gtx_1080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB</span> | <span style="color:gray">GPU0*: 94%<br>GPU1*: 86%<br>GPU2*: 9%</span> | <span style="color:gray">13.6/24</span> | <span style="color:gray">4.0/17.6/109.4</span> | <span style="color:gray">3 (091, 096, 119)</span> |
| <span style="color:gray">ai05 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 12%<br>GPU1*: 94%<br>GPU2*: 10%</span> | <span style="color:gray">12.1/26</span> | <span style="color:gray">4.4/17.6/125.0</span> | <span style="color:gray">3 (041, 105, 107)</span> |
| <span style="color:gray">ai10 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 10%<br>GPU1*: 11%<br>GPU2*: 98%</span> | <span style="color:gray">12.4/24</span> | <span style="color:gray">4.8/17.6/125.0</span> | <span style="color:gray">3 (029, 036, 115)</span> |
| jinx01 | gtx_1080 | 2/2 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 11%<br>GPU1*: 97% | 7.5/16 | 2.8/14.6/214.8 | 2 (004, 068) |
| **TOTAL (owner)** | — | **41/41** | **190.8/610.9 GB** | **mean 48% (all listed GPUs)** | **238.8/480** | **87.0/394.0/7277.3** | **57** |
| <span style="color:gray">**TOTAL (collaborators)**</span> | <span style="color:gray">—</span> | <span style="color:gray">**45/45**</span> | <span style="color:gray">**151.7/587.9 GB**</span> | <span style="color:gray">**mean 36% (all listed GPUs)**</span> | <span style="color:gray">**202.3/362**</span> | <span style="color:gray">**72.3/263.7/5857.4**</span> | <span style="color:gray">**45**</span> |
| **TOTAL** | — | **86/86** | **342.6/1198.8 GB** | **mean 42% (all listed GPUs)** | **441.1/842** | **159.3/657.7/13134.8** | **102** |


## 2026-08-06, seventh tick — the first extrinsic reward

105 runs executing, 45 pending, none failed, none orphaned. 33 jobs (owner 12, collaborator 21).
Mean 13,800,369 steps per run — 0.69% of each run's target, 0.48% of the 150-run budget. Aggregate
316,677 steps/s. Zero invariant violations and zero non-finite gradient norms.

**Eleven runs have scored above zero on the 200-episode extrinsic mean**, and two of them
substantially: run 21 (`arm2_no_rnd_grad_clip`, seed 5) reached 252.0 at 19.7M steps, and run 63
(`arm4_shallower_predictor`, seed 13) reached 265.5 at 16.4M steps. On Montezuma's Revenge the first
room pays 100 for the key and 300 for the door, so a 200-episode mean in the 250s means those agents
are opening the first door reliably rather than once by luck.

The per-arm counts of seeds that have scored at all are 1 of 20, 5 of 21, 0 of 20, 3 of 21, 2 of 22
for arms 1 to 5. **Nothing should be read into that ordering.** Every run has between 3 and 6 logged
rows, the runs are at 0.7% of their step budget, and the quantity being counted is whether a
high-variance early episode happened to land — not a property of the arm. The comparison that will
matter is the one made at a matched step with all 30 seeds, and it is a long way off.

No top-up: `gpu` at 384 of 400 CPU threads and `gnolim` at 64 of 80, both at the owner's headroom.
The collaborator's uid is what adds capacity from here.
## 2026-08-06 03:27  (jobs: 33 running (owner 12, yuxinchen 21), 0 pending | runs: 106 running, 44 pending, 0 done, 0 failed)

| node | gpu type | gpus ours/node | gpu memory used/total (every gpu) | gpu usage (every gpu) | cpu busy/asked (threads) | sys memory used/asked/node (GB) | running runs |
|---|---|---|---|---|---|---|---|
| <span style="color:gray">jaguar06 (yuxinchen)</span> | <span style="color:gray">a40</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.3/45.0 GB</span> | <span style="color:gray">GPU0*: 91%</span> | <span style="color:gray">5.2/8</span> | <span style="color:gray">1.6/5.9/123.0</span> | <span style="color:gray">1 (048)</span> |
| <span style="color:gray">cheetah02 (yuxinchen)</span> | <span style="color:gray">rtx_4000_ada</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.3/20.0 GB<br>GPU1*: 3.3/20.0 GB<br>GPU2*: 3.3/20.0 GB<br>GPU3*: 3.3/20.0 GB</span> | <span style="color:gray">GPU0*: 96%<br>GPU1*: 6%<br>GPU2*: 5%<br>GPU3*: 5%</span> | <span style="color:gray">18.6/32</span> | <span style="color:gray">6.3/23.4/1000.0</span> | <span style="color:gray">4 (066–067, 085–086)</span> |
| jaguar03 | rtx_a4500 | 8/8 | GPU0*: 10.7/20.0 GB<br>GPU1*: 10.0/20.0 GB<br>GPU2*: 10.1/20.0 GB<br>GPU3*: 9.7/20.0 GB<br>GPU4*: 10.0/20.0 GB<br>GPU5*: 9.9/20.0 GB<br>GPU6*: 10.0/20.0 GB<br>GPU7*: 10.0/20.0 GB | GPU0*: 23%<br>GPU1*: 60%<br>GPU2*: 100%<br>GPU3*: 99%<br>GPU4*: 99%<br>GPU5*: 98%<br>GPU6*: 97%<br>GPU7*: 21% | 91.0/192 | 37.8/175.8/1000.0 | 24 (001, 005, 007, 010–022, 024–026, 028, 030–031, 038, 044) |
| <span style="color:gray">cheetah08 (yuxinchen)</span> | <span style="color:gray">rtx_a4000</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB</span> | <span style="color:gray">GPU0*: 93%<br>GPU1*: 8%<br>GPU2*: 90%<br>GPU3*: 94%</span> | <span style="color:gray">18.6/32</span> | <span style="color:gray">6.3/23.4/500.0</span> | <span style="color:gray">4 (058, 061, 073, 078)</span> |
| cheetah09 | rtx_a4000 | 4/4 | GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB | GPU0*: 94%<br>GPU1*: 11%<br>GPU2*: 94%<br>GPU3*: 97% | 18.7/32 | 6.2/29.3/500.0 | 4 (003, 023, 034–035) |
| <span style="color:gray">jaguar02 (yuxinchen)</span> | <span style="color:gray">a16</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.2/15.0 GB<br>GPU1*: 3.1/15.0 GB<br>GPU2*: 3.2/15.0 GB</span> | <span style="color:gray">GPU0*: 21%<br>GPU1*: 21%<br>GPU2*: 18%</span> | <span style="color:gray">8.6/24</span> | <span style="color:gray">4.6/17.6/1000.0</span> | <span style="color:gray">3 (084, 101, 112)</span> |
| lotus | rtx_6000 | 8/8 | GPU0*: 3.4/24.0 GB<br>GPU1*: 3.4/24.0 GB<br>GPU2*: 3.4/24.0 GB<br>GPU3*: 3.4/24.0 GB<br>GPU4*: 3.4/24.0 GB<br>GPU5*: 3.4/24.0 GB<br>GPU6*: 3.4/24.0 GB<br>GPU7*: 3.4/24.0 GB | GPU0*: 94%<br>GPU1*: 8%<br>GPU2*: 95%<br>GPU3*: 7%<br>GPU4*: 5%<br>GPU5*: 7%<br>GPU6*: 8%<br>GPU7*: 8% | 36.6/64 | 12.1/58.6/250.0 | 8 (002, 008, 032, 039, 054, 072, 074–075) |
| <span style="color:gray">affogato11 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 8%</span> | <span style="color:gray">5.2/8</span> | <span style="color:gray">2.0/5.9/125.0</span> | <span style="color:gray">1 (109)</span> |
| <span style="color:gray">ai01 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 32%<br>GPU1*: 6%<br>GPU2*: 5%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (040, 047, 069)</span> |
| <span style="color:gray">ai02 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 92%<br>GPU1*: 90%<br>GPU2*: 6%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (037, 045, 094)</span> |
| <span style="color:gray">ai03 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 4.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 37%<br>GPU1*: 35%<br>GPU2*: 6%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (059, 076, 081)</span> |
| <span style="color:gray">ai04 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 6%<br>GPU2*: 6%</span> | <span style="color:gray">14.4/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (083, 088, 097)</span> |
| ai06 (owner, yuxinchen) | rtx_2080_ti | 1/1 | GPU0*: 3.4/11.0 GB | GPU0*: 6% | 5.0/16 | 1.5/7.3/62.5 | 1 (098) |
| cheetah03 | rtx_2080_ti | 2/2 | GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB | GPU0*: 8%<br>GPU1*: 15% | 8.9/16 | 3.0/14.6/1000.0 | 2 (055, 064) |
| <span style="color:gray">lynx10 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 34%<br>GPU2*: 55%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">4.7/17.6/62.5</span> | <span style="color:gray">3 (113, 122, 124)</span> |
| <span style="color:gray">adriatic01 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.1/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.1/8.0 GB</span> | <span style="color:gray">GPU0*: 96%<br>GPU1*: 8%<br>GPU2*: 12%</span> | <span style="color:gray">12.6/24</span> | <span style="color:gray">4.5/17.6/1000.0</span> | <span style="color:gray">3 (090, 114, 123)</span> |
| adriatic02 | quadro_rtx_4000 | 3/3 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB | GPU0*: 14%<br>GPU1*: 94%<br>GPU2*: 68% | 12.8/24 | 4.6/17.6/1000.0 | 3 (027, 033, 050) |
| adriatic03 | quadro_rtx_4000 | 3/3 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.2/8.0 GB | GPU0*: 10%<br>GPU1*: 10%<br>GPU2*: 12% | 12.9/24 | 4.6/17.6/1000.0 | 3 (051, 060, 062) |
| adriatic04 (owner, yuxinchen) | quadro_rtx_4000 | 2/2 | GPU0*: 4.4/8.0 GB<br>GPU1*: 3.2/8.0 GB | GPU0*: 10%<br>GPU1*: 61% | 8.5/24 | 3.1/11.7/1000.0 | 2 (042, 046) |
| adriatic05 (owner, yuxinchen) | quadro_rtx_4000 | 2/2 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 12%<br>GPU1*: 93% | 8.2/24 | 3.1/11.7/1000.0 | 2 (100, 106) |
| <span style="color:gray">adriatic06 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 10%<br>GPU1*: 98%<br>GPU2*: 91%</span> | <span style="color:gray">12.9/24</span> | <span style="color:gray">5.1/17.6/1000.0</span> | <span style="color:gray">3 (103, 108, 116)</span> |
| <span style="color:gray">jaguar05 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.4/8.0 GB</span> | <span style="color:gray">GPU0*: 11%</span> | <span style="color:gray">4.6/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (089)</span> |
| <span style="color:gray">lynx01 (yuxinchen)</span> | <span style="color:gray">titan_xp</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/12.0 GB</span> | <span style="color:gray">GPU0*: 89%</span> | <span style="color:gray">4.9/8</span> | <span style="color:gray">2.0/5.9/62.5</span> | <span style="color:gray">1 (077)</span> |
| ai07 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 7%<br>GPU1*: 96%<br>GPU2*: 7% | 13.5/24 | 4.2/17.6/125.0 | 3 (009, 049, 063) |
| ai08 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 6%<br>GPU1*: 94%<br>GPU2*: 8% | 13.5/24 | 4.2/17.6/125.0 | 3 (053, 065, 070) |
| <span style="color:gray">ai09 (yuxinchen)</span> | <span style="color:gray">gtx_1080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 7%<br>GPU2*: 7%</span> | <span style="color:gray">13.6/24</span> | <span style="color:gray">4.2/17.6/109.4</span> | <span style="color:gray">3 (091, 096, 119)</span> |
| <span style="color:gray">ai05 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 12%<br>GPU1*: 9%<br>GPU2*: 96%</span> | <span style="color:gray">12.4/26</span> | <span style="color:gray">4.4/17.6/125.0</span> | <span style="color:gray">3 (041, 105, 107)</span> |
| <span style="color:gray">ai10 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 10%<br>GPU1*: 11%<br>GPU2*: 98%</span> | <span style="color:gray">12.5/24</span> | <span style="color:gray">4.8/17.6/125.0</span> | <span style="color:gray">3 (029, 036, 115)</span> |
| jinx01 | gtx_1080 | 2/2 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 8%<br>GPU1*: 9% | 7.5/16 | 2.8/14.6/214.8 | 2 (004, 068) |
| **TOTAL (owner)** | — | **41/41** | **190.8/610.9 GB** | **mean 43% (all listed GPUs)** | **237.4/480** | **87.0/394.0/7277.3** | **57** |
| <span style="color:gray">**TOTAL (collaborators)**</span> | <span style="color:gray">—</span> | <span style="color:gray">**45/45**</span> | <span style="color:gray">**151.0/558.9 GB**</span> | <span style="color:gray">**mean 37% (all listed GPUs)**</span> | <span style="color:gray">**201.2/362**</span> | <span style="color:gray">**73.0/263.7/5732.4**</span> | <span style="color:gray">**45**</span> |
| **TOTAL** | — | **86/86** | **341.9/1169.8 GB** | **mean 40% (all listed GPUs)** | **438.5/842** | **160.0/657.7/13009.8** | **102** |

## 2026-08-06 03:27  (jobs: 33 running (owner 12, yuxinchen 21), 0 pending | runs: 106 running, 44 pending, 0 done, 0 failed)

| node | gpu type | gpus ours/node | gpu memory used/total (every gpu) | gpu usage (every gpu) | cpu busy/asked (threads) | sys memory used/asked/node (GB) | running runs |
|---|---|---|---|---|---|---|---|
| <span style="color:gray">jaguar06 (yuxinchen)</span> | <span style="color:gray">a40</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.3/45.0 GB</span> | <span style="color:gray">GPU0*: 91%</span> | <span style="color:gray">5.2/8</span> | <span style="color:gray">1.6/5.9/123.0</span> | <span style="color:gray">1 (048)</span> |
| <span style="color:gray">cheetah02 (yuxinchen)</span> | <span style="color:gray">rtx_4000_ada</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.3/20.0 GB<br>GPU1*: 3.3/20.0 GB<br>GPU2*: 3.3/20.0 GB<br>GPU3*: 3.3/20.0 GB</span> | <span style="color:gray">GPU0*: 96%<br>GPU1*: 6%<br>GPU2*: 5%<br>GPU3*: 5%</span> | <span style="color:gray">18.6/32</span> | <span style="color:gray">6.3/23.4/1000.0</span> | <span style="color:gray">4 (066–067, 085–086)</span> |
| jaguar03 | rtx_a4500 | 8/8 | GPU0*: 10.7/20.0 GB<br>GPU1*: 10.0/20.0 GB<br>GPU2*: 10.1/20.0 GB<br>GPU3*: 9.7/20.0 GB<br>GPU4*: 10.0/20.0 GB<br>GPU5*: 9.9/20.0 GB<br>GPU6*: 10.0/20.0 GB<br>GPU7*: 10.0/20.0 GB | GPU0*: 23%<br>GPU1*: 60%<br>GPU2*: 100%<br>GPU3*: 99%<br>GPU4*: 99%<br>GPU5*: 98%<br>GPU6*: 97%<br>GPU7*: 21% | 91.0/192 | 37.8/175.8/1000.0 | 24 (001, 005, 007, 010–022, 024–026, 028, 030–031, 038, 044) |
| <span style="color:gray">cheetah08 (yuxinchen)</span> | <span style="color:gray">rtx_a4000</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB</span> | <span style="color:gray">GPU0*: 93%<br>GPU1*: 8%<br>GPU2*: 90%<br>GPU3*: 94%</span> | <span style="color:gray">18.6/32</span> | <span style="color:gray">6.3/23.4/500.0</span> | <span style="color:gray">4 (058, 061, 073, 078)</span> |
| cheetah09 | rtx_a4000 | 4/4 | GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB | GPU0*: 94%<br>GPU1*: 11%<br>GPU2*: 94%<br>GPU3*: 97% | 18.7/32 | 6.2/29.3/500.0 | 4 (003, 023, 034–035) |
| <span style="color:gray">jaguar02 (yuxinchen)</span> | <span style="color:gray">a16</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.2/15.0 GB<br>GPU1*: 3.1/15.0 GB<br>GPU2*: 3.2/15.0 GB</span> | <span style="color:gray">GPU0*: 21%<br>GPU1*: 21%<br>GPU2*: 18%</span> | <span style="color:gray">8.6/24</span> | <span style="color:gray">4.6/17.6/1000.0</span> | <span style="color:gray">3 (084, 101, 112)</span> |
| lotus | rtx_6000 | 8/8 | GPU0*: 3.4/24.0 GB<br>GPU1*: 3.4/24.0 GB<br>GPU2*: 3.4/24.0 GB<br>GPU3*: 3.4/24.0 GB<br>GPU4*: 3.4/24.0 GB<br>GPU5*: 3.4/24.0 GB<br>GPU6*: 3.4/24.0 GB<br>GPU7*: 3.4/24.0 GB | GPU0*: 94%<br>GPU1*: 8%<br>GPU2*: 95%<br>GPU3*: 7%<br>GPU4*: 5%<br>GPU5*: 7%<br>GPU6*: 8%<br>GPU7*: 8% | 36.6/64 | 12.1/58.6/250.0 | 8 (002, 008, 032, 039, 054, 072, 074–075) |
| <span style="color:gray">affogato11 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 8%</span> | <span style="color:gray">5.2/8</span> | <span style="color:gray">2.0/5.9/125.0</span> | <span style="color:gray">1 (109)</span> |
| <span style="color:gray">ai01 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 32%<br>GPU1*: 6%<br>GPU2*: 5%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (040, 047, 069)</span> |
| <span style="color:gray">ai02 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 92%<br>GPU1*: 90%<br>GPU2*: 6%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (037, 045, 094)</span> |
| <span style="color:gray">ai03 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 4.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 37%<br>GPU1*: 35%<br>GPU2*: 6%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (059, 076, 081)</span> |
| <span style="color:gray">ai04 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 6%<br>GPU2*: 6%</span> | <span style="color:gray">14.4/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (083, 088, 097)</span> |
| ai06 (owner, yuxinchen) | rtx_2080_ti | 1/1 | GPU0*: 3.4/11.0 GB | GPU0*: 6% | 5.0/16 | 1.5/7.3/62.5 | 1 (098) |
| cheetah03 | rtx_2080_ti | 2/2 | GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB | GPU0*: 8%<br>GPU1*: 15% | 8.9/16 | 3.0/14.6/1000.0 | 2 (055, 064) |
| <span style="color:gray">lynx10 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 34%<br>GPU2*: 55%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">4.7/17.6/62.5</span> | <span style="color:gray">3 (113, 122, 124)</span> |
| <span style="color:gray">adriatic01 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.1/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.1/8.0 GB</span> | <span style="color:gray">GPU0*: 96%<br>GPU1*: 8%<br>GPU2*: 12%</span> | <span style="color:gray">12.6/24</span> | <span style="color:gray">4.5/17.6/1000.0</span> | <span style="color:gray">3 (090, 114, 123)</span> |
| adriatic02 | quadro_rtx_4000 | 3/3 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB | GPU0*: 14%<br>GPU1*: 94%<br>GPU2*: 68% | 12.8/24 | 4.6/17.6/1000.0 | 3 (027, 033, 050) |
| adriatic03 | quadro_rtx_4000 | 3/3 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.2/8.0 GB | GPU0*: 10%<br>GPU1*: 10%<br>GPU2*: 12% | 12.9/24 | 4.6/17.6/1000.0 | 3 (051, 060, 062) |
| adriatic04 (owner, yuxinchen) | quadro_rtx_4000 | 2/2 | GPU0*: 4.4/8.0 GB<br>GPU1*: 3.2/8.0 GB | GPU0*: 10%<br>GPU1*: 61% | 8.5/24 | 3.1/11.7/1000.0 | 2 (042, 046) |
| adriatic05 (owner, yuxinchen) | quadro_rtx_4000 | 2/2 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 12%<br>GPU1*: 93% | 8.2/24 | 3.1/11.7/1000.0 | 2 (100, 106) |
| <span style="color:gray">adriatic06 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 10%<br>GPU1*: 98%<br>GPU2*: 91%</span> | <span style="color:gray">12.9/24</span> | <span style="color:gray">5.1/17.6/1000.0</span> | <span style="color:gray">3 (103, 108, 116)</span> |
| <span style="color:gray">jaguar05 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.4/8.0 GB</span> | <span style="color:gray">GPU0*: 11%</span> | <span style="color:gray">4.6/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (089)</span> |
| <span style="color:gray">lynx01 (yuxinchen)</span> | <span style="color:gray">titan_xp</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/12.0 GB</span> | <span style="color:gray">GPU0*: 89%</span> | <span style="color:gray">4.9/8</span> | <span style="color:gray">2.0/5.9/62.5</span> | <span style="color:gray">1 (077)</span> |
| ai07 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 7%<br>GPU1*: 96%<br>GPU2*: 7% | 13.5/24 | 4.2/17.6/125.0 | 3 (009, 049, 063) |
| ai08 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 6%<br>GPU1*: 94%<br>GPU2*: 8% | 13.5/24 | 4.2/17.6/125.0 | 3 (053, 065, 070) |
| <span style="color:gray">ai09 (yuxinchen)</span> | <span style="color:gray">gtx_1080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 7%<br>GPU2*: 7%</span> | <span style="color:gray">13.6/24</span> | <span style="color:gray">4.2/17.6/109.4</span> | <span style="color:gray">3 (091, 096, 119)</span> |
| <span style="color:gray">ai05 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 12%<br>GPU1*: 9%<br>GPU2*: 96%</span> | <span style="color:gray">12.4/26</span> | <span style="color:gray">4.4/17.6/125.0</span> | <span style="color:gray">3 (041, 105, 107)</span> |
| <span style="color:gray">ai10 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 10%<br>GPU1*: 11%<br>GPU2*: 98%</span> | <span style="color:gray">12.5/24</span> | <span style="color:gray">4.8/17.6/125.0</span> | <span style="color:gray">3 (029, 036, 115)</span> |
| jinx01 | gtx_1080 | 2/2 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 8%<br>GPU1*: 9% | 7.5/16 | 2.8/14.6/214.8 | 2 (004, 068) |
| **TOTAL (owner)** | — | **41/41** | **190.8/610.9 GB** | **mean 43% (all listed GPUs)** | **237.4/480** | **87.0/394.0/7277.3** | **57** |
| <span style="color:gray">**TOTAL (collaborators)**</span> | <span style="color:gray">—</span> | <span style="color:gray">**45/45**</span> | <span style="color:gray">**151.0/558.9 GB**</span> | <span style="color:gray">**mean 37% (all listed GPUs)**</span> | <span style="color:gray">**201.2/362**</span> | <span style="color:gray">**73.0/263.7/5732.4**</span> | <span style="color:gray">**45**</span> |
| **TOTAL** | — | **86/86** | **341.9/1169.8 GB** | **mean 40% (all listed GPUs)** | **438.5/842** | **160.0/657.7/13009.8** | **102** |

## 2026-08-06 03:50  (jobs: 33 running (owner 12, yuxinchen 21), 0 pending | runs: 106 running, 44 pending, 0 done, 0 failed)

| node | gpu type | gpus ours/node | gpu memory used/total (every gpu) | gpu usage (every gpu) | cpu busy/asked (threads) | sys memory used/asked/node (GB) | running runs |
|---|---|---|---|---|---|---|---|
| <span style="color:gray">jaguar06 (yuxinchen)</span> | <span style="color:gray">a40</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.3/45.0 GB</span> | <span style="color:gray">GPU0*: 6%</span> | <span style="color:gray">5.2/8</span> | <span style="color:gray">1.6/5.9/123.0</span> | <span style="color:gray">1 (048)</span> |
| <span style="color:gray">cheetah02 (yuxinchen)</span> | <span style="color:gray">rtx_4000_ada</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.3/20.0 GB<br>GPU1*: 3.3/20.0 GB<br>GPU2*: 3.3/20.0 GB<br>GPU3*: 3.3/20.0 GB</span> | <span style="color:gray">GPU0*: 5%<br>GPU1*: 5%<br>GPU2*: 5%<br>GPU3*: 4%</span> | <span style="color:gray">18.8/32</span> | <span style="color:gray">6.3/23.4/1000.0</span> | <span style="color:gray">4 (066–067, 085–086)</span> |
| jaguar03 | rtx_a4500 | 8/8 | GPU0*: 10.7/20.0 GB<br>GPU1*: 10.0/20.0 GB<br>GPU2*: 10.1/20.0 GB<br>GPU3*: 9.7/20.0 GB<br>GPU4*: 10.0/20.0 GB<br>GPU5*: 9.9/20.0 GB<br>GPU6*: 10.0/20.0 GB<br>GPU7*: 10.0/20.0 GB | GPU0*: 99%<br>GPU1*: 98%<br>GPU2*: 95%<br>GPU3*: 100%<br>GPU4*: 100%<br>GPU5*: 100%<br>GPU6*: 100%<br>GPU7*: 100% | 91.6/192 | 37.8/175.8/1000.0 | 24 (001, 005, 007, 010–022, 024–026, 028, 030–031, 038, 044) |
| <span style="color:gray">cheetah08 (yuxinchen)</span> | <span style="color:gray">rtx_a4000</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 9%<br>GPU2*: 96%<br>GPU3*: 92%</span> | <span style="color:gray">18.6/32</span> | <span style="color:gray">6.3/23.4/500.0</span> | <span style="color:gray">4 (058, 061, 073, 078)</span> |
| cheetah09 | rtx_a4000 | 4/4 | GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB | GPU0*: 7%<br>GPU1*: 8%<br>GPU2*: 96%<br>GPU3*: 96% | 18.8/32 | 6.2/29.3/500.0 | 4 (003, 023, 034–035) |
| <span style="color:gray">jaguar02 (yuxinchen)</span> | <span style="color:gray">a16</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.2/15.0 GB<br>GPU1*: 3.1/15.0 GB<br>GPU2*: 3.2/15.0 GB</span> | <span style="color:gray">GPU0*: 97%<br>GPU1*: 97%<br>GPU2*: 27%</span> | <span style="color:gray">8.2/24</span> | <span style="color:gray">4.6/17.6/1000.0</span> | <span style="color:gray">3 (084, 101, 112)</span> |
| lotus | rtx_6000 | 8/8 | GPU0*: 3.4/24.0 GB<br>GPU1*: 3.4/24.0 GB<br>GPU2*: 3.4/24.0 GB<br>GPU3*: 3.4/24.0 GB<br>GPU4*: 3.4/24.0 GB<br>GPU5*: 3.4/24.0 GB<br>GPU6*: 3.4/24.0 GB<br>GPU7*: 3.4/24.0 GB | GPU0*: 95%<br>GPU1*: 7%<br>GPU2*: 8%<br>GPU3*: 6%<br>GPU4*: 94%<br>GPU5*: 91%<br>GPU6*: 12%<br>GPU7*: 41% | 36.4/64 | 12.1/58.6/250.0 | 8 (002, 008, 032, 039, 054, 072, 074–075) |
| <span style="color:gray">affogato11 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 26%</span> | <span style="color:gray">5.2/8</span> | <span style="color:gray">2.0/5.9/125.0</span> | <span style="color:gray">1 (109)</span> |
| <span style="color:gray">ai01 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 43%<br>GPU2*: 8%</span> | <span style="color:gray">14.1/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (040, 047, 069)</span> |
| <span style="color:gray">ai02 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 6%<br>GPU2*: 6%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (037, 045, 094)</span> |
| <span style="color:gray">ai03 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 4.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 14%<br>GPU2*: 25%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (059, 076, 081)</span> |
| <span style="color:gray">ai04 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 28%<br>GPU1*: 7%<br>GPU2*: 51%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (083, 088, 097)</span> |
| ai06 (owner, yuxinchen) | rtx_2080_ti | 1/1 | GPU0*: 3.4/11.0 GB | GPU0*: 30% | 5.0/16 | 2.1/5.9/62.5 | 1 (092) |
| cheetah03 | rtx_2080_ti | 2/2 | GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB | GPU0*: 57%<br>GPU1*: 17% | 9.4/16 | 3.0/14.6/1000.0 | 2 (055, 064) |
| <span style="color:gray">lynx10 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 5%<br>GPU2*: 6%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">4.7/17.6/62.5</span> | <span style="color:gray">3 (113, 122, 124)</span> |
| <span style="color:gray">adriatic01 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.1/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.1/8.0 GB</span> | <span style="color:gray">GPU0*: 96%<br>GPU1*: 94%<br>GPU2*: 98%</span> | <span style="color:gray">13.1/24</span> | <span style="color:gray">4.6/17.6/1000.0</span> | <span style="color:gray">3 (090, 114, 123)</span> |
| adriatic02 | quadro_rtx_4000 | 3/3 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB | GPU0*: 19%<br>GPU1*: 98%<br>GPU2*: 94% | 12.7/24 | 4.6/17.6/1000.0 | 3 (027, 033, 050) |
| adriatic03 | quadro_rtx_4000 | 3/3 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.2/8.0 GB | GPU0*: 97%<br>GPU1*: 98%<br>GPU2*: 94% | 12.8/24 | 4.6/17.6/1000.0 | 3 (051, 060, 062) |
| adriatic04 (owner, yuxinchen) | quadro_rtx_4000 | 2/2 | GPU0*: 4.4/8.0 GB<br>GPU1*: 3.2/8.0 GB | GPU0*: 14%<br>GPU1*: 96% | 8.8/24 | 3.1/11.7/1000.0 | 2 (042, 046) |
| adriatic05 (owner, yuxinchen) | quadro_rtx_4000 | 2/2 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 8%<br>GPU1*: 10% | 8.6/24 | 3.1/11.7/1000.0 | 2 (100, 106) |
| <span style="color:gray">adriatic06 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 13%<br>GPU1*: 97%<br>GPU2*: 12%</span> | <span style="color:gray">12.5/24</span> | <span style="color:gray">5.1/17.6/1000.0</span> | <span style="color:gray">3 (103, 108, 116)</span> |
| <span style="color:gray">jaguar05 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.4/8.0 GB</span> | <span style="color:gray">GPU0*: 17%</span> | <span style="color:gray">4.6/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (089)</span> |
| <span style="color:gray">lynx01 (yuxinchen)</span> | <span style="color:gray">titan_xp</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/12.0 GB</span> | <span style="color:gray">GPU0*: 7%</span> | <span style="color:gray">4.9/8</span> | <span style="color:gray">2.0/5.9/62.5</span> | <span style="color:gray">1 (077)</span> |
| ai07 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 96%<br>GPU1*: 7%<br>GPU2*: 7% | 13.3/24 | 4.2/17.6/125.0 | 3 (009, 049, 063) |
| ai08 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 6%<br>GPU1*: 7%<br>GPU2*: 98% | 13.5/24 | 4.2/17.6/125.0 | 3 (053, 065, 070) |
| <span style="color:gray">ai09 (yuxinchen)</span> | <span style="color:gray">gtx_1080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 6%<br>GPU2*: 7%</span> | <span style="color:gray">13.7/24</span> | <span style="color:gray">4.2/17.6/109.4</span> | <span style="color:gray">3 (091, 096, 119)</span> |
| <span style="color:gray">ai05 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 9%<br>GPU1*: 8%<br>GPU2*: 8%</span> | <span style="color:gray">12.2/26</span> | <span style="color:gray">4.4/17.6/125.0</span> | <span style="color:gray">3 (041, 105, 107)</span> |
| <span style="color:gray">ai10 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 98%<br>GPU1*: 97%<br>GPU2*: 13%</span> | <span style="color:gray">12.1/24</span> | <span style="color:gray">4.8/17.6/125.0</span> | <span style="color:gray">3 (029, 036, 115)</span> |
| jinx01 | gtx_1080 | 2/2 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 97%<br>GPU1*: 81% | 7.6/16 | 2.8/14.6/214.8 | 2 (004, 068) |
| **TOTAL (owner)** | — | **41/41** | **190.9/610.9 GB** | **mean 61% (all listed GPUs)** | **238.6/480** | **87.6/392.6/7277.3** | **57** |
| <span style="color:gray">**TOTAL (collaborators)**</span> | <span style="color:gray">—</span> | <span style="color:gray">**45/45**</span> | <span style="color:gray">**151.0/558.9 GB**</span> | <span style="color:gray">**mean 31% (all listed GPUs)**</span> | <span style="color:gray">**200.1/362**</span> | <span style="color:gray">**73.0/263.7/5732.4**</span> | <span style="color:gray">**45**</span> |
| **TOTAL** | — | **86/86** | **341.9/1169.8 GB** | **mean 45% (all listed GPUs)** | **438.7/842** | **160.6/656.2/13009.8** | **102** |


## 2026-08-06, eighth tick — a checkpoint gate that would have fired on every walltime kill

106 runs executing, 45 pending, none failed. Mean 17.9M steps per run, 0.62% of the 150-run budget,
aggregate 320,805 steps/s. Zero invariant violations, zero non-finite gradient norms. Eighteen runs
have now scored above zero on the extrinsic mean, the best at **1,175**.

**The collaborator reported a slow node and it uncovered a trainer defect.** cheetah01's A100
delivered 1,349 steps/s against a same-arm median of 2,955 — 46%, and below every GTX 1080 and P100
in the set — while holding its full 8 threads at 665% CPU with the GPU at 2% utilisation. Not the
allocation, not the arm. Its submission script is parked in
`slurm/submission_script/parked/`, which both launchers enforce by refusing a node with no script.
The campaign is capped on CPU threads rather than GPUs, so a node returning half the steps per thread
costs the campaign directly.

**The defect underneath it is the important part.** Run 79 ran 79 minutes on that node and left **no
checkpoint**, so its requeue restarts from zero. The trainer's log shows it caught the signal and was
killed before it could write one. The checkpoint was gated on a logging update:

```python
due = time.time() - last_checkpoint_time >= args.checkpoint_every_seconds
if due and update % args.log_every_updates == 0:
```

so an hourly checkpoint could only be written at a multiple of 200 updates. On a fast node a logging
interval is about 16 minutes and the rounding is invisible; at 1,349 steps/s it is 40 minutes, so the
first checkpoint could not land before 80 minutes. The two problems compounded precisely — the slow
node doubled the checkpoint gap, and the run was moved off it at 79 minutes.

Left alone this would have fired on **every walltime kill of every slow run for the rest of the
campaign**, and every run spans several 4-day segments. Fixed: the checkpoint is written at any
iteration boundary once due. The alignment it was protecting is unnecessary — a format-2 checkpoint
carries its own history, so a resume rebuilds from it and truncates the record to that position at
whatever update it sits. 103 of the 106 runs past the one-hour mark already have checkpoints, so the
mechanism was sound and only its gate was wrong.

**Also fixed:** the cancelled job's log held 85 bytes — only slurmstepd's line. `worker_manager.py`'s
own output was block-buffered and lost exactly when it was needed for diagnosis. Every submission
script now runs it under `python3 -u`.

Both changes reach runs started from now on; runs already executing carry the old code until they are
restarted.
## 2026-08-06 04:07  (jobs: 33 running (owner 12, yuxinchen 21), 0 pending | runs: 105 running, 45 pending, 0 done, 0 failed)

| node | gpu type | gpus ours/node | gpu memory used/total (every gpu) | gpu usage (every gpu) | cpu busy/asked (threads) | sys memory used/asked/node (GB) | running runs |
|---|---|---|---|---|---|---|---|
| <span style="color:gray">jaguar06 (yuxinchen)</span> | <span style="color:gray">a40</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.3/45.0 GB</span> | <span style="color:gray">GPU0*: 6%</span> | <span style="color:gray">5.2/8</span> | <span style="color:gray">1.6/5.9/123.0</span> | <span style="color:gray">1 (048)</span> |
| <span style="color:gray">cheetah02 (yuxinchen)</span> | <span style="color:gray">rtx_4000_ada</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.3/20.0 GB<br>GPU1*: 3.3/20.0 GB<br>GPU2*: 3.3/20.0 GB<br>GPU3*: 3.3/20.0 GB</span> | <span style="color:gray">GPU0*: 5%<br>GPU1*: 5%<br>GPU2*: 35%<br>GPU3*: 94%</span> | <span style="color:gray">18.6/32</span> | <span style="color:gray">6.3/23.4/1000.0</span> | <span style="color:gray">4 (066–067, 085–086)</span> |
| jaguar03 | rtx_a4500 | 8/8 | GPU0*: 10.7/20.0 GB<br>GPU1*: 10.0/20.0 GB<br>GPU2*: 10.1/20.0 GB<br>GPU3*: 9.7/20.0 GB<br>GPU4*: 10.0/20.0 GB<br>GPU5*: 9.9/20.0 GB<br>GPU6*: 10.0/20.0 GB<br>GPU7*: 10.0/20.0 GB | GPU0*: 97%<br>GPU1*: 99%<br>GPU2*: 25%<br>GPU3*: 100%<br>GPU4*: 98%<br>GPU5*: 33%<br>GPU6*: 19%<br>GPU7*: 99% | 88.9/192 | 37.8/175.8/1000.0 | 24 (001, 005, 007, 010–022, 024–026, 028, 030–031, 038, 044) |
| <span style="color:gray">cheetah08 (yuxinchen)</span> | <span style="color:gray">rtx_a4000</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB</span> | <span style="color:gray">GPU0*: 94%<br>GPU1*: 95%<br>GPU2*: 6%<br>GPU3*: 95%</span> | <span style="color:gray">18.9/32</span> | <span style="color:gray">6.3/23.4/500.0</span> | <span style="color:gray">4 (058, 061, 073, 078)</span> |
| cheetah09 | rtx_a4000 | 4/4 | GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB | GPU0*: 10%<br>GPU1*: 90%<br>GPU2*: 96%<br>GPU3*: 94% | 18.7/32 | 6.2/29.3/500.0 | 4 (003, 023, 034–035) |
| <span style="color:gray">jaguar02 (yuxinchen)</span> | <span style="color:gray">a16</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.2/15.0 GB<br>GPU1*: 3.1/15.0 GB<br>GPU2*: 3.2/15.0 GB</span> | <span style="color:gray">GPU0*: 25%<br>GPU1*: 96%<br>GPU2*: 97%</span> | <span style="color:gray">8.2/24</span> | <span style="color:gray">4.6/17.6/1000.0</span> | <span style="color:gray">3 (084, 101, 112)</span> |
| lotus | rtx_6000 | 8/8 | GPU0*: 3.4/24.0 GB<br>GPU1*: 3.4/24.0 GB<br>GPU2*: 3.4/24.0 GB<br>GPU3*: 3.4/24.0 GB<br>GPU4*: 3.4/24.0 GB<br>GPU5*: 3.4/24.0 GB<br>GPU6*: 3.4/24.0 GB<br>GPU7*: 3.4/24.0 GB | GPU0*: 6%<br>GPU1*: 97%<br>GPU2*: 94%<br>GPU3*: 8%<br>GPU4*: 9%<br>GPU5*: 8%<br>GPU6*: 9%<br>GPU7*: 7% | 36.6/64 | 12.1/58.6/250.0 | 8 (002, 008, 032, 039, 054, 072, 074–075) |
| <span style="color:gray">affogato11 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 11%</span> | <span style="color:gray">5.2/8</span> | <span style="color:gray">2.0/5.9/125.0</span> | <span style="color:gray">1 (109)</span> |
| <span style="color:gray">ai01 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 43%<br>GPU2*: 5%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (040, 047, 069)</span> |
| <span style="color:gray">ai02 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 93%<br>GPU1*: 6%<br>GPU2*: 6%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (037, 045, 094)</span> |
| <span style="color:gray">ai03 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 4.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 89%<br>GPU2*: 5%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (059, 076, 081)</span> |
| <span style="color:gray">ai04 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 8%<br>GPU2*: 6%</span> | <span style="color:gray">14.1/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (083, 088, 097)</span> |
| ai06 (owner, yuxinchen) | rtx_2080_ti | 1/1 | GPU0*: 3.4/11.0 GB | GPU0*: 7% | 5.0/16 | 1.5/7.3/62.5 | 1 (098) |
| cheetah03 | rtx_2080_ti | 2/2 | GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB | GPU0*: 69%<br>GPU1*: 60% | 9.1/16 | 3.0/14.6/1000.0 | 2 (055, 064) |
| <span style="color:gray">lynx10 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 8%<br>GPU1*: 57%<br>GPU2*: 31%</span> | <span style="color:gray">14.2/24</span> | <span style="color:gray">4.7/17.6/62.5</span> | <span style="color:gray">3 (113, 122, 124)</span> |
| <span style="color:gray">adriatic01 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.1/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.1/8.0 GB</span> | <span style="color:gray">GPU0*: 10%<br>GPU1*: 10%<br>GPU2*: 7%</span> | <span style="color:gray">12.7/24</span> | <span style="color:gray">4.6/17.6/1000.0</span> | <span style="color:gray">3 (090, 114, 123)</span> |
| adriatic02 | quadro_rtx_4000 | 3/3 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB | GPU0*: 9%<br>GPU1*: 98%<br>GPU2*: 12% | 12.5/24 | 4.6/17.6/1000.0 | 3 (027, 033, 050) |
| adriatic03 | quadro_rtx_4000 | 3/3 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.2/8.0 GB | GPU0*: 96%<br>GPU1*: 95%<br>GPU2*: 9% | 13.0/24 | 4.6/17.6/1000.0 | 3 (051, 060, 062) |
| adriatic04 (owner, yuxinchen) | quadro_rtx_4000 | 2/2 | GPU0*: 4.4/8.0 GB<br>GPU1*: 3.2/8.0 GB | GPU0*: 8%<br>GPU1*: 8% | 8.6/24 | 3.1/11.7/1000.0 | 2 (042, 046) |
| adriatic05 (owner, yuxinchen) | quadro_rtx_4000 | 2/2 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 10%<br>GPU1*: 12% | 8.6/24 | 3.1/11.7/1000.0 | 2 (100, 106) |
| <span style="color:gray">adriatic06 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 12%<br>GPU1*: 7%<br>GPU2*: 59%</span> | <span style="color:gray">12.9/24</span> | <span style="color:gray">5.1/17.6/1000.0</span> | <span style="color:gray">3 (103, 108, 116)</span> |
| <span style="color:gray">jaguar05 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.4/8.0 GB</span> | <span style="color:gray">GPU0*: 11%</span> | <span style="color:gray">4.7/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (089)</span> |
| <span style="color:gray">lynx01 (yuxinchen)</span> | <span style="color:gray">titan_xp</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/12.0 GB</span> | <span style="color:gray">GPU0*: 92%</span> | <span style="color:gray">4.9/8</span> | <span style="color:gray">2.0/5.9/62.5</span> | <span style="color:gray">1 (077)</span> |
| ai07 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 97%<br>GPU1*: 7%<br>GPU2*: 9% | 13.4/24 | 4.2/17.6/125.0 | 3 (009, 049, 063) |
| ai08 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 8%<br>GPU1*: 95%<br>GPU2*: 8% | 13.6/24 | 4.2/17.6/125.0 | 3 (053, 065, 070) |
| <span style="color:gray">ai09 (yuxinchen)</span> | <span style="color:gray">gtx_1080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 95%<br>GPU2*: 9%</span> | <span style="color:gray">13.8/24</span> | <span style="color:gray">4.2/17.6/109.4</span> | <span style="color:gray">3 (091, 096, 119)</span> |
| <span style="color:gray">ai05 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 97%<br>GPU1*: 11%<br>GPU2*: 10%</span> | <span style="color:gray">12.0/26</span> | <span style="color:gray">4.4/17.6/125.0</span> | <span style="color:gray">3 (041, 105, 107)</span> |
| <span style="color:gray">ai10 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 10%<br>GPU1*: 97%<br>GPU2*: 96%</span> | <span style="color:gray">12.0/24</span> | <span style="color:gray">4.8/17.6/125.0</span> | <span style="color:gray">3 (029, 036, 115)</span> |
| jinx01 | gtx_1080 | 2/2 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 8%<br>GPU1*: 17% | 7.5/16 | 2.8/14.6/214.8 | 2 (004, 068) |
| **TOTAL (owner)** | — | **41/41** | **190.8/610.9 GB** | **mean 45% (all listed GPUs)** | **235.5/480** | **87.1/394.0/7277.3** | **57** |
| <span style="color:gray">**TOTAL (collaborators)**</span> | <span style="color:gray">—</span> | <span style="color:gray">**45/45**</span> | <span style="color:gray">**151.0/558.9 GB**</span> | <span style="color:gray">**mean 37% (all listed GPUs)**</span> | <span style="color:gray">**200.1/362**</span> | <span style="color:gray">**73.0/263.7/5732.4**</span> | <span style="color:gray">**45**</span> |
| **TOTAL** | — | **86/86** | **341.9/1169.8 GB** | **mean 41% (all listed GPUs)** | **435.7/842** | **160.1/657.7/13009.8** | **102** |


## 2026-08-06, ninth tick — quiet; the clip's effect on the predictor has settled

105 runs executing, 45 pending, none failed, none orphaned, no open problem reports. 33 jobs
(owner 12, collaborator 21). Mean 24,997,303 steps per run — 1.25% of each run's target, 0.87% of the
150-run budget. Aggregate 323,654 steps/s. Zero invariant violations, zero non-finite gradient norms.
104 of 105 records have a checkpoint; the one without is run 79, restarted from zero after yesterday's
requeue.

**The quantity the ablation exists to measure has stopped moving.** Mean scale applied to the
predictor's gradient, at three matched steps:

| step | arm 1 | arm 3 | arm 4 | arms 2 and 5 |
|---|---|---|---|---|
| 3,276,800 | 0.859 | 0.872 | 0.878 | 1.000 |
| 9,830,400 | 0.409 | 0.358 | 0.439 | 1.000 |
| 19,660,800 | 0.382 | 0.381 | 0.389 | 1.000 |

So in the three joint-clip arms the RND predictor now trains at **about 38% of the gradient it would
otherwise receive**, and that figure has settled rather than continuing to fall. The predictor still
contributes well under 1% of the joint squared norm, so this is entirely the policy's norm deciding
how fast the predictor learns. That is the mechanism arms 2 and 5 remove, and it is now a large,
stable, well-measured effect rather than the 1.4% it looked like at 3.3M steps.

Extrinsic reward at 19,660,800 steps, with about 20 seeds per arm: arm 4 mean 80.8 (4 of 21 seeds
have scored), arm 5 44.2 (4 of 20), arm 2 18.0 (4 of 21), arm 3 3.8 (1 of 20), arm 1 0.0 (0 of 20).
**These means are each carried by one or two seeds** and the runs are at 1% of their budget, so the
ordering is not yet evidence. The one line worth remembering for later is that arm 1, CleanRL as
published, is so far the only arm with no seed scoring at that step.

Best single run so far: 1,365 on the 200-episode mean.

No top-up: `gpu` at 384 of 400 CPU threads, `gnolim` at 64 of 80, both at the owner's headroom.
## 2026-08-06 04:27  (jobs: 33 running (owner 12, yuxinchen 21), 0 pending | runs: 105 running, 45 pending, 0 done, 0 failed)

| node | gpu type | gpus ours/node | gpu memory used/total (every gpu) | gpu usage (every gpu) | cpu busy/asked (threads) | sys memory used/asked/node (GB) | running runs |
|---|---|---|---|---|---|---|---|
| <span style="color:gray">jaguar06 (yuxinchen)</span> | <span style="color:gray">a40</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.3/45.0 GB</span> | <span style="color:gray">GPU0*: 6%</span> | <span style="color:gray">5.3/8</span> | <span style="color:gray">1.6/5.9/123.0</span> | <span style="color:gray">1 (048)</span> |
| <span style="color:gray">cheetah02 (yuxinchen)</span> | <span style="color:gray">rtx_4000_ada</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.3/20.0 GB<br>GPU1*: 3.3/20.0 GB<br>GPU2*: 3.3/20.0 GB<br>GPU3*: 3.3/20.0 GB</span> | <span style="color:gray">GPU0*: 5%<br>GPU1*: 92%<br>GPU2*: 5%<br>GPU3*: 4%</span> | <span style="color:gray">18.6/32</span> | <span style="color:gray">6.3/23.4/1000.0</span> | <span style="color:gray">4 (066–067, 085–086)</span> |
| jaguar03 | rtx_a4500 | 8/8 | GPU0*: 10.7/20.0 GB<br>GPU1*: 10.0/20.0 GB<br>GPU2*: 10.1/20.0 GB<br>GPU3*: 9.7/20.0 GB<br>GPU4*: 10.0/20.0 GB<br>GPU5*: 9.9/20.0 GB<br>GPU6*: 10.0/20.0 GB<br>GPU7*: 10.0/20.0 GB | GPU0*: 23%<br>GPU1*: 20%<br>GPU2*: 23%<br>GPU3*: 95%<br>GPU4*: 25%<br>GPU5*: 17%<br>GPU6*: 100%<br>GPU7*: 21% | 92.6/192 | 37.8/175.8/1000.0 | 24 (001, 005, 007, 010–022, 024–026, 028, 030–031, 038, 044) |
| <span style="color:gray">cheetah08 (yuxinchen)</span> | <span style="color:gray">rtx_a4000</span> | <span style="color:gray">4/4</span> | <span style="color:gray">GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 7%<br>GPU2*: 96%<br>GPU3*: 8%</span> | <span style="color:gray">18.9/32</span> | <span style="color:gray">6.3/23.4/500.0</span> | <span style="color:gray">4 (058, 061, 073, 078)</span> |
| cheetah09 | rtx_a4000 | 4/4 | GPU0*: 3.2/16.0 GB<br>GPU1*: 3.4/16.0 GB<br>GPU2*: 3.2/16.0 GB<br>GPU3*: 3.2/16.0 GB | GPU0*: 8%<br>GPU1*: 9%<br>GPU2*: 5%<br>GPU3*: 96% | 18.5/32 | 6.2/29.3/500.0 | 4 (003, 023, 034–035) |
| <span style="color:gray">jaguar02 (yuxinchen)</span> | <span style="color:gray">a16</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.2/15.0 GB<br>GPU1*: 3.1/15.0 GB<br>GPU2*: 3.2/15.0 GB</span> | <span style="color:gray">GPU0*: 20%<br>GPU1*: 97%<br>GPU2*: 22%</span> | <span style="color:gray">8.1/24</span> | <span style="color:gray">4.6/17.6/1000.0</span> | <span style="color:gray">3 (084, 101, 112)</span> |
| lotus | rtx_6000 | 8/8 | GPU0*: 3.4/24.0 GB<br>GPU1*: 3.4/24.0 GB<br>GPU2*: 3.4/24.0 GB<br>GPU3*: 3.4/24.0 GB<br>GPU4*: 3.4/24.0 GB<br>GPU5*: 3.4/24.0 GB<br>GPU6*: 3.4/24.0 GB<br>GPU7*: 3.4/24.0 GB | GPU0*: 9%<br>GPU1*: 90%<br>GPU2*: 9%<br>GPU3*: 9%<br>GPU4*: 45%<br>GPU5*: 75%<br>GPU6*: 9%<br>GPU7*: 95% | 36.3/64 | 12.1/58.6/250.0 | 8 (002, 008, 032, 039, 054, 072, 074–075) |
| <span style="color:gray">affogato11 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 59%</span> | <span style="color:gray">5.0/8</span> | <span style="color:gray">2.0/5.9/125.0</span> | <span style="color:gray">1 (109)</span> |
| <span style="color:gray">ai01 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 43%<br>GPU1*: 5%<br>GPU2*: 6%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (040, 047, 069)</span> |
| <span style="color:gray">ai02 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 7%<br>GPU1*: 6%<br>GPU2*: 90%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (037, 045, 094)</span> |
| <span style="color:gray">ai03 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 4.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 18%<br>GPU1*: 6%<br>GPU2*: 63%</span> | <span style="color:gray">14.1/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (059, 076, 081)</span> |
| <span style="color:gray">ai04 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 81%<br>GPU1*: 36%<br>GPU2*: 5%</span> | <span style="color:gray">14.3/24</span> | <span style="color:gray">5.1/17.6/62.5</span> | <span style="color:gray">3 (083, 088, 097)</span> |
| ai06 (owner, yuxinchen) | rtx_2080_ti | 1/1 | GPU0*: 3.4/11.0 GB | GPU0*: 6% | 5.0/16 | 1.5/7.3/62.5 | 1 (098) |
| cheetah03 | rtx_2080_ti | 2/2 | GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB | GPU0*: 8%<br>GPU1*: 7% | 9.0/16 | 3.0/14.6/1000.0 | 2 (055, 064) |
| <span style="color:gray">lynx10 (yuxinchen)</span> | <span style="color:gray">rtx_2080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.4/11.0 GB<br>GPU1*: 3.4/11.0 GB<br>GPU2*: 3.4/11.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 7%<br>GPU2*: 6%</span> | <span style="color:gray">14.4/24</span> | <span style="color:gray">4.7/17.6/62.5</span> | <span style="color:gray">3 (113, 122, 124)</span> |
| <span style="color:gray">adriatic01 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.1/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.1/8.0 GB</span> | <span style="color:gray">GPU0*: 9%<br>GPU1*: 6%<br>GPU2*: 9%</span> | <span style="color:gray">12.7/24</span> | <span style="color:gray">4.6/17.6/1000.0</span> | <span style="color:gray">3 (090, 114, 123)</span> |
| adriatic02 | quadro_rtx_4000 | 3/3 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB | GPU0*: 98%<br>GPU1*: 96%<br>GPU2*: 11% | 12.7/24 | 4.6/17.6/1000.0 | 3 (027, 033, 050) |
| adriatic03 | quadro_rtx_4000 | 3/3 | GPU0*: 3.2/8.0 GB<br>GPU1*: 3.2/8.0 GB<br>GPU2*: 3.2/8.0 GB | GPU0*: 8%<br>GPU1*: 9%<br>GPU2*: 95% | 12.8/24 | 4.6/17.6/1000.0 | 3 (051, 060, 062) |
| adriatic04 (owner, yuxinchen) | quadro_rtx_4000 | 2/2 | GPU0*: 4.4/8.0 GB<br>GPU1*: 3.2/8.0 GB | GPU0*: 98%<br>GPU1*: 14% | 8.3/24 | 3.1/11.7/1000.0 | 2 (042, 046) |
| adriatic05 (owner, yuxinchen) | quadro_rtx_4000 | 1/1 | GPU0*: 3.3/8.0 GB | GPU0*: 97% | 4.6/24 | 1.5/5.9/1000.0 | 1 (071) |
| <span style="color:gray">adriatic06 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 18%<br>GPU1*: 98%<br>GPU2*: 97%</span> | <span style="color:gray">12.6/24</span> | <span style="color:gray">5.1/17.6/1000.0</span> | <span style="color:gray">3 (103, 108, 116)</span> |
| <span style="color:gray">jaguar05 (yuxinchen)</span> | <span style="color:gray">quadro_rtx_4000</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 4.4/8.0 GB</span> | <span style="color:gray">GPU0*: 95%</span> | <span style="color:gray">4.6/8</span> | <span style="color:gray">2.1/5.9/250.0</span> | <span style="color:gray">1 (089)</span> |
| <span style="color:gray">lynx01 (yuxinchen)</span> | <span style="color:gray">titan_xp</span> | <span style="color:gray">1/1</span> | <span style="color:gray">GPU0*: 3.4/12.0 GB</span> | <span style="color:gray">GPU0*: 7%</span> | <span style="color:gray">4.8/8</span> | <span style="color:gray">2.0/5.9/62.5</span> | <span style="color:gray">1 (077)</span> |
| ai07 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 7%<br>GPU1*: 7%<br>GPU2*: 6% | 13.7/24 | 4.2/17.6/125.0 | 3 (009, 049, 063) |
| ai08 | gtx_1080_ti | 3/3 | GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB | GPU0*: 9%<br>GPU1*: 95%<br>GPU2*: 95% | 13.4/24 | 4.2/17.6/125.0 | 3 (053, 065, 070) |
| <span style="color:gray">ai09 (yuxinchen)</span> | <span style="color:gray">gtx_1080_ti</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/11.0 GB<br>GPU1*: 3.3/11.0 GB<br>GPU2*: 3.3/11.0 GB</span> | <span style="color:gray">GPU0*: 6%<br>GPU1*: 96%<br>GPU2*: 9%</span> | <span style="color:gray">13.6/24</span> | <span style="color:gray">4.2/17.6/109.4</span> | <span style="color:gray">3 (091, 096, 119)</span> |
| <span style="color:gray">ai05 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 12%<br>GPU1*: 11%<br>GPU2*: 97%</span> | <span style="color:gray">12.2/26</span> | <span style="color:gray">4.4/17.6/125.0</span> | <span style="color:gray">3 (041, 105, 107)</span> |
| <span style="color:gray">ai10 (yuxinchen)</span> | <span style="color:gray">gtx_1080</span> | <span style="color:gray">3/3</span> | <span style="color:gray">GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB<br>GPU2*: 3.3/8.0 GB</span> | <span style="color:gray">GPU0*: 63%<br>GPU1*: 10%<br>GPU2*: 11%</span> | <span style="color:gray">12.3/24</span> | <span style="color:gray">4.8/17.6/125.0</span> | <span style="color:gray">3 (029, 036, 115)</span> |
| jinx01 | gtx_1080 | 2/2 | GPU0*: 3.3/8.0 GB<br>GPU1*: 3.3/8.0 GB | GPU0*: 33%<br>GPU1*: 98% | 7.6/16 | 2.8/14.6/214.8 | 2 (004, 068) |
| **TOTAL (owner)** | — | **40/40** | **187.7/602.9 GB** | **mean 42% (all listed GPUs)** | **234.6/480** | **85.6/388.2/7277.3** | **56** |
| <span style="color:gray">**TOTAL (collaborators)**</span> | <span style="color:gray">—</span> | <span style="color:gray">**45/45**</span> | <span style="color:gray">**151.0/558.9 GB**</span> | <span style="color:gray">**mean 33% (all listed GPUs)**</span> | <span style="color:gray">**200.1/362**</span> | <span style="color:gray">**73.0/263.7/5732.4**</span> | <span style="color:gray">**45**</span> |
| **TOTAL** | — | **85/85** | **338.7/1161.8 GB** | **mean 37% (all listed GPUs)** | **434.7/842** | **158.6/651.9/13009.8** | **101** |


## 2026-08-06, tenth tick — past 1% of the campaign; a first hint on the clip

105 runs executing, 45 pending, none failed, none orphaned, no open problem reports. Mean 28,711,010
steps per run — 1.44% of each run's target, **1.00% of the 150-run budget**. Aggregate 324,328
steps/s. Zero invariant violations, zero non-finite gradient norms. 104 of 105 records checkpointed.

Best single run: **3,502** on the 200-episode mean (run 39, `arm5_all`, seed 8). The top three are all
`arm5_all`.

**A first hint that the predictor's clip matters, and how much weight it can bear.** Comparing runs
only at a step they have all reached, so a faster node cannot look like a better arm:

| at 26,214,400 steps | runs reaching it | scored | rate | mean | median |
|---|---|---|---|---|---|
| arm1_original | 14 | 2 | 14% | 0.3 | 0.0 |
| arm2_no_rnd_grad_clip | 14 | 7 | **50%** | 79.1 | 3.0 |
| arm3_update_proportion_1 | 12 | 2 | 17% | 1.0 | 0.0 |
| arm4_shallower_predictor | 15 | 3 | 20% | 87.1 | 0.0 |
| arm5_all | 12 | 6 | **50%** | 147.9 | 0.2 |
| **predictor clipped (1, 3, 4)** | 41 | 7 | **17%** | 32.3 | — |
| **predictor unclipped (2, 5)** | 26 | 13 | **50%** | 110.9 | — |

The same split at 19,660,800 steps reads 13% against 26%, so the gap is in the same direction and
widening.

**What this is not.** Every run is at 1.4% of its 2e9 budget; the median is 0.0 in every arm, so the
means are carried by a handful of seeds; and this is a count of "has scored at all", which on a
sparse-reward game is a coarse proxy for exploration. It is consistent with the mechanism the
gradient statistics have already established — the joint clip scales the predictor's gradient to
about 38%, so arms 2 and 5 give the predictor roughly 2.6 times the learning signal — but a
difference in first-reward timing at 1% of the budget is not a difference in final performance.

The comparison that decides anything is at a matched step with all 30 seeds, and that is far away.
Recording it now because the direction is worth watching, not because it settles anything.

No top-up: `gpu` at 384 of 400 CPU threads, `gnolim` at 64 of 80.
