# Actual-training throughput recheck after the 2026-07-31 jaguar03 reboot: FIXED

**Date:** 2026-07-31 23:27–23:43. **Jobs:** 6528732 (jaguar03, exclusive, reservation `sl5nw_151`),
6528733 (adriatic06 healthy control, exclusive). Ids recorded in
`../../slurm/submitted_jobids_jaguar03_probe.txt`.

## Method

Identical to the 2026-07-23 light-load reproduction (`../run_experiment.py`, layout `L8`):

1. 8 copies of the real run-5 original-small training workload (`train.py`, `rnd_next_state`,
   beta 1000, `mse_mean` readout, lr 1e-4, reward normalization, fast `space_sample` obs warmup),
   each pinned to its own physical core with 1 thread, node otherwise idle (exclusive allocation).
2. Frequent logging so the number is measurable inside a 20-minute job: each run flushes its step
   count every 50 env steps (`--eval_freq=50`), and the driver polls every 60 s into
   `<node>/data/L8/progress_L8.csv`.
3. 15-minute measured window. Define per-run throughput $v$ (steps/min) as
   $v = 60\,\hat\beta$, where $\hat\beta$ is the least-squares slope of the (elapsed seconds,
   step) pairs with elapsed at least 240 s — the launch/import ramp is excluded, same as the
   original report's setup-excluded slope. Reported per node as the median of $v$ over the 8 runs.
4. Before training, the single-thread random-access memory-latency probe (`../latency.c`, 256 MB
   buffer) — the root-cause metric from the 2026-07-23 correction.

## Results

| node, date | per-run training steps/min (median of 8) | memory latency ns per access |
|---|---|---|
| jaguar03, 2026-07-31 (after reboot) | **1,731** | **107** |
| adriatic06, 2026-07-31 (healthy control, same day) | <u>1,382</u> | <u>111</u> |
| jaguar03, 2026-07-23/24 (sick) | 168 | 413–425 |
| adriatic02, 2026-07-27 (healthy reference from the original repro) | 1,301 | not measured |

Higher is better for steps/min, lower for latency; best in bold, second best underlined.
Per-run spread was tight on both nodes (jaguar03 1,693–1,735; adriatic06 1,363–1,384).

## Findings

1. **jaguar03 is fixed for the actual training workload.** 1,731 steps/min per run is about 10x its
   sick value (168) and about 25% faster than the same-day healthy control (1,382) — reasonable,
   since its EPYC 7663 cores are newer than the control's Xeon Silver 4208.
2. **The root cause (memory latency) is back in the healthy range**: 107 ns per random access,
   versus 413–425 ns when sick and 99–138 ns on healthy nodes. The 2026-07-24 reboot had NOT fixed
   this (413 ns after it); the 2026-07-31 reboot did.
3. The `/proc/cpuinfo` frequency counter still reads oddly (median 1500 MHz on jaguar03 during the
   run vs 2100 on adriatic06) — as established before, that counter is unreliable here; throughput
   is the ground truth.
