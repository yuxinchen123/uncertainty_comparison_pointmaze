# jaguar03 CPU-threading experiment — which layout to fill the node with?

**Question.** jaguar03 runs RND training far slower than every other node. When we fill it (192 single-thread tasks, 2 per core), it finished **zero** of its runs in ~30 h while other nodes finish a run in ~15 h. This experiment measures which thread-to-core layout gives the most training throughput when jaguar03 is fully loaded, and records why the node is slow in the first place.

## Background — the measured slowness (before this experiment)

- Historical per-node finish times: every non-jaguar03 node finishes a 1M-step run in ~15–21 h; **jaguar03 finished 0 runs in ~30 h**, its in-flight runs reaching only ~12 % of 1M steps.
- Controlled single-core fixed-work benchmark under load: **jaguar03 400 MHz, compute 34.7 s** vs puma01 2300 MHz, compute 1.33 s (26× slower); memory only 1.6× slower — so the bottleneck is the **CPU clock**, not memory bandwidth.
- Hardware: 2× AMD EPYC 7663 (112 cores / 224 threads). cpufreq shows `scaling_min=1.5 GHz, scaling_max=2.0 GHz`, yet under full all-core load the actual clock is ~400 MHz (below the OS minimum), with only 31/224 threads above 1 GHz. At 32-core load the same node runs at 1500 MHz; only at full load does it collapse to 400 MHz.

## Experiment setup

- **Node fully loaded** (this matters — the 400 MHz throttle only appears at full all-core load, which is how the node is used in production). The OS reserves 4 of the node's 112 cores, so all three layouts load the **108 usable physical cores / 216 hardware threads** (jaguar03 topology: physical core *K* owns hardware threads *{K, K+112}*), so the node throttles identically for all three and the **only** difference is the thread-to-run mapping:
  - **A** — 108 runs, 2 threads/run (OMP=2), 1 run per physical core.
  - **B** — 54 runs, 4 threads/run (OMP=4), 1 run per 2 physical cores.
  - **C** — 216 runs, 1 thread/run (OMP=1), 2 runs share a physical core. *(This is what the real sweep used.)*
- **Layouts run strictly sequentially** (A → B → C); each is fully killed and reaped before the next launches, so their loads never overlap.
- **Workload**: the real run-5 "original-small" arm (`rnd_next_state`, β=1000, `mse_mean` readout, lr 1e-4, LeakyReLU(0.2), predictor +1 layer, reward normalization), with fast `space_sample` obs warmup so the measured window is training, not warmup.
- **Logging**: each run flushes its step count every 50 env steps; the driver also polls every 60 s into `data/<layout>/progress_<layout>.csv` (elapsed_s, run_id, step, runtime_s, median_MHz).
- **Metric**: per-run throughput = steps/min from the steady-state slope of (step vs runtime, excluding the setup ramp); reported as **median per-run steps/min**, **total node steps/min** (sum over the layout's runs — the number that decides how fast the whole node clears work), and **per-core steps/min**.

## Results

All three layouts ran at the full-load 400 MHz throttle (confirmed by the `median MHz` column), so the comparison isolates the thread-to-run mapping. Per-run throughput is the steady-state slope of step-vs-runtime (setup excluded); total-node is the sum over the layout's runs (the metric that decides how fast the whole node clears work).

| layout | runs | threads/run | median per-run steps/min | **total-node steps/min** | per-core steps/min | median MHz |
|---|---|---|---|---|---|---|
| A | 108 | 2 | 159 | 17,242 | 160 | 400 |
| B | 54 | 4 | 211 | 11,286 | 105 | 400 |
| **C** | **216** | **1** | **82** | **17,801** | **165** | **400** |

- **C (1 thread/run, 2 runs per core) has the highest total-node throughput — 17,801 steps/min — but only about 3% above A (17,242).** This confirms that two independent single-thread runs on a core's two hardware threads slightly beat one two-thread run, but the margin is small.
- **B (4 threads/run) is much worse — about 37% below C.** This workload is single-core-bound, so giving each run four threads wastes cores; the extra threads add little per-run speed (211 vs A's 159 steps/min is far less than 2x) while halving the number of runs.
- **Per-run latency runs the other way.** Projected wall time for one 1M-step run at 400 MHz: B ~79 h (fewest runs, so each is fastest), A ~105 h, C ~203 h (each run gets only half a core). A gives nearly C's total throughput (within 3%) with each run finishing in half C's time.

## Recommendation

**To fill jaguar03 for maximum total throughput: layout C (216 runs, 1 thread each, 2 runs per core) — but layout A (108 runs, 2 threads each) is within 3% and finishes each run in half the time, so A is the better practical choice unless you specifically want to maximize completed-runs-per-hour and do not care about individual run latency.** Never use B — four threads per run wastes this single-core-bound workload's cores.

Two caveats that matter more than the layout choice:

1. **The layout barely matters; the throttle is the whole problem.** Switching from the worst layout (B) to the best (C) changes total throughput by ~1.6x, and C-vs-A is only ~3%. The 400 MHz throttle itself costs about **26x** versus a healthy node. No thread layout makes jaguar03 competitive: even the best case (C) needs ~203 h per 1M-step run, and the fastest-per-run case (B) still needs ~79 h — against roughly **15 h on a healthy node**. So the real fix is the hardware throttle (see "Is the jaguar03 throttling expected?"), not the layout.
2. **The real sweep already used layout C** (192 single-thread tasks, 2 per core), i.e. the highest-total-throughput layout. So the sweep's slowness was not a layout mistake — it was the node throttling to 400 MHz. Re-running the tail with a different layout on jaguar03 would gain at most ~3% and is not worth it; the tail was moved to healthy cpu-partition nodes instead.

These numbers are at the throttled 400 MHz. On a healthy jaguar03 (2.0 GHz base) all three would be roughly 5x higher, and the same ranking (C ≈ A > B) would hold.

## CORRECTION (2026-07-23): the real cause is MEMORY LATENCY, not a CPU throttle

Everything below this section was written under the belief that jaguar03 throttles its CPU clock. A
later, controlled reproduction shows that is wrong. The real fault is the memory subsystem:

- **Direct reproduction with the actual training, at LIGHT load (8 runs, 1 thread per core, so NOT a
  full-node effect):** the identical training runs at **168 steps/min/run on jaguar03 vs 1,301 on a
  healthy node (adriatic02) — 7.7x slower.** So jaguar03 is slow for this workload per-process, even
  lightly loaded; it is not a full-node power/thermal throttle.
- **Mechanism, via a single-thread random-access latency test (`latency.c`, pointer-chase over 256 MB,
  no dependencies):** jaguar03 **425 ns per random memory access vs 99 ns on a healthy node (ai06) —
  4.3x worse memory latency.**
- This reconciles every earlier puzzle: the RND/SAC training does constant small RANDOM memory reads
  (replay-buffer sampling), so it is memory-latency-bound and runs ~8x slower; a cache-resident compute
  benchmark (numpy matmul, AVX FMA burn) never touches main memory, so it ran at FULL speed on jaguar03
  and no CPU stress test reproduced the problem. The "~400 MHz" figure was an acpi-cpufreq mis-report,
  not a real clock; the CPU is fine.
- Root cause is therefore a MEMORY fault (a degraded DIMM, a channel dropped to a lower speed/rank, a
  memory-controller issue, or a NUMA/interleave change), recent (this same training ran at healthy speed
  in early July -- see the five-run table), and node-specific. It is checkable by admins: DIMM speeds
  (`dmidecode -t memory`), memory error counts (EDAC / `ras-mc-ctl`), memtest, NUMA config.
- Self-contained reproducer for admins: `gcc -O3 -o latency latency.c && ./latency 256` on jaguar03 vs a
  healthy node -> ~425 ns vs ~100 ns.

The sections below are kept for the record but read "throttle/clock" as "memory latency"; the timing,
recency, and node-specific findings all still hold.

## Is the jaguar03 throttling expected?

Under sustained load on all 112 cores, jaguar03's cores drop to about 400 MHz. A controlled single-core fixed-work benchmark on the node (recorded in run 5's `infra_history.md`, 2026-07-22T21:20) ran 34.69 s of compute versus 1.33 s on puma01 (2300 MHz) and 1.63 s on ai06 (1200 MHz) — about 26 times slower than puma01. Because the same fixed amount of work takes 26 times as long, the low clock is real and not just a misreport from the OS frequency counters.

The decisive fact is that the measured clock (~400 MHz) sits below the operating system's `scaling_min_freq` of 1500 MHz. That value is only the lowest frequency the Linux governor will ever *request*; it is not a hardware floor (https://docs.kernel.org/admin-guide/pm/cpufreq.html). The real delivered clock is set by the on-die System Management Unit (SMU), a firmware agent below the OS, which lowers the clock below the requested minimum when it hits a power, current, or thermal limit, or when an external signal (PROCHOT / a BMC power cap / APML) is asserted (https://docs.kernel.org/admin-guide/pm/amd-pstate.html, https://access.redhat.com/solutions/7002736). A measured clock below the OS minimum therefore proves the OS is not in control — no governor change, `scaling_min_freq` write, or `acpi-cpufreq`-to-`amd-pstate` swap can raise it.

Verdict: this is not expected EPYC 7663 behavior. That part's base clock (the guaranteed all-core sustained frequency) is 2.0 GHz, so a healthy node under full load should sit near or above 2.0 GHz — as puma01 does at ~2.3 GHz on the same reservation (https://www.amd.com/en/products/processors/server/epyc/7003-series/amd-epyc-7663.html). A stable ~400 MHz floor under full, non-oversubscribed, non-memory-starved load, on this one node, is a hardware or firmware protective throttle — a fault or misconfiguration, not normal power management. (Separately, the reported 2.0 GHz maximum instead of the rated 3.5 GHz means Core Performance Boost is disabled; that is a common, benign determinism setting and does not explain dropping *below* base.) The exact trigger cannot be pinned down from the available evidence alone — it requires reading the node's BMC/firmware state, which needs administrator access.

Likely causes (ranked by fit; the specific one is not confirmed):
1. An external protective throttle specific to jaguar03 — PROCHOT / a BMC or APML node power cap / a VRM or board thermal sensor — asserting only when all 112 cores draw current at once. Best fit: below the OS minimum, load-dependent, node-specific, die temperatures normal, no kernel log (https://forums.tomshardware.com/threads/bd-prochot-throttle-is-always-on-even-though-temps-are-low.3695109/, https://forum.level1techs.com/t/amd-epyc-9175f-fails-mysteriously-clocks-down-to-around-600-mhz-on-all-cores/247742).
2. A firmware/BIOS fault or a transient power-delivery event that latched the CPU into a low-power protective state — the closest published analog (dual EPYC 7763 stuck at 400 MHz under heavy load, normal temperatures) was traced to BIOS/BMC firmware corruption and fixed by a BIOS update (https://forums.servethehome.com/index.php?threads/amd-milan-7763-stuck-at-400-mhz-during-heavy-io-load.33584/).
3. A very low BIOS package power limit (cTDP/PPT) or determinism setting on this node. This is cheap to check but by itself normally yields a floor near 1.5–2.0 GHz, not 400 MHz (https://kb.storpool.com/install/amd-epyc-tuning.html).

Likely fixes (administrator-side; OS-side changes will not help):
1. Full power cycle of jaguar03 — not just a reboot; a power cycle is what cleared the latched throttle in every analogous report.
2. Read the BMC/IPMI system event log for PROCHOT, power-cap, thermal, or VRM events; check for and clear any node power cap (APML/Redfish/DCMI).
3. Update BIOS and BMC firmware; compare jaguar03's BIOS power/determinism settings against healthy puma01, and raise cTDP/package power limit to the part's rating (240 W) if set low.
4. Check cooling and power delivery under all-core load (inlet and VRM temperatures, PSU headroom) — a marginal VRM on a fully populated dual-socket board is a known cause.

One caveat on the evidence: the 26x fixed-work benchmark is the load-bearing proof (busy cores genuinely throttled), not the median-clock number, because a low median over allocated cores could in principle include idle cores parked at 400 MHz.

**How light is "light"? — the throttle needs SUSTAINED heavy load, not an instantaneous core count.** A controlled load-sweep on jaguar03 (2026-07-23, `load_sweep.py`, measuring the per-worker matmul rate, which is the true speed, since acpi-cpufreq misreports the MHz here — it read 400 MHz even at 1 busy core running fast) shows:
- **A single busy core runs the fastest of any node measured** — ~3,388 matmuls/s, above puma01's ~2,256/s — so jaguar03 is not throttled at rest; it boosts normally.
- **Pure compute does NOT throttle even at full load**: 40-second matmul-only bursts held full per-thread speed at 1 thread/core up to 108 cores, and at 2 threads/core all the way to 216 threads. No slowdown.
- **A memory-heavy burst (random gather over a 256 MB DRAM buffer per worker + compute, mimicking SAC replay sampling) at 2 threads/core** slowed only ~1.3x above ~128 threads — that is memory-bandwidth contention, not the frequency collapse.
- Neither 40-second burst reproduced the training's ~26–40x collapse. But the real SAC+RND training at 216 threads sustained for hours DID collapse (400 MHz). So the throttle is triggered by **sustained, memory-and-compute-heavy, near-full-node load held for minutes to hours** — the signature of a thermal limit (heat builds up over time) or a time-integrated package-power/EDC limit — not by reaching a specific core count in an instant. This is why a quick check of the node looks healthy while long training jobs collapse, and it is consistent with the node being FINE under the identical sustained full-node load in early July (run 3.2.1): the node's sustained-power/thermal headroom degraded around July 8–11 so it can no longer hold full-load training, though it still runs light and short work at full speed. The practical density threshold for training would require running the real workload at reduced densities for tens of minutes each to pin down; the short sweeps only bound it from below (jaguar03 is fine for light or brief work).

## Was jaguar03 always this slow? (history of use on this project)

Verdict: the slowdown is RECENT, not chronic. jaguar03 ran this project's training at normal-to-fastest speed under the same full-node load as recently as July 5–8, but is throttled the very next time it is loaded, July 11 (run 3.2.3). So the change happened **between July 8 and July 11**, and it has persisted since — runs 3.2.3, 3.2.4, and 5 all hit it.

Timeline of sl5nw's jaguar03 use (full `sacct` history, 922 jobs, continuous 2026-05-01 through 2026-07-22 — an earlier count was truncated by a `head -400` pipe and later corrected):
- 2026-04-08: first-ever jaguar03 jobs for this user, but these are tinker negotiation eval jobs, not RND. All May–early-June jaguar03 use is tinker work.
- 2026-06-24: first RND use of jaguar03 (run 2, 13 worker jobs in a mixed-node sweep).
- 2026-06-25 to 2026-07-08: jaguar03 was the fastest node in every RND run. Median wall-clock hours per completed 1M-step run — run 3.1.1 (Jun 26): 8.5 h on jaguar03 vs 10.4 h on other nodes; run 3.1.2 (Jul 2): 8.3 h vs 11.1 h; run 3.2.1 (Jul 4–8): 13.9 h vs 16.1 h.
- 2026-07-11 (run 3.2.3) and 2026-07-15 (run 3.2.4): each sent 4 jaguar03 worker jobs (32 tasks = 192 threads, full-node density). Every one claimed only ~1 run per worker, completed **zero**, and hit the 96 h walltime (TIMEOUT) — the same throttle signature as run 5. (An earlier forensic pass wrongly reported these as "barely used / not used"; it missed the `fossil1` and `valid32` jaguar03 jobs, so this line is the correction.)
- 2026-07-20 to 2026-07-22: run 5. jaguar03 completed zero runs in ~30 hours; other nodes ran at ~17.5 h median.

The decisive same-load comparison uses `sacct -X` (job-to-node) plus the job logs (per-run durations), because the per-run JSON records contain no node field:
- Run 3.2.1, early July: jobs 6387153–6387157 and 6387168 — exactly 6 jaguar03 jobs, each 32 tasks (32 AllocCPUS), all COMPLETED, started 2026-07-05T09:39:58, elapsed 2 d 13 h to 2 d 17 h (~61.5–65.8 h). That is 192 threads (2 per core, ~full node) held continuously for about 2.5 days. Per-run durations parsed from two of these jobs: job 6387153 median 13.99 h (p10–p90 13.71–14.22 h, max 14.46 h, n=106 runs); job 6387168 median 13.91 h (p10–p90 13.58–14.25 h, max 14.89 h, n=109). Pooled across all nodes run 3.2.1 was 15.11 h median with a tail to the 24 h cap, so jaguar03 with no tail was among the faster nodes.
- Run 5, now: jobs 6516452–6516457 — again exactly 6 jaguar03 jobs, each 32 tasks (32 AllocCPUS), identical 192-thread shape, started 2026-07-21T15:15:36, elapsed 1 d 06 h (~30.3 h), all CANCELLED with zero completed runs. In-flight runs averaged ~123k of 1M steps (~4.1k steps/hr, versus ~50–67k steps/hr elsewhere).

Same node, same user, same 32-task/192-thread full-node load, same 1M-step pytorch workload, 16 days apart: about 14 hours per run then, zero completions in 30 hours now. Three days of identical all-core load in early July with no throttling rules out "jaguar03 always collapses under this load."

The transition is visible across FIVE consecutive same-setup runs (each an identical SAC+RND job: 32 tasks = 192 threads, 2 per core, 64 G, same env), so it is not a one-off. The clearest signal is "runs finished per worker" = claimed-runs (from the job logs) / 32 workers: a healthy worker finishes its ~14 h run and claims the next, so a long job completes several per worker; a throttled worker claims one run and never finishes it.

| Run | Launched | jaguar03 jobs (32-task / 192-thread) | Runs finished per worker | Outcome |
|---|---|---|---|---|
| 3.2.1 | Jul 5  | 6, all COMPLETED (ran 61-66 h) | ~4    | fast (~14 h/run) |
| 3.2.2 | Jul 7  | 5, all COMPLETED (ran 11-23 h) | ~1    | fast (~16 h/run) |
| 3.2.3 | Jul 11 | 4 TIMEOUT at 96 h (+1 brief spare) | <1 (stuck) | throttled, 0 finished |
| 3.2.4 | Jul 15 | 4 TIMEOUT at 96 h (+2 brief spares) | <1 (stuck) | throttled, 0 finished |
| 5     | Jul 21 | 6, all CANCELLED (ran 30 h) | 1, never done | throttled, 0 finished |

Two runs before (Jul 5, Jul 7) succeeded and three after (Jul 11, 15, 21) failed, all identical jobs on the same node -- pinning the onset to July 7-11.

Two corrections that do not change the verdict:
1. puma01 was also slowed by the concurrent run-5 load — its first completed run took 30.47 h (job 6516458), not the ~14 h implied elsewhere. So the clean jaguar03-vs-puma01 contrast is the point-in-time 400 MHz vs 2300 MHz benchmark, not a "puma01 finishes in 14 h" figure. puma01 was slow; jaguar03 was far worse (zero completions).
2. Some historical contention did exist on other nodes (run 3.2.4 notes ~71 run-3.2.1 runs slowed to about half speed against the old 24 h cap), but the log parse shows those were not on jaguar03 — jaguar03's max was 14.89 h with no tail. jaguar03's 26x / 400 MHz collapse is genuinely new and specific to jaguar03 now.

Three RND runs sent worker jobs to jaguar03 after the throttle appeared — 3.2.3 (Jul 11), 3.2.4 (Jul 15), and 5 (Jul 20) — and in each, the jaguar03 jobs claimed runs, completed zero, and timed out or were cancelled. **No run lost data or got wrong numbers.** The throttle only prevents a run from finishing; it never changes a completed run's result (same seed and same computation give the same final reward, just slower). In all three cases the runs the jaguar03 jobs failed to finish were requeued and re-ran on healthy nodes, and every queue drained fully: run 3.2.3 has 4,652 completed runs (queue pending=0), run 3.2.4 has 498 (pending=0), and run 5 is draining its requeued tail on the cpu partition now (the 6 jaguar03 jobs were cancelled 2026-07-22T21:41, ids 6516452–6516457, and their 192 in-flight runs requeued). The only cost was wasted jaguar03 compute and delay. Runs before the onset (2, 4, 3.1.1, 3.1.2, 3.2.1, 3.2.2) used jaguar03 while it was fast and were unaffected. The throttle diagnosis and the takeaway ("jaguar03 is unsuitable for CPU-bound all-core training; prefer puma01/bigcat/cortado") are recorded in run 5's `infra_history.md`.
