# Train run 5 — infrastructure history

Sweep `2026-07-20-16-55_set-baseline`. Owner sl5nw. cpu + nolim only (no reservation/gpu/gnolim). Monitored every ~20 min.

## 2026-07-20T18:26:59 — launch
- Committed+pushed code e56dbbd before submit (commit-before-submit rule).
- Canary 6514315 verified AllocCPUS=32, correct run-5 params, no errors.
- Full wave: 12 cpu (384t, 16 headroom) + 1 nolim (32t). Over-cap 13th cpu job (6514327) cancelled to keep headroom.
- Owner id file: 13 ids. Durable nohup monitor_loop.sh running (requeue+prune+completion).
- Collaborator packet generated (for_collaborator/, cpu+nolim only).

## 2026-07-20T18:28:21 — first health tick
- 416 checkpoint JSONs written (runs at ~50-100k/1M steps, completed=false — normal early progress).
- Worker CPU ~99.9% on bigcat01 (correct ntasks-per-core=2 packing; not the half-speed pathology).
- Collaborator yuxinchen submitted 6514352 (collab0, 16 cpu workers) — packet works. Total 432 slots.
- 0 failures, 0 problem reports. Healthy.

## 2026-07-20T18:53:44 — tick
- Owner 13 jobs RUNNING (384 cpu + 32 nolim = my full headroom-limited allocation). No deaths.
- Collaborator yuxinchen SCALED UP: ~14 run5 jobs, ~240 workers. Total running workers=656.
- Queue: running=656 pending=2344 done=0 failed=0 pruned=0. 416 checkpoint JSONs (completed=false); 0 complete yet (~2h in, 1M steps ~15h).
- Pruning not yet active (bar=None) — correct: the rule scores only COMPLETED runs, none complete yet; betas reach n>=30 around ~15-20h.
- CAVEAT: the cspartcpu QOS counter reads 0 despite 12 running cpu jobs (unreliable, as at launch). Do NOT top up from it — use id-file accounting (I am at 384 cpu / 32 nolim = my headroom cap, so NO top-up). 0 failures, 0 problem reports.

## 2026-07-21T00:56:19 — BUG FIX (prune controller key reconstruction)
- Found: train.py's output JSON does NOT carry the queue-level config_key/variant, only the individual
  rnd_ fields. prune_controller.load_scores read d.get("config_key") -> None for all records, so it
  would group everything under None and NEVER prune once runs complete.
- Not yet manifested (0 completed records). Fixed before completions (~3h out): added
  prune_controller.key_from_record(d) reconstructing the 8-field key from the record fields
  (readout=='mse_mean' -> variant origsmall). Verified: real origsmall record -> correct key in
  ORIGSMALL_KEYS; +1 test (key_from_record == build_queue.config_key for all 10 configs); 6 tests pass.
- Live on the next prune cycle (nohup monitor_loop spawns a fresh python process each cycle).

## 2026-07-21T11:33:12 — first pruning executed (controller working)
- bar=36.72 (b1000, n=30, the leading original-small beta). b10000 PRUNED (mean 11.70, upper99
  21.55 < bar): 184 pending markers moved to pruned/. Race so far: b1000 36.72 > b100 25.34 >
  b0.1 15.19 > (b10000 pruned 11.70). Consistent with run-3's high-beta reward-norm RND finding,
  but the sweet spot is b1000 not b10000 (too much intrinsic weight destabilizes).
- done=305, no failures, no problem reports. The prune-controller fix (key reconstruction) is
  validated on real data end-to-end.

## 2026-07-21T15:16:12 — reservation re-authorized by user; added jaguar03 + puma01
- User re-authorized reservation sl5nw_151 (extended to 2026-08-04). Both nodes idle.
- Submitted 9 jobs: jaguar03 6x32x1 (192/222 threads, 2G/cpu, --partition=gpu --gpus-per-node=0),
  puma01 3x32x1 (96/158 threads, 1900M/cpu, --partition=cpu). All RUNNING. +288 workers.
- ids 6516452-6516460 appended to owner id file (now 22 ids). Speeds the survivor-arm tail.

## 2026-07-22T14:52 — reservation tail is slow (not stuck); 192 in-flight C2 runs crawling
- State: done=1689, pruned=1039, running=272, pending=0, failed=0. No orphans, no problem reports.
- Diagnosed the slow tail: of 272 in-flight (completed=false) records, 192 are stuck at their
  first eval flush (step ~100k of 1e6), last written 188-278 min ago; 68 are at 600-900k and 12
  at >900k and ARE writing fresh (progressing normally).
- The 192 are almost all C2 seeds, claimed ~23h ago (15:16 on 07-21) by the 8 alive reservation
  jobs (6516452-6516457 on jaguar03, 6516458-6516459 on puma01). They are NOT orphans and NOT
  hung: sstat shows ~100% CPU per task (23h39m CPU over ~23h wall). They are simply running ~6x
  slower than the open-partition workers did (~5k vs ~33k steps/hr): jaguar03 packs 6x32=192
  tasks (2 per physical core) and was co-loaded by the whole sweep during 15:00-05:00.
- requeue_orphans correctly leaves them (claiming job alive => not an orphan; the 6h staleness
  fallback only fires when there is NO claim line). This is expected behavior, not a bug.
- Consequence: at this rate the 192 runs need many more hours (worst case ~1-2 days) to reach 1e6.
  They are mostly redundant C2 seeds -- C2 already has n=234, N1 n=233, b1000 n=227, all far above
  the sample needed for stable estimates (the substitution test already confirms C2/N1 stability).
  Surfaced the finish-now vs wait-for-300 tradeoff to the user.

## 2026-07-22T21:20 — HARD EVIDENCE: jaguar03 hardware-throttles to 400 MHz under load
- Historical per-node finish times: every node finishes a run in 15-21h; jaguar03 has finished
  ZERO runs in ~30h. jaguar03 in-flight runs average ~123k of 1e6 steps (median 100k) = ~12% after
  30h (~4.1k steps/hr) vs ~50-67k steps/hr elsewhere.
- Controlled single-core fixed-work benchmark (3000 256x256 matmuls + random-gather over 256MB),
  run via srun on each node under current load:
    puma01   (reservation, 64-task load): 2300 MHz  compute=1.33s  mem=1.98s
    ai06     (open partition):            1200 MHz  compute=1.63s  mem=0.43s
    jaguar03 (reservation, 192-task load): 400 MHz  compute=34.69s mem=3.15s
  => jaguar03 compute is 26x slower than puma01 / 21x slower than ai06, but memory only ~1.6x
  slower. So the bottleneck is CLOCK, not memory bandwidth (earlier bandwidth guess REFUTED).
- Root cause (cpufreq read on jaguar03 under load): 2x AMD EPYC 7663 (112 cores/224 threads),
  governor schedutil, cpu0 scaling_min=1.5GHz scaling_max=2.0GHz but scaling_cur=400MHz -- the
  ACTUAL clock is BELOW the OS-requested minimum, and only 31/224 cores are >1GHz. Actual freq
  below the software floor = HARDWARE-ENFORCED throttling (power/thermal cap on the dual 240W
  EPYC under all-core load). Node is also admin-capped at 2.0GHz max (chip boosts to 3.5GHz).
  puma01 on the same reservation but lighter load holds 2.3GHz and finishes runs in ~14h.
- Takeaway for future sweeps: jaguar03 is unsuitable for CPU-bound all-core training (it collapses
  to ~400MHz); prefer puma01/bigcat/cortado. No jobs touched pending the user's decision.

## 2026-07-22T21:41 — cancelled the 6 jaguar03 jobs (user ok), requeued their 192 orphans
- User authorized freeing jaguar03 for a threading-config experiment. scancelled ONLY the 6
  jaguar03 ids from the owner id file (6516452-6516457); puma01 jobs 6516458/6516459 left running.
- BUG FOUND + FIXED in requeue_orphans.py: marker_config_key called
  build_queue.config_key(spec["params"], spec["beta"]) with 2 args, but config_key(cfg_spec) takes
  1 (the whole config dict). Phase-2 (failed/->pending/) would have crashed on the first orphan,
  stranding markers in failed/. Fixed to config_key(spec). This bug never fired before because no
  orphan had ever existed this sweep (all prior job exits were clean COMPLETED).
- Ran requeue_orphans (fixed): 192 jaguar03 orphans moved running/->failed/->pending/ (43 C2, 47
  N1, 48 b100, 54 b1000). puma01's live runs correctly left in running/ (their job is still alive).
- Ran prune_controller --once: the 48 requeued b100 markers belong to a pruned arm, re-pruned them
  (pruned 1039->1087). Net queue: pending=143 (42 C2, 47 N1, 54 b1000 -- wanted arms), running=31
  (puma01), done=1739, pruned=1087, failed=0; total 3000. No run lost; jaguar03 is now free.

## 2026-07-22T23:52 — jaguar03 threading experiment DONE (report in jaguar03_threading_experiment/report.md)
- Compared 3 thread->core layouts at full 108-core/400 MHz load: A=108 runs/2thr, B=54 runs/4thr,
  C=216 runs/1thr (the layout the real sweep used). Steady-state steps/min per run, node totals:
    A: per-run 159, TOTAL 17242 steps/min, per-core 160, 400 MHz
    B: per-run 211, TOTAL 11286 steps/min, per-core 105, 400 MHz  (worst: 4 threads waste cores)
    C: per-run  82, TOTAL 17801 steps/min, per-core 165, 400 MHz  (best total, ~3% over A)
- Verdict: C wins total node throughput but only ~3% over A; A gives ~same total with 2x better
  per-run latency (1M-step run ~105h vs C's ~203h at 400 MHz). B is ~37% worse. The real sweep
  already used C, so its slowness was the 400 MHz throttle (~26x), not a layout mistake -- the
  layout is a ~3% lever, the throttle a ~26x one. No layout makes jaguar03 usable while throttled.
- Web research + past-run forensics (6-agent verified workflow) added to report.md: the ~400 MHz is
  BELOW the OS scaling_min (1.5 GHz) => firmware/SMU throttle (power/thermal/PROCHOT), a node FAULT,
  fixable only admin-side (power cycle, BMC event log, BIOS cTDP/determinism, cooling). And it is
  RECENT: jaguar03 ran the identical 192-thread load at ~14h/run in early July (run 3.2.1, fastest
  node) but completed 0 runs in 30h now (run 5). Recommend reporting jaguar03 to cluster admins.
- Experiment folder holds: report.md, run_experiment.py, data/{A,B,C}/progress_*.csv, logs/.
  First C attempt (0.2s stagger) hit a 216-process filesystem-import herd (91/216 stuck in setup);
  re-run with 0.8s stagger + 30min window was clean (all 216 at steady 400 MHz). jaguar03 released.

## 2026-07-23T00:10 — CORRECTION: throttle onset is Jul 8-11; runs 3.2.3 + 3.2.4 also hit it (no data lost)
- The 6-agent research had said runs 3.2.3/3.2.4 "barely used / did not use" jaguar03. WRONG -- it
  missed their jaguar03 jobs (job-name fossil1 / valid32). Corrected via sacct -X on each run's id file:
    run 3.2.3 (Jul 11): 4 jaguar03 jobs (fossil1 6485364-67), 32 tasks each = 192 threads; each claimed
      ~1 run/worker, completed 0, TIMEOUT at 96h. = throttle signature (healthy would do ~6 runs/worker).
    run 3.2.4 (Jul 15): 4 jaguar03 jobs (valid32 6497224-27), same shape, claimed ~28-32, completed 0,
      TIMEOUT at 96h.
    run 5 (Jul 20): 6 jaguar03 jobs, completed 0 in 30h, cancelled+requeued.
- So the throttle onset is BETWEEN Jul 8 and Jul 11 (jaguar03 fast through Jul 5-8 run 3.2.1 ~14h/run;
  throttled by Jul 11), narrower than the earlier "after Jul 8". It has persisted since.
- DATA INTEGRITY: no run lost data or got wrong numbers. The throttle only prevents completion; a
  completed run's value is unchanged (same seed+compute). The runs the jaguar03 jobs failed to finish
  were requeued to healthy nodes; all queues drained fully -- 3.2.3: 4652 completed (pending=0);
  3.2.4: 498 (pending=0); run 5: draining on cpu partition. Only cost = wasted jaguar03 compute + delay.
- report.md "Was jaguar03 always this slow?" section corrected accordingly.

## 2026-07-23T00:45 — CONFOUND CHECK: same Slurm settings AND same env, fast then vs slow now
- Question: is the jaguar03 "collapse" actually a different submission setting, not the hardware?
  Compared the FAST early jaguar03 job (run 3.2.1, 6387153/6387168, Jul 5, ~14h/run) vs the SLOW
  ones (run 3.2.3 fossil1 6485364 Jul 11; run 3.2.4 valid32 6497224 Jul 14; run 5 run5res 6516452
  Jul 21).
- sacct AllocTRES IDENTICAL for all: cpu=32, mem=64G, node=1. Worker scripts byte-identical except
  the run-folder path and the python path. Both use: --ntasks=32 --cpus-per-task=1
  --ntasks-per-core=2 (192 threads, 2 per core), --mem-per-cpu=2G (64G/job), OMP_NUM_THREADS=1,
  srun --wait=0. Same packing, same memory, same thread cap.
- ENV also ruled out: run 3.2.1 (fast) AND run 3.2.3 (throttled) BOTH used the SAME private env
  /u/sl5nw/.conda/envs/exploration. (run 3.2.4 and 5 used the shared clone, but 3.2.3 on the private
  env was already throttled, so the env change is not the cause.)
- Conclusion: same submission + same env + same node + same 192-thread/2-per-core workload, fast on
  Jul 5, collapsed on Jul 11. Only variable is time. Rules out the settings/env confounds ->
  the collapse is a genuine jaguar03 hardware/firmware degradation ~Jul 8-11, not a user-side change.

## 2026-07-23T02:15 — REPRODUCED + RE-DIAGNOSED: jaguar03 fault is MEMORY LATENCY, not a CPU throttle
- Controlled reproduction with the REAL training at LIGHT load (8 runs, 1 thread/core, NOT full node):
  jaguar03 168 steps/min/run vs healthy adriatic02 1301 steps/min/run = 7.7x slower. So it is per-process
  and load-independent, not a full-node power/thermal throttle.
- Mechanism: single-thread random-access latency test (latency.c, 256MB pointer-chase, no deps):
  jaguar03 425 ns/access vs healthy ai06 99 ns/access = 4.3x worse memory latency.
- This reconciles the earlier puzzles: the RND/SAC training is memory-latency-bound (replay-buffer random
  sampling) so jaguar03 runs it ~8x slower; cache-resident compute (numpy matmul, AVX FMA burn) never
  hits DRAM so it ran at FULL speed on jaguar03 (why no CPU stress test reproduced it, and why the raw
  compute microbench looked fast). The "~400 MHz" was an acpi-cpufreq MISREPORT, not a real clock -- the
  CPU is fine.
- Corrected root cause: a MEMORY fault (degraded DIMM / channel dropped to lower speed-rank / memory
  controller / NUMA-interleave change), RECENT (same training ran ~14h/run in early July), node-specific.
  Admin checks: dmidecode -t memory (DIMM speeds), EDAC/ras-mc-ctl (memory errors), memtest, NUMA config.
- Self-contained repro for admins (no training env): gcc -O3 -o latency latency.c && ./latency 256
  -> jaguar03 ~425 ns vs healthy ~100 ns. report.md updated with a CORRECTION section at the top.

## 2026-07-23T02:30 — ruled out "slow by nature": same-architecture control confirms degradation
- Concern: is jaguar03's 426 ns latency just a naturally-slow (big dual-socket / more NUMA hops) node?
- Controlled test, identical latency.c (256MB random pointer-chase), against a SAME-ARCHITECTURE node:
    jaguar03   : 2 sockets, 2 NUMA nodes -> 426 ns/access
    adriatic02 : 2 sockets, 2 NUMA nodes (healthy) -> 112 ns/access
  Same dual-socket/2-NUMA architecture, same test -> jaguar03 is 3.8x worse. So NOT a by-design NUMA
  effect (a healthy dual-EPYC is ~100 ns local, ~200-250 ns worst-case cross-socket; 426 ns is beyond
  any healthy config). (numactl absent on both nodes, so NUMA-local binding not tested, but the
  same-architecture healthy control already controls for socket/NUMA count.)
- Independent confirmation (workload, not microbench): jaguar03 ran this SAME memory-bound training at
  ~14 h/run on Jul 5/7. If its memory were slow by nature the training could not have been fast then.
  => memory latency was normal then and degraded since. Two independent lines both => genuine
  degradation, not a naturally-slow node. Diagnosis (recent memory-subsystem fault) stands.

## 2026-07-23T02:53 — both puma01 reservation jobs completed cleanly; sweep in final drain
- puma01 jobs 6516458 (ended 02:22:07) and 6516459 (ended 02:34:55) both State=COMPLETED after ~1d11h.
  Clean exit (queue empty for them) -> no orphaned running/ markers. Reservation-node work is done.
- Queue now: done=1774, pruned=1087, running=139, pending=0 (accounts for all 3000). Final drain: the
  remaining 139 full 1M-step runs are all running concurrently on the 5 cpu-partition jobs (160 task
  slots > 139 runs, so no queueing). SWEEP_COMPLETE expected when the slowest finishes (~14-16h/run on
  healthy cpu nodes) -> later on 2026-07-23. No action.

## 2026-07-23T11:50 — fleet-wide latency survey rules out "jaguar03 slow by nature"
Concern: is jaguar03's 424 ns just normal for a big/many-CPU node? Measured latency.c (256MB random
pointer-chase) across the gpu partition, spanning 16 -> 256 CPUs:
  jaguar05(16)=111  nekomata01(24)=99  jaguar02(32)=101  ai06(32)=105  adriatic06(32,idle)=112
  cheetah01(32,busy)=286  jaguar06(48)=96  jaguar01(64)=104  serval06(64)=129  cheetah02(72)=109
  lotus(80)=107  serval03(128)=131  puma01(160,busy)=803  cheetah04(256)=145  jaguar03(224,idle)=424
Findings:
 - Healthy band is a FLAT ~95-145 ns from 16 to 256 CPUs -> node size does NOT predict latency.
 - DECISIVE: cheetah04 (256 CPUs/128 cores, BIGGER than jaguar03) = 145 ns; jaguar03 (224/112) = 424 ns.
   The larger node is 2.9x faster -> "big node = slow memory" is false.
 - jaguar03 siblings jaguar01/02/05/06 (same vendor family, one with same 1TB RAM) = 96-111 ns -> not a
   model trait; jaguar03-specific.
 - jaguar03 measured IDLE (best case) while most comparators were busy (contended, inflated) -> the true
   gap is even larger. jaguar03 is a lone 3-4x outlier => genuine fault, not natural slowness. CONFIRMED.
 - Caveats: cheetah01(286) and puma01(803) read high but were BUSY (co-tenant memory contention), not
   clean idle readings. puma01 is the other reserved node -> worth a separate idle check (may be load, or
   may also need a look); does not affect the jaguar03 conclusion.
Data: jaguar03_threading_experiment/latency_fleet_survey.txt

## 2026-07-23T12:15 — REVISED DIAGNOSIS: memory BANDWIDTH is the dominant fault, not latency
Prompted by: (a) training slowdown (~10x) >> latency penalty (~4x), so latency alone never fit; and
(b) puma01 (788ns) appeared to train OK, questioning the latency story.

Combined lat-vs-train probe (clean idle latency + real single-thread RND training rate, back to back):
  node        latency   train_rate(steps/min)
  jaguar01    105 ns    2829   (healthy)
  adriatic06  111 ns    1693   (healthy, idle)
  jaguar03    424 ns     171   (idle -> 9.9x slower than adriatic06)
  puma01      788 ns     879   (BUSY: 118/160 cpus on other pmam-* jobs -> all readings contended)

Memory bandwidth (STREAM-triad, bandwidth.c):
  node        1-thread    16-thread
  jaguar01    13.4 GB/s   69.9 GB/s   (healthy)
  adriatic06  12.0 GB/s   54.2 GB/s   (healthy)
  jaguar03     2.7 GB/s    8.3 GB/s   (IDLE -> 4.4x/6.5x worse -> THE smoking gun)
  puma01       2.3 GB/s    (n/a)      (BUSY, contended -> not a clean hardware reading)

jaguar03 (idle) vs adriatic06 (idle): latency 3.8x worse, bandwidth 4.4x (1thr)/6.5x (16thr) worse,
training 9.9x slower. CPU compute is fine (AVX burn fast). => jaguar03's MEMORY SUBSYSTEM is degraded in
BOTH bandwidth and latency; bandwidth is the LARGER deficit and the one that matches the sweep slowdown
(~6.5x aggregate-bandwidth deficit ~ the earlier ~7-8x full-node training slowdown). Latency (4x) is a
secondary symptom.

puma01 is NOT a counterexample: it is 74% loaded by non-this-session jobs (6519833/34/35/859, ~118 cpus),
so its 788ns/2.3GB/s/879-steps are all contention, not hardware. Cannot get a clean puma01 reading without
cancelling others' jobs (forbidden). puma01's true hardware health is UNKNOWN (worth a clean idle check
later); it does not change the jaguar03 conclusion.

Bottom line: earlier "latency fault" was incomplete. Correct diagnosis = jaguar03 memory subsystem
(bandwidth-dominant + latency) degraded; CPU fine; training ~10x slow. Two independent reproducers now:
bandwidth.c (2.7 vs 12 GB/s) and latency.c (424 vs 111 ns).
Data: lat_vs_train_results.txt, bandwidth_results.txt, puma01_bw.txt

## 2026-07-24T16:50 — admin could not reproduce; re-checked post-reboot: FAULT PERSISTS, block-size ruled out
Admin (Jed) could not reproduce, found no system errors, suggested adjusting block-size. Re-checked:
- jaguar03 was REBOOTED (up since 2026-07-24 13:02); reboot did NOT clear the fault.
- Re-measured (idle, exclusive) 2026-07-24: jaguar03 1-thread BW 2.7-2.9 GB/s (all buffer sizes),
  4-thread 4.3 GB/s, latency 413 ns. adriatic06 (healthy): 1-thread ~11 GB/s, 4-thread 22 GB/s,
  latency 138 ns. Same ~4x (BW) / 3x (latency) gap as 2026-07-23. Persistent.
- BLOCK-SIZE SWEEP (single-thread, working set 3MB->768MB), directly answering Jed's suggestion:
    working_set   adriatic06(healthy)   jaguar03
    3 MB          19.8 GB/s             3.1 GB/s
    12 MB         11.2                  3.0
    24 MB         10.7                  3.1
    48 MB         10.8                  3.0
    96 MB         10.9                  2.7
    768 MB        11.1                  2.9
  jaguar03 is 4-6x slower at EVERY block size -> no block size makes it healthy; block-size is not the
  cause and cannot be a fix.
- NEW (broadens diagnosis): jaguar03 is slow even for tiny cache-resident buffers (3MB: 3.1 vs 19.8),
  not just DRAM. Register-only AVX compute is full-speed. So the degradation is in the data-movement
  path (loads/stores at ALL cache levels + DRAM), pointing at the memory controller / Infinity Fabric
  (uncore/FCLK) rather than a single DIMM. A perf degradation of this kind logs NO system errors
  (consistent with Jed finding none) and hides behind jaguar03's large 512MB L3 for small tests.
- Suggested checks broaden: Infinity Fabric / uncore (FCLK) clock, memory controller, in addition to
  DIMM speed/ECC; a plain reboot did not help (already tried) -> full power cycle or hardware inspection.
Data: recheck_2026-07-24.txt, jaguar03_blocksize_sweep.txt, adriatic06_blocksize_sweep.txt

## 2026-07-24T17:20 — ROOT CAUSE CORRECTED: CPU clock throttled to ~400 MHz node-wide (not a memory fault)
The memory-bandwidth diagnosis was a SYMPTOM. Controlling for the AVX-512(adriatic)/AVX2(jaguar) ISA
difference with a portable scalar compute loop (clockcheck.c, no -march=native) exposed the real cause:
- Register-only compute (no memory traffic): jaguar03 0.49 G(mul-add)/s vs adriatic06 2.45 -> 5x slower.
- scaling_cur_freq UNDER LOAD, all cores both sockets (0,28,56,84,112,140,168,196): ALL ~399,700-399,985
  kHz = ~400 MHz. scaling_min=1,500,000, scaling_max/bios_limit=2,000,000. So cores run BELOW the OS
  minimum (1.5 GHz) -> not the governor (schedutil); a firmware/hardware clamp below OS control.
- /proc/cpuinfo shows a misleading 1500 MHz (stale); scaling_cur_freq shows the true 400 MHz. This is why
  the admin (and my earlier check) found nothing -- and why I WRONGLY called the 400 MHz an acpi-cpufreq
  misread on 07-23: I had only compared jaguar03 to itself. Cross-node compute proves 400 MHz is REAL.
Single root cause explains everything: compute 5x slow (400MHz vs 2000 nominal), single-thread memory
bandwidth 4x slow (slow core issues loads slowly), latency 3x (SoC/fabric likely throttled too),
training ~8-10x slow. No system errors (a power/firmware clamp does not log). Reboot (up 2026-07-24
13:02) did NOT clear it -> persistent firmware/hardware throttle.
Likely mechanism (freq pinned below OS min): power delivery (VRM), thermal (stuck PROCHOT / sensor), or
BMC/firmware power-cap (PPT). Fix: full power cycle / BMC reset / check power+thermal via BMC-IPMI; a
plain reboot already failed.
Admin verification (seconds): `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq` under load ->
~400000 on all cores (should be >=1500000). Data: clockcheck.c, and the compute/freq run above.

## 2026-07-29 — admin rebuttal (cpuinfo ramps to 2.0GHz); re-tested with ground-truth throughput
Admin (Jed): did a "full power drain", says hardware is fine, recommends using another node. His evidence:
cpuinfo_cur_freq (the ACTUAL hardware freq, root-only) ramps 1.5->2.0 GHz under all-core load; scaling_cur_freq
is not the real freq; and 2.5 GHz was an unfair baseline (7663 maxes at 2.0). He is right on those points:
- Confirmed scaling_cur_freq is UNRELIABLE: adriatic06 read 800MHz while its compute was 2.49 GHz. So my
  earlier "400 MHz" scaling_cur_freq reading is not a trustworthy frequency.
- cpuinfo_cur_freq is Permission denied for a non-root Slurm job -> I never could read the true hw freq.
- jaguar03 uptime still 2026-07-24 13:02 -> this node was NOT actually rebooted/drained (despite the note).
BUT the ground-truth signal (compute throughput = work/time, needs no freq counter) still shows jaguar03 slow,
today, under FULL all-core load:
  jaguar03   all 222 threads busy -> per-core effective 0.28 GHz/thread (~0.55/core)
  adriatic06 all 30 threads busy  -> 2.49 GHz/thread
  jaguar03 single core -> 0.51 GHz ; --cpu-freq=Performance did NOT change it (0.52).
Register-only loop (no memory), same binary -> Zen3 has >= the Xeon's FP IPC, so 0.5 vs 2.5 G/s means the
CORE CLOCK my jobs get is ~5x low. This is real for MY jobs.
Reconciliation (Jed's 2.0GHz root reading vs my 0.5GHz throughput on the same un-rebooted node): the frequency
my Slurm jobs get differs from root's direct benchmark. Two possibilities, not yet distinguished:
  (A) my Slurm jobs are frequency/power-capped on jaguar03 (Slurm power/cpufreq plugin, reservation, or cgroup)
      while root is not -> hardware fine, fix is Slurm config or just use another node.
  (B) hardware genuinely throttled AND cpuinfo_cur_freq also misreports (the acpi-cpufreq driver is already
      wrong for scaling_cur_freq) -> hardware/firmware fault.
Decisive test I cannot run (non-root): have Jed run clockcheck.c AS ROOT and report effective GHz. Fast(~2)=>A,
slow(~0.5)=>B. Practical: Jed recommends another node; run 5's jobs are already all on other nodes. Verdict for
the user: jaguar03 is STILL slow for our jobs (measured), regardless of which counter is right; move off it.
Data: allcore_probe.sh output above.
