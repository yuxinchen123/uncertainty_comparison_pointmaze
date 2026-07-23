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
