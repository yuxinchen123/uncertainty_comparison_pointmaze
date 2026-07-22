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
