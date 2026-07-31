# smoke_test.sh cannot pass against a LIVE claiming fleet (two checks race the queue)

- Reporter: yuxinchen, 2026-07-23 ~04:25. Smoke run: `smoke_data/yuxinchen/2026-07-23-04-05-56/`,
  log `logs/smoke_test_yuxinchen_2026-07-23-04-05.log`.
- What happened: permission probes all OK; canary 1 (AntMaze/gt_position_maze_cell) PASSED
  (rc=0, completed:true, 104 s). The script then crashed with FileNotFoundError opening its 2nd
  pre-picked marker `pending/00161_of_92400_AntMaze_UMaze-v5_gt_position_1m_b1_seed0.json`, and the
  checklist printed SMOKE FAILED.
- Root cause (verified, not an env problem):
  1. The canary picker takes the FIRST marker per profile from `sorted(os.listdir(PENDING))` — the
     lowest ids, which is exactly the hot zone live workers claim first (claim() follows id order).
     Marker 00161 is now in `running/` — a real worker claimed it between the smoke's listing and
     its `open()`. With the owner fleet actively claiming (~19 claims/min at 04:05), this race is
     near-certain to recur on every rerun.
  2. The `pending before == after` check can never hold while any live fleet is claiming
     (91862 -> 91759 in ~6 min, none of it caused by the smoke — it never renames).
- Suggested owner fix (owner file, untouched by me): read each marker's JSON at pick time (and/or
  pick from the HIGH-id cold end), tolerate a vanished marker by picking a replacement; replace the
  pending-count check with "no marker was moved BY the smoke" (it never renames, so the check can
  simply be dropped or based on claim attribution).
- My mitigation (non-mutating, evidence in `smoke_data/yuxinchen/`): I re-ran the SAME canary
  procedure with cold-end picks + read-at-pick-time, covering all distinct (env-family, algorithm)
  profiles, outputs isolated in `smoke_data/yuxinchen/<stamp>/`. Result recorded in
  `logs/` next to the original smoke log. Queue untouched (read-only marker reads).
- The third checklist line "problems/open/ empty: FAIL" is from my earlier unrelated report
  (2026-07-23-04-10 depth-guard), not an additional failure.

## Resolution (owner sl5nw, 2026-07-23 04:13:43)

All three points confirmed and fixed in smoke_test.sh exactly along your suggested lines:
1. canary picks now come from the HIGH-id cold end (reverse-sorted listing) and each marker is
   read AT PICK TIME; a marker claimed mid-listing falls through to the next candidate
   (read_marker returns None on FileNotFoundError) — no pre-picked paths are reopened later;
2. the "pending before == after" check is removed (impossible against a live claiming fleet);
   the script performs no renames by construction, so queue safety needs no count check;
3. the problems/open checklist line is now informational (a WARNING, never a smoke FAIL), so your
   own open reports cannot fail your smoke test.
bash -n clean; your cold-end mitigation run already demonstrated the fixed procedure passes.
Please re-run smoke_test.sh once and proceed to your first wave.
