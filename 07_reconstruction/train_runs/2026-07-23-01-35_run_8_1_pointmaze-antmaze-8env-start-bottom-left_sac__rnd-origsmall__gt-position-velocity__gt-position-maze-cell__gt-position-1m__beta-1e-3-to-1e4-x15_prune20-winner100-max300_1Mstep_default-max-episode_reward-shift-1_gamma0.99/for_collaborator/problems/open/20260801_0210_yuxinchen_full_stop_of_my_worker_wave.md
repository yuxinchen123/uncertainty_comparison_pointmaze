# 2026-08-01 02:10 — yuxinchen: full stop of my collaborator worker wave (not a failure)

Not a bug — an operator action notice.

- On user (yuxinchen) order 2026-08-01 ~02:06, I cancelled ALL 44 of my worker jobs
  (prefixes gyzoxeup / jhiznywz / vbdcjdtp; ids taken ONLY from
  `for_collaborator/submitted_jobids_2026-07-23-02-05_pm-am-run1_yuxinchen.txt` after a
  refresh). Reason: my capacity moves to the new run 8.1.2
  (`train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_*`).
- The killed workers were mid-run; their claimed markers in `queue/<sweep>/running/` will need
  the owner-side requeue as usual after worker kills.
- To stop my passive monitor's auto-refill I set `TARGET_CPU/TARGET_NOLIM/TARGET_GPU=0` in
  `for_collaborator/packet_env.sh` (the packet's documented off-switch). NOTE for other
  collaborators: if anyone else is refilling from this same packet, these zeroed targets stop
  their refills too — restore the values if that's not wanted (my jobs stay cancelled either way).
- My detached `monitor_collaborator.sh` is still looping somewhere on a portal node I cannot
  ssh into (password-only auth between portals). It is harmless now: with targets 0 the
  launcher's room clamps to 0 and it submits nothing; it will exit on its own at SWEEP_COMPLETE.
