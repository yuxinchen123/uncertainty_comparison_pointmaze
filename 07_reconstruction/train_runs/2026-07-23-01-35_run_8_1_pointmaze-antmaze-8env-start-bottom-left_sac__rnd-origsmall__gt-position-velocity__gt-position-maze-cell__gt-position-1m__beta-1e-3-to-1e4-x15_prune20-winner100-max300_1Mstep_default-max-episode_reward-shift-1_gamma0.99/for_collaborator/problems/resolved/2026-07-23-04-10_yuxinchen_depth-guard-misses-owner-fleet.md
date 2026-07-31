# Depth guard in launch_workers_collaborator.sh does not count the owner fleet (harmless today, matters at the tail)

- Reporter: yuxinchen, 2026-07-23 ~04:10 (before my first wave; smoke test in progress).
- Observation: `launch_workers_collaborator.sh` computes live worker CPUs with
  `awk -v re="^(r81|gyzoxeup|vbdcjdtp|jhiznywz)"` over `squeue -h -t R -o "%C %j"`, i.e. it expects
  the OWNER fleet to be named `r81*`. The owner fleet actually running now is named `pmam-*`
  (`pmam-w30-N`, `pmam-w14-N`, `pmam-wg8*`), so the guard printed
  `live-worker-cpus(all submitters)=0` while ~200+ owner worker CPUs were live.
- Impact: none today (pending = 91,861, so `remaining-unclaimed` is huge either way). At the SWEEP
  TAIL the guard will over-estimate remaining work and my launcher/monitor could submit worker jobs
  with no work left — they would exit quickly on an empty queue, wasting a little scheduling, not
  corrupting anything.
- Suggested owner fix (owner-owned file, so I did not touch it): change `r81` in the launcher's
  depth-guard regex to `pmam` (or whatever the fleet base names are), or add the owner prefixes to
  `packet_env.sh` and build the regex from there.
- No action needed on my side; my first wave proceeds under the normal caps.

## Resolution (owner sl5nw, 2026-07-23 04:08:53)

Confirmed and fixed: the depth-guard regex now matches the real owner fleet base name `pmam`
(`^(pmam|<collab prefixes>)`), so live owner worker CPUs are counted. Verified with bash -n and a
live squeue count. Thank you for the precise report — no collaborator-side action needed.
