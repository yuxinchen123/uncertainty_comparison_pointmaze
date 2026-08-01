#!/bin/bash
# Smoke test (run under YOUR uid, in minutes, WITHOUT touching the real queue). It proves three
# things before you submit any worker job:
#   1. non-mutating permission probes: you can read/enter/write BOTH pending pools, the other queue
#      state dirs, the data dir and the log dir, and read a marker from each pool (this catches an
#      owner-side permission regression FIRST);
#   2. >= 5 short canary runs of the REAL entry point (train.py, built from real queue configs via
#      the run's own worker.build_cmd), covering both environments, all four new algorithm arms and
#      the task-R baseline, each with a hard `timeout 1200` (< 20 min) and outputs isolated under
#      smoke_data/<user>/<timestamp>/;
#   3. every canary finished COMPLETED and wrote a JSON with "completed": true.
#
# HOW THIS STAYS CLEAR OF THE LIVE QUEUE (the two run-8.1 traps, already fixed there — do not undo):
#   - it never renames a marker: it never calls worker.claim(), it only READS marker JSON, so no
#     queue entry ever changes state because of the smoke test;
#   - it picks from the COLD (high-id) end of each pool and reads each marker AT PICK TIME. Live
#     workers claim from the LOW-id end (worker.claim() picks randomly among the first 32 sorted
#     names), and the high ids belong to the last seed block, so a smoke pick is never in the hot
#     zone; a marker that vanishes between listing and open simply falls through to the next
#     candidate (run-8.1 problems/resolved/2026-07-23-04-25);
#   - it writes ONLY under smoke_data/<user>/<timestamp>/ — never into data/<sweep_id>/;
#   - there is no "pending count unchanged" check: a live claiming fleet moves markers constantly,
#     so such a check can never hold and says nothing about this script (which renames nothing).
# Only after all checks pass do you move to the normal wave (launch_workers_collaborator.sh + monitor).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/packet_env.sh"
[[ "$USER" == "sl5nw" ]] && { echo "you are the sweep owner — the owner fleet already ran its checks"; exit 3; }
[[ -f "$COMPLETE_SENTINEL" ]] && { echo "sweep complete — nothing to test against"; exit 0; }

Q="$RUN_DIR/queue/$SWEEP_ID"
STAMP=$(date '+%Y-%m-%d-%H-%M-%S')
SMOKE_OUT="$FC/smoke_data/${USER}/${STAMP}"
mkdir -p "$SMOKE_OUT" "$LOGDIR"
fail=0

echo "=== [1/3] non-mutating permission probes (no submission, no queue change) ==="
# both pending pools + running/done/failed: a worker renames markers through these -> need r-w-x.
for d in pending_1m pending_10m running done failed; do
  { test -r "$Q/$d" && test -x "$Q/$d" && test -w "$Q/$d"; } \
    || { echo "  FAIL: need r-w-x on $Q/$d"; fail=1; }
done
# pruned/ is owner-only (a collaborator never writes there) -> read+enter is enough
{ test -r "$Q/pruned" && test -x "$Q/pruned"; } || echo "  note: $Q/pruned not readable (owner-only; harmless)"
# workers create data/<sweep>/local under data/ -> need write there
test -w "$RUN_DIR/data" || { echo "  FAIL: need w on $RUN_DIR/data (worker JSON output)"; fail=1; }
test -w "$LOGDIR"       || { echo "  FAIL: need w on $LOGDIR"; fail=1; }
# read exactly one marker from EACH pool (read-only; both markers stay where they are)
for d in pending_1m pending_10m; do
  one=$(ls "$Q/$d" 2>/dev/null | tail -1)
  { [[ -n "$one" ]] && test -r "$Q/$d/$one"; } || { echo "  FAIL: cannot read a marker in $Q/$d"; fail=1; }
done
if (( fail )); then
  echo "  permission probes FAILED — do NOT submit; tell the owner (this is an owner-side regression)."
  exit 1
fi
echo "  OK: both pools + running/done/failed writable, data dir and log dir writable, markers readable."

echo "=== [2/3] >= 5 short canary runs of train.py (isolated under $SMOKE_OUT) ==="
export RUN_DIR PROJ_DIR SWEEP_ID SMOKE_OUT
export WORKER_DEVICE=cpu          # collaborators run CPU-only
export WORKER_POOLS="pending_1m"  # only satisfies worker.py's import; claim() is never called
"$PY" - << 'PYEOF'
import glob
import json
import os
import subprocess
import sys
import time

# reuse the run's own worker module for an EXACT train.py argv (params + fixed + device override)
RUN = os.environ["RUN_DIR"]; SMOKE_OUT = os.environ["SMOKE_OUT"]
sys.path.insert(0, os.path.join(RUN, "slurm"))
import worker
worker.DATA = SMOKE_OUT                        # redirect train.py's --local_log_dir to the isolated dir
QUEUE = os.path.join(RUN, "queue", os.environ["SWEEP_ID"])
POOLS = ["pending_10m", "pending_1m"]          # 10M pool first so the task-R baseline is always covered

# marker names look like
#   before: "23999_of_24200_AntMaze_Medium-v5_alg2.3_lr0.01_b10000_seed99.json"  (pool pending_1m)
#           "24199_of_24200_AntMaze_Medium-v5_baseline_lr0.0001_b10000_seed99.json" (pool pending_10m)
#   after : profile ("Medium", "alg2.3") / ("Medium", "baseline") — one canary per distinct profile
ARMS = ("alg2.1", "alg2.2", "alg2.3", "alg1", "baseline")   # longest-first: "alg2.1" before "alg1"


def profile(name):
    """The (environment, algorithm-arm) pair a marker filename belongs to."""
    env = "UMaze" if "UMaze" in name else ("Medium" if "Medium" in name else "other")
    for a in ARMS:
        if f"_{a}_" in name:
            return (env, a)
    return (env, "unknown")


def read_marker(pool, nm):
    """Return the marker's parsed JSON, or None if a live worker claimed it in this instant."""
    try:
        with open(os.path.join(QUEUE, pool, nm)) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# COLD-END, READ-AT-PICK selection: the live fleet claims from the LOW-id end, so iterate each pool's
# listing reversed (highest ids = the last seed block) and read each candidate immediately.
# 8 picks = both environments' task-R baseline plus six (environment, arm) combinations of task S,
# so every new algorithm arm and both AntMaze environments are exercised at about 100 s per canary.
picked, seen = [], set()
for pool in POOLS:
    for nm in sorted(os.listdir(os.path.join(QUEUE, pool)), reverse=True):
        if len(picked) >= 8:
            break
        p = profile(nm)
        if p in seen:
            continue
        cfg = read_marker(pool, nm)
        if cfg is None:                        # claimed between listing and open: take the next one
            continue
        seen.add(p); picked.append((pool, nm, cfg))
print("  picked %d canary configs: %s" % (len(picked),
      ", ".join(f"{profile(n)[0]}/{profile(n)[1]}({p})" for p, n, _ in picked)))

results = []
for pool, nm, cfg in picked:
    # shrink to a fast canary: a few thousand steps, two evals, so it finishes well under the 1200s cap
    cfg["fixed"]["total_timesteps"] = 2000
    cfg["fixed"]["eval_freq"] = 1000
    cfg["fixed"]["n_eval_episodes"] = 2
    argv = ["timeout", "1200"] + worker.build_cmd(cfg)   # hard per-run 20-min timeout
    t0 = time.time()
    rc = subprocess.call(argv, cwd=worker.PROJ)
    dt = time.time() - t0
    # verify a JSON for THIS run_id was written into the isolated dir with completed: true
    done = False
    for jp in glob.glob(os.path.join(SMOKE_OUT, "local", "*.json")):
        try:
            r = json.load(open(jp))
        except Exception:
            continue
        if r.get("run_id") == cfg["run_id"] and r.get("completed") is True:
            done = True; break
    ok = (rc == 0 and done)
    results.append(ok)
    print(f"  [{'OK ' if ok else 'BAD'}] {profile(nm)[0]}/{profile(nm)[1]} ({pool}) "
          f"rc={rc} in {dt:.0f}s completed={done}")

n_ok = sum(results)
print(f"  canary result: {n_ok}/{len(results)} completed with completed:true")
sys.exit(0 if (n_ok == len(results) and len(results) >= 5) else 2)
PYEOF
canary_rc=$?

echo "=== [3/3] checklist ==="
echo "  - canary runs COMPLETED with completed:true : $([[ $canary_rc -eq 0 ]] && echo PASS || echo FAIL)"
echo "  - this script performed no queue renames (by construction; outputs isolated) : PASS"
echo "  - problems/open/ reports : $([[ -z "$(ls -A "$FC/problems/open" 2>/dev/null | grep -v '^.gitkeep$')" ]] && echo 'none' || echo 'WARNING: open reports present (informational — read them; not a smoke failure)')"
echo "  - isolated outputs under : $SMOKE_OUT/local/"
if [[ $canary_rc -eq 0 ]]; then
  echo "SMOKE PASSED — start the first wave: bash launch_workers_collaborator.sh, then the monitor loop."
  exit 0
else
  echo "SMOKE FAILED — do NOT submit a wave; tell the owner (do not debug shared state yourself)."
  exit 1
fi
