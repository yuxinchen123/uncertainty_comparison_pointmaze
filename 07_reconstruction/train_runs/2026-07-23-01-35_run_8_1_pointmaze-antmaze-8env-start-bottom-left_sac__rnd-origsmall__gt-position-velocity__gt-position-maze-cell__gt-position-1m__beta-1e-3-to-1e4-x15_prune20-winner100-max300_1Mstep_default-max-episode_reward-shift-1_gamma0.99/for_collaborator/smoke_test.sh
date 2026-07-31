#!/bin/bash
# Smoke test (run under YOUR uid, in minutes, WITHOUT touching the real queue). It proves three
# things before you submit any worker job:
#   1. non-mutating permission probes: you can read/enter/write the queue state dirs + data dir + log
#      dir and read a pending marker (this catches an owner-side permission regression FIRST);
#   2. >= 5 short canary runs of the REAL entry point (train.py, built from real pending configs via
#      the run's own worker.build_cmd), covering PointMaze + AntMaze and several algorithms, each with
#      a hard `timeout 1200` (< 20 min) and outputs isolated under smoke_data/<user>/<timestamp>/ —
#      the queue is NEVER claimed or renamed (configs are read-copied, not moved);
#   3. every canary finished COMPLETED and wrote a JSON with "completed": true.
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
# pending/running/done/failed: the worker renames markers through these -> need read+enter+write.
for d in pending running done failed; do
  { test -r "$Q/$d" && test -x "$Q/$d" && test -w "$Q/$d"; } \
    || { echo "  FAIL: need r-w-x on $Q/$d"; fail=1; }
done
# pruned/ is owner-only (a collaborator never writes there) -> read+enter is enough
{ test -r "$Q/pruned" && test -x "$Q/pruned"; } || echo "  note: $Q/pruned not readable (owner-only; harmless)"
# workers create data/<sweep>/local under data/ -> need write there
test -w "$RUN_DIR/data" || { echo "  FAIL: need w on $RUN_DIR/data (worker JSON output)"; fail=1; }
test -w "$LOGDIR"       || { echo "  FAIL: need w on $LOGDIR"; fail=1; }
# read exactly one pending marker (read-only; the marker stays in pending/)
one=$(ls "$Q/pending" 2>/dev/null | head -1)
{ [[ -n "$one" ]] && test -r "$Q/pending/$one"; } || { echo "  FAIL: cannot read a pending marker"; fail=1; }
if (( fail )); then
  echo "  permission probes FAILED — do NOT submit; tell the owner (this is an owner-side regression)."
  exit 1
fi
echo "  OK: queue state dirs, data dir, log dir writable; a pending marker is readable."

echo "=== [2/3] >= 5 short canary runs of train.py (isolated under $SMOKE_OUT) ==="
export RUN_DIR PROJ_DIR SWEEP_ID SMOKE_OUT
"$PY" - << 'PYEOF'
import json, os, subprocess, sys, time, glob
# reuse the run's own worker module for an EXACT train.py argv (env_setup + device override)
RUN = os.environ["RUN_DIR"]; SMOKE_OUT = os.environ["SMOKE_OUT"]
os.environ["WORKER_DEVICE"] = "cpu"           # collaborators run CPU-only
sys.path.insert(0, os.path.join(RUN, "slurm"))
import worker
worker.DATA = SMOKE_OUT                        # redirect train.py's --local_log_dir to the isolated dir
PENDING = os.path.join(RUN, "queue", os.environ["SWEEP_ID"], "pending")

# pick >= 5 markers spanning distinct (env-family, algorithm) profiles for coverage
# before: pending names like "01234_of_92400_AntMaze_Large-v5_rnd_next_state_b0.1_seed3.json"
# after : one representative marker per distinct (PointMaze|AntMaze, algorithm) profile, up to 6
def profile(name):
    fam = "AntMaze" if "AntMaze" in name else ("PointMaze" if "PointMaze" in name else "other")
    for a in ("no_exploration", "rnd_next_state", "gt_position_velocity",
              "gt_position_maze_cell", "gt_position_1m"):
        if a in name:
            return (fam, a)
    return (fam, "unknown")

# COLD-END, READ-AT-PICK selection (owner fix after yuxinchen's 2026-07-23-04-25 report): the live
# fleet claims markers in id order from the LOW end, so pick from the HIGH-id end and read each
# marker's JSON immediately; a marker claimed between listing and open simply falls through to the
# next candidate. The queue is never renamed by this script.
names = sorted(os.listdir(PENDING), reverse=True)


def read_marker(nm):
    """Return the marker's parsed JSON, or None if a live worker claimed it in this instant."""
    try:
        with open(os.path.join(PENDING, nm)) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


picked, seen = [], set()
for nm in names:
    p = profile(nm)
    if p in seen:
        continue
    cfg = read_marker(nm)
    if cfg is None:
        continue
    seen.add(p); picked.append((nm, cfg))
for nm in names:                               # top up to >= 5 if fewer distinct profiles exist
    if len(picked) >= 5: break
    if any(nm == pn for pn, _ in picked): continue
    cfg = read_marker(nm)
    if cfg is not None:
        picked.append((nm, cfg))
picked = picked[:6]
print(f"  picked {len(picked)} canary configs: " + ", ".join(profile(n)[0] + "/" + profile(n)[1] for n, _ in picked))

results = []
for nm, cfg in picked:
    # shrink to a fast canary: a few thousand steps, one eval, so it finishes well under the 1200s cap
    cfg["fixed"]["total_timesteps"] = 2000
    cfg["fixed"]["eval_freq"] = 1000
    cfg["fixed"]["n_eval_episodes"] = 2
    argv = ["timeout", "1200"] + worker.build_cmd(cfg)   # hard per-run 20-min timeout
    t0 = time.time()
    rc = subprocess.call(argv, cwd=worker.PROJ)
    dt = time.time() - t0
    # verify a JSON for THIS run_id was written with completed: true
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
    print(f"  [{'OK ' if ok else 'BAD'}] {profile(nm)[0]}/{profile(nm)[1]} rc={rc} in {dt:.0f}s completed={done}")

n_ok = sum(results)
print(f"  canary result: {n_ok}/{len(results)} completed with completed:true")
sys.exit(0 if (n_ok == len(results) and len(results) >= 5) else 2)
PYEOF
canary_rc=$?

echo "=== [3/3] checklist ==="
echo "  - canary runs COMPLETED with completed:true : $([[ $canary_rc -eq 0 ]] && echo PASS || echo FAIL)"
# The old "pending count unchanged" check is GONE (owner fix, 2026-07-23): a live claiming fleet
# moves pending markers constantly, so the count can never hold. This script performs no renames
# at all, so queue safety is by construction; only the smoke's own outputs are checked.
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
