#!/bin/bash
# Collaborator launcher for Convergence run 1: adds worker jobs to the EXISTING sweep under YOUR
# user's own Slurm caps. It never builds a queue, never touches the owner's controller/monitor,
# never uses a reservation. It self-gates so it is a clean NO-OP once the sweep is (almost) done.
#   preview :  DRY=1 bash launch_workers_collaborator.sh     (prints the sbatch plan, submits nothing)
#   submit  :  bash launch_workers_collaborator.sh
# Conventions: /p/rlprojects/.claude/skills/submit-cpu-sweep/SKILL.md (the one CPU submission rule).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/packet_env.sh"
DRY="${DRY:-0}"

# ---- guards -------------------------------------------------------------------------------
# the OWNER submits through slurm/launch_queue.sh (reservation, id file); this script is for
# everyone ELSE adding capacity under their own caps (DRY=1 preview is allowed for the owner)
[[ "$USER" == "sl5nw" && "$DRY" != "1" ]] && { echo "you are the sweep owner — use slurm/launch_queue.sh instead (DRY=1 preview is allowed)"; exit 3; }
# a finished sweep needs no workers (the owner's monitor writes this sentinel when the queue drains)
[[ -f "$RUN_DIR/optuna/SWEEP_COMPLETE" ]] && { echo "sweep complete — nothing to do"; exit 0; }
PENDING_DIR="$RUN_DIR/queue/$SWEEP_ID/pending"
[[ -d "$PENDING_DIR" ]] || { echo "ERROR: no queue at $PENDING_DIR"; exit 1; }

# ---- queue-depth guard: never submit more worker slots than there is work left --------------
# live worker slots across ALL submitters: the owner's fleet (OWNER_JOBNAME) plus every
# collaborator's per-partition prefixes (they all source THIS packet, so all share these names).
pending=$(ls "$PENDING_DIR" | wc -l)
SLOTS_RE="$(all_slots_regex)"
slots=$(squeue -h -t R -o "%C %j" 2>/dev/null | awk -v re="$SLOTS_RE" '$2 ~ re {s+=$1} END{print s+0}')
remaining=$(( pending - slots ))
echo "[depth] pending=$pending  live-worker-slots=$slots  remaining-unclaimed=$remaining"
if (( remaining <= 0 )); then
  echo "[depth] enough workers are already running for the remaining queue — not submitting"
  echo "[depth] (this sweep is tiny and may finish within the hour — a no-op here means you are done)"
  exit 0
fi

# ---- my own usage per partition (each user has independent caps: cpu 400 / gpu 400 / nolim 80) --
my_usage() { squeue -u "$USER" -h -o "%C %P" | awk -v p="$1" '$2==p{s+=$1} END{print s+0}'; }
USED_CPU=$(my_usage cpu); USED_GPU=$(my_usage gpu); USED_NOLIM=$(my_usage nolim)
echo "[caps] my usage: cpu=$USED_CPU/400  gpu=$USED_GPU/400  nolim=$USED_NOLIM/80 (16-CPU headroom kept per pool)"

mkdir -p "$LOGDIR"
submitted=0
budget=$remaining   # never plan more worker slots than unclaimed queue entries

sub() {  # sub <script> <ntasks> <mem-per-cpu> <jobname> <extra sbatch args...>
  local script="$1" ntasks="$2" mem="$3" name="$4"; shift 4
  (( budget <= 0 )) && { echo "[depth] budget used up — skipping $name $*"; return; }
  local cmd=(sbatch --job-name="$name" --chdir="$PROJ_DIR" --nodes=1 --ntasks="$ntasks" \
        --cpus-per-task=1 --ntasks-per-core=2 --mem-per-cpu="$mem" \
        --output="$LOGDIR/${name}_%j.log" --time=4-00:00:00 \
        --comment="${SWEEP_ID}_${USER}" --export=ALL,SWEEP_ID="$SWEEP_ID" "$@" "$HERE/$script")
  if [[ "$DRY" == "1" ]]; then echo "[dry] ${cmd[*]}"; budget=$((budget-ntasks)); return; fi
  local out id
  out=$("${cmd[@]}")
  id=$(grep -oP '[0-9]+$' <<< "$out")
  # flock: the launcher and monitor_collaborator.sh may both append this file
  [[ -n "$id" ]] && flock "$IDFILE.lock" bash -c "echo $id >> '$IDFILE'"
  echo "[sub] $name ntasks=$ntasks -> ${id:-FAILED}  $*"
  budget=$((budget-ntasks)); submitted=$((submitted+1))
  if (( submitted % 10 == 0 )); then echo "[throttle] $submitted submitted; sleep 30"; sleep 30; fi
}

idle_threads() { sinfo -n "$1" -h -o "%C" 2>/dev/null | cut -d/ -f2; }

# ---- bucket 1: open cpu partition (cumulative ceiling TARGET_CPU, 32x1 no-nodelist) ----------
# cap = 400 - 16 headroom - my current cpu usage; room = TARGET_CPU - my usage; take the smaller.
cap=$(( 400 - 16 - USED_CPU )); room=$(( TARGET_CPU - USED_CPU ))
(( room > cap )) && room=$cap; (( room < 0 )) && room=0
njobs=$(( room / 32 ))
echo "[cpu] target=$TARGET_CPU room=$room -> $njobs x 32x1 jobs (no nodelist)"
for i in $(seq "$njobs" 2>/dev/null); do sub worker_32x1_collab.slurm 32 2G "$(prefix_for_partition cpu)" --partition=cpu; done

# ---- bucket 2: gpu partition, LOWEST-capability GPU nodes first ------------------------------
# The eligible classes and their capability_rank come live from the shared node catalog (single
# source of truth); CPU-only jobs fill the least capable GPU nodes first so better GPUs' CPUs stay
# free for real GPU jobs. Only clearly-lower-tier families are eligible (include-list, never
# --exclude, so a future high-end GPU can never auto-qualify).
cap=$(( 400 - 16 - USED_GPU )); room=$(( TARGET_GPU - USED_GPU ))
(( room > cap )) && room=$cap; (( room < 0 )) && room=0
echo "[gpu] target=$TARGET_GPU room=$room, nodes in ascending GPU capability:"
CATALOG="/p/rlprojects/.claude/skills/submit-gpu-sweep/server_introduction/server_introduction.json"
# emits "node threads" lines, least-capable class first (descending capability_rank)
gpu_nodes=$("$PY" - "$CATALOG" << 'PYEOF'
import json, sys
ALLOW = {"tesla_p100", "gtx_1080_ti", "titan_xp", "titan_x", "rtx_2080_ti", "quadro_rtx_4000"}
cat = json.load(open(sys.argv[1]))
assert cat.get("schema_version", 0) >= 2, "server_introduction.json schema too old"
rows = []
for c in cat["classes"]:
    fam = (c.get("gres_type") or c.get("family") or "").lower()
    if any(a in fam for a in ALLOW) and c.get("partition") == "gpu":
        for node in c["nodes"]:
            rows.append((c["capability_rank"], node, c["cpu_threads"]))
for rank, node, threads in sorted(rows, key=lambda r: (-r[0], r[1])):
    print(node, threads)
PYEOF
) || { echo "[gpu] catalog read failed — skipping the gpu bucket"; gpu_nodes=""; }
while read -r node threads; do
  [[ -z "${node:-}" ]] && continue
  (( room < 14 )) && break
  idle=$(idle_threads "$node"); [[ -z "$idle" ]] && continue
  if (( threads >= 64 )); then
    # >=32-core class: one 32x1 job when the node is free enough
    if (( idle >= 32 && room >= 32 )); then
      sub worker_32x1_collab.slurm 32 2G "$(prefix_for_partition gpu)" --partition=gpu --nodelist="$node" --gpus-per-node=0
      room=$((room-32))
    fi
  else
    # 16-core/32-thread class: the 16x1 + 14x1 pair (ntasks capped by physical cores); 64G-total
    # nodes (ai01-06) take 2G + 1900M, everything else 2G + 2G
    if (( idle >= 30 && room >= 30 )); then
      mem14=2G; [[ "$node" =~ ^ai0[1-6]$ ]] && mem14=1900M
      sub worker_16x1_collab.slurm 16 2G "$(prefix_for_partition gpu)" --partition=gpu --nodelist="$node" --gpus-per-node=0
      sub worker_16x1_collab.slurm 14 "$mem14" "$(prefix_for_partition gpu)" --partition=gpu --nodelist="$node" --gpus-per-node=0
      room=$((room-30))
    fi
  fi
done <<< "$gpu_nodes"

# ---- bucket 3: nolim (cumulative ceiling TARGET_NOLIM of the 80 cap) -------------------------
cap=$(( 80 - 16 - USED_NOLIM )); room=$(( TARGET_NOLIM - USED_NOLIM ))
(( room > cap )) && room=$cap; (( room < 0 )) && room=0
echo "[nolim] target=$TARGET_NOLIM room=$room"
if (( room >= 32 )) && sinfo -p nolim -N -h -o "%C" | awk -F/ '$2>=32{f=1} END{exit !f}'; then
  sub worker_32x1_collab.slurm 32 2G "$(prefix_for_partition nolim)" --partition=nolim
  room=$((room-32))
fi
if (( room >= 16 )) && sinfo -p nolim -N -h -o "%C" | awk -F/ '$2>=16{f=1} END{exit !f}'; then
  sub worker_16x1_collab.slurm 16 2G "$(prefix_for_partition nolim)" --partition=nolim
fi

echo "[done] submitted $submitted jobs; ids in $IDFILE"
echo "[next] verify the first job of each shape: sacct -j <id> --format=JobID,JobName,AllocCPUS,ReqMem,NodeList"
echo "[next] keep the passive monitor running: nohup bash $HERE/monitor_collaborator.sh >> $LOGDIR/monitor_collaborator.log 2>&1 &"
