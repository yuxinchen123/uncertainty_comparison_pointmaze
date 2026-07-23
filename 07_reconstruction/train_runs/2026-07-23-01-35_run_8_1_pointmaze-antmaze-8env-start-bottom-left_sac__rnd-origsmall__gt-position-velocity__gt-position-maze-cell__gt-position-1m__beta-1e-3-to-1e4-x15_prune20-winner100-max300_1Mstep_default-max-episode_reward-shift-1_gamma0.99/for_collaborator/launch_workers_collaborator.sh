#!/bin/bash
# Collaborator launcher for Run 8.1: adds worker jobs to the EXISTING sweep under YOUR user's own
# Slurm caps. It never builds a queue, never prunes, never requeues, never uses a reservation, and
# never submits a GPU-using job. Buckets (open partitions of your own pools only):
#   1. cpu   partition                       30x1 (remainder 14x1)
#   2. gpu   partition, CPU-ONLY, least-capable GPU nodes first (--gpus-per-node=0)   30x1 / 14x1+12x1
#   3. nolim partition                       30x1 (remainder 14x1)
#   preview :  DRY=1 bash launch_workers_collaborator.sh     (prints the sbatch plan, submits nothing)
#   submit  :  bash launch_workers_collaborator.sh
# Conventions: /p/rlprojects/.claude/skills/submit-cpu-sweep/SKILL.md (the one CPU submission rule).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"     # interactive script: dirname "$0" is fine (NOT an sbatch'ed .slurm)
source "$HERE/packet_env.sh"
DRY="${DRY:-0}"

# ---- guards -------------------------------------------------------------------------------
# the OWNER submits through slurm/launch_queue.sh (reservation, prune controller); this script is for
# everyone else adding capacity under their own caps
[[ "$USER" == "sl5nw" && "$DRY" != "1" ]] && { echo "you are the sweep owner — use slurm/launch_queue.sh instead (DRY=1 preview is allowed)"; exit 3; }
# a finished sweep needs no workers (the owner's monitor_loop.sh touches this sentinel when the queue drains)
[[ -f "$COMPLETE_SENTINEL" ]] && { echo "sweep complete ($COMPLETE_SENTINEL exists) — nothing to do"; exit 0; }
PENDING_DIR="$RUN_DIR/queue/$SWEEP_ID/pending"
[[ -d "$PENDING_DIR" ]] || { echo "ERROR: no queue at $PENDING_DIR"; exit 1; }

# ---- maintenance-aware walltime -------------------------------------------------------------
# --time = min(4 days, next-maintenance-start - now - 30 min). A 1M-step run is 9-25 h, so a claiming
# worker drains several runs before its walltime; on the OPEN partitions a job whose walltime would
# outlast the next maintenance is DEFERRED to after it (never starts now), so the cap is mandatory.
now=$(date +%s)
TLIM_SECS=$(( 4*24*3600 ))   # 4-day floor (cpu partition limit; also <= nolim's 20-day limit)
maint=$(scontrol show reservation -o 2>/dev/null | grep -i 'MAINT' | grep -oP 'StartTime=\K\S+' \
        | while read -r t; do date -d "$t" +%s 2>/dev/null; done | sort -n | awk -v n="$now" '$1>n{print; exit}')
if [[ -n "${maint:-}" ]]; then
  gap=$(( maint - now - 1800 ))
  (( gap < TLIM_SECS )) && TLIM_SECS=$gap
fi
if (( TLIM_SECS < 3600 )); then
  echo "[time] next maintenance is under ~1 h away — not submitting (a job would be deferred past it)"
  exit 0
fi
# format seconds -> D-HH:MM:SS
d=$(( TLIM_SECS/86400 )); r=$(( TLIM_SECS%86400 )); h=$(( r/3600 )); m=$(( (r%3600)/60 )); s=$(( r%60 ))
TLIM=$(printf '%d-%02d:%02d:%02d' "$d" "$h" "$m" "$s")
echo "[time] --time=$TLIM"

# ---- queue-depth guard: never submit more worker slots than there is work left --------------
# live worker CPUs across ALL submitters: the owner fleet (job names r81*) + every collaborator using
# THIS packet (the three random prefixes baked into packet_env.sh). Never over-provision the queue.
pending=$(ls "$PENDING_DIR" | wc -l)
slots=$(squeue -h -t R -o "%C %j" 2>/dev/null \
        | awk -v re="^(r81|$PREFIX_CPU|$PREFIX_NOLIM|$PREFIX_GPU)" '$2 ~ re {s+=$1} END{print s+0}')
remaining=$(( pending - slots ))
echo "[depth] pending=$pending  live-worker-cpus(all submitters)=$slots  remaining-unclaimed=$remaining"
if (( remaining <= 0 )); then
  echo "[depth] enough workers are already running for the remaining queue — not submitting"
  exit 0
fi

# ---- my own usage per partition (each user has independent caps: cpu 400 / gpu 400 / nolim 80) -----
my_usage() { squeue -u "$USER" -h -o "%C %P" | awk -v p="$1" '$2==p{s+=$1} END{print s+0}'; }
USED_CPU=$(my_usage cpu); USED_GPU=$(my_usage gpu); USED_NOLIM=$(my_usage nolim)
echo "[caps] my usage: cpu=$USED_CPU/400  gpu=$USED_GPU/400  nolim=$USED_NOLIM/80 (16-CPU headroom kept per pool)"

mkdir -p "$LOGDIR"
submitted=0
budget=$remaining   # never plan more worker CPUs than unclaimed queue entries

# sub <script> <ntasks> <mem-per-cpu> <partition> [extra sbatch args...]
# The job name is the random prefix for <partition>; --ntasks OVERRIDES the script's #SBATCH default
# (so one wrapper serves any fragment shape); the id is captured under a flock (the launcher and the
# monitor may both append IDFILE) and tagged --comment=<sweep>_<user> for requeue recovery.
sub() {
  local script="$1" ntasks="$2" mem="$3" part="$4"; shift 4
  (( budget <= 0 )) && { echo "[depth] budget used up — skipping $script ntasks=$ntasks part=$part"; return; }
  local name; name="$(prefix_for_partition "$part")"
  local cmd=(sbatch --job-name="$name" --chdir="$PROJ_DIR" --partition="$part" \
        --nodes=1 --ntasks="$ntasks" --cpus-per-task=1 --ntasks-per-core=2 --mem-per-cpu="$mem" \
        --output="$LOGDIR/${name}_%j.log" --time="$TLIM" \
        --comment="${SWEEP_ID}_${USER}" --export=ALL,SWEEP_ID="$SWEEP_ID" "$@" "$HERE/$script")
  if [[ "$DRY" == "1" ]]; then echo "[dry] ${cmd[*]}"; budget=$((budget-ntasks)); return; fi
  local out id
  out=$("${cmd[@]}")
  id=$(grep -oP '[0-9]+$' <<< "$out")
  [[ -n "$id" ]] && flock "$IDFILE.lock" bash -c "echo $id >> '$IDFILE'"
  echo "[sub] $name $script ntasks=$ntasks part=$part -> ${id:-FAILED}  $*"
  budget=$((budget-ntasks)); submitted=$((submitted+1))
  if (( submitted % 10 == 0 )); then echo "[throttle] $submitted submitted; sleep 30"; sleep 30; fi
}

# free idle threads on a node (sinfo %C = A/I/O/T CPUs, where CPU=hardware thread here)
idle_threads() { sinfo -n "$1" -h -o "%C" 2>/dev/null | cut -d/ -f2; }
# does partition <p> have any node with >= <n> idle threads?
part_has_idle() { sinfo -p "$1" -N -h -o "%C" | awk -F/ -v n="$2" '$2>=n{f=1} END{exit !f}'; }

# ---- bucket 1: open cpu partition (cumulative ceiling TARGET_CPU, 30x1 + 14x1 remainder) -----------
# TARGET_* are CUMULATIVE ceilings (usage never exceeds the target), not per-cycle additions, so the
# monitor's auto-refill cannot creep the pool up to cap-16 over many cycles.
cap=$(( 400 - 16 - USED_CPU )); room=$(( TARGET_CPU - USED_CPU ))
(( room > cap )) && room=$cap; (( room < 0 )) && room=0
njobs=$(( room / 30 ))
echo "[cpu] target=$TARGET_CPU room=$room -> $njobs x 30x1 (no nodelist)"
for i in $(seq "$njobs"); do sub worker_30x1_collab.slurm 30 2G cpu; room=$((room-30)); done
if (( room >= 14 )) && part_has_idle cpu 14; then sub worker_14x1_collab.slurm 14 2G cpu; room=$((room-14)); fi

# ---- bucket 2: gpu partition, CPU-ONLY, LOWEST-capability GPU nodes first ---------------------------
# The eligible classes and their capability_rank come live from the shared node catalog (single source
# of truth); CPU-only jobs fill the LEAST capable GPU nodes first so better GPUs' CPUs stay free for
# real GPU jobs. Only clearly-lower-tier families are eligible (the ALLOW include-list). These are
# CPU-ONLY jobs: --gpus-per-node=0 and WORKER_DEVICE=cpu (set in the .slurm) — never a GPU job.
cap=$(( 400 - 16 - USED_GPU )); room=$(( TARGET_GPU - USED_GPU ))
(( room > cap )) && room=$cap; (( room < 0 )) && room=0
echo "[gpu] target=$TARGET_GPU room=$room (CPU-only jobs on gpu nodes, ascending GPU capability):"
CATALOG="/p/rlprojects/.claude/skills/submit-gpu-sweep/server_introduction/server_introduction.json"
# emits "node threads" lines, least-capable class first (descending capability_rank)
gpu_nodes=$("$PY" - "$CATALOG" << 'PYEOF'
import json, sys
# include-list of clearly-lower-tier GPU families whose CPUs we fill first (least capable)
ALLOW = {"tesla_p100", "gtx_1080_ti", "titan_xp", "titan_x", "rtx_2080_ti", "quadro_rtx_4000"}
cat = json.load(open(sys.argv[1]))
assert cat.get("schema_version", 0) >= 2, "server_introduction.json schema too old"
rows = []
# collect (capability_rank, node, cpu_threads) for every allowed gpu-partition class
for c in cat["classes"]:
    fam = (c.get("gres_type") or c.get("family") or "").lower()
    if any(a in fam for a in ALLOW) and c.get("partition") == "gpu":
        for node in c["nodes"]:
            rows.append((c["capability_rank"], node, c["cpu_threads"]))
# least capable first (descending capability_rank -> higher rank = less capable in this catalog)
for rank, node, threads in sorted(rows, key=lambda r: (-r[0], r[1])):
    print(node, threads)
PYEOF
) || { echo "[gpu] catalog read failed — skipping the gpu bucket"; gpu_nodes=""; }
while read -r node threads; do
  [[ -z "${node:-}" ]] && continue
  (( room < 14 )) && break
  idle=$(idle_threads "$node"); [[ -z "$idle" ]] && continue
  if (( threads >= 64 )); then
    # >=32-physical-core class: one 30x1 job (2 cores of headroom) when the node is free enough
    if (( idle >= 30 && room >= 30 )); then
      sub worker_30x1_collab.slurm 30 2G gpu --nodelist="$node" --gpus-per-node=0
      room=$((room-30))
    fi
  else
    # 16-physical-core / 32-thread class: 14x1 + 12x1 (16x1 is rejected on this class; the pair packs
    # 26 threads / 13 cores, leaving headroom)
    if (( idle >= 26 && room >= 26 )); then
      sub worker_14x1_collab.slurm 14 2G gpu --nodelist="$node" --gpus-per-node=0
      sub worker_12x1_collab.slurm 12 2G gpu --nodelist="$node" --gpus-per-node=0
      room=$((room-26))
    fi
  fi
done <<< "$gpu_nodes"

# ---- bucket 3: nolim (cumulative ceiling TARGET_NOLIM of the 80 cap, 30x1 + 14x1 remainder) ---------
cap=$(( 80 - 16 - USED_NOLIM )); room=$(( TARGET_NOLIM - USED_NOLIM ))
(( room > cap )) && room=$cap; (( room < 0 )) && room=0
echo "[nolim] target=$TARGET_NOLIM room=$room"
while (( room >= 30 )) && part_has_idle nolim 30; do sub worker_30x1_collab.slurm 30 2G nolim; room=$((room-30)); done
if (( room >= 14 )) && part_has_idle nolim 14; then sub worker_14x1_collab.slurm 14 2G nolim; room=$((room-14)); fi

echo "[done] submitted $submitted jobs; ids in $IDFILE"
echo "[next] verify the first job of each shape: sacct -j <id> --format=JobID,JobName,AllocCPUS,ReqMem,NodeList"
echo "[next]   AllocCPUS must EQUAL ntasks (30/14/12); if it is doubled, --ntasks-per-core=2 was lost — report a problem"
echo "[next] keep the passive monitor running: nohup bash $HERE/monitor_collaborator.sh >> $LOGDIR/monitor_collaborator.log 2>&1 &"
