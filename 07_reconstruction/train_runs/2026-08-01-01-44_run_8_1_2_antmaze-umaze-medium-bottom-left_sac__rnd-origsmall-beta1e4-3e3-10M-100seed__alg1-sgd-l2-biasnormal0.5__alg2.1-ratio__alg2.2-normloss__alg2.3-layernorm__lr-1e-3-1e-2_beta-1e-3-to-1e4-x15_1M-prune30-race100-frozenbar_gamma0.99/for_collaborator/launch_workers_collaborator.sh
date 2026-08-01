#!/bin/bash
# Collaborator launcher for Run 8.1.2: adds worker jobs to the EXISTING sweep under YOUR user's own
# Slurm caps. It never builds a queue, never prunes, never requeues, never uses a reservation, and
# never submits a GPU-USING job (the run's task-R cuda jobs are owner-only). Buckets, in submission
# order (open partitions, least-contended first — submit-cpu-sweep section 8):
#   1. gpu    partition, CPU-ONLY (--gpus-per-node=0), least-capable GPU nodes first   -> pending_1m
#   2. gnolim partition, CPU-ONLY (--gpus-per-node=0), ai / jinx / titanx nodes        -> pending_10m
#                                                                                         then _1m
#   3. cpu    partition, unpinned                          30x1 (remainder 14x1)       -> pending_1m
#   4. nolim  partition, unpinned                          30x1 (remainder 14x1)       -> pending_1m
#   preview :  DRY=1 bash launch_workers_collaborator.sh     (prints the sbatch plan, submits nothing)
#   submit  :  bash launch_workers_collaborator.sh
# Conventions: /p/rlprojects/.claude/skills/submit-cpu-sweep/SKILL.md (the one CPU submission rule).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"     # interactive script: dirname "$0" is fine (NOT an sbatch'ed .slurm)
source "$HERE/packet_env.sh"
DRY="${DRY:-0}"

# ---- guards -------------------------------------------------------------------------------
# the OWNER submits through the run's own slurm/ scripts (reservation, cuda jobs, prune controller);
# this script is for everyone else adding capacity under their own caps
[[ "$USER" == "sl5nw" && "$DRY" != "1" ]] && { echo "you are the sweep owner — use the run's slurm/ launcher instead (DRY=1 preview is allowed)"; exit 3; }
# a finished sweep needs no workers (the owner's monitor_loop.slurm touches this sentinel)
[[ -f "$COMPLETE_SENTINEL" ]] && { echo "sweep complete ($COMPLETE_SENTINEL exists) — nothing to do"; exit 0; }
[[ -d "$POOL_1M"  ]] || { echo "ERROR: no task-S queue pool at $POOL_1M"; exit 1; }
[[ -d "$POOL_10M" ]] || { echo "ERROR: no task-R queue pool at $POOL_10M"; exit 1; }

# ---- maintenance-aware walltime ---------------------------------------------------------------
# Two limits, because the partition caps differ: cpu/gpu are 4-day partitions, nolim/gnolim 20-day.
# Both are additionally capped by the next maintenance window (a job whose walltime would outlast it
# is DEFERRED to after it and never starts now), leaving a 30-minute margin.
# Consequence for gnolim (deliberate, see the .slurm header): a gnolim job submitted BEFORE the
# maintenance gets a short walltime, so the run's walltime guard (>= 170 h for a cpu 10M claim)
# blocks the 10M pool and the job serves task S; a 20-day gnolim job submitted after the maintenance
# takes the 10M task-R runs first.
now=$(date +%s)
CAP_SHORT=$(( 4*24*3600 ))     # cpu / gpu partition limit
CAP_LONG=$(( 20*24*3600 ))     # nolim / gnolim partition limit
maint=$(scontrol show reservation -o 2>/dev/null | grep -i 'MAINT' | grep -oP 'StartTime=\K\S+' \
        | while read -r t; do date -d "$t" +%s 2>/dev/null; done | sort -n | awk -v n="$now" '$1>n{print; exit}')
if [[ -n "${maint:-}" ]]; then
  gap=$(( maint - now - 1800 ))
  (( gap < CAP_SHORT )) && CAP_SHORT=$gap
  (( gap < CAP_LONG  )) && CAP_LONG=$gap
  echo "[time] next maintenance starts $(date -d "@$maint" '+%Y-%m-%dT%H:%M:%S'); every job must end before it"
fi
if (( CAP_SHORT < 3600 )); then
  echo "[time] next maintenance is under ~1 h away — not submitting (a job would be deferred past it)"
  exit 0
fi
# format seconds -> D-HH:MM:SS
fmt_time() { local s=$1; printf '%d-%02d:%02d:%02d' $(( s/86400 )) $(( (s%86400)/3600 )) $(( (s%3600)/60 )) $(( s%60 )); }
TLIM_SHORT=$(fmt_time "$CAP_SHORT")   # cpu, gpu
TLIM_LONG=$(fmt_time "$CAP_LONG")     # nolim, gnolim
echo "[time] --time=$TLIM_SHORT (cpu, gpu)   --time=$TLIM_LONG (nolim, gnolim)"
# what the gnolim jobs of THIS launch will serve, given their walltime (170 h = the cpu 10M guard)
if (( CAP_LONG >= 170*3600 )); then
  echo "[time] gnolim jobs get >= 170 h -> they claim task-R 10M runs first, then drain into task S"
else
  echo "[time] gnolim jobs get < 170 h -> the run's walltime guard skips the 10M pool; they serve task S"
fi

# ---- queue-depth guard: never submit more worker slots than there is work left ------------------
# Name-independent by construction: `running/` holds exactly one marker per BUSY worker of EVERY
# submitter (owner + every collaborator), so subtracting it needs no job-name regex. (The run-8.1
# packet matched the owner fleet by job name and silently counted 0 when that name changed —
# problems/resolved/2026-07-23-04-10. This form cannot drift.)
p1m=$(ls "$POOL_1M" | wc -l); p10m=$(ls "$POOL_10M" | wc -l)
running=$(ls "$RUN_DIR/queue/$SWEEP_ID/running" 2>/dev/null | wc -l)
rem_1m=$(( p1m - running )); (( rem_1m < 0 )) && rem_1m=0
rem_all=$(( p1m + p10m - running )); (( rem_all < 0 )) && rem_all=0
echo "[depth] pending_1m=$p1m  pending_10m=$p10m  running(all submitters)=$running"
echo "[depth] unclaimed for task-S shapes=$rem_1m   unclaimed for gnolim (both pools)=$rem_all"
if (( rem_all <= 0 )); then
  echo "[depth] enough workers are already running for the remaining queue — not submitting"
  exit 0
fi

# ---- my own usage per partition (each user has independent caps) --------------------------------
my_usage() { squeue -u "$USER" -h -o "%C %P" | awk -v p="$1" '$2==p{s+=$1} END{print s+0}'; }
USED_CPU=$(my_usage cpu); USED_GPU=$(my_usage gpu); USED_NOLIM=$(my_usage nolim); USED_GNOLIM=$(my_usage gnolim)
echo "[caps] my usage: cpu=$USED_CPU/400  gpu=$USED_GPU/400  nolim=$USED_NOLIM/80  gnolim=$USED_GNOLIM/80"
echo "[caps] (16-CPU headroom kept under every pool, for your own interactive jobs)"

mkdir -p "$LOGDIR"
submitted=0
budget=$rem_all   # ONE queue-depth budget across all four buckets: never plan more worker slots in
                  # this pass than there are unclaimed runs left in the whole sweep

# sub <script> <ntasks> <partition> <timelimit> [extra sbatch args...]
# The job name is the random prefix for <partition>; --ntasks OVERRIDES the script's #SBATCH default
# (so one wrapper serves any fragment shape); the id is captured under a flock (the launcher and the
# monitor may both append IDFILE) and tagged --comment=<sweep>_<user> for requeue recovery.
sub() {
  local script="$1" ntasks="$2" part="$3" tlim="$4"; shift 4
  (( budget <= 0 )) && { echo "[depth] budget used up — skipping $script ntasks=$ntasks part=$part"; return; }
  budget=$(( budget - ntasks ))
  local name; name="$(prefix_for_partition "$part")"
  local cmd=(sbatch --job-name="$name" --chdir="$PROJ_DIR" --partition="$part" \
        --nodes=1 --ntasks="$ntasks" --cpus-per-task=1 --ntasks-per-core=2 --mem-per-cpu=2G \
        --output="$LOGDIR/${name}_%j.log" --time="$tlim" \
        --comment="${SWEEP_ID}_${USER}" --export=ALL,SWEEP_ID="$SWEEP_ID" "$@" "$HERE/$script")
  if [[ "$DRY" == "1" ]]; then echo "[dry] ${cmd[*]}"; return; fi
  local out id
  out=$("${cmd[@]}")
  id=$(grep -oP '[0-9]+$' <<< "$out")
  [[ -n "$id" ]] && flock "$IDFILE.lock" bash -c "echo $id >> '$IDFILE'"
  echo "[sub] $name $script ntasks=$ntasks part=$part -> ${id:-FAILED}  $*"
  submitted=$((submitted+1))
  if (( submitted % 10 == 0 )); then echo "[throttle] $submitted submitted; sleep 30"; sleep 30; fi
}

# usable_threads <node> <alloc_threads> -> threads this node can still take (0 when unknown/none).
# sinfo %C is A/I/O/T (allocated/idle/other/total hardware threads); a node's usable slice is
# min(idle, CPUEfctv - allocated) — a node showing 32 idle threads takes at most 30 (CPUEfctv=CPUTot-2).
usable_threads() {
  local c; c=$(sinfo -n "$1" -h -o "%C" 2>/dev/null | head -1)
  [[ -z "$c" ]] && { echo 0; return; }
  awk -F/ -v alloc="$2" '{a=$1; i=$2; u=alloc-a; if (i<u) u=i; if (u<0) u=0; print u}' <<< "$c"
}
# pick_shape <max_tasks> -> the largest shipped shape that fits, or empty
pick_shape() { for s in 30 28 14 12; do (( s <= $1 )) && { echo "$s"; return; }; done; }
# NODE_SPARE threads are left unallocated on every PINNED node so a real GPU job can still get CPUs
# there (a 30-allocatable-thread node then takes 14+12=26, as the run-8.1 packet did).
NODE_SPARE=4
# script_for_shape <shape> <cpu|gnolim>
script_for_shape() {
  case "$2" in
    gnolim) case "$1" in 14) echo worker_gnolim_14x1_collab.slurm ;; 12) echo worker_gnolim_12x1_collab.slurm ;;
                         *)  echo "no gnolim script for shape $1" >&2; return 1 ;; esac ;;
    *)      echo "worker_${1}x1_collab.slurm" ;;
  esac
}
# does partition <p> have any node with >= <n> idle threads?
part_has_idle() { sinfo -p "$1" -N -h -o "%C" | awk -F/ -v n="$2" '$2>=n{f=1} END{exit !f}'; }

CATALOG="/p/rlprojects/.claude/skills/submit-gpu-sweep/server_introduction/server_introduction.json"

# ---- bucket 1: gpu partition, CPU-ONLY, LOWEST-capability GPU nodes first -----------------------
# The eligible classes and their capability_rank come live from the shared node catalog (single source
# of truth); CPU-only jobs fill the LEAST capable GPU nodes first so better GPUs' CPUs stay free for
# real GPU jobs. Only clearly-lower-tier families are eligible (the ALLOW include-list), and the
# nodes the OWNER's task-R cuda jobs are submitted on are excluded outright — filling their CPUs
# would block the 10M GPU runs this sweep depends on. These are CPU-ONLY jobs: --gpus-per-node=0 and
# WORKER_DEVICE=cpu (set in the .slurm) — never a GPU job.
cap=$(( 400 - 16 - USED_GPU )); room=$(( TARGET_GPU - USED_GPU ))
(( room > cap )) && room=$cap; (( room < 0 )) && room=0
(( room > rem_1m )) && room=$rem_1m
echo "[gpu] target=$TARGET_GPU room=$room (CPU-only jobs on gpu nodes, ascending GPU capability):"
# emits "node cores alloc_threads" lines, least-capable class first (descending capability_rank)
gpu_nodes=$("$PY" - "$CATALOG" << 'PYEOF'
import json, sys
# include-list of clearly-lower-tier GPU families whose CPUs we fill first (least capable)
ALLOW = {"tesla_p100", "p100", "gtx_1080_ti", "titan_xp", "titan_x", "rtx_2080_ti", "quadro_rtx_4000"}
# the OWNER's task-R cuda host nodes: never take their CPUs (a 10M GPU run needs 2 free CPUs there)
OWNER_TASK_R_NODES = {"cheetah02", "cheetah04", "cheetah08", "cheetah09",
                      "jaguar01", "jaguar02",
                      "adriatic01", "adriatic02", "adriatic03", "adriatic04", "adriatic05", "adriatic06"}
cat = json.load(open(sys.argv[1]))
assert cat.get("schema_version", 0) >= 2, "server_introduction.json schema too old"
rows = []
# collect (capability_rank, node, physical_cores, allocatable_threads) for every allowed gpu class
for c in cat["classes"]:
    fam = (c.get("gres_type") or c.get("family") or "").lower()
    if c.get("partition") != "gpu" or not any(a in fam for a in ALLOW):
        continue
    threads = c["cpu_threads"]
    cores = c.get("cpu_cores") or threads // 2      # ThreadsPerCore=2 on every node here
    alloc = c.get("cpu_alloc_threads") or (threads - 2)
    for node in c["nodes"]:
        if node in OWNER_TASK_R_NODES:
            continue
        rows.append((c["capability_rank"], node, cores, alloc))
# least capable first (higher capability_rank = less capable in this catalog)
for rank, node, cores, alloc in sorted(rows, key=lambda r: (-r[0], r[1])):
    print(node, cores, alloc)
PYEOF
) || { echo "[gpu] catalog read failed — skipping the gpu bucket"; gpu_nodes=""; }
while read -r node cores alloc; do
  [[ -z "${node:-}" ]] && continue
  (( room < 12 )) && break
  # fill this node with as many shapes as its free slice takes (each job's ntasks <= physical cores),
  # leaving NODE_SPARE threads free; `left` tracks the slice already spoken for in this pass
  left=$(usable_threads "$node" "$alloc"); left=$(( left - NODE_SPARE ))
  while (( left >= 12 && room >= 12 )); do
    max=$(( cores < left ? cores : left )); (( max > room )) && max=$room
    shape=$(pick_shape "$max"); [[ -z "$shape" ]] && break
    sub "$(script_for_shape "$shape" cpu)" "$shape" gpu "$TLIM_SHORT" --nodelist="$node" --gpus-per-node=0
    room=$(( room - shape )); left=$(( left - shape ))
  done
done <<< "$gpu_nodes"

# ---- bucket 2: gnolim partition, CPU-ONLY (pending_10m first, pending_1m fallback) ---------------
# gnolim's Pascal/Maxwell GPUs cannot run this run's torch build, so every gnolim job is CPU-only
# (--gpus-per-node=0, WORKER_DEVICE=cpu). The 16-core ai class takes 14x1; the 12-core jinx/titanx
# class takes 12x1 (sbatch validates --ntasks against PHYSICAL CORES).
cap=$(( 80 - 16 - USED_GNOLIM )); room=$(( TARGET_GNOLIM - USED_GNOLIM ))
(( room > cap )) && room=$cap; (( room < 0 )) && room=0
(( room > rem_all )) && room=$rem_all
echo "[gnolim] target=$TARGET_GNOLIM room=$room"
gnolim_nodes=$("$PY" - "$CATALOG" << 'PYEOF'
import json, sys
cat = json.load(open(sys.argv[1]))
rows = []
# every gnolim class is eligible: for this run gnolim is a CPU pool (its GPUs are unusable here)
for c in cat["classes"]:
    if c.get("partition") != "gnolim":
        continue
    threads = c["cpu_threads"]
    cores = c.get("cpu_cores") or threads // 2
    alloc = c.get("cpu_alloc_threads") or (threads - 2)
    for node in c["nodes"]:
        rows.append((c["capability_rank"], node, cores, alloc))
for rank, node, cores, alloc in sorted(rows, key=lambda r: (-r[0], r[1])):
    print(node, cores, alloc)
PYEOF
) || { echo "[gnolim] catalog read failed — skipping the gnolim bucket"; gnolim_nodes=""; }
while read -r node cores alloc; do
  [[ -z "${node:-}" ]] && continue
  (( room < 12 )) && break
  left=$(usable_threads "$node" "$alloc"); left=$(( left - NODE_SPARE ))
  while (( left >= 12 && room >= 12 )); do
    max=$(( cores < left ? cores : left )); (( max > room )) && max=$room
    (( max > 14 )) && max=14      # only 14x1 and 12x1 gnolim wrappers ship in this packet
    shape=$(pick_shape "$max"); [[ -z "$shape" ]] && break
    sub "$(script_for_shape "$shape" gnolim)" "$shape" gnolim "$TLIM_LONG" --nodelist="$node" --gpus-per-node=0
    room=$(( room - shape )); left=$(( left - shape ))
  done
done <<< "$gnolim_nodes"

# ---- bucket 3: open cpu partition (cumulative ceiling TARGET_CPU, 30x1 + 14x1 remainder) ---------
# TARGET_* are CUMULATIVE ceilings (usage never exceeds the target), not per-cycle additions, so the
# monitor's auto-refill cannot creep the pool up to cap-16 over many cycles.
cap=$(( 400 - 16 - USED_CPU )); room=$(( TARGET_CPU - USED_CPU ))
(( room > cap )) && room=$cap; (( room < 0 )) && room=0
(( room > rem_1m )) && room=$rem_1m
njobs=$(( room / 30 ))
echo "[cpu] target=$TARGET_CPU room=$room -> $njobs x 30x1 (no nodelist)"
for i in $(seq "$njobs"); do sub worker_30x1_collab.slurm 30 cpu "$TLIM_SHORT"; room=$((room-30)); done
if (( room >= 14 )) && part_has_idle cpu 14; then sub worker_14x1_collab.slurm 14 cpu "$TLIM_SHORT"; room=$((room-14)); fi

# ---- bucket 4: nolim (cumulative ceiling TARGET_NOLIM of the 80 cap, 30x1 + 14x1 remainder) ------
cap=$(( 80 - 16 - USED_NOLIM )); room=$(( TARGET_NOLIM - USED_NOLIM ))
(( room > cap )) && room=$cap; (( room < 0 )) && room=0
(( room > rem_1m )) && room=$rem_1m
echo "[nolim] target=$TARGET_NOLIM room=$room"
while (( room >= 30 )) && part_has_idle nolim 30; do sub worker_30x1_collab.slurm 30 nolim "$TLIM_LONG"; room=$((room-30)); done
if (( room >= 14 )) && part_has_idle nolim 14; then sub worker_14x1_collab.slurm 14 nolim "$TLIM_LONG"; room=$((room-14)); fi

echo "[done] submitted $submitted jobs; ids in $IDFILE"
echo "[next] verify the first job of each shape: sacct -j <id> --format=JobID,JobName,AllocCPUS,ReqMem,NodeList"
echo "[next]   AllocCPUS must EQUAL ntasks (30/28/14/12); if it is doubled, --ntasks-per-core=2 was lost — report a problem"
echo "[next] keep the passive monitor running: nohup bash $HERE/monitor_collaborator.sh >> $LOGDIR/monitor_collaborator.log 2>&1 &"
