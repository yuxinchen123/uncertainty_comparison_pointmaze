#!/bin/bash
# Launch ONE Train-run-4 sweep. Usage:  bash launch_queue.sh [tag]
# Generates a sweep id  SWEEP_ID=<YYYY-MM-DD-HH-MM>_<tag>  (tag defaults to "sweep"), builds that sweep's
# queue (queue/<SWEEP_ID>/) and data dir (data/<SWEEP_ID>/), then submits ~64 jobs (8 workers each => 512
# workers, ~64 squeue IDs), each injected with SWEEP_ID so its workers only touch this sweep. Every sweep is
# isolated by id, so legacy/other sweeps in the same run folder never collide (sweep-id convention,
# .claude/rules/run-id-and-logging.md). Job ids go to submitted_jobids_<SWEEP_ID>.txt (cluster hard rule:
# scancel only ids from a run's own file). Reservation-aware; --mem-per-cpu=2G (1e6 replay buffer ~1 GB/run).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(dirname "$HERE")"
PROJ=/p/rlprojects/RND/07_reconstruction
LOGDIR="$RUN_DIR/logs"; mkdir -p "$LOGDIR"
WORKER="$HERE/worker.slurm"
VENVPY="/p/rlprojects/RND/.venvs/jax_bench/bin/python"

TAG="${1:-sweep}"
SWEEP_ID="$(date +%Y-%m-%d-%H-%M)_${TAG}"
JOBIDS="$HERE/submitted_jobids_${SWEEP_ID}.txt"
echo "[sweep] SWEEP_ID=$SWEEP_ID"

# build this sweep's queue (queue/<SWEEP_ID>/pending/) + manifest row in data/SWEEPS.md
"$VENVPY" "$HERE/build_queue.py" --sweep_id "$SWEEP_ID" || { echo "build_queue failed"; exit 1; }

NTASKS=8; CPT=2; PERJOB=$((NTASKS*CPT))           # 16 cpus per job
RES_JOBS_PER_NODE="${RES_JOBS_PER_NODE:-14}"
N_NOLIM="${N_NOLIM:-10}"; N_CPU="${N_CPU:-20}"; N_GPU="${N_GPU:-10}"

submitted=0
sub() {  # sub <jobname> <extra sbatch args...>
  local name="$1"; shift
  local out id
  out=$(sbatch --job-name="$name" --chdir="$PROJ" --nodes=1 --ntasks="$NTASKS" --cpus-per-task="$CPT" \
        --mem-per-cpu=2G --output="$LOGDIR/${name}_%j.log" --time=1-00:00:00 \
        --export=ALL,SWEEP_ID="$SWEEP_ID" "$@" "$WORKER")
  id=$(grep -oP '[0-9]+$' <<< "$out"); [[ -n "$id" ]] && echo "$id" >> "$JOBIDS"
  submitted=$((submitted+1))
  if (( submitted % 10 == 0 )); then echo "[throttle] $submitted submitted; sleep 20"; sleep 20; fi
}

# ---- reservation (dynamic; sometimes none) ----
RES=$(scontrol show reservation -o 2>/dev/null | grep -i 'Users=.*sl5nw' | grep -oP 'ReservationName=\K\S+' | head -1)
declare -A RES_NAME=( [jaguar03]=diff-prior [puma01]=seq2graph )
RESERVED=""
if [[ -n "${RES:-}" ]]; then
  NODES=$(scontrol show reservation "$RES" -o | grep -oP 'Nodes=\K\S+'); RESERVED="$NODES"
  echo "[reservation] active: $RES nodes=$NODES"
  IFS=',' read -ra NLIST <<< "$NODES"
  for node in "${NLIST[@]}"; do
    cpus=$(scontrol show node "$node" 2>/dev/null | grep -oP 'CPUTot=\K[0-9]+')
    part=$(scontrol show node "$node" 2>/dev/null | grep -oP 'Partitions=\K\S+' | cut -d, -f1)
    [[ -z "${cpus:-}" || -z "${part:-}" ]] && { echo "[reservation] $node: missing cpus/part, skip"; continue; }
    njobs=$(( cpus / PERJOB )); (( njobs > RES_JOBS_PER_NODE )) && njobs=$RES_JOBS_PER_NODE
    name="${RES_NAME[$node]:-resv_$node}"
    echo "[reservation] $node part=$part cpus=$cpus -> $njobs jobs ($name)"
    for ((i=0;i<njobs;i++)); do sub "$name" --partition="$part" --reservation="$RES" --nodelist="$node" --qos=csresnolim; done
  done
else
  echo "[reservation] none active; skipping reserved-node jobs"
fi

# ---- open partitions (curated gpu allowlist: include-only, lower-tier nodes) ----
GPU_ALLOW=(adriatic01 adriatic02 adriatic03 adriatic04 adriatic05 adriatic06 affogato11 affogato13 affogato14 affogato15 \
           ai01 ai02 ai03 ai04 ai06 cheetah03 jaguar05 lynx01 lynx02 lynx03 lynx04 lynx05 lynx06 lynx07 lynx10)
if [[ "$RESERVED" != *jaguar03* ]]; then GPU_ALLOW+=(jaguar03); fi
for ((i=0;i<N_NOLIM;i++)); do sub causal-rep --partition=nolim; done
for ((i=0;i<N_CPU;i++));   do sub tab-bench  --partition=cpu;   done
n=${#GPU_ALLOW[@]}
for ((i=0;i<N_GPU;i++)); do node="${GPU_ALLOW[$(( i % n ))]}"; sub meta-icl --partition=gpu --nodelist="$node" --gpus-per-node=0; done

echo "[done] sweep $SWEEP_ID: submitted $submitted jobs (=> $submitted slurm IDs, $((submitted*NTASKS)) workers)"
echo "[done] job ids in $JOBIDS  | queue: queue/$SWEEP_ID/  | data: data/$SWEEP_ID/local/"
