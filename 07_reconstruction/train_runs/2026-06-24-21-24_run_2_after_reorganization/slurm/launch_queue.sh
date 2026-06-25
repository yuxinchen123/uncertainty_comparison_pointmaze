#!/bin/bash
# Launch the local-work-queue fleet: 64 slurm jobs, each ntasks=8 (8 workers) => 512 workers but only
# 64 IDs in squeue (the admin sees 64, not 512). Each job runs worker.slurm (NO wandb agent). Tracks every
# job id into submitted_jobids.txt (cluster-slurm hard rule: only ever scancel ids from that file).
# Reservation-aware; --mem-per-cpu=2G (SAC's 1e6 replay buffer ~1 GB/run). Counts overridable via env.
# Run from a shell with the `exploration` conda env active:  bash launch_queue.sh
set -u
PROJ=/p/rlprojects/RND/07_reconstruction
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(dirname "$HERE")"
LOGDIR="$RUN_DIR/logs"; mkdir -p "$LOGDIR"
WORKER="$HERE/worker.slurm"
JOBIDS="$HERE/submitted_jobids.txt"
NTASKS=8; CPT=2; PERJOB=$((NTASKS*CPT))           # 16 cpus per job
# Job counts (=> 64 IDs). Override via env. Reserved nodes are filled fully; open partitions are capped by
# the per-user QOS cpu limit (extra jobs just pend, still <=64 IDs).
RES_JOBS_PER_NODE="${RES_JOBS_PER_NODE:-14}"
N_NOLIM="${N_NOLIM:-10}"; N_CPU="${N_CPU:-20}"; N_GPU="${N_GPU:-10}"

submitted=0
sub() {  # sub <jobname> <extra sbatch args...>
  local name="$1"; shift
  local out id
  out=$(sbatch --job-name="$name" --chdir="$PROJ" --nodes=1 --ntasks="$NTASKS" --cpus-per-task="$CPT" \
        --mem-per-cpu=2G --output="$LOGDIR/${name}_%j.log" --time=2-00:00:00 "$@" "$WORKER")
  id=$(grep -oP '[0-9]+$' <<< "$out"); [[ -n "$id" ]] && echo "$id" >> "$JOBIDS"
  submitted=$((submitted+1))
  if (( submitted % 10 == 0 )); then echo "[throttle] $submitted submitted; sleep 20"; sleep 20; fi
}

# ---- reservation (dynamic) ----
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

# ---- open partitions ----
GPU_ALLOW=(adriatic01 adriatic02 adriatic03 adriatic04 adriatic05 adriatic06 affogato11 affogato13 affogato14 affogato15 \
           ai01 ai02 ai03 ai04 ai06 cheetah03 jaguar05 lynx01 lynx02 lynx03 lynx04 lynx05 lynx06 lynx07 lynx10)
if [[ "$RESERVED" != *jaguar03* ]]; then GPU_ALLOW+=(jaguar03); fi
for ((i=0;i<N_NOLIM;i++)); do sub causal-rep --partition=nolim; done
for ((i=0;i<N_CPU;i++));   do sub tab-bench  --partition=cpu;   done
n=${#GPU_ALLOW[@]}
for ((i=0;i<N_GPU;i++)); do node="${GPU_ALLOW[$(( i % n ))]}"; sub meta-icl --partition=gpu --nodelist="$node" --gpus-per-node=0; done

echo "[done] submitted $submitted jobs (=> $submitted slurm IDs, $((submitted*NTASKS)) workers)"
