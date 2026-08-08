#!/bin/bash
# The standing monitoring loop AS A SLURM JOB (shared sweep-monitoring skill): one 1-cpu task runs
# one monitor.sh tick every INTERVAL seconds until the run's completion sentinel appears. Every path
# is absolute — sbatch spools this script, so dirname "$0" would point at the spool directory. The
# sweep id is read from the run's slurm/SWEEP_ID.txt, so this script takes no arguments.
# Partition, --time and --job-name are supplied on the sbatch command line.
#
# COMPLETION = pending/ empty AND running/ empty AND all 120 configurations TERMINALLY decided
# (truncated, stopped_at_100 or winner_complete in the decision log). 120 = 4 algorithm arms x 2
# predictor learning rates x 15 bonus weights.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=0
#SBATCH --mem=16G
#SBATCH --output=/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-08-02-00_run_6_pointmaze-large-topright__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1Mstep_seed1200-1499_prune20-race100-winner300_bar-run5-origrnd_cpu-nolim-resv/logs/monitor_loop_%j.log
set -u
RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-08-02-00_run_6_pointmaze-large-topright__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1Mstep_seed1200-1499_prune20-race100-winner300_bar-run5-origrnd_cpu-nolim-resv"
MONITOR="$RUN_DIR/slurm/monitor.sh"
SENTINEL="$RUN_DIR/SWEEP_COMPLETE"
SWEEP_ID="$(cat "$RUN_DIR/slurm/SWEEP_ID.txt")"
INTERVAL="${INTERVAL:-1200}"   # 20 minutes between ticks
Q="$RUN_DIR/queue/$SWEEP_ID"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
export PYTHONNOUSERSITE=1
N_CONFIGS=120

echo "[$(date '+%Y-%m-%dT%H:%M:%S')] monitor_loop.slurm start job=${SLURM_JOB_ID:-?} node=$(hostname) sweep=$SWEEP_ID interval=${INTERVAL}s"
while true; do
  if [[ -f "$SENTINEL" ]]; then
    echo "[$(date '+%Y-%m-%dT%H:%M:%S')] SWEEP_COMPLETE present; monitor loop exiting"
    break
  fi
  bash "$MONITOR" "$SWEEP_ID"
  pend=$(ls "$Q/pending" 2>/dev/null | wc -l)
  run=$(ls "$Q/running" 2>/dev/null | wc -l)
  dec=$("$PY" - "$RUN_DIR/slurm/truncation_decisions_${SWEEP_ID}.jsonl" <<'EOF'
import json, sys, os
# count UNIQUE decided configuration keys in the decision log (0 when the log does not exist yet)
path = sys.argv[1]
keys = set()
if os.path.exists(path):
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("verdict") in ("truncated", "stopped_at_100", "winner_complete"):
            keys.add(d["config_key"])
print(len(keys))
EOF
)
  echo "[$(date '+%Y-%m-%dT%H:%M:%S')] state: pending=$pend running=$run decided=$dec/$N_CONFIGS"
  if (( pend == 0 && run == 0 && dec >= N_CONFIGS )); then
    echo "[$(date '+%Y-%m-%dT%H:%M:%S')] SWEEP_COMPLETE (queue empty, every configuration decided); writing sentinel and exiting"
    touch "$SENTINEL"
    break
  fi
  sleep "$INTERVAL"
done
