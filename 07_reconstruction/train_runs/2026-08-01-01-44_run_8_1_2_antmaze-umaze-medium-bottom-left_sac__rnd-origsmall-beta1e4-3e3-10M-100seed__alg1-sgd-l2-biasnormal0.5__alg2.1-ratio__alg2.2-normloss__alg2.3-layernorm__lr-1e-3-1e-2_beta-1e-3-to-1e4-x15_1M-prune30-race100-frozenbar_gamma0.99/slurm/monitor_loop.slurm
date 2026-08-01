#!/bin/bash
# Run 8.1.2 standing monitor loop AS A SLURM JOB (shared sweep-monitoring skill). One 1-cpu task on
# an open partition runs one monitor.sh tick every INTERVAL seconds until the run's completion
# sentinel appears. Everything uses ABSOLUTE paths (sbatch spools this script; dirname "$0" would
# point at the spool dir). The sweep id is read from the run's slurm/SWEEP_ID.txt.
# The partition, --time and --job-name are supplied on the sbatch command line.
# COMPLETION = both pending pools empty AND running empty AND all 240 task-S configs decided
# (pruned or survivor in the stage-1 decision log).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=0
#SBATCH --mem=32G
#SBATCH --output=/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99/logs/monitor_loop_%j.log
set -u
RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99"
MONITOR="$RUN_DIR/slurm/monitor.sh"
SENTINEL="$RUN_DIR/SWEEP_COMPLETE"
SWEEP_ID="$(cat "$RUN_DIR/slurm/SWEEP_ID.txt")"
INTERVAL="${INTERVAL:-1200}"   # 20 min between ticks
Q="$RUN_DIR/queue/$SWEEP_ID"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
N_CONFIGS_S=240

echo "[$(date '+%Y-%m-%dT%H:%M:%S')] monitor_loop.slurm start job=${SLURM_JOB_ID:-?} node=$(hostname) sweep=$SWEEP_ID interval=${INTERVAL}s"
# loop: run one tick, stop when the sweep is complete, otherwise sleep and repeat
while true; do
  if [[ -f "$SENTINEL" ]]; then
    echo "[$(date '+%Y-%m-%dT%H:%M:%S')] SWEEP_COMPLETE present; monitor loop exiting"
    break
  fi
  bash "$MONITOR" "$SWEEP_ID"
  # completion: both pools + running empty AND every task-S config has a stage-1 verdict
  p1=$(ls "$Q/pending_1m" 2>/dev/null | wc -l); p10=$(ls "$Q/pending_10m" 2>/dev/null | wc -l)
  run=$(ls "$Q/running" 2>/dev/null | wc -l)
  dec=$("$PY" - "$RUN_DIR/slurm/stage1_decisions_${SWEEP_ID}.jsonl" <<'EOF'
import json, sys, os
# count UNIQUE decided config keys in the stage-1 decision log (0 when the log is missing)
path = sys.argv[1]
keys = set()
if os.path.exists(path):
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("verdict") in ("pruned", "survivor"):
            keys.add(d["config_key"])
print(len(keys))
EOF
)
  echo "[$(date '+%Y-%m-%dT%H:%M:%S')] state: pending_1m=$p1 pending_10m=$p10 running=$run decided=$dec/$N_CONFIGS_S"
  if (( p1 == 0 && p10 == 0 && run == 0 && dec >= N_CONFIGS_S )); then
    echo "[$(date '+%Y-%m-%dT%H:%M:%S')] SWEEP_COMPLETE (pools empty, running empty, all configs decided); writing sentinel and exiting"
    touch "$SENTINEL"
    break
  fi
  sleep "$INTERVAL"
done
