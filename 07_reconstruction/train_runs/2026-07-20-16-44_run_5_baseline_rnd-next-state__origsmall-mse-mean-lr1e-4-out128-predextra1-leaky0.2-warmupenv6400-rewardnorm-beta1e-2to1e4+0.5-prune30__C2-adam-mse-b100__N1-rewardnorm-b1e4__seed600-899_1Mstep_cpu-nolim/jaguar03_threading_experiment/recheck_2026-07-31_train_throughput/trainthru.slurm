#!/bin/bash
#SBATCH --exclusive
#SBATCH --gpus-per-node=0
#SBATCH --mem=0
#SBATCH --ntasks=1
#SBATCH --time=00:20:00

# Actual-training throughput recheck after the 2026-07-31 jaguar03 reboot (14:38).
# Layout L8 = 8 copies of the real run-5 origsmall training workload, 1 thread pinned per
# physical core, node otherwise idle (exclusive allocation). Identical harness, workload, and
# metric as the 2026-07-23 reproduction (sick jaguar03: 168 steps/min/run; healthy adriatic02:
# 1,301 steps/min/run). The driver flushes each run's step count every 50 env steps and polls a
# progress CSV every 60 s, so the throughput slope is measurable inside the 20-minute cap.
# Usage: sbatch [placement flags] trainthru.slurm <workdir with a run_experiment.py copy>
set -u
WORKDIR="$1"
cd "$WORKDIR"
echo "host=$(hostname)  date=$(date '+%F %T')  boot=$(uptime -s)"

# Root-cause metric first: single-thread random-access memory latency over a 256 MB buffer.
# Sick jaguar03 measured 413-425 ns per access; healthy nodes 99-138 ns.
gcc -O3 -o ./latency ../latency.c
echo -n "LATENCY: "
./latency 256

# 15-minute measured window of the real training (the driver launches, polls every 60 s, kills).
WINDOW_SEC=900 /p/rlprojects/RND/.venvs/exploration/bin/python run_experiment.py L8
