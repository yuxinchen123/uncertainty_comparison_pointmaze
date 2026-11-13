#!/bin/bash
# Submit one diagnosis run to a named node and card type and record its job id at once.
# Usage: submit_diagnosis_job.sh <node> <card type> <run name> <configuration file> <bonus options or "none"> [more options]
set -euo pipefail
node="$1"; card="$2"; shift 2
folder="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
run="$(cd "$folder/../.." && pwd)"
ledger="$run/slurm/submitted_jobids_diagnosis_sl5nw.txt"
mkdir -p "$folder/logs"
id=$(sbatch --parsable --job-name "maze-diagnosis_$1" --partition gpu --nodes 1 --ntasks 1 --cpus-per-task 2 --mem 4G \
     --time 0:40:00 --output "$folder/logs/$1_%j.log" --gres "gpu:$card:1" --nodelist "$node" \
     --export=ALL,DIAGNOSIS_FOLDER="$folder" "$folder/scripts/diagnosis_job.sh" "$@" < /dev/null)
echo "$id" >> "$ledger"
echo "$id"
