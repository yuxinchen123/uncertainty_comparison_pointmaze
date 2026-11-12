#!/bin/bash
# Submit one test job to puma01 through the reservation, record its id, wait for it, print its log.
#   submit_tests.sh <pytest arguments ...>      (paths relative to 10_jax_exploration_platform)
# The id goes to slurm/submitted_jobids_tests_sl5nw.txt of this run folder; only ids from that file
# are ever cancelled. Exit code: 0 when the job COMPLETED, 1 otherwise.
set -uo pipefail
RUN=$(cd "$(dirname "$0")/.." && pwd)
RES=$(scontrol show reservation -o 2>/dev/null | grep -i "Users=.*$USER" | grep -oP 'ReservationName=\K\S+' | head -1)
[ -n "$RES" ] || { echo "no reservation found for $USER" >&2; exit 2; }
id=$(sbatch --parsable --reservation="$RES" -w puma01 -o "$RUN/logs/tests_%j.out" "$RUN/infrastructure/run_tests_cpu.sh" "$@" < /dev/null)
[ -n "$id" ] || { echo "sbatch returned no id" >&2; exit 2; }
echo "$id" >> "$RUN/slurm/submitted_jobids_tests_sl5nw.txt"
echo "submitted test job $id"
sleep 20
while true; do
    state=$(sacct -j "$id" -X -n -P -o State 2>/dev/null | head -1)
    case "$state" in
        COMPLETED) cat "$RUN/logs/tests_$id.out"; exit 0;;
        FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) cat "$RUN/logs/tests_$id.out" 2>/dev/null; echo "test job $id ended in state $state"; exit 1;;
    esac
    sleep 15
done
