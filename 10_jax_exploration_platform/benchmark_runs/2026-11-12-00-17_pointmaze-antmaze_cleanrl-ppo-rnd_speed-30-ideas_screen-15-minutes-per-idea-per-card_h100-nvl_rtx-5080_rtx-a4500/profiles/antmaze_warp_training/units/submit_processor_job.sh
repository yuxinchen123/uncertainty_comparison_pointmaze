#!/bin/bash
# Submit one processor job script to puma01 through the reservation, record its id, wait for it, print its log.
#   submit_processor_job.sh <sbatch script> <its arguments ...>
# The same steps as infrastructure/submit_tests.sh, for a job script other than run_tests_cpu.sh (the Warp
# interpreter's tests and single scripts: profiles/antmaze_warp/units/run_warp_tests_cpu.sh, run_warp_script_cpu.sh).
# The id goes to slurm/submitted_jobids_tests_sl5nw.txt of this run folder; only ids from that file are ever
# cancelled. Exit code: 0 when the job COMPLETED, 1 otherwise.
set -uo pipefail
RUN=$(cd "$(dirname "$0")/../../.." && pwd)
JOB=$1; shift
RES=$(scontrol show reservation -o 2>/dev/null | grep -i "Users=.*$USER" | grep -oP 'ReservationName=\K\S+' | head -1)
[ -n "$RES" ] || { echo "no reservation found for $USER" >&2; exit 2; }
id=$(sbatch --parsable --reservation="$RES" -w puma01 -o "$RUN/logs/tests_%j.out" "$JOB" "$@" < /dev/null)
[ -n "$id" ] || { echo "sbatch returned no id" >&2; exit 2; }
echo "$id" >> "$RUN/slurm/submitted_jobids_tests_sl5nw.txt"
echo "submitted processor job $id"
sleep 20
while true; do
    state=$(sacct -j "$id" -X -n -P -o State 2>/dev/null | head -1)
    case "$state" in
        COMPLETED) cat "$RUN/logs/tests_$id.out"; exit 0;;
        FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) cat "$RUN/logs/tests_$id.out" 2>/dev/null; echo "processor job $id ended in state $state"; exit 1;;
    esac
    sleep 15
done
