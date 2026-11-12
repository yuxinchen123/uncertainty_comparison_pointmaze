#!/bin/bash
# Submit one 15-minute speed profile to one card and record its id.
#   submit_profile.sh <card> <name> <python script, relative to the platform folder> <its arguments ...>
# <card>: rtx-a4500 (jaguar03, through the reservation), rtx-5080-nekomata01 or rtx-5080-nekomata02 (partition gpu).
# The id is appended to slurm/submitted_jobids_profiles_sl5nw.txt of this run folder; only ids from that file are
# ever cancelled. At most 3 profile jobs of that file wait or run at one time: a fourth is refused (exit 3).
set -uo pipefail
RUN=$(cd "$(dirname "$0")/.." && pwd)
IDS="$RUN/slurm/submitted_jobids_profiles_sl5nw.txt"
card=$1; name=$2; shift 2
touch "$IDS"
active=0
if [ -s "$IDS" ]; then
    active=$(squeue -h -j "$(paste -sd, "$IDS")" -t PENDING,RUNNING,CONFIGURING -o %i 2>/dev/null | wc -l)
fi
[ "$active" -lt 3 ] || { echo "3 profile jobs already wait or run; not submitted" >&2; exit 3; }
case "$card" in
    rtx-a4500)
        RES=$(scontrol show reservation -o 2>/dev/null | grep -i "Users=.*$USER" | grep -oP 'ReservationName=\K\S+' | head -1)
        [ -n "$RES" ] || { echo "no reservation found for $USER" >&2; exit 2; }
        where=(--reservation="$RES" --qos=csresnolim -p gpu -w jaguar03);;
    rtx-5080-nekomata01) where=(-p gpu -w nekomata01);;
    rtx-5080-nekomata02) where=(-p gpu -w nekomata02);;
    *) echo "unknown card $card" >&2; exit 2;;
esac
# PROFILE_CPUS (optional, default 8): processors asked for; fewer are used when a node's other jobs leave fewer
# idle (the iteration time barely depends on the processor count; compilation takes longer with fewer)
# Memory: 24G with the usual 8 processors; 12G when fewer processors are asked for, which is when the node is
# shared and its free memory is short (measured peak of finished profile jobs: 2.1 to 4.8 GB). PROFILE_MEM
# replaces either.
mem=${PROFILE_MEM:-$([ "${PROFILE_CPUS:-8}" -lt 8 ] && echo 12G || echo 24G)}
id=$(sbatch --parsable "${where[@]}" --gpus-per-node=1 -c "${PROFILE_CPUS:-8}" --mem="$mem" -t 00:15:00 \
     -o "$RUN/logs/${name}_%j.out" "$RUN/infrastructure/profile_job.sh" "$@" < /dev/null)
[ -n "$id" ] || { echo "sbatch returned no id" >&2; exit 2; }
echo "$id" >> "$IDS"
echo "$id"
