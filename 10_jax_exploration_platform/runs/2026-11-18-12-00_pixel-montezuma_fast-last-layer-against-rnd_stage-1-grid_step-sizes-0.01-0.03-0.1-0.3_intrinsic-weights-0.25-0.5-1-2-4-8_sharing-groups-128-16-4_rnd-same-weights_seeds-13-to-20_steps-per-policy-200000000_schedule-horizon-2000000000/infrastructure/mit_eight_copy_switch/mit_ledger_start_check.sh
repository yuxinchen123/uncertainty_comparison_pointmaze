#!/usr/bin/env bash
# Read the card count of every MIT job of a run's MIT ledger within a minute of its start, the jobs submitted later
# included: every 45 s one command through the shared connection on portal04 (never a login: ProxyCommand=false)
# reads the state, start, requested and allocated cards of every ledger id and the card indices held; each job
# seen started (not PENDING) for the first time is appended to OUT once. Jobs already started at the first reading
# are noted as such and not read again (earlier watchers read them). Ends after LIMIT seconds, or when the shared
# connection is closed.
#   bash mit_ledger_start_check.sh LEDGER OUT LIMIT
set -u
LEDGER=$1 OUT=$2 LIMIT=$3
started=$(date +%s)
declare -A read_already
first=yes
pacific() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
echo "$(pacific) ledger watcher starts (process $$), ledger $LEDGER, at most $LIMIT s" >> "$OUT"
while [ $(( $(date +%s) - started )) -lt "$LIMIT" ]; do
    ssh -O check mit > /dev/null 2>&1 || { echo "$(pacific) the shared connection to MIT is closed; this check ends" >> "$OUT"; exit 1; }
    # the ledger's ids, read again at every pass, so jobs submitted later are watched too
    mapfile -t JOBS < <(grep -E '^[0-9]+$' "$LEDGER" | sort -u)
    list=$(IFS=,; echo "${JOBS[*]}")
    answer=$(ssh -o BatchMode=yes -o ControlMaster=no -o ProxyCommand=false mit "sacct -X -n -P -j $list -o JobID,State,Start,ReqTRES,AllocTRES,NodeList; for j in ${JOBS[*]}; do echo \"gres \$j \$(scontrol show job -dd \$j 2>/dev/null | grep -oE 'GRES=\S+\(IDX:[^)]*\)' | head -n 1)\"; done" < /dev/null)
    for job in "${JOBS[@]}"; do
        [ -n "${read_already[$job]:-}" ] && continue
        line=$(echo "$answer" | grep "^$job|" | head -n 1)
        state=$(echo "$line" | cut -d'|' -f2)
        case "$state" in
            PENDING|"") ;;
            *)
                if [ "$first" = yes ]; then
                    echo "$(pacific) already started or ended at the first reading: $job $state" >> "$OUT"
                else
                    echo "$(pacific) read: $line $(echo "$answer" | grep "^gres $job " | cut -d' ' -f3-)" >> "$OUT"
                fi
                read_already[$job]=1 ;;
        esac
    done
    first=no
    read -t 45 <> <(:) || true
done
echo "$(pacific) limit reached; this check ends" >> "$OUT"
