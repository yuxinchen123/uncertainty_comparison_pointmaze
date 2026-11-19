#!/usr/bin/env bash
# Copy the records of the eight-copy check job from MIT to UVA through the shared connection (no login), and print
# the readings: the job's accounting, its output, the unit's claim record, progress, worker lines, run.json's copies,
# and the seconds per update from the metric lines (median gap between writes, the first two updates left out).
#   bash read_eight_copy_check.sh JOB OUT_FOLDER
set -u
JOB=$1 OUT=$2
UNIT=grid_fast-last-layer_eight-copies_step-sizes-0.01-0.03-0.1-0.3_sharing-group-4_intrinsic-weights-0.25-0.5_seed-15
RUN=/home/shuzel/rnd_pixel/runs/2026-11-18-12-00_pixel-montezuma_fast-last-layer-against-rnd_stage-1-grid_step-sizes-0.01-0.03-0.1-0.3_intrinsic-weights-0.25-0.5-1-2-4-8_sharing-groups-128-16-4_rnd-same-weights_seeds-13-to-20_steps-per-policy-200000000_schedule-horizon-2000000000
SSH="ssh -o BatchMode=yes -o ControlMaster=no -o ProxyCommand=false"
ssh -O check mit > /dev/null 2>&1 || { echo "the shared connection to MIT is closed; nothing read"; exit 1; }
mkdir -p "$OUT"
$SSH mit "sacct -j $JOB -n -P -o JobID,State,Elapsed,Start,End,NodeList,ReqTRES,AllocTRES,MaxRSS; cd $RUN; ls queue/*/$UNIT.json" < /dev/null > "$OUT/sacct_and_queue.txt"
rsync -rtO -e "$SSH" --ignore-missing-args "mit:$RUN/logs/job_$JOB.out" "mit:$RUN/logs/${UNIT}_job-$JOB.log" \
    "mit:$RUN/data/$UNIT/run/progress.json" "mit:$RUN/data/$UNIT/run/events.jsonl" "mit:$RUN/data/$UNIT/run/run.json" \
    "mit:$RUN/data/$UNIT/run/metrics.jsonl" "mit:$RUN/data/$UNIT/run/game_first_times.json" \
    "mit:$RUN/queue/pending/$UNIT.json" "mit:$RUN/queue/running/$UNIT.json" "$OUT/"
cat "$OUT/sacct_and_queue.txt"
echo "--- job output (first and last lines)"; head -n 12 "$OUT/job_$JOB.out" 2>/dev/null; echo ...; tail -n 8 "$OUT/job_$JOB.out" 2>/dev/null
python3 - "$OUT" "$UNIT" "$JOB" <<'EOF'
import json, statistics, sys
from pathlib import Path
out, unit, job = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
# the claim record: the last attempt's card, memory reading, speed and outcome
record = next((json.loads(path.read_text()) for path in (out / f"{unit}.json",) if path.exists()), None)
if record:
    last = record["history"][-1] if record.get("history") else {}
    print("--- claim record:", {key: record.get(key) for key in ("copies", "execution_profile", "minimum_copy_number", "attempts", "failures")})
    print("    environment of the profile:", record.get("execution_profile_settings", {}).get("environment"))
    print("    last attempt:", {key: last.get(key) for key in ("job_id", "card", "card_memory_mb", "outcome", "exit_code",
                                                         "update_at_start", "update_at_end", "seconds_per_update",
                                                         "card_memory_used_mb", "wall_seconds", "copy_number")})
for name in ("progress.json", "run.json"):
    path = out / name
    if path.exists():
        data = json.loads(path.read_text())
        print(f"--- {name}:", data if name == "progress.json" else
              {"copies": data.get("learner", {}).get("n_copies"), "copy_seeds": data.get("copy_seeds"),
               "per_copy": data.get("learner", {}).get("per_copy_scalar_settings"),
               "reset_lane_budget": data.get("environment", {}).get("reset_lane_budget")})
events = out / "events.jsonl"
if events.exists():
    for line in events.read_text().splitlines():
        event = json.loads(line)
        if event.get("event", "").startswith("worker_") or event.get("event") in ("started", "resumed", "stopped", "completed"):
            print("--- event:", {key: event.get(key) for key in ("event", "device", "card_memory_mb", "execution_profile",
                                                               "copies", "update", "outcome", "slurm_job_id")})
metrics = out / "metrics.jsonl"
if metrics.exists():
    lines = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    gaps = [after["unix"] - before["unix"] for before, after in zip(lines[2:], lines[3:])]
    finite = all(isinstance(value, (int, float)) and value == value and abs(value) != float("inf")
                 for line in lines for key, values in line.items() if isinstance(values, list)
                 for value in values if isinstance(value, (int, float)) and not isinstance(value, bool))
    print(f"--- metrics: {len(lines)} updates, copies per line {len(lines[0]['loss']) if lines else None}, "
          f"median seconds per update {statistics.median(gaps) if gaps else None} over {len(gaps)} gaps, "
          f"range {(min(gaps), max(gaps)) if gaps else None}, every number finite: {finite}")
EOF
