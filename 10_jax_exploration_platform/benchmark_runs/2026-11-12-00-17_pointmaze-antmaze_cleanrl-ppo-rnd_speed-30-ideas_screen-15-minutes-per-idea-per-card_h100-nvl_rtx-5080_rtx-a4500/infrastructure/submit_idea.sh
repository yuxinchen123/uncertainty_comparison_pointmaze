#!/bin/bash
# Submit the paired timing of one PointMaze speed idea on one card (one 15-minute job, both rollout shapes).
#   submit_idea.sh <card> <idea slug> <shapes> <arm> [<arm> ...]
# <card>: as for submit_profile.sh (rtx-a4500, rtx-5080-nekomata01, rtx-5080-nekomata02).
# <idea slug>: the idea's number and name, e.g. 20_target-features-once-per-iteration.
# <shapes>: <environments per policy>:<policies>, comma-separated, e.g. 16:256,128:64.
# <arm>: name[:key=value,...]; the first arm is the reference (see benchmarks/harness/compare_study_arms.py).
# The record goes to profiles/pointmaze/ideas/idea-<slug>_<card model><RECORD_SUFFIX>.json; the job id is printed.
# RECORD_SUFFIX (optional, e.g. _environments-per-policy-4) keeps a second job of one idea on one card from
# replacing the first job's record. RECORD_FOLDER (optional) replaces the folder name "ideas".
set -uo pipefail
RUN=$(cd "$(dirname "$0")/.." && pwd)
card=$1; idea=$2; shapes=$3; shift 3
model=${card%-nekomata0?}
name="idea-${idea}_${model}${RECORD_SUFFIX:-}"
folder="$RUN/profiles/pointmaze/${RECORD_FOLDER:-ideas}"
mkdir -p "$folder"
arms=()
for arm in "$@"; do arms+=(--arm "$arm"); done
bash "$RUN/infrastructure/submit_profile.sh" "$card" "$name" benchmarks/harness/compare_study_arms.py \
    --idea "$idea" --shapes "$shapes" "${arms[@]}" --output "$folder/$name.json"
