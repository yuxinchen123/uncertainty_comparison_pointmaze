#!/bin/bash
# Refresh YOUR OWN id file, then print your live jobs. Run this before any scancel.
set -uo pipefail
source /p/rlprojects/RND/08_cleanrl_ppo_rnd/train_runs/2026-08-05-17-45_cleanrl_train_run_1_ppo-rnd_montezuma-revenge-v5__envpool-autoreset-fixed__128env-128step-2e9step__int-coef1-ext-coef2-updateproportion0.25__seed1-30__checkpoint-8h-resumable__gpu-gnolim/for_collaborator/packet_env.sh
touch "$IDFILE"
# Recover any job of yours that carries this run's comment tag but is missing from the file, which
# happens if a submission was interrupted between sbatch returning and the append.
for id in $(squeue -u "$USER" -h -o "%i %k" | awk -v t="${SWEEP_ID}_${USER}" '$2==t{print $1}'); do
  grep -q "^$id " "$IDFILE" || { echo "$id RECOVERED_BY_COMMENT_TAG $(date -Is)" >> "$IDFILE"; echo "recovered $id"; }
done
echo "your ids for this run:"
awk '/^[0-9]/{print $1}' "$IDFILE" | sort -u | while read -r id; do
  printf "  %-10s %s\n" "$id" "$(sacct -j "$id" -n -X -o State 2>/dev/null | head -1 | tr -d ' ')"
done
