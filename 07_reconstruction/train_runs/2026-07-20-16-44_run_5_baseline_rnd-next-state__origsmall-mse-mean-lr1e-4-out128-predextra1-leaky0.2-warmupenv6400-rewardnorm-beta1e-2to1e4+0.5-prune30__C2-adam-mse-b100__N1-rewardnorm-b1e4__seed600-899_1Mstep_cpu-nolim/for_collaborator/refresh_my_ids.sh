#!/bin/bash
# Refresh YOUR id file from squeue (read-only otherwise). Run this BEFORE any manual scancel so a
# requeued job's new id is recorded — ids in this file are the ONLY jobs you may ever cancel.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/packet_env.sh"
squeue -u "$USER" -h -o "%i %j" | grep -E " ${PREFIX}[0-9]" | while read -r id name; do
  grep -qx "$id" "$IDFILE" 2>/dev/null || \
    { flock "$IDFILE.lock" bash -c "echo $id >> '$IDFILE'"; echo "appended new id $id ($name)"; }
done
echo "current ids in $IDFILE:"
cat "$IDFILE" 2>/dev/null || echo "(none yet)"
