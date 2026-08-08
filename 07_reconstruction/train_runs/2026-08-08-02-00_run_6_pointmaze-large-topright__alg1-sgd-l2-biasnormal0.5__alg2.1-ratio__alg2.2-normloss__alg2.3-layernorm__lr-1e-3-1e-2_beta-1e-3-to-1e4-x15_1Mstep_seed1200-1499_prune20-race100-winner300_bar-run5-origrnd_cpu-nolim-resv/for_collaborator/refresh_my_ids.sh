#!/bin/bash
# Refresh YOUR OWN id file, and print the current state of your jobs. Run this before any manual
# scancel — the id file is the only list you may cancel from, and it is the only list that cannot
# accidentally include somebody else's work.
#
# Discovery is deliberately belt-and-braces: your recorded ids, plus any live job of yours whose
# name carries one of this packet's random prefixes, plus any carrying the --comment recovery tag.
# The extra two catch ids that were submitted but never made it into the file.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/packet_env.sh"
mkdir -p "$(dirname "$IDFILE")"
touch "$IDFILE"

# live jobs of mine matching this packet's prefixes or its recovery comment tag
recovered=$(squeue -u "$USER" -h -O "JobID:.20,Name:.40,Comment:.80" 2>/dev/null \
  | awk -v re="$PREFIX_RE" -v tag="${SWEEP_ID}_${USER}" \
        '$2 ~ re || $3 == tag {print $1}')
{ cat "$IDFILE"; echo "$recovered"; } | grep -E '^[0-9]+$' | sort -u > "$IDFILE.new"
mv "$IDFILE.new" "$IDFILE"

n=$(wc -l < "$IDFILE")
echo "[ids] $n recorded in $IDFILE"
if (( n > 0 )); then
  squeue -u "$USER" -j "$(paste -sd, "$IDFILE")" -h -o "%.10i %.9P %.14j %.2t %.5C %.20R" 2>/dev/null
  echo "[states] $(squeue -u "$USER" -j "$(paste -sd, "$IDFILE")" -h -o '%T' 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
fi
