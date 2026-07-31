#!/bin/bash
# Compute-throughput scaling test at 1, 2, 8 PHYSICAL cores (cpus 0..n-1 = first thread of each of the
# first n physical cores; no hyperthread sibling used, so no HT contention). Throughput = work/time,
# independent of any frequency counter. Prints CPU model + per-core and aggregate effective GHz.
EXP="$(cd "$(dirname "$0")" && pwd)"
gcc -O3 -o "/tmp/cc_$$" "$EXP/clockcheck.c" 2>/dev/null
echo "host=$(hostname)  cpu=$(lscpu | grep 'Model name' | sed 's/.*: *//')  max_MHz=$(lscpu | awk -F: '/CPU max MHz/{gsub(/ /,"",$2);print $2}')"
for n in 1 2 8; do
  for c in $(seq 0 $((n-1))); do taskset -c "$c" "/tmp/cc_$$" >"/tmp/w_${$}_$c" 2>&1 & done
  wait
  vals=$(for c in $(seq 0 $((n-1))); do grep -oP 'effective_GHz=\K[0-9.]+' "/tmp/w_${$}_$c"; done)
  med=$(echo "$vals" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  total=$(echo "$vals" | awk '{s+=$1} END{printf "%.2f", s}')
  echo "  cores=$n  per_core_GHz=$med  aggregate_GHz=$total"
  rm -f /tmp/w_${$}_*
done
rm -f "/tmp/cc_$$"
