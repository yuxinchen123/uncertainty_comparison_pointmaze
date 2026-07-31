#!/bin/bash
# One isolated memory measurement, meant to run as the ONLY job on an otherwise-idle node.
# Usage: mem_probe.sh <nthreads>   (run once with 1, then once with 4 -- sequentially, never together)
# Prints node model/topology (on the 1-thread run) + memory bandwidth (STREAM triad) at <nthreads>,
# and single-thread memory latency. No dependencies beyond gcc.
EXP="$(cd "$(dirname "$0")" && pwd)"
NT="${1:-1}"
if [ "$NT" = "1" ]; then
  echo "MODEL: $(lscpu | grep 'Model name' | sed 's/.*: *//')"
  echo "TOPO: sockets=$(lscpu | awk '/^Socket/{print $2}') cores_per_socket=$(lscpu | awk -F: '/Core\(s\) per socket/{gsub(/ /,"",$2);print $2}')"
  gcc -O3 -o "/tmp/lt_$$" "$EXP/latency.c" 2>/dev/null && { echo -n "LAT_1thr: "; "/tmp/lt_$$" 256; rm -f "/tmp/lt_$$"; }
fi
MB=128; [ "$NT" = "4" ] && MB=96
gcc -O3 -fopenmp -o "/tmp/bw_$$" "$EXP/bandwidth.c" 2>/dev/null && { echo -n "BW_${NT}thr: "; OMP_NUM_THREADS="$NT" "/tmp/bw_$$" "$MB"; rm -f "/tmp/bw_$$"; }
