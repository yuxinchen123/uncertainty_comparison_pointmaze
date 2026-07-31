#!/bin/bash
# Load EVERY allowed CPU with the register-only compute loop (full all-core load), then report the
# per-core effective GHz. Tests whether the node ramps to full clock under all-core load (the admin's
# claim). Throughput is ground truth, independent of any frequency counter.
EXP="$(cd "$(dirname "$0")" && pwd)"
gcc -O3 -o "/tmp/cc_$$" "$EXP/clockcheck.c" 2>/dev/null
# expand this job's allowed cpu list (e.g. "0-110,112-222") into individual cpu ids
cpus=$(taskset -cp $$ 2>/dev/null | grep -oP 'list: \K.*' | tr ',' '\n' | while IFS=- read a b; do
  if [ -n "$b" ]; then seq "$a" "$b"; else echo "$a"; fi; done)
ncpu=$(echo "$cpus" | grep -c .)
echo "  host=$(hostname) loading $ncpu cpus (full all-core load)"
for c in $cpus; do taskset -c "$c" "/tmp/cc_$$" >"/tmp/w_${$}_$c" 2>&1 & done
sleep 5
echo "  scaling_cur_freq sample (unreliable, for reference) cores 0,56,112: $(for c in 0 56 112; do printf '%s ' "$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null)"; done)"
wait
ghz=$(grep -h effective /tmp/w_${$}_* 2>/dev/null | grep -oP 'effective_GHz=\K[0-9.]+' | sort -n)
n=$(echo "$ghz" | grep -c .)
med=$(echo "$ghz" | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
echo "  per-core effective_GHz under FULL load: n=$n  min=$(echo "$ghz"|head -1)  median=$med  max=$(echo "$ghz"|tail -1)"
rm -f "/tmp/cc_$$" /tmp/w_${$}_*
