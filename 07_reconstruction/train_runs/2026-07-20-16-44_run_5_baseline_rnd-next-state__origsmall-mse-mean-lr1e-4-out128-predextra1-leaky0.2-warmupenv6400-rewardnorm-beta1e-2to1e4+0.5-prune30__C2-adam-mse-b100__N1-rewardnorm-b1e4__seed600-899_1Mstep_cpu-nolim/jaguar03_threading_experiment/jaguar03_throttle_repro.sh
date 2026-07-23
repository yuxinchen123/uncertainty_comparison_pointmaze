#!/bin/bash
# Reproduce + observe the jaguar03 CPU throttle, with NO application dependencies.
# It puts every core under a sustained memory+compute load and logs the ACHIEVED clock over time.
#
# IMPORTANT: read the clock with turbostat (APERF/MPERF), NOT /sys .../scaling_cur_freq -- on this
# node the acpi-cpufreq /sys value misreports (it reads ~400 MHz even when a core is running fast),
# so only turbostat's Bzy_MHz is trustworthy.
#
# Run it on the SUSPECT node and on a KNOWN-GOOD node (e.g. puma01) as a control, and compare.
# The throttle needs SUSTAINED load to appear (a 1-minute test can look fine), so run >= 20-30 min.
#
# Usage:  ./jaguar03_throttle_repro.sh [duration_seconds]      # default 1800 (30 min)

set -u
DUR="${1:-1800}"
NODE=$(hostname)
LOG="throttle_repro_${NODE}_$(date +%Y%m%d-%H%M%S).log"

{
  echo "node=$NODE  duration=${DUR}s"
  cf=/sys/devices/system/cpu/cpu0/cpufreq
  echo "cpufreq: governor=$(cat $cf/scaling_governor 2>/dev/null) driver=$(cat $cf/scaling_driver 2>/dev/null)" \
       "scaling_min=$(cat $cf/scaling_min_freq 2>/dev/null) scaling_max=$(cat $cf/scaling_max_freq 2>/dev/null) KHz"
  echo "online CPUs: $(nproc)"
} | tee "$LOG"

# 1) sustained all-core memory+compute load (matrix ops touch DRAM + FPU, like the real workload)
if command -v stress-ng >/dev/null 2>&1; then
  stress-ng --matrix 0 --matrix-size 256 --timeout "${DUR}s" >/dev/null 2>&1 &
  LOAD=$!
  echo "load: stress-ng --matrix 0 --matrix-size 256  (pid $LOAD)" | tee -a "$LOG"
else
  echo "NOTE: stress-ng not found. Install it, or start any all-core load yourself" | tee -a "$LOG"
  echo "      (e.g.  for i in \$(seq \$(nproc)); do openssl speed -multi 1 >/dev/null & done ). Monitoring only below." | tee -a "$LOG"
  LOAD=""
fi

# 2) log the true achieved clock + package temp/power every 30 s
if ! command -v turbostat >/dev/null 2>&1; then
  echo "WARNING: turbostat not found (part of linux-tools / cpupower). Without it the frequency" | tee -a "$LOG"
  echo "         reading is unreliable on this node. Please install turbostat and re-run." | tee -a "$LOG"
fi
echo "time     Bzy_MHz  PkgTmp  PkgWatt   (Bzy_MHz is the real clock; watch it fall over the run on the faulty node)" | tee -a "$LOG"
END=$((SECONDS + DUR))
while [ $SECONDS -lt $END ]; do
  if command -v turbostat >/dev/null 2>&1; then
    row=$(turbostat --quiet --Summary --interval 1 --num_iterations 1 2>/dev/null \
          | awk 'NR==1{for(i=1;i<=NF;i++){if($i=="Bzy_MHz")b=i; if($i=="PkgTmp")t=i; if($i=="PkgWatt")w=i}} NR==2{printf "%8s %7s %8s", $b, $t, $w}')
    echo "$(date +%H:%M:%S)  $row" | tee -a "$LOG"
  else
    echo "$(date +%H:%M:%S)  (install turbostat for a reliable reading)" | tee -a "$LOG"
  fi
  sleep 30
done
[ -n "$LOAD" ] && kill "$LOAD" 2>/dev/null

echo "DONE -> $LOG" | tee -a "$LOG"
echo "Healthy node: Bzy_MHz stays near/above the 2.0 GHz base for the whole run." | tee -a "$LOG"
echo "Faulty node (jaguar03): Bzy_MHz falls toward a few-hundred MHz (below scaling_min) as load is sustained." | tee -a "$LOG"
