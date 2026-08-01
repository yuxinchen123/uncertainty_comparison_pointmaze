#!/bin/bash
# Prepare train run 8.1.2 (AntMaze UMaze + Medium; task S = 24,000 1M-step runs in pending_1m/,
# task R = 200 10M-step baseline runs in pending_10m/). This script BUILDS (or reuses) the work queue
# and then PRINTS a ready-to-paste submission plan — it NEVER runs sbatch itself (submission is a
# deliberate, reviewed owner step). Bucket order follows the shared submission convention, open
# partitions first and the reserved node last:
#   cpu (task S) -> nolim (task S) -> gpu (task R) -> gnolim (task R first, task S fallback)
#   -> reservation on jaguar03 (task S).
#   fresh :  bash launch_queue.sh [tag]                 (builds a new queue, prints the plan)
#   refill:  SWEEP_ID=<existing id> bash launch_queue.sh (reuse the queue; print the plan again)
set -u
[[ "$USER" == "sl5nw" ]] || { echo "owner-only script; refusing to run as $USER"; exit 3; }
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(dirname "$HERE")"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
mkdir -p "$RUN_DIR/logs"

# build (fresh) or reuse (refill) the queue; never submit
if [[ -n "${SWEEP_ID:-}" ]]; then
  [[ -d "$RUN_DIR/queue/$SWEEP_ID/pending_1m" ]] || { echo "ERROR: no queue for SWEEP_ID=$SWEEP_ID"; exit 1; }
  n1m=$(ls "$RUN_DIR/queue/$SWEEP_ID/pending_1m" 2>/dev/null | wc -l)
  n10m=$(ls "$RUN_DIR/queue/$SWEEP_ID/pending_10m" 2>/dev/null | wc -l)
  echo "[refill] reusing sweep $SWEEP_ID (pending_1m=$n1m, pending_10m=$n10m)"
else
  TAG="${1:-run812}"
  SWEEP_ID="$(date +%Y-%m-%d-%H-%M)_${TAG}"
  "$PY" "$HERE/build_queue.py" --sweep_id "$SWEEP_ID" || { echo "build_queue failed"; exit 1; }
fi
JOBIDS="$HERE/submitted_jobids_${SWEEP_ID}.txt"
# monitor_loop.slurm reads the sweep id from this file (it is spooled, so it takes no arguments)
echo "$SWEEP_ID" > "$HERE/SWEEP_ID.txt"
echo "[sweep] $SWEEP_ID  ids -> $JOBIDS  (sweep id also written to $HERE/SWEEP_ID.txt)"
echo

# Live walltime facts, printed so the TIME_* values below are set from today's numbers, not memory.
MAINT_START=$(scontrol show reservation -o 2>/dev/null | grep -i 'MAINT' | grep -oP 'StartTime=\K\S+' | sort | head -1)
echo "[time] next maintenance start : ${MAINT_START:-none found}"
echo "[time] partition MaxTime      : cpu=$(scontrol show partition cpu | grep -oP 'MaxTime=\K\S+' | head -1)" \
     "gpu=$(scontrol show partition gpu | grep -oP 'MaxTime=\K\S+' | head -1)" \
     "nolim=$(scontrol show partition nolim | grep -oP 'MaxTime=\K\S+' | head -1)" \
     "gnolim=$(scontrol show partition gnolim | grep -oP 'MaxTime=\K\S+' | head -1)"
echo "[time] hours from now to the 2026-08-05 07:00 cutoff: $(( ($(date -d '2026-08-05 07:00' +%s) - $(date +%s)) / 3600 ))"
echo

# The plan below is PRINTED, not executed. It is self-contained: it sets SWEEP_ID / HERE / JOBIDS,
# defines the sub() (id capture + batch-of-10 sleep-30 etiquette) and pool_free() / gpus_free()
# helpers, then the per-bucket loops. Copy the whole block into a shell to actually submit.
echo "# ========================= COPY-PASTE FROM HERE TO SUBMIT ========================="
echo "SWEEP_ID=\"$SWEEP_ID\""
echo "HERE=\"$HERE\""
echo "JOBIDS=\"$JOBIDS\""
cat <<'PLAN'
# ############################################################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!  EDIT THE --time PLACEHOLDERS FIRST  !!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Every TIME_* value below MUST satisfy BOTH caps, whichever is smaller:
#   1. the job must END BEFORE 2026-08-05 07:00 — the cs_admin_maint maintenance reservation
#      starts 2026-08-05T07:30 on every node, and a job still running then is NOT started now
#      (it is deferred to a start time after maintenance, i.e. it never runs in this window);
#   2. the partition MaxTime: cpu and gpu 4-00:00:00, nolim and gnolim 20-00:00:00. This cluster
#      sets EnforcePartLimits=NO, so an over-limit --time is ACCEPTED and then held PENDING
#      (PartitionTimeLimit) forever — nothing warns you.
#   3. a job pinned to the reservation must also finish before the reservation's EndTime.
# The launcher printed today's numbers above. sub() REFUSES to submit while a value still reads
# REPLACE_ME, so pasting this block unedited submits nothing.
# ############################################################################################
TIME_CPU="REPLACE_ME"        # cpu partition, task-S workers      (<= 4-00:00:00 and the cutoff)
TIME_NOLIM="REPLACE_ME"      # nolim partition, task-S workers    (<= 20-00:00:00 and the cutoff)
TIME_GPU="REPLACE_ME"        # gpu partition, task-R 10M workers  (<= 4-00:00:00 and the cutoff;
                             #   worker.py claims a 10M item only with >= 90 h REMAINING)
TIME_GNOLIM="REPLACE_ME"     # gnolim partition                   (<= 20-00:00:00 and the cutoff;
                             #   a cpu 10M claim needs >= 170 h remaining, else it serves task S)
TIME_RES="REPLACE_ME"        # reserved jaguar03                  (<= 4-00:00:00, the cutoff, and
                             #   the reservation EndTime)

PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
submitted=0

# sub <jobname> <worker_script> <partition> <time> [extra sbatch flags...]
# Each worker_*.slurm already carries --nodes/--ntasks/--cpus-per-task/--ntasks-per-core/--mem/
# --gpus-per-node/--output and its own WORKER_POOLS; sub() only adds identity, placement, walltime and
# the SWEEP_ID export. --parsable makes sbatch print the bare id, `cut -d';' -f1` drops the optional
# ";cluster" suffix so the id file stays digits-only, and `tee -a` appends it to the id file the moment
# it is returned (the only-cancel-your-own-ids rule).
sub() {
  local name="$1" script="$2" part="$3" tlim="$4"; shift 4
  local id
  case "$tlim" in REPLACE_ME*) echo "[skip] $name: --time placeholder not edited"; return 0;; esac
  id=$(sbatch --parsable --job-name="$name" --partition="$part" --time="$tlim" \
       --export=ALL,SWEEP_ID="$SWEEP_ID" "$@" "$HERE/$script" | cut -d';' -f1 | tee -a "$JOBIDS")
  echo "[sub] $name script=$script part=$part time=$tlim -> ${id:-FAILED}"
  submitted=$((submitted+1))
  if (( submitted % 10 == 0 )); then echo "[throttle] $submitted submitted; sleep 30"; sleep 30; fi
}

# free threads under a per-user partition QOS cap, leaving `head` unallocated (read from scontrol)
pool_free() {  # pool_free <qos> <cap> <head>
  local qos="$1" cap="$2" head="$3" used
  used=$(scontrol show assoc_mgr qos="$qos" flags=qos 2>/dev/null \
         | grep -oP "MaxTRESPU=cpu=$cap\(\K[0-9]+" | head -1)
  used=${used:-0}
  echo $(( cap - used - head ))
}

# free GPUs on one node = configured GPUs - allocated GPUs (0 for a down/drained node)
gpus_free() {  # gpus_free <node>
  local n="$1" info tot alloc
  info=$(scontrol show node "$n" 2>/dev/null) || { echo 0; return; }
  case "$(grep -oP 'State=\K\S+' <<< "$info")" in *DOWN*|*DRAIN*|"") echo 0; return;; esac
  tot=$(grep -oP 'CfgTRES=\S*gres/gpu=\K[0-9]+' <<< "$info" | head -1)
  alloc=$(grep -oP 'AllocTRES=\S*gres/gpu=\K[0-9]+' <<< "$info" | head -1)
  echo $(( ${tot:-0} - ${alloc:-0} ))
}

# ---- 1. cpu partition (open; cap 400 threads) : TASK S, 30x1 shape then the smaller remainders ----
# Shapes: 30x1 for ~32-core nodes, 28x1 for ~28-core, 14x1 for the 16-core class (which rejects 16x1
# outright), 12x1 for a smaller/older node. All of them claim pending_1m only.
CPU_FREE=$(pool_free cspartcpu 400 16); echo "[cpu] free (cap400-used-16)=$CPU_FREE"
while (( CPU_FREE >= 30 )); do sub r812cpu worker_30x1.slurm cpu "$TIME_CPU"; CPU_FREE=$((CPU_FREE-30)); done
(( CPU_FREE >= 28 )) && { sub r812cpu worker_28x1.slurm cpu "$TIME_CPU"; CPU_FREE=$((CPU_FREE-28)); }
(( CPU_FREE >= 14 )) && { sub r812cpu worker_14x1.slurm cpu "$TIME_CPU"; CPU_FREE=$((CPU_FREE-14)); }
(( CPU_FREE >= 12 )) && { sub r812cpu worker_12x1.slurm cpu "$TIME_CPU"; CPU_FREE=$((CPU_FREE-12)); }

# ---- 2. nolim partition (open; cap 80 threads) : TASK S, same shapes ------------------------------
NOLIM_FREE=$(pool_free cspartnolim 80 16); echo "[nolim] free (cap80-used-16)=$NOLIM_FREE"
while (( NOLIM_FREE >= 30 )); do sub r812nl worker_30x1.slurm nolim "$TIME_NOLIM"; NOLIM_FREE=$((NOLIM_FREE-30)); done
(( NOLIM_FREE >= 28 )) && { sub r812nl worker_28x1.slurm nolim "$TIME_NOLIM"; NOLIM_FREE=$((NOLIM_FREE-28)); }
(( NOLIM_FREE >= 14 )) && { sub r812nl worker_14x1.slurm nolim "$TIME_NOLIM"; NOLIM_FREE=$((NOLIM_FREE-14)); }
(( NOLIM_FREE >= 12 )) && { sub r812nl worker_12x1.slurm nolim "$TIME_NOLIM"; NOLIM_FREE=$((NOLIM_FREE-12)); }

# ---- 3. gpu partition : TASK R (10M baseline), 1 GPU x 2 cuda workers, FAST HOSTS ONLY -----------
# Host CPU generation sets 10M speed, not the GPU class: submit ONLY on the classes measured at
# <= 7 h per 1M steps. One job per FREE GPU on those hosts, in submission batches of ~20 GPUs — wait
# until this batch is RUNNING before printing/submitting the next one (the submission-batch rule).
FAST_GPU_HOSTS=(cheetah02 jaguar01 cheetah04 adriatic01 adriatic02 adriatic03 adriatic04 adriatic05 \
                adriatic06 cheetah08 cheetah09 jaguar02)
GPU_BATCH=20
placed=0
for host in "${FAST_GPU_HOSTS[@]}"; do
  free=$(gpus_free "$host")
  echo "[gpu] $host free GPUs=$free"
  while (( free > 0 && placed < GPU_BATCH )); do
    sub r812gpu worker_gpu_10m_1x2.slurm gpu "$TIME_GPU" --nodelist="$host"
    free=$((free-1)); placed=$((placed+1))
  done
  (( placed >= GPU_BATCH )) && break
done
echo "[gpu] placed $placed of the $GPU_BATCH-GPU submission batch"

# ---- 4. gnolim partition (old-tier GPUs; separate QOS cspartgnolim, cap 80 threads) --------------
# CPU-only there (--gpus-per-node=0 is in the worker script: the Pascal/Maxwell cards cannot run
# torch cu128). WORKER_POOLS="pending_10m pending_1m": it takes task-R 10M items first when the job
# has >= 170 h of remaining walltime, and falls back to task-S 1M items otherwise.
# jinx01-02 and titanx03 are a SMALLER core class — do NOT send the 14x1 shape there; copy
# worker_gnolim_14x1.slurm to a 12x1 variant (--ntasks=12, --mem=24G) first if you want to use them.
GNOLIM_FREE=$(pool_free cspartgnolim 80 16); echo "[gnolim] free (cap80-used-16)=$GNOLIM_FREE"
while (( GNOLIM_FREE >= 14 )); do
  sub r812gnl worker_gnolim_14x1.slurm gnolim "$TIME_GNOLIM" --qos=cspartgnolim
  GNOLIM_FREE=$((GNOLIM_FREE-14))
done

# ---- 5. reservation on jaguar03 (LAST; task S, 100 single-cpu workers) ---------------------------
# Reserved capacity is admitted OVER the partition caps, so it is submitted last: filling the open
# pools first yields all the open caps PLUS the reserved node on top. Discover MY reservation name
# dynamically (it changes over time — never hardcode); add --qos=csresnolim if the submit is
# rejected for qos. jaguar03 sits in the gpu partition but serves CPU work here.
RES=$(scontrol show reservation -o 2>/dev/null | grep -i "Users=.*$USER" \
      | grep -oP 'ReservationName=\K\S+' | head -1)
if [[ -n "$RES" ]]; then
  echo "[reservation] using $RES on jaguar03 (ends $(scontrol show reservation "$RES" | grep -oP 'EndTime=\K\S+' | head -1))"
  sub r812res worker_jaguar03_100x1.slurm gpu "$TIME_RES" --reservation="$RES"
else
  echo "[reservation] none for $USER; skipping the jaguar03 bucket"
fi

echo "[done] submitted $submitted worker jobs; ids in $JOBIDS"
echo "[verify] sacct -j \$(head -1 \"$JOBIDS\") --format=JobID,JobName,AllocCPUS,ReqMem,NodeList,Timelimit"
PLAN
echo "# ========================== END COPY-PASTE BLOCK =================================="
echo
echo "[canary] before the full wave, run the canary sweep (its own sweep id, 60000-step runs):"
echo "       $PY $HERE/build_canary_queue.py --sweep_id \$(date +%Y-%m-%d-%H-%M)_run812-canary --count 40"
echo "       canary worker jobs need --export=ALL,SWEEP_ID=<canary id>,WORKER_REQUIRED_10M_HOURS=0"
echo
echo "[next] after submitting, arm the 20-min monitoring loop — its own 1-cpu Slurm job, which reads"
echo "       the sweep id from $HERE/SWEEP_ID.txt (already written above), and whose id goes into the"
echo "       same id file:"
echo "       sbatch --parsable --job-name=r812mon --partition=nolim --time=<same walltime rule> \\"
echo "              $HERE/monitor_loop.slurm | cut -d';' -f1 | tee -a $JOBIDS"
