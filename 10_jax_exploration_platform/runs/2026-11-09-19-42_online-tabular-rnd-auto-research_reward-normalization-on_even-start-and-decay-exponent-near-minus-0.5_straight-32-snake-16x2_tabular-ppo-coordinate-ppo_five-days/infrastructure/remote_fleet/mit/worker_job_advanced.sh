#!/bin/bash
# Worker job of the RND grid study on mit, pool advanced. Written by remote_fleet.py: change that, not this copy.
# One card per job. It checks the card, compares one small processor-backend training with UVA's (once per server),
# then runs units from queue/pending one at a time until none is left or too little walltime remains.
# No speed test or benchmark is run.
RUN=/orcd/scratch/orcd/008/shuzel/rnd_grid/runs/auto_research
PY=/orcd/scratch/orcd/008/shuzel/envs/rnd_platform_jax/bin/python
CODE=/orcd/scratch/orcd/008/shuzel/rnd_grid/code/current/10_jax_exploration_platform
UNIT_SECONDS=4000
LIMIT_SECONDS=43200
export GRID_GPU_CLASS=h100_mit
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
# Module-provided CUDA libraries would be found before the environment's own (Gilbreth, DeltaAI).
unset LD_LIBRARY_PATH
JOB=${SLURM_JOB_ID:?this script runs only as a Slurm job}
cd "$RUN" || exit 2
NEED_CARDS=1
RUN_DIR="$RUN"
GUARD_ACTION=${GUARD_ACTION:-requeue}
# CARD GUARD (rule ~/.claude/rules/mit-single-gpu-request.md; P10 of ../submission_principle.md).
# Paste right after the #SBATCH lines, before any worker, or copy this file into the run folder and `source` it.
# Set NEED_CARDS and a PY with torch first. Ends the job within seconds when the allocation is short or long:
# nvidia-smi is compared first, and torch (its import took 273 to 433 s on a cold node) is imported only when
# nvidia-smi already shows the right count.
#   GUARD_ACTION=exit     (default) print the counts and exit 3
#   GUARD_ACTION=requeue  requeue this job (keeps the id, resets the age), at most
#                         GUARD_MAX_RESTARTS times (default 2), then exit 3
# GUARD_ACTION comes from the environment or, when that is unset, from the file $RUN_DIR/guard_action, so a job
# that is already pending can be switched to requeue by writing that file before it starts.
NEED_CARDS=${NEED_CARDS:?set NEED_CARDS to the number of cards the run needs}
GUARD_ACTION=${GUARD_ACTION:-$(tr -d '[:space:]' 2>/dev/null < "${RUN_DIR:-/nonexistent}/guard_action")}
GUARD_ACTION=${GUARD_ACTION:-exit}
have_smi=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
if [ "$have_smi" -eq "$NEED_CARDS" ]; then
  have_torch=$("${PY:-python}" -c 'import jax; print(len(jax.devices("gpu")))' 2>/dev/null || echo "?")
else
  have_torch="not-checked"
fi
echo "[card guard] $(date -Is) job=${SLURM_JOB_ID:-?} node=$(hostname) need=$NEED_CARDS nvidia-smi=$have_smi torch=$have_torch restart=${SLURM_RESTART_COUNT:-0} action=$GUARD_ACTION"
if [ "$have_smi" -ne "$NEED_CARDS" ] || [ "$have_torch" != "$NEED_CARDS" ]; then
  echo "[card guard] MISMATCH: the job holds $have_smi card(s) by nvidia-smi and $have_torch by torch, needs $NEED_CARDS."
  if [ "$GUARD_ACTION" = requeue ] && [ "${SLURM_RESTART_COUNT:-0}" -lt "${GUARD_MAX_RESTARTS:-2}" ]; then
    echo "[card guard] requeueing job $SLURM_JOB_ID (scontrol requeue) so it is allocated afresh; never cancel it"
    scontrol requeue "$SLURM_JOB_ID"
    sleep 30
  fi
  echo "[card guard] exiting 3 so the card(s) go back to the pool at once"
  exit 3
fi
echo "[card guard] ok: $have_smi card(s), as needed"

# Record the start, the node and the card.
echo "$JOB $(hostname) $(date -Is) restart=${SLURM_RESTART_COUNT:-0}" >> slurm/job_starts.txt
echo "job $JOB on $(hostname), start $(date -Is), class $GRID_GPU_CLASS"
CARD=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -n 1)
echo "card: $CARD MiB"

# A requeued job starts again under the same id: a unit claimed by its earlier start goes back to pending.
for claim in queue/running/*.job; do
  [ -e "$claim" ] || continue
  grep -q "^job=$JOB " "$claim" || continue
  unit=$(basename "$claim" .job)
  mv "queue/running/$unit.json" "queue/pending/$unit.json" && echo "released $unit, claimed by an earlier start of this job"
  mv "$claim" "logs/${unit}_${JOB}_released_$(date +%s).job"
done

# Nothing to do: give the card back before starting Python.
if ! ls queue/pending/*.json > /dev/null 2>&1; then echo "nothing pending; this job ends"; exit 0; fi

# The environment must use the card, or every unit taken would fail.
"$PY" -c 'import jax; print("jax devices:", jax.devices()); assert jax.default_backend() == "gpu"' \
  || { echo "jax does not use the card; this job ends without taking a unit"; exit 3; }

# First-unit check, once per server: the same small training as on UVA, on the processor backend, must give the
# same arrays (floats within a relative difference of 1e-4, integers equal). A failure leaves every unit unclaimed.
if ! grep -q '"passed": true' first_unit_check.json 2> /dev/null; then
  DUMP="first_unit_check/dump_$JOB.npz"
  JAX_PLATFORMS=cpu PYTHONPATH="$CODE/src" "$PY" "$CODE/scripts/grid_initialization_campaign/check_online_path_unchanged.py" --dump "$DUMP" \
    && "$PY" first_unit_check/remote_first_unit_compare.py first_unit_check/reference_uva.npz "$DUMP" \
         --output first_unit_check.json --server mit
  if ! grep -q '"passed": true' first_unit_check.json 2> /dev/null; then
    echo "the first-unit check did not pass; this job ends without taking a unit"; exit 4
  fi
fi

END=${SLURM_JOB_END_TIME:-$(( $(date +%s) + LIMIT_SECONDS ))}
STATUS=0
while :; do
  # Stop while a whole unit still fits, so no unit is cut off by the time limit.
  LEFT=$(( END - $(date +%s) ))
  if [ "$LEFT" -lt "$UNIT_SECONDS" ]; then echo "walltime left $LEFT s is below $UNIT_SECONDS s; this job ends"; break; fi
  # Claim the first pending unit: the move succeeds for exactly one job.
  unit=""
  for file in queue/pending/*.json; do
    [ -e "$file" ] || continue
    name=$(basename "$file" .json)
    if mv "$file" "queue/running/$name.json" 2> /dev/null; then unit=$name; break; fi
  done
  if [ -z "$unit" ]; then echo "nothing pending; this job ends"; break; fi
  printf 'job=%s host=%s gpu=%s started=%s\n' "$JOB" "$(hostname)" "$(echo "$CARD" | tr ' ' '_')" "$(date -Is)" > "queue/running/$unit.job"
  echo "$(date -Is) start $unit"
  BEGAN=$(date +%s)
  "$PY" "$CODE/scripts/grid_initialization_campaign/train_unit.py" --run-dir "$RUN" \
    --specification "$RUN/specifications/$unit.json" > "logs/${unit}_${JOB}.log" 2>&1
  CODE_OF_UNIT=$?
  printf 'exit=%s wall_seconds=%s ended=%s\n' "$CODE_OF_UNIT" "$(( $(date +%s) - BEGAN ))" "$(date -Is)" >> "queue/running/$unit.job"
  # The claim record moves first, so a unit seen in done/ or failed/ always has its record beside it.
  TO=failed; [ "$CODE_OF_UNIT" -eq 0 ] && TO=done
  mv "queue/running/$unit.job" "queue/$TO/$unit.job"
  mv "queue/running/$unit.json" "queue/$TO/$unit.json"
  echo "$(date -Is) $TO $unit (exit $CODE_OF_UNIT)"
  # One fault must not empty the queue into failed/: after a failed unit this job takes no more.
  if [ "$CODE_OF_UNIT" -ne 0 ]; then echo "a unit failed; this job ends"; STATUS=1; break; fi
done
echo "job $JOB ends $(date -Is)"
exit $STATUS
