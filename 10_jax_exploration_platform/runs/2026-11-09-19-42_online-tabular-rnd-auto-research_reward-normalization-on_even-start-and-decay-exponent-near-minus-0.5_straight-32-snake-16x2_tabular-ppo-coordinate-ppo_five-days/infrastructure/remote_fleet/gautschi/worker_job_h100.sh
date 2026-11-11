#!/bin/bash
# Worker job of the RND grid study on gautschi, pool h100. Written by remote_fleet.py: change that, not this copy.
# One card per job. It checks the card, compares one small processor-backend training with UVA's (once per server),
# then runs units from queue/pending one at a time until none is left or too little walltime remains.
# No speed test or benchmark is run.
RUN=/scratch/gautschi/liu4869/rnd_grid/runs/auto_research
PY=/scratch/gautschi/liu4869/envs/rnd_platform_jax/bin/python
CODE=/scratch/gautschi/liu4869/rnd_grid/code/current/10_jax_exploration_platform
UNIT_SECONDS=4500
LIMIT_SECONDS=43200
export GRID_GPU_CLASS=h100_sxm_gautschi
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
# Module-provided CUDA libraries would be found before the environment's own (Gilbreth, DeltaAI).
unset LD_LIBRARY_PATH
JOB=${SLURM_JOB_ID:?this script runs only as a Slurm job}
cd "$RUN" || exit 2
# (no card guard on this server)
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
         --output first_unit_check.json --server gautschi
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
