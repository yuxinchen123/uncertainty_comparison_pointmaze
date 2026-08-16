#!/bin/bash
# The four configurations of the learning-outcome campaign, in sequence on serval05.
#
# Order matters: the two reduced-precision runs come first, because they answer the primary
# question (do the two implementations learn the same thing) with the configuration both trainers
# actually ship; the two exact-precision runs answer the secondary one.
#
# Every run is skipped if its record.json already exists, and every run resumes from its own
# checkpoint, so re-running this script after any interruption continues rather than restarts.
set -uo pipefail
RUN="$1"
CODE="$RUN/code"
TORCH=/localtmp/sl5nw/venvs/rnd09_torch/bin/python
JAX=/localtmp/sl5nw/venvs/rnd09_jax/bin/python
export PYTHONNOUSERSITE=1

for SPEC in "torch reduced" "jax reduced" "torch exact" "jax exact"; do
  set -- $SPEC
  FW=$1; PREC=$2
  PY=$TORCH; [ "$FW" = jax ] && PY=$JAX
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') starting $FW $PREC ==="
  # a run that dies is retried twice: it resumes from its last checkpoint, so a retry costs one
  # chunk rather than the run
  for ATTEMPT in 1 2 3; do
    $PY "$CODE/train_learning_outcome.py" --framework $FW --precision $PREC --outdir "$RUN" \
        --copies-per-rate 1024 --iterations 19531 --history-every 200 \
        --checkpoint-every 2000 --log-every 600 && break
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') $FW $PREC attempt $ATTEMPT failed, retrying ==="
    sleep 30
  done
done
echo CAMPAIGN_DONE
