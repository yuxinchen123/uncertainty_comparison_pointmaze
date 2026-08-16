#!/bin/bash
# Second pilot: (a) the resume test done correctly — the interrupted run keeps the full run's
# annealing denominator, which the first attempt did not, so the two runs differed for a reason
# that had nothing to do with resuming; (b) the per-iteration cost and memory of all four
# configurations at the campaign's own size, 8,192 copies, which is what the campaign is budgeted
# from.
set -euo pipefail
RUN="$1"
CODE="$RUN/code"
PILOT="$RUN/data/pilot"
TORCH=/localtmp/sl5nw/venvs/rnd09_torch/bin/python
JAX=/localtmp/sl5nw/venvs/rnd09_jax/bin/python
export PYTHONNOUSERSITE=1
CK=/localtmp/sl5nw/rnd09_pilot2
rm -rf "$PILOT/whole2" "$PILOT/resumed2" "$CK"
mkdir -p "$CK"

# --- (a) resume equals uninterrupted ------------------------------------------------------
for FW in torch jax; do
  PY=$TORCH; [ "$FW" = jax ] && PY=$JAX
  $PY "$CODE/train_learning_outcome.py" --framework $FW --precision reduced \
      --outdir "$PILOT/whole2" --ckpt-root "$CK/whole_$FW" --copies-per-rate 8 \
      --iterations 60 --history-every 10 --checkpoint-every 100 --log-every 60
  $PY "$CODE/train_learning_outcome.py" --framework $FW --precision reduced \
      --outdir "$PILOT/resumed2" --ckpt-root "$CK/resumed_$FW" --copies-per-rate 8 \
      --iterations 60 --stop-after 20 --history-every 10 --checkpoint-every 20 --log-every 60
  $PY "$CODE/train_learning_outcome.py" --framework $FW --precision reduced \
      --outdir "$PILOT/resumed2" --ckpt-root "$CK/resumed_$FW" --copies-per-rate 8 \
      --iterations 60 --history-every 10 --checkpoint-every 20 --log-every 60
done

# --- (b) cost of one iteration at the campaign's size -------------------------------------
# checkpoint-every is set above the iteration count on purpose: a 6 GB write inside the timed
# window would be counted as training time.
for FW in torch jax; do
  PY=$TORCH; [ "$FW" = jax ] && PY=$JAX
  for PREC in reduced exact; do
    rm -rf "$CK/timing_${FW}_${PREC}"
    $PY "$CODE/train_learning_outcome.py" --framework $FW --precision $PREC \
        --outdir "$PILOT/timing" --ckpt-root "$CK/timing_${FW}_${PREC}" \
        --copies-per-rate 1024 --iterations 19531 --stop-after 60 \
        --history-every 20 --checkpoint-every 100000 --log-every 60
    mv "$PILOT/timing/data/${FW}_${PREC}/history.jsonl" \
       "$PILOT/timing_${FW}_${PREC}_history.jsonl"
    rm -rf "$PILOT/timing/data/${FW}_${PREC}"
  done
done
echo PILOT2_DONE
