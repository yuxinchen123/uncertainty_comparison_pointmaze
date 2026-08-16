#!/bin/bash
# Pilot for the learning-outcome campaign, run on serval05 under the exclusive lock.
# Four things are checked before any card time is spent on the real runs:
#   1. the driver runs end to end for both frameworks and both precisions;
#   2. a resumed run reproduces an uninterrupted one exactly (the resume is correct, not merely
#      "it kept going");
#   3. the precision knob changes the parameters the REAL trainer produces (not just a bare
#      matrix multiplication), and the same precision twice is bitwise identical;
#   4. the requested precision is what the card actually used.
set -euo pipefail
RUN="$1"                      # the run folder
CODE="$RUN/code"
PILOT="$RUN/data/pilot"
TORCH=/localtmp/sl5nw/venvs/rnd09_torch/bin/python
JAX=/localtmp/sl5nw/venvs/rnd09_jax/bin/python
export PYTHONNOUSERSITE=1
CK=/localtmp/sl5nw/rnd09_pilot
rm -rf "$PILOT" "$CK"
mkdir -p "$PILOT" "$CK"

# --- 1 + 2. driver, then resume-equals-uninterrupted, for each framework -------------------
for FW in torch jax; do
  PY=$TORCH; [ "$FW" = jax ] && PY=$JAX
  # (a) uninterrupted 60 iterations
  $PY "$CODE/train_learning_outcome.py" --framework $FW --precision reduced \
      --outdir "$PILOT/whole" --ckpt-root "$CK/whole_$FW" --copies-per-rate 8 \
      --iterations 60 --history-every 10 --checkpoint-every 20 --log-every 5
  # (b) stop at 20, then continue to 60 in a second process
  $PY "$CODE/train_learning_outcome.py" --framework $FW --precision reduced \
      --outdir "$PILOT/resumed" --ckpt-root "$CK/resumed_$FW" --copies-per-rate 8 \
      --iterations 20 --history-every 10 --checkpoint-every 20 --log-every 5
  rm -f "$PILOT/resumed/data/${FW}_reduced/record.json"
  $PY "$CODE/train_learning_outcome.py" --framework $FW --precision reduced \
      --outdir "$PILOT/resumed" --ckpt-root "$CK/resumed_$FW" --copies-per-rate 8 \
      --iterations 60 --history-every 10 --checkpoint-every 20 --log-every 5
done

# --- 3. the precision knob reaches the real trainer ----------------------------------------
for FW in torch jax; do
  PY=$TORCH; [ "$FW" = jax ] && PY=$JAX
  $PY "$CODE/precision_effect_check.py" dump --framework $FW --precision reduced \
      --out "$PILOT/${FW}_reduced_a.npy"
  $PY "$CODE/precision_effect_check.py" dump --framework $FW --precision reduced \
      --out "$PILOT/${FW}_reduced_b.npy"
  $PY "$CODE/precision_effect_check.py" dump --framework $FW --precision exact \
      --out "$PILOT/${FW}_exact.npy"
  $PY "$CODE/precision_effect_check.py" compare \
      --repeat "$PILOT/${FW}_reduced_a.npy" "$PILOT/${FW}_reduced_b.npy" \
      --other "$PILOT/${FW}_exact.npy" --out "$PILOT/precision_effect_${FW}.json"
done

# --- 4. exact-precision drivers also start (their probe is the gate) -----------------------
for FW in torch jax; do
  PY=$TORCH; [ "$FW" = jax ] && PY=$JAX
  $PY "$CODE/train_learning_outcome.py" --framework $FW --precision exact \
      --outdir "$PILOT/whole" --ckpt-root "$CK/whole_${FW}_exact" --copies-per-rate 8 \
      --iterations 30 --history-every 10 --checkpoint-every 30 --log-every 5
done
echo PILOT_DONE
