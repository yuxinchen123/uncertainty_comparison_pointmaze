#!/bin/bash
# Smoke test: prove YOUR setup for this sweep in a couple of minutes WITHOUT submitting a Slurm job
# and WITHOUT touching the shared queue. Two parts:
#   1. non-mutating permission probes (test -r/-w/-x) on the queue state dirs, the data dir, the log
#      dir, and a read of one pending marker FILE — this catches owner-side permission regressions
#      (e.g. mode-600 markers) before any submission.
#   2. >=5 short canary invocations of the REAL entry point (convergence_train.py, ~15 s each) into
#      an isolated for_collaborator/smoke_data/<user>/<ts>/ tree, each under `timeout 1200`. The
#      canaries cover every config family: 3 point sets x 4 inits x both optimizers.
# Every canary run writes its own JSON with completed:true; the queue is never read or renamed.
# Run this before your first real wave; scale up (launcher + monitor) only after every check passes.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/packet_env.sh"
[[ "$USER" == "sl5nw" ]] && { echo "you are the sweep owner — the owner fleet already validated its env"; exit 3; }

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/tmp
Q="$RUN_DIR/queue/$SWEEP_ID"
LOCAL="$RUN_DIR/data/$SWEEP_ID/local"
TS="$(date +%Y-%m-%d-%H-%M-%S)"
SMOKE="$FC/smoke_data/$USER/$TS"
mkdir -p "$SMOKE"
fails=0

# ---- part 1: non-mutating permission probes -------------------------------------------------
echo "=== part 1: permission probes (nothing is created, renamed, or deleted) ==="
probe() {  # probe <flag> <path> <why>: print PASS/FAIL for `test -<flag> <path>`
  local flag="$1" path="$2" why="$3"
  if test -"$flag" "$path"; then echo "  PASS  -$flag  $why  ($path)"
  else echo "  FAIL  -$flag  $why  ($path)"; fails=$((fails+1)); fi
}
# atomic claim = rename pending/ -> running/ and running/ -> done|failed/: needs r+w+x on each dir
for sub in pending running done failed; do
  probe r "$Q/$sub" "read $sub markers";  probe w "$Q/$sub" "claim into/out of $sub";  probe x "$Q/$sub" "traverse $sub"
done
probe w "$LOCAL"  "write per-run JSON output"
probe w "$LOGDIR" "write your job logs"
# FILE-mode probe: a pending marker must be group-readable (the mode-600 regression, 2026-07-12)
marker="$(ls "$Q/pending" 2>/dev/null | head -1)"
if [[ -n "$marker" ]]; then probe r "$Q/pending/$marker" "read a pending marker FILE"
else echo "  NOTE  pending/ is empty (sweep nearly/fully done) — no marker to read-probe"; fi

# ---- part 2: canary invocations of the real trainer (queue untouched) -----------------------
echo
echo "=== part 2: 6 canary convergence_train.py runs into $SMOKE ==="
pending_before="$(ls "$Q/pending" 2>/dev/null | wc -l)"

# each row: label | point_set weight_init bias_init optimizer  <optimizer-specific args>  seed
run_canary() {  # run_canary <k> <label> <trainer args...>
  local k="$1" label="$2"; shift 2
  local outdir="$SMOKE/run${k}"
  echo "-- canary $k: $label"
  timeout 1200 "$PY" "$PROJ_DIR/convergence_train.py" "$@" \
      --a_seed="$k" --n_steps=4096 --run_total=0 --sweep_id=smoke --local_log_dir="$outdir"
  local rc=$?
  # completed:true is the proof the trainer ran end-to-end and flushed its record
  local js; js="$(ls "$outdir/local/"*.json 2>/dev/null | head -1)"
  if (( rc == 0 )) && [[ -n "$js" ]] && grep -q '"completed": true' "$js"; then
    echo "   OK   rc=0, completed:true -> $(basename "$js")"
  else
    echo "   BAD  rc=$rc  json=${js:-<none>}"; fails=$((fails+1))
  fi
}
run_canary 1 "center_square  I1-zero      adam lr1e-03"  --point_set=center_square  --rnd_weight_init=orthogonal      --rnd_bias_init=zero            --rnd_optimizer=adam  --rnd_lr=1e-03
run_canary 2 "top_right_cell I4-normal0.5 adam lr1e-05"  --point_set=top_right_cell --rnd_weight_init=orthogonal      --rnd_bias_init=normal_0.5      --rnd_optimizer=adam  --rnd_lr=1e-05
run_canary 3 "cell_midpoints I3-ptfull    sgd1t e1 t1e4"  --point_set=cell_midpoints --rnd_weight_init=pytorch_default --rnd_bias_init=pytorch_default --rnd_optimizer=sgd1t --rnd_sgd_eta0=1e-01 --rnd_sgd_t0=1e4
run_canary 4 "center_square  I2-ptbias    sgd1t e3 t1e2"  --point_set=center_square  --rnd_weight_init=orthogonal      --rnd_bias_init=pytorch_default --rnd_optimizer=sgd1t --rnd_sgd_eta0=1e-03 --rnd_sgd_t0=1e2
run_canary 5 "top_right_cell I1-zero      adam lr1e-04"  --point_set=top_right_cell --rnd_weight_init=orthogonal      --rnd_bias_init=zero            --rnd_optimizer=adam  --rnd_lr=1e-04
run_canary 6 "cell_midpoints I4-normal0.5 sgd1t e2 t1e3"  --point_set=cell_midpoints --rnd_weight_init=orthogonal      --rnd_bias_init=normal_0.5      --rnd_optimizer=sgd1t --rnd_sgd_eta0=1e-02 --rnd_sgd_t0=1e3

pending_after="$(ls "$Q/pending" 2>/dev/null | wc -l)"

# ---- checklist ------------------------------------------------------------------------------
echo
echo "=== checklist ==="
echo "  [ ] all permission probes PASS (part 1: no FAIL lines above)"
echo "  [ ] all 6 canaries printed 'OK rc=0, completed:true'"
echo "  [ ] queue untouched: pending $pending_before -> $pending_after (should be unchanged by this test)"
open_reports="$(ls "$FC/problems/open/" 2>/dev/null | wc -l)"
echo "  [ ] problems/open/ empty: $open_reports report(s)"
echo "  smoke outputs (safe to delete): $SMOKE"
if (( fails == 0 )) && [[ "$pending_before" == "$pending_after" ]]; then
  echo "SMOKE PASS -> start the monitor, then run launch_workers_collaborator.sh"
  exit 0
else
  echo "SMOKE FAIL ($fails check(s) failed) -> do NOT submit; write a problem file (README step 7) and stop"
  exit 1
fi
