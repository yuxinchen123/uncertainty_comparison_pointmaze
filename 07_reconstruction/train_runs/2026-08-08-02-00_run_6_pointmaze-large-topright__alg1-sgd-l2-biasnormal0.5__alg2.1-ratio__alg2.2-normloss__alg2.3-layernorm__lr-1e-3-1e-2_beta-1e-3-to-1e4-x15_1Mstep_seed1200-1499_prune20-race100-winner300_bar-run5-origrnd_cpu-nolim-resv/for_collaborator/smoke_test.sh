#!/bin/bash
# Prove YOUR setup works, in minutes, WITHOUT touching the real queue.
#
# What it does, in order:
#   1. non-mutating permission probes on everything a worker of yours must read or write — the queue
#      state directories, one pending marker FILE (not just the directory), the data directory and
#      your log directory. This catches an owner-side permission regression before you submit.
#   2. five short training runs through the REAL entry point, covering all three environments of the
#      sweep, each with a hard 20-minute timeout so the smoke can never hang.
#   3. a printed checklist.
#
# Outputs go to for_collaborator/smoke_data/$USER/<timestamp>/ and nowhere else. No queue marker is
# read-modified, claimed, or moved.
#
#   bash smoke_test.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/packet_env.sh"
STAMP=$(date +%Y-%m-%d-%H-%M-%S)
OUT="$FC/smoke_data/$USER/$STAMP"
mkdir -p "$OUT"
export PYTHONNOUSERSITE=1
fails=0

echo "=== 1. permission probes (nothing is modified) ==="
Q="$RUN_DIR/queue/$SWEEP_ID"
for d in "$Q/pending" "$Q/running" "$Q/done" "$Q/failed" "$RUN_DIR/data/$SWEEP_ID/local" "$LOGDIR"; do
  perm=""
  [[ -r "$d" ]] && perm+="r"; [[ -w "$d" ]] && perm+="w"; [[ -x "$d" ]] && perm+="x"
  if [[ "$perm" == "rwx" ]]; then
    echo "  OK   $d  ($perm)"
  else
    echo "  FAIL $d  (need rwx, have '${perm:-none}')"; fails=$((fails+1))
  fi
done
# a worker reads a marker FILE before claiming it; a directory-only check would pass while every
# marker is mode 600 (a real incident on an earlier packet)
# find -print -quit, NOT a shell glob: this queue holds 36,000 markers and the expanded glob
# exceeds the kernel argument-size limit, so `ls .../*.json` never runs (collaborator report
# 2026-08-08; the addendum queue was small enough to hide this)
marker=$(find "$Q/pending" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)
if [[ -n "$marker" && -r "$marker" ]]; then
  echo "  OK   one pending marker is readable: $(basename "$marker")"
else
  echo "  FAIL cannot read a pending marker file"; fails=$((fails+1))
fi
echo "  env  $("$PY" -c 'import sys, torch; print("python", sys.version.split()[0], "torch", torch.__version__)' 2>&1)"

echo
echo "=== 2. five short runs through the real entry point (hard 20-minute timeout each) ==="
# one row per smoke run: arm, learning rate, bonus weight — one smoke run per run-6 arm plus a
# second learning rate on alg1, all on the sweep's one environment. Extra flags per arm mirror
# slurm/build_queue.py's ARM_EXTRAS.
RUNS=(
  "alg1 0.001 1000 "
  "alg1 0.01 0.001 "
  "alg2.1 0.001 30 --rnd_readout_norm_init=True"
  "alg2.2 0.01 30 --rnd_readout_norm_init=True --rnd_predictor_loss=mse_init_normalized"
  "alg2.3 0.001 10000 --rnd_readout_norm_init=True --rnd_layer_norm=True"
)
i=0
for row in "${RUNS[@]}"; do
  set -- $row
  arm="$1"; lr="$2"; beta="$3"; shift 3; extras="$*"
  echo "  [run $i] $arm lr $lr bonus weight $beta"
  timeout 1200 "$PY" "$PROJ_DIR/train.py" \
    --algorithm=rnd_next_state --beta="$beta" --a_seed=$((90000 + i)) \
    --env_setup=initial_single_large_pointmaze_max_400 \
    --rnd_optimizer=sgd --rnd_bonus_readout=l2 --rnd_lr="$lr" $extras \
    --rnd_update_proportion=1.0 --rnd_activation=leaky_relu --rnd_predictor_extra_layers=1 \
    --rnd_obs_warmup_mode=env_steps --rnd_obs_warmup_steps=6400 \
    --rnd_reward_norm=False \
    --rnd_bias_init=normal_0.5 --rnd_weight_init=orthogonal \
    --rnd_obs_norm=True --rnd_distance=mse --rnd_output_dim=128 --n_predictors=1 \
    --total_timesteps=6000 --eval_freq=3000 --n_eval_episodes=2 \
    --eval_standalone=False --log_distance=False --device=cpu --use_wandb=False \
    --z_logging_mode=local --local_log_dir="$OUT" --run_id=$i --run_total=5 \
    > "$OUT/run_$i.log" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then echo "    OK (exit 0)"; else echo "    FAIL (exit $rc; see $OUT/run_$i.log)"; fails=$((fails+1)); fi
  i=$((i+1))
done

echo
echo "=== 3. checklist ==="
recs=$(ls "$OUT/local"/*.json 2>/dev/null | wc -l)
complete=$("$PY" - "$OUT/local" <<'EOF'
import glob, json, os, sys
# count the smoke records that were written at their natural end
d = sys.argv[1]
print(sum(1 for p in glob.glob(os.path.join(d, "*.json")) if json.load(open(p)).get("completed")))
EOF
)
echo "  records written : $recs (expect 5)"
echo "  completed=true  : $complete (expect 5)"
pend_now=$(ls "$Q/pending" 2>/dev/null | wc -l)
echo "  queue untouched : pending is $pend_now (it must be whatever it was before this script)"
echo "  open problems   : $(ls "$FC/problems/open"/*.md 2>/dev/null | wc -l) (expect 0)"
[[ "$recs" -eq 5 && "$complete" -eq 5 ]] || fails=$((fails+1))
echo
if (( fails == 0 )); then
  echo "SMOKE PASS — you are ready to run launch_workers_collaborator.sh"
  exit 0
fi
echo "SMOKE FAIL ($fails problem(s)). Do NOT submit worker jobs."
echo "Write one file under $FC/problems/open/ describing what failed, and tell the owner."
exit 1
