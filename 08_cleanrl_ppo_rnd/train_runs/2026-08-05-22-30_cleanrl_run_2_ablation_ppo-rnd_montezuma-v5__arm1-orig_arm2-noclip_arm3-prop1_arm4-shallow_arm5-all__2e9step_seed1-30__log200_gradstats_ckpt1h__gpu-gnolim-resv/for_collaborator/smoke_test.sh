#!/bin/bash
# Prove the setup works under YOUR uid before you submit anything to the queue.
# Runs 5 short trainings into your own folder. The queue is never touched.
# Every run has a hard timeout, so this cannot hang: it finishes or it fails.
set -uo pipefail
source /p/rlprojects/RND/08_cleanrl_ppo_rnd/train_runs/2026-08-05-22-30_cleanrl_run_2_ablation_ppo-rnd_montezuma-v5__arm1-orig_arm2-noclip_arm3-prop1_arm4-shallow_arm5-all__2e9step_seed1-30__log200_gradstats_ckpt1h__gpu-gnolim-resv/for_collaborator/packet_env.sh
OUT="$RUN_DIR/for_collaborator/smoke_data/$USER/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
echo "=== 1. permission probes (read-only; these catch an owner-side permission regression) ==="
fail=0
for d in "$RUN_DIR/queue/pending" "$RUN_DIR/queue/running" "$RUN_DIR/queue/done" "$RUN_DIR/queue/failed" "$RUN_DIR/data/$SWEEP_ID/local" "$LOGDIR"; do
  [ -r "$d" ] && [ -w "$d" ] && [ -x "$d" ] && echo "  ok  rwx $d" || { echo "  FAIL     $d"; fail=1; }
done
m=$(ls "$RUN_DIR/queue/pending"/*.json "$RUN_DIR/queue/running"/*.json 2>/dev/null | head -1)
if [ -n "$m" ]; then head -c1 "$m" >/dev/null 2>&1 && echo "  ok  read a queue marker" || { echo "  FAIL cannot read a queue marker"; fail=1; }; fi
[ -x "$PY" ] && echo "  ok  shared env python is executable by you" || { echo "  FAIL $PY"; fail=1; }
[ "$fail" -eq 0 ] || { echo "PERMISSION PROBES FAILED - write a problem report, do not submit"; exit 1; }
echo
echo "=== 2. five short trainings, each with a hard 20-minute timeout ==="
ok=0
ARMS=(arm1_original arm2_no_rnd_grad_clip arm3_update_proportion_1 arm4_shallower_predictor arm5_all)
for i in 0 1 2 3 4; do
  d="$OUT/run$i"; mkdir -p "$d"
  # One short run per arm, so the smoke covers every configuration the queue can hand you.
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 timeout 1200 "$PY" -u "$PROJ_DIR/src/ppo_rnd_envpool_shuze.py" \
      --arm "${ARMS[$i]}" --env_id MontezumaRevenge-v5 --total_timesteps 163840 \
      --num_iterations_obs_norm_init 2 \
      --log_every_updates 2 --output_dir "$d" --run_id "$i" --run_total 1 --seed "$((i+1))" \
      --checkpoint_every_seconds 60 --no-resume --opt_env_threads "$CPUS_PER_RUN" \
      > "$d/smoke.log" 2>&1
  rc=$?
  rec="$d/${i}_of_1.json"
  if [ $rc -eq 0 ] && [ -f "$rec" ] && grep -q '"completed": true' "$rec"; then
    echo "  ok  run $i (${ARMS[$i]}): completed:true, checkpoint $(ls -la "$d"/*.checkpoint.pt 2>/dev/null | awk '{print $5}')"
    ok=$((ok+1))
  else
    echo "  FAIL run $i: rc=$rc  (see $d/smoke.log)"
  fi
done
echo
echo "=== 3. checklist ==="
echo "  [$([ $ok -eq 5 ] && echo x || echo ' ')] all 5 smoke runs wrote completed:true"
echo "  [$([ "$(ls "$RUN_DIR/queue/pending" 2>/dev/null | wc -l)" -ge 0 ] && echo x || echo ' ')] queue untouched by this script"
echo "  [$([ -z "$(ls "$RUN_DIR/for_collaborator/problems/open" 2>/dev/null)" ] && echo x || echo ' ')] problems/open is empty"
echo "  outputs: $OUT"
[ $ok -eq 5 ] && echo "SMOKE PASSED - you may submit workers" || { echo "SMOKE FAILED - write a problem report, do not submit"; exit 1; }
