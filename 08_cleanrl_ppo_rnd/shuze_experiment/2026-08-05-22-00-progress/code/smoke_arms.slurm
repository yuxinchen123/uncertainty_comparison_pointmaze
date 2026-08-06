#!/bin/bash
# Smoke test for the corrected clip semantics. Two phases, both on one node, inside 20 minutes.
#
# Phase 1 runs all five arms AT ONCE, one GPU each, so the whole ablation is exercised in the time
# one arm would take on its own. Logging is dense (every 10 updates) because the point here is to see
# several rows per arm, not to reproduce the production cadence — the short run does that.
#
# Phase 2 kills a run with SIGTERM and resumes it, which is how a run will actually end on the
# cluster: the walltime kill arrives as a signal, the run checkpoints at the next update boundary,
# and the next job continues from there. The clipping and the gradient statistics both changed, so
# the checkpoint's contents changed with them and the resume has to be re-checked.
#SBATCH --job-name=rndsmoke
#SBATCH --ntasks=1
#SBATCH --time=00:35:00
set -uo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
P=/p/rlprojects/RND/08_cleanrl_ppo_rnd
PY=/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python
OUT="$PROG_DIR/data/smoke"
rm -rf "$OUT"; mkdir -p "$OUT"
ARMS=(arm1_original arm2_no_rnd_grad_clip arm3_update_proportion_1 arm4_shallower_predictor arm5_all)

echo "=== phase 1: five arms at once, one GPU each ==="
date
for i in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$i timeout 720 "$PY" -u "$P/src/ppo_rnd_envpool_shuze.py" \
    --arm "${ARMS[$i]}" --env_id MontezumaRevenge-v5 \
    --total_timesteps 655360 --num_iterations_obs_norm_init 2 \
    --log_every_updates 10 --checkpoint_every_seconds 100 \
    --output_dir "$OUT" --run_id "$i" --run_total 6 --seed $((i+1)) \
    --opt_env_threads 8 --no-resume > "$OUT/${ARMS[$i]}.log" 2>&1 &
done
wait
date
for i in 0 1 2 3 4; do
  echo "  ${ARMS[$i]}: $(grep -c . "$OUT/${ARMS[$i]}.log") log lines, record $([ -f "$OUT/${i}_of_6.json" ] && echo present || echo MISSING)"
done

echo
echo "=== phase 2: kill with SIGTERM at an update boundary, then resume ==="
# A large budget the run will not reach, so the only way it stops is the signal.
CUDA_VISIBLE_DEVICES=5 "$PY" -u "$P/src/ppo_rnd_envpool_shuze.py" \
  --arm arm2_no_rnd_grad_clip --env_id MontezumaRevenge-v5 \
  --total_timesteps 3276800 --num_iterations_obs_norm_init 2 \
  --log_every_updates 10 --checkpoint_every_seconds 100 \
  --output_dir "$OUT" --run_id 5 --run_total 6 --seed 6 \
  --opt_env_threads 8 --resume > "$OUT/resume_first.log" 2>&1 &
FIRST=$!
sleep 240
echo "  sending SIGTERM to $FIRST"
kill -TERM $FIRST
wait $FIRST
echo "  first segment exit=$?"
grep -E "terminat|checkpoint" "$OUT/resume_first.log" | tail -4

echo "  --- resuming the same run ---"
CUDA_VISIBLE_DEVICES=5 timeout 420 "$PY" -u "$P/src/ppo_rnd_envpool_shuze.py" \
  --arm arm2_no_rnd_grad_clip --env_id MontezumaRevenge-v5 \
  --total_timesteps 3276800 --num_iterations_obs_norm_init 2 \
  --log_every_updates 10 --checkpoint_every_seconds 100 \
  --output_dir "$OUT" --run_id 5 --run_total 6 --seed 6 \
  --opt_env_threads 8 --resume > "$OUT/resume_second.log" 2>&1
echo "  second segment exit=$?"
grep -E "resum" "$OUT/resume_second.log" | head -4
echo "=== smoke done ==="
date
