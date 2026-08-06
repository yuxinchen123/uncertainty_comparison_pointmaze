#!/bin/bash
# Smoke: all five arms, one after another, on one GPU. Must finish well inside 20 minutes.
# Each arm trains briefly with the real production flags and writes a real record.
#SBATCH --job-name=rndsmoke
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
set -uo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
P=/p/rlprojects/RND/08_cleanrl_ppo_rnd
OUT="$PROG_DIR/data/smoke"
mkdir -p "$OUT"
PY=/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python
i=0
for arm in arm1_original arm2_no_rnd_grad_clip arm3_update_proportion_1 arm4_shallower_predictor arm5_all; do
  echo "=== $arm ==="
  timeout 420 "$PY" -u "$P/src/ppo_rnd_envpool_shuze.py" \
    --arm "$arm" --env_id MontezumaRevenge-v5 \
    --total_timesteps 3276800 --num_iterations_obs_norm_init 2 \
    --log_every_updates 20 --checkpoint_every_seconds 120 \
    --output_dir "$OUT" --run_id $i --run_total 5 --seed $((i+1)) \
    --opt_env_threads "${SLURM_CPUS_PER_TASK:-8}" --no-resume
  echo "  $arm exit=$?"
  i=$((i+1))
done
echo "=== smoke done ==="
