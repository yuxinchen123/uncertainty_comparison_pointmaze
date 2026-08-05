#!/bin/bash
#SBATCH --job-name=rndshape
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
BASE=/p/rlprojects/RND/08_cleanrl_ppo_rnd/shuze_experiment/2026-08-05_profilling
SRC=/p/rlprojects/RND/08_cleanrl_ppo_rnd/src
PY=/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python
OUT="$BASE/data/shape_test"; mkdir -p "$OUT"
COMMON="--env_id MontezumaRevenge-v5 --num_iterations_obs_norm_init 2 --total_timesteps 278528 \
 --log_every_updates 1 --profile_iterations 12 --run_id 0 --run_total 1 --seed 1 \
 --checkpoint_every_seconds 1e9 --no-resume --opt_env_threads ${SLURM_CPUS_PER_TASK}"
run () {  # name, extra flags
  local name="$1"; shift
  local d; d=$(mktemp -d)
  $PY "$SRC/ppo_rnd_envpool_shuze.py" $COMMON --output_dir "$d" --profile_tag "$name" "$@" \
      > "$OUT/${name}_${SLURMD_NODENAME}.log" 2>&1 || true
  $PY - "$d/0_of_1.json" "$name" <<'PYEOF'
import json,sys,statistics
rows=json.load(open(sys.argv[1]))["eval_history"][4:]
it=[r["charts/iteration_seconds"] for r in rows]
up=[r["charts/update_seconds"] for r in rows]
ro=[r["charts/rollout_seconds"] for r in rows]
print(f'{sys.argv[2]:34s} steps/s {16384/statistics.fmean(it):6.0f} | rollout {statistics.fmean(ro):5.2f}s | update {statistics.fmean(up):5.2f}s')
PYEOF
  rm -rf "$d"
}
echo "node=$SLURMD_NODENAME cpus=$SLURM_CPUS_PER_TASK"
run "A_nofix_benchmark"          --no-fix_envpool_autoreset --opt_cudnn_benchmark --no-opt_fixed_minibatch_shape
run "B_fix_benchmark_varyshape"  --fix_envpool_autoreset --opt_cudnn_benchmark --no-opt_fixed_minibatch_shape
run "C_fix_nobenchmark_varyshape" --fix_envpool_autoreset --no-opt_cudnn_benchmark --no-opt_fixed_minibatch_shape
run "D_fix_benchmark_fixedshape" --fix_envpool_autoreset --opt_cudnn_benchmark --opt_fixed_minibatch_shape
