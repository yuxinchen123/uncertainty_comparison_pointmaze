#!/usr/bin/env bash
#SBATCH --account=biwj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=rnd-atari-pixel-check
#SBATCH --output=/projects/biwj/sliu51/rnd_atari_pixels/runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/slurm/deltaai_pixel_check_%j.out
#SBATCH --error=/projects/biwj/sliu51/rnd_atari_pixels/runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/slurm/deltaai_pixel_check_%j.out
#SBATCH --no-requeue
set -euo pipefail
export PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=true
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_PLATFORMS=cuda
export XDG_DATA_HOME=/projects/biwj/sliu51/rnd_atari_pixels/assets
export JAXTARI_SKIP_SPRITE_UPDATE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
unset LD_LIBRARY_PATH
run_directory=/projects/biwj/sliu51/rnd_atari_pixels/runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1
job_directory="$run_directory/slurm/jobs/${SLURM_JOB_ID}_${HOSTNAME}"
mkdir -p "$job_directory"
cp "$run_directory/slurm/deltaai_pixel_check.sbatch" "$job_directory/job.sbatch"
printf '%s\n' '{"unit_id":"deltaai_original_pixel_real_game_copies-1_environments-2_rollout-2_updates-2","source_commit":"dd2709f928cb22bccd23ad0395a1d60b97ff3884"}' > "$job_directory/assignment.json"
exec > >(tee "$job_directory/job.log") 2>&1
cd /projects/biwj/sliu51/rnd_atari_pixels/code/dd2709f928cb22bccd23ad0395a1d60b97ff3884
/projects/biwj/sliu51/envs/rnd_atari_pixels_jax/bin/python /projects/biwj/sliu51/rnd_atari_pixels/code/dd2709f928cb22bccd23ad0395a1d60b97ff3884/10_jax_exploration_platform/scripts/atari_pixel_campaign/run_unit.py --run-dir /projects/biwj/sliu51/rnd_atari_pixels/runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1 --unit-id deltaai_original_pixel_real_game_copies-1_environments-2_rollout-2_updates-2 --mode smoke --copies 1 --envs-per-copy 2 --rollout-steps 2 --epochs 1 --minibatches 1 --warmup-rollouts 1 --iterations 2
cp "$run_directory/data/deltaai_original_pixel_real_game_copies-1_environments-2_rollout-2_updates-2/hardware.json" "$job_directory/hardware.json"
