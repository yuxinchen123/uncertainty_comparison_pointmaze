#!/usr/bin/env bash
#SBATCH --account=158590754315
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --job-name=rnd-atari-aces-reference-d28e673
#SBATCH --output=/scratch/user/u.sl359075/rnd_atari_pixels/runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/slurm/aces_reference_%j.out
#SBATCH --error=/scratch/user/u.sl359075/rnd_atari_pixels/runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/slurm/aces_reference_%j.out
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export ATARI_PIXEL_ASSETS=/scratch/user/u.sl359075/rnd_atari_pixels/assets
source /scratch/user/u.sl359075/rnd_atari_pixels/code/d28e6735e79788f618acb0ebfde8609db1f776af/10_jax_exploration_platform/scripts/atari_pixel_campaign/worker_environment.sh
export JAX_PLATFORMS=cuda
run_directory=/scratch/user/u.sl359075/rnd_atari_pixels/runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1
job_directory="$run_directory/slurm/jobs/${SLURM_JOB_ID}_${HOSTNAME}"
mkdir -p "$job_directory"
cp "$run_directory/slurm/aces_reference_d28e673.sbatch" "$job_directory/job.sbatch"
printf '%s\n' '{"unit_id":"aces_reference_copies_1_source_d28e6735e797","source_commit":"d28e6735e79788f618acb0ebfde8609db1f776af","classification":"resource_probe_only"}' > "$job_directory/assignment.json"
exec > >(tee "$job_directory/job.log") 2>&1
cd /scratch/user/u.sl359075/rnd_atari_pixels/code/d28e6735e79788f618acb0ebfde8609db1f776af
set +e
/scratch/user/u.sl359075/envs/rnd_atari_pixels_jax/bin/python /scratch/user/u.sl359075/rnd_atari_pixels/code/d28e6735e79788f618acb0ebfde8609db1f776af/10_jax_exploration_platform/scripts/atari_pixel_campaign/run_unit.py --run-dir /scratch/user/u.sl359075/rnd_atari_pixels/runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1 --unit-id aces_reference_copies_1_source_d28e6735e797 --mode benchmark --configuration /scratch/user/u.sl359075/rnd_atari_pixels/code/d28e6735e79788f618acb0ebfde8609db1f776af/10_jax_exploration_platform/configs/experiments/montezuma_pixels_original_cleanrl.toml --copies 1 --envs-per-copy 128 --rollout-steps 128 --epochs 4 --minibatches 4 --warmup-rollouts 50 --rounds 7 --iterations 10 --seed-start 1
probe_status=$?
set -e
if test -f "$run_directory/data/aces_reference_copies_1_source_d28e6735e797/hardware.json"; then
    cp "$run_directory/data/aces_reference_copies_1_source_d28e6735e797/hardware.json" "$job_directory/hardware.json"
fi
printf '%s\n' "$probe_status" > "$job_directory/exit_code.txt"
exit "$probe_status"
