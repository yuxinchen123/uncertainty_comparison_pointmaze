#!/bin/bash
#SBATCH --job-name=profile-target-qualification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=jaguar03
#SBATCH --account=sl5nw_base
#SBATCH --qos=csresnolim
#SBATCH --gres=gpu:nvidia_rtx_a4500:1
#SBATCH --no-requeue
#SBATCH --reservation=sl5nw_168
#SBATCH --output=/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/slurm/jobs/profile-target-%j.out
#SBATCH --error=/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/slurm/jobs/profile-target-%j.err
set -euo pipefail
umask 077
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
cd /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1
sha256sum --check --strict /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-05-14-01_profile-source-qualification_jaguar03-rtx-a4500_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-start-1_hours-3_helper-source-2aa6df92f7d9/source_checksums.sha256
SECONDS=0
while [ ! -f /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-05-14-01_profile-source-qualification_jaguar03-rtx-a4500_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-start-1_hours-3_helper-source-2aa6df92f7d9/owned_allocation_inputs.json ]; do
  if [ "$SECONDS" -ge 60 ]; then exit 75; fi
  sleep 0.1
done
exec srun --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores --gres=gpu:nvidia_rtx_a4500:1 --kill-on-bad-exit=1 /usr/bin/python3 -I -S -B /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-05-14-01_profile-source-qualification_jaguar03-rtx-a4500_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-start-1_hours-3_helper-source-2aa6df92f7d9_controller/code/profile_target_qualification.py --prepared-unit /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-05-14-01_profile-source-qualification_jaguar03-rtx-a4500_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-start-1_hours-3_helper-source-2aa6df92f7d9_controller
