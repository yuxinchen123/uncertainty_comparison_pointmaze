# Submission commands (command.txt holds the training script's own command)

Eight workers on jaguar03 (jobs 6728389-6728396):

    sbatch --parsable -J rd-batch1 -p gpu --reservation=sl5nw_165 --qos=csresnolim --nodelist=jaguar03 --nodes=1 --ntasks=1 -c 8 --mem=16G --gres=gpu:nvidia_rtx_a4500:1 -t 03:30:00 -o /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7/slurm/logs/worker_%j.out --wrap 'bash /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7/code/worker.sh 8'

Three workers on RTX 2080 Ti cards (jobs 6728397-6728399):

    sbatch --parsable -J rd-batch1 -p gpu --nodes=1 --ntasks=1 -c 8 --mem=16G --gres=gpu:nvidia_geforce_rtx_2080_ti:1 -t 1-00:00:00 -o /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7/slurm/logs/worker_%j.out --wrap 'bash /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7/code/worker.sh 8'

Second attempt of A3 (job 6728404), outside the queue:

    sbatch --parsable -J rd-batch1-a3b -p gpu -c 8 --mem=16G --gres=gpu:nvidia_geforce_rtx_2080_ti:1 -t 1-00:00:00 -o /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7/slurm/logs/a3_second_attempt_%j.out /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7/code/a3_second_attempt.sh
