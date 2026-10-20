# Submission commands (command.txt holds the training script's own command)

Canary (job 6728253, after 6728236 failed and 6728252 was moved off a full node):

    sbatch --parsable -J rd-screen-canary -p gpu -c 8 --mem=16G --gres=gpu:nvidia_geforce_rtx_2080_ti:1 -t 00:40:00 -o /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-16-18_map-screening_umaze-cap150_medium-cap250_medium-cap400_large-cap400_rnd-weights-1-10-100-1000_visit-count-sqrt-weight-1_no-bonus_learning-rate-1e-3_steps-30M-large-50M_seeds-0-127/slurm/logs/canary_%j.out /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-16-18_map-screening_umaze-cap150_medium-cap250_medium-cap400_large-cap400_rnd-weights-1-10-100-1000_visit-count-sqrt-weight-1_no-bonus_learning-rate-1e-3_steps-30M-large-50M_seeds-0-127/code/canary.sh

Workers, 12 times (jobs 6728254-6728265), generic, claiming from queue/:

    sbatch --parsable -J rd-screen -p gpu --nodes=1 --ntasks=1 -c 8 --mem=16G --gres=gpu:nvidia_geforce_rtx_2080_ti:1 -t 1-00:00:00 -o /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-16-18_map-screening_umaze-cap150_medium-cap250_medium-cap400_large-cap400_rnd-weights-1-10-100-1000_visit-count-sqrt-weight-1_no-bonus_learning-rate-1e-3_steps-30M-large-50M_seeds-0-127/slurm/logs/worker_%j.out --wrap 'bash /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-16-18_map-screening_umaze-cap150_medium-cap250_medium-cap400_large-cap400_rnd-weights-1-10-100-1000_visit-count-sqrt-weight-1_no-bonus_learning-rate-1e-3_steps-30M-large-50M_seeds-0-127/code/worker.sh 8'
