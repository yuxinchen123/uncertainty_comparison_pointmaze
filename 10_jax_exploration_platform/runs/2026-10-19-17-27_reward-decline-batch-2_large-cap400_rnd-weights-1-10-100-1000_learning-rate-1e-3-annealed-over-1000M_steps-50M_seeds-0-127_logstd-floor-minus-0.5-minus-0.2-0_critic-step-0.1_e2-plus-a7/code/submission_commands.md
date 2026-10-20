# Submission commands (command.txt holds the training script's own command)

Three workers on jaguar03 under sl5nw (jobs 6728524-6728526):

    sbatch --parsable -J rd-batch2 -p gpu --reservation=sl5nw_165 --qos=csresnolim --nodelist=jaguar03 --nodes=1 --ntasks=1 -c 8 --mem=16G --gres=gpu:nvidia_rtx_a4500:1 -t 03:15:00 -o /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-17-27_reward-decline-batch-2_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_logstd-floor-minus-0.5-minus-0.2-0_critic-step-0.1_e2-plus-a7/slurm/logs/worker_%j.out --wrap 'bash /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-17-27_reward-decline-batch-2_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_logstd-floor-minus-0.5-minus-0.2-0_critic-step-0.1_e2-plus-a7/code/worker.sh 8'

Workers on RTX 2080 Ti cards under yuxinchen (ids in slurm/submitted_jobids_batch-2_yuxinchen.txt):

    bash /p/rlprojects/.claude/skills/submit-as-account/scripts/submit_as.sh uva-yuxinchen --idfile /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-17-27_reward-decline-batch-2_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_logstd-floor-minus-0.5-minus-0.2-0_critic-step-0.1_e2-plus-a7/slurm/submitted_jobids_batch-2_yuxinchen.txt -- -J rd-batch2 -p gpu --nodes=1 --ntasks=1 -c 8 --mem=16G --gres=gpu:nvidia_geforce_rtx_2080_ti:1 -t 1-00:00:00 -o /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-17-27_reward-decline-batch-2_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_logstd-floor-minus-0.5-minus-0.2-0_critic-step-0.1_e2-plus-a7/slurm/logs/worker_%j.out --wrap 'bash /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-17-27_reward-decline-batch-2_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_logstd-floor-minus-0.5-minus-0.2-0_critic-step-0.1_e2-plus-a7/code/worker.sh 8'

The first two (6728527, 6728528) failed every unit within two seconds: git, run as yuxinchen,
refused the sl5nw-owned repository when the first unit recorded the git state. worker.sh now sets
safe.directory through GIT_CONFIG_*; the five markers went back to pending.
