# Submission commands (command.txt holds the training script's own command)

See the run's slurm/ id files for the ids. Workers are generic and claim from queue/.

    sbatch --parsable -J <rd-p4a|rd-p4b> -p gpu --nodes=1 --ntasks=1 -c 8 --mem=16G --gres=gpu:nvidia_geforce_rtx_2080_ti:1 -t 1-00:00:00 -o /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-19-27_reward-decline-confirmation-part-a_large-cap400_learning-rate-1e-3_steps-100M_rnd-weights-1-10-100-1000_visit-count-sqrt-weights-1-10_seeds-0-127_control-a8-b5-c2/slurm/logs/worker_%j.out --wrap 'bash /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-19-27_reward-decline-confirmation-part-a_large-cap400_learning-rate-1e-3_steps-100M_rnd-weights-1-10-100-1000_visit-count-sqrt-weights-1-10_seeds-0-127_control-a8-b5-c2/code/worker.sh 8'

Part A: 4 such jobs under sl5nw and 4 under yuxinchen (submit_as.sh uva-yuxinchen, id file
slurm/submitted_jobids_phase-4_yuxinchen.txt). Part B: 4 under sl5nw on RTX 2080 Ti cards, and 4
under sl5nw asking --gres=gpu:nvidia_rtx_a4500:1 without the reservation, which start on jaguar03
once the sl5nw_165 reservation ends (2026-10-19 20:59 PT).
