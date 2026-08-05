# Every volatile value for this run, in one sourced file. Sourced by ABSOLUTE path everywhere:
# under sbatch, $0 points at Slurm's spool copy, so a relative path resolves to the wrong place.
export RUN_DIR="/p/rlprojects/RND/08_cleanrl_ppo_rnd/train_runs/2026-08-05-17-45_cleanrl_train_run_1_ppo-rnd_montezuma-revenge-v5__envpool-autoreset-fixed__128env-128step-2e9step__int-coef1-ext-coef2-updateproportion0.25__seed1-30__checkpoint-8h-resumable__gpu-gnolim"
export PROJ_DIR="/p/rlprojects/RND/08_cleanrl_ppo_rnd"
export PY="/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python"
export SWEEP_ID="ppo-rnd-atari"
export SUBMISSION_SCRIPT_DIR="$RUN_DIR/slurm/submission_script"
# One id file per submitter. Never shared: one writer per file is the whole coordination rule.
export IDFILE="$RUN_DIR/for_collaborator/submitted_jobids_${SWEEP_ID}_${USER}.txt"
export LOGDIR="$RUN_DIR/for_collaborator/logs"
export CPUS_PER_RUN=8
# Random job-name prefixes, one per partition, fixed at packet generation.
export PREFIX_GPU="wrmniogq"
export PREFIX_GNOLIM="eimbancf"
prefix_for_partition () { case "$1" in gpu) echo "$PREFIX_GPU";; gnolim) echo "$PREFIX_GNOLIM";; *) echo "$PREFIX_GPU";; esac; }
export -f prefix_for_partition
