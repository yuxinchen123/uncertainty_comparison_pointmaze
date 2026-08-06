# Every volatile value for this run, in one sourced file. Sourced by ABSOLUTE path everywhere:
# under sbatch, $0 resolves to Slurm's spool copy, so a relative path finds the wrong file.
export RUN_DIR="/p/rlprojects/RND/08_cleanrl_ppo_rnd/train_runs/2026-08-05-22-30_cleanrl_run_2_ablation_ppo-rnd_montezuma-v5__arm1-orig_arm2-noclip_arm3-prop1_arm4-shallow_arm5-all__2e9step_seed1-30__log200_gradstats_ckpt1h__gpu-gnolim-resv"
export PROJ_DIR="/p/rlprojects/RND/08_cleanrl_ppo_rnd"
export PY="/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python"
export SWEEP_ID="2026-08-05-22-30_ablation"
export SUBMISSION_SCRIPT_DIR="$RUN_DIR/slurm/submission_script"
# One id file per submitter. Never shared: one writer per file is the whole coordination rule.
export IDFILE="$RUN_DIR/for_collaborator/submitted_jobids_${SWEEP_ID}_${USER}.txt"
export LOGDIR="$RUN_DIR/for_collaborator/logs"
export CPUS_PER_RUN=8
# Random job-name prefixes, one per partition, fixed when this packet was generated. A job name
# must not reveal whose sweep it belongs to, and a random prefix cannot collide with another
# sweep's name-matching guards.
export PREFIX_GPU="vlmapiik"
export PREFIX_GNOLIM="nzzxkbsi"
prefix_for_partition () { case "$1" in gnolim) echo "$PREFIX_GNOLIM";; *) echo "$PREFIX_GPU";; esac; }
export -f prefix_for_partition
