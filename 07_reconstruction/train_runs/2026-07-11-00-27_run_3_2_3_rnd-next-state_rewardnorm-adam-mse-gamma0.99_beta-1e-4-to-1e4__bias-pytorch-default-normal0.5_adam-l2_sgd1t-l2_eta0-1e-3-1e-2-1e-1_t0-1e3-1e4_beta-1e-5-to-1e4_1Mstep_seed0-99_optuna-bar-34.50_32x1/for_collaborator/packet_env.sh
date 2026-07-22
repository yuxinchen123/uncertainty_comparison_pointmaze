# Volatile values for the run-3.2.3 collaborator packet — the ONE place they live.
# Every packet script sources this file; if the run folder moves or the env changes, fix it here
# (or ask the owner to regenerate the packet with the collab-handbook skill).
RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-07-11-00-27_run_3_2_3_rnd-next-state_rewardnorm-adam-mse-gamma0.99_beta-1e-4-to-1e4__bias-pytorch-default-normal0.5_adam-l2_sgd1t-l2_eta0-1e-3-1e-2-1e-1_t0-1e3-1e4_beta-1e-5-to-1e4_1Mstep_seed0-99_optuna-bar-34.50_32x1"
PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"   # shared env (canonical; see .venvs/ENVS.md)
SWEEP_ID="2026-07-11-00-57_init-rewardnorm-bar"
PREFIX="collab"        # collaborator job-name prefix (collab1..collab4) — NEVER lattice/fossil,
                       # those are the owner's names; the split keeps logs and squeue rows apart
FC="$RUN_DIR/for_collaborator"
IDFILE="$FC/submitted_jobids_${SWEEP_ID}_${USER}.txt"  # per-submitter id file: YOUR only legal scancel source
LOGDIR="$FC/logs"                                       # collaborator logs (owner logs stay in $RUN_DIR/logs)
# first-wave partition-CPU targets (CPU-mode packet, user-set 2026-07-11; cpu raised to 400 per
# user request 2026-07-12); the launcher trims them live against your own caps (16-CPU headroom
# -> effective max 384), node availability, and remaining queue depth
TARGET_CPU=400
TARGET_GPU=0    # user 2026-07-12: no gpu-partition jobs at all (cpu + nolim only)
TARGET_NOLIM=80
