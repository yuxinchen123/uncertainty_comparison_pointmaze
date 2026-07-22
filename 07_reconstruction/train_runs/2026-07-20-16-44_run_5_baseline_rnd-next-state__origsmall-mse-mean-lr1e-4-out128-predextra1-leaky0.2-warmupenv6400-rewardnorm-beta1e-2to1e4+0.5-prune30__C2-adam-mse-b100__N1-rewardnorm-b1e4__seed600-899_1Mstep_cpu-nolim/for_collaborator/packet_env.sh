# Volatile values for the Train run 5 collaborator packet — the ONE place they live.
# Every packet script sources this file; if the run folder moves or the env changes, fix it here.
RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-07-20-16-44_run_5_baseline_rnd-next-state__origsmall-mse-mean-lr1e-4-out128-predextra1-leaky0.2-warmupenv6400-rewardnorm-beta1e-2to1e4+0.5-prune30__C2-adam-mse-b100__N1-rewardnorm-b1e4__seed600-899_1Mstep_cpu-nolim"
PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"   # shared canonical env (see .venvs/ENVS.md)
SWEEP_ID="2026-07-20-16-55_set-baseline"
PREFIX="collab"        # collaborator job-name prefix (collab1..) — distinct from the owner's run5cpu/run5nl
FC="$RUN_DIR/for_collaborator"
IDFILE="$FC/submitted_jobids_${SWEEP_ID}_${USER}.txt"  # per-submitter id file: YOUR only legal scancel source
LOGDIR="$FC/logs"                                       # collaborator logs (owner logs stay in $RUN_DIR/logs)
# Partition-CPU targets. User directive 2026-07-20: cpu + nolim ONLY (NO gpu partition, NO gnolim,
# NO reservation). The launcher trims these live against YOUR own caps (16-CPU headroom per pool),
# node availability, and remaining queue depth.
TARGET_CPU=400
TARGET_GPU=0    # NO gpu-partition jobs (user 2026-07-20)
TARGET_NOLIM=80
