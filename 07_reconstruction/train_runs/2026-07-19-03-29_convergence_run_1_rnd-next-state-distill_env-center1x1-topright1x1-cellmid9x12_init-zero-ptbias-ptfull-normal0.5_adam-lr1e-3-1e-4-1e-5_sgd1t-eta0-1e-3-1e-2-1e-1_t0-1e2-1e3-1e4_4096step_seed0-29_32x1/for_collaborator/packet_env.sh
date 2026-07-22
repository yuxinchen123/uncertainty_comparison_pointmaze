# Volatile values for the Convergence-run-1 collaborator packet — the ONE place they live.
# Every packet script sources this file; if the run folder moves or the env changes, fix it here
# (or ask the owner to regenerate the packet with the collab-handbook skill).
RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-07-19-03-29_convergence_run_1_rnd-next-state-distill_env-center1x1-topright1x1-cellmid9x12_init-zero-ptbias-ptfull-normal0.5_adam-lr1e-3-1e-4-1e-5_sgd1t-eta0-1e-3-1e-2-1e-1_t0-1e2-1e3-1e4_4096step_seed0-29_32x1"
PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"   # shared canonical env (see .venvs/ENVS.md)
SWEEP_ID="2026-07-19-03-32_convergence-run1"
FC="$RUN_DIR/for_collaborator"
IDFILE="$FC/submitted_jobids_${SWEEP_ID}_${USER}.txt"  # per-submitter id file: YOUR only legal scancel source
LOGDIR="$FC/logs"                                       # collaborator logs (owner logs stay in $RUN_DIR/logs)

# Job-name prefixes: RANDOM 8-lowercase-letter strings generated once at packet time, ONE PER
# PARTITION (same prefix within a partition, different across partitions). They are deliberately
# meaningless — a job name must not reveal that a collaborator is submitting for the owner's sweep,
# and a random string can never collide with another sweep's name-matching guards. NEVER edit these
# to a descriptive name. Own-job discovery matches ANY of these prefixes + the per-user id file +
# the --comment=<sweep>_$USER tag. The owner's own jobs use OWNER_JOBNAME (queue-depth guard only).
PREFIX_CPU="ggxcpfux"
PREFIX_GPU="xkdgymoq"
PREFIX_NOLIM="uepebczt"
ALL_PREFIXES="$PREFIX_CPU $PREFIX_GPU $PREFIX_NOLIM"
OWNER_JOBNAME="convrate1"   # the owner's launch_queue.sh base job name (queue-depth guard counts it)

# echo the job-name prefix for a partition (used by the launcher when it sets --job-name)
prefix_for_partition() {
  case "$1" in
    cpu)   echo "$PREFIX_CPU"   ;;
    gpu)   echo "$PREFIX_GPU"   ;;
    nolim) echo "$PREFIX_NOLIM" ;;
    *)     echo "$PREFIX_CPU"   ;;
  esac
}

# echo an anchored egrep alternation of MY prefixes (matches a bare job-name field exactly), used to
# find my own jobs in squeue for the id-file refresh and my-slot counts.
my_prefix_regex() {
  local p out=""
  for p in $ALL_PREFIXES; do out="${out:+$out|}$p"; done
  echo "^($out)$"
}

# echo the same alternation PLUS the owner's base name, for the queue-depth guard (ALL submitters).
all_slots_regex() {
  local p out="$OWNER_JOBNAME"
  for p in $ALL_PREFIXES; do out="$out|$p"; done
  echo "^($out)$"
}

# First-wave partition-CPU targets (CUMULATIVE ceilings, not per-cycle additions — usage never
# exceeds the target). The launcher trims them live against your OWN caps (16-CPU headroom kept per
# pool), live node availability, and the REMAINING unclaimed queue depth. This sweep is tiny (4320
# runs of ~15 s each ~= 18 core-hours) and may already be complete, so the depth guard usually
# trims these to near zero — that is the expected clean no-op, not an error.
TARGET_CPU=128
TARGET_GPU=64
TARGET_NOLIM=32
