# Volatile values for the collaborator packet of the Adam learning-rate 1e-3 addendum run — the ONE
# place they live. Every packet script sources this file by absolute path; if the run folder moves or
# the environment changes, fix it here and nowhere else.
RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-05-16-05_run_5_8_1_2_addendum_rnd-origsmall-adam-lr1e-3__pointmaze-large-topright-seed900-999__antmaze-umaze-medium-bottom-left-seed0-99__beta-1e-3-to-1e4-x15_1Mstep-100seed_prune30-race100-frozenbar-vs-adam-lr1e-4_cpu-nolim"
PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"   # shared canonical env (.venvs/ENVS.md); Python 3.11
SWEEP_ID="2026-08-05-16-05_lr1e3"                                # also in $RUN_DIR/slurm/SWEEP_ID.txt
FC="$RUN_DIR/for_collaborator"
IDFILE="$FC/submitted_jobids_${SWEEP_ID}_${USER}.txt"  # per-submitter id file: YOUR ONLY legal scancel source
LOGDIR="$FC/logs"                                      # your logs (the owner's stay in $RUN_DIR/logs)
COMPLETE_SENTINEL="$RUN_DIR/SWEEP_COMPLETE"            # the owner's monitor writes this when the sweep finishes
POOL="$RUN_DIR/queue/$SWEEP_ID/pending"                # ONE pending pool: every run is 1,000,000 steps on cpu

# Job-name prefixes: RANDOM 8-lowercase-letter strings, one PER PARTITION, generated once when this
# packet was written. The same prefix within a partition, a different prefix across partitions — a
# random prefix never reveals whose sweep this is and never collides with another sweep's
# name-matching guards. Own-job discovery matches ANY of these prefixes, the per-user id file, AND
# the --comment=<sweep>_$USER recovery tag — never a meaningful name.
PREFIX_CPU="zqohtynr"     # jobs on the open cpu partition
PREFIX_NOLIM="wghqovmf"   # jobs on the open nolim partition
PREFIXES=("$PREFIX_CPU" "$PREFIX_NOLIM")
PREFIX_RE="($PREFIX_CPU|$PREFIX_NOLIM)"

# prefix_for_partition <cpu|nolim> -> echoes the job-name prefix for that partition.
prefix_for_partition() {
  case "$1" in
    cpu)   echo "$PREFIX_CPU" ;;
    nolim) echo "$PREFIX_NOLIM" ;;
    *)     echo "prefix_for_partition: unknown partition '$1'" >&2; return 1 ;;
  esac
}

# This run submits to the OPEN cpu and nolim partitions ONLY — no gpu, no gnolim, no reservation
# (the owner's instruction for this sweep). The launcher trims these ceilings live against YOUR OWN
# per-user caps with a 16-CPU headroom per pool, against node availability, and against how much
# work is actually left in the queue.
TARGET_CPU=400      # per-user cap of the cpu partition
TARGET_NOLIM=80     # per-user cap of the nolim partition
