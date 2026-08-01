# Volatile values for the Run 8.1 (point maze + ant maze) collaborator packet — the ONE place they
# live. Every packet script sources this file; if the run folder moves or the env changes, fix it here.
RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-07-23-01-35_run_8_1_pointmaze-antmaze-8env-start-bottom-left_sac__rnd-origsmall__gt-position-velocity__gt-position-maze-cell__gt-position-1m__beta-1e-3-to-1e4-x15_prune20-winner100-max300_1Mstep_default-max-episode_reward-shift-1_gamma0.99"
PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"   # shared canonical env (see .venvs/ENVS.md); Python 3.11
SWEEP_ID="2026-07-23-02-05_pm-am-run1"
FC="$RUN_DIR/for_collaborator"
IDFILE="$FC/submitted_jobids_${SWEEP_ID}_${USER}.txt"   # per-submitter id file: YOUR ONLY legal scancel source
LOGDIR="$FC/logs"                                        # collaborator logs (the owner's logs stay in $RUN_DIR/logs)
COMPLETE_SENTINEL="$RUN_DIR/SWEEP_COMPLETE"             # owner's monitor_loop.sh touches this when the queue drains
                                                        # (run 8.1 has no optuna/ dir — the sentinel sits at the run root)

# Job-name prefixes: RANDOM 8-lowercase-letter strings, one PER PARTITION, generated once at packet
# time (skill rule 2026-07-12). The same prefix within a partition, a different prefix across
# partitions — a random prefix never reveals whose sweep this is and never collides with another
# sweep's name-matching guards. Own-job discovery (monitor / refresh) matches ANY of these prefixes,
# the per-user id file, AND the --comment=<sweep>_$USER recovery tag — never a meaningful name.
PREFIX_CPU="gyzoxeup"     # jobs on the open cpu partition
PREFIX_NOLIM="vbdcjdtp"   # jobs on the open nolim partition
PREFIX_GPU="jhiznywz"     # CPU-ONLY jobs placed on low-capability gpu-partition nodes (never a GPU job)
PREFIXES=("$PREFIX_CPU" "$PREFIX_NOLIM" "$PREFIX_GPU")
# a single regex that matches a job name from any of the three prefixes (used by monitor/refresh)
PREFIX_RE="($PREFIX_CPU|$PREFIX_NOLIM|$PREFIX_GPU)"

# prefix_for_partition <cpu|nolim|gpu> -> echoes the job-name prefix for that partition.
prefix_for_partition() {
  case "$1" in
    cpu)   echo "$PREFIX_CPU" ;;
    nolim) echo "$PREFIX_NOLIM" ;;
    gpu)   echo "$PREFIX_GPU" ;;
    *)     echo "prefix_for_partition: unknown partition '$1'" >&2; return 1 ;;
  esac
}

# Per-partition CPU targets (CUMULATIVE ceilings the launcher never lets YOUR usage exceed; the
# launcher trims these live against your OWN caps with a 16-CPU headroom per pool, node availability,
# and the remaining queue depth). Collaborators use the OPEN partitions of their own pools ONLY:
#   - cpu   (open; per-user cap 400 CPUs)
#   - nolim (open; per-user cap  80 CPUs)
#   - gpu   (open; per-user cap 400 CPUs) — CPU-ONLY worker jobs (--gpus-per-node=0) on the LEAST
#           capable gpu nodes first, so good GPUs' CPUs stay free for real GPU jobs.
# NO owner reservation (puma01/jaguar03), NO gnolim, and NO GPU-using jobs (the sweep's GPU packing
# count W is still being probed by the owner — device=cuda submission stays owner-only for now).
TARGET_CPU=0     # 2026-08-01 yuxinchen: user ordered full stop of my wave (capacity moves to run 8.1.2)
TARGET_NOLIM=0   # 2026-08-01 yuxinchen: full stop (was 80)
TARGET_GPU=0     # gpu-partition CPU-ONLY jobs only; set to 0 to disable the gpu-partition bucket
                 # 2026-08-01 yuxinchen: full stop (was 400)
