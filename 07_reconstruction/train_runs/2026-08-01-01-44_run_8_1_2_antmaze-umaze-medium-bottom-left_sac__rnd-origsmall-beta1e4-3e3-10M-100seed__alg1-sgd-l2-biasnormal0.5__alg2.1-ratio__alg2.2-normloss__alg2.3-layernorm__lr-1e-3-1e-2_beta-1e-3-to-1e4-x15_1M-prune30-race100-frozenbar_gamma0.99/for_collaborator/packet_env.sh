# Volatile values for the Run 8.1.2 (AntMaze UMaze + Medium, four new RND arms vs a frozen RND bar)
# collaborator packet — the ONE place they live. Every packet script sources this file; if the run
# folder moves or the environment changes, fix it here and nowhere else.
RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99"
PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"   # shared canonical env (see .venvs/ENVS.md); Python 3.11
SWEEP_ID="2026-08-01-02-03_run812"                     # also in $RUN_DIR/slurm/SWEEP_ID.txt
FC="$RUN_DIR/for_collaborator"
IDFILE="$FC/submitted_jobids_${SWEEP_ID}_${USER}.txt"   # per-submitter id file: YOUR ONLY legal scancel source
LOGDIR="$FC/logs"                                       # collaborator logs (the owner's logs stay in $RUN_DIR/logs)
COMPLETE_SENTINEL="$RUN_DIR/SWEEP_COMPLETE"             # the owner's monitor_loop.slurm touches this when the
                                                        # sweep finishes (both pools empty, running empty, every
                                                        # task-S config decided)

# TWO pending pools under ONE sweep (this is the run-8.1.2 change over run 8.1):
#   pending_1m  — task S: 24,000 markers, 1,000,000-step runs of the four new algorithm arms.
#   pending_10m — task R: 200 markers, 10,000,000-step RND baseline runs (100 seeds x 2 environments).
# A worker's WORKER_POOLS variable (space-separated, priority order) says which pool(s) it claims
# from. Your CPU task-S shapes claim "pending_1m" only; your gnolim shapes claim
# "pending_10m pending_1m" (10M first, 1M as fallback) and the run's own worker.py decides whether a
# 10M claim is allowed from the job's REMAINING walltime (>= 170 h on cpu) — nothing to configure.
POOL_1M="$RUN_DIR/queue/$SWEEP_ID/pending_1m"
POOL_10M="$RUN_DIR/queue/$SWEEP_ID/pending_10m"

# Job-name prefixes: RANDOM 8-lowercase-letter strings, one PER PARTITION, generated once at packet
# time (skill rule 2026-07-12). The same prefix within a partition, a different prefix across
# partitions — a random prefix never reveals whose sweep this is and never collides with another
# sweep's name-matching guards. Own-job discovery (monitor / refresh) matches ANY of these prefixes,
# the per-user id file, AND the --comment=<sweep>_$USER recovery tag — never a meaningful name.
PREFIX_CPU="hxmxxwbg"     # jobs on the open cpu partition
PREFIX_NOLIM="nroewpyx"   # jobs on the open nolim partition
PREFIX_GPU="xtxolziq"     # CPU-ONLY jobs placed on low-capability gpu-partition nodes (never a GPU job)
PREFIX_GNOLIM="eiyfdxmo"  # CPU-ONLY jobs on gnolim nodes (--gpus-per-node=0; their Pascal/Maxwell
                          # GPUs cannot run torch 2.10+cu128, so gnolim is a CPU pool for this run)
PREFIXES=("$PREFIX_CPU" "$PREFIX_NOLIM" "$PREFIX_GPU" "$PREFIX_GNOLIM")
# a single regex matching a job name from any of the four prefixes (used by monitor/refresh)
PREFIX_RE="($PREFIX_CPU|$PREFIX_NOLIM|$PREFIX_GPU|$PREFIX_GNOLIM)"

# prefix_for_partition <cpu|nolim|gpu|gnolim> -> echoes the job-name prefix for that partition.
prefix_for_partition() {
  case "$1" in
    cpu)    echo "$PREFIX_CPU" ;;
    nolim)  echo "$PREFIX_NOLIM" ;;
    gpu)    echo "$PREFIX_GPU" ;;
    gnolim) echo "$PREFIX_GNOLIM" ;;
    *)      echo "prefix_for_partition: unknown partition '$1'" >&2; return 1 ;;
  esac
}

# Per-partition CPU targets (CUMULATIVE ceilings the launcher never lets YOUR usage exceed; the
# launcher trims these live against your OWN caps with a 16-CPU headroom per pool, node availability,
# and the remaining queue depth). Collaborators use the OPEN partitions of their own pools ONLY:
#   - cpu    (open; per-user cap 400 CPUs)                     -> pending_1m
#   - nolim  (open; per-user cap  80 CPUs)                     -> pending_1m
#   - gpu    (open; per-user cap 400 CPUs) — CPU-ONLY worker jobs (--gpus-per-node=0) on the LEAST
#            capable GPU nodes first, so good GPUs' CPUs stay free for real GPU jobs -> pending_1m
#   - gnolim (open; per-user cap  80 CPUs) — CPU-ONLY worker jobs -> pending_10m first, pending_1m
# NO owner reservation (puma01/jaguar03) and NO GPU-USING job: the run's task-R cuda jobs
# (--gpus-per-node=1, WORKER_DEVICE=cuda) are OWNER-ONLY and this packet does not ship them.
TARGET_CPU=400
TARGET_NOLIM=80
TARGET_GPU=400      # gpu-partition CPU-ONLY jobs only; set to 0 to disable the gpu-partition bucket
TARGET_GNOLIM=80    # gnolim CPU-ONLY jobs;              set to 0 to disable the gnolim bucket
