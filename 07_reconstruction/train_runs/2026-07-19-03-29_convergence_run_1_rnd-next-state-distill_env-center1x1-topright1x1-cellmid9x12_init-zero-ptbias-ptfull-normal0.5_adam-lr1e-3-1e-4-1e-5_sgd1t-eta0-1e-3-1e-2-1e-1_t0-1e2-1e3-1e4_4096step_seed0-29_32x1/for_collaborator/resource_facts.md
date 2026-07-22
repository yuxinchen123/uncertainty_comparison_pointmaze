# Convergence run 1 — per-worker resource facts (for sizing your own submissions)

Use these measured facts, together with the shared rule
`/p/rlprojects/.claude/skills/submit-cpu-sweep/SKILL.md`, to decide what to submit, where, and how
to pack — including writing NEW worker slurm scripts for node classes the packet does not cover
(keep new scripts inside `for_collaborator/`, source `packet_env.sh` by absolute path, run
`worker_collab.py`, log to `for_collaborator/logs/`, record ids in your own id file).

This is a much lighter and shorter workload than the run-3.2.x SAC sweeps: each run is pure RND
distillation with no RL loop, no environment stepping, and no replay buffer — so it finishes in
seconds and uses well under a gigabyte.

| fact | value | source |
|---|---|---|
| trainer | `07_reconstruction/convergence_train.py` (pure distillation; no SAC, no env stepping, no replay buffer) | run design |
| CPUs per worker | 1 (single hardware thread; always pair with `--ntasks-per-core=2`) | this run's fleet, verified `AllocCPUS == ntasks` at launch |
| thread policy | `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` exported by the worker slurm scripts; the trainer runs on CPU | `worker_*_collab.slurm` |
| GPU | none — CPU-only distillation; gpu-partition jobs use `--gpus-per-node=0` | run design |
| run duration | about 15 s per run (4096 full-batch steps, 1 CPU); a worker executes runs back-to-back until the queue empties | measured, owner fleet |
| memory used | peak resident set size about 830 MB per worker | measured, owner fleet |
| memory ask per worker | `--mem-per-cpu=2G` (1900M for the 14x1 job on 64G-total nodes ai01-06, so the 16x1 + 14x1 pair fits: 32768 + 26600 = 59368M <= 64000M). 2G is a comfortable ceiling over the ~830M peak | packet worker scripts |
| total runs | 4320 (3 point sets x 4 inits x 12 optimizer configs x 30 seeds); about 18 core-hours total. Seed-outermost run ids, no pruning (every config runs all 30 seeds) | `slurm/build_queue.py` |
| live check | `sstat -j <jobid> --format=JobID,MaxRSS -a \| sort -k2 -h \| tail -3` on one of your running jobs | -- |
| job shapes | 32x1 on nodes with >= 32 physical cores; 16x1 + 14x1 pair on the 16-core/32-thread class (adriatic/affogato/ai; `sbatch` REJECTS ntasks > physical cores) | submit-cpu-sweep SKILL.md |
| walltime | `--time=4-00:00:00` (cpu/gpu partition maximum; workers exit as soon as the queue empties, so a run needs only minutes) | partition limits |

## Note: this sweep is small and may finish before you submit

The whole sweep is about 18 core-hours. With the owner's fleet already running, the queue can drain
within the hour of launch. The launcher's queue-depth guard means that once the remaining unclaimed
work is smaller than the workers already running, it submits nothing and prints a clean "not
submitting" line. If that happens, you are done — it is the expected outcome, not an error.
