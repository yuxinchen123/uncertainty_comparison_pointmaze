# Run 3.2.3 — per-worker resource facts (for sizing your own submissions)

Use these measured facts, together with the shared rule
`/p/rlprojects/.claude/skills/submit-cpu-sweep/SKILL.md`, to decide what to submit, where, and
how to pack — including writing NEW worker slurm scripts for node classes the packet does not
cover (keep new scripts inside `for_collaborator/`, source `packet_env.sh`, run
`worker_collab.py`, log to `for_collaborator/logs/`, record ids in your own id file).

| fact | value | source |
|---|---|---|
| CPUs per worker | 1 (single hardware thread; always pair with `--ntasks-per-core=2`) | this run's fleet, verified `AllocCPUS == ntasks` at launch |
| thread policy | `OMP_NUM_THREADS=1` etc. exported by the worker slurm scripts; `train.py` caps torch threads from it | `worker_*_collab.slurm` |
| GPU | none — the workload is CPU-bound SAC; gpu-partition jobs use `--gpus-per-node=0` | run design |
| memory ask per worker | `--mem-per-cpu=2G` (1900M for the 14×1 job on 64G-total nodes ai01–06, so the 16×1+14×1 pair fits: 32768 + 26600 = 59368M ≤ 64000M) | run-3.2.1/3.2.3 launches |
| memory actually used | ~1.0–1.6G resident per worker late in a run (the SAC replay buffer holds 10^6 transitions ≈ 1G); the 2G ask is the safe ceiling — do NOT ask less than 2G | prior sweep failures: under-asked memory OOM-kills workers silently |
| run duration | ~10–14 h per run (10^6 steps, 1 CPU); a worker executes runs back-to-back until the queue empties | run-3.2.1/3.2.2 timing |
| live check | `sstat -j <jobid> --format=JobID,MaxRSS -a \| sort -k2 -h \| tail -3` on one of your running jobs | — |
| job shapes | 32×1 on nodes with ≥ 32 physical cores; 16×1 + 14×1 pair on the 16-core/32-thread class (adriatic/affogato/ai; `sbatch` REJECTS ntasks > physical cores) | submit-cpu-sweep SKILL.md |
| walltime | `--time=4-00:00:00` (cpu/gpu partition maximum; nolim allows more but 4 d is plenty — workers exit when the queue empties) | partition limits |
