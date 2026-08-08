# Measured per-worker facts for this run

Everything below is measured on this cluster from the two runs that share this run's trainer,
environment stack and step budget — train run 5 (PointMaze, 300 completed 1M-step runs) and train
run 1.2 (AntMaze, 6,210 completed 1M-step runs). Use them to size any new node-class script, and to
judge whether a job you are watching is healthy.

## One run

| fact | value |
|---|---|
| what a run is | one configuration and one seed: 1,000,000 env steps of SAC with an RND bonus |
| cores per run | 1 (`--cpus-per-task=1`); the trainer barely parallelizes and the thread caps are set to 1 in every worker script |
| wall time, PointMaze | median 17.2 h, maximum observed 20.9 h (n = 300, train run 5) |
| wall time, AntMaze | median 19.6 h, maximum observed 28.2 h (n = 400 sampled, train run 1.2) |
| memory, resident | plateaus at about 1.2 GB per worker on cpu; the replay buffer (1e6 transitions) dominates and does not grow after it fills |
| memory asked | 2 GB per worker where the node can afford it; the planner drops to 92% of `RealMemory / CPUEfctv` on high-core-count nodes and skips any node that cannot give a worker 1.4 GB |
| checkpoints | none. A run killed mid-flight is re-queued and re-run from the start, so a job's walltime should be the partition maximum |
| output | one JSON per run under `data/<sweep>/local/`, rewritten atomically at every 50,000-step evaluation with `completed: false`, and once at the end with `completed: true` |

## Healthy-worker signs

- `ps -o pcpu` on the node shows about 100% per worker (`--cpus-per-task=1`). Half that means two
  tasks landed on one hardware thread — the `--ntasks-per-core=2` flag was lost.
- `sacct -j <id> --format=AllocCPUS` equals the job's `--ntasks`. Double means the same thing.
- The job log gets a `claimed ...` line within a minute of the worker's startup jitter (up to 30 s),
  then nothing for hours — that is normal; the next line is the finish line 17–20 h later.
- The sweep's live progress signal is the record count, not the job log:
  `ls <run>/data/<sweep>/local/*.json | wc -l`.

## Thread policy

Every worker script exports `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. This
project's training gains only about 1.3x from a second thread while a second independent worker on
the core's sibling thread gains far more, so one thread per worker and two workers per core is the
measured best use of a core.

Every worker script also exports `PYTHONNOUSERSITE=1`. That is not hygiene: the owner's home
directory carries a `site-packages` that sits ahead of the shared environment on `sys.path` and
supplies a different torch build, while your uid has no such directory. Without the variable, the
owner's workers and yours would run different torch versions inside one sweep.

## Node sizing (what the planner does for you)

`slurm/plan_jobs.py` reads `scontrol show node` live and derives every job's shape:

- `--ntasks` ≤ the node's physical cores (`Sockets × CoresPerSocket`) — sbatch rejects more, whatever
  the thread count says;
- `--ntasks` ≤ its free allocatable threads (`CPUEfctv − CPUAlloc`); `CPUEfctv` is `CPUTot` minus the
  system's reserved core, so a set of jobs summing to `CPUTot` leaves the last one pending forever;
- `--ntasks-per-core=2` only where `ThreadsPerCore` is 2;
- shapes are capped at 32 tasks per job and floored at 6;
- reserved nodes are excluded entirely — a normal job cannot run on one and would pend forever.
