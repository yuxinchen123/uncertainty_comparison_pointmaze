## Purpose

Measure how fast the batched PointMaze environment runs on ordinary processor cores, so the
graphics-processor numbers in the main report have a reference point. Two questions: what
aggregate throughput does a whole 160-processor node reach, and what rate does an individual
environment advance at.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| node | puma01 (2 sockets x 40 cores x 2 threads = 160 logical processors, Ice Lake, 246 GiB) |
| allocation | one job, 80 cores requested (a request for 160 logical processors is rejected: a single task may not exceed the node's physical core count) |
| reservation | sl5nw_156 (covers jaguar03 and puma01, ends 2026-08-19) |
| partition / qos | cpu / csresnolim |
| walltime | 4 hours |
| implementation measured | the PyTorch environment (`pointmaze/torch_env`), unchanged, on the processor |
| thread study | one process given 1, 2, 4, 8, 16, 32, 64, 80, 160 threads, at 10,000 and 100,000 environments |
| process study | 1 to 160 independent processes, one thread each, 1,000 and 10,000 environments per process |
| measurement | each point runs for a fixed 6-second budget after a warm-up, and reports steps completed |

## Code and config changes

New benchmark `benchmarks/bench_env_cpu.py`; job script in `slurm/cpu_bench.slurm`. No change to
the environment implementation — the same code that runs on the graphics processor runs here with
the device set to the processor.

## Git state

Branch Use-RLexplore-RND at the commit recorded in each result file's `git` field.
