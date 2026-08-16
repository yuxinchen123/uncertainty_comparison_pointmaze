## Purpose

Answer whether a processor with fewer but individually stronger cores beats the 224-thread node
already measured (jaguar03, AMD EPYC 7663) on this project's training workload, and whether it
would beat that node if it had the same number of threads.

The workload is one single-thread PyTorch process per worker, each running a small policy network
plus a physics environment. It is made of small matrix multiplications and many tiny dependent
operations, not large parallel arithmetic, so clock speed and per-core cache were expected to
matter more than core count. This run tests that expectation.

Both nodes are held exclusively, because a job sharing the node contaminates the timing and the
jaguar03 reference was itself measured on an exclusively held node.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| node under test | jaguar02: Intel Xeon Gold 6334, 2 sockets x 8 cores x 2 threads = 32 hardware threads, 1,000 GiB memory |
| why this node | the highest base clock on the cluster (3.60 GHz) and the fewest cores per socket of any Ice Lake part here, and it was completely idle (0 of 32 allocated) at submit time |
| nodes rejected | serval03 (AMD EPYC 9534, Zen 4) is in a maintenance reservation until 2026-08-31; serval06-09 (AMD EPYC 9354, Zen 4) each already carried another user's multi-day job, so none could be held exclusively; jaguar06 (Xeon Gold 5317, 3.00 GHz) went from idle to allocated during the search; every other idle node is Skylake, Broadwell or Haswell, older than both nodes compared here |
| reference node | jaguar03: AMD EPYC 7663, 2 sockets x 56 cores x 2 threads = 224 hardware threads, 1,000 GiB memory |
| exclusivity | `--exclusive` on both; the jaguar03 job additionally sits inside reservation sl5nw_156, whose IGNORE_JOBS and SPEC_NODES flags already exclude other users |
| partition / walltime | gpu with `--gpus-per-node=0`; `--time=4-00:00:00`, the smallest of the gpu partition limit (4 days), the next maintenance window (none covers either node; the only maintenance reservation covers serval03 alone) and the reservation end (2026-08-19 23:59, further out than 4 days) |
| parallel form | independent worker processes, one thread each (`--mode processes`), never threads inside one process |
| update convention | one update per batch (`--style full_batch`) |
| copies per worker | 1 (`--copies-per-proc 1`) |
| worker counts, jaguar02 | 1, 2, 4, 8, 16, 32. 16 is one worker per physical core; 32 uses both hardware threads of every core |
| worker counts, jaguar03 | 1, 2, 4, 16 measured here; 8, 32, 112 and 224 already measured in `benchmarks/results/2026-08-15-18-53-15_trainbench_cpu_processes_j3_p1_styleA.json` |
| measurement | median of 5 timed iterations after 2 warm-up iterations; each iteration collects 512 environment steps per copy |
| implementation | unchanged trainer with every graphics-processor-only feature off: no recorded operation sequences, no reduced-precision matrix mode, no fused optimiser, eager rollout |
| system-reserved cores | both nodes set `CoreSpecCount=1`, so a job sees 30 of 32 processors on jaguar02 and 222 of 224 on jaguar03. The top worker count on each node therefore exceeds the visible processors by exactly 2, identically on both |

## Code and config changes

- No change to the trainer or to `benchmarks/bench_train_cpu.py`; the run only calls the existing
  script with new worker counts and a new tag.
- Job scripts `slurm/cpu_train_jaguar02.slurm` and `slurm/cpu_train_jaguar03_fill.slurm`; job ids
  appended at submit time to `slurm/submitted_jobids_cpu_node.txt`.
- Each job first writes what the machine says about itself to `logs/hardware_<node>.txt` (`lscpu`,
  `lscpu -C`, `/proc/cpuinfo`, `numactl -H`, `getconf` cache sizes, `dmidecode -t memory`, the
  cpufreq driver and governor), so every hardware figure in the comparison can be traced to a
  command run on the node it describes.
- Each job appends one line per finished configuration to `logs/progress.log`, and the benchmark
  runs under `python -u`, so a row appears in the file as soon as its worker count is timed.
- The first jaguar03 submission (6537879) failed after 2 seconds: the hardware-reporting block ends
  several pipelines in `head`, and on a 224-processor node `head` exits before `grep` has finished
  writing, which under `set -o pipefail` aborted the job. The reporting block now runs in a
  subshell with the error and pipefail options off. Resubmitted as 6537881.

## Git state

Commit: `d9208327d6c7580c453f5deb4b376068b1069db5` (branch `Use-RLexplore-RND`).

The working tree is dirty: three other sessions are editing this repository at the same time. The
exact commit each result file was produced from is recorded in the `git` field inside the file.

## What the run added after it started

Two things were added once the first results were in, both because the first results could not
be trusted as they stood.

1. **A hardware probe on each node** (`code/cpu_probe.c`, job script `slurm/probe_node.slurm`).
   `lscpu` gives cache sizes but not the clock a core actually runs at, and jaguar02 exposes no
   cpufreq interface at all — `/proc/cpuinfo` there reports a fixed nominal 3600 whatever the
   processor is doing. The probe times a chain of additions in which each addition needs the
   previous one's result, which advances one addition per clock cycle, so additions divided by
   seconds is the clock. It also times one dependent memory access and one core's streaming read
   at four working-set sizes. It is written in C because a dependent memory access takes about
   100 nanoseconds and a Python loop iteration about 30, so a Python version would have measured
   Python. The method checks out against a known value: on jaguar03 it read 3.52 GHz against that
   part's rated 3.5 GHz boost. The probe was run twice per node — once idle, once while the real
   training workload occupied every other core.
2. **Sustained measurements** (`slurm/sustained_node.slurm`, `slurm/confirm_node.slurm`). The
   five-iteration setting used by the earlier processor work times about two seconds of work.
   Two effects make that read high, and both grow with the worker count: a server processor runs
   above its sustained clock for the first seconds of a load, and over seven iterations several
   hundred worker processes are still starting at different moments, so the benchmark's sum of
   each worker's own rate describes a load that never existed. Every point was therefore measured
   again at 150 iterations, and every worker count the argument rests on was measured three or
   four times.

## Outcome

All jobs completed without error. Job ids in `slurm/submitted_jobids_cpu_node.txt`; one early
failure (6537879) is explained in the section above. Result files carry the tags `_j2_`, `_j3_`,
`_sustained_` and `_confirm_` under `benchmarks/results/`.

**The five-iteration numbers were wrong, and wrong worst where the argument is decided.** At 150
iterations jaguar03 falls 27% at 112 workers and 68% at 224; jaguar02 is unaffected, because 16
cores can hold a clock that 112 cannot (jaguar02 measured 3.56 GHz with every core busy against
its 3.59 idle; jaguar03 2.60 against 3.52).

Sustained results, independent worker processes, one copy each, one update per batch:

| node | one worker per physical core | one worker per hardware thread | second thread |
|---|---|---|---|
| jaguar02 (16 cores / 32 threads) | 16 workers, 0.0204 M steps/s, 1.27 k per copy | 32 workers, 0.0230 M, 0.72 k per copy | **+12%** |
| jaguar03 (112 cores / 224 threads) | 112 workers, 0.1296 M steps/s, 1.16 k per copy | 224 workers, 0.1138 M, 0.51 k per copy | **-12%** |

So jaguar03's best setting is 112 workers, not 224 — filling its second hardware threads makes
it slower. The five-iteration numbers had reported that same step as nearly doubling its total.

At equal load jaguar02 gives one worker 10% more steps per second on a clock 37% higher, so
jaguar03 does 1.24 times as much of this work per clock cycle. The difference tracks the
last-level cache and nothing else measured here: at an 8 MiB working set one dependent access
takes 44.3 ns on jaguar02 against 15.6 ns on jaguar03, and jaguar03 has 4.57 MiB of level-3 cache
per core against 2.25 MiB. It is not the vector units — the array library compiled itself for
AVX512 on jaguar02 and only AVX2 on jaguar03, and the AVX512 machine is the one doing less per
cycle, which is what a workload of many small dependent operations looks like.

Scaled to 112 cores, jaguar02's throughput per physical core projects to 0.161 M steps/s, about
124% of jaguar03's best. That projection assumes the clock, the memory bandwidth per core and the
cache per core all survive a sevenfold increase in core count, and jaguar03 is itself the
demonstration that they do not, so it is an upper bound.

These measurements feed the section generated by
`report/2026-08-15-pointmaze-gpu-parallelization/code/cpu_node_comparison.py`
(`sec_cpu_node_comparison()`), which reads every number from the result files rather than
carrying any of them in its text.
