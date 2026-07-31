# jaguar03: memory bandwidth is degraded (diagnosis + 30-second reproducer)

**Date:** 2026-07-23
**Node:** jaguar03 (gpu partition)
**Reporter:** Shuze

## Summary

jaguar03's main memory delivers about **5–12× less bandwidth** than other nodes — including
nodes whose memory is *lower* spec than jaguar03's — and its bandwidth **does not increase when
more threads are added**. Memory latency on the node is also about 4× worse. CPU compute (AVX/FMA)
runs at full speed, and the node does not crash. So this is not a CPU, thermal, or software problem:
it is a fault in the memory subsystem.

The practical effect: my training jobs do a lot of small, memory-heavy work, so they run roughly
8–10× slower on jaguar03 than on any other node. The same jobs completed normally on jaguar03 in
early July, so this developed recently.

A self-contained test that reproduces the problem in about 30 seconds is included at the bottom.

## Measurement (each test was the only job on the node)

To rule out interference from other jobs, each node was **completely idle**, and I ran the
**1-thread** test and the **4-thread** test **one at a time** (never together). Numbers are from a
simple STREAM-style memory bandwidth test; the absolute value depends on the test, but both nodes ran
the identical program, so the ratio between them is what matters.

| Node | CPU (memory spec) | bandwidth, 1 thread | bandwidth, 4 threads | latency, 1 thread |
|---|---|---|---|---|
| **jaguar03** | 2× AMD EPYC 7663 — 8-channel DDR4-3200 | **2.6 GB/s** | **2.0 GB/s** | **432 ns** |
| adriatic06 (healthy) | 2× Intel Xeon Silver 4208 — 6-channel DDR4-2400 | 12.3 GB/s | 24.8 GB/s | 111 ns |

Two things stand out:

1. **jaguar03 is far slower:** about 5× lower bandwidth at 1 thread, and about 12× lower at
   4 threads.
2. **jaguar03's bandwidth does not scale:** it goes *down* slightly from 1 to 4 threads
   (2.6 → 2.0 GB/s), while the healthy node goes up (12.3 → 24.8 GB/s). A healthy memory system
   delivers more total bandwidth as more cores ask for it. jaguar03 behaves as if it is already
   saturated at a single thread — consistent with most of its memory channels not contributing.

## Why this is not simply "jaguar03 is a big node, so it is slower by nature"

1. **jaguar03's memory is higher spec than the healthy comparison node, not lower.** jaguar03 has
   8 memory channels of DDR4-3200 per socket (about 410 GB/s rated for the node); adriatic06 has
   6 channels of DDR4-2400 per socket (about 230 GB/s rated). jaguar03 is rated for roughly **1.8×
   more** memory bandwidth, yet measures 5–12× **less**. If anything it should be faster.
2. **It is a lone outlier across the whole cluster.** I ran the same test on a range of nodes from
   16 to 256 cores; they all land 5–8× above jaguar03. That includes cheetah04, which has *more*
   cores than jaguar03 (256 vs 224) and measures about 8× more bandwidth. Node size does not explain
   this — only jaguar03 is slow.
3. **The CPU is fine.** A pure-compute (AVX/FMA) test runs at full speed on jaguar03, so the cores
   and clocks are healthy. The deficit is specifically in main-memory bandwidth (and latency).

## Why SLURM did not flag it

SLURM checks the *amount* of memory and the GPU count. On jaguar03 those are still correct — the
full 1 TB is present — so there is no mismatch for SLURM to catch. What is wrong is the memory
*bandwidth and latency*, which SLURM does not monitor.

## Reproduce it (about 30 seconds, no dependencies)

On jaguar03 and on any other node, with the node otherwise idle, save the program below as
`bandwidth.c`, then build and run it once with 1 thread and once with 4 threads (separately):

```bash
gcc -O3 -fopenmp -o bandwidth bandwidth.c
OMP_NUM_THREADS=1 ./bandwidth 128     # 1-thread test
OMP_NUM_THREADS=4 ./bandwidth 96      # 4-thread test (run after the 1-thread test, not together)
```

Expected result: on jaguar03 about **2–3 GB/s**, and flat or lower going from 1 to 4 threads; on any
healthy node about **12 GB/s** at 1 thread, rising as threads increase.

```c
/* bandwidth.c -- sustained memory bandwidth (STREAM-triad), needs only gcc + -fopenmp.
 * Each thread streams a[i] = b[i] + s*c[i] over large per-thread arrays (well beyond cache), so the
 * loop is bound by sustained DRAM bandwidth, not by cache, latency, or clock.
 * Build:  gcc -O3 -fopenmp -o bandwidth bandwidth.c
 * Run  :  OMP_NUM_THREADS=<n> ./bandwidth [MB_per_thread]   (default 128 MB/thread)
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

int main(int argc, char** argv){
    long mb = argc>1 ? atol(argv[1]) : 128;                 /* per-thread array size */
    long n  = mb*1024L*1024L/sizeof(double);                /* elements per array per thread */
    int nthr = 1;
    #pragma omp parallel
    {
        #pragma omp master
        nthr = omp_get_num_threads();     /* record how many threads OpenMP gave us */
    }
    /* per-thread arrays so threads never share cache lines */
    double** a=malloc(nthr*sizeof(double*)); double** b=malloc(nthr*sizeof(double*)); double** c=malloc(nthr*sizeof(double*));
    #pragma omp parallel
    {
        int t=omp_get_thread_num();
        a[t]=malloc(n*sizeof(double)); b[t]=malloc(n*sizeof(double)); c[t]=malloc(n*sizeof(double));
        for(long i=0;i<n;i++){ b[t][i]=1.0; c[t][i]=2.0; a[t][i]=0.0; }   /* first-touch = NUMA-local */
    }
    const double s=3.0; int REP=20;
    double t0=now_s();
    #pragma omp parallel
    {
        int t=omp_get_thread_num();
        for(int r=0;r<REP;r++)
            for(long i=0;i<n;i++) a[t][i]=b[t][i]+s*c[t][i];   /* triad: 2 reads + 1 write per element */
    }
    double dt=now_s()-t0;
    double gb = (double)nthr*REP*n*3.0*sizeof(double)/1e9;
    if(a[0][0]==-12345.0) printf("x");                        /* keep arrays live */
    printf("threads=%d  %.1f GB/s aggregate  (%.2f GB/s per thread)\n", nthr, gb/dt, gb/dt/nthr);
    return 0;
}
```

## Suggested checks

The evidence points at the memory subsystem, so the useful things to check are:

1. DIMM population and speed (`dmidecode -t memory`) — whether a channel has downclocked or DIMMs
   are populated so that most channels are inactive.
2. Correctable memory-error counts (EDAC / `ras-mc-ctl --error-count`) — a failing DIMM or channel
   often shows a rising correctable-error count and gets throttled or retrained.
3. BIOS/BMC memory-training and event logs.
4. A memtest pass.
5. A **full power cycle** (not just a reboot), which re-trains the memory controller and often clears
   this kind of fault.

You are welcome to take jaguar03 out of my reservation while you look — nothing of mine is running on
it now, and I am happy to run any test workload you would like.
