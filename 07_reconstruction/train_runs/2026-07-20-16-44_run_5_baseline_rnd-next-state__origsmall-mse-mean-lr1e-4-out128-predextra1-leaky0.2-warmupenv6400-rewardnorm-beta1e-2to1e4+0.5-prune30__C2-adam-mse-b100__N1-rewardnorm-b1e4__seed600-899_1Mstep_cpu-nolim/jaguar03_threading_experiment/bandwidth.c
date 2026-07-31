/* bandwidth.c -- multi-thread sustained memory bandwidth (STREAM-triad-like), no deps (gcc + -fopenmp).
 * Each thread streams a triad a[i] = b[i] + s*c[i] over large per-thread arrays (well beyond cache), so
 * the loop is bound by sustained DRAM BANDWIDTH, not latency and not clock. Reports aggregate GB/s across
 * N threads. A latency fault and a bandwidth fault are different failures; the training uses both (random
 * sampling = latency; large tensor ops = bandwidth), so this complements latency.c. Compare suspect vs
 * healthy node at the SAME thread count.
 *
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
        nthr = omp_get_num_threads();
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
    /* triad moves 3 doubles (2 read + 1 write) * 8 bytes per element, REP times, across nthr threads */
    double gb = (double)nthr*REP*n*3.0*sizeof(double)/1e9;
    if(a[0][0]==-12345.0) printf("x");                        /* keep arrays live */
    printf("threads=%d  %.1f GB/s aggregate  (%.2f GB/s per thread)\n", nthr, gb/dt, gb/dt/nthr);
    return 0;
}
