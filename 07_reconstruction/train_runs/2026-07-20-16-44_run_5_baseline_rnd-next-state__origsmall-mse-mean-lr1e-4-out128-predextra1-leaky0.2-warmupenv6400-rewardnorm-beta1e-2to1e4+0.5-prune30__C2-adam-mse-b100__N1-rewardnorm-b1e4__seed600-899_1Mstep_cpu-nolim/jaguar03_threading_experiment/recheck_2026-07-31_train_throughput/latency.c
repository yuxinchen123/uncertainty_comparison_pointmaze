/* latency.c -- single-thread memory-latency benchmark (random pointer-chase), no deps (gcc only).
 * Builds a random permutation cycle through a buffer bigger than cache, then chases it: each load
 * depends on the previous, so the loop time is pure memory-access LATENCY (not bandwidth, not clock).
 * This is what a small-random-access workload (like SAC replay sampling) is bound by. A cache-resident
 * compute benchmark can look fast on a node whose memory latency has degraded, while this exposes it.
 *
 * Build:  gcc -O3 -o latency latency.c
 * Run  :  ./latency [buffer_MB]        (default 256 MB, well beyond L3)
 * Prints ns per dependent random access. Compare the suspect node against a healthy node.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

int main(int argc, char** argv){
    long mb = argc>1 ? atol(argv[1]) : 256;
    long n = mb*1024L*1024L/sizeof(long);          /* number of slots */
    long* a = malloc(n*sizeof(long));
    for(long i=0;i<n;i++) a[i]=i;
    /* Fisher-Yates shuffle into a single random permutation, then turn it into a pointer cycle */
    unsigned long r=88172645463325252ULL;
    for(long i=n-1;i>0;i--){ r^=r<<13; r^=r>>7; r^=r<<17; long j=r%(i+1); long t=a[i];a[i]=a[j];a[j]=t; }
    long* idx = malloc(n*sizeof(long));
    for(long i=0;i<n;i++) idx[a[i]]=a[(i+1)%n];    /* idx[p] = next node in the cycle */
    /* warm + chase */
    long p=0; long STEPS=40000000L;
    for(long i=0;i<n;i++) p=idx[p];
    double t0=now_s();
    for(long i=0;i<STEPS;i++) p=idx[p];            /* dependent random loads -> latency-bound */
    double dt=now_s()-t0;
    if(p==0xdeadbeef) printf("x");                  /* keep p live */
    printf("buffer=%ld MB  %.2f ns per random dependent access  (%.0f Maccess/s)\n",
           mb, dt*1e9/STEPS, STEPS/dt/1e6);
    return 0;
}
