/* avxburn.c -- self-contained heavy AVX FMA load + throttle probe, no external deps (gcc only).
 *
 * Forks N workers pinned one per hardware thread; each runs a sustained dense fused-multiply-add loop
 * (with -O3 -march=native this vectorizes to AVX/AVX-512 FMA -- a high-power "power-virus" load, close
 * to the power a neural-net training draws, much hotter than a light memory loop). Each worker also
 * times a FIXED chunk of that work and writes (seconds_since_start, chunk_time_ms) to its own file, so
 * the achieved speed over time is visible WITHOUT turbostat/root: chunk_time_ms is proportional to
 * 1/frequency, so it GROWS if the node throttles under sustained load.
 *
 * Build:  gcc -O3 -march=native -o avxburn avxburn.c
 * Run  :  ./avxburn <duration_sec> <n_workers> <outdir>     e.g.  ./avxburn 600 216 /tmp/burn
 *         (n_workers 216 = 2 per core on a 112-core node; use your core count for 1-per-core.)
 * Read the *.w files: chunk_ms flat = healthy; chunk_ms rising over the run = throttling.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sched.h>
#include <signal.h>
#include <sys/wait.h>

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

/* one fixed chunk of dense FMA work; 8 independent accumulators keep the FMA units saturated. Returns a
   value derived from the accumulators so the compiler cannot optimize the loop away. */
static double fma_chunk(long iters){
    double a[8]; for(int i=0;i<8;i++) a[i]=0.1*i+1.0;
    const double b=1.0000000001, c=0.5;
    for(long i=0;i<iters;i++)
        for(int j=0;j<8;j++) a[j]=a[j]*b+c;   /* -> vectorized AVX FMA */
    double s=0; for(int j=0;j<8;j++) s+=a[j];
    return s;
}

static volatile double SINK;

/* worker: pin to `cpu`, run fma_chunk forever, every ~2 s append (elapsed, chunk_ms) to outpath. */
static void worker(int cpu, const char* outpath, double t0){
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu,&set); sched_setaffinity(0,sizeof(set),&set);
    const long CHUNK = 30000000L;              /* ~fixed work per chunk; tune for ~0.1-0.3 s at full clock */
    FILE* f = fopen(outpath,"w");
    double last = now_s();
    while(1){
        double s0 = now_s();
        SINK = fma_chunk(CHUNK);
        double dt = now_s() - s0;
        if(now_s()-last >= 2.0){
            fprintf(f,"%.3f %.3f\n", now_s()-t0, dt*1000.0); fflush(f);
            last = now_s();
        }
    }
}

int main(int argc, char** argv){
    int dur   = argc>1 ? atoi(argv[1]) : 600;
    int nw    = argc>2 ? atoi(argv[2]) : 216;
    const char* outdir = argc>3 ? argv[3] : "/tmp/avxburn";
    char cmd[512]; snprintf(cmd,sizeof(cmd),"mkdir -p %s && rm -f %s/*.w 2>/dev/null", outdir, outdir); if(system(cmd)){}
    double t0 = now_s();
    pid_t* pids = malloc(nw*sizeof(pid_t));
    for(int j=0;j<nw;j++){
        int core = j/2, cpu = (j%2)? core+112 : core;   /* 2 per core; sibling thread = core+112 */
        pid_t p = fork();
        if(p==0){ char path[512]; snprintf(path,sizeof(path),"%s/%d.w",outdir,j); worker(cpu,path,t0); _exit(0); }
        pids[j]=p;
    }
    /* let it run for `dur`, then stop the workers */
    sleep(dur);
    for(int j=0;j<nw;j++) kill(pids[j], SIGKILL);
    for(int j=0;j<nw;j++) waitpid(pids[j], NULL, 0);
    printf("done: %d workers x %ds -> %s/*.w\n", nw, dur, outdir);
    return 0;
}
