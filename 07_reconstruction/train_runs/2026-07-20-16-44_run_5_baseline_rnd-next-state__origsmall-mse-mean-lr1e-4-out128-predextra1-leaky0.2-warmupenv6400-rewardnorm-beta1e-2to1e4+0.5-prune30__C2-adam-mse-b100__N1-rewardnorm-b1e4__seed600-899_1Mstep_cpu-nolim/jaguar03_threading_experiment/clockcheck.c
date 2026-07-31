/* clockcheck.c -- portable single-core compute-rate probe to detect a CPU CLOCK throttle.
 * A tight loop of independent scalar mul+add over 8 accumulators (registers only -- NO memory traffic).
 * Compiled WITHOUT -march=native so both nodes run the identical baseline (SSE2 scalar) instructions,
 * so the only thing that differs is how fast the core retires them = effective clock. Prints the
 * achieved billions-of-mul-add-pairs/sec; with ~1 mul+1 add retired per cycle at full ILP this number
 * is close to the core's effective GHz. A throttled core prints a much lower number.
 * Build:  gcc -O3 -o clockcheck clockcheck.c      (NO -march=native, on purpose)
 */
#include <stdio.h>
#include <time.h>
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
int main(void){
    double a[8]; for(int i=0;i<8;i++) a[i]=1.0+0.1*i;   /* 8 independent chains -> hide latency, expose throughput */
    const double b=1.0000000001, c=0.5;
    long K=1000000000L;                                  /* 1e9 outer iterations * 8 = 8e9 mul-add pairs */
    double t0=now_s();
    for(long i=0;i<K;i++)
        for(int j=0;j<8;j++) a[j]=a[j]*b+c;              /* register-only: no loads/stores in the hot loop */
    double dt=now_s()-t0;
    double s=0; for(int j=0;j<8;j++) s+=a[j];
    printf("%.3f s  %.2f G(mul-add)/s  ~effective_GHz=%.2f  sink=%g\n", dt, K*8/dt/1e9, K*8/dt/1e9, s);
    return 0;
}
