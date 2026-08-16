/* Read from a processor the figures the training comparison needs but lscpu does not give.
 *
 * Written in C because all three quantities are destroyed by interpreter overhead: one memory
 * access takes about 100 nanoseconds, while one python loop iteration takes about 30, so a
 * python version of this would mostly measure python.
 *
 *   clock       timed as a chain of additions in which each addition needs the previous one's
 *               result. Such a chain advances one addition per clock cycle on both processor
 *               designs compared here, so additions divided by seconds is the clock. This is
 *               the only way to read the clock on jaguar02, which exposes no cpufreq interface
 *               and reports a fixed nominal 3600 in /proc/cpuinfo.
 *   latency     time for one dependent memory access, walking a random cycle through a buffer.
 *               Each address comes from the previous read, so the processor cannot fetch ahead
 *               and the time is one full access. Run at sizes that land in each cache level.
 *   bandwidth   bytes per second read by one core streaming through a buffer, at the same
 *               sizes, which is the throughput number rather than the latency number.
 *
 * Output is one JSON object on standard output.
 *
 * Build: gcc -O2 -o cpu_probe cpu_probe.c
 * Run:   ./cpu_probe
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static double now_seconds(void)
{
    /* goal: a monotonic clock that a suspended machine cannot move backwards */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double dependent_add_ghz(uint64_t iterations)
{
    /* goal: the clock, from a chain of additions that cannot overlap with each other */
    uint64_t x = 1;
    /* eight additions per loop iteration, each needing the previous result, so the loop's own
     * counter and branch run alongside the chain instead of being counted by it */
    double t0 = now_seconds();
    for (uint64_t i = 0; i < iterations; i++) {
        __asm__ volatile("addq $1, %0" : "+r"(x));
        __asm__ volatile("addq $1, %0" : "+r"(x));
        __asm__ volatile("addq $1, %0" : "+r"(x));
        __asm__ volatile("addq $1, %0" : "+r"(x));
        __asm__ volatile("addq $1, %0" : "+r"(x));
        __asm__ volatile("addq $1, %0" : "+r"(x));
        __asm__ volatile("addq $1, %0" : "+r"(x));
        __asm__ volatile("addq $1, %0" : "+r"(x));
    }
    double seconds = now_seconds() - t0;
    /* the chain must have advanced exactly eight times per iteration; if it did not, the
     * compiler removed part of it and every number below would be wrong */
    if (x != 1 + 8 * iterations) {
        fprintf(stderr, "addition chain was optimised away (x=%lu)\n", (unsigned long)x);
        exit(2);
    }
    /* example: 8 x 100,000,000 additions in 0.25 s -> 800e6 / 0.25 / 1e9 = 3.2 GHz */
    return (double)(8 * iterations) / seconds / 1e9;
}

static void build_cycle(uint64_t *next, uint64_t n_slots, unsigned seed)
{
    /* goal: link every slot into one single cycle, so a walk visits all of them in random order */
    uint64_t *order = malloc(n_slots * sizeof(uint64_t));
    for (uint64_t i = 0; i < n_slots; i++) order[i] = i;
    /* Fisher-Yates shuffle: example with 4 slots, order [0,1,2,3] may become [2,0,3,1] */
    srand(seed);
    for (uint64_t i = n_slots - 1; i > 0; i--) {
        uint64_t j = ((uint64_t)rand() * 2654435761u + rand()) % (i + 1);
        uint64_t t = order[i]; order[i] = order[j]; order[j] = t;
    }
    /* then each slot points at the next one in the shuffled order, and the last wraps round:
     * order [2,0,3,1] gives next[2]=0, next[0]=3, next[3]=1, next[1]=2 */
    for (uint64_t i = 0; i < n_slots; i++) next[order[i]] = order[(i + 1) % n_slots];
    free(order);
}

static double latency_ns(uint64_t size_bytes, uint64_t n_accesses)
{
    /* goal: nanoseconds for one dependent memory access in a buffer of the given size */
    uint64_t n_slots = size_bytes / sizeof(uint64_t);
    uint64_t *next = malloc(n_slots * sizeof(uint64_t));
    build_cycle(next, n_slots, 12345);
    /* warm the buffer so the first pass does not pay for page faults */
    uint64_t idx = 0;
    for (uint64_t i = 0; i < n_slots; i++) idx = next[idx];
    double t0 = now_seconds();
    for (uint64_t i = 0; i < n_accesses; i++) idx = next[idx];
    double seconds = now_seconds() - t0;
    /* consume idx so the walk cannot be removed */
    if (idx == 0xFFFFFFFFFFFFFFFFULL) printf(" ");
    free(next);
    /* example: 20,000,000 accesses in 2.0 s -> 2.0 / 20e6 * 1e9 = 100 ns per access */
    return seconds / (double)n_accesses * 1e9;
}

static double read_bandwidth_gb_s(uint64_t size_bytes, uint64_t n_passes)
{
    /* goal: bytes per second one core reads streaming through a buffer of the given size */
    uint64_t n = size_bytes / sizeof(double);
    double *a = malloc(n * sizeof(double));
    for (uint64_t i = 0; i < n; i++) a[i] = 1.0;
    /* eight running totals rather than one: a single total would make each addition wait for
     * the previous one, which measures addition latency instead of how fast memory arrives */
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    double t0 = now_seconds();
    for (uint64_t p = 0; p < n_passes; p++)
        for (uint64_t i = 0; i + 7 < n; i += 8) {
            s0 += a[i]; s1 += a[i+1]; s2 += a[i+2]; s3 += a[i+3];
            s4 += a[i+4]; s5 += a[i+5]; s6 += a[i+6]; s7 += a[i+7];
        }
    double seconds = now_seconds() - t0;
    double sum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    if (sum < 0.0) printf(" ");            /* consume sum so the loop cannot be removed */
    free(a);
    /* example: 64 MiB read 16 times in 0.5 s -> 1 GiB / 0.5 s = about 2.1 GB/s */
    return (double)size_bytes * (double)n_passes / seconds / 1e9;
}

int main(int argc, char **argv)
{
    /* goal: a short mode that reads only the clock, so it can be run alongside the real
     * training workload without taking a core away from it for long */
    int clock_only = (argc > 1 && strcmp(argv[1], "--clock-only") == 0);
    if (clock_only) {
        double best = 0.0;
        for (int r = 0; r < 5; r++) {
            double g = dependent_add_ghz(30000000);
            if (g > best) best = g;
        }
        printf("{\"clock_ghz_dependent_add_chain\": %.3f}\n", best);
        return 0;
    }

    /* the four working-set sizes, chosen to land in level 1, level 2, level 3 and main memory
     * on both processors compared here */
    const uint64_t sizes[4] = {24 * 1024, 384 * 1024, 8ULL * 1024 * 1024, 256ULL * 1024 * 1024};
    const char *names[4] = {"24KiB_level1", "384KiB_level2", "8MiB_level3", "256MiB_main_memory"};

    /* the clock: take the fastest of five, since only interference can make it slower */
    double best_ghz = 0.0;
    for (int r = 0; r < 5; r++) {
        double g = dependent_add_ghz(30000000);
        if (g > best_ghz) best_ghz = g;
    }

    printf("{\n \"clock_ghz_dependent_add_chain\": %.3f,\n", best_ghz);

    printf(" \"latency_ns_per_dependent_access\": {\n");
    for (int i = 0; i < 4; i++) {
        uint64_t accesses = (i == 3) ? 4000000 : 20000000;
        printf("  \"%s\": %.2f%s\n", names[i], latency_ns(sizes[i], accesses),
               i < 3 ? "," : "");
    }
    printf(" },\n");

    printf(" \"read_bandwidth_gb_per_s_one_core\": {\n");
    for (int i = 0; i < 4; i++) {
        uint64_t passes = (i == 3) ? 4 : (i == 2 ? 64 : 4096);
        printf("  \"%s\": %.2f%s\n", names[i], read_bandwidth_gb_s(sizes[i], passes),
               i < 3 ? "," : "");
    }
    printf(" }\n}\n");
    return 0;
}
