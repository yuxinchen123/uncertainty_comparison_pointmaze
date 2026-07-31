# jaguar03: CPU is clock-throttled to ~400 MHz (diagnosis + reproducer)

**Date:** 2026-07-24
**Node:** jaguar03 (gpu partition)
**Reporter:** Shuze

*(This supersedes my earlier note that framed the problem as memory bandwidth. That was a real
symptom, but the root cause is simpler: the CPU cores are running far below their rated clock, which
slows everything else — compute, memory, and my training jobs — as a side effect.)*

## Summary

Every CPU core on jaguar03 is running at about **400 MHz** while under load, on both sockets. The
node's rated clock is 2000 MHz, and the operating system's own configured minimum is 1500 MHz — so the
cores are stuck **below even the OS minimum**. The Linux governor cannot go below its own minimum, so
this is a firmware/hardware-level throttle, not an OS setting. A reboot today (the node came up at
13:02) did not clear it.

The effect: compute runs about 5× slower, memory access about 4× slower, and my training jobs about
8–10× slower than on any other node. There are no system-error log entries, because a power/firmware
clock clamp does not raise an error — it just runs slow.

This is easy to miss, which is probably why it looked fine on your side: `/proc/cpuinfo` reports a
normal-looking **1500 MHz** (a stale value), while the *real* running frequency in
`scaling_cur_freq` is **400 MHz**. You only see it if you read `scaling_cur_freq` under load, or
measure actual compute throughput.

## Direct evidence

Measured on jaguar03 today while a compute loop was running (node otherwise idle):

| What | jaguar03 | healthy node (adriatic06) | expected |
|---|---|---|---|
| `scaling_cur_freq`, all cores (both sockets) | **~400 MHz** | ~1450 MHz | ≥ 1500 MHz (the OS minimum) |
| `scaling_min` / `scaling_max` | 1500 / 2000 MHz | — | current should be within this range |
| effective clock from a compute benchmark | **~0.49 GHz** | ~2.45 GHz | ~2 GHz |
| `/proc/cpuinfo` "cpu MHz" (unreliable here) | 1500 (stale) | — | — |

The per-core `scaling_cur_freq` read 399,647–399,985 kHz across cores 0, 28, 56, 84, 112, 140, 168,
196 — i.e. the whole node, both sockets, is clamped at ~400 MHz.

Impact of the throttle (all measured jaguar03 vs the same healthy node):

1. **Compute:** a register-only loop (no memory access) runs ~5× slower — this is the throttle itself.
2. **Memory bandwidth:** ~2.8 vs ~11 GB/s (a slow core issues memory requests slowly).
3. **Memory latency:** ~413 vs ~138 ns.
4. **My training jobs:** ~8–10× slower. The same jobs ran at normal speed on jaguar03 in early July,
   so this developed recently.

## Reproduce it (seconds)

**Fastest check — read the running frequency (no build needed).** With any load running on the node:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq   # jaguar03: ~400000 ; healthy: ~1.5-2 million
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq   # 1500000  -> current is BELOW this
```

**Confirm with actual compute throughput** (rules out any frequency-reporting quirk). Save the program
below as `clockcheck.c`, build **without** `-march=native` (so the instruction set is identical on
every node), and run it:

```bash
gcc -O3 -o clockcheck clockcheck.c
./clockcheck
```

On jaguar03 it reports about **0.5** for `~effective_GHz`; on any healthy node about **2**. Because it
does no memory access, this number tracks the core's real clock.

```c
/* clockcheck.c -- portable single-core compute-rate probe for a CPU clock throttle.
 * A tight loop of independent scalar mul+add over 8 accumulators (registers only, NO memory traffic).
 * Build WITHOUT -march=native so every node runs identical baseline instructions; then the only thing
 * that differs is how fast the core retires them = effective clock.
 * Build:  gcc -O3 -o clockcheck clockcheck.c      (no -march=native, on purpose)
 */
#include <stdio.h>
#include <time.h>
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
int main(void){
    double a[8]; for(int i=0;i<8;i++) a[i]=1.0+0.1*i;   /* 8 independent chains -> expose throughput */
    const double b=1.0000000001, c=0.5;
    long K=1000000000L;                                  /* 1e9 * 8 = 8e9 mul-add pairs */
    double t0=now_s();
    for(long i=0;i<K;i++)
        for(int j=0;j<8;j++) a[j]=a[j]*b+c;              /* register-only: no loads/stores in the hot loop */
    double dt=now_s()-t0;
    double s=0; for(int j=0;j<8;j++) s+=a[j];
    printf("%.3f s  %.2f G(mul-add)/s  ~effective_GHz=%.2f  sink=%g\n", dt, K*8/dt/1e9, K*8/dt/1e9, s);
    return 0;
}
```

## Likely cause and suggested checks

The frequency is pinned *below* the OS minimum, so the OS/governor is not doing it — something below
the OS is forcing it. The usual causes are:

1. **Power delivery** — a failing or current-limited VRM / power stage forcing the cores down.
2. **Thermal** — a stuck thermal sensor or a `PROCHOT` line held asserted, which clamps the clock even
   when the part is not actually hot.
3. **A BMC / firmware power cap** (package power limit) set or stuck low.

Because it does not log an OS error, the place to look is the **BMC / IPMI**: power and thermal
sensor readings, the power-cap / package-power-limit setting, and the hardware event log. A plain
reboot did **not** clear it (already tried today), so the next steps would be a **full power cycle**,
a **BMC reset**, and a check of the power-delivery and thermal hardware.

You are welcome to take jaguar03 out of my reservation while you look — nothing of mine is running on
it now, and I am happy to run any test you would like.
