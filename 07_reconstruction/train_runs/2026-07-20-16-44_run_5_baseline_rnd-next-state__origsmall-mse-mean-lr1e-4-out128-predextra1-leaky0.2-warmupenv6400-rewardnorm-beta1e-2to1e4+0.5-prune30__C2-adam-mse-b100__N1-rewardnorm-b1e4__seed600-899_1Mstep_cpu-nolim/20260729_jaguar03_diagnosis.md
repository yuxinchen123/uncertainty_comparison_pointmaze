# jaguar03: cores run at ~0.5 GHz, about a quarter of the rated 2.0 GHz (throughput comparison)

**Date:** 2026-07-29
**Node:** jaguar03 (gpu partition)
**Reporter:** Shuze

## Summary

Measured by **actual compute throughput** (work done ÷ time — this needs no frequency counter, so it
is not affected by any counter being misreported), jaguar03's cores run at about **0.52 GHz**, at
1, 2, and 8 cores alike. The chip is rated for 2.0 GHz, so it is running at roughly **a quarter of its
own rating**. Two healthy nodes measured the same way run at essentially their own rated clock.

I am using throughput, not `scaling_cur_freq`, on purpose: `scaling_cur_freq` is unreliable on this
cluster — on the healthy node adriatic06 it read 800 MHz while the cores were doing 2.5 GHz of work — so
I do not rely on it. Throughput is the ground truth.

## Method

- Each node was measured **exclusively** (the only job on the node).
- The load is a register-only multiply-add loop (`clockcheck.c` below) — **no memory access**, so it
  measures core execution rate only. Built **without** `-march=native`, so every node runs the
  identical instructions.
- Cores 1, 2, 8 are pinned to distinct **physical** cores (cpus 0..n-1, first thread of each core, no
  hyperthread sibling), so there is no hyperthread contention.
- `effective_GHz` reported by the program is billions of mul-add pairs per second; at ~1 pair per
  cycle it equals the core's effective clock. (Cross-check: it reads 3.6 on a 3.6 GHz Xeon and 3.16 on
  a 3.2 GHz Xeon, so it tracks true clock closely.)

## Results (per-core effective GHz = throughput)

| cores loaded | jaguar03 — EPYC 7663 (rated max 2.0 GHz) | adriatic06 — Xeon Silver 4208 (rated max 3.2 GHz) | jaguar02 — Xeon Gold 6334 (rated max 3.6 GHz) |
|---|---|---|---|
| 1 | **0.52** | 3.16 | 3.61 |
| 2 | **0.52** | 3.07 | 3.62 |
| 8 | **0.53** | 2.48 | 3.58 |

Reading of the table:

1. **jaguar03 is flat at ~0.52 GHz** whether 1, 2, or 8 cores are busy — it never rises toward its
   2.0 GHz rating. That is about **26% of its own rated clock**.
2. Both healthy nodes run at essentially their rated clock (adriatic06 boosts to 3.16 at one core and
   holds 2.48 across eight; jaguar02 holds ~3.6). This is what a healthy node looks like, and it is why
   the comparison is fair: each node is judged against its own rating, and only jaguar03 is far below.
3. The same ~4× shortfall (0.52 vs the 2.0 it should reach) is what makes my training jobs run ~8–10×
   slower on jaguar03 than elsewhere.

Note on the earlier disagreement: read as root, `cpuinfo_cur_freq` on jaguar03 shows 2.0 GHz under
load. But the throughput above (measured inside a normal Slurm job) is ~0.5 GHz. Both cannot be true of
the same running code, so either (a) jobs on jaguar03 are being given a low frequency that root's direct
test is not, or (b) the hardware is throttled and `cpuinfo_cur_freq` is also misreporting. The one test
that settles it is to run `clockcheck` (below) **as root** and read its `effective_GHz`: ~2 means (a),
~0.5 means (b).

## Scripts I ran

Two files. `clockcheck.c` is the compute kernel; `scaling_probe.sh` runs it at 1, 2, and 8 cores and
prints the table above.

`clockcheck.c`:

```c
/* clockcheck.c -- portable single-core compute-rate probe (register-only, NO memory traffic).
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

`scaling_probe.sh` (put it next to `clockcheck.c`):

```bash
#!/bin/bash
# Compute-throughput scaling test at 1, 2, 8 PHYSICAL cores (cpus 0..n-1 = first thread of each core,
# no hyperthread sibling -> no HT contention). Throughput = work/time, independent of any freq counter.
EXP="$(cd "$(dirname "$0")" && pwd)"
gcc -O3 -o "/tmp/cc_$$" "$EXP/clockcheck.c" 2>/dev/null
echo "host=$(hostname)  cpu=$(lscpu | grep 'Model name' | sed 's/.*: *//')  max_MHz=$(lscpu | awk -F: '/CPU max MHz/{gsub(/ /,"",$2);print $2}')"
for n in 1 2 8; do
  for c in $(seq 0 $((n-1))); do taskset -c "$c" "/tmp/cc_$$" >"/tmp/w_${$}_$c" 2>&1 & done
  wait
  vals=$(for c in $(seq 0 $((n-1))); do grep -oP 'effective_GHz=\K[0-9.]+' "/tmp/w_${$}_$c"; done)
  med=$(echo "$vals" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  total=$(echo "$vals" | awk '{s+=$1} END{printf "%.2f", s}')
  echo "  cores=$n  per_core_GHz=$med  aggregate_GHz=$total"
  rm -f /tmp/w_${$}_*
done
rm -f "/tmp/cc_$$"
```

## Reproduce it

On jaguar03 and on any other node, with the node otherwise idle:

```bash
gcc -O3 -o clockcheck clockcheck.c
chmod +x scaling_probe.sh
./scaling_probe.sh
```

Expected: jaguar03 prints ~0.52 at every core count; a healthy node prints its rated clock (2.5–3.6).
To settle whether it is a per-job cap or the hardware, also run `./clockcheck` **as root** on jaguar03
and read `effective_GHz` (~2 = per-job cap, hardware fine; ~0.5 = hardware/firmware throttle).

## What I think is going on and suggested checks

The cores deliver ~1/4 of their rated clock for my jobs, and this has been steady for days (a reboot
did not change it, and the node's uptime shows it has not been power-cycled since 2026-07-24). Because
`cpuinfo_cur_freq` shows 2.0 GHz to root while my Slurm jobs measure ~0.5 GHz, the most likely places to
look are:

1. Whether jobs on jaguar03 are getting a **CPU-frequency / power cap** that root's direct test does not
   — e.g. a Slurm power or cpufreq plugin, a per-partition or per-reservation `--cpu-freq` default, or a
   cgroup power limit. (Running `clockcheck` as root, above, confirms or rules this out in ~15 seconds.)
2. If root also measures ~0.5 GHz, then it is the hardware after all — a power-delivery, thermal
   (stuck PROCHOT), or BMC power-cap issue — and worth a full power cycle plus a look at the BMC/IPMI
   power and thermal readings.

Either way, I have moved my jobs off jaguar03, so there is no urgency — I just want it recorded so it can
be sorted out. Happy to run anything that helps.

## Follow-up 2026-07-31: fixed after the reboot

The node was rebooted 2026-07-31 14:38 (BootTime). I re-ran the identical probe (same `clockcheck.c`,
built without `-march=native`, loads pinned to distinct physical cores, node otherwise idle) as Slurm
job 6528713 under reservation `sl5nw_151`, plus an outbound-internet check (job 6528714). Job ids are in
`slurm/submitted_jobids_jaguar03_probe.txt`; probe files and raw output in
`slurm/clockcheck_probe_20260731/`.

| cores loaded | per-core G(mul-add)/s, 2026-07-29 (before reboot) | per-core G(mul-add)/s, 2026-07-31 (after reboot) |
|---|---|---|
| 1 | 0.52 | 4.62 |
| 2 | 0.52 | 4.67 |
| 8 | 0.53 | 4.66 |

Reading of the result:

1. **The problem is gone.** Throughput is ~9x what it was before the reboot — the same factor as the
   8–10x training slowdown that was observed.
2. On why the after-reboot number (4.66) is above the 2.0 GHz rating: this AMD core (Zen 3) can retire
   about two mul-add pairs per cycle for this kernel, while the calibration Xeons retire about one per
   cycle. So 4.66 G pairs/s corresponds to an actual clock of roughly 2.3 GHz — at or slightly above the
   rated 2.0 GHz, i.e. healthy. The decisive comparison is same node, identical binary, before vs
   after: 0.52 → 4.66.
3. Outbound internet from the node works again (HTTP 200 from huggingface.co).
4. **The actual training workload is also back to full speed** (checked the same evening, since the
   compute probe above does not touch memory and the true root cause was memory latency — see the
   2026-07-23 correction in `jaguar03_threading_experiment/report.md`): 8 copies of the real run-5
   training, 1 thread per physical core, ran at a median 1,731 steps/min per run on jaguar03 vs
   1,382 on the same-day healthy control adriatic06 (sick value was 168), and memory latency
   measured 107 ns per random access (sick: 413–425 ns; healthy range: 99–138 ns). Full method and
   table: `jaguar03_threading_experiment/recheck_2026-07-31_train_throughput/results.md`.
