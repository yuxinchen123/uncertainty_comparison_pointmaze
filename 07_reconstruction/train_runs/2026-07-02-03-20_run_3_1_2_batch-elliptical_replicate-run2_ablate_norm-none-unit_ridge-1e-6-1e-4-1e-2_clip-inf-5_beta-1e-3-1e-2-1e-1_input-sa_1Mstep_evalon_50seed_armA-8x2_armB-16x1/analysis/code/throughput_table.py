#!/usr/bin/env python
"""Two-arm throughput table (MEAN steps per CPU-hour), fixed-shape arm-B era only.

Broken-binding arm-B attempts are excluded by construction: their partial JSONs were archived at the
2026-07-02 reshape, and run->bucket attribution uses only logs of jobs in the two id files whose logs
carry the fat-era ids never re-claimed... more simply: a run_id claimed twice is attributed to its LAST
claim (the requeued attempt), and all post-reshape arm-B jobs run the corrected shape. Finished runs use
1M / runtime; in-flight runs use their last checkpoint (>= 1 h runtime). Per-CPU = pace / billed CPUs per
worker (arm A worker = 2 Slurm CPUs, arm B = 1)."""
import glob
import json
import os
import re
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(os.path.dirname(HERE))
SWEEPS = {"A": "2026-07-02-03-52_armA-8tasks-2cpu_replicate-ablate",
          "B": "2026-07-02-03-52_armB-16tasks-1cpu_replicate-ablate"}
CPUS_PER_WORKER = {"A": 2, "B": 1}
BUCKETS = {"graph-prune": ("A", "jaguar03"), "spec-decode": ("B", "jaguar03"),
           "kv-cache": ("A", "puma01"), "lr-warmup": ("B", "puma01"),
           "tok-merge": ("A", "cpu-partition"), "ffn-gate": ("B", "cpu-partition"),
           "attn-sink": ("A", "gpu-allowlist"), "rope-scale": ("B", "gpu-allowlist")}
# fat-era arm-B job ids (killed at the reshape): claims in their logs are superseded by any later claim
FAT_ERA_MAX_JOBID = 6270522

# run -> bucket from worker logs; later-jobid claims win (the requeued attempt supersedes the killed one)
claims = {}
for lf in glob.glob(os.path.join(RUN, "logs", "*_*.log")):
    base = os.path.basename(lf)
    name, jid = base.rsplit("_", 1)
    # only worker-job logs are <bucket>_<numeric jobid>.log; skip other logs (e.g. the monitor's)
    if name not in BUCKETS or not jid.replace(".log", "").isdigit():
        continue
    jid = int(jid.replace(".log", ""))
    arm = BUCKETS[name][0]
    for m in re.finditer(r"claimed (\d+)_of_900", open(lf, errors="ignore").read()):
        key = (arm, int(m.group(1)))
        if key not in claims or jid > claims[key][0]:
            claims[key] = (jid, name)

paces = defaultdict(list)
armpaces = defaultdict(list)
for arm, sw in SWEEPS.items():
    for f in glob.glob(os.path.join(RUN, "data", sw, "local", "*.json")):
        try:
            r = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        hours = (r.get("runtime_seconds") or 0) / 3600.0
        if r.get("completed", True):
            steps = r.get("total_timesteps", 1000000)
        else:
            steps = max((e.get("step", 0) for e in r.get("train_history", [])), default=0)
            if hours < 1.0 or steps == 0:
                continue
        got = claims.get((arm, int(r["run_id"])))
        if got:
            paces[got[1]].append(steps / hours)
            armpaces[arm].append(steps / hours)

print("=" * 88)
print(f"{'bucket (node class)':<26s} {'arm':<4s} {'n runs':>6s} {'MEAN steps/h':>13s} "
      f"{'cpus/worker':>11s} {'MEAN steps per CPU-hour':>23s}")
print("-" * 88)
for name, (arm, nodeclass) in BUCKETS.items():
    p = paces.get(name, [])
    if not p:
        continue
    mean = statistics.mean(p)
    print(f"{nodeclass+' ('+name+')':<26s} {arm:<4s} {len(p):>6d} {mean:>13,.0f} "
          f"{CPUS_PER_WORKER[arm]:>11d} {mean/CPUS_PER_WORKER[arm]:>23,.0f}")
print("-" * 88)
for arm in ("A", "B"):
    p = armpaces[arm]
    if p:
        mean = statistics.mean(p)
        print(f"{'ALL BUCKETS':<26s} {arm:<4s} {len(p):>6d} {mean:>13,.0f} "
              f"{CPUS_PER_WORKER[arm]:>11d} {mean/CPUS_PER_WORKER[arm]:>23,.0f}")
print("-" * 88)
for nodeclass, ab, bb in (("jaguar03", "graph-prune", "spec-decode"), ("puma01", "kv-cache", "lr-warmup"),
                          ("cpu-partition", "tok-merge", "ffn-gate"), ("gpu-allowlist", "attn-sink", "rope-scale")):
    pa, pb = paces.get(ab, []), paces.get(bb, [])
    if pa and pb:
        ma, mb = statistics.mean(pa), statistics.mean(pb)
        print(f"{nodeclass}: per-CPU-hour mean B/A = {mb/(ma/2):.2f}")
