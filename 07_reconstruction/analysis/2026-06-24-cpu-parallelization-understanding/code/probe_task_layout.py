"""
Per-task layout probe for an srun step. Run as `srun python probe_task_layout.py`
inside an allocation; each task prints one JSON line describing the CPU set and
default thread count IT actually sees. Reveals:
  - how many cpus each task is bound to (its sched_getaffinity), and whether the
    tasks' cpu sets are DISJOINT (each task its own cores) or SHARED (all see all),
  - what torch.get_num_threads() defaults to per task with NO OMP_NUM_THREADS set
    (i.e. does a 4-cpu-per-task layout give 4 threads/task, or does each task grab
    the whole node?),
  - the memory limit the task sees.
No thread env vars are set here, so torch's number is the genuine DEFAULT.
"""
import json
import os
import socket

aff = sorted(os.sched_getaffinity(0))
out = {
    "host": socket.gethostname(),
    "SLURM_PROCID": os.environ.get("SLURM_PROCID"),
    "SLURM_LOCALID": os.environ.get("SLURM_LOCALID"),
    "SLURM_NTASKS": os.environ.get("SLURM_NTASKS"),
    "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
    "SLURM_MEM_PER_NODE_MB": os.environ.get("SLURM_MEM_PER_NODE"),
    "SLURM_MEM_PER_CPU_MB": os.environ.get("SLURM_MEM_PER_CPU"),
    "OMP_NUM_THREADS_env": os.environ.get("OMP_NUM_THREADS"),
    "affinity_ncpus": len(aff),
    "affinity_cpus": aff,
}
try:
    import torch
    out["torch_default_num_threads"] = torch.get_num_threads()
except Exception as e:
    out["torch_default_num_threads"] = f"err: {e}"
try:
    import psutil
    vm = psutil.virtual_memory()
    out["node_total_ram_gb"] = round(vm.total / 1e9, 1)
except Exception:
    pass
print(json.dumps(out), flush=True)
