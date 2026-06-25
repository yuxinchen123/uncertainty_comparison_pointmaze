import os, torch, numpy as np
info = {"SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
        "sched_affinity_cpus": len(os.sched_getaffinity(0)),
        "OMP_NUM_THREADS_env": os.environ.get("OMP_NUM_THREADS"),
        "torch_get_num_threads": torch.get_num_threads()}
try:
    import threadpoolctl
    info["threadpools"] = [(d.get("internal_api"), d.get("num_threads")) for d in threadpoolctl.threadpool_info()]
except Exception as e:
    info["threadpools"] = str(e)
print(info)
