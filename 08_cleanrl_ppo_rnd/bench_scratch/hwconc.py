"""Reports what envpool's default thread count resolves to inside a Slurm cgroup."""
import os, json, time, numpy as np, envpool
ncpu = len(os.sched_getaffinity(0))
import subprocess
hw = subprocess.run(["nproc","--all"], capture_output=True, text=True).stdout.strip()
print(json.dumps({"host": os.uname().nodename, "cgroup_cpus": ncpu, "nproc_all(machine)": hw}))
def sps(ne, nt):
    kw = dict(task_id="MontezumaRevenge-v5", env_type="gym", num_envs=ne, episodic_life=True,
              reward_clip=True, seed=1, repeat_action_probability=0.25)
    if nt: kw["num_threads"] = nt
    e = envpool.make(**kw); e.reset()
    a = np.random.randint(0, e.action_space.n, size=(ne,), dtype=np.int32)
    for _ in range(20): e.step(a)
    it = 40000 // ne; t0 = time.perf_counter()
    for _ in range(it): e.step(a)
    d = time.perf_counter()-t0; e.close(); return it*ne/d
for nt in [0, ncpu, 128]:
    v = sps(128, nt); print(f"  num_threads={nt if nt else 'default(0)':>12}  {v:9.1f} agent-steps/s", flush=True)
