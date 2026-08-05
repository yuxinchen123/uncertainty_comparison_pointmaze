"""Measures envpool Atari throughput for the ppo_rnd_envpool.py settings across
num_envs / batch_size / num_threads, in synchronous and asynchronous mode."""
import json, os, sys, time
import numpy as np
import envpool

TARGET_STEPS = 40000   # agent steps per measurement


def make(num_envs, batch_size, num_threads):
    """Builds a MontezumaRevenge pool with the exact CleanRL wrapper settings."""
    kw = dict(task_id="MontezumaRevenge-v5", env_type="gym", num_envs=num_envs,
              episodic_life=True, reward_clip=True, seed=1, repeat_action_probability=0.25)
    if batch_size: kw["batch_size"] = batch_size
    if num_threads: kw["num_threads"] = num_threads
    return envpool.make(**kw)


def bench_sync(num_envs, num_threads):
    """Times synchronous stepping (batch_size == num_envs), returns agent steps per second."""
    e = make(num_envs, 0, num_threads)
    e.reset()
    n_act = e.action_space.n
    acts = np.random.randint(0, n_act, size=(num_envs,), dtype=np.int32)
    for _ in range(20): e.step(acts)          # warmup
    iters = max(20, TARGET_STEPS // num_envs)
    t0 = time.perf_counter()
    for _ in range(iters): e.step(acts)
    dt = time.perf_counter() - t0
    e.close()
    return iters * num_envs / dt


def bench_async(num_envs, batch_size, num_threads):
    """Times asynchronous send/recv stepping, returns agent steps per second."""
    e = make(num_envs, batch_size, num_threads)
    e.async_reset()
    n_act = e.action_space.n
    got = 0
    for _ in range(20):
        out = e.recv(); env_id = out[-1]["env_id"]
        e.send(np.random.randint(0, n_act, size=(len(env_id),), dtype=np.int32), env_id)
    iters = max(20, TARGET_STEPS // batch_size)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = e.recv(); env_id = out[-1]["env_id"]; got += len(env_id)
        e.send(np.random.randint(0, n_act, size=(len(env_id),), dtype=np.int32), env_id)
    dt = time.perf_counter() - t0
    e.close()
    return got / dt


if __name__ == "__main__":
    ncpu = len(os.sched_getaffinity(0))
    res = {"host": os.uname().nodename, "cpus_in_cgroup": ncpu,
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "sync": {}, "async": {}}
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}), flush=True)
    for ne in [128, 256]:
        for nt in [0, ncpu, 2 * ncpu, 128]:
            key = f"num_envs={ne},num_threads={nt if nt else 'default(0)'}"
            try:
                sps = bench_sync(ne, nt)
                res["sync"][key] = round(sps, 1)
                print(f"  SYNC  {key:44s} {sps:10.1f} agent-steps/s  ({sps*4:.0f} frames/s)", flush=True)
            except Exception as ex:
                res["sync"][key] = f"FAILED {type(ex).__name__}: {str(ex)[:120]}"
                print(f"  SYNC  {key:44s} FAILED {ex}", flush=True)
    for ne, bs in [(256, 128), (192, 128), (160, 128), (256, 64)]:
        for nt in [0, ncpu]:
            key = f"num_envs={ne},batch_size={bs},num_threads={nt if nt else 'default(0)'}"
            try:
                sps = bench_async(ne, bs, nt)
                res["async"][key] = round(sps, 1)
                print(f"  ASYNC {key:44s} {sps:10.1f} agent-steps/s  ({sps*4:.0f} frames/s)", flush=True)
            except Exception as ex:
                res["async"][key] = f"FAILED {type(ex).__name__}: {str(ex)[:120]}"
                print(f"  ASYNC {key:44s} FAILED {ex}", flush=True)
    print("=== RESULT JSON ===", flush=True)
    print(json.dumps(res, indent=2), flush=True)
    with open(sys.argv[1], "w") as f: json.dump(res, f, indent=2)
