"""Measure what the logging actually costs, at several cadences, with and without gradient stats.

Runs the real trainer in a fresh process per configuration, on the same node, and reports
steady-state throughput. The question is whether logging is worth worrying about at all.
"""
import json, os, shutil, statistics, subprocess, sys, tempfile, time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
TRAINER = os.path.join(REPO, "src", "ppo_rnd_envpool_shuze.py")
PYTHON = "/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python"
BATCH = 128 * 128


def run(name, extra, iterations, warmup, workdir, log_dir):
    """Run one configuration and return its steady-state throughput."""
    out = os.path.join(workdir, name); os.makedirs(out, exist_ok=True)
    cmd = [PYTHON, "-u", TRAINER, "--env_id", "MontezumaRevenge-v5",
           "--num_iterations_obs_norm_init", "2",
           "--total_timesteps", str(BATCH * (iterations + 5)),
           "--profile_iterations", str(iterations), "--output_dir", out,
           "--run_id", "0", "--run_total", "1", "--seed", "1", "--profile_tag", name,
           "--checkpoint_every_seconds", "1e9", "--no-resume",
           "--opt_env_threads", os.environ.get("SLURM_CPUS_PER_TASK", "8")] + extra
    env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1")
    t0 = time.time()
    with open(os.path.join(log_dir, f"{name}.log"), "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=5400)
    wall = time.time() - t0
    rec = os.path.join(out, "0_of_1.json")
    if p.returncode != 0 or not os.path.exists(rec):
        return {"variant": name, "error": f"rc={p.returncode}"}
    d = json.load(open(rec))
    if not d["eval_history"]:
        return {"variant": name, "error": "no logged rows"}
    last = d["eval_history"][-1]
    side = rec[:-5] + ".episodes.jsonl"
    res = {"variant": name, "steps_per_second": last["charts/steps_per_second_cumulative"],
           "wall_seconds": wall, "rows": len(d["eval_history"]),
           "record_bytes": os.path.getsize(rec),
           "sidecar_bytes": os.path.getsize(side) if os.path.exists(side) else 0,
           "episodes_kept": d.get("episodes_kept", 0)}
    shutil.rmtree(out, ignore_errors=True)
    return res


def main():
    node = os.environ.get("SLURMD_NODENAME", os.uname().nodename)
    job = os.environ.get("SLURM_JOB_ID", "local")
    out_dir = sys.argv[1]; os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, f"logs_{node}"); os.makedirs(log_dir, exist_ok=True)
    ITER, WARM = 40, 6
    # Cadence 1 logs every update, which is far denser than anything real: it is the upper bound on
    # what logging could ever cost. Then the candidate cadences, then no gradient statistics, so the
    # gradient cost is separable from the flush cost.
    configs = [
        ("every_1_update",        ["--log_every_updates", "1"]),
        ("every_25_updates",      ["--log_every_updates", "25"]),
        ("every_100_updates",     ["--log_every_updates", "100"]),
        ("every_200_updates",     ["--log_every_updates", "200"]),
        ("every_400_updates",     ["--log_every_updates", "400"]),
        ("no_logging_at_all",     ["--log_every_updates", "1000000"]),
        ("every_1_no_gradstats",  ["--log_every_updates", "1", "--no-log_gradient_statistics"]),
        ("every_200_no_gradstats",["--log_every_updates", "200", "--no-log_gradient_statistics"]),
    ]
    report = {"node": node, "slurm_job_id": job,
              "cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
              "iterations": ITER, "warmup": WARM, "variants": []}
    path = os.path.join(out_dir, f"logging_{node}_{job}.json")
    with tempfile.TemporaryDirectory(prefix="logcost_") as wd:
        for name, extra in configs:
            r = run(name, extra, ITER, WARM, wd, log_dir)
            report["variants"].append(r)
            print(f"[{node}] {name}: {r.get('steps_per_second', r.get('error'))}", flush=True)
            json.dump(report, open(path, "w"), indent=1)
    os.chmod(path, 0o660)
    print(f"[{node}] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
