"""Measure how many training runs one GPU should carry at once.

A single run leaves the GPU idle for more than half of every iteration: the rollout is a serial
alternation of a small GPU forward pass and a CPU-side environment step, so while envpool steps, the
GPU has nothing to do. Packing several runs onto one GPU should fill those gaps. This script tests
that directly rather than reasoning about it: it launches N copies of the real trainer on the same
GPU, all at once, and reports the AGGREGATE throughput.

What matters is aggregate steps per second per GPU, and steps per second per cpu — the cpu is the
capped resource on this cluster, so a packing factor that raises the GPU's aggregate rate but lowers
the per-cpu rate is not a win.

Run inside a Slurm job holding one GPU and N * cpus_per_run cpus:
    PYTHONNOUSERSITE=1 <env>/bin/python pack_test.py --n 4 --cpus_per_run 8 --out <dir>
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
TRAINER = os.path.join(REPO, "src", "ppo_rnd_envpool_shuze.py")
PYTHON = "/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python"


def launch(idx, cpus_per_run, iterations, workdir, log_dir, env_id):
    """Start one trainer process in the run configuration and return the handle plus its paths."""
    out = os.path.join(workdir, f"run{idx}")
    os.makedirs(out, exist_ok=True)
    cmd = [
        PYTHON, TRAINER, "--env_id", env_id,
        "--num_iterations_obs_norm_init", "2",
        "--total_timesteps", str(128 * 128 * (iterations + 5)),
        "--log_every_updates", "1",
        "--profile_iterations", str(iterations),
        "--output_dir", out, "--run_id", str(idx), "--run_total", "1",
        # A different seed per copy, so the copies are not accidentally doing identical work and
        # sharing cache lines in a way a real sweep would not.
        "--seed", str(idx + 1),
        "--profile_tag", f"pack{idx}",
        "--checkpoint_every_seconds", "1e9", "--no-resume",
        "--opt_env_threads", str(cpus_per_run),
    ]
    logf = open(os.path.join(log_dir, f"pack_run{idx}.log"), "w")
    env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    return {"idx": idx, "proc": proc, "logf": logf, "out": out}


def read_throughput(out, idx, warmup):
    """Read one finished copy's steady-state throughput from its JSON record."""
    path = os.path.join(out, f"{idx}_of_1.json")
    if not os.path.exists(path):
        return None
    rec = json.load(open(path))
    rows = rec["eval_history"][warmup:]
    if not rows:
        return None
    its = [r["charts/iteration_seconds"] for r in rows]
    return {
        "steps_per_second": 128 * 128 / statistics.fmean(its),
        "iteration_seconds_mean": statistics.fmean(its),
        "gpu_memory_peak_mb": max(r["charts/gpu_memory_peak_mb"] for r in rows),
        "gpu_memory_reserved_mb": max(r["charts/gpu_memory_reserved_mb"] for r in rows),
    }


def gpu_memory_used_mb():
    """Ask the driver how much memory is actually in use on the GPU right now."""
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    used, total = r.stdout.strip().splitlines()[0].split(",")
    return float(used), float(total)


def main():
    """Launch N concurrent trainers on one GPU and report the aggregate throughput."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="concurrent training runs on this one GPU")
    ap.add_argument("--cpus_per_run", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--env_id", default="MontezumaRevenge-v5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    node = os.environ.get("SLURMD_NODENAME", os.uname().nodename)
    job = os.environ.get("SLURM_JOB_ID", "local")
    os.makedirs(args.out, exist_ok=True)
    log_dir = os.path.join(args.out, f"logs_{node}_n{args.n}")
    os.makedirs(log_dir, exist_ok=True)

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    report = {
        "node": node, "slurm_job_id": job, "n_concurrent": args.n,
        "cpus_per_run": args.cpus_per_run, "cpus_total": len(os.sched_getaffinity(0)),
        "gpu_smi": gpu, "env_id": args.env_id, "iterations": args.iterations,
    }

    with tempfile.TemporaryDirectory(prefix="pack_") as workdir:
        started = time.time()
        # Start every copy before waiting on any of them, so they genuinely overlap.
        handles = [launch(i, args.cpus_per_run, args.iterations, workdir, log_dir, args.env_id)
                   for i in range(args.n)]
        # Sample the driver's memory figure once the copies are past start-up, which is the number
        # that decides whether the packing factor actually fits on a smaller card.
        time.sleep(90)
        try:
            used, total = gpu_memory_used_mb()
            report["gpu_memory_used_mb_while_running"] = used
            report["gpu_memory_total_mb"] = total
        except Exception as exc:
            report["gpu_memory_used_mb_while_running"] = f"unreadable: {exc}"
        for h in handles:
            h["proc"].wait(timeout=7200)
            h["logf"].close()
        report["wall_seconds"] = time.time() - started

        per_run = []
        for h in handles:
            t = read_throughput(h["out"], h["idx"], args.warmup)
            per_run.append({"idx": h["idx"], "returncode": h["proc"].returncode, **(t or {"error": "no record"})})
        report["per_run"] = per_run
        shutil.rmtree(workdir, ignore_errors=True)

    ok = [r for r in per_run if "steps_per_second" in r]
    report["runs_succeeded"] = len(ok)
    if ok:
        agg = sum(r["steps_per_second"] for r in ok)
        report["aggregate_steps_per_second"] = agg
        report["mean_steps_per_second_per_run"] = agg / len(ok)
        report["steps_per_second_per_cpu"] = agg / (len(ok) * args.cpus_per_run)
    out_path = os.path.join(args.out, f"pack_{node}_n{args.n}_{job}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    os.chmod(out_path, 0o660)
    print(json.dumps({k: v for k, v in report.items() if k != "per_run"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
