"""Run the throughput-option ladder on whichever GPU node this job landed on.

Each variant is a set of flags for `src/ppo_rnd_envpool_shuze.py`, run as a FRESH PROCESS — cuDNN's
autotune cache and the CUDA allocator carry state across configurations inside one process and would
manufacture speedups that are not there.

The ladder is cumulative: variant N is variant N-1 plus one option. That is what makes a "before and
after" table honest — every row's gain is attributable to the one option it added. Two isolated
measurements bracket it: the published code path at the top, and the configuration the real 30-seed
run will use at the bottom.

Steady-state throughput is `batch_size / iteration_seconds`, averaged over the updates after a
warm-up, NOT the script's own cumulative `steps_per_second` (which is dragged down by start-up).

Run:
    PYTHONNOUSERSITE=1 <env>/bin/python profile_variants.py --out <dir> --iterations 12
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
TRAINER = os.path.join(REPO, "src", "ppo_rnd_envpool_shuze.py")
PYTHON = "/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python"

# Every throughput option, in the order the ladder switches them on. The names match the flags.
SAME_NUMERICS = [
    "opt_env_threads_on",   # handled specially: it is an int, not a bool
    "opt_gpu_obs_rms",
    "opt_fused_policy_pass",
    "opt_rnd_no_grad",
    "opt_uint8_obs",
    "opt_gpu_norm_stats",
    "opt_no_sync_update",
    "opt_cudnn_benchmark",
]
DIFFERENT_NUMERICS = ["opt_amp_fp16", "opt_channels_last", "opt_matmul_tf32", "opt_torch_compile"]
ALL_BOOL_OPTS = [o for o in SAME_NUMERICS if o != "opt_env_threads_on"] + DIFFERENT_NUMERICS


def build_ladder(cpus):
    """Build the list of (name, description, flag-dict) variants, cumulative then bracketed."""
    # Start from the published code path: every option off, the bug fix off, and envpool left to
    # its own thread default (which reads the machine's core count, not the job's allocation).
    off = {o: False for o in ALL_BOOL_OPTS}
    off["opt_fast_obs_norm_init"] = False
    off["fix_envpool_autoreset"] = False
    off["opt_env_threads"] = 0

    ladder = [("00_upstream", "the published CleanRL code path, envpool thread default", dict(off))]

    # Cumulative: add one same-numerics option at a time.
    cur = dict(off)
    cur["opt_env_threads"] = cpus
    ladder.append(("01_env_threads", f"envpool num_threads set to the job's {cpus} cpus", dict(cur)))
    for i, opt in enumerate([o for o in SAME_NUMERICS if o != "opt_env_threads_on"], start=2):
        cur = dict(cur)
        cur[opt] = True
        ladder.append((f"{i:02d}_{opt}", f"plus {opt}", dict(cur)))

    # The configuration the real run uses: every same-numerics option, the fast start-up, and the
    # bug fix.
    real = dict(cur)
    real["opt_fast_obs_norm_init"] = True
    real["fix_envpool_autoreset"] = True
    ladder.append(("10_tier_a_plus_bugfix", "every same-numerics option, plus the auto-reset fix "
                                            "(this is the 30-seed run's configuration)", dict(real)))

    # Options that change the numbers, each measured on top of the real configuration.
    for name, extra in [
        ("11_amp_fp16_channels_last", {"opt_amp_fp16": True, "opt_channels_last": True}),
        ("12_matmul_tf32", {"opt_matmul_tf32": True}),
        ("13_torch_compile", {"opt_torch_compile": True}),
    ]:
        v = dict(real)
        v.update(extra)
        ladder.append((name, f"the run configuration plus {', '.join(extra)} (CHANGES THE NUMBERS)", v))
    return ladder


def flags_to_argv(flags):
    """Turn a flag dict into tyro command-line arguments."""
    argv = []
    for k, v in flags.items():
        if isinstance(v, bool):
            argv.append(f"--{k}" if v else f"--no-{k}")
        else:
            argv += [f"--{k}", str(v)]
    return argv


def run_variant(name, flags, iterations, warmup, cpus, workdir, env_id, log_dir):
    """Run one variant in a fresh process and return its steady-state throughput measurements."""
    out = os.path.join(workdir, name)
    os.makedirs(out, exist_ok=True)
    cmd = [
        PYTHON, TRAINER,
        "--env_id", env_id,
        # Two normalisation-init iterations instead of fifty: the init does not affect steady-state
        # throughput, and at fifty it would dominate the profiling budget. Its real cost is measured
        # separately by measure_startup().
        "--num_iterations_obs_norm_init", "2",
        "--total_timesteps", str(128 * 128 * (iterations + 5)),
        "--log_every_updates", "1",
        "--profile_iterations", str(iterations),
        "--output_dir", out,
        "--run_id", "0", "--run_total", "1", "--seed", "1",
        "--profile_tag", name,
        "--checkpoint_every_seconds", "1e9",
        "--no-resume",
    ] + flags_to_argv(flags)

    env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1")
    started = time.time()
    with open(os.path.join(log_dir, f"{name}.log"), "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, timeout=5400)
    wall = time.time() - started

    result = {"variant": name, "returncode": proc.returncode, "wall_seconds": wall}
    record_path = os.path.join(out, "0_of_1.json")
    if proc.returncode != 0 or not os.path.exists(record_path):
        result["error"] = f"returncode {proc.returncode}; see {name}.log"
        return result

    rec = json.load(open(record_path))
    rows = rec["eval_history"]
    # Drop the warm-up updates: the first few carry cuDNN autotune, allocator growth and, when
    # torch.compile is on, the whole compilation.
    steady = rows[warmup:]
    if not steady:
        result["error"] = f"only {len(rows)} updates logged, need more than {warmup}"
        return result
    its = [r["charts/iteration_seconds"] for r in steady]
    batch = 128 * 128
    result.update({
        "iterations_measured": len(steady),
        "iteration_seconds_mean": statistics.fmean(its),
        "iteration_seconds_min": min(its),
        "steps_per_second": batch / statistics.fmean(its),
        "steps_per_second_best": batch / min(its),
        "rollout_seconds_mean": statistics.fmean(r["charts/rollout_seconds"] for r in steady),
        "update_seconds_mean": statistics.fmean(r["charts/update_seconds"] for r in steady),
        "gpu_memory_peak_mb": max(r["charts/gpu_memory_peak_mb"] for r in steady),
        "gpu_memory_reserved_mb": max(r["charts/gpu_memory_reserved_mb"] for r in steady),
        "burned_rows_dropped": statistics.fmean(r["charts/burned_rows_dropped"] for r in steady),
        "gpu_name": rec["gpu_name"],
        "env_threads": rec["env_threads"],
    })
    # The per-variant record is large and regenerable; keep the numbers, drop the file.
    shutil.rmtree(out, ignore_errors=True)
    return result


def measure_startup(cpus, workdir, env_id, log_dir):
    """Measure the observation-normalisation start-up loop, the slow way and the fast way."""
    # This is the one option whose cost is entirely in start-up, so it needs the real setting of
    # fifty init iterations and only one training update.
    out = {}
    for name, fast in [("startup_original", False), ("startup_fast", True)]:
        d = os.path.join(workdir, name)
        os.makedirs(d, exist_ok=True)
        cmd = [
            PYTHON, TRAINER, "--env_id", env_id,
            "--num_iterations_obs_norm_init", "50",
            "--total_timesteps", str(128 * 128 * 6),
            "--log_every_updates", "1", "--profile_iterations", "1",
            "--output_dir", d, "--run_id", "0", "--run_total", "1", "--seed", "1",
            "--profile_tag", name, "--checkpoint_every_seconds", "1e9", "--no-resume",
            "--opt_env_threads", str(cpus),
            "--opt_fast_obs_norm_init" if fast else "--no-opt_fast_obs_norm_init",
        ]
        env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1")
        with open(os.path.join(log_dir, f"{name}.log"), "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, timeout=5400)
        rec_path = os.path.join(d, "0_of_1.json")
        if proc.returncode == 0 and os.path.exists(rec_path):
            rec = json.load(open(rec_path))
            out[name] = rec["eval_history"][0]["charts/obs_norm_init_seconds"]
        else:
            out[name] = None
        shutil.rmtree(d, ignore_errors=True)
    return out


def main():
    """Run the ladder plus the start-up measurement and write one JSON file for this node."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory the result JSON is written to")
    ap.add_argument("--iterations", type=int, default=12, help="updates to run per variant")
    ap.add_argument("--warmup", type=int, default=4, help="updates discarded before averaging")
    ap.add_argument("--env_id", default="MontezumaRevenge-v5")
    ap.add_argument("--skip_startup", action="store_true", help="skip the start-up measurement")
    ap.add_argument("--only_variant", default="", help="run just this one variant by name prefix")
    args = ap.parse_args()

    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    node = os.environ.get("SLURMD_NODENAME", os.uname().nodename)
    job = os.environ.get("SLURM_JOB_ID", "local")
    os.makedirs(args.out, exist_ok=True)
    log_dir = os.path.join(args.out, f"logs_{node}")
    os.makedirs(log_dir, exist_ok=True)

    # Read the GPU's identity once, up front, so a variant that fails still lands in a labelled file.
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()

    # Slurm counts hardware threads, so a 16-cpu allocation is 8 physical cores with both threads.
    # Record what the process can actually see, so the cores-versus-throughput curve is honest.
    cgroup_cpus = len(os.sched_getaffinity(0))
    report = {
        "node": node, "slurm_job_id": job, "cpus_per_task": cpus, "gpu_smi": gpu,
        "cgroup_visible_cpus": cgroup_cpus, "machine_cpus": os.cpu_count(),
        "env_id": args.env_id, "iterations": args.iterations, "warmup": args.warmup,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "variants": [],
    }
    out_path = os.path.join(args.out, f"profile_{node}_{job}.json")

    ladder = build_ladder(cpus)
    if args.only_variant:
        ladder = [v for v in ladder if v[0].startswith(args.only_variant)]
        if not ladder:
            raise ValueError(f"no variant matches prefix {args.only_variant!r}")
        # A cores-versus-throughput sweep runs one variant per cpu count, so name the file by the
        # cpu count too or the sweep's jobs overwrite each other on the same node.
        out_path = os.path.join(args.out, f"profile_{node}_{cpus}cpu_{job}.json")
        report["cpu_sweep"] = True

    with tempfile.TemporaryDirectory(prefix="prof_") as workdir:
        for name, description, flags in ladder:
            print(f"[{node}] === {name}: {description}", flush=True)
            res = run_variant(name, flags, args.iterations, args.warmup, cpus, workdir,
                              args.env_id, log_dir)
            res["description"] = description
            report["variants"].append(res)
            print(f"[{node}] {name}: {res.get('steps_per_second', 'FAILED')}", flush=True)
            # Flush after every variant so a job killed at its walltime still leaves its results.
            with open(out_path, "w") as f:
                json.dump(report, f, indent=1)

        if not args.skip_startup:
            print(f"[{node}] === start-up measurement", flush=True)
            report["startup"] = measure_startup(cpus, workdir, args.env_id, log_dir)
            with open(out_path, "w") as f:
                json.dump(report, f, indent=1)

    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    os.chmod(out_path, 0o660)
    print(f"[{node}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
