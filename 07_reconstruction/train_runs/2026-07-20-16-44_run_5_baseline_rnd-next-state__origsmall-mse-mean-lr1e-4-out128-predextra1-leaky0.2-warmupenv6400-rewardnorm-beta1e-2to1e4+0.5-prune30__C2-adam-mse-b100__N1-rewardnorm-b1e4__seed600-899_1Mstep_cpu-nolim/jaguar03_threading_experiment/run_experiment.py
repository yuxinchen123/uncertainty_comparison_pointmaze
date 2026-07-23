#!/usr/bin/env python
"""jaguar03 CPU-threading experiment: which thread->core layout maximizes RND-training throughput?

Runs three layouts SEQUENTIALLY on jaguar03, each at the SAME load (56 physical cores of socket 0 =
112 hardware threads), so the node's frequency-throttle is identical and only the mapping differs:
  A: 56 runs, OMP=2, 1 run per physical core (both HW threads of the core)      -> 112 threads/56 cores
  B: 28 runs, OMP=4, 1 run per 2 physical cores (4 HW threads)                   -> 112 threads/56 cores
  C: 112 runs, OMP=1, 2 runs share a physical core (1 HW thread each)            -> 112 threads/56 cores
(jaguar03 topology: physical core k owns HW threads {k, k+112}; socket 0 = cores 0..55.)

Each run is the real run-5 origsmall workload (train.py) with a huge step budget so it never finishes;
we run it for WINDOW_SEC and measure how many env steps it completed. train.py flushes its JSON every
--eval_freq steps (set small = 200), and this driver ALSO polls every POLL_SEC and appends
(elapsed_s, run_id, step, runtime_s) to progress_<cfg>.csv -> the frequent step log. It samples the
node's median core MHz too, so the report shows the throttle regime each layout ran under.

Run (on jaguar03, whole node): srun --exclusive --nodelist=jaguar03 ... python run_experiment.py
"""
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
PROJ = "/p/rlprojects/RND/07_reconstruction"
PY = "/p/rlprojects/RND/.venvs/exploration/bin/python"
TRAIN = os.path.join(PROJ, "train.py")

WINDOW_SEC = int(os.environ.get("WINDOW_SEC", "1080"))   # 18 min of training measured per layout
STAGGER_SEC = float(os.environ.get("STAGGER_SEC", "0.2"))  # delay between run launches; raise it for the
#                                                          216-run layout C so 216 simultaneous torch
#                                                          imports do not saturate the shared filesystem
POLL_SEC = 60                                            # step-progress + freq sample cadence
NCORES = 108                                             # FULL usable node: the OS reserves cores 108..111
#                                                          (srun affinity is 0-107,112-219), so 108 physical
#                                                          cores are usable. cores 0..107, sibling thread =
#                                                          core+112. Every layout loads all 108 cores / 216
#                                                          HW threads -> the real all-core 400 MHz throttle;
#                                                          only the thread->run mapping differs across A/B/C

# the run-5 origsmall arm workload, minus the slow env-steps obs warmup (space_sample keeps the same
# network/algorithm but a fast startup so the measured window is training, not warmup)
ARM = [
    "--algorithm=rnd_next_state", "--beta=1000",
    "--rnd_optimizer=adam", "--rnd_bonus_readout=mse_mean", "--rnd_lr=0.0001",
    "--rnd_update_proportion=1.0", "--rnd_activation=leaky_relu", "--rnd_predictor_extra_layers=1",
    "--rnd_obs_warmup_mode=space_sample", "--rnd_obs_warmup_steps=200",
    "--rnd_reward_norm=True", "--rnd_reward_norm_gamma=0.99",
    "--rnd_bias_init=zero", "--rnd_weight_init=orthogonal",
    "--total_timesteps=100000000", "--eval_freq=50", "--n_eval_episodes=3",
    "--z_logging_mode=local", "--use_wandb=False",
]


def masks_for(cfg):
    """Return the list of (run_id, omp_threads, taskset_cpu_list) for a layout, all within cores 0..55.
    A: run i -> core i's two threads {i, i+112}, OMP 2.
    B: run i -> cores {2i, 2i+1} (four threads), OMP 4.
    C: run j -> ONE thread (core j//2, thread 0 if j even else thread 1 = +112), OMP 1, 2 runs/core."""
    out = []
    if cfg.startswith("L"):                           # "L<n>": n runs, 1 thread each, 1 PER CORE (no HT
        n = int(cfg[1:])                              # sibling used) -> a LIGHT-load layout for the
        for i in range(n):                            # reproduction test (compare light vs full-node C)
            out.append((i, 1, f"{i}"))
        return out
    if cfg == "A":
        for i in range(NCORES):                       # 56 runs
            out.append((i, 2, f"{i},{i + 112}"))
    elif cfg == "B":
        for i in range(NCORES // 2):                  # 28 runs
            c0, c1 = 2 * i, 2 * i + 1
            out.append((i, 4, f"{c0},{c1},{c0 + 112},{c1 + 112}"))
    elif cfg == "C":
        for j in range(2 * NCORES):                   # 112 runs
            core = j // 2
            cpu = core if j % 2 == 0 else core + 112
            out.append((j, 1, f"{cpu}"))
    return out


def median_mhz():
    """Median live core MHz across all logical CPUs (the node's current throttle level)."""
    mhz = sorted(float(l.split(":")[1]) for l in open("/proc/cpuinfo") if l.startswith("cpu MHz"))
    return round(mhz[len(mhz) // 2], 1) if mhz else -1.0


def run_json(datadir, run_id, run_total):
    """Path of a run's per-run JSON (train.py names it <run_id 0-padded>_of_<total>.json under local/)."""
    name = f"{run_id:0{len(str(run_total))}d}_of_{run_total}.json"
    return os.path.join(datadir, "local", name)


def read_step_runtime(path):
    """(max step, runtime_seconds) from a run's JSON via a fast head+tail read, or (0, 0.0) if absent.
    step lives in train_history rows near the file head; runtime_seconds is a top-level field."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            head = fh.read(8192).decode("utf-8", "replace")
            if size > 8192:
                fh.seek(max(0, size - 4096))
                head += fh.read(4096).decode("utf-8", "replace")
    except OSError:
        return 0, 0.0
    steps = [int(x) for x in re.findall(r'"step":\s*(\d+)', head)]
    rt = re.search(r'"runtime_seconds":\s*([\d.]+)', head)
    return (max(steps) if steps else 0), (float(rt.group(1)) if rt else 0.0)


def launch_layout(cfg):
    """Launch every run of one layout, pinned via taskset, return the list of (run_id, Popen, datadir)."""
    datadir = os.path.join(HERE, "data", cfg)
    logdir = os.path.join(HERE, "logs", cfg)
    os.makedirs(logdir, exist_ok=True)
    spec = masks_for(cfg)
    total = len(spec)
    procs = []
    # launch each run pinned to its cpu list with its OMP thread budget; stagger 0.2s to ease import herd
    for run_id, omp, cpulist in spec:
        env = dict(os.environ, OMP_NUM_THREADS=str(omp), MKL_NUM_THREADS=str(omp),
                   OPENBLAS_NUM_THREADS=str(omp))
        argv = ["taskset", "-c", cpulist, PY, TRAIN, *ARM,
                f"--a_seed={600 + run_id}", f"--run_id={run_id}", f"--run_total={total}",
                f"--local_log_dir={datadir}"]
        logf = open(os.path.join(logdir, f"run_{run_id}.log"), "w")
        # new session so we can kill the whole process group cleanly at the window end
        p = subprocess.Popen(argv, cwd=PROJ, env=env, stdout=logf, stderr=subprocess.STDOUT,
                             start_new_session=True)
        procs.append((run_id, p, datadir, total))
        time.sleep(STAGGER_SEC)
    return procs


def poll_and_wait(cfg, procs):
    """For WINDOW_SEC, every POLL_SEC append each run's (elapsed, run_id, step, runtime) to a CSV and
    sample node MHz; return the list of MHz samples. This is the frequent step log the report reads."""
    csv_path = os.path.join(HERE, "data", cfg, f"progress_{cfg}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    mhz_samples = []
    with open(csv_path, "w") as csv:
        csv.write("elapsed_s,run_id,step,runtime_s,median_mhz\n")
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            mhz = median_mhz()
            mhz_samples.append(mhz)
            # snapshot every run's current step/runtime from its JSON
            for run_id, _p, datadir, total in procs:
                step, rt = read_step_runtime(run_json(datadir, run_id, total))
                csv.write(f"{elapsed:.0f},{run_id},{step},{rt:.1f},{mhz}\n")
            csv.flush()
            print(f"[{cfg}] t={elapsed:6.0f}s  median_MHz={mhz:7.1f}  "
                  f"steps(min/med/max)={_step_stats(procs)}", flush=True)
            if elapsed >= WINDOW_SEC:
                break
            time.sleep(POLL_SEC)
    return mhz_samples


def _step_stats(procs):
    """min/median/max current step across a layout's runs (for the live progress print)."""
    steps = sorted(read_step_runtime(run_json(d, r, t))[0] for r, _p, d, t in procs)
    if not steps:
        return "0/0/0"
    return f"{steps[0]}/{steps[len(steps) // 2]}/{steps[-1]}"


def kill_layout(procs):
    """Terminate every run's process group (SIGTERM, then SIGKILL), then BLOCK until every child has
    actually exited -- so no run from this layout is still on a core when the next layout launches
    (strict sequential isolation: layouts must not overlap or their loads/throttle would mix)."""
    for _run_id, p, _d, _t in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(8)
    for _run_id, p, _d, _t in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    # wait (up to ~45s) for every child process to be reaped before returning
    deadline = time.time() + 45
    while time.time() < deadline:
        alive = sum(1 for _r, p, _d, _t in procs if p.poll() is None)
        if alive == 0:
            break
        time.sleep(2)
    alive = sum(1 for _r, p, _d, _t in procs if p.poll() is None)
    print(f"[kill] {len(procs) - alive}/{len(procs)} exited; {alive} still alive after wait", flush=True)


def main():
    """Run layouts A, B, C in sequence; each launches, polls for WINDOW_SEC, then is killed."""
    order = sys.argv[1:] or ["A", "B", "C"]
    print(f"node={os.uname().nodename} window_sec={WINDOW_SEC} order={order}", flush=True)
    for cfg in order:
        print(f"==== layout {cfg} START {time.strftime('%H:%M:%S')} ====", flush=True)
        procs = launch_layout(cfg)
        print(f"[{cfg}] launched {len(procs)} runs", flush=True)
        mhz = poll_and_wait(cfg, procs)
        kill_layout(procs)
        med = sorted(mhz)[len(mhz) // 2] if mhz else -1
        print(f"==== layout {cfg} END   {time.strftime('%H:%M:%S')}  median_MHz~{med} ====", flush=True)
        time.sleep(20)  # let the node settle before the next layout
    print("EXPERIMENT_DONE", flush=True)


if __name__ == "__main__":
    main()
