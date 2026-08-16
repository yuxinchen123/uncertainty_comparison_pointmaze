#!/usr/bin/env python
"""Run the ext4m training workload under a list of optimization CONDITIONS, one after another on an
idle exclusive node, and report both numbers that matter for each: how fast it ran, and whether the
run it produced is still the run the sweep would have produced.

Each condition is one subprocess of the production trainer (through patched_entry.py, so the live
repo is never edited) driven by a REAL ext4m queue marker — the same algorithm, beta, seed and env
knobs the sweep uses — shortened to --steps timesteps so a condition costs minutes instead of days.

Two outputs per condition:
  speed     wall seconds for the identical workload (and the record's own runtime_seconds);
            steps/s = steps / runtime_seconds. Conditions are compared only against the baseline
            measured on the SAME node in the SAME job, so node class and contention cancel.
  validity  the run's record JSON with `runtime_seconds` dropped, compared field-by-field against
            the baseline condition's record for the same config. Equal => the optimization does not
            move a single logged number, i.e. every result already produced stays valid. Anything
            else => the candidate is a FUTURE change, not an applied one.

Usage:
  python run_conditions.py --markers <dir> --conditions baseline,optflags,... \
      --steps 10000 --eval_freq 2500 --out ../../data/results_<jobid>.json
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EFF = os.path.dirname(os.path.dirname(HERE))          # the 2026-08-13-ext4m-efficiency folder
PROJ = "/p/rlprojects/RND/07_reconstruction"
PY = "/p/rlprojects/RND/.venvs/exploration/bin/python"
TCMALLOC = "/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

# name -> (extra CLI flags for train.py, extra environment). Everything else is production.
CONDITIONS = {
    # the reference: exactly what the sweep runs today
    "baseline":    ([], {}),
    # the two bit-exact switches that already ship in train.py but have never been turned on in a
    # production sweep (built + measured 2026-06-25: +6.1% combined on a 2-thread node)
    "optflags":    (["--opt_polyak_foreach=True", "--opt_torch_reward=True"], {}),
    # candidate: foreach Adam/SGD (torch's CPU default is the single-tensor python loop)
    "adamforeach": ([], {"EFF_PATCHES": "adam_foreach,sgd_foreach"}),
    # candidate: cache the observation-normalization tensors between RunningMeanStd updates
    "normcache":   ([], {"EFF_PATCHES": "norm_cache"}),
    # candidate: stop rescanning the long-lived env/model objects in every gc generation
    "gcfreeze":    ([], {"EFF_PATCHES": "gc_freeze"}),
    # candidate: tcmalloc instead of glibc malloc (allocator only — cannot change a single number)
    "tcmalloc":    ([], {"LD_PRELOAD": TCMALLOC}),
    # candidate: one inter-op thread (the pool is sized from the node's 46 cores by default)
    "interop1":    ([], {"EFF_PATCHES": "interop1"}),
    # candidate: skip oneDNN for these small MLPs (may not be record-identical — the check decides)
    "mkldnnoff":   ([], {"EFF_PATCHES": "mkldnn_off"}),
    # candidate: inference_mode instead of no_grad for the bonus forwards
    "infmode":     ([], {"EFF_PATCHES": "inference_mode"}),
    # candidate: one malloc arena instead of glibc's per-thread arenas (allocator only)
    "arena1":      ([], {"MALLOC_ARENA_MAX": "1"}),
    # the surviving stack plus oneDNN off (screened identical twice on alg23; the record check
    # re-decides it on the other two configs)
    "combined2":   (["--opt_polyak_foreach=True", "--opt_torch_reward=True"],
                    {"EFF_PATCHES": "adam_foreach,sgd_foreach,norm_cache,gc_freeze,mkldnn_off",
                     "LD_PRELOAD": TCMALLOC}),
    # everything that survives, together
    "combined":    (["--opt_polyak_foreach=True", "--opt_torch_reward=True"],
                    {"EFF_PATCHES": "adam_foreach,sgd_foreach,norm_cache,gc_freeze",
                     "LD_PRELOAD": TCMALLOC}),
}


def marker_args(marker, out_dir, steps, eval_freq):
    """Turn one ext4m queue marker into the trainer's argv — the worker's build_cmd, with the
    checkpoint plumbing disabled and the horizon shortened.

    before: marker {"algorithm": "rnd_next_state", "beta": "30", "a_seed": 1500, "params": {...},
                    "fixed": {"total_timesteps": 4000000, "eval_freq": 50000, ...}}
    after:  [--algorithm=rnd_next_state, --beta=30, ..., --total_timesteps=10000, --eval_freq=2500]
            (the overrides come LAST, so they win over the marker's own values in argparse)
    """
    args = [
        f'--ckpt_dir={os.path.join(out_dir, "ckpt")}',
        # far above the horizon so no checkpoint is ever written, and a multiple of eval_freq
        # (train4m exits at startup otherwise -- it cost one dead job to learn that)
        f'--ckpt_every={eval_freq * 1000000}',
        '--suspend_end_epoch=0',         # no walltime suspend
        f'--algorithm={marker["algorithm"]}',
        f'--beta={marker["beta"]}',
        f'--a_seed={marker["a_seed"]}',
        f'--env_setup={marker["env_setup"]}',
        '--z_logging_mode=local',
        '--use_wandb=False',
        f'--local_log_dir={out_dir}',
        f'--run_id={marker["run_id"]}',
        f'--run_total={marker["run_total"]}',
    ]
    for k, v in marker.get("params", {}).items():
        args.append(f"--{k}={v}")
    for k, v in marker["fixed"].items():
        args.append(f"--{k}={v}")
    args += [f"--total_timesteps={steps}", f"--eval_freq={eval_freq}"]
    return args


def record_of(out_dir):
    """The single run record written under <out_dir>/local/, as a dict (or None if absent)."""
    local = os.path.join(out_dir, "local")
    files = sorted(f for f in os.listdir(local) if f.endswith(".json")) if os.path.isdir(local) else []
    if not files:
        return None
    with open(os.path.join(local, files[0])) as fh:
        return json.load(fh)


def science_of(record):
    """The part of a record that must not move: everything except the wall-clock it took.
    before: {..., "runtime_seconds": 431.7, "eval_history": [...]}
    after:  {..., "eval_history": [...]}  (runtime dropped, nothing else touched)"""
    if record is None:
        return None
    trimmed = copy.deepcopy(record)
    trimmed.pop("runtime_seconds", None)
    return trimmed


def first_difference(a, b, path="record"):
    """A human-readable path to the first place two records differ, or None when identical."""
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f"{path}: keys {sorted(set(a) ^ set(b))[:4]}"
        for k in a:
            d = first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if a != b:
        return f"{path}: {a!r} vs {b!r}"
    return None


def run_one(name, marker, marker_name, steps, eval_freq, data_dir, log_dir, rep=0):
    """Run one (condition, config) pair to completion; return its measurement row.

    `rep` distinguishes repeats of the same pair. Repeating the BASELINE is itself a measurement:
    two baselines must come out `IDENTICAL`, which is what makes the record check meaningful for
    every other condition."""
    flags, env_extra = CONDITIONS[name]
    tag = name if rep == 0 else f"{name}#{rep}"
    out_dir = os.path.join(data_dir, f"{marker_name}__{tag}")
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    # production worker environment: one thread per worker, no user-site shadowing of the shared env
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "PYTHONNOUSERSITE": "1", "TMPDIR": "/tmp"})
    env.update(env_extra)
    argv = [PY, os.path.join(HERE, "patched_entry.py")] + marker_args(marker, out_dir, steps, eval_freq) + flags
    log_path = os.path.join(log_dir, f"{marker_name}__{tag}.log")
    t0 = time.time()
    with open(log_path, "w") as log:
        rc = subprocess.call(argv, cwd=PROJ, env=env, stdout=log, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    record = record_of(out_dir)
    runtime = record.get("runtime_seconds") if record else None
    row = {"condition": name, "rep": rep, "config": marker_name, "rc": rc, "wall_s": round(wall, 2),
           "runtime_s": None if runtime is None else round(runtime, 2),
           "steps_per_s": None if not runtime else round(steps / runtime, 3),
           "log": log_path, "record_dir": out_dir}
    print(f"[bench] {marker_name:<12} {tag:<14} rc={rc} wall={wall:7.1f}s "
          f"steps/s={row['steps_per_s']}", flush=True)
    return row, science_of(record)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--markers", default=os.path.join(EFF, "data", "markers"))
    p.add_argument("--configs", default="")           # marker basenames without .json; "" = all
    p.add_argument("--conditions", default="baseline,optflags,adamforeach,normcache,gcfreeze,tcmalloc,combined")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--eval_freq", type=int, default=2500)
    p.add_argument("--repeat", type=int, default=1, help="times to run each (config, condition)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    names = [n.strip() for n in args.conditions.split(",") if n.strip()]
    unknown = [n for n in names if n not in CONDITIONS]
    if unknown:
        sys.exit(f"unknown condition(s): {unknown}; known: {sorted(CONDITIONS)}")
    markers = {}
    for fn in sorted(os.listdir(args.markers)):
        if not fn.endswith(".json"):
            continue
        key = fn[:-5]
        if args.configs and key not in args.configs.split(","):
            continue
        with open(os.path.join(args.markers, fn)) as fh:
            markers[key] = json.load(fh)

    data_dir = os.path.join(EFF, "data", f"runs_{os.environ.get('SLURM_JOB_ID', 'local')}")
    log_dir = os.path.join(EFF, "logs", f"conditions_{os.environ.get('SLURM_JOB_ID', 'local')}")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    rows, baseline_science = [], {}
    for marker_name, marker in markers.items():
        for name in names:
          for rep in range(args.repeat):
            row, science = run_one(name, marker, marker_name, args.steps, args.eval_freq,
                                   data_dir, log_dir, rep)
            if name == "baseline" and marker_name not in baseline_science:
                baseline_science[marker_name] = science
            base = baseline_science.get(marker_name)
            if science is None:
                row["validity"] = "NO RECORD (run failed)"
            elif base is None:
                row["validity"] = "no baseline yet"
            else:
                diff = first_difference(base, science)
                row["validity"] = "IDENTICAL" if diff is None else f"DIFFERS at {diff}"
            print(f"        validity: {row['validity']}", flush=True)
            rows.append(row)
            with open(args.out, "w") as fh:      # rewrite after every run: a killed job keeps its data
                json.dump({"node": os.environ.get("SLURMD_NODENAME", ""),
                           "job": os.environ.get("SLURM_JOB_ID", ""),
                           "steps": args.steps, "eval_freq": args.eval_freq, "rows": rows}, fh, indent=1)

    # speedups against each config's baseline
    print("\n[bench] summary (steps/s, ratio vs baseline, validity)", flush=True)
    for marker_name in markers:
        base = next((r for r in rows if r["config"] == marker_name and r["condition"] == "baseline"
                     and r["steps_per_s"]), None)
        for r in rows:
            if r["config"] != marker_name:
                continue
            ratio = (r["steps_per_s"] / base["steps_per_s"]) if base and r["steps_per_s"] else float("nan")
            r["ratio_vs_baseline"] = round(ratio, 4)
            print(f"  {marker_name:<12} {r['condition']:<12} {r['steps_per_s']} "
                  f"x{ratio:.4f}  {r['validity']}", flush=True)
    with open(args.out, "w") as fh:
        json.dump({"node": os.environ.get("SLURMD_NODENAME", ""),
                   "job": os.environ.get("SLURM_JOB_ID", ""),
                   "steps": args.steps, "eval_freq": args.eval_freq, "rows": rows}, fh, indent=1)
    print(f"\n[bench] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
