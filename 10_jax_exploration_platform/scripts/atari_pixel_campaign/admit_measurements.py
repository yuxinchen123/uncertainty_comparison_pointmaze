"""Admit paired measurements only after recorded checks pass on both GPU types."""

import argparse
import hashlib
import json
from pathlib import Path

from controller import (REFERENCE_ALGORITHM, SENTINEL_NAME, atomic_json,
                        read_json, ready_specification, tick, verify_source)


def verified_check(path, commit, context, kind):
    """Validate the actual command, GPU, runtime settings and source of one check."""
    path = path.resolve()
    record = read_json(path)
    if record.get("passed") is not True or record.get("exit_code") != 0:
        raise ValueError(f"required check has not passed: {path}")
    if record.get("source_commit") != commit:
        raise ValueError(f"check used another source commit: {path}")
    hardware = record["hardware"]
    query = hardware["gpu_query"]
    if context == "h100":
        if "H100" not in query:
            raise ValueError("the H100 check must record an actual H100")
    elif hardware["hostname"] != "jaguar03" or "A4500" not in query:
        raise ValueError("the A4500 check must have run on jaguar03")
    runtime = hardware["runtime_environment"]
    if (runtime.get("JAX_ENABLE_X64") != "true"
            or runtime.get("JAX_DEFAULT_MATMUL_PRECISION") != "highest"
            or "--xla_gpu_exclude_nondeterministic_ops=true" not in runtime.get("XLA_FLAGS", "")):
        raise ValueError("required deterministic full-precision settings were not recorded")
    command = record["command"]
    if kind == "learner":
        if "pytest" not in command or "-k" in command or "-m" not in command:
            raise ValueError("learner evidence must include the full pytest suite")
        if not any(value.endswith("tests/atari_pixels") for value in command):
            raise ValueError("learner evidence does not include the pixel learner tests")
        if not any(value.endswith("tests/test_atari_pixel_environment.py") for value in command):
            raise ValueError("learner evidence does not include the environment checks")
    elif (not any(value.endswith("/check_rendering.py") for value in command)
          or command[-2:] != ["--candidate", "all"]):
        raise ValueError("renderer evidence must check all renderer candidates")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "command": command, "hardware": hardware}


def admit(args):
    """Write the parent's explicit source record and queue the thirty initial ideas."""
    source_root, run_dir = args.source_root.resolve(), args.run_dir.resolve()
    verify_source(source_root, args.source_commit)
    counts = sorted(set(args.matched_copy_counts))
    if 1 not in counts or min(counts) < 1:
        raise ValueError("matched copy counts must be positive and include one")
    configuration = read_json(args.reference_configuration_source)
    configuration["n_copies"] = 1
    if any(configuration.get(name) != value for name, value in REFERENCE_ALGORITHM.items()):
        raise ValueError("the reference differs from the agreed CleanRL configuration")
    if any(configuration.get(name) for name in (
            "per_copy_learning_rates", "per_copy_intrinsic_coefficients", "per_copy_scalar_settings")):
        raise ValueError("this infrastructure study uses one common algorithm setting")
    destination = run_dir / "data/source_checks" / args.source_commit
    gate_paths = {}
    for context, learner, renderer in (
            ("h100", args.h100_learner_record, args.h100_renderer_record),
            ("jaguar03_rtx_a4500", args.a4500_learner_record, args.a4500_renderer_record)):
        evidence = {"learner": verified_check(learner, args.source_commit, context, "learner"),
                    "renderer": verified_check(renderer, args.source_commit, context, "renderer")}
        path = destination / f"{context}.json"
        atomic_json(path, {"passed": True, "source_commit": args.source_commit,
                           "hardware_class": context, "checks": evidence})
        gate_paths[context] = str(path)
    reference_path = destination / "reference_configuration.json"
    atomic_json(reference_path, configuration)
    specification = {"authorized": True, "source_root": str(source_root),
                     "source_commit": args.source_commit, "contexts": list(gate_paths),
                     "matched_copy_counts": counts, "gate_records": gate_paths,
                     "reference_configuration_path": str(reference_path),
                     "seed_start": args.seed_start, "rounds": 7, "iterations": 10}
    path = run_dir / SENTINEL_NAME
    if path.exists() and read_json(path) != specification:
        raise ValueError("an existing source admission differs; review the study revision explicitly")
    atomic_json(path, specification)
    if ready_specification(run_dir) is None:
        raise ValueError("the controller did not accept the recorded source checks")
    state = tick(run_dir)
    state.pop("remaining_work", None)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--reference-configuration-source", type=Path, required=True)
    parser.add_argument("--matched-copy-counts", nargs="+", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=1)
    for hardware in ("h100", "a4500"):
        for kind in ("learner", "renderer"):
            parser.add_argument(f"--{hardware}-{kind}-record", type=Path, required=True)
    admit(parser.parse_args())
