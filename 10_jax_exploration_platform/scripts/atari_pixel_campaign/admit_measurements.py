"""Admit paired measurements after complete checks and full-size capacity measurements."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from controller import (REFERENCE_ALGORITHM, SENTINEL_NAME, read_json,
                        ready_specification, tick, verify_source)


CONTEXTS = ("h100", "jaguar03_rtx_a4500")


def evidence_file(path):
    """Read and hash the same original bytes so an admission identifies its exact evidence."""
    path = Path(path).resolve()
    contents = path.read_bytes()
    return json.loads(contents), {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest()}


def verified_hardware(hardware):
    """Identify the actual GPU context and require the agreed deterministic precision settings."""
    query = hardware["gpu_query"]
    if "H100" in query:
        context = "h100"
    elif hardware["hostname"] == "jaguar03" and "A4500" in query:
        context = "jaguar03_rtx_a4500"
    else:
        raise ValueError("evidence must record an H100 or an RTX A4500 on jaguar03")
    runtime = hardware["runtime_environment"]
    deterministic = [value for value in shlex.split(runtime.get("XLA_FLAGS", ""))
                     if value.startswith("--xla_gpu_exclude_nondeterministic_ops=")]
    if (runtime.get("JAX_ENABLE_X64") != "true"
            or runtime.get("JAX_DEFAULT_MATMUL_PRECISION") != "highest"
            or deterministic != ["--xla_gpu_exclude_nondeterministic_ops=true"]):
        raise ValueError("required deterministic full-precision settings were not recorded")
    return context


def source_path(command_path, relative, commit, run_dir, source_root=None):
    """Accept only the named file in this commit's recorded snapshot or verified frozen checkout."""
    path = Path(command_path)
    roots = [Path(run_dir) / "data/code_snapshots" / commit]
    if source_root is not None:
        roots.append(Path(source_root))
    expected = [root / "10_jax_exploration_platform" / relative for root in roots]
    if not path.is_absolute() or path not in expected or not path.exists():
        raise ValueError(f"command path does not name the requested source commit: {path}")
    if source_root is not None and path == expected[-1]:
        # The shared source check covers the learner and campaign scripts. Include
        # the test files here before accepting a checkout instead of its archive.
        verify_source(source_root, commit)
        changes = subprocess.run(["git", "-C", str(source_root), "status", "--porcelain",
            "--untracked-files=all", "--", "10_jax_exploration_platform/tests/atari_pixels",
            "10_jax_exploration_platform/tests/test_atari_pixel_environment.py"],
            capture_output=True, text=True, check=True).stdout.strip()
        if changes:
            raise ValueError("frozen checkout test files differ from the recorded commit: " + changes)
    return path


def verified_check(path, commit, context, kind, run_dir, source_root):
    """Require the exact complete check command, source commit, hardware and precision settings."""
    record, evidence = evidence_file(path)
    if record.get("passed") is not True or record.get("exit_code") != 0:
        raise ValueError(f"required check has not passed: {path}")
    if record.get("source_commit") != commit:
        raise ValueError(f"check used another source commit: {path}")
    hardware = record["hardware"]
    if verified_hardware(hardware) != context:
        raise ValueError(f"check used another hardware context: {path}")
    command = record["command"]
    if (not isinstance(command, list) or not command
            or not all(isinstance(value, str) for value in command)
            or not re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(command[0]).name)):
        raise ValueError("check command must invoke Python directly")
    if kind == "learner":
        if len(command) != 6 or command[1:4] != ["-m", "pytest", "-q"]:
            raise ValueError("learner evidence must run the exact full pytest suite without selectors")
        first = source_path(command[4], "tests/atari_pixels", commit, run_dir, source_root)
        second = source_path(command[5], "tests/test_atari_pixel_environment.py", commit, run_dir, source_root)
        if first.parents[1] != second.parents[1]:
            raise ValueError("both learner test paths must come from the same frozen source")
    elif kind == "renderer":
        if len(command) != 4 or command[2:] != ["--candidate", "all"]:
            raise ValueError("renderer evidence must check exactly all renderer candidates")
        source_path(command[1], "scripts/atari_pixel_campaign/check_rendering.py",
                    commit, run_dir, source_root)
    else:
        raise ValueError(f"unknown check kind: {kind}")
    return {**evidence, "command": command, "hardware": hardware}


def capacity_source(run_dir, unit_id, hardware):
    """Recover the original capacity source from the exact submitted job or recorded direct process."""
    run_dir = Path(run_dir)
    if hardware.get("slurm_job_id"):
        rows = [line.split("\t") for line in (run_dir / "slurm/submissions.tsv").read_text().splitlines()]
        matches = [(row[2], Path(row[3])) for row in rows
                   if len(row) == 4 and row[0] == hardware["slurm_job_id"]]
    else:
        matches = [(record["source_commit"], Path(record["script"]))
                   for record in read_json(run_dir / "processes.json") if record.get("unit") == unit_id]
    if len(matches) != 1:
        raise ValueError("capacity evidence must identify exactly one recorded submission")
    commit, script = matches[0]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("capacity submission must record its original full source commit")
    text = script.read_text()
    commands = [shlex.split(line) for line in text.splitlines() if "run_unit.py" in line]
    if len(commands) != 1:
        raise ValueError("capacity script must contain exactly one run_unit command")
    command = commands[0]
    if len(command) < 2 or not re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(command[0]).name):
        raise ValueError("capacity script must invoke Python directly")
    runner = source_path(command[1], "scripts/atari_pixel_campaign/run_unit.py", commit, run_dir)
    for name, value in (("--unit-id", unit_id), ("--run-dir", str(run_dir)), ("--mode", "benchmark")):
        if command.count(name) != 1 or command[command.index(name) + 1] != value:
            raise ValueError("capacity command differs from its recorded unit, run directory or mode")
    return {"source_commit": commit, "source_root": str(runner.parents[3]), "command": command,
            "script": {"path": str(script.resolve()), "sha256": hashlib.sha256(text.encode()).hexdigest()}}


def verified_capacity(path, run_dir, configuration, counts):
    """Require seven completed full-size reference rounds and preserve original source evidence."""
    path = Path(path).resolve()
    if path.name != "result.json" or path.parent.parent != Path(run_dir) / "data":
        raise ValueError("capacity result must belong to a direct data unit in this run")
    result, result_evidence = evidence_file(path)
    arguments, argument_evidence = evidence_file(path.with_name("arguments.json"))
    actual, configuration_evidence = evidence_file(path.with_name("config_resolved.json"))
    hardware, hardware_evidence = evidence_file(path.with_name("hardware.json"))
    context = verified_hardware(hardware)
    copies = result.get("copies")
    if type(copies) is not int or copies not in counts or copies <= 1:
        raise ValueError("capacity evidence must match a requested copy count above one")
    expected = {**configuration, "n_copies": copies}
    if actual != expected:
        raise ValueError("capacity measurement differs from the full reference learner configuration")
    required = {"mode": "benchmark", "variant": "reference", "copies": copies,
                "envs_per_copy": 128, "rollout_steps": 128, "iterations": 10,
                "warmup_rollouts": 50, "epochs": 4, "minibatches": 4, "rounds": 7,
                "unit_id": path.parent.name, "run_dir": str(run_dir)}
    if any(arguments.get(name) != value for name, value in required.items()):
        raise ValueError("capacity arguments must record the complete seven-round reference benchmark")
    if arguments.get("environment_configuration") is not None or arguments.get("configuration") is not None:
        raise ValueError("capacity proof must use the original environment and learner defaults")
    rounds = result.get("rounds", [])
    if (result.get("completed") is not True or result.get("mode") != "benchmark"
            or result.get("variant") != "reference" or len(rounds) != 7):
        raise ValueError("capacity benchmark has not completed all seven rounds")
    transitions = copies * 128 * 128 * 10
    for index, record in enumerate(rounds):
        if (record.get("record") != "benchmark" or record.get("round") != index
                or record.get("copies") != copies or record.get("variant") != "reference"
                or record.get("transitions") != transitions
                or not isinstance(record.get("seconds"), (int, float))
                or not math.isfinite(record["seconds"]) or record["seconds"] <= 0):
            raise ValueError("capacity benchmark contains incomplete or differently sized rounds")
    source = capacity_source(run_dir, path.parent.name, hardware)
    return {"hardware_class": context, "copies": copies, "result": result_evidence,
            "arguments": argument_evidence, "configuration_record": configuration_evidence,
            "configuration": actual, "hardware_record": hardware_evidence,
            "hardware": hardware, **source}


@contextmanager
def admission_lock(run_dir):
    """Serialize admission publication while keeping all existing JSON records immutable."""
    with (Path(run_dir) / ".source_admission.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def check_existing(publications):
    """Reject every content collision before publishing any proposed admission JSON."""
    for path, value in publications:
        json.dumps(value, allow_nan=False)
        if path.exists() and read_json(path) != value:
            raise ValueError(f"an immutable admission record already has different contents: {path}")


def publish_once(path, value):
    """Create a complete JSON file exclusively, retaining identical existing files byte for byte."""
    if path.exists():
        check_existing([(path, value)])
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            check_existing([(path, value)])
    finally:
        temporary.unlink()


def admit(args):
    """Validate every input before publishing immutable source evidence and queue authorization."""
    source_root, run_dir = args.source_root.resolve(), args.run_dir.resolve()
    verify_source(source_root, args.source_commit)
    counts = sorted(set(args.matched_copy_counts))
    if not counts or 1 not in counts or any(type(value) is not int or value < 1 for value in counts):
        raise ValueError("matched copy counts must be positive and include one")
    if type(args.seed_start) is not int or args.seed_start < 0:
        raise ValueError("the starting seed must be a nonnegative integer")
    configuration, configuration_evidence = evidence_file(args.reference_configuration_source)
    configuration = {**configuration, "n_copies": 1}
    if any(configuration.get(name) != value for name, value in REFERENCE_ALGORITHM.items()):
        raise ValueError("the reference differs from the agreed CleanRL configuration")
    if any(configuration.get(name) for name in (
            "per_copy_learning_rates", "per_copy_intrinsic_coefficients", "per_copy_scalar_settings")):
        raise ValueError("this infrastructure study uses one common algorithm setting")
    capacities = [verified_capacity(path, run_dir, configuration, counts) for path in args.capacity_record]
    keys = [(record["hardware_class"], record["copies"]) for record in capacities]
    required = {(context, copies) for context in CONTEXTS for copies in counts if copies > 1}
    if len(keys) != len(set(keys)) or set(keys) != required:
        raise ValueError("each requested copy count above one requires one capacity record from each GPU context")
    destination = run_dir / "data/source_checks" / args.source_commit
    publications, gate_paths = [], {}
    for context, learner, renderer in (
            ("h100", args.h100_learner_record, args.h100_renderer_record),
            ("jaguar03_rtx_a4500", args.a4500_learner_record, args.a4500_renderer_record)):
        evidence = {"learner": verified_check(learner, args.source_commit, context, "learner", run_dir, source_root),
                    "renderer": verified_check(renderer, args.source_commit, context, "renderer", run_dir, source_root)}
        path = destination / f"{context}.json"
        publications.append((path, {"passed": True, "source_commit": args.source_commit,
            "hardware_class": context, "checks": evidence,
            "capacity_checks": [record for record in capacities if record["hardware_class"] == context]}))
        gate_paths[context] = str(path)
    reference_path = destination / "reference_configuration.json"
    publications.append((reference_path, configuration))
    specification = {"authorized": True, "source_root": str(source_root),
                     "source_commit": args.source_commit, "contexts": list(gate_paths),
                     "matched_copy_counts": counts, "gate_records": gate_paths,
                     "reference_configuration_path": str(reference_path),
                     "reference_configuration_source": configuration_evidence,
                     "seed_start": args.seed_start, "rounds": 7, "iterations": 10}
    publications.append((run_dir / SENTINEL_NAME, specification))
    with admission_lock(run_dir):
        check_existing(publications)
        for path, value in publications:
            publish_once(path, value)
    if ready_specification(run_dir) is None:
        raise ValueError("the controller did not accept the recorded source checks")
    # Before the adaptive phase, run_case must pass reference_environment from its
    # immutable specification to paired_benchmark. The initial original baseline
    # uses no environment overrides; this admission does not authorize that change.
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
    parser.add_argument("--capacity-record", action="append", type=Path, default=[])
    parser.add_argument("--seed-start", type=int, default=1)
    for hardware in ("h100", "a4500"):
        for kind in ("learner", "renderer"):
            parser.add_argument(f"--{hardware}-{kind}-record", type=Path, required=True)
    admit(parser.parse_args())
