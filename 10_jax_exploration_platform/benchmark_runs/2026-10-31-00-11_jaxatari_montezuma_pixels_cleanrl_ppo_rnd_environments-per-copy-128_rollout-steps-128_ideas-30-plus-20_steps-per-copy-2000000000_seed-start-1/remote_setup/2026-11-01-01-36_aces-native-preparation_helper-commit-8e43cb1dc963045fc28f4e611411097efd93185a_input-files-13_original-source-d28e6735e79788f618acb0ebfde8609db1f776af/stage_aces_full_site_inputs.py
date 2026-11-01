"""Stage one fixed ACES input list and run only native metadata preparation.

Execution requires portal05 and an already open ACES master. There is one SSH
session, no authentication fallback, no master check, no retry and no scheduler
action. Original raw admission files, old checkouts and completed packets are
never replaced. Stdlib-only --describe validates local inputs without networking.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import shlex
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading


PLAN_SHA256 = "30892127adbc487f157dcb36d89e79635041278b09ca746d7b9e7139d9cb9d8f"
BASE = Path("/scratch/user/u.sl359075/rnd_atari_pixels")
PYTHON = "/scratch/user/u.sl359075/envs/rnd_atari_pixels_jax/bin/python"
HELPER_COMMIT = "8e43cb1dc963045fc28f4e611411097efd93185a"
SOURCE_COMMIT = "d28e6735e79788f618acb0ebfde8609db1f776af"
CLEANRL_COMMIT = "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
BLOCK = 1024 * 1024
SSH_TIMEOUT = 600
MAX_HEADER = 2 * 1024 * 1024
MAX_RETURN_FILE = 8 * 1024 * 1024
SSH_OPTIONS = ["ssh", "-T", "-o", "ControlMaster=no", "-o", "ProxyCommand=false", "-o", "BatchMode=yes",
    "-o", "NumberOfPasswordPrompts=0", "-o", "ConnectionAttempts=1", "-o", "ConnectTimeout=10", "aces"]
NUMERICAL_MODULES = ("jax", "jaxlib", "numpy", "torch", "flax", "optax")
REMOTE_COMMANDS = []


def digest(contents):
    """Hash exact bytes rather than a reserialized document."""
    return hashlib.sha256(contents).hexdigest()


def utc_now():
    """Use an explicit UTC timestamp for each actual metadata observation."""
    return datetime.now(timezone.utc).isoformat()


def ordinary(path, *, directory=False):
    """Reject changed lexical paths, symlinks and unexpected file types."""
    path = Path(path)
    if not path.is_absolute() or str(path) != str(PurePosixPath(path)) or path != path.resolve(strict=True):
        raise ValueError("an exact ordinary absolute path is required: " + str(path))
    mode = path.lstat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError("a staging artifact has the wrong ordinary type: " + str(path))
    return path


def file_record(path):
    """Hash a regular file incrementally and retain its exact path and length."""
    path = ordinary(path)
    hasher, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BLOCK), b""):
            hasher.update(chunk)
            size += len(chunk)
    return {"path": str(path), "sha256": hasher.hexdigest(), "bytes": size}


def checked_plan(contents):
    """Accept only the reviewed exact plan and its bounded path and command layout."""
    if digest(contents) != PLAN_SHA256:
        raise ValueError("the native staging plan differs from the reviewed SHA256")
    plan = json.loads(contents)
    root = Path(plan["remote_inputs_directory"])
    helper = BASE / "git/checkouts/study_helpers_8e43cb1dc963"
    source = plan["native_numerical_checkout_required"]
    cleanrl = plan["native_cleanrl_checkout_required"]
    if (plan.get("schema_version") != 1 or plan.get("kind") != "aces_full_site_native_preparation_staging_plan"
            or root.parent != BASE / "remote_setup" or not root.name.endswith("_aces-full-site-inputs_original-source-d28e673_driver-580.178.04")
            or plan.get("published_helper_commit") != HELPER_COMMIT or Path(plan["native_helper_root"]) != helper
            or source["path"] != str(root / "numerical_source") or source["git_repository"] != str(BASE / "git/repository.git")
            or source["commit"] != SOURCE_COMMIT or cleanrl["path"] != str(root / "cleanrl_source") or cleanrl["commit"] != CLEANRL_COMMIT
            or plan["native_prepare_parameters"]["python"] != PYTHON
            or plan["native_prepare_parameters"]["helper-root"] != str(helper)):
        raise ValueError("the native staging plan has an unexpected target, source or Python")
    rows = plan["exact_copy_allowlist"]
    targets = [row["remote_path"] for row in rows]
    if len(rows) != 13 or len(set(targets)) != 13 or len({row["local_path"] for row in rows}) != 13:
        raise ValueError("the exact thirteen distinct files are required")
    for row in rows:
        path = Path(row["remote_path"])
        if (set(row) != {"local_path", "remote_path", "sha256", "bytes"} or not path.is_relative_to(root)
                or any(part in (".", "..") for part in PurePosixPath(row["remote_path"]).parts)
                or str(path) != row["remote_path"] or type(row["bytes"]) is not int or not 0 <= row["bytes"] <= 200 * 1024 * 1024):
            raise ValueError("a copied input is outside the exact bounded file layout")
    expected = [PYTHON, "-I", str(helper / PREFIX / "aces_slurm_site_validation.py"), "prepare",
        *[part for key, value in plan["native_prepare_parameters"].items() for part in ("--" + key, value)]]
    if plan["native_prepare_command"] != expected:
        raise ValueError("only the exact native metadata prepare argv is allowed")
    return plan


def validate_local(plan_path):
    """Reject foreign plans and changed bytes before starting any network process."""
    contents = ordinary(plan_path).read_bytes()
    plan = checked_plan(contents)
    for row in plan["exact_copy_allowlist"]:
        actual = file_record(row["local_path"])
        if actual != {"path": row["local_path"], "sha256": row["sha256"], "bytes": row["bytes"]}:
            raise ValueError("an exact local input changed: " + row["local_path"])
    return plan, contents


def make_parents(path):
    """Create only missing ordinary directories below the fixed private input root."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        return ordinary(path, directory=True)
    make_parents(path.parent)
    path.mkdir()
    return ordinary(path, directory=True)


def fsync_directory(path):
    """Make a new exclusive directory entry durable before reporting publication."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def receive_file(stream, target, expected):
    """Read exactly one file and publish without replacing existing different bytes."""
    target = Path(target)
    make_parents(target.parent)
    existing = target.exists() or target.is_symlink()
    if existing and file_record(target) != {"path": str(target), "sha256": expected["sha256"], "bytes": expected["bytes"]}:
        raise ValueError("different bytes already occupy an immutable staging path: " + str(target))
    descriptor, temporary = tempfile.mkstemp(prefix=".received-", dir=target.parent)
    temporary = Path(temporary)
    try:
        hasher, remaining = hashlib.sha256(), expected["bytes"]
        with os.fdopen(descriptor, "wb") as output:
            while remaining:
                chunk = stream.read(min(BLOCK, remaining))
                if not chunk:
                    raise ValueError("the input stream ended before one complete staging file")
                output.write(chunk)
                hasher.update(chunk)
                remaining -= len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if hasher.hexdigest() != expected["sha256"]:
            raise ValueError("streamed staging bytes differ from their reviewed SHA256")
        # An exclusive link prevents replacing a file created after the initial check.
        try:
            os.link(temporary, target)
        except FileExistsError:
            if file_record(target) != {"path": str(target), "sha256": expected["sha256"], "bytes": expected["bytes"]}:
                raise ValueError("conflicting bytes appeared during exclusive staging publication")
        fsync_directory(target.parent)
    finally:
        temporary.unlink()
    return file_record(target)


def write_record(path, value):
    """Publish a new metadata record exclusively and preserve any previous bytes."""
    contents = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    import io
    return receive_file(io.BytesIO(contents), path, {"sha256": digest(contents), "bytes": len(contents)})


def execute(command, commands, *, environment=None, timeout=120):
    """Capture each exact Git or single native-prepare command without a shell or retry."""
    if command[0] != "git" and command[:4] != [PYTHON, "-I", str(BASE / "git/checkouts/study_helpers_8e43cb1dc963" / PREFIX / "aces_slurm_site_validation.py"), "prepare"]:
        raise ValueError("this staging driver cannot execute a different program or helper action")
    result = subprocess.run(command, env=environment, capture_output=True, timeout=timeout)
    row = {"argv": command, "recorded_at_utc": utc_now(), "returncode": result.returncode,
        "stdout": result.stdout.decode(), "stderr": result.stderr.decode()}
    commands.append(row)
    if result.returncode != 0:
        raise ValueError("a native metadata command failed: " + shlex.join(command))
    if len(result.stdout) + len(result.stderr) > MAX_HEADER:
        raise ValueError("a native metadata command exceeded its bounded output")
    return result.stdout.decode().strip()


def git_identity(root, commands, expected=None):
    """Record an exact existing private Git root without altering its files or refs."""
    root = ordinary(root, directory=True)
    base = ["git", "-C", str(root)]
    top = execute([*base, "rev-parse", "--show-toplevel"], commands)
    head = execute([*base, "rev-parse", "HEAD"], commands)
    branch = execute([*base, "rev-parse", "--abbrev-ref", "HEAD"], commands)
    status = execute([*base, "status", "--porcelain", "--untracked-files=all"], commands)
    if top != str(root) or branch != "HEAD" or status or (expected is not None and head != expected):
        raise ValueError("an existing private Git checkout is not its exact clean detached source")
    return {"root": str(root), "commit": head, "tree": execute([*base, "rev-parse", "HEAD^{tree}"], commands), "status": status}


def old_roots(plan, commands):
    """Inspect all existing checkouts only within this user's dedicated study directory."""
    parent = ordinary(BASE / "git/checkouts", directory=True)
    roots = {str(path): git_identity(path, commands) for path in sorted(parent.iterdir())}
    required = {str(parent / "numerical_source_d28e6735e797"): SOURCE_COMMIT,
        str(parent / "jaxtari_source_04ab9aec318c"): "04ab9aec318c72162e2e53503a594a489d7e162b",
        plan["native_helper_root"]: HELPER_COMMIT}
    if any(path not in roots or roots[path]["commit"] != commit for path, commit in required.items()):
        raise ValueError("the established numerical/JAXtari and separately staged helper roots are required")
    return roots


def helper_module(plan):
    """Load only the exact published stdlib helper after checking every helper file hash."""
    identity = plan["expected_native_helper_identity"]
    for relative, expected in identity["files"].items():
        if file_record(Path(identity["source_root"]) / relative)["sha256"] != expected:
            raise ValueError("a published helper file differs from the reviewed source closure")
    path = Path(identity["source_root"]) / PREFIX / "aces_slurm_site_validation.py"
    specification = importlib.util.spec_from_file_location("staged_actual_aces_site", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.common().verify_source_identity(identity)
    return module


def checkout_commands(plan):
    """Describe only fresh numerical and dedicated CleanRL Git checkout construction."""
    source = plan["native_numerical_checkout_required"]
    cleanrl = plan["native_cleanrl_checkout_required"]
    bare = str(Path(plan["remote_inputs_directory"]) / "cleanrl_repository.git")
    return [["git", "clone", "--no-checkout", source["git_repository"], source["path"]],
        ["git", "-C", source["path"], "sparse-checkout", "init", "--cone"],
        ["git", "-C", source["path"], "sparse-checkout", "set", *source["sparse_cones"]],
        ["git", "-C", source["path"], "checkout", "--detach", SOURCE_COMMIT],
        ["git", "bundle", "list-heads", cleanrl["bundle"]],
        ["git", "clone", "--bare", cleanrl["bundle"], bare],
        ["git", "clone", "--no-checkout", bare, cleanrl["path"]],
        ["git", "-C", cleanrl["path"], "sparse-checkout", "init", "--cone"],
        ["git", "-C", cleanrl["path"], "sparse-checkout", "set", "cleanrl"],
        ["git", "-C", cleanrl["path"], "checkout", "--detach", CLEANRL_COMMIT]]


def mapping_check(helper, plan):
    """Rebuild all seventy-two exact mappings and the original admission before or after prepare."""
    parameters = plan["native_prepare_parameters"]
    binding = {"schema_version": 1, "kind": "aces_original_source_portable_admission",
        "campaign_run_dir": parameters["admission-run-dir"], "evidence_manifest": {
            "path": parameters["evidence-manifest"], "sha256": parameters["evidence-manifest-sha256"]}}
    adapter = helper.load("aces_source_admission")
    proof = adapter.evidence(binding)
    if len(proof.resolver.files) != 72 or len(proof.resolver.git_roots) != 1:
        raise ValueError("the complete reviewed seventy-two-file source/admission mapping is required")
    admission, source, configuration = adapter.snapshot(binding)
    if source["source_root"] != plan["native_numerical_checkout_required"]["path"]:
        raise ValueError("the manifest mapped source differs from the fresh dedicated checkout")
    return {"binding": binding, "admission": admission, "numerical_source": source, "configuration": configuration}


def native_environment(plan):
    """Set actual metadata preparation controls without importing numerical libraries."""
    environment = dict(os.environ)
    if any(value and (name.startswith("PYTEST_") or name == "PYTHONOPTIMIZE") for name, value in environment.items()):
        raise ValueError("native metadata preparation cannot inherit pytest selectors or Python optimization")
    environment.update(PYTHONNOUSERSITE="1", TZ="UTC", XDG_DATA_HOME=str(BASE / "assets"))
    reference = str(Path(plan["native_cleanrl_checkout_required"]["path"]) / "cleanrl/ppo_rnd_envpool.py")
    if environment.get("CLEANRL_REFERENCE_FILE") not in (None, "", reference):
        raise ValueError("native metadata preparation inherited a different CleanRL file")
    environment["CLEANRL_REFERENCE_FILE"] = reference
    return environment


def prepared_files(plan):
    """List only the exact packet records and four corresponding data-standard copies."""
    rows = [("packet/" + name, Path(plan["remote_packet_directory"]) / name) for name in plan["prepared_packet_standard_files_expected"]]
    rows += [("data/" + name, Path(plan["remote_data_unit"]) / name) for name in
        ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt")]
    return rows


def remote_receive():
    """Receive fixed inputs, verify fresh checkouts and prepare metadata on actual ACES only."""
    if (os.getuid() != 56626 or pwd.getpwuid(os.getuid()).pw_name != "u.sl359075"
            or socket.gethostname().split(".")[0] != "alogin1" or str(Path(sys.executable).absolute()) != PYTHON):
        raise ValueError("native staging requires the actual ACES owner, login host and project Python")
    raw_header = sys.stdin.buffer.readline(MAX_HEADER + 1)
    if len(raw_header) > MAX_HEADER or not raw_header.endswith(b"\n"):
        raise ValueError("the staging header is absent or exceeds its bound")
    header = json.loads(raw_header)
    if set(header) != {"plan", "driver_sha256"}:
        raise ValueError("the exact staging header is required")
    # Record the actual -c code supplied to this own process, not a claimed source hash.
    command_line = Path("/proc/self/cmdline").read_bytes().split(b"\0")
    code = command_line[command_line.index(b"-c") + 1]
    if digest(code) != header["driver_sha256"]:
        raise ValueError("the actual remote staging code differs from its recorded source")
    plan = checked_plan(header["plan"].encode())
    commands, copied = REMOTE_COMMANDS, []
    root = Path(plan["remote_inputs_directory"])
    targets = [Path(plan["native_numerical_checkout_required"]["path"]), Path(plan["native_cleanrl_checkout_required"]["path"]),
        root / "cleanrl_repository.git", Path(plan["remote_packet_directory"]), Path(plan["remote_data_unit"])]
    if any(path.exists() or path.is_symlink() for path in targets):
        raise ValueError("native source, CleanRL bare/root and packet/data targets must all be fresh")
    ordinary(BASE / "git/repository.git", directory=True)
    before = old_roots(plan, commands)
    helper = helper_module(plan)
    make_parents(root)
    write_record(root / "old_roots_before.json", before)
    for row in plan["exact_copy_allowlist"]:
        copied.append(receive_file(sys.stdin.buffer, row["remote_path"], row))
    if sys.stdin.buffer.read(1):
        raise ValueError("the input stream contains bytes beyond the thirteen fixed files")
    for command in checkout_commands(plan):
        output = execute(command, commands)
        if command[1:3] == ["bundle", "list-heads"] and output != CLEANRL_COMMIT + " HEAD":
            raise ValueError("the CleanRL full bundle has a different exact advertised commit/ref")
    git_identity(plan["native_numerical_checkout_required"]["path"], commands, SOURCE_COMMIT)
    git_identity(plan["native_cleanrl_checkout_required"]["path"], commands, CLEANRL_COMMIT)
    mapped_before = mapping_check(helper, plan)
    environment = native_environment(plan)
    execute(plan["native_prepare_command"], commands, environment=environment, timeout=240)
    # The parent observer uses the same explicit metadata environment as the actual preparer.
    os.environ.clear()
    os.environ.update(environment)
    directory, unit, actual = helper.prepared(Path(plan["remote_packet_directory"]))
    helper.checked_software(actual)
    mapped_after = mapping_check(helper, plan)
    after = old_roots(plan, commands)
    if before != after or mapped_before != mapped_after:
        raise ValueError("old roots or original mapped admission changed during metadata preparation")
    if any(name in sys.modules for name in NUMERICAL_MODULES):
        raise ValueError("native metadata staging imported a numerical module")
    write_record(root / "old_roots_after.json", after)
    records = prepared_files(plan) + [("staging/old_roots_before.json", root / "old_roots_before.json"),
        ("staging/old_roots_after.json", root / "old_roots_after.json")]
    files = [{"relative_path": relative, **file_record(path)} for relative, path in records]
    if any(row["bytes"] > MAX_RETURN_FILE for row in files):
        raise ValueError("a native prepared artifact exceeds the bounded returned file size")
    receipt = {"schema_version": 1, "kind": "aces_actual_native_metadata_preparation", "passed": True,
        "recorded_at_utc": utc_now(), "plan_sha256": PLAN_SHA256, "driver_sha256": header["driver_sha256"],
        "hostname": socket.gethostname(), "uid": os.getuid(), "python": str(Path(sys.executable).absolute()),
        "copied": copied, "old_roots_before": before, "old_roots_after": after, "mapping": mapped_after,
        "commands": commands, "prepared_files": list(files), "numerical_imports": [], "jobs_submitted": 0,
        "workers_permitted": False, "case_credit": 0, "idea_credit": 0, "admission_credit": 0}
    receipt_path = root / "native_preparation_receipt.json"
    write_record(receipt_path, receipt)
    files.append({"relative_path": "staging/native_preparation_receipt.json", **file_record(receipt_path)})
    sys.stdout.buffer.write((json.dumps({"receipt": receipt, "files": files}, separators=(",", ":")) + "\n").encode())
    for row in files:
        send_file(sys.stdout.buffer, row["path"], row)
    sys.stdout.buffer.flush()


def ssh_command(source):
    """Reuse only an existing master and make a missing-master connection fail before authentication."""
    remote = "exec " + shlex.join([PYTHON, "-I", "-c", source, "--remote-receive"])
    return [*SSH_OPTIONS, remote]


def send_file(stream, path, expected):
    """Stream only the recorded file length and verify both EOF and the incremental SHA256."""
    remaining, hasher = expected["bytes"], hashlib.sha256()
    with ordinary(path).open("rb") as source:
        while remaining:
            chunk = source.read(min(BLOCK, remaining))
            if not chunk:
                raise ValueError("a streamed file became shorter than its recorded length")
            stream.write(chunk)
            hasher.update(chunk)
            remaining -= len(chunk)
        if source.read(1) or hasher.hexdigest() != expected["sha256"]:
            raise ValueError("a streamed file changed its recorded length or SHA256")


def send_inputs(stream, plan, contents, source_hash, errors):
    """Stream each already verified local file and reject changes while sending."""
    try:
        stream.write((json.dumps({"plan": contents.decode(), "driver_sha256": source_hash}) + "\n").encode())
        for row in plan["exact_copy_allowlist"]:
            send_file(stream, row["local_path"], row)
        stream.flush()
    except Exception as error:
        errors.append(type(error).__name__ + ": " + str(error))
    finally:
        stream.close()


def accept_response(path, directory, plan, source_hash):
    """Save complete raw remote files only beneath new bounded local mirror names."""
    allowed = {relative: str(remote) for relative, remote in prepared_files(plan)}
    root = Path(plan["remote_inputs_directory"])
    allowed.update({"staging/" + name: str(root / name) for name in
        ("old_roots_before.json", "old_roots_after.json", "native_preparation_receipt.json")})
    with ordinary(path).open("rb") as stream:
        line = stream.readline(MAX_HEADER + 1)
        if len(line) > MAX_HEADER or not line.endswith(b"\n"):
            raise ValueError("the returned native metadata header is absent or exceeds its bound")
        header = json.loads(line)
        if set(header) != {"receipt", "files"}:
            raise ValueError("the exact returned receipt and artifact roster are required")
        receipt = header["receipt"]
        if (receipt.get("kind") != "aces_actual_native_metadata_preparation" or receipt.get("passed") is not True
                or receipt.get("plan_sha256") != PLAN_SHA256 or receipt.get("driver_sha256") != source_hash
                or receipt.get("uid") != 56626 or receipt.get("python") != PYTHON
                or receipt.get("old_roots_before") != receipt.get("old_roots_after")
                or receipt.get("numerical_imports") != [] or receipt.get("jobs_submitted") != 0
                or receipt.get("workers_permitted") is not False
                or type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1
                or any(type(receipt.get(name)) is not int or receipt[name] != 0 for name in
                    ("jobs_submitted", "case_credit", "idea_credit", "admission_credit"))):
            raise ValueError("returned metadata does not prove this exact zero-job native preparation")
        rows = header["files"]
        if len(rows) != len(allowed) or {row["relative_path"] for row in rows} != set(allowed):
            raise ValueError("the returned artifact roster is incomplete or duplicated")
        for row in rows:
            if (set(row) != {"relative_path", "path", "sha256", "bytes"} or allowed[row["relative_path"]] != row["path"]
                    or type(row["bytes"]) is not int or not 0 <= row["bytes"] <= MAX_RETURN_FILE):
                raise ValueError("a returned artifact changed its exact path or size")
            receive_file(stream, directory / "received" / row["relative_path"], row)
        if stream.read(1):
            raise ValueError("the remote response contains trailing unregistered bytes")
    saved = json.loads((directory / "received/staging/native_preparation_receipt.json").read_bytes())
    if saved != receipt:
        raise ValueError("the returned native receipt bytes differ from the actual header record")
    return receipt


def local_execute(plan, contents, receipt_directory):
    """Run one bounded existing-master SSH transfer and preserve exact raw completion or failure evidence."""
    if socket.gethostname().split(".")[0] != "portal05":
        raise ValueError("run the native staging driver only on portal05")
    directory = Path(receipt_directory).absolute()
    ordinary(directory.parent, directory=True)
    directory.mkdir()
    source = ordinary(Path(__file__).absolute()).read_text()
    source_hash = digest(source.encode())
    command = ssh_command(source)
    write_record(directory / "invocation.json", {"recorded_at_utc": utc_now(), "plan_sha256": PLAN_SHA256,
        "driver_sha256": source_hash, "ssh_options": SSH_OPTIONS, "remote_code_bytes": len(source.encode()),
        "remote_code_sha256": source_hash, "timeout_seconds": SSH_TIMEOUT, "master_separately_checked": False,
        "authentication_retry_permitted": False, "submission_permitted": False})
    (directory / "executed_driver.py").write_text(source)
    errors = []
    with (directory / "stdout.raw").open("xb") as output, (directory / "stderr.raw").open("xb") as error:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output, stderr=error)
        writer = threading.Thread(target=send_inputs, args=(process.stdin, plan, contents, source_hash, errors), daemon=True)
        writer.start()
        try:
            returncode = process.wait(timeout=SSH_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            errors.append("SSH exceeded the fixed staging timeout; no automatic retry is permitted")
            returncode = process.returncode
        writer.join(timeout=10)
    if writer.is_alive():
        errors.append("the input sender did not finish after the SSH process ended")
    transport = {"recorded_at_utc": utc_now(), "returncode": returncode, "send_errors": errors,
        "stdout": file_record(directory / "stdout.raw"), "stderr": file_record(directory / "stderr.raw"),
        "remote_state_after_failure_may_require_read_only_inspection": bool(returncode or errors)}
    write_record(directory / "transport_result.json", transport)
    if returncode or errors:
        raise ValueError("native staging did not complete; exact raw transport evidence is preserved")
    return accept_response(directory / "stdout.raw", directory, plan, source_hash)


def metadata_timeout(signum, frame):
    """Stop this own metadata process after the bounded remote staging deadline."""
    raise TimeoutError("native metadata staging exceeded its fixed 540-second deadline")


def main():
    """Describe reviewed operations or execute one explicit metadata-only staging attempt."""
    if sys.argv[1:] == ["--remote-receive"]:
        signal.signal(signal.SIGALRM, metadata_timeout)
        signal.alarm(540)
        try:
            remote_receive()
        except Exception as error:
            # Preserve actual failed command records; do not retry or infer that partial targets are usable.
            failure = {"schema_version": 1, "kind": "aces_native_metadata_staging_failure",
                "recorded_at_utc": utc_now(), "error_type": type(error).__name__, "error": str(error),
                "commands": REMOTE_COMMANDS, "jobs_submitted": 0, "automatic_retry_permitted": False}
            sys.stdout.buffer.write((json.dumps(failure) + "\n").encode())
            sys.stdout.buffer.flush()
            raise
        finally:
            signal.alarm(0)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--receipt-directory", type=Path)
    arguments = parser.parse_args()
    plan, contents = validate_local(arguments.plan)
    if arguments.execute:
        if arguments.receipt_directory is None:
            raise ValueError("execution needs a fresh local receipt directory")
        result = local_execute(plan, contents, arguments.receipt_directory)
        print(json.dumps({"passed": result["passed"], "receipt_directory": str(arguments.receipt_directory),
            "remote_packet_directory": plan["remote_packet_directory"], "jobs_submitted": 0}, indent=2))
    else:
        print(json.dumps({"plan_sha256": PLAN_SHA256, "copy_files": 13, "ssh_options": SSH_OPTIONS,
            "master_separately_checked": False, "planned_checkout_commands": checkout_commands(plan),
            "native_prepare_command": plan["native_prepare_command"], "network_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
