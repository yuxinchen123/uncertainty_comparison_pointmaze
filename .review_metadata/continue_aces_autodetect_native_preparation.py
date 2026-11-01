"""Prepare one fresh ACES packet from reviewed incremental source without submitting work.

The default command only checks local metadata. Root may use --execute after
review and publication. No installation, numerical import, job action or retry
is implemented. Existing and incomplete destinations are always retained.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import resource
import re
import shlex
import socket
import subprocess
import sys
import threading
import types


PLAN_SHA = "9c1700a4e1fe7f526563d0f59243afec9f41b7dda16be95975e98ca6b8ff8fdc"
TIP = "bad2ee3736a1451094d5d1644b36b8cfc2d91c38"
BASIS = "8e43cb1dc963045fc28f4e611411097efd93185a"
ROOT = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl")
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
SSH = ["ssh", "-T", "-o", "ControlMaster=no", "-o", "ProxyCommand=false", "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=5", "aces"]
FORBIDDEN = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "jaxtari", "exploration_platform"}
MAGIC = b"RND_ACES_NATIVE_PREPARATION_V1\n"
PACKET_FILES = {"site_validation_plan.json", "software_at_prepare.json", "job.sbatch", "command.txt",
                "config_resolved.yaml", "experiment_background.md", "manifest.yaml"}
STANDARD_FILES = {"command.txt", "config_resolved.yaml", "experiment_background.md", "manifest.yaml"}
DOCUMENTS = {"continuation", "commands", "reused", "verification", "environment_plan", "metadata_helper",
             "base_snapshot", "private_snapshot", "installed_files", "preserved", "native_setup_receipt", "software", "old_plan"}


def require(condition, message):
    """Reject missing, changed or ambiguous evidence before continuing."""
    if not condition:
        raise ValueError(message)


def sha(path):
    """Hash exact file bytes incrementally without reading a wheel or library into memory."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def ordinary(path, *, directory=False):
    """Require an existing exact physical path with no symlink component."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path
        and (path.is_dir() if directory else path.is_file()), "exact ordinary path required: " + str(path))
    return path


def sync_directory(path):
    """Make new file and directory names durable before recording later actions."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_bytes(path, content):
    """Write a new durable file exclusively, preserving any conflicting or partial earlier file."""
    path = Path(path)
    ordinary(path.parent, directory=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def save(path, value):
    """Retain one literal JSON record without replacing an earlier observation."""
    save_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def receipt(path):
    """Bind the original path, SHA256 and length of an ordinary file."""
    path = ordinary(path)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def no_numerical_modules():
    """Reject a numerical package already loaded in this metadata-only process."""
    require(not any(name.split(".")[0] in FORBIDDEN for name in sys.modules), "numerical packages are forbidden")


def checked_payload(payload):
    """Verify the complete thirteen-document reviewed input graph before any transport or mutation."""
    raw = payload["plan"].encode()
    require(hashlib.sha256(raw).hexdigest() == PLAN_SHA, "the exact reviewed driver plan changed")
    plan = json.loads(raw)
    require(type(plan.get("schema_version")) is int and plan["schema_version"] == 1
        and plan.get("kind") == "aces_autodetect_incremental_native_preparation_driver_plan"
        and plan["source_commit"] == TIP and plan["basis_commit"] == BASIS
        and set(plan["documents"]) == DOCUMENTS and set(payload["documents"]) == DOCUMENTS, "unsupported driver input graph")
    for name, expected in plan["documents"].items():
        content = payload["documents"][name].encode()
        require(len(content) == expected["bytes"] and hashlib.sha256(content).hexdigest() == expected["sha256"],
                "reviewed input changed: " + name)
    data = {name: json.loads(value) for name, value in payload["documents"].items()
            if name not in {"verification", "metadata_helper"}}
    continuation, commands, reused = (data[name] for name in ("continuation", "commands", "reused"))
    require(continuation["source_commit"] == TIP and continuation["basis_commit"] == BASIS
        and continuation["expected_helper_count"] == len(continuation["expected_helper_closure"]) == 18
        and commands["ssh_prefix"] == SSH and commands["native_post_prepare_verification_argv"][:4]
            == [reused["private_python"], "-I", "-B", "-c"]
        and commands["native_post_prepare_verification_argv"][4] == payload["documents"]["verification"]
        and commands["native_prepare_argv"][:5] == [reused["private_python"], "-I", "-B",
            continuation["new_helper_checkout_root"] + "/" + PREFIX + "aces_slurm_site_validation.py", "prepare"]
        and data["native_setup_receipt"]["passed"] is True and reused["private_package_count"] == 7
        and reused["original_file_mapping_count"] == 72 and reused["expected_asset_file_count"] == 4531,
        "reviewed continuation, interpreter, metadata command or original proof differs")
    for name in ("workers_permitted", "campaign_claims_permitted", "physical_gres_mapping_verified", "kernel_device_enforcement_verified"):
        require(continuation.get(name) is False, "preparation cannot grant a site or worker claim")
    require(plan["git_packing_threads"] == 2 and plan["git_address_space_limit_bytes"] == 2 * 2 ** 30
        and data["environment_plan"]["private_python"] == reused["private_python"]
        and data["private_snapshot"]["python"] == reused["private_python"]
        and len(data["installed_files"]) == 7, "the preserved environment or bounded Git shape changed")
    no_numerical_modules()
    return plan, data


def local_payload(path):
    """Read exact existing local inputs without making network calls or creating execution outputs."""
    path = ordinary(path)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PLAN_SHA, "driver plan bytes changed before transport")
    plan = json.loads(raw)
    documents = {}
    for name, expected in plan["documents"].items():
        source = ordinary(expected["path"])
        require(receipt(source) == expected, "local reviewed input changed: " + name)
        documents[name] = source.read_text()
    payload = {"plan": raw.decode(), "documents": documents}
    checked_payload(payload)
    return payload


def git_limit():
    """Bound every Git process to two GiB of virtual address space."""
    resource.setrlimit(resource.RLIMIT_AS, (2 * 2 ** 30, 2 * 2 ** 30))


def invoke(command, directory, *, environment=None, git=False, accepted=(0,), timeout=180):
    """Save each exact bounded command and its raw returned bytes before interpreting its status."""
    directory = Path(directory)
    index = len(list(directory.glob("command-*.intent.json"))) + 1
    stem = "command-" + str(index).zfill(4)
    started = datetime.now(timezone.utc).isoformat()
    save(directory / (stem + ".intent.json"), {"argv": command, "started_at_utc": started,
        "timeout_seconds": timeout, "address_space_limit_bytes": 2 * 2 ** 30 if git else None})
    failure = None
    try:
        result = subprocess.run(command, capture_output=True, env=environment, timeout=timeout,
                                preexec_fn=git_limit if git else None)
    except (subprocess.TimeoutExpired, OSError) as error:
        failure = error
        result = types.SimpleNamespace(returncode=None, stdout=getattr(error, "stdout", None) or b"",
                                       stderr=getattr(error, "stderr", None) or b"")
    save_bytes(directory / (stem + ".stdout.raw"), result.stdout)
    save_bytes(directory / (stem + ".stderr.raw"), result.stderr)
    save(directory / (stem + ".json"), {"argv": command, "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "returncode": result.returncode,
        "execution_error": None if failure is None else repr(failure),
        "stdout": receipt(directory / (stem + ".stdout.raw")), "stderr": receipt(directory / (stem + ".stderr.raw"))})
    if failure is not None:
        raise failure
    require(result.returncode in accepted, "bounded metadata command failed: " + repr(command[:5]))
    return result


def bundle_header(path, tip, reference, basis):
    """Verify the one advertised reference and every explicit incremental prerequisite."""
    header, prerequisites, references = [], [], []
    with ordinary(path).open("rb") as stream:
        first = stream.readline(128)
        require(first in (b"# v2 git bundle\n", b"# v3 git bundle\n"), "unsupported incremental bundle header")
        while True:
            line = stream.readline(8193)
            require(line and len(line) <= 8192 and sum(map(len, header)) < 65536, "incomplete or oversized bundle header")
            if line == b"\n":
                break
            header.append(line)
            text = line.decode().rstrip("\n")
            if text.startswith("@"):
                require(text == "@object-format=sha1", "unsupported bundle capability")
            elif text.startswith("-"):
                commit = text[1:].split(" ", 1)[0]
                require(len(commit) == 40 and all(value in "0123456789abcdef" for value in commit), "invalid prerequisite")
                prerequisites.append(commit)
            else:
                references.append(text)
    require(references == [tip + " " + reference] and basis in prerequisites
        and len(prerequisites) == len(set(prerequisites)), "incremental bundle reference/prerequisites differ")
    return prerequisites


def publish_bundle(temporary, target):
    """Publish only new complete bytes without overwriting a conflict, then retire the owned temporary name."""
    temporary, target = ordinary(temporary), Path(target)
    require(temporary.parent == target.parent and temporary.stat().st_nlink == 1, "bundle publication paths are ambiguous")
    os.link(temporary, target)
    sync_directory(target.parent)
    temporary.unlink()
    sync_directory(target.parent)


def build_bundle(root, audit, continuation):
    """Create and verify only objects after the exact existing basis, retaining failures and exact raw commands."""
    tip, basis = continuation["source_commit"], continuation["basis_commit"]
    reference = "refs/transfer/aces_helpers_" + tip
    target = Path(continuation["new_local_bundle_planned_path"])
    temporary = target.with_name(target.name + ".incomplete")
    require(not target.exists() and not temporary.exists() and not target.is_symlink() and not temporary.is_symlink(),
            "a complete or partial local bundle already exists; preserve it")
    run = lambda command, **options: invoke(command, audit, git=True, **options)
    run(["git", "-C", str(root), "merge-base", "--is-ancestor", basis, tip])
    origin = run(["git", "-C", str(root), "rev-parse", "refs/remotes/origin/merge-atari-cleanrl"]).stdout.decode().strip()
    run(["git", "-C", str(root), "merge-base", "--is-ancestor", tip, origin])
    require(run(["git", "-C", str(root), "rev-parse", tip + "^{tree}"]).stdout.decode().strip()
        == continuation["source_tree"], "published helper tree differs")
    created = False
    try:
        run(["git", "-C", str(root), "update-ref", reference, tip, "0" * 40])
        created = True
        run(["git", "-C", str(root), "-c", "pack.threads=2", "-c", "pack.windowMemory=64m", "-c",
             "pack.deltaCacheSize=32m", "-c", "pack.window=5", "-c", "pack.depth=20", "bundle", "create",
             str(temporary), reference, "^" + basis], timeout=300)
        prerequisites = bundle_header(temporary, tip, reference, basis)
        for commit in prerequisites:
            run(["git", "-C", str(root), "merge-base", "--is-ancestor", commit, basis])
        run(["git", "-C", str(root), "bundle", "verify", str(temporary)])
        require(run(["git", "bundle", "list-heads", str(temporary)]).stdout.decode().strip()
            == tip + " " + reference, "bundle advertised an unexpected source")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        publish_bundle(temporary, target)
    finally:
        if created:
            run(["git", "-C", str(root), "update-ref", "-d", reference, tip])
    result = {**receipt(target), "prerequisites": prerequisites, "reference": reference,
        "source_commit": tip, "basis_commit": basis, "source_tree": continuation["source_tree"],
        "packing_threads": 2, "address_space_limit_bytes": 2 * 2 ** 30, "temporary_reference_removed": True}
    save(Path(audit) / "local_bundle.json", result)
    return result


def expected_roots(preserved):
    """Flatten the recorded old and dedicated native roots without discarding a conflicting duplicate."""
    result = {}
    for group in preserved["roots"].values():
        for name, value in group.items():
            path = value["root"]
            require(not Path(name).is_absolute() or name == path, "recorded Git root key differs from its literal root")
            require(path not in result or result[path] == value, "preserved Git root has conflicting identities")
            result[path] = value
    return result


def check_roots(expected, audit):
    """Check exact old checkout tops, detached commits, trees and clean status without changing them."""
    result = {}
    for path, value in expected.items():
        root = ordinary(path, directory=True)
        query = lambda *arguments, **options: invoke(["git", "-C", str(root), *arguments], audit, git=True, **options)
        require(query("rev-parse", "--show-toplevel").stdout.decode().strip() == path
            and query("rev-parse", "HEAD").stdout.decode().strip() == value["commit"]
            and query("rev-parse", "HEAD^{tree}").stdout.decode().strip() == value["tree"]
            and query("symbolic-ref", "--quiet", "HEAD", accepted=(0, 1)).returncode == 1
            and query("status", "--porcelain", "--untracked-files=all").stdout == b"", "an old detached source root changed: " + path)
        result[path] = value
    return result


def check_preserved_files(data):
    """Rehash every preserved input, mapping, private installed file, retained wheel and physical asset."""
    reused, previous = data["reused"], data["preserved"]
    for expected in reused["preserved_original_inputs"]:
        require(receipt(expected["path"]) == expected, "an original thirteen-file input changed")
    numerical = reused["numerical_source"]
    require(len(numerical["files"]) == 32, "the complete original numerical file roster changed")
    for relative, digest in numerical["files"].items():
        require(sha(ordinary(Path(numerical["source_root"]) / relative)) == digest, "an original numerical source file changed")
    reference = reused["cleanrl_reference"]["file"]
    require(sha(ordinary(reference["path"])) == reference["sha256"], "the pinned original CleanRL reference changed")
    manifest_record = reused["portable_admission"]["evidence_manifest"]
    manifest_path = ordinary(manifest_record["path"])
    require(sha(manifest_path) == manifest_record["sha256"], "the exact portable mapping changed")
    mapping = json.loads(manifest_path.read_bytes())
    require(len(mapping["files"]) == 72, "the original mapping must retain exactly72 files")
    for row in mapping["files"]:
        relative = PurePosixPath(row["relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == row["relative_path"], "mapped file escaped its input root")
        path = ordinary(manifest_path.parent / relative)
        require(sha(path) == row["sha256"], "a mapped original byte copy changed")
    installed_count = 0
    for distribution in data["installed_files"]:
        for path, digest in distribution["verified_files"].items():
            require(Path(path).is_relative_to(reused["private_prefix"]) and sha(ordinary(path)) == digest, "an installed private wheel file changed")
            installed_count += 1
    wheels = []
    for row in data["environment_plan"]["wheels"]:
        path = Path(data["environment_plan"]["setup_directory"]) / "wheels" / Path(row["path"]).name
        actual = receipt(path)
        require((actual["sha256"], actual["bytes"]) == (row["sha256"], row["bytes"]), "a retained locked wheel changed")
        wheels.append(actual)
    record = reused["asset_record"]
    require(json.loads(ordinary(reused["asset_record_file"]).read_bytes()) == record
        and sha(ordinary(record["manifest"])) == record["sha256"], "the operational exact asset binding changed")
    files = json.loads(Path(record["manifest"]).read_bytes())["files"]
    root, actual_files = ordinary(record["root"], directory=True), {}
    for directory, names, filenames in os.walk(root, followlinks=False):
        require(all(not (Path(directory) / name).is_symlink() for name in names), "asset directory symlink is forbidden")
        for name in filenames:
            path = ordinary(Path(directory) / name)
            require(path.stat().st_nlink == 1, "the physical asset view cannot contain hard links")
            actual_files[str(path.relative_to(root))] = sha(path)
    require(len(actual_files) == 4531 and actual_files == files, "the exact4531-file physical asset view changed")
    private_python = ordinary(reused["private_python"])
    old_python = Path(data["environment_plan"]["old_python"])
    ordinary(old_python.parent, directory=True)
    old_target = ordinary(old_python.resolve(strict=True))
    require(old_target.is_relative_to(data["environment_plan"]["base_prefix"])
        and sha(private_python) == sha(old_target), "the copied private interpreter differs from its original base interpreter")
    interpreter_files = [receipt(private_python), receipt(Path(reused["private_prefix"]) / "pyvenv.cfg"),
        {"requested_path": str(old_python), "link_target": os.readlink(old_python) if old_python.is_symlink() else None,
         "resolved_target": receipt(old_target)}]
    return {"mapping_sha256": manifest_record["sha256"], "mapping_files": 72, "installed_locked_files": installed_count,
        "wheels": wheels, "asset_files": len(actual_files), "asset_manifest_sha256": record["sha256"],
        "inputs": previous["inputs"], "interpreter_files": interpreter_files}


def metadata_probe(payload, data, audit, python, shared):
    """Observe providers in the explicitly selected old or private interpreter without importing their packages."""
    source = payload["documents"]["metadata_helper"]
    code = "import json,types\nm=types.ModuleType('pinned_metadata')\nexec(" + repr(source) + ",m.__dict__)\nprint(json.dumps(m.snapshot(hash_shared=" + repr(shared) + "),sort_keys=True))"
    observed = invoke([python, "-I", "-B", "-c", code], audit, timeout=300)
    result = json.loads(observed.stdout)
    expected = data["base_snapshot"] if shared else data["private_snapshot"]
    require(result == expected, "actual preserved base/private interpreter, package metadata or shared bytes changed")
    return result


def native_environment(data):
    """Preserve the reviewed numerical declarations and exact asset root without any installation settings."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
    environment.update(data["commands"]["native_prepare_environment"])
    environment["XDG_DATA_HOME"] = data["reused"]["asset_record"]["root"]
    return environment


def preserve(payload, data, audit, phase):
    """Check the complete previously verified environment and inputs before and after fresh metadata preparation."""
    roots = check_roots(expected_roots(data["preserved"]), audit)
    files = check_preserved_files(data)
    base = metadata_probe(payload, data, audit, data["environment_plan"]["old_python"], True)
    private = metadata_probe(payload, data, audit, data["reused"]["private_python"], False)
    module = types.ModuleType("checked_dependency_metadata")
    exec(payload["documents"]["metadata_helper"], module.__dict__)
    closure = module.dependency_check(list(private["effective"].values()), private["environment"])
    require(closure["active_dependency_count"] == 128, "the exact complete installed dependency closure changed")
    result = {"roots": roots, "files": files, "base_metadata_and_shared_files_unchanged": True,
        "private_metadata_unchanged": True, "base_package_count": len(base["effective"]),
        "private_package_count": len(private["effective"]) - len(base["effective"]), "active_dependency_count": 128}
    save(Path(audit) / ("preserved_" + phase + ".json"), result)
    return result


def receive_bundle(stream, target, expected):
    """Stream only the exact new bundle to an exclusive partial path, retaining every failed transfer."""
    target = Path(target)
    temporary = target.with_name(target.name + ".incomplete")
    require(not target.exists() and not target.is_symlink(), "the new remote bundle target already exists")
    ordinary(target.parent, directory=True)
    value, count = hashlib.sha256(), 0
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        remaining = expected["bytes"]
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            require(block, "incomplete bundle stream; retain the partial file")
            output.write(block)
            value.update(block)
            count += len(block)
            remaining -= len(block)
        output.flush()
        os.fsync(output.fileno())
    require(not stream.read(1) and count == expected["bytes"] and value.hexdigest() == expected["sha256"], "bundle stream length/hash differs; retain partial evidence")
    publish_bundle(temporary, target)


def stage_helper(data, bundle, audit):
    """Import verified incremental objects and create only the fresh detached sparse helper checkout."""
    continuation = data["continuation"]
    root = Path(continuation["new_helper_checkout_root"])
    repository = root.parent.parent / "repository.git"
    ordinary(repository, directory=True)
    run = lambda arguments, **options: invoke(arguments, audit, git=True, **options)
    require(run(["git", "--git-dir", str(repository), "rev-parse", "--is-bare-repository"]).stdout.strip() == b"true", "dedicated native bare repository required")
    prerequisites = bundle_header(bundle["path"], TIP, bundle["reference"], BASIS)
    require(prerequisites == bundle["prerequisites"], "transferred bundle prerequisites changed")
    for commit in prerequisites:
        run(["git", "--git-dir", str(repository), "cat-file", "-e", commit + "^{commit}"])
    run(["git", "--git-dir", str(repository), "bundle", "verify", bundle["path"]])
    reference = continuation["new_remote_reference"]
    require(run(["git", "--git-dir", str(repository), "show-ref", "--verify", "--quiet", reference], accepted=(0, 1)).returncode == 1,
            "the new source reference already exists; preserve it")
    # Import objects without updating another reference; the named new reference uses compare-and-swap.
    run(["git", "--git-dir", str(repository), "-c", "pack.threads=2", "bundle", "unbundle", bundle["path"]])
    run(["git", "--git-dir", str(repository), "update-ref", reference, TIP, "0" * 40])
    run(["git", "--git-dir", str(repository), "merge-base", "--is-ancestor", BASIS, TIP])
    run(["git", "--git-dir", str(repository), "worktree", "add", "--detach", "--no-checkout", str(root), TIP])
    run(["git", "-C", str(root), "sparse-checkout", "init", "--cone"])
    sparse = ["10_jax_exploration_platform/src", "10_jax_exploration_platform/configs", "10_jax_exploration_platform/scripts/atari_pixel_campaign"]
    run(["git", "-C", str(root), "sparse-checkout", "set", *sparse])
    run(["git", "-C", str(root), "checkout", "--detach", TIP])
    check_roots({str(root): {"root": str(root), "commit": TIP, "tree": continuation["source_tree"], "status": ""}}, audit)
    for relative, digest in continuation["expected_helper_closure"].items():
        path = ordinary(root / relative)
        require(sha(path) == digest, "one committed new helper changed")
        require(run(["git", "-C", str(root), "show", TIP + ":" + relative]).stdout == path.read_bytes(), "new helper differs from its Git blob")
    save(Path(audit) / "new_helper.json", {"root": str(root), "source_commit": TIP, "source_tree": continuation["source_tree"],
        "helper_count": 18, "fresh": True, "bundle": bundle, "sparse_paths": sparse})


def output_files(data, audit):
    """List only new standard packet/data files and the bounded new audit records for exact raw return."""
    continuation = data["continuation"]
    packet = ordinary(continuation["planned_native_packet_directory"], directory=True)
    unit = ordinary(continuation["planned_native_data_unit"], directory=True)
    archive = data["old_plan"]["admission_archive"]["files"]
    names = PACKET_FILES | {"admission_files/" + Path(row["captured"]["path"]).name for row in archive}
    require(len(archive) == 4, "exactly four original admission archive records required")
    entries = []
    for directory, prefix, allowed in ((packet, "packet", names), (unit, "data", STANDARD_FILES)):
        actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
        require(actual == allowed and not any(path.is_symlink() for path in directory.rglob("*")), "new packet/data contains an unexpected or missing file")
        entries += [{"relative_path": prefix + "/" + name, **receipt(directory / name)} for name in sorted(allowed)]
    entries += [{"relative_path": "audit/" + path.name, **receipt(path)} for path in sorted(Path(audit).iterdir()) if path.is_file()]
    require(len(entries) <= 512 and sum(row["bytes"] for row in entries) <= 64 * 2 ** 20, "native metadata return exceeds its declared bound")
    return entries


def emit(entries, result):
    """Return bounded exact native files with their original paths and byte hashes, never base64 or reconstructed records."""
    raw = (json.dumps({"schema_version": 1, "kind": "aces_autodetect_native_preparation_return", "result": result, "files": entries},
                      allow_nan=False) + "\n").encode()
    stream = sys.stdout.buffer
    stream.write(MAGIC + str(len(raw)).encode() + b"\n" + raw)
    for row in entries:
        require(receipt(row["path"]) == {key: row[key] for key in ("path", "sha256", "bytes")}, "a returned native file changed")
        with Path(row["path"]).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                stream.write(block)
    stream.flush()


def remote_main(payload):
    """Run only fresh source transport and actual native metadata preparation under the existing ACES owner."""
    plan, data = checked_payload(payload)
    reused, continuation = data["reused"], data["continuation"]
    require(os.getuid() == 56626 and pwd.getpwuid(os.getuid()).pw_name == "u.sl359075"
        and socket.gethostname().split(".")[0] == "alogin1" and sys.dont_write_bytecode
        and str(Path(sys.executable).absolute()) == reused["private_python"], "exact ACES login owner/private interpreter required")
    destinations = [plan["remote_execution_directory"], continuation["new_helper_checkout_root"],
        continuation["new_remote_bundle"], continuation["new_remote_bundle"] + ".incomplete",
        continuation["planned_native_packet_directory"], continuation["planned_native_data_unit"]]
    require(all(not Path(path).exists() and not Path(path).is_symlink() for path in destinations), "an execution, source, bundle or packet destination already exists; never reuse partial work")
    audit = Path(plan["remote_execution_directory"])
    ordinary(audit.parent, directory=True)
    audit.mkdir()
    sync_directory(audit.parent)
    try:
        save(audit / "intent.json", {"plan_sha256": PLAN_SHA, "source_commit": TIP, "basis_commit": BASIS,
            "driver_sha256": payload["driver_sha256"], "bundle": payload["bundle"], "created_at_utc": datetime.now(timezone.utc).isoformat()})
        before = preserve(payload, data, audit, "before")
        # Read the single streamed bundle only after every existing native byte/input check passed.
        bundle = {**payload["bundle"], "path": continuation["new_remote_bundle"]}
        require(0 < bundle["bytes"] <= plan["maximum_bundle_bytes"], "incremental source bundle exceeds the fixed bound")
        receive_bundle(sys.stdin.buffer, bundle["path"], bundle)
        stage_helper(data, bundle, audit)
        environment = native_environment(data)
        invoke(data["commands"]["native_prepare_argv"], audit, environment=environment, timeout=300)
        checked = invoke(data["commands"]["native_post_prepare_verification_argv"], audit, environment=environment, timeout=300)
        verification = json.loads(checked.stdout)
        require(verification.get("passed") is True and verification.get("helper_commit") == TIP
            and verification.get("helper_count") == 18 and verification.get("workers_permitted") is False
            and verification.get("actual_site_success") is False, "native metadata verification did not pass its zero-credit boundary")
        save(audit / "native_verification.json", verification)
        after = preserve(payload, data, audit, "after")
        require(before == after, "old native files, sources, software or inputs changed during metadata preparation")
        no_numerical_modules()
        result = {"schema_version": 1, "kind": "aces_actual_autodetect_incremental_native_preparation", "passed": True,
            "plan_sha256": PLAN_SHA, "source_commit": TIP, "source_tree": continuation["source_tree"], "basis_commit": BASIS,
            "driver_sha256": payload["driver_sha256"], "native_packet": continuation["planned_native_packet_directory"],
            "native_data_unit": continuation["planned_native_data_unit"], "private_python": reused["private_python"],
            "preserved_before_after": True, "actual_site_success": False, "numerical_imports": [], "jobs_submitted": 0,
            "workers_permitted": False, "scientific_hypothesis_credit": 0, "comparison_credit": 0, "idea_credit": 0, "admission_credit": 0}
        save(audit / "result.json", result)
        emit(output_files(data, audit), result)
    except BaseException as error:
        save(audit / "failure.json", {"error_type": type(error).__name__, "error": str(error),
            "at_utc": datetime.now(timezone.utc).isoformat(), "retry_permitted": False, "jobs_submitted": 0})
        raise



def validate_return_roster(rows, plan, data):
    """Bound original native paths and require the complete exact packet/data roster before copying bytes."""
    require(isinstance(rows, list) and 0 < len(rows) <= 512
        and all(set(row) == {"relative_path", "path", "sha256", "bytes"}
            and type(row["bytes"]) is int and 0 <= row["bytes"] <= 4 * 2 ** 20
            and isinstance(row["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in rows)
        and sum(row["bytes"] for row in rows) <= plan["maximum_return_bytes"]
        and len({row["relative_path"] for row in rows}) == len(rows), "native return roster or size is ambiguous")
    actual = {"packet": set(), "data": set(), "audit": set()}
    for row in rows:
        relative = PurePosixPath(row["relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == row["relative_path"]
            and len(relative.parts) >= 2 and relative.parts[0] in actual, "returned relative path escaped its receiver")
        prefix = {"packet": data["continuation"]["planned_native_packet_directory"],
            "data": data["continuation"]["planned_native_data_unit"], "audit": plan["remote_execution_directory"]}[relative.parts[0]]
        require(row["path"] == str(Path(prefix).joinpath(*relative.parts[1:])), "returned original native path changed")
        actual[relative.parts[0]].add(str(PurePosixPath(*relative.parts[1:])))
    archive = {"admission_files/" + Path(row["captured"]["path"]).name for row in data["old_plan"]["admission_archive"]["files"]}
    require(actual["packet"] == PACKET_FILES | archive and actual["data"] == STANDARD_FILES,
            "returned packet/data roster omitted or introduced an artifact")
    required_audit = {"intent.json", "preserved_before.json", "preserved_after.json", "new_helper.json", "native_verification.json", "result.json"}
    require(required_audit <= actual["audit"] and all(name in required_audit
        or re.fullmatch(r"command-\d{4}\.(intent\.json|json|stdout\.raw|stderr\.raw)", name)
        for name in actual["audit"]), "returned audit roster contains an unknown or missing record")


def validate_received(received, header, data):
    """Require actual returned metadata to preserve original inputs and every zero-credit boundary."""
    packet, audit = Path(received) / "packet", Path(received) / "audit"
    result = header["result"]
    require(json.loads((audit / "result.json").read_bytes()) == result and result.get("source_commit") == TIP
        and result.get("basis_commit") == BASIS and result.get("preserved_before_after") is True
        and result.get("native_packet") == data["continuation"]["planned_native_packet_directory"]
        and result.get("native_data_unit") == data["continuation"]["planned_native_data_unit"], "returned result changed its actual prepared source or native identity")
    for name, expected in {"workers_permitted": False, "actual_site_success": False, "jobs_submitted": 0,
            "scientific_hypothesis_credit": 0, "comparison_credit": 0, "idea_credit": 0, "admission_credit": 0}.items():
        require(type(result.get(name)) is type(expected) and result[name] == expected, "metadata preparation cannot grant credit or admission")
    current = json.loads((packet / "site_validation_plan.json").read_bytes())
    for name in ("python", "numerical_source", "cleanrl_reference", "portable_admission", "asset_record", "jaxtari_source_root",
            "admission", "configuration", "runtime_environment", "gate_environment_policy", "job_shape", "reference_updates", "child_order"):
        require(current[name] == data["old_plan"][name], "returned native preparation changed an original input: " + name)
    require(current["helpers"]["source_commit"] == TIP
        and current["helpers"]["source_root"] == data["continuation"]["new_helper_checkout_root"]
        and current["helpers"]["files"] == data["continuation"]["expected_helper_closure"]
        and current["qualification_profile"]["name"] == data["continuation"]["profile_name"]
        and current["qualification_profile"]["physical_gres_mapping_verified"] is False
        and current["qualification_profile"]["kernel_device_enforcement_verified"] is False
        and json.loads((packet / "software_at_prepare.json").read_bytes()) == data["software"], "returned actual native source/software/profile changed")
    manifest = json.loads((packet / "manifest.yaml").read_bytes())
    require(set(manifest["artifacts"]) == PACKET_FILES - {"manifest.yaml"}, "returned standard preparation manifest is incomplete")
    for name, value in manifest["artifacts"].items():
        require(value["path"] == str(Path(data["continuation"]["planned_native_packet_directory"]) / name)
            and sha(packet / name) == value["sha256"], "returned prepared manifest does not bind its actual native bytes")
    require(current["software_at_prepare"] == manifest["artifacts"]["software_at_prepare.json"]
        and json.loads((packet / "config_resolved.yaml").read_bytes()) == current,
        "returned resolved configuration or actual software receipt differs from the prepared plan")
    shape = data["old_plan"]["job_shape"]
    command = ["sbatch", "--parsable", "--hold", *["--" + flag + "=" + str(shape[key]) for flag, key in
        (("account", "account"), ("partition", "partition"), ("qos", "qos"), ("nodes", "nodes"), ("ntasks", "ntasks"),
         ("gres", "gres"), ("cpus-per-task", "cpus_per_task"), ("mem", "memory"), ("time", "time"))],
        "--job-name=ACES_H100_full_site_qualification", "--output=" + current["campaign_run_dir"] + "/slurm/jobs/aces_full_site_qualification_%j.log",
        data["continuation"]["planned_native_packet_directory"] + "/job.sbatch"]
    require(shlex.split((packet / "command.txt").read_text()) == manifest["submission_command"] == command,
            "returned held command changed its exact reviewed resources or destination")
    for name in STANDARD_FILES:
        require((packet / name).read_bytes() == (Path(received) / "data" / name).read_bytes(), "returned initial data bytes differ from their packet")
    before = json.loads((audit / "preserved_before.json").read_bytes())
    require(before == json.loads((audit / "preserved_after.json").read_bytes()),
            "returned original environment/input observations changed")
    require(before["roots"] == expected_roots(data["preserved"])
        and before["base_package_count"] == 77 and before["private_package_count"] == 7
        and before["active_dependency_count"] == 128 and before["base_metadata_and_shared_files_unchanged"] is True
        and before["private_metadata_unchanged"] is True and before["files"]["mapping_files"] == 72
        and before["files"]["asset_files"] == 4531
        and before["files"]["installed_locked_files"] == sum(len(row["verified_files"]) for row in data["installed_files"]),
        "returned preservation proof omitted original sources, software or exact inputs")
    intent = json.loads((audit / "intent.json").read_bytes())
    helper = json.loads((audit / "new_helper.json").read_bytes())
    verification = json.loads((audit / "native_verification.json").read_bytes())
    require(intent["plan_sha256"] == PLAN_SHA and intent["driver_sha256"] == result["driver_sha256"]
        and helper["source_commit"] == TIP and helper["source_tree"] == data["continuation"]["source_tree"]
        and helper["root"] == data["continuation"]["new_helper_checkout_root"] and helper["helper_count"] == 18
        and verification["passed"] is True and verification["private_python"] == data["reused"]["private_python"]
        and verification["helper_commit"] == TIP and verification["helper_count"] == 18
        and verification["actual_site_success"] is False and verification["workers_permitted"] is False,
        "returned actual metadata validation did not bind this source, private Python or zero-credit scope")


def accept_return(path, directory, payload):
    """Parse only an already durable raw stream and copy each exact allowed native byte to a fresh local receiver."""
    plan, data = checked_payload(payload)
    received = Path(directory) / "received"
    require(not received.exists() and not received.is_symlink(), "a prior returned-file directory already exists")
    require(ordinary(path).stat().st_size <= plan["maximum_return_bytes"] + 2 ** 20, "raw native return exceeds its fixed bound")
    with Path(path).open("rb") as stream:
        require(stream.read(len(MAGIC)) == MAGIC, "native return has an unexpected prefix; preserve raw bytes")
        length = stream.readline(32)
        require(length.strip().isdigit() and 0 < int(length) <= 2 ** 20, "native frame header length differs")
        raw = stream.read(int(length))
        header = json.loads(raw)
        require(header.get("kind") == "aces_autodetect_native_preparation_return" and type(header.get("schema_version")) is int and header["schema_version"] == 1
            and header["result"].get("passed") is True and header["result"].get("plan_sha256") == PLAN_SHA
            and header["result"].get("driver_sha256") == payload["driver_sha256"], "native result did not bind this exact one-shot source")
        rows = header["files"]
        validate_return_roster(rows, plan, data)
        received.mkdir()
        sync_directory(received.parent)
        for row in rows:
            relative = PurePosixPath(row["relative_path"])
            destination = received / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            content = stream.read(row["bytes"])
            require(len(content) == row["bytes"] and hashlib.sha256(content).hexdigest() == row["sha256"], "returned exact native bytes changed")
            save_bytes(destination, content)
        require(not stream.read(1), "unexpected bytes follow the native metadata return")
    validate_received(received, header, data)
    save(Path(directory) / "accepted_return.json", header)
    return header


def send_payload(stream, payload, errors):
    """Send one bounded metadata header and its exact incremental bundle without retries."""
    try:
        raw = (json.dumps(payload, allow_nan=False) + "\n").encode()
        require(len(raw) < 16 * 2 ** 20, "transport metadata header exceeds its fixed bound")
        stream.write(raw)
        with ordinary(payload["bundle"]["path"]).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                stream.write(block)
        stream.flush()
    except BaseException as error:
        errors.append(repr(error))
    finally:
        stream.close()


def local_execute(payload, source):
    """Use one existing-master-only SSH operation and retain raw returns before parsing or publishing any native result."""
    plan, data = checked_payload(payload)
    require(socket.gethostname().split(".")[0] == "portal05" and pwd.getpwuid(os.getuid()).pw_name == "sl5nw",
            "actual native preparation may execute only from portal05 as the campaign owner")
    audit = Path(plan["local_execution_directory"])
    require(not audit.exists() and not audit.is_symlink(), "a previous local execution or partial intent already exists")
    ordinary(audit.parent, directory=True)
    audit.mkdir()
    sync_directory(audit.parent)
    payload = {**payload, "driver_sha256": hashlib.sha256(source).hexdigest()}
    save(audit / "intent.json", {"plan_sha256": PLAN_SHA, "driver_sha256": payload["driver_sha256"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "existing_master_only": True,
        "separate_master_checks": 0, "authentication_attempts": 0, "retry_permitted": False})
    try:
        payload["bundle"] = build_bundle(ROOT, audit, data["continuation"])
        require(payload["bundle"]["bytes"] <= plan["maximum_bundle_bytes"], "incremental bundle exceeds its fixed transfer bound")
        payload["driver_source"] = source.decode()
        bootstrap = "import hashlib,json,sys,types\np=json.loads(sys.stdin.buffer.readline(16*2**20))\ns=p.pop('driver_source')\nassert hashlib.sha256(s.encode()).hexdigest()==p['driver_sha256']\nm=types.ModuleType('reviewed_continuation')\nexec(compile(s,'<reviewed-continuation>','exec'),m.__dict__)\nm.remote_main(p)"
        command = SSH + [shlex.join([data["reused"]["private_python"], "-I", "-B", "-c", bootstrap])]
        save(audit / "ssh_invocation.json", {"argv": command, "plan_sha256": PLAN_SHA, "driver_sha256": payload["driver_sha256"], "bundle": payload["bundle"]})
        errors, started = [], datetime.now(timezone.utc).isoformat()
        stdout, stderr = audit / "stdout.raw", audit / "stderr.raw"
        with stdout.open("xb") as out, stderr.open("xb") as err:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err)
            sender = threading.Thread(target=send_payload, args=(child.stdin, payload, errors), daemon=True)
            sender.start()
            try:
                returncode = child.wait(timeout=plan["transport_timeout_seconds"])
            except subprocess.TimeoutExpired:
                child.kill()
                returncode = child.wait(timeout=30)
                errors.append("the single metadata transport exceeded its declared timeout; never retry")
            sender.join(timeout=30)
            if sender.is_alive():
                errors.append("the owned transport sender did not stop within its bound")
            out.flush()
            err.flush()
            os.fsync(out.fileno())
            os.fsync(err.fileno())
        sync_directory(audit)
        save(audit / "transport_result.json", {"started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": returncode, "send_errors": errors, "stdout": receipt(stdout), "stderr": receipt(stderr),
            "existing_master_only": True, "authentication_attempts": 0, "master_checks": 0, "ssh_calls": 1, "retries": 0})
        require(returncode == 0 and not errors, "native metadata operation failed; preserve the complete raw return and all partial targets")
        return accept_return(stdout, audit, payload)
    except BaseException as error:
        save(audit / "failure.json", {"error_type": type(error).__name__, "error": str(error),
            "at_utc": datetime.now(timezone.utc).isoformat(), "retry_permitted": False, "jobs_submitted": 0})
        raise



def main():
    """Check local input bytes by default; execute a single reviewed operation only under the explicit root flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    payload = local_payload(arguments.plan)
    if arguments.execute:
        result = local_execute(payload, ordinary(__file__).read_bytes())
        print(json.dumps({"passed": True, "native_packet": result["result"]["native_packet"],
            "returned_files": len(result["files"]), "jobs_submitted": 0, "workers_permitted": False}))
    else:
        print(json.dumps({"local_inputs_verified": True, "plan_sha256": PLAN_SHA, "source_commit": TIP,
            "basis_commit": BASIS, "documents": len(payload["documents"]), "ssh_calls": 0, "jobs_submitted": 0}))


if __name__ == "__main__":
    main()
