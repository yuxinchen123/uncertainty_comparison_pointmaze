"""Copy an explicit evidence list and verify pre-existing staged Git roots.

Input JSON has schema_version=1, kind='portable_evidence_input_list', files
containing original_path/sha256, and git_roots containing the exact resolver's
original_root/relative_path/commit fields. Git checkouts already exist inside
the staging root. No repository is fetched, guessed, or copied by this program.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import uuid

def resolver_module():
    """Use the adjacent existing resolver even when the builder itself is imported under an alias."""
    specification = importlib.util.spec_from_file_location("portable_package_exact_resolver",
        Path(__file__).resolve().with_name("portable_evidence.py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


portable = resolver_module()


MANIFEST = "manifest.json"
INPUT_COPY = "transport/input_list.json"
RECEIPT = "transport/preparation_receipt.json"
BLOCK_BYTES = 1024 * 1024


def json_bytes(value):
    """Serialize complete finite transport metadata without changing any scientific file."""
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def sha256(value):
    """Require one explicit complete lowercase SHA256."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("portable input needs a complete lowercase SHA256")
    return value


def relative_path(value):
    """Reject absolute, normalized, empty and traversal aliases before creating any target."""
    if (not isinstance(value, str) or not value or "\0" in value or value == "."
            or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts
            or str(PurePosixPath(value)) != value):
        raise ValueError("portable staging targets must be canonical relative paths")
    return value


def target_path(root, relative, exists=False):
    """Keep all actual target components inside staging and reject symlink-dependent transport paths."""
    relative = relative_path(relative)
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("a portable staging target contains a symlink: " + relative)
    resolved = path.resolve(strict=exists)
    if not resolved.is_relative_to(root):
        raise ValueError("a portable staging target escapes its staging root")
    return path


def input_document(contents):
    """Build the exact resolver manifest from a finite explicit list, with no path discovery."""
    supplied = portable.document(contents)
    if (not isinstance(supplied, dict) or set(supplied) != {"schema_version", "kind", "files", "git_roots"}
            or type(supplied["schema_version"]) is not int or supplied["schema_version"] != 1
            or supplied["kind"] != "portable_evidence_input_list"
            or not isinstance(supplied["files"], list) or not isinstance(supplied["git_roots"], list)
            or not supplied["files"] and not supplied["git_roots"]):
        raise ValueError("supply the exact version-one explicit portable input list")
    files, roots, identities, original_roots, targets = [], [], set(), set(), {}
    # Before: two originals with one SHA. After: two exact identities sharing one byte copy.
    for item in supplied["files"]:
        if not isinstance(item, dict) or set(item) != {"original_path", "sha256"}:
            raise ValueError("portable file inputs need exactly original_path and sha256")
        original, expected = portable.original_path(item["original_path"]), sha256(item["sha256"])
        if original in identities:
            raise ValueError("a portable original file path is duplicated")
        identities.add(original)
        files.append({"original_path": original, "relative_path": "files/" + expected, "sha256": expected})
    for item in supplied["git_roots"]:
        if not isinstance(item, dict) or set(item) != {"original_root", "relative_path", "commit"}:
            raise ValueError("portable Git inputs need original_root, relative_path and commit")
        original = portable.original_path(item["original_root"])
        relative, commit = relative_path(item["relative_path"]), item["commit"]
        if original in original_roots or not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("a portable Git identity is duplicated or its commit is invalid")
        if (PurePosixPath(relative).parts[0] in {"files", "transport"} or relative == MANIFEST
                or relative in targets and targets[relative] != commit):
            raise ValueError("portable Git targets conflict with transport files or another commit")
        original_roots.add(original)
        targets[relative] = commit
        roots.append({"original_root": original, "relative_path": relative, "commit": commit})
    return {"schema_version": 1, "kind": "portable_raw_evidence", "files": files, "git_roots": roots}


def git_environment():
    """Reject process overrides that could make another object store appear to be the staged repository."""
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY", "GIT_INDEX_FILE", "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        if os.environ.get(name):
            raise ValueError("portable Git verification cannot inherit " + name)
    return {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"}


def git_command(path, *arguments):
    """Run local Git metadata checks with network-backed lazy object retrieval disabled."""
    return subprocess.check_output(["git", "-C", str(path), *arguments], env=git_environment(), text=True).strip()


def verify_metadata_tree(directory):
    """Reject metadata symlinks without restricting legitimate tracked source-code symlinks."""
    pending = [directory]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ValueError("portable Git metadata cannot contain a symlink: " + entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))


def verify_git_root(root, entry):
    """Require real clean detached Git metadata and all reachable objects inside the package."""
    path = target_path(root, entry["relative_path"], exists=True)
    if not (path / ".git").exists():
        raise ValueError("staged Git source must have its own Git metadata")
    top = Path(git_command(path, "rev-parse", "--show-toplevel")).resolve(strict=True)
    commit = git_command(path, "rev-parse", "HEAD")
    branch = git_command(path, "rev-parse", "--abbrev-ref", "HEAD")
    changed = git_command(path, "status", "--porcelain", "--untracked-files=all")
    if top != path or commit != entry["commit"] or branch != "HEAD" or changed:
        raise ValueError("staged Git source must be its own clean detached root at the exact commit")
    directories = {}
    for name, argument in (("git_directory", "--absolute-git-dir"), ("common_directory", "--git-common-dir")):
        value = Path(git_command(path, "rev-parse", argument))
        actual = (path / value if not value.is_absolute() else value).resolve(strict=True)
        if not actual.is_relative_to(root):
            raise ValueError("staged Git metadata depends on a directory outside the package")
        relative = actual.relative_to(root).as_posix()
        target_path(root, relative, exists=True)
        directories[name] = relative
    for relative in set(directories.values()):
        verify_metadata_tree(root / relative)
    common = root / directories["common_directory"]
    if (common / "objects/info/alternates").exists() or os.environ.get("GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        raise ValueError("portable Git objects may not depend on external alternates")
    # Hash only Git object identities; large object-list output is consumed one line at a time.
    environment = git_environment()
    process = subprocess.Popen(["git", "-C", str(path), "rev-list", "--objects", "--missing=print",
        "--no-object-names", commit], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    count, objects = 0, hashlib.sha256()
    try:
        for line in process.stdout:
            if line.startswith(b"?") or not re.fullmatch(rb"[0-9a-f]{40}\n", line):
                raise ValueError("staged Git source has missing or invalid local objects")
            objects.update(line)
            count += 1
        error = process.stderr.read()
        if process.wait() != 0:
            raise ValueError("local Git object verification failed: " + error.decode(errors="replace"))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        process.stdout.close()
        process.stderr.close()
    return {**entry, "tree": git_command(path, "rev-parse", commit + "^{tree}"), **directories,
            "reachable_object_count": count, "reachable_object_list_sha256": objects.hexdigest()}


def publish(path, contents):
    """Publish complete immutable metadata without replacing earlier differing bytes."""
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError("an earlier portable transport record differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".temporary-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def copy_file(root, entry):
    """Stream and hash every original, safely sharing one complete staged file for identical hashes."""
    source = Path(entry["original_path"])
    target = target_path(root, entry["relative_path"])
    if not stat.S_ISREG(source.stat().st_mode):
        raise ValueError("portable inputs must be ordinary files")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".temporary-" + uuid.uuid4().hex)
    calculated, size = hashlib.sha256(), 0
    try:
        with source.open("rb") as stream, temporary.open("xb") as output:
            before = os.fstat(stream.fileno())
            for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
                calculated.update(block)
                size += len(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
            after = os.fstat(stream.fileno())
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        current = source.stat()
        if any(getattr(before, field) != getattr(after, field) or getattr(after, field) != getattr(current, field) for field in fields):
            raise ValueError("an original file changed while being copied")
        if calculated.hexdigest() != entry["sha256"]:
            raise ValueError("an original file differs from its explicitly expected SHA256: " + str(source))
        if target.exists():
            if not target.is_file() or portable.digest(target) != entry["sha256"]:
                raise ValueError("an existing staged hash target has different bytes")
        else:
            os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {**entry, "bytes": size}


@contextmanager
def package_lock(root):
    """Prevent concurrent preparations from publishing different transport records."""
    path = target_path(root, "transport/package.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def prepare(input_path, staging_root):
    """Copy only caller-listed evidence and emit resolver-compatible immutable transport metadata."""
    input_path = Path(portable.original_path(input_path))
    root = Path(portable.original_path(staging_root)).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("use an explicit existing staging directory")
    contents = input_path.read_bytes()
    manifest = input_document(contents)
    with package_lock(root):
        roots = [verify_git_root(root, entry) for entry in manifest["git_roots"]]
        # Scientific bytes are copied as bytes; no embedded original path is rewritten.
        files = [copy_file(root, entry) for entry in manifest["files"]]
        input_copy, manifest_path = target_path(root, INPUT_COPY), target_path(root, MANIFEST)
        publish(input_copy, contents)
        publish(manifest_path, json_bytes(manifest))
        resolver = portable.ExactPathResolver(manifest_path, portable.digest(manifest_path))
        roots_after = [verify_git_root(root, entry) for entry in manifest["git_roots"]]
        if roots_after != roots:
            raise ValueError("staged Git source changed during evidence preparation")
        receipt = {"schema_version": 1, "kind": "portable_evidence_preparation",
            "input_list": {"original_path": str(input_path), "relative_path": INPUT_COPY, "sha256": portable.digest(input_copy)},
            "manifest": {"relative_path": MANIFEST, "sha256": resolver.manifest_sha256},
            "files": files, "git_roots": roots,
            "producer_files": {name: portable.digest(Path(__file__).with_name(name))
                for name in ("prepare_portable_evidence.py", "portable_evidence.py")}}
        receipt_path = target_path(root, RECEIPT)
        publish(receipt_path, json_bytes(receipt))
        result = verify(root, resolver.manifest_sha256, portable.digest(receipt_path))
        return {**result, "transport_receipt_path": str(receipt_path)}


def verify(staging_root, manifest_sha256, transport_sha256):
    """Finalize a transferred package from exact expected hashes without reading any original path."""
    root = Path(portable.original_path(staging_root)).resolve(strict=True)
    receipt_path = target_path(root, RECEIPT, exists=True)
    if portable.digest(receipt_path) != sha256(transport_sha256):
        raise ValueError("the portable transport receipt differs from its expected SHA256")
    receipt = portable.document(receipt_path.read_bytes())
    if (set(receipt) != {"schema_version", "kind", "input_list", "manifest", "files", "git_roots", "producer_files"}
            or type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1
            or receipt["kind"] != "portable_evidence_preparation"
            or receipt["input_list"]["relative_path"] != INPUT_COPY
            or receipt["manifest"] != {"relative_path": MANIFEST, "sha256": sha256(manifest_sha256)}):
        raise ValueError("portable transport metadata differs from its exact preparation contract")
    if (set(receipt["input_list"]) != {"original_path", "relative_path", "sha256"}
            or set(receipt["producer_files"]) != {"prepare_portable_evidence.py", "portable_evidence.py"}):
        raise ValueError("portable transport receipt has incomplete original input or producer identities")
    portable.original_path(receipt["input_list"]["original_path"])
    for value in receipt["producer_files"].values():
        sha256(value)
    copied_input = target_path(root, INPUT_COPY, exists=True)
    if portable.digest(copied_input) != receipt["input_list"]["sha256"]:
        raise ValueError("the copied explicit input list changed")
    manifest = input_document(copied_input.read_bytes())
    resolver = portable.ExactPathResolver(target_path(root, MANIFEST, exists=True), manifest_sha256)
    if portable.document(resolver.manifest_path.read_bytes()) != manifest:
        raise ValueError("the portable resolver manifest changed its exact input mapping")
    files = [{**entry, "bytes": resolver.file_path(entry["original_path"]).stat().st_size} for entry in manifest["files"]]
    roots = [verify_git_root(root, entry) for entry in manifest["git_roots"]]
    if files != receipt["files"] or roots != receipt["git_roots"]:
        raise ValueError("portable staged files or real Git identities changed after preparation")
    return {"verified": True, "staging_root": str(root), "manifest_path": str(resolver.manifest_path),
        "manifest_sha256": resolver.manifest_sha256, "transport_receipt_sha256": transport_sha256,
        "mapped_files": len(files), "unique_staged_files": len({entry["relative_path"] for entry in files}),
        "mapped_git_roots": len(roots), "mapped_bytes": sum(entry["bytes"] for entry in files)}


def main():
    """Prepare an explicit local package or verify a transferred package; never submit jobs or contact remotes."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--inputs", type=Path, required=True)
    preparation.add_argument("--staging-root", type=Path, required=True)
    verification = commands.add_parser("verify")
    verification.add_argument("--staging-root", type=Path, required=True)
    verification.add_argument("--manifest-sha256", required=True)
    verification.add_argument("--transport-sha256", required=True)
    arguments = parser.parse_args()
    result = (prepare(arguments.inputs, arguments.staging_root) if arguments.command == "prepare"
              else verify(arguments.staging_root, arguments.manifest_sha256, arguments.transport_sha256))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
