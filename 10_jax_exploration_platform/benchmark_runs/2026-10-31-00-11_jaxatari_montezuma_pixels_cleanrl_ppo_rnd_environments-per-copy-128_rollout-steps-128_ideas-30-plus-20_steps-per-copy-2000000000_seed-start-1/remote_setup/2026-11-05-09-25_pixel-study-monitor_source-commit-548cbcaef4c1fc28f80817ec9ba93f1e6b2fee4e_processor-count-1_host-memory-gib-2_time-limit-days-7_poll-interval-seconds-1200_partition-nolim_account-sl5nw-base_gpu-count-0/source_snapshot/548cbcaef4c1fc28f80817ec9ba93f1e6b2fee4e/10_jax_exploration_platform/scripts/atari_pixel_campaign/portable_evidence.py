"""Resolve unchanged historical evidence through explicit, hash-pinned staging entries.

A manifest contains schema_version=1, kind='portable_raw_evidence', files with
original_path/relative_path/sha256, and git_roots with original_root/relative_path/
commit. Relative paths are inside the manifest directory. Original paths are
identities, not locations to try when an entry is missing. This module never
rewrites a record, creates an output directory, or imports an array library.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


def document(contents):
    """Reject duplicate JSON object keys before they can replace a manifest entry."""
    def unique(pairs):
        """Retain every supplied field exactly once."""
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("portable evidence contains a duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(contents, object_pairs_hook=unique)


def original_path(value):
    """Require one canonical absolute POSIX identity without consulting the current host."""
    value = str(value) if isinstance(value, Path) else value
    if (not isinstance(value, str) or not value or "\0" in value or not value.startswith("/")
            or value.startswith("//") or ".." in PurePosixPath(value).parts or str(PurePosixPath(value)) != value):
        raise ValueError("portable evidence needs a canonical absolute original path: " + repr(value))
    return value


def digest(path):
    """Hash staged bytes in bounded chunks, including large evidence files."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class ExactPathResolver:
    """Keep logical paths unchanged while requiring exact staged files and detached Git roots."""

    def __init__(self, manifest_path, expected_sha256=None):
        """Validate the complete explicit mapping and pin the original manifest bytes."""
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        self.staging_root = self.manifest_path.parent
        contents = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(contents).hexdigest()
        if expected_sha256 is not None and self.manifest_sha256 != expected_sha256:
            raise ValueError("the portable evidence manifest differs from its required SHA256")
        manifest = document(contents)
        if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "kind", "files", "git_roots"}
                or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
                or manifest["kind"] != "portable_raw_evidence"
                or not isinstance(manifest["files"], list) or not isinstance(manifest["git_roots"], list)):
            raise ValueError("portable evidence requires the exact version-one file and Git-root manifest")
        self.files, self.git_roots = {}, {}
        targets = {}
        # Before: one original path and one relative staged file; after: an exact lookup, never a prefix rule.
        for entry in manifest["files"]:
            if not isinstance(entry, dict) or set(entry) != {"original_path", "relative_path", "sha256"}:
                raise ValueError("a portable file entry requires original_path, relative_path and sha256")
            key = original_path(entry["original_path"])
            if key in self.files or not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
                raise ValueError("a portable file identity is duplicated or has an invalid SHA256: " + key)
            path = self.staged_path(entry["relative_path"])
            if not path.is_file() or path == self.manifest_path:
                raise ValueError("a portable file target must be an ordinary staged file: " + key)
            if str(path) in targets and targets[str(path)] != entry["sha256"]:
                raise ValueError("one staged file has conflicting portable SHA256 values")
            targets[str(path)] = entry["sha256"]
            self.files[key] = dict(entry)
            self.file_path(key)
        for entry in manifest["git_roots"]:
            if not isinstance(entry, dict) or set(entry) != {"original_root", "relative_path", "commit"}:
                raise ValueError("a portable Git entry requires original_root, relative_path and commit")
            key = original_path(entry["original_root"])
            if key in self.git_roots or not isinstance(entry["commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", entry["commit"]):
                raise ValueError("a portable Git identity is duplicated or has an invalid commit: " + key)
            self.staged_path(entry["relative_path"])
            self.git_roots[key] = dict(entry)
            self.git_root(key, entry["commit"])

    def unchanged_manifest(self):
        """Reject a manifest edited after this resolver pinned its bytes."""
        if digest(self.manifest_path) != self.manifest_sha256:
            raise ValueError("the portable evidence manifest changed during validation")

    def staged_path(self, relative):
        """Require a relative target whose actual symlink-resolved location stays inside staging."""
        if (not isinstance(relative, str) or not relative or "\0" in relative
                or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts
                or str(PurePosixPath(relative)) != relative or relative == "."):
            raise ValueError("portable targets must be canonical relative paths inside staging")
        path = (self.staging_root / relative).resolve(strict=True)
        if not path.is_relative_to(self.staging_root):
            raise ValueError("a portable target resolves outside its staging directory")
        return path

    def file_path(self, path):
        """Resolve only a listed file and recheck its actual bytes against the pinned SHA256."""
        self.unchanged_manifest()
        key = original_path(path)
        if key not in self.files:
            raise ValueError("portable evidence has no exact file entry for original path: " + key)
        entry = self.files[key]
        actual = self.staged_path(entry["relative_path"])
        if not actual.is_file() or digest(actual) != entry["sha256"]:
            raise ValueError("staged evidence differs from its pinned SHA256: " + key)
        return actual

    def read_bytes(self, path):
        """Return unchanged raw bytes and detect changes between path checking and reading."""
        key = original_path(path)
        contents = self.file_path(key).read_bytes()
        if hashlib.sha256(contents).hexdigest() != self.files[key]["sha256"]:
            raise ValueError("staged evidence changed while being read: " + key)
        return contents

    def git_root(self, root, commit):
        """Require the exact mapped directory to be a clean detached Git top at the pinned commit."""
        self.unchanged_manifest()
        key = original_path(root)
        if key not in self.git_roots:
            raise ValueError("portable evidence has no exact Git-root entry for original path: " + key)
        entry = self.git_roots[key]
        if entry["commit"] != commit:
            raise ValueError("portable Git root does not admit the requested original commit: " + key)
        path = self.staged_path(entry["relative_path"])
        top = subprocess.check_output(["git", "-C", str(path), "rev-parse", "--show-toplevel"], text=True).strip()
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.run(["git", "-C", str(path), "symbolic-ref", "--quiet", "HEAD"], capture_output=True, text=True)
        changed = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"], text=True).strip()
        if Path(top).resolve(strict=True) != path or actual != commit or branch.returncode != 1 or changed:
            raise ValueError("portable source must be a clean detached Git root at its pinned commit: " + key)
        return path

    def receipt(self):
        """Describe transport separately from the original scientific evidence receipts."""
        self.unchanged_manifest()
        return {"kind": "portable_raw_evidence", "schema_version": 1,
                "manifest_path": str(self.manifest_path), "manifest_sha256": self.manifest_sha256,
                "mapped_files": len(self.files), "mapped_git_roots": len(self.git_roots)}
