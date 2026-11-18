"""Describe the installed distributions of one environment, so that two environments can be compared file by file.

    python3 distribution_records.py SITE_PACKAGES > records.json
    python3 distribution_records.py --compare UVA.json SERVER.json

Standard library only; it reads the dist-info folders and imports nothing from the environment. For each
distribution it records the version and one SHA-256 over the lines of its RECORD file (the installed files with
their own hashes), leaving out the lines that differ between two installs of the same wheel: compiled .pyc files,
scripts in bin/ (their first line names the interpreter), and the dist-info files pip writes at install time.
Two installs of the same wheel, or of the same source tree, therefore give the same digest.
"""
import hashlib
import json
from pathlib import Path
import sys

INSTALL_TIME_FILES = ("/INSTALLER", "/REQUESTED", "/RECORD", "/direct_url.json")


def normalized(name):
    """Spell a distribution name the way pip compares names."""
    # before: "Farama_Notifications" ; after: "farama-notifications"
    return name.lower().replace("_", "-").replace(".", "-")


def record_digest(record):
    """One SHA-256 over the RECORD lines that two installs of the same files share."""
    # before: ["jax/__init__.py,sha256=abc,1234", "jax/__pycache__/x.cpython-312.pyc,,", "../../../bin/x,sha256=..,238"]
    # after:  sha256 of "jax/__init__.py,sha256=abc,1234" alone
    kept = sorted(line for line in record.read_text().splitlines()
                  if line.strip() and not line.startswith("../") and ".pyc," not in line
                  and not any(line.split(",")[0].endswith(name) for name in INSTALL_TIME_FILES))
    return hashlib.sha256("\n".join(kept).encode()).hexdigest(), len(kept)


def describe(site_packages):
    """Record the version and the RECORD digest of every distribution of one site-packages folder."""
    # after: {"jax": {"version": "0.11.0", "record_sha256": "5f1e...", "files": 1234}, ...}
    records = {}
    for info in sorted(Path(site_packages).glob("*.dist-info")):
        metadata = (info / "METADATA").read_text(errors="replace").splitlines()
        name = next(line.split(":", 1)[1].strip() for line in metadata if line.startswith("Name:"))
        version = next(line.split(":", 1)[1].strip() for line in metadata if line.startswith("Version:"))
        digest, files = record_digest(info / "RECORD")
        records[normalized(name)] = {"version": version, "record_sha256": digest, "files": files}
    return records


def compare(reference, other):
    """Print the distributions whose version or installed files differ; returns the number that differ."""
    differing, names = 0, sorted(set(reference) | set(other))
    # every distribution of either side, in name order
    for name in names:
        left, right = reference.get(name), other.get(name)
        if left is None or right is None:
            differing += 1
            print(f"{name}: only in {'the second' if left is None else 'the first'} environment")
        elif left["version"] != right["version"]:
            differing += 1
            print(f"{name}: version {left['version']} against {right['version']}")
        elif left["record_sha256"] != right["record_sha256"]:
            differing += 1
            print(f"{name} {left['version']}: same version, installed files differ "
                  f"({left['files']} against {right['files']} files)")
    print(f"{len(names) - differing} distributions equal file by file, {differing} differ")
    return differing


def main(argv):
    """Describe one environment, or compare two descriptions."""
    if argv[:1] == ["--compare"]:
        reference, other = (json.loads(Path(path).read_text()) for path in argv[1:3])
        return 1 if compare(reference, other) else 0
    print(json.dumps(describe(argv[0]), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
