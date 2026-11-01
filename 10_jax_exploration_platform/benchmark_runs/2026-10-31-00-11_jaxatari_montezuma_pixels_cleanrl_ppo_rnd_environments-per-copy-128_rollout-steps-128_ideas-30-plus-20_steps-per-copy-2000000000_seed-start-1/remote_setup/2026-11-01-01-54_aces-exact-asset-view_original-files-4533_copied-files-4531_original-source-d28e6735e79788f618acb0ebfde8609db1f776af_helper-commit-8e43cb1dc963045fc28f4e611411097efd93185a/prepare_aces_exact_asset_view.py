"""Copy one exact ACES asset view and prepare native metadata without running jobs.

The original 4533-file asset root, failed staging inputs, existing checkouts and
failure evidence are read-only inputs. A fresh physical view contains exactly
the 4531 original manifest files. No source helper or expected hash is changed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
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
import types


REPAIR_SHA = "cb3185e5cbedacb31a8aebc801f78a69112c91591588a5582c59fd3d3a65756c"
BASE_SHA = "95e99f9c213b23c6f8f82c8e8123d25d2a71acf3f4166217a872b0d1126e410a"
ORIGINAL_SHA = "30892127adbc487f157dcb36d89e79635041278b09ca746d7b9e7139d9cb9d8f"
OBSERVATION_SHA = "62706387149a3cf1718caab9ea76f6ba75e5561fa3fb883ee0486e7f3545bc69"
MANIFEST_SHA = "049004b3b7e2eb3b12f72579a175ac41afad3b5f91a088d7ba1b5065aed31bb6"
BASE = Path("/scratch/user/u.sl359075/rnd_atari_pixels")
PYTHON = "/scratch/user/u.sl359075/envs/rnd_atari_pixels_jax/bin/python"
REMOTE_COMMANDS = []
BOOTSTRAP = '''import hashlib,json,sys
line=sys.stdin.buffer.readline(2097153)
if len(line)>2097152 or not line.endswith(b"\\n"): raise ValueError("bounded repair header required")
payload=json.loads(line)
if hashlib.sha256(payload["driver_source"].encode()).hexdigest()!=sys.argv[1]: raise ValueError("repair source SHA changed")
namespace={"__name__":"aces_asset_view_remote","__file__":"<reviewed asset view repair>"}
exec(compile(payload["driver_source"],"<reviewed asset view repair>","exec"),namespace)
try:
 namespace["remote_prepare"](payload)
except Exception as error:
 failure={"schema_version":1,"kind":"aces_exact_asset_view_preparation_failure","error_type":type(error).__name__,"error":str(error),"commands":namespace["REMOTE_COMMANDS"],"automatic_retry_permitted":False,"jobs_submitted":0}
 sys.stdout.buffer.write((json.dumps(failure)+"\\n").encode());sys.stdout.buffer.flush()
 raise
'''


def digest(contents):
    """Hash exact reviewed bytes without reserializing the input documents."""
    return hashlib.sha256(contents).hexdigest()


def base_module(source):
    """Reuse only the exact published stdlib staging utilities without running their main entry point."""
    if digest(source.encode()) != BASE_SHA:
        raise ValueError("the preserved base staging utilities changed")
    module = types.ModuleType("aces_asset_view_staging_utilities")
    module.__file__ = "<published fixed staging utilities>"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def checked_payload(payload):
    """Bind the original failure, roster, source plan and separate asset repair plan exactly."""
    if set(payload) != {"repair_plan", "original_plan", "observation", "failed_execution", "base_source", "driver_source"}:
        raise ValueError("the exact six-document repair payload is required")
    repair_bytes = payload["repair_plan"].encode()
    if digest(repair_bytes) != REPAIR_SHA:
        raise ValueError("the asset-view repair plan differs from its reviewed hash")
    repair = json.loads(repair_bytes)
    base = base_module(payload["base_source"])
    for key, name, expected in (("original_plan", "original_staging_plan", ORIGINAL_SHA),
            ("observation", "asset_observation", OBSERVATION_SHA), ("failed_execution", "failed_execution", repair["failed_execution"]["sha256"])):
        if digest(payload[key].encode()) != expected or repair[name]["sha256"] != expected:
            raise ValueError("a frozen input changed before the separate asset-view repair")
    original = base.checked_plan(payload["original_plan"].encode())
    observation = json.loads(payload["observation"])
    failure = json.loads(payload["failed_execution"])
    root = Path(repair["repair_directory"])
    packet = Path(repair["native_packet_directory"])
    if (type(repair.get("schema_version")) is not int or repair["schema_version"] != 1
            or repair.get("kind") != "aces_exact_asset_view_native_preparation_plan"
            or repair.get("base_driver", {}).get("sha256") != BASE_SHA
            or repair.get("expected_manifest_sha256") != MANIFEST_SHA
            or root.parent != BASE / "remote_setup" or Path(repair["asset_view"]) != root / "asset_view"
            or Path(repair["asset_record"]) != root / "asset_record.json"
            or packet.parent != Path(original["remote_campaign_run_dir"]) / "remote_setup"
            or packet.name != repair["new_packet_name"] or "_aces-full-site_" not in packet.name
            or repair["native_data_unit"] != str(Path(original["remote_campaign_run_dir"]) / "data" / packet.name)
            or observation.get("kind") != "aces_actual_asset_roster_observation"
            or observation.get("root") != str(BASE / "assets") or observation.get("numerical_imports") is not False
            or type(observation.get("jobs_submitted")) is not int or observation["jobs_submitted"] != 0
            or failure.get("kind") != "aces_native_metadata_staging_failure" or failure.get("jobs_submitted") != 0):
        raise ValueError("the frozen repair does not describe this exact failed native preparation")
    return base, repair, original, observation


def local_payload(plan_path):
    """Validate every local repair input and all thirteen original files before any SSH."""
    path = Path(plan_path)
    if not path.is_absolute() or path != path.resolve(strict=True) or not path.is_file():
        raise ValueError("an exact ordinary absolute local repair plan is required")
    contents = path.read_bytes()
    if digest(contents) != REPAIR_SHA:
        raise ValueError("the local repair plan differs from its reviewed hash")
    repair = json.loads(contents)
    payload = {"repair_plan": contents.decode(), "driver_source": Path(__file__).read_text()}
    for key, name in (("original_plan", "original_staging_plan"), ("observation", "asset_observation"),
            ("failed_execution", "failed_execution"), ("base_source", "base_driver")):
        record = repair[name]
        source = Path(record["path"])
        if not source.is_absolute() or source != source.resolve(strict=True) or not source.is_file():
            raise ValueError("a local frozen repair input is not its ordinary exact path")
        raw = source.read_bytes()
        if digest(raw) != record["sha256"]:
            raise ValueError("a local frozen repair input changed")
        payload[key] = raw.decode()
    base, repair, original, observation = checked_payload(payload)
    base.validate_local(repair["original_staging_plan"]["path"])
    return payload, base, repair, original, observation


def asset_roster(base, root):
    """Hash every original ordinary asset file and reject links anywhere in that existing tree."""
    root = base.ordinary(root, directory=True)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("the original asset tree contains a symlink")
        if path.is_file():
            record = base.file_record(path)
            files[str(path.relative_to(root))] = {name: record[name] for name in ("sha256", "bytes")}
        elif not path.is_dir():
            raise ValueError("the original asset tree contains a nonregular entry")
    return files


def expected_manifest(base, original, observation):
    """Require all 4531 original hashes and only the two observed additional operational files."""
    row, = [row for row in original["exact_copy_allowlist"] if Path(row["remote_path"]).name == "expected_assets_manifest.json"]
    if base.file_record(row["remote_path"])["sha256"] != MANIFEST_SHA:
        raise ValueError("the original exhaustive expected asset manifest changed")
    manifest = json.loads(Path(row["remote_path"]).read_bytes())
    files = manifest["files"]
    if (len(files) != 4531 or manifest.get("source_commit") != "04ab9aec318c72162e2e53503a594a489d7e162b"
            or set(observation["files"]) != set(files) | {"asset_verification.log", "assets_tree.sha256"}
            or any(observation["files"][name]["sha256"] != expected for name, expected in files.items())):
        raise ValueError("the observed source assets do not contain every unchanged original asset")
    return row["remote_path"], files


def copy_exact_assets(base, source, destination, files, observed):
    """Create new physical files exclusively, using every expected byte hash without linking originals."""
    source = base.ordinary(source, directory=True)
    destination = Path(destination)
    destination.mkdir()
    copied = []
    for relative, expected in sorted(files.items()):
        name = PurePosixPath(relative)
        if name.is_absolute() or str(name) != relative or ".." in name.parts:
            raise ValueError("an expected asset name is outside its exact tree")
        original = base.ordinary(source / relative)
        target = destination / relative
        base.make_parents(target.parent)
        hasher, size = hashlib.sha256(), 0
        descriptor = os.open(original, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as incoming, target.open("xb") as outgoing:
            original_stat = os.fstat(incoming.fileno())
            if not stat.S_ISREG(original_stat.st_mode):
                raise ValueError("an opened source asset is not an ordinary file")
            while True:
                chunk = incoming.read(base.BLOCK)
                if not chunk:
                    break
                outgoing.write(chunk)
                hasher.update(chunk)
                size += len(chunk)
                if size > observed[relative]["bytes"]:
                    raise ValueError("a source asset grew beyond its observed length while copying")
            outgoing.flush()
            os.fsync(outgoing.fileno())
            copied_stat = os.fstat(outgoing.fileno())
        if (hasher.hexdigest() != expected or size != observed[relative]["bytes"]
                or copied_stat.st_nlink != 1 or (copied_stat.st_dev, copied_stat.st_ino) == (original_stat.st_dev, original_stat.st_ino)):
            raise ValueError("an exclusive physical asset copy changed bytes or reused an original inode")
        base.fsync_directory(target.parent)
        copied.append({"relative_path": relative, "sha256": expected, "bytes": size,
            "original_inode": [original_stat.st_dev, original_stat.st_ino], "copied_inode": [copied_stat.st_dev, copied_stat.st_ino],
            "copied_link_count": copied_stat.st_nlink})
    expected_roster = {relative: {"sha256": expected, "bytes": observed[relative]["bytes"]} for relative, expected in files.items()}
    if asset_roster(base, destination) != expected_roster:
        raise ValueError("the fresh physical view does not contain exactly the expected asset bytes")
    if any((destination / relative).stat().st_nlink != 1 for relative in files):
        raise ValueError("a fresh asset copy acquired an unexpected hardlink")
    base.fsync_directory(destination)
    return copied


def input_records(base, original):
    """Recheck all thirteen failed-attempt input files without changing their bytes."""
    records = []
    for row in original["exact_copy_allowlist"]:
        actual = base.file_record(row["remote_path"])
        if actual != {"path": row["remote_path"], "sha256": row["sha256"], "bytes": row["bytes"]}:
            raise ValueError("an original staging input changed after the preserved failure")
        records.append(actual)
    return records


def cloned_roots(base, original):
    """Verify the already completed D28/CleanRL clones and dedicated CleanRL bare repository read-only."""
    roots = {"numerical": base.git_identity(original["native_numerical_checkout_required"]["path"], REMOTE_COMMANDS, base.SOURCE_COMMIT),
        "cleanrl": base.git_identity(original["native_cleanrl_checkout_required"]["path"], REMOTE_COMMANDS, base.CLEANRL_COMMIT)}
    bare = str(Path(original["remote_inputs_directory"]) / "cleanrl_repository.git")
    if (base.execute(["git", "--git-dir", bare, "rev-parse", "--is-bare-repository"], REMOTE_COMMANDS) != "true"
            or base.execute(["git", "--git-dir", bare, "rev-parse", "HEAD"], REMOTE_COMMANDS) != base.CLEANRL_COMMIT):
        raise ValueError("the preserved CleanRL bare repository changed")
    return roots


def effective_plan(base, repair, original):
    """Change only new operational output and asset-record paths in the existing prepare command."""
    plan = json.loads(json.dumps(original))
    plan.update(remote_packet_directory=repair["native_packet_directory"], remote_data_unit=repair["native_data_unit"])
    parameters = plan["native_prepare_parameters"]
    parameters.update({"output-dir": repair["native_packet_directory"], "asset-record": repair["asset_record"]})
    plan["native_prepare_command"] = [PYTHON, "-I", str(Path(original["native_helper_root"]) / base.PREFIX / "aces_slurm_site_validation.py"),
        "prepare", *[part for key, value in parameters.items() for part in ("--" + key, value)]]
    return plan


def returned_files(base, repair, original):
    """Bound all new packet records and repair observations to explicit remote paths."""
    plan = effective_plan(base, repair, original)
    root = Path(repair["repair_directory"])
    return base.prepared_files(plan) + [("repair/" + name, root / name) for name in
        ("assets_before.json", "assets_after.json", "inputs_before.json", "inputs_after.json", "roots_before.json", "roots_after.json",
         "asset_copy_receipt.json", "asset_record.json")]


def repair_timeout(signum, frame):
    """Stop only this own metadata process at its fixed bounded repair deadline."""
    raise TimeoutError("exact asset-view metadata preparation exceeded 540 seconds")


def remote_prepare(payload):
    """Preserve existing evidence, copy a fresh exact view and invoke only metadata preparation."""
    signal.signal(signal.SIGALRM, repair_timeout)
    signal.alarm(540)
    try:
        if (os.getuid() != 56626 or pwd.getpwuid(os.getuid()).pw_name != "u.sl359075"
                or socket.gethostname().split(".")[0] != "alogin1" or str(Path(sys.executable).absolute()) != PYTHON):
            raise ValueError("asset-view preparation requires actual ACES owner, login host and project Python")
        actual_command = Path("/proc/self/cmdline").read_bytes().split(b"\0")
        if digest(actual_command[actual_command.index(b"-c") + 1]) != digest(BOOTSTRAP.encode()):
            raise ValueError("the actual remote repair wrapper differs from the reviewed bootstrap")
        base, repair, original, observation = checked_payload(payload)
        root = Path(repair["repair_directory"])
        if any(Path(path).exists() or Path(path).is_symlink() for path in
                (root, repair["native_packet_directory"], repair["native_data_unit"])):
            raise ValueError("the new asset view, packet and data targets must be fresh")
        inputs_before = input_records(base, original)
        roots_before = {"old": base.old_roots(original, REMOTE_COMMANDS), "cloned": cloned_roots(base, original)}
        helper = base.helper_module(original)
        mapped_before = base.mapping_check(helper, original)
        manifest_path, expected = expected_manifest(base, original, observation)
        assets_before = asset_roster(base, BASE / "assets")
        if assets_before != observation["files"]:
            raise ValueError("the actual original asset roster changed after its read-only observation")
        root.mkdir()
        for name, value in (("assets_before.json", assets_before), ("inputs_before.json", inputs_before), ("roots_before.json", roots_before)):
            base.write_record(root / name, value)
        copied = copy_exact_assets(base, BASE / "assets", repair["asset_view"], expected, assets_before)
        base.write_record(root / "asset_copy_receipt.json", {"kind": "exclusive_physical_exact_asset_view", "files": copied,
            "source_root": str(BASE / "assets"), "view_root": repair["asset_view"], "expected_manifest_sha256": MANIFEST_SHA,
            "source_files_changed": False, "asset_symlinks_created": 0, "asset_hardlinks_created": 0})
        base.write_record(repair["asset_record"], {"root": repair["asset_view"], "manifest": manifest_path, "sha256": MANIFEST_SHA})
        plan = effective_plan(base, repair, original)
        environment = base.native_environment(plan)
        environment["XDG_DATA_HOME"] = repair["asset_view"]
        base.execute(plan["native_prepare_command"], REMOTE_COMMANDS, environment=environment, timeout=240)
        os.environ.clear()
        os.environ.update(environment)
        directory, unit, actual = helper.prepared(Path(repair["native_packet_directory"]))
        helper.checked_software(actual)
        mapped_after = base.mapping_check(helper, original)
        roots_after = {"old": base.old_roots(original, REMOTE_COMMANDS), "cloned": cloned_roots(base, original)}
        inputs_after, assets_after = input_records(base, original), asset_roster(base, BASE / "assets")
        if (roots_before != roots_after or inputs_before != inputs_after or assets_before != assets_after or mapped_before != mapped_after
                or any(name in sys.modules for name in base.NUMERICAL_MODULES)):
            raise ValueError("original assets, inputs, roots or mappings changed during exact-view preparation")
        for name, value in (("assets_after.json", assets_after), ("inputs_after.json", inputs_after), ("roots_after.json", roots_after)):
            base.write_record(root / name, value)
        files = [{"relative_path": relative, **base.file_record(path)} for relative, path in returned_files(base, repair, original)]
        if any(row["bytes"] > base.MAX_RETURN_FILE for row in files):
            raise ValueError("a returned repair artifact exceeds its explicit size bound")
        receipt = {"schema_version": 1, "kind": "aces_actual_exact_asset_view_metadata_preparation", "passed": True,
            "at_utc": datetime.now(timezone.utc).isoformat(), "repair_plan_sha256": REPAIR_SHA,
            "driver_sha256": digest(payload["driver_source"].encode()), "base_driver_sha256": BASE_SHA,
            "asset_observation_sha256": OBSERVATION_SHA, "original_staging_plan_sha256": ORIGINAL_SHA,
            "asset_view": repair["asset_view"], "expected_manifest_sha256": MANIFEST_SHA, "copied_file_count": len(copied),
            "mapping": mapped_after, "commands": REMOTE_COMMANDS, "prepared_files": list(files),
            "old_assets_inputs_roots_unchanged": True, "numerical_imports": [], "jobs_submitted": 0,
            "workers_permitted": False, "case_credit": 0, "idea_credit": 0, "admission_credit": 0}
        final = root / "native_asset_view_preparation_receipt.json"
        base.write_record(final, receipt)
        files.append({"relative_path": "repair/native_asset_view_preparation_receipt.json", **base.file_record(final)})
        sys.stdout.buffer.write((json.dumps({"receipt": receipt, "files": files}) + "\n").encode())
        for row in files:
            base.send_file(sys.stdout.buffer, row["path"], row)
        sys.stdout.buffer.flush()
    finally:
        signal.alarm(0)


def accept_response(base, path, directory, repair, original, source_hash):
    """Save only exact new metadata files and bind their receipt to this reviewed repair."""
    allowed = {relative: str(remote) for relative, remote in returned_files(base, repair, original)}
    allowed["repair/native_asset_view_preparation_receipt.json"] = str(Path(repair["repair_directory"]) / "native_asset_view_preparation_receipt.json")
    with base.ordinary(path).open("rb") as stream:
        line = stream.readline(base.MAX_HEADER + 1)
        if len(line) > base.MAX_HEADER or not line.endswith(b"\n"):
            raise ValueError("the returned asset-view metadata header is absent or oversized")
        header = json.loads(line)
        if set(header) != {"receipt", "files"}:
            raise ValueError("an exact repair receipt and file roster are required")
        receipt, rows = header["receipt"], header["files"]
        if (receipt.get("kind") != "aces_actual_exact_asset_view_metadata_preparation" or receipt.get("passed") is not True
                or receipt.get("repair_plan_sha256") != REPAIR_SHA or receipt.get("driver_sha256") != source_hash
                or receipt.get("base_driver_sha256") != BASE_SHA or receipt.get("expected_manifest_sha256") != MANIFEST_SHA
                or receipt.get("old_assets_inputs_roots_unchanged") is not True or receipt.get("copied_file_count") != 4531
                or receipt.get("asset_view") != repair["asset_view"] or receipt.get("numerical_imports") != []
                or receipt.get("workers_permitted") is not False
                or any(type(receipt.get(name)) is not int or receipt[name] != 0 for name in
                    ("jobs_submitted", "case_credit", "idea_credit", "admission_credit"))):
            raise ValueError("the returned record does not prove this exact zero-job asset-view metadata preparation")
        if len(rows) != len(allowed) or {row["relative_path"] for row in rows} != set(allowed):
            raise ValueError("the returned repair artifact roster is incomplete or duplicated")
        for row in rows:
            if (set(row) != {"relative_path", "path", "sha256", "bytes"} or allowed[row["relative_path"]] != row["path"]
                    or type(row["bytes"]) is not int or not 0 <= row["bytes"] <= base.MAX_RETURN_FILE):
                raise ValueError("a returned repair artifact changed its exact path or size")
            base.receive_file(stream, directory / "received" / row["relative_path"], row)
        if stream.read(1):
            raise ValueError("the asset-view response has unregistered trailing bytes")
    if json.loads((directory / "received/repair/native_asset_view_preparation_receipt.json").read_bytes()) != receipt:
        raise ValueError("the actual returned repair receipt differs from its header")
    return receipt


def local_execute(payload, base, repair, original, directory):
    """Use one existing-master-only SSH call and retain raw evidence without automatic retry."""
    if socket.gethostname().split(".")[0] != "portal05":
        raise ValueError("the separate asset-view preparation must start only from portal05")
    directory = Path(directory).absolute()
    base.ordinary(directory.parent, directory=True)
    directory.mkdir()
    source_hash = digest(payload["driver_source"].encode())
    command = [*base.SSH_OPTIONS, "exec " + shlex.join([PYTHON, "-I", "-c", BOOTSTRAP, source_hash])]
    header = (json.dumps(payload) + "\n").encode()
    if len(header) > base.MAX_HEADER:
        raise ValueError("the fixed repair metadata payload exceeds its transport bound")
    base.write_record(directory / "invocation.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
        "repair_plan_sha256": REPAIR_SHA, "driver_sha256": source_hash, "bootstrap_sha256": digest(BOOTSTRAP.encode()),
        "base_driver_sha256": BASE_SHA, "ssh_options": base.SSH_OPTIONS, "master_separately_checked": False,
        "authentication_retry_permitted": False, "jobs_submitted": 0, "timeout_seconds": 600})
    (directory / "executed_driver.py").write_text(payload["driver_source"])
    timeout = False
    with (directory / "stdout.raw").open("xb") as output, (directory / "stderr.raw").open("xb") as error:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output, stderr=error)
        try:
            process.communicate(input=header, timeout=600)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            timeout = True
    base.write_record(directory / "transport_result.json", {"returncode": process.returncode, "timeout": timeout,
        "stdout": base.file_record(directory / "stdout.raw"), "stderr": base.file_record(directory / "stderr.raw"),
        "automatic_retry_permitted": False})
    if process.returncode or timeout:
        raise ValueError("asset-view preparation did not complete; exact raw failure evidence is preserved")
    return accept_response(base, directory / "stdout.raw", directory, repair, original, source_hash)


def main():
    """Validate and describe the exact repair by default; execute only when root explicitly invokes it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--receipt-directory", type=Path)
    arguments = parser.parse_args()
    payload, base, repair, original, observation = local_payload(arguments.plan)
    plan = effective_plan(base, repair, original)
    if arguments.execute:
        if arguments.receipt_directory is None:
            raise ValueError("execution requires a new absolute metadata receipt directory")
        result = local_execute(payload, base, repair, original, arguments.receipt_directory)
        print(json.dumps({"passed": result["passed"], "receipt_directory": str(arguments.receipt_directory),
            "native_packet_directory": repair["native_packet_directory"], "jobs_submitted": 0}))
    else:
        print(json.dumps({"repair_plan_sha256": REPAIR_SHA, "expected_file_count": 4531,
            "original_asset_files_preserved": 4533, "asset_view": repair["asset_view"], "copy_mode": "new exclusive physical files",
            "native_prepare_command": plan["native_prepare_command"], "old_inputs_and_roots_reused_read_only": True,
            "ssh_options": base.SSH_OPTIONS, "master_separately_checked": False, "network_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
