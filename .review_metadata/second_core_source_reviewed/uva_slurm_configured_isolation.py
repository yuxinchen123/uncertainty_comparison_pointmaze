"""Qualify one UVA device on an explicit configuration-trust basis.

Logical scheduler indices never select a physical device. The frozen plan leaves
that identity unresolved; a current process records it separately after CUDA and
JAX agree. This module grants no source, worker, or scientific admission.
"""

import csv
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import selectors
import stat
import subprocess
import sys
import time
import uuid


PROFILE = {
    "schema_version": 1,
    "name": "uva-jaguar03-autodetect-nvml-configured-isolation-version-1",
    "cluster": "uva", "hostname": "jaguar03", "username": "sl5nw", "uid": 2618919,
    "configuration_service_name": "slurm", "configuration_service_uid": 64030,
    "model": "NVIDIA RTX A4500", "memory_mib": 20470, "driver": "580.159.04",
    "gpu_count": 1, "physical_gres_mapping_verified": False,
    "kernel_device_enforcement_verified": False, "installed_binary_matches_upstream_verified": False,
    "isolation_basis": "explicit trust in the five reviewed administrator-managed UVA configuration files",
}
CAPABILITY = "unresolved_plan_original_selector_current_cuda_jax_binding_v1"
HELPERS = ("uva_slurm_configured_isolation.py", "device_memory_limit.py", "h100_nvl_slurm_site_validation.py",
    "uva_active_configuration_observation.py", "passive_installed_slurm_observation.py")
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
CONFIG_FILES = {
    "/etc/slurm/slurm.conf": "213efdf97aeae42c5e4697171a49d195a801a67c232293cf0c0b04e15a57c47a",
    "/etc/slurm/nodes.conf": "1c9dc0acd626a1f794feddae773da4295d5807e06e872a9330a722eac2e59a5c",
    "/etc/slurm/partitions.conf": "161b848c4ee1281341d7791391e516df99fb4a98ea16d07d57355b4553871d8c",
    "/etc/slurm/gres.conf": "1d7d1d7dd21d54a549b74a80ebaf0fe3b5016fa5ae702de945b81a43b9b76af7",
    "/etc/slurm/cgroup.conf": "b2d8db246a72c9463d782c8c25b0f56ab5940e3fcba194342440d7c898caadc9",
}
BOUNDARY = {"physical_gres_mapping_verified": False, "kernel_device_enforcement_verified": False,
    "installed_binary_matches_upstream_verified": False, "worker_admitted": False,
    "admission_granted": False, "case_credit": 0, "idea_credit": 0, "site_qualification_credit": 0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def adjacent(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("uva_configuration_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def exact(value, expected):
    return type(value) is type(expected) and (set(value) == set(expected)
        and all(exact(value[k], expected[k]) for k in expected) if isinstance(expected, dict)
        else len(value) == len(expected) and all(exact(a, b) for a, b in zip(value, expected))
        if isinstance(expected, list) else value == expected)


def require_profile(value):
    require(exact(value, PROFILE), "the complete separate UVA configured-isolation profile is required")
    return value


def unresolved_identity():
    return {"schema_version": 1, "kind": "unresolved_owned_cuda_device", "gpu_uuid": None,
        "pci_address": None, "physical_minor": None}


def ordinary(path):
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path and path.is_file(), "evidence must use an ordinary exact file path")
    return path


def receipt(path):
    path = ordinary(path)
    return {"path": str(path), "sha256": digest(path.read_bytes())}


def read_receipt(value):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}
        and receipt(value["path"]) == value, "evidence bytes or path changed")
    return adjacent("device_memory_limit").json_document(Path(value["path"]).read_bytes())


def publish(path, value):
    """Publish one durable record without replacing any existing attempt."""
    path = Path(path).absolute()
    require(path.parent.resolve(strict=True) == path.parent, "record directory must be an ordinary exact path")
    with path.open("xb") as stream:
        stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return receipt(path)


def validate_plan(plan, consumer=None):
    value = plan.get("configured_isolation")
    require(isinstance(value, dict) and set(value) == {"schema_version", "profile", "capability", "consumer",
        "resources", "physical_identity", "owned_inputs_path"}, "explicit unresolved configuration-trust plan required")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1
        and value["capability"] == CAPABILITY, "unsupported configured-isolation capability")
    require_profile(value["profile"])
    require(value["consumer"] in {"selected_measurement", "component_profile"}
        and (consumer is None or value["consumer"] == consumer), "unsupported configured-isolation consumer")
    require(exact(value["physical_identity"], unresolved_identity()) and plan.get("gpu_uuid", "absent") is None
        and plan.get("hardware_class") == "jaguar03_rtx_a4500", "the immutable plan must leave physical identity unresolved")
    require(exact(value["resources"], {"cpus": 8, "host_memory_gib": 64}), "only the reviewed single-card resource shape is supported")
    require(type(plan["unit_id"]) is str and Path(plan["unit_id"]).name == plan["unit_id"]
        and plan["unit_id"] not in {"", ".", ".."}, "unit identity must be one literal directory name")
    directory = Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
    require(value["owned_inputs_path"] == str(directory / "owned_allocation_inputs.json"), "owned launch binding must be this exact unit")
    require((value["consumer"] == "selected_measurement" and plan.get("kind") == "selected_configuration_measurement"
        and plan.get("mode") in {"correctness", "confirmation"}
        or value["consumer"] == "component_profile" and plan.get("kind") == "selected_configuration_profile" and plan.get("mode") in {None, "profile"})
        and not any(key in plan for key in ("device_memory_limit_protocol", "final_protocol", "final_confirmation_protocol", "completed_study", "completed_study_manifest", "fixed_total_scaling", "scaling_protocol")),
        "packing, instrumented, final, scaling and portable consumers need separate explicit support")
    return value


def source_identity(plan):
    """Bind this staged core and its direct dependencies to exact committed helper bytes."""
    value = plan["helpers"]
    require(re.fullmatch(r"[0-9a-f]{40}", value["source_commit"]), "a full helper commit is required")
    root = Path(value["source_root"])
    require(root.resolve(strict=True) == root, "helper Git root cannot use path aliases")
    for name in HELPERS:
        relative = PREFIX + name
        expected = value["files"].get(relative)
        raw = subprocess.check_output(["git", "-C", str(root), "show", value["source_commit"] + ":" + relative])
        require(expected == digest(raw) == receipt(root / relative)["sha256"]
            == receipt(Path(__file__).with_name(name))["sha256"], "a staged or committed qualifier dependency changed")
    return value


def owned_inputs(plan):
    """Read only durable launch records; no job number is guessed from a queue."""
    options = validate_plan(plan)
    path = Path(options["owned_inputs_path"])
    value = read_receipt(receipt(path))
    require(set(value) == {"schema_version", "kind", "job_id", "plan", "ledger", "submissions", "submission", "release", "submission_row"}
        and type(value["schema_version"]) is int and value["schema_version"] == 1
        and value["kind"] == "uva_configured_isolation_owned_inputs", "unsupported owned launch input schema")
    job = value["job_id"]
    require(isinstance(job, str) and re.fullmatch(r"[1-9][0-9]*", job), "an exact submitted job is required")
    unit = Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
    require(value["plan"] == receipt(unit / "measurement_plan.json")
        and read_receipt(value["plan"]) == plan, "the launch binding changed the immutable plan")
    script = str(Path(plan["campaign_run_dir"]) / "remote_setup" / plan["unit_id"] / "job.sbatch")
    row = [job, plan["kind"], plan["helpers"]["source_commit"], script]
    require(value["submission_row"] == row and all(type(x) is str for x in value["submission_row"]), "the exact submission row changed")
    for key, filename, member in (("ledger", "submitted_jobids_uva.txt", job), ("submissions", "submissions.tsv", "\t".join(row))):
        item = value[key]
        snapshot_name = "configured_isolation_owned_job_ids.txt" if key == "ledger" else "configured_isolation_owned_submissions.tsv"
        require(set(item) == {"snapshot", "source_path"} and item["source_path"] == str(Path(plan["campaign_run_dir"]) / "slurm" / filename)
            and item["snapshot"]["path"] == str(unit / snapshot_name)
            and receipt(item["snapshot"]["path"]) == item["snapshot"], "owned ledger snapshot or source changed")
        for candidate in (item["snapshot"]["path"], item["source_path"]):
            lines = ordinary(candidate).read_text().splitlines()
            require(lines.count(member) == 1 and (key != "ledger" or len(lines) == len(set(lines))), "missing or duplicate owned submission")
            if key == "submissions":
                require([line for line in lines if line.split("\t", 1)[0] == job] == [member], "conflicting rows for the exact owned job")
    submission, release = read_receipt(value["submission"]), read_receipt(value["release"])
    for document, kind in ((submission, "owned_uva_configured_isolation_submission"), (release, "owned_uva_configured_isolation_release")):
        require(type(document.get("schema_version")) is int and document["schema_version"] == 1
            and document.get("kind") == kind
            and document.get("job_id") == job and document.get("cluster") == "uva"
            and document.get("plan_sha256") == value["plan"]["sha256"] and document.get("data_unit") == str(unit)
            and type(document.get("returncode")) is int and document["returncode"] == 0, "launch receipt identity changed")
    command = submission.get("held_command")
    require(isinstance(command, list) and command and command[0] == "sbatch" and "--hold" in command
        and "--no-requeue" in command and command[-1] == script and submission.get("submission_row") == row
        and release.get("command") == ["scontrol", "release", job]
        and release.get("submission_receipt_sha256") == value["submission"]["sha256"], "held submission/release chain changed")
    return value


def config_file(path, observation=None):
    """Reuse the reviewed pinned ordinary reader; never retain private configuration bytes."""
    path = Path(path)
    require(str(path) in CONFIG_FILES, "only one of the five reviewed configuration paths may be read")
    row = {} if observation is None else observation
    row.update(path=str(path), read_status="started", read_trace={})
    reader = adjacent("uva_active_configuration_observation")
    try:
        raw = reader.pinned_configuration_bytes(path, row["read_trace"], 32768)
        trace = row["read_trace"]
        before = trace["descriptor_before"]
        row.update(uid=before["owner_uid"], mode=before["mode"], regular_file=before["regular_file"],
            symlink=False, sha256=digest(raw), settings=[], includes=[])
        row["directory_chain"] = [{"path": x["path"], "uid": x["owner_uid"], "mode": x["mode"], "inode": x["inode"],
            "device": x["device"], "directory": x["directory"], "symlink": False} for x in trace["pinned_directory_chain"]]
        row["directory_chain_after"] = []
        for x in row["directory_chain"]:
            info = Path(x["path"]).lstat()
            row["directory_chain_after"].append({"path": x["path"], "uid": info.st_uid, "mode": stat.S_IMODE(info.st_mode),
                "inode": info.st_ino, "device": info.st_dev, "directory": stat.S_ISDIR(info.st_mode), "symlink": stat.S_ISLNK(info.st_mode)})
        require(trace["complete_file"] is True and trace["inode_changed_during_read"] is False
            and trace["path_replaced_during_read"] is False, "configuration changed or exceeded its read limit")
        # A changed file may contain private fields. Hash it, but disclose no excerpts from it.
        require(row["sha256"] == CONFIG_FILES[str(path)], "reviewed configuration bytes changed")
        parser = adjacent("passive_installed_slurm_observation")
        row.update(settings=parser.setting_rows(raw, parser.FILE_SETTINGS), includes=parser.include_rows(raw), read_status="read")
    except BaseException as error:
        row.update(read_status="failed", error_type=type(error).__name__)
        raise
    return row


def validate_configuration(value):
    require(value.get("service") == {"name": "slurm", "uid": 64030, "gid": 64030}, "the exact configuration service identity changed")
    require(value.get("cache_absent") is True and list(value["files"]) == list(CONFIG_FILES), "configuration scope or competing cache changed")
    for path, expected in CONFIG_FILES.items():
        row = value["files"][path]
        require(row["path"] == path and row.get("regular_file") is True and row.get("symlink") is False
            and type(row["uid"]) is int and row["uid"] == 64030 and type(row["mode"]) is int and row["mode"] == 0o644
            and row["sha256"] == expected and row.get("read_status") == "read" and "raw" not in row
            and "raw_base64" not in row, "one of the five exact reviewed configuration files changed")
        trace = row["read_trace"]
        require(trace["complete_file"] is True and trace["inode_changed_during_read"] is False
            and trace["path_replaced_during_read"] is False and trace["descriptor_before"] == trace["descriptor_after"]
            and type(trace["bytes_read"]) is int and trace["bytes_read"] == trace["descriptor_before"]["byte_count"]
            and trace["descriptor_before"]["path"] == path and trace["descriptor_before"]["owner_uid"] == row["uid"]
            and trace["descriptor_before"]["mode"] == row["mode"] and trace["descriptor_before"]["regular_file"] is True,
            "a complete stable ordinary-file read is required")
        require(isinstance(row["settings"], list) and all(set(x) == {"line", "name", "value"}
            and type(x["line"]) is int and x["line"] > 0 and type(x["name"]) is str
            and x["name"].lower() in {name.lower() for name in adjacent("passive_installed_slurm_observation").FILE_SETTINGS}
            and type(x["value"]) is str for x in row["settings"]), "only the reviewed nonsecret settings may be retained")
        expected_includes = [{"line": 121, "token": "/etc/slurm/nodes.conf"}, {"line": 122, "token": "/etc/slurm/partitions.conf"}] if path == "/etc/slurm/slurm.conf" else []
        require(row["includes"] == expected_includes, "the reviewed exact include graph changed")
        chain = row["directory_chain"]
        require(chain == row["directory_chain_after"] and [x["path"] for x in chain] == ["/", "/etc", "/etc/slurm"]
            and [x["uid"] for x in chain] == [0, 0, 64030]
            and all(type(x["uid"]) is int and type(x["mode"]) is int and x["mode"] == 0o755
                and x["directory"] is True and x["symlink"] is False and type(x["inode"]) is int and x["inode"] > 0 for x in chain),
            "the exact protected configuration directory chain changed")
    # Exact full-file hashes preserve the reviewed include graph and all duplicate/override decisions.
    return value


def bounded_query(command, configuration=False, byte_limit=32768, timeout=20):
    """Bound both pipes and time; retain only nonsecret controller excerpts on success."""
    value = {"command": command, "returncode": None, "capture_status": "started"}
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    selector, process = selectors.DefaultSelector(), None
    deadline = time.monotonic() + timeout
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        for name in buffers:
            stream = getattr(process, name); os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        value["capture_status"] = "completed"
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                value["capture_status"] = "timeout"; break
            for event, unused in selector.select(remaining):
                stream, name = event.fileobj, event.data
                try:
                    raw = os.read(stream.fileno(), min(65536, byte_limit + 1 - len(buffers[name])))
                except BlockingIOError:
                    continue
                if not raw:
                    selector.unregister(stream); continue
                buffers[name].extend(raw)
                if len(buffers[name]) > byte_limit:
                    value["capture_status"] = "response_limit_exceeded"; break
            if value["capture_status"] != "completed":
                break
        if value["capture_status"] != "completed" and process.poll() is None:
            process.kill()
        try:
            value["returncode"] = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            value["capture_status"] = "timeout"; process.kill(); value["returncode"] = process.wait(timeout=5)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        value.update(capture_status="failed", error_type=type(error).__name__)
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill(); process.wait(timeout=5)
            for name in buffers:
                getattr(process, name).close()
        selector.close()
    for name, raw in buffers.items():
        value[name + "_full_sha256"] = digest(raw)
        value[name + "_captured_bytes"] = len(raw)
    if configuration:
        value["stdout"], value["stderr"] = "", "" if not buffers["stderr"] else "controller stderr omitted; exact hash retained"
        if value["capture_status"] == "completed" and value["returncode"] == 0 and not buffers["stderr"]:
            parser = adjacent("passive_installed_slurm_observation")
            try:
                rows = parser.setting_rows(bytes(buffers["stdout"]), parser.SETTINGS, comments=False)
                value["stdout"] = "".join(x["name"] + " = " + x["value"] + "\n" for x in rows)
            except (UnicodeError, ValueError):
                value["capture_status"] = "unsupported_configuration_text"
    else:
        for name, raw in buffers.items():
            value[name] = raw.decode("utf-8", errors="replace")
    return value


def collect_metadata(plan, directory):
    """Preserve raw own-ID evidence before any physical-device operation or rejection."""
    directory = Path(directory)
    result = {"schema_version": 1, "kind": "uva_configured_isolation_metadata", "profile": PROFILE,
        "capability": CAPABILITY, **BOUNDARY}
    try:
        require(not any(name.split(".", 1)[0] in {"jax", "jaxlib", "numpy", "torch"} for name in sys.modules),
            "metadata admission must precede numerical imports")
        validate_plan(plan); result["source"] = source_identity(plan)
        result["inputs"] = owned_inputs(plan); result["inputs_receipt"] = receipt(plan["configured_isolation"]["owned_inputs_path"])
        helper = adjacent("device_memory_limit")
        result["process"] = helper.process_identity()
        result["environment"] = {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "SLURM_CONF", "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_CPUS_PER_TASK") if k in os.environ}
        job, step = result["environment"].get("SLURM_JOB_ID", ""), result["environment"].get("SLURM_STEP_ID", "")
        require(job == result["inputs"]["job_id"], "actual own job differs from the durable submitted ID")
        helper.v2_paths(job, step)
        require(result["process"]["hostname"] == "jaguar03" and type(result["process"]["uid"]) is int
            and result["process"]["uid"] == 2618919, "fixed own UID and host must pass before scheduler queries")
        wrapped = {"ownership_profile": helper.UVA_CGROUP_V2_OWNERSHIP_PROFILE, "device_memory_limit_protocol": {"allocation_kind": "slurm", "cluster": "uva"}}
        result["cgroup_capture_path"] = str(directory / "configured_isolation_process.json")
        result["cgroup"] = helper.capture_v2_ownership(wrapped, result["process"], job, step, Path(result["cgroup_capture_path"]))
        for key, command in (("job", ["scontrol", "show", "job", "-d", job]), ("step", ["scontrol", "show", "step", "-d", job + "." + step]),
                ("running_configuration", ["scontrol", "show", "config"])):
            result[key] = {"command": command, "status": "started"}
            result[key] = bounded_query(command, configuration=key == "running_configuration")
        account = pwd.getpwnam("slurm")
        result["configuration"] = {"service": {"name": account.pw_name, "uid": account.pw_uid, "gid": account.pw_gid},
            "cache_absent": not os.path.lexists("/var/spool/slurmd/conf-cache"), "files": {}}
        for path in CONFIG_FILES:
            result["configuration"]["files"][path] = {}
            config_file(path, result["configuration"]["files"][path])
        validate_metadata(plan, result)
    except BaseException as error:
        result["capture_error"] = {"type": type(error).__name__, "error": str(error)}
        raise
    finally:
        publish(directory / "configured_isolation_metadata.json", result)
    return result


def resources(text):
    pairs = [x.split("=", 1) for x in text.split(",")]
    require(all(len(x) == 2 for x in pairs) and len({x[0] for x in pairs}) == len(pairs), "duplicate or invalid scheduler resource")
    return dict(pairs)


def validate_metadata(plan, value):
    options = validate_plan(plan)
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1 and value.get("kind") == "uva_configured_isolation_metadata"
        and value.get("capability") == CAPABILITY and "capture_error" not in value, "partial or unsupported configuration-trust metadata")
    require_profile(value["profile"])
    require(all(exact(value.get(k), v) for k, v in BOUNDARY.items()), "configuration-trust metadata cannot grant admission")
    require(value["source"] == source_identity(plan) and value["inputs"] == owned_inputs(plan)
        and value["inputs_receipt"] == receipt(options["owned_inputs_path"]), "source or owned launch binding changed")
    process, environment, inputs = value["process"], value["environment"], value["inputs"]
    job, step = environment.get("SLURM_JOB_ID", ""), environment.get("SLURM_STEP_ID", "")
    helper = adjacent("device_memory_limit"); helper.v2_paths(job, step)
    require(process["hostname"] == "jaguar03" and type(process["uid"]) is int and process["uid"] == 2618919
        and job == inputs["job_id"] and environment.get("SLURM_CONF") == "/etc/slurm/slurm.conf"
        and environment.get("SLURM_GPUS_ON_NODE") == "1" and environment.get("SLURM_CPUS_PER_TASK") == "8"
        and re.fullmatch(r"(?:0|[1-9][0-9]*)", environment.get("SLURM_JOB_GPUS", ""))
        and environment.get("SLURM_JOB_GPUS") == environment.get("SLURM_STEP_GPUS")
        and isinstance(environment.get("CUDA_VISIBLE_DEVICES"), str) and re.fullmatch(r"(?:0|[1-9][0-9]*|GPU-[0-9a-fA-F-]{36})", environment["CUDA_VISIBLE_DEVICES"])
        and len(process["cpu_affinity"]) == len(set(process["cpu_affinity"])) == 8
        and all(type(cpu) is int and cpu >= 0 for cpu in process["cpu_affinity"]), "the exact own single-GPU numeric allocation or original selector changed")
    helper.validate_v2_ownership(value["cgroup"], process, job, step)
    for key, command in (("job", ["scontrol", "show", "job", "-d", job]), ("step", ["scontrol", "show", "step", "-d", job + "." + step]),
            ("running_configuration", ["scontrol", "show", "config"])):
        row = value[key]
        require(row["command"] == command and row.get("capture_status") == "completed"
            and type(row["returncode"]) is int and row["returncode"] == 0 and row["stderr"] == "", "actual own scheduler query failed or changed")
    field = helper.scheduler_field
    row, task = value["job"]["stdout"], value["step"]["stdout"]
    require(field(row, "JobId") == job and field(row, "UserId") == "sl5nw(2618919)" and field(row, "JobState") == "RUNNING"
        and field(row, "Account") == "sl5nw_base" and field(row, "Partition") == "gpu" and field(row, "QOS") == "csresnolim"
        and field(row, "NodeList") == "jaguar03" and field(row, "NumNodes") == "1" and field(row, "NumCPUs") == "8"
        and field(row, "CPUs/Task") == "8" and field(row, "NumTasks") == "1" and field(row, "TresPerNode") == "gres/gpu:nvidia_rtx_a4500:1"
        and field(row, "Requeue") == "0" and field(row, "Restarts") == "0", "actual job owner or reviewed typed resource shape changed")
    for text, key in ((row, "AllocTRES"), (row, "ReqTRES"), (task, "TRES")):
        actual = resources(field(text, key))
        require(actual.get("cpu") == "8" and actual.get("node") == "1" and actual.get("gres/gpu") == "1"
            and all(k == "gres/gpu:nvidia_rtx_a4500" and v == "1" for k, v in actual.items() if k.startswith("gres/gpu:")), "one exact A4500/eight-CPU scheduler allocation is required")
        require(actual.get("mem") in {"64G", "65536M"}, "the actual host memory allocation changed")
    require(field(task, "StepId") == job + "." + step and field(task, "State") == "RUNNING"
        and field(task, "UserId") in {"2618919", "sl5nw(2618919)"} and field(task, "NodeList") == "jaguar03"
        and field(task, "Tasks") == "1", "actual numeric step ownership changed")
    running = value["running_configuration"]["stdout"]
    for key, expected in (("SLURM_CONF", "/etc/slurm/slurm.conf"), ("SlurmUser", "slurm(64030)"), ("ProctrackType", "proctrack/cgroup"), ("GresTypes", "gpu")):
        found = re.findall(r"^\s*" + re.escape(key) + r"\s*=\s*(.*?)\s*$", running, re.M)
        require(found == [expected], "controller configuration identity changed")
    found = re.findall(r"^\s*TaskPlugin\s*=\s*(.*?)\s*$", running, re.M)
    require(len(found) == 1 and len(found[0].split(",")) == 2 and set(found[0].split(",")) == {"task/cgroup", "task/affinity"}, "only the two reviewed task plugins are supported")
    validate_configuration(value["configuration"])
    return value


def pci(value):
    require(isinstance(value, str) and re.fullmatch(r"(?:0000)?[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", value), "invalid physical PCI address")
    return value.lower()[-12:]


def native_identity(device, observation=None):
    """Query only the already mapped driver, preserving the original CUDA selector."""
    require(device.platform == "gpu" and device.device_kind == PROFILE["model"]
        and type(device.local_hardware_id) is int and device.local_hardware_id == 0, "one actual local JAX A4500 is required")
    value = {} if observation is None else observation
    value.update(pid=os.getpid(), jax_local_hardware_id=device.local_hardware_id, jax_device_kind=device.device_kind,
        jax_identity_basis="sole local JAX GPU, sole CUDA-visible ordinal, and that device's current CUDA context")
    with Path("/proc/self/maps").open() as stream:
        maps = stream.read(1024 * 1024 + 1)
    require(len(maps.encode()) <= 1024 * 1024, "own mapped-image metadata exceeds its read limit")
    selected = "".join(line + "\n" for line in maps.splitlines() if line.split() and Path(line.split()[-1]).name.startswith("libcuda.so"))
    paths = {line.split()[-1] for line in selected.splitlines()}
    value.update(mapped_driver_bytes=selected, mapped_driver_sha256=digest(selected.encode()))
    require(len(paths) == 1, "one actually mapped CUDA driver is required")
    path, = paths
    require(path.endswith("." + PROFILE["driver"]), "actual CUDA driver differs from the reviewed UVA target")
    value.update(driver_library=path, driver_library_sha256=receipt(path)["sha256"])
    driver = ctypes.CDLL(path, mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
    signatures = {"cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)], "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDeviceGetUuid_v2": [ctypes.c_void_p, ctypes.c_int], "cuDeviceGetPCIBusId": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
        "cuCtxGetCurrent": [ctypes.POINTER(ctypes.c_void_p)], "cuCtxGetDevice": [ctypes.POINTER(ctypes.c_int)]}
    for name, arguments in signatures.items():
        function = getattr(driver, name); function.argtypes = arguments; function.restype = ctypes.c_int
    count, handle, current_device, context = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_void_p()
    codes = {"cuDeviceGetCount": driver.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGet": driver.cuDeviceGet(ctypes.byref(handle), 0)}
    codes["cuCtxGetCurrent"] = driver.cuCtxGetCurrent(ctypes.byref(context))
    codes["cuCtxGetDevice"] = driver.cuCtxGetDevice(ctypes.byref(current_device))
    value.update(api_return_codes=codes, cuda_device_count=count.value, cuda_device_handle=handle.value,
        current_context=context.value, current_context_device=current_device.value)
    require(all(type(x) is int and x == 0 for x in codes.values()) and count.value == 1
        and context.value and current_device.value == handle.value, "one visible CUDA device and its existing current context are required")
    raw_uuid, bus = (ctypes.c_ubyte * 16)(), ctypes.create_string_buffer(32)
    codes["cuDeviceGetUuid_v2"] = driver.cuDeviceGetUuid_v2(ctypes.byref(raw_uuid), handle.value)
    codes["cuDeviceGetPCIBusId"] = driver.cuDeviceGetPCIBusId(bus, len(bus), handle.value)
    require(all(type(x) is int and x == 0 for x in codes.values()) and count.value == 1, "exactly one current CUDA device must be visible")
    value.update(gpu_uuid="GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid))), pci_address=pci(bus.value.decode()))
    return value


def bind_process(plan, metadata, jax_device, directory):
    """Preserve one attempted resolution, including failures, before learner work."""
    directory = Path(directory)
    require(not any((directory / name).exists() for name in ("resolved_device_binding.json", "resolved_device_binding_attempt.json")),
        "preserve the prior physical binding attempt; do not retry this immutable unit")
    attempt = {"schema_version": 1, "kind": "uva_current_process_device_binding_attempt", "profile": PROFILE,
        "capability": CAPABILITY, "native": {}, **BOUNDARY}
    try:
        value = bind_process_attempt(plan, metadata, jax_device, directory, attempt)
        publish(directory / "resolved_device_binding.json", value)
        attempt.update(passed=True, binding=receipt(directory / "resolved_device_binding.json"))
        return value
    except BaseException as error:
        attempt.update(passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        publish(directory / "resolved_device_binding_attempt.json", attempt)


def bind_process_attempt(plan, metadata, jax_device, directory, attempt):
    """Resolve only after current metadata and the sole JAX device have passed."""
    directory = Path(directory)
    validate_metadata(plan, metadata)
    helper = adjacent("device_memory_limit")
    before = helper.process_identity()
    require(before == metadata["process"] and all(os.environ.get(k) == v for k, v in metadata["environment"].items()), "process or original selector changed before native CUDA resolution")
    require("jax" in sys.modules and sys.modules["jax"].local_devices(backend="gpu") == [jax_device],
        "the actual process must expose exactly this one local JAX GPU")
    attempt.update(plan_sha256=metadata["inputs"]["plan"]["sha256"], metadata_sha256=digest(encoded(metadata)), process=before)
    native = native_identity(jax_device, attempt["native"])
    gpu, bus = native["gpu_uuid"], native["pci_address"]
    command = ["nvidia-smi", "-i", gpu, "--query-gpu=uuid,name,pci.bus_id,memory.total,driver_version", "--format=csv,noheader,nounits"]
    attempt["hardware"] = bounded_query(command)
    query = attempt["hardware"]
    require(query["capture_status"] == "completed" and type(query["returncode"]) is int
        and query["returncode"] == 0 and query["stderr"] == "", "assigned-device hardware query failed")
    proc_path = Path("/proc/driver/nvidia/gpus") / bus / "information"
    attempt["proc"] = {"path": str(proc_path), "read_status": "started"}
    with proc_path.open() as stream:
        proc_raw = stream.read(32769)
    attempt["proc"] = {"path": str(proc_path), "raw": proc_raw, "sha256": digest(proc_raw.encode())}
    require(len(proc_raw.encode()) <= 32768, "assigned-device proc metadata exceeds its read limit")
    minors = re.findall(r"^Device Minor:\s*([0-9]+)\s*$", proc_raw, re.M)
    require(len(minors) == 1, "current CUDA PCI address must identify one physical device minor")
    device_path = Path("/dev/nvidia" + minors[0]); information = device_path.lstat()
    attempt["device"] = {"path": str(device_path), "character_device": stat.S_ISCHR(information.st_mode), "symlink": stat.S_ISLNK(information.st_mode),
        "major": os.major(information.st_rdev), "minor": os.minor(information.st_rdev)}
    attempt["driver_capture_expected_path"] = str(directory / "captured_driver" / Path(native["driver_library"]).name)
    capture = adjacent("h100_nvl_slurm_site_validation").capture_driver_image(native, directory)
    attempt["driver_capture"] = capture
    value = {"schema_version": 1, "kind": "uva_current_process_resolved_device_binding", "profile": PROFILE,
        "capability": CAPABILITY, "plan_sha256": metadata["inputs"]["plan"]["sha256"], "metadata_sha256": digest(encoded(metadata)),
        "process": before, "original_environment": metadata["environment"], "native": native,
        "gpu_uuid": gpu, "pci_address": bus, "physical_minor": int(minors[0]), "driver_capture": capture,
        "hardware": query, "proc": attempt["proc"], "device": attempt["device"],
        "process_after": helper.process_identity(), "environment_after": {k: os.environ.get(k) for k in metadata["environment"]},
        "startup_memory_observed": False, "other_gpu_processes_observed": False, **BOUNDARY}
    validate_binding(plan, metadata, value, directory)
    return value


def validate_binding(plan, metadata, value, directory=None):
    """Rebuild one binding from saved bytes; this grants no portable or formal admission."""
    validate_metadata(plan, metadata)
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1 and value.get("kind") == "uva_current_process_resolved_device_binding"
        and value.get("capability") == CAPABILITY, "unsupported resolved physical binding")
    require_profile(value["profile"])
    require(all(exact(value.get(k), v) for k, v in BOUNDARY.items())
        and value.get("startup_memory_observed") is False and value.get("other_gpu_processes_observed") is False, "the binding cannot invent mapping, startup capture, process visibility or admission")
    require(value["plan_sha256"] == metadata["inputs"]["plan"]["sha256"] and value["metadata_sha256"] == digest(encoded(metadata))
        and value["process"] == value["process_after"] == metadata["process"]
        and value["original_environment"] == value["environment_after"] == metadata["environment"], "resolved process, immutable input or original selector changed")
    native = value["native"]
    require(type(native["pid"]) is int and native["pid"] == metadata["process"]["pid"]
        and type(native["cuda_device_count"]) is int and native["cuda_device_count"] == 1
        and type(native["jax_local_hardware_id"]) is int and native["jax_local_hardware_id"] == 0
        and type(native["cuda_device_handle"]) is int and native["cuda_device_handle"] >= 0
        and type(native.get("current_context")) is int and native["current_context"] > 0
        and type(native.get("current_context_device")) is int and native["current_context_device"] == native["cuda_device_handle"]
        and native.get("jax_identity_basis") == "sole local JAX GPU, sole CUDA-visible ordinal, and that device's current CUDA context"
        and native["jax_device_kind"] == PROFILE["model"]
        and exact(native["api_return_codes"], {k: 0 for k in ("cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid_v2", "cuDeviceGetPCIBusId", "cuCtxGetCurrent", "cuCtxGetDevice")})
        and re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", native["gpu_uuid"])
        and native["gpu_uuid"] == value["gpu_uuid"] and pci(native["pci_address"]) == value["pci_address"], "same-process native CUDA/JAX identity changed")
    require(native["driver_library"].endswith("." + PROFILE["driver"])
        and digest(native["mapped_driver_bytes"].encode()) == native["mapped_driver_sha256"]
        and {line.split()[-1] for line in native["mapped_driver_bytes"].splitlines()} == {native["driver_library"]}, "the actual mapped driver path or bytes changed")
    query = value["hardware"]
    require(query["command"] == ["nvidia-smi", "-i", value["gpu_uuid"], "--query-gpu=uuid,name,pci.bus_id,memory.total,driver_version", "--format=csv,noheader,nounits"]
        and query.get("capture_status") == "completed" and type(query["returncode"]) is int
        and query["returncode"] == 0 and query["stderr"] == "", "only the actual assigned-UUID hardware query is supported")
    rows = list(csv.reader(query["stdout"].splitlines()))
    require(len(rows) == 1 and len(rows[0]) == 5, "one current physical hardware row is required")
    gpu, model, bus, memory, driver = [x.strip() for x in rows[0]]
    require(gpu == value["gpu_uuid"] and model == PROFILE["model"] and pci(bus) == value["pci_address"]
        and memory == str(PROFILE["memory_mib"]) and driver == PROFILE["driver"], "actual UUID/model/PCI/memory/driver changed")
    proc = value["proc"]
    require(proc["path"] == "/proc/driver/nvidia/gpus/" + value["pci_address"] + "/information"
        and digest(proc["raw"].encode()) == proc["sha256"], "physical proc metadata changed")
    for key, expected in (("GPU UUID", gpu), ("Model", model), ("Bus Location", value["pci_address"]), ("Device Minor", str(value["physical_minor"]))):
        found = re.findall(r"^" + re.escape(key) + r":\s*(.*?)\s*$", proc["raw"], re.M)
        require(found == [expected], "native CUDA UUID/PCI does not join exact physical proc/minor metadata")
    device = value["device"]
    require(type(value["physical_minor"]) is int and value["physical_minor"] >= 0 and exact(device, {"path": "/dev/nvidia" + str(value["physical_minor"]),
        "character_device": True, "symlink": False, "major": 195, "minor": value["physical_minor"]}), "physical Linux device metadata changed")
    directory = Path(directory) if directory is not None else Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
    capture = value["driver_capture"]
    target = directory / "captured_driver" / Path(native["driver_library"]).name
    require(type(capture.get("schema_version")) is int and capture["schema_version"] == 1 and capture.get("kind") == "captured_mapped_cuda_driver_image"
        and type(capture.get("pid")) is int and capture["pid"] == native["pid"]
        and capture["original"] == {"path": native["driver_library"], "sha256": native["driver_library_sha256"]}
        and capture["captured"] == {"path": str(target), "sha256": native["driver_library_sha256"]}
        and type(capture["bytes"]) is int and capture["bytes"] > 0
        and receipt(target) == capture["captured"] and target.stat().st_size == capture["bytes"], "immutable captured mapped driver changed")
    return value
