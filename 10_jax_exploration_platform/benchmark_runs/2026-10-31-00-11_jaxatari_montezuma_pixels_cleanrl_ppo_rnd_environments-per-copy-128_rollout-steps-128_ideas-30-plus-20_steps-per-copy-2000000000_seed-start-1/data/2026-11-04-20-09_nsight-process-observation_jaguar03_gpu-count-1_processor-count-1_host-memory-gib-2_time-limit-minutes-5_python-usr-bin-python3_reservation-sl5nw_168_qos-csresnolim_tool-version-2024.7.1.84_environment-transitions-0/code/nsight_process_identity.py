"""Observe only the new Nsight process and its owned descendants before numerical work."""
import hashlib
import json
import os
from pathlib import Path
import re
import select
import signal
import stat
import time

PROTOCOL = "nsight-owned-descendant-observation-version-1"
MAX_PROCESSES = 32
MAX_THREADS = 128
MAX_DEPTH = 8
MAX_BYTES = 65536
STABLE_FIELDS = ("pid", "parent_pid", "process_group", "session", "process_start_time",
                 "uids", "cgroup", "command", "executable")


def require(value, message):
    """Stop before release when an identity or its observed relationship is unsupported."""
    if not value:
        raise ValueError(message)


def digest(raw):
    """Hash exact source, plan and observation bytes."""
    return hashlib.sha256(raw).hexdigest()


def receipt(path):
    """Bind an exact ordinary file without following a path alias."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
            and not path.is_symlink(), "an exact ordinary process artifact is required")
    return {"path": str(path), "sha256": digest(path.read_bytes())}


def read_json(path):
    """Refuse duplicate JSON keys in a process claim or release."""
    receipt(path)
    def unique(pairs):
        """Keep one unambiguous value for each observed field."""
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate process-artifact key")
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def publish(path, value):
    """Create and fsync one process artifact, preserving every previous attempt."""
    path = Path(path)
    require(path.parent.resolve(strict=True) == path.parent, "process-artifact parent is not exact")
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    pending = path.with_name("." + path.name + ".pending")
    require(not os.path.lexists(path) and not os.path.lexists(pending),
            "preserve the existing final or partial process artifact")
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    # Before: only a complete fsynced temporary exists; after: one atomically visible final inode.
    # link refuses an existing final path. A failed attempt keeps its temporary bytes for review.
    os.link(pending, path, follow_symlinks=False)
    pending.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return receipt(path)


def parsed_stat(raw):
    """Read stable Linux process fields while retaining the original stat bytes separately."""
    require(isinstance(raw, str) and ") " in raw, "invalid process stat")
    # Before: "123 (python) S 100 123 123 ..."; after: typed PID, parent, group, session and birth.
    first, remainder = raw.rsplit(") ", 1)
    pid = first.split(" (", 1)[0]
    fields = remainder.split()
    require(pid.isdigit() and len(fields) >= 20 and all(fields[i].isdigit() for i in (1, 2, 3, 19)),
            "incomplete process stat")
    require(str(int(pid)) == pid and int(pid) > 0
            and all(str(int(fields[i])) == fields[i] for i in (1, 2, 3, 19)),
            "process stat identities must use canonical digits")
    require(fields[0] not in {"Z", "X", "x"}, "the observed process has already exited")
    return {"pid": int(pid), "parent_pid": int(fields[1]), "process_group": int(fields[2]),
            "session": int(fields[3]), "process_start_time": fields[19]}


def parsed_uids(raw):
    """Require all four kernel-reported user identifiers."""
    rows = [line.split()[1:] for line in raw.splitlines() if line.startswith("Uid:")]
    require(len(rows) == 1 and len(rows[0]) == 4 and all(item.isdigit() for item in rows[0]),
            "missing or invalid actual process user identifiers")
    return [int(value) for value in rows[0]]


def stable(record):
    """Compare identity fields without comparing changing CPU accounting counters."""
    return {name: record[name] for name in STABLE_FIELDS}


class ProcReader:
    """Read only a caller-supplied owned PID and its own thread-child lists."""
    def __init__(self, root=Path("/proc")):
        """Pin the proc namespace used by this observation or a labelled test fixture."""
        self.root = Path(root)
        require(self.root.is_absolute() and self.root.resolve(strict=True) == self.root,
                "the process namespace cannot use an alias")

    def bounded(self, directory, name):
        """Read one bounded regular proc file through its pinned process directory."""
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            require(stat.S_ISREG(os.fstat(descriptor).st_mode), "a process observation is not a regular file")
            raw = os.read(descriptor, MAX_BYTES + 1)
            require(len(raw) <= MAX_BYTES, "a process observation exceeded its bound")
            return raw
        finally:
            os.close(descriptor)

    def snapshot(self, pid, owner_uid, expected_cgroup=None):
        """Check user and cgroup before retaining this known descendant's command."""
        require(type(pid) is int and pid > 0 and type(owner_uid) is int and owner_uid >= 0,
                "a typed owned process identity is required")
        descriptor = os.open(self.root / str(pid), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            # User and allocation membership precede command/executable inspection.
            status = self.bounded(descriptor, "status").decode("utf-8", errors="strict")
            uids = parsed_uids(status)
            require(uids == [owner_uid] * 4, "the observed descendant is not owned by this user")
            cgroup = self.bounded(descriptor, "cgroup").decode("utf-8", errors="strict")
            require(expected_cgroup is None or cgroup == expected_cgroup,
                    "the descendant left the exact owned allocation and numeric step")
            before = self.bounded(descriptor, "stat").decode("utf-8", errors="strict")
            identity = parsed_stat(before)
            require(identity["pid"] == pid, "the process directory changed its PID")
            command_raw = self.bounded(descriptor, "cmdline")
            require(command_raw.endswith(b"\0") and command_raw != b"\0", "the actual process has no complete command")
            command = [item.decode("utf-8", errors="strict") for item in command_raw[:-1].split(b"\0")]
            executable = os.readlink("exe", dir_fd=descriptor)
            after = self.bounded(descriptor, "stat").decode("utf-8", errors="strict")
            status_after = self.bounded(descriptor, "status").decode("utf-8", errors="strict")
            require(parsed_stat(before) == parsed_stat(after)
                    and parsed_uids(status_after) == uids
                    and self.bounded(descriptor, "cgroup").decode() == cgroup
                    and self.bounded(descriptor, "cmdline") == command_raw
                    and os.readlink("exe", dir_fd=descriptor) == executable,
                    "the process changed while its identity was being observed")
        finally:
            os.close(descriptor)
        return {**identity, "uids": uids, "cgroup": cgroup, "command": command, "executable": executable,
                "stat_before": before, "stat_after": after, "status": status, "status_after": status_after,
                "cmdline": command_raw.decode("utf-8", errors="strict"), "cmdline_sha256": digest(command_raw),
                "observed_at_seconds": time.time()}

    def children(self, pid):
        """Discover candidates only through threads of an already proved owned process."""
        directory = self.root / str(pid) / "task"
        threads = list(directory.iterdir())
        require(0 < len(threads) <= MAX_THREADS and all(path.name.isdigit() and not path.is_symlink() for path in threads),
                "unsupported owned-process thread roster")
        threads.sort(key=lambda path: int(path.name))
        rows, children = [], set()
        for thread in threads:
            descriptor = os.open(thread, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                raw = self.bounded(descriptor, "children")
            finally:
                os.close(descriptor)
            entries = raw.decode("ascii").split()
            require(all(value.isdigit() and int(value) > 0 for value in entries), "invalid owned thread-child list")
            rows.append({"tid": int(thread.name), "raw": raw.decode("ascii"), "sha256": digest(raw)})
            children.update(int(value) for value in entries)
        return sorted(children), rows


def require_process_control_apis():
    """Refuse this interpreter when either required process-descriptor API is absent or not callable."""
    required = {"os.pidfd_open": getattr(os, "pidfd_open", None),
                "signal.pidfd_send_signal": getattr(signal, "pidfd_send_signal", None)}
    missing = [name for name, function in required.items() if not callable(function)]
    require(not missing, "this Python interpreter lacks required process-control APIs: " + ", ".join(missing))
    return {name: True for name in required}


def open_process_handle(pid):
    """Pin this actual PID before trusting a process edge or sending a signal."""
    require_process_control_apis()
    return os.pidfd_open(pid, 0)


def handle_exited(descriptor, milliseconds=0):
    """Observe exit through a pinned process descriptor without guessing its return status."""
    waiter = select.poll()
    waiter.register(descriptor, select.POLLIN)
    return bool(waiter.poll(milliseconds))


def signal_handle(descriptor, number):
    """Signal only the process referred to by this already verified descriptor."""
    signal.pidfd_send_signal(descriptor, number, None, 0)


def close_process_handle(descriptor):
    """Close this observer's own pidfd without selecting a process by a later PID lookup."""
    os.close(descriptor)


class OwnedProcesses:
    """Hold pinned identities for positively observed descendants, without global PID scans."""
    def __init__(self, root_pid, root_birth, parent, reader=None):
        """Start from the actual new Popen PID, never from a claimed arbitrary ancestor."""
        self.reader = reader or ProcReader()
        self.owner_uid = parent["uids"][0]
        self.cgroup = parent["cgroup"]
        self.root_pid, self.root_birth = root_pid, root_birth
        self.parent = parent
        self.handles, self.nodes, self.edges, self.child_lists = {}, {}, [], []
        self.pin(root_pid, parent["pid"], root_birth)
        self.root_initial = self.nodes[root_pid]

    def pin(self, pid, parent_pid, birth=None):
        """Pin and recheck a candidate from an observed owned-parent child list."""
        require(pid not in self.handles and len(self.handles) < MAX_PROCESSES, "duplicate or oversized owned process tree")
        descriptor = open_process_handle(pid)
        try:
            current = self.reader.snapshot(pid, self.owner_uid, self.cgroup)
            require(current["parent_pid"] == parent_pid and (birth is None or current["process_start_time"] == birth)
                    and not handle_exited(descriptor), "the owned process edge or birth changed")
        except BaseException:
            close_process_handle(descriptor)
            raise
        self.handles[pid], self.nodes[pid] = descriptor, current
        return current

    def observe(self, capture_pid, capture_birth, expected_command):
        """Prove a positive ancestry path and recheck it before any numerical release."""
        # The pinned launcher may execute another image before its blocked payload appears.
        root = self.reader.snapshot(self.root_pid, self.owner_uid, self.cgroup)
        invariant = ("pid", "parent_pid", "process_start_time", "uids", "cgroup")
        require(all(root[key] == self.root_initial[key] for key in invariant)
                and not handle_exited(self.handles[self.root_pid]),
                "the pinned Nsight root identity changed before release")
        self.nodes[self.root_pid] = root
        pending = [(self.root_pid, 0)]
        while pending:
            parent_pid, depth = pending.pop(0)
            require(depth < MAX_DEPTH, "the observed Nsight process tree exceeded its depth bound")
            children, raw_lists = self.reader.children(parent_pid)
            self.child_lists.append({"pid": parent_pid, "process_start_time": self.nodes[parent_pid]["process_start_time"],
                                     "thread_children": raw_lists})
            for child_pid in children:
                require(child_pid != parent_pid, "a process cannot be its own child")
                if child_pid in self.nodes:
                    require(self.nodes[child_pid]["parent_pid"] == parent_pid, "conflicting observed parent edges")
                    continue
                child = self.pin(child_pid, parent_pid)
                parent = self.nodes[parent_pid]
                require(int(parent["process_start_time"]) <= int(child["process_start_time"]), "a child predates its observed parent")
                self.edges.append({"parent_pid": parent_pid, "parent_start_time": parent["process_start_time"],
                                   "child_pid": child_pid, "child_start_time": child["process_start_time"]})
                pending.append((child_pid, depth + 1))
            if capture_pid in self.nodes:
                break
        require(capture_pid in self.nodes, "the capture is not a positively observed descendant of this Nsight process")
        capture = self.nodes[capture_pid]
        require(capture["process_start_time"] == capture_birth and capture["command"] == expected_command,
                "the observed capture birth or exact source command changed")
        # Raw thread-child lists give positive edges only; they do not prove whole-tree completeness.
        for pid, before in self.nodes.items():
            after = self.reader.snapshot(pid, self.owner_uid, self.cgroup)
            require(stable(before) == stable(after) and not handle_exited(self.handles[pid]),
                    "an owned process identity or edge changed before release")
        return {"schema_version": 1, "protocol": PROTOCOL, "root_pid": self.root_pid,
                "root_start_time": self.root_birth, "parent": self.parent, "nodes": self.nodes,
                "root_initial": self.root_initial,
                "edges": self.edges, "child_lists": self.child_lists, "capture_pid": capture_pid,
                "positive_ancestry_only": True, "whole_tree_completeness_established": False,
                "pidfds_opened_before_identity_reads": True, "pidfds_retained_through_release": True}

    def stop(self, child, timeout=120):
        """Stop pinned observed descendants, then wait on the actual owned Popen process."""
        signals = []
        for pid, descriptor in reversed(list(self.handles.items())):
            if not handle_exited(descriptor):
                try:
                    signal_handle(descriptor, signal.SIGTERM)
                except ProcessLookupError:
                    require(handle_exited(descriptor), "a signalled owned handle has an unexplained state")
                signals.append({"pid": pid, "process_start_time": self.nodes[pid]["process_start_time"], "signal": int(signal.SIGTERM)})
        deadline = time.monotonic() + timeout
        for pid, descriptor in reversed(list(self.handles.items())):
            remaining = max(0, int((deadline - time.monotonic()) * 1000))
            if not handle_exited(descriptor, remaining):
                try:
                    signal_handle(descriptor, signal.SIGKILL)
                except ProcessLookupError:
                    require(handle_exited(descriptor), "a killed owned handle has an unexplained state")
                signals.append({"pid": pid, "process_start_time": self.nodes[pid]["process_start_time"], "signal": int(signal.SIGKILL)})
                require(handle_exited(descriptor, 10000), "a pinned owned process did not exit after SIGKILL")
        returncode = child.wait()
        return {"signals": signals, "waited_popen_pid": child.pid, "popen_returncode": returncode,
                "known_pinned_processes_exited": True, "unobserved_descendant_exit_not_claimed": True}

    def close(self):
        """Release only this observer's own process descriptors."""
        for descriptor in self.handles.values():
            close_process_handle(descriptor)
        self.handles.clear()


def validate_snapshot(record, uid, cgroup):
    """Reconstruct saved identities from raw process fields instead of trusting summaries."""
    before, after = parsed_stat(record["stat_before"]), parsed_stat(record["stat_after"])
    require(before == after and all(type(record[key]) is type(value) and record[key] == value for key, value in before.items()),
            "saved process fields differ from their actual stat bytes")
    require(type(uid) is int and uid >= 0 and isinstance(record["uids"], list)
            and len(record["uids"]) == 4 and all(type(value) is int for value in record["uids"])
            and parsed_uids(record["status"]) == parsed_uids(record["status_after"])
            == record["uids"] == [uid] * 4 and record["cgroup"] == cgroup,
            "saved process ownership or allocation differs")
    raw = record["cmdline"].encode("utf-8")
    require(raw.endswith(b"\0") and digest(raw) == record["cmdline_sha256"]
            and record["command"] == [part.decode() for part in raw[:-1].split(b"\0")],
            "saved process command differs from its raw bytes")
    return stable(record)


def validate_observation(value, claim, expected_command, parent):
    """Require the claimed capture to lie on the saved actual owned Nsight ancestry path."""
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1
            and value.get("protocol") == PROTOCOL and value.get("positive_ancestry_only") is True
            and value.get("whole_tree_completeness_established") is False
            and value.get("pidfds_opened_before_identity_reads") is True
            and value.get("pidfds_retained_through_release") is True,
            "unsupported process observation protocol")
    uid, cgroup = parent["uids"][0], parent["cgroup"]
    validate_snapshot(parent, uid, cgroup)
    require(stable(value["parent"]) == stable(parent), "the observed parent differs from the actual supervisor")
    require(all(type(pid) is str and pid.isdigit() and str(int(pid)) == pid for pid in value["nodes"]),
            "saved process keys must be exact canonical PID strings")
    nodes = {int(pid): row for pid, row in value["nodes"].items()}
    require(0 < len(nodes) <= MAX_PROCESSES and type(value["root_pid"]) is int
            and value["root_pid"] in nodes and type(value["capture_pid"]) is int, "missing owned Nsight root")
    for pid, row in nodes.items():
        require(type(pid) is int and pid == row["pid"], "saved process PID key changed")
        validate_snapshot(row, uid, cgroup)
    root = nodes[value["root_pid"]]
    initial = value["root_initial"]
    validate_snapshot(initial, uid, cgroup)
    require(all(initial[key] == root[key] for key in ("pid", "parent_pid", "process_start_time", "uids", "cgroup")),
            "the initial pinned launcher differs from its blocked-time root")
    require(root["process_start_time"] == value["root_start_time"] and root["parent_pid"] == parent["pid"],
            "the actual Nsight Popen process changed")
    parents = {}
    listed = {}
    for row in value["child_lists"]:
        parent_pid = row["pid"]
        require(type(parent_pid) is int and parent_pid not in listed and parent_pid in nodes
                and row["process_start_time"] == nodes[parent_pid]["process_start_time"]
                and 0 < len(row["thread_children"]) <= MAX_THREADS,
                "a saved thread-child list has another parent")
        children = set()
        tids = set()
        for thread in row["thread_children"]:
            require(type(thread["tid"]) is int and thread["tid"] > 0 and thread["tid"] not in tids,
                    "saved thread identifiers must be unique exact integers")
            tids.add(thread["tid"])
            raw = thread["raw"].encode("ascii")
            require(len(raw) <= MAX_BYTES and digest(raw) == thread["sha256"]
                    and all(part.isdigit() and int(part) > 0 and str(int(part)) == part for part in raw.decode().split()),
                    "a saved owned thread-child list changed")
            children.update(int(part) for part in raw.decode().split())
        listed[parent_pid] = children
    for edge in value["edges"]:
        child, ancestor = edge["child_pid"], edge["parent_pid"]
        require(type(child) is int and type(ancestor) is int and child > 0 and ancestor > 0
                and child not in parents and child in nodes and ancestor in nodes and child in listed.get(ancestor, set())
                and type(edge["child_start_time"]) is str and type(edge["parent_start_time"]) is str
                and re.fullmatch(r"0|[1-9][0-9]*", edge["child_start_time"])
                and re.fullmatch(r"0|[1-9][0-9]*", edge["parent_start_time"])
                and nodes[child]["parent_pid"] == ancestor
                and nodes[child]["process_start_time"] == edge["child_start_time"]
                and nodes[ancestor]["process_start_time"] == edge["parent_start_time"],
                "saved owned ancestry edge changed")
        parents[child] = ancestor
    require(set(parents) == set(nodes) - {value["root_pid"]}, "a saved process has no observed owned parent")
    capture_pid = value["capture_pid"]
    require(capture_pid == claim["pid"] and capture_pid in nodes
            and nodes[capture_pid]["process_start_time"] == claim["process_start_time"]
            and nodes[capture_pid]["command"] == expected_command, "saved actual capture identity changed")
    path, current = [], capture_pid
    while current != value["root_pid"]:
        require(current not in path and current in parents and len(path) < MAX_DEPTH, "capture has no exact bounded Nsight ancestry")
        path.append(current); current = parents[current]
    return {"capture": nodes[capture_pid], "path": list(reversed(path)), "root": root}


def commands(directory, plan):
    """Record the exact waiter and capture source commands without changing learner settings."""
    if plan.get("kind") == "nsight_process_identity_diagnostic":
        entry = str(Path(directory) / "code/diagnose_nsight_process_tree.py")
        return ([plan["python"], "-I", "-S", "-B", entry, "--waiter-unit", str(directory)],
                [plan["python"], "-I", "-S", "-B", entry, "--probe-unit", str(directory)])
    entry = str(Path(directory) / "code/profile_selected.py")
    return ([plan["python"], "-I", "-S", "-B", entry, "--owned-waiter-unit", str(directory)],
            [plan["python"], entry, "--capture-unit", str(directory)])


def nsight_command(directory, plan):
    """Keep the actual approved profiling options and launch only the metadata waiter."""
    waiter, unused_capture = commands(directory, plan)
    return [plan["profile_tool"]["path"], "profile", "--trace=cuda,cudnn", "--sample=none", "--cpuctxsw=none",
            "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
            "--stop-on-exit=true", "--force-overwrite=false", "--output=" + str(Path(directory) / "trace/capture"),
            *waiter]


def validate_saved_intent(directory, plan, intent):
    """Reconstruct source, command and owned-step scope without inspecting historical PIDs."""
    directory = Path(directory)
    waiter, capture = commands(directory, plan)
    entry_name = "diagnose_nsight_process_tree.py" if plan.get("kind") == "nsight_process_identity_diagnostic" else "profile_selected.py"
    require(intent.get("protocol") == PROTOCOL and type(intent.get("schema_version")) is int
            and intent["schema_version"] == 1 and type(intent.get("challenge")) is str
            and re.fullmatch(r"[0-9a-f]{64}", intent["challenge"])
            and intent["plan"] == receipt(directory / "measurement_plan.json")
            and intent["source"] == receipt(directory / "code" / entry_name)
            and intent["waiter_command"] == waiter and intent["capture_command"] == capture
            and intent["nsight_command"] == nsight_command(directory, plan)
            and intent["source"]["sha256"] == plan["helpers"]["files"][
                "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + entry_name],
            "the one-attempt process challenge, source or plan changed")
    require(type(intent["job_id"]) is str and type(intent["step_id"]) is str
            and re.fullmatch(r"[1-9][0-9]*", intent["job_id"])
            and re.fullmatch(r"[0-9]+", intent["step_id"]), "the process challenge belongs to another owned step")
    parent = intent["parent"]
    validate_snapshot(parent, parent["uids"][0], parent["cgroup"])
    require(re.search(r"/job_" + re.escape(intent["job_id"]) + r"/step_" + re.escape(intent["step_id"]) + r"(?:/|$)",
                      parent["cgroup"]), "the process challenge lacks the actual numeric Slurm step")
    return intent


def validate_intent(directory, plan, intent):
    """Require this immutable challenge to belong to the current owned numeric step."""
    validate_saved_intent(directory, plan, intent)
    require(intent["job_id"] == os.environ.get("SLURM_JOB_ID")
            and intent["step_id"] == os.environ.get("SLURM_STEP_ID")
            and intent["parent"]["uids"] == [os.getuid()] * 4,
            "the process challenge is not in this actual owned numeric step")
    return intent


def make_intent(directory, plan, command):
    """Publish one fresh challenge without touching any scientific random stream."""
    directory = Path(directory)
    parent = ProcReader().snapshot(os.getpid(), os.getuid())
    waiter, capture = commands(directory, plan)
    # This entropy is an attempt identifier, not an environment or learner RNG draw.
    entry_name = "diagnose_nsight_process_tree.py" if plan.get("kind") == "nsight_process_identity_diagnostic" else "profile_selected.py"
    value = {"schema_version": 1, "protocol": PROTOCOL, "challenge": os.urandom(32).hex(),
             "plan": receipt(directory / "measurement_plan.json"),
             "source": receipt(directory / "code" / entry_name), "parent": parent,
             "nsight_command": command, "waiter_command": waiter, "capture_command": capture,
             "job_id": os.environ.get("SLURM_JOB_ID"), "step_id": os.environ.get("SLURM_STEP_ID")}
    validate_intent(directory, plan, value)
    publish(directory / "profile_process_intent.json", value)
    return value


def claim_self(directory, plan):
    """Save the actual capture identity before it can receive permission for numerical work."""
    directory = Path(directory)
    intent = validate_intent(directory, plan, read_json(directory / "profile_process_intent.json"))
    current = ProcReader().snapshot(os.getpid(), os.getuid(), intent["parent"]["cgroup"])
    require(current["command"] == intent["capture_command"], "the capture Python source command changed")
    value = {"schema_version": 1, "protocol": PROTOCOL, "hostname": os.uname().nodename.split(".")[0],
             "pid": current["pid"], "process_start_time": current["process_start_time"],
             "process": current, "challenge": intent["challenge"],
             "intent": receipt(directory / "profile_process_intent.json"),
             "source": intent["source"], "plan": intent["plan"]}
    publish(directory / "capture_identity.json", value)
    return value


def parent_handle(intent):
    """Pin the known supervisor and refuse a stale PID before waiting for release."""
    parent = intent["parent"]
    descriptor = open_process_handle(parent["pid"])
    try:
        actual = ProcReader().snapshot(parent["pid"], parent["uids"][0], parent["cgroup"])
        require(stable(actual) == stable(parent) and not handle_exited(descriptor),
                "the actual supervisor disappeared or changed before release")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def record_root_wait(directory, intent, owner, returncode):
    """Save the actual original Nsight Popen wait separately from the payload's actual parent wait."""
    require(type(returncode) is int, "the actual Nsight wait return must be an integer")
    return publish(Path(directory) / "profiler_actual_wait.json", {
        "schema_version": 1, "protocol": PROTOCOL, "kind": "original_nsight_popen_wait",
        "waited_pid": owner["pid"], "process_start_time": owner["process_start_time"],
        "command": intent["nsight_command"], "returncode": returncode, "exited_normally": returncode >= 0,
        "launch": receipt(Path(directory) / "profiler_process.json"), "interventions": []})


def validate_root_wait(directory, intent, owner):
    """Require the successful, unchanged wait on the actual original Nsight Popen process."""
    waited = read_json(Path(directory) / "profiler_actual_wait.json")
    require(type(waited.get("schema_version")) is int and waited["schema_version"] == 1
            and waited.get("protocol") == PROTOCOL and waited.get("kind") == "original_nsight_popen_wait"
            and type(waited["waited_pid"]) is int and waited["waited_pid"] == owner["pid"]
            and waited["process_start_time"] == owner["process_start_time"]
            and waited["command"] == intent["nsight_command"] and type(waited["returncode"]) is int
            and waited["returncode"] == 0 and waited["exited_normally"] is True and waited["interventions"] == []
            and waited["launch"] == receipt(Path(directory) / "profiler_process.json"),
            "the original Nsight process has no successful exact Popen wait")
    return waited


def validate_claim(directory, plan, intent, claim):
    """Require the claim to carry the same source, plan and attempt challenge."""
    validate_intent(directory, plan, intent)
    require(type(claim.get("schema_version")) is int and claim["schema_version"] == 1
            and claim.get("protocol") == PROTOCOL and claim.get("challenge") == intent["challenge"]
            and claim.get("intent") == receipt(Path(directory) / "profile_process_intent.json")
            and claim.get("plan") == intent["plan"] and claim.get("source") == intent["source"],
            "the child claim belongs to another attempt, plan or source")
    process = claim["process"]
    validate_snapshot(process, intent["parent"]["uids"][0], intent["parent"]["cgroup"])
    require(type(claim["pid"]) is int and claim["pid"] == process["pid"]
            and claim["process_start_time"] == process["process_start_time"]
            and process["command"] == intent["capture_command"], "the child claim changed its actual process")
    return claim


def validate_live_observation(directory, plan, claim):
    """Recheck every positively observed process immediately before entering numerical work."""
    directory = Path(directory)
    intent = validate_intent(directory, plan, read_json(directory / "profile_process_intent.json"))
    validate_claim(directory, plan, intent, claim)
    observation = read_json(directory / "capture_process_observation.json")
    rebuilt = validate_observation(observation, claim, intent["capture_command"], intent["parent"])
    for row in observation["nodes"].values():
        actual = ProcReader().snapshot(row["pid"], row["uids"][0], row["cgroup"])
        require(stable(actual) == stable(row), "an observed process changed before numerical release")
    require(stable(ProcReader().snapshot(os.getpid(), os.getuid(), claim["process"]["cgroup"])) == stable(claim["process"]),
            "the live capture differs from its blocked pre-numerical identity")
    return rebuilt


def validate_saved(directory, plan, result):
    """Validate the actual source/ancestry/release and direct wait records without reading historical PIDs."""
    directory = Path(directory)
    intent = validate_saved_intent(directory, plan, read_json(directory / "profile_process_intent.json"))
    # Saved reading keeps the recorded job/step facts; it never inspects a possibly reused current PID.
    require(intent["plan"] == receipt(directory / "measurement_plan.json")
            and intent["source"] == receipt(directory / "code/profile_selected.py")
            and intent["source"]["sha256"] == plan["helpers"]["files"][
                "10_jax_exploration_platform/scripts/atari_pixel_campaign/profile_selected.py"],
            "saved profile process evidence changed its source or plan")
    waiter_command, capture_command = commands(directory, plan)
    require(intent["waiter_command"] == waiter_command and intent["capture_command"] == capture_command
            and intent["nsight_command"] == nsight_command(directory, plan)
            and intent["nsight_command"] == read_json(directory / "executed_command.json"),
            "saved profile process commands changed")
    claim = read_json(directory / "capture_identity.json")
    require(type(claim.get("schema_version")) is int and claim["schema_version"] == 1
            and claim.get("protocol") == PROTOCOL
            and claim["challenge"] == intent["challenge"] and claim["intent"] == receipt(directory / "profile_process_intent.json")
            and claim["source"] == intent["source"] and claim["plan"] == intent["plan"],
            "saved capture challenge or source changed")
    validate_snapshot(claim["process"], intent["parent"]["uids"][0], intent["parent"]["cgroup"])
    require(type(claim["pid"]) is int and claim["pid"] == claim["process"]["pid"]
            and claim["process_start_time"] == claim["process"]["process_start_time"]
            and claim["process"]["command"] == capture_command,
            "saved actual capture claim differs from its raw process")
    observation = read_json(directory / "capture_process_observation.json")
    rebuilt = validate_observation(observation, claim, capture_command, intent["parent"])
    require(stable(rebuilt["capture"]) == stable(claim["process"]),
            "the saved blocked capture differs from the released observed process")
    owner = read_json(directory / "profiler_process.json")
    require(owner["pid"] == observation["root_pid"] and owner["process_start_time"] == observation["root_start_time"]
            and owner["command"] == intent["nsight_command"] and owner["parent_pid"] == intent["parent"]["pid"]
            and owner["parent_start_time"] == intent["parent"]["process_start_time"],
            "saved Nsight launch differs from its actual Popen identity")
    validate_root_wait(directory, intent, owner)
    launch = read_json(directory / "capture_actual_launch.json")
    waited = read_json(directory / "capture_actual_wait.json")
    require(launch["challenge"] == waited["challenge"] == intent["challenge"]
            and launch["claim"] == receipt(directory / "capture_identity.json")
            and launch["child"]["pid"] == claim["pid"]
            and stable(launch["child"]) == stable(claim["process"])
            and launch["command"] == waited["command"] == capture_command
            and waited["launch"] == receipt(directory / "capture_actual_launch.json")
            and waited["waited_pid"] == claim["pid"] and type(waited["returncode"]) is int
            and waited["returncode"] == 0 and waited["exited_normally"] is True
            and waited["interventions"] == [], "the actual capture was not waited successfully by its owned parent")
    waiter = launch["waiter"]
    validate_snapshot(waiter, claim["process"]["uids"][0], claim["process"]["cgroup"])
    require(waiter["command"] == waiter_command and claim["process"]["parent_pid"] == waiter["pid"]
            and str(waiter["pid"]) in observation["nodes"]
            and stable(observation["nodes"][str(waiter["pid"])]) == stable(waiter),
            "saved actual Python parent is not the observed Nsight descendant")
    release = read_json(directory / "capture_release.json")
    require(release == {**owner, "capture_identity": claim,
                       "process_observation": receipt(directory / "capture_process_observation.json")},
            "saved numerical release differs from the actual observed source child")
    identity = result["execution_identity"]
    require(identity["pid"] == claim["pid"] and identity["process_start_time"] == claim["process_start_time"]
            and result.get("capture_process_protocol") == PROTOCOL
            and result.get("capture_process_observation") == receipt(directory / "capture_process_observation.json")
            and result.get("capture_actual_wait") == receipt(directory / "capture_actual_wait.json"),
            "the captured numerical result belongs to another process or omitted its actual wait")
    require(result.get("profiler_actual_wait") == receipt(directory / "profiler_actual_wait.json"),
            "the numerical result omitted the actual original Nsight wait")
    return {"capture_identity": receipt(directory / "capture_identity.json"),
            "process_observation": receipt(directory / "capture_process_observation.json"),
            "capture_wait": receipt(directory / "capture_actual_wait.json"), "capture_pid": rebuilt["capture"]["pid"]}
