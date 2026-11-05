"""Share one metadata deadline and positive-owned descendant scope; never admit profiling."""
import ast
from contextlib import contextmanager
import errno
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import threading
import time

NAME = "profile_metadata_runtime.py"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
PROTOCOL = {"schema_version": 1, "capability": "owned_profile_metadata_single_origin_and_descendant_closure",
    "work_seconds": 240, "cleanup_seconds": 60, "total_seconds": 300, "operational_admission": False}
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}
PROC_ROOT = Path("/proc")
MAX_NODES, MAX_EVENTS, MAX_BYTES = 256, 512, 1024 * 1024
_NATIVE = None
_REQUEST_COUNTS = {"PR_GET_CHILD_SUBREAPER": 0, "PR_SET_CHILD_SUBREAPER": 0}


def require(value, message):
    if not value:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def capability():
    """Refuse before any Popen or native bootstrap if exact process-handle APIs are absent."""
    require(sys.platform == "linux" and callable(getattr(os, "pidfd_open", None))
        and callable(getattr(signal, "pidfd_send_signal", None)) and callable(getattr(os, "waitid", None)),
        "closed metadata supervision requires system-Python pidfd/waitid APIs; no numeric-PID or native-provider fallback")


def process_observer(plan):
    """Load only the exact current profile process observer under a private metadata name."""
    path = Path(__file__).with_name("nsight_process_identity.py")
    raw = path.read_bytes()
    require(plan["helpers"]["files"].get(PREFIX + path.name) == digest(raw),
        "metadata process observer is outside the current profile helper closure")
    spec = importlib.util.spec_from_file_location("_profile_metadata_private_process_observer", path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def origin(plan, owner, *, started=None):
    """Create the single root origin before the outer metadata Popen."""
    started = time.monotonic() if started is None else started
    require(finite(started), "actual initial metadata monotonic time required")
    return {"schema_version": 1, "kind": "profile_metadata_single_actual_origin",
        "protocol": PROTOCOL, "plan_sha256": digest(encoded(plan)), "owner": owner,
        "clock_domain": {"hostname": socket.gethostname().split(".")[0],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "clock": "CLOCK_MONOTONIC"}, "started_monotonic": started,
        "work_end_monotonic": started + 240, "total_end_monotonic": started + 300,
        "created_at_seconds": time.time()}


class Budget:
    """Every role uses the same immutable host/boot/monotonic origin and total end."""
    def __init__(self, value, plan, *, clock=time.monotonic, verify_host=True):
        require(type(value) is dict and set(value) == {"schema_version", "kind", "protocol", "plan_sha256", "owner",
            "clock_domain", "started_monotonic", "work_end_monotonic", "total_end_monotonic", "created_at_seconds"}
            and type(value["schema_version"]) is int and value["schema_version"] == 1
            and value["kind"] == "profile_metadata_single_actual_origin"
            and encoded(value["protocol"]) == encoded(PROTOCOL)
            and value["plan_sha256"] == digest(encoded(plan))
            and all(finite(value[k]) for k in ("started_monotonic", "work_end_monotonic", "total_end_monotonic", "created_at_seconds"))
            and value["work_end_monotonic"] == value["started_monotonic"] + 240
            and value["total_end_monotonic"] == value["started_monotonic"] + 300,
            "metadata budget must retain its single exact reviewed origin")
        domain = value["clock_domain"]
        require(type(domain) is dict and set(domain) == {"hostname", "boot_id", "clock"}
            and domain["clock"] == "CLOCK_MONOTONIC"
            and type(domain["hostname"]) is str and domain["hostname"]
            and type(domain["boot_id"]) is str and len(domain["boot_id"]) == 36, "metadata clock domain incomplete")
        if verify_host:
            require(domain["hostname"] == socket.gethostname().split(".")[0]
                and domain["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "metadata origin belongs to another actual host or boot")
        self.origin, self.clock = value, clock
        self.cleanup_started = None

    def remaining(self, phase="work", *, allow_zero=False):
        end = self.origin["work_end_monotonic"] if phase == "work" else self.origin["total_end_monotonic"]
        if phase == "cleanup":
            require(self.cleanup_started is not None, "cleanup cannot start a fresh independent deadline")
            end = min(end, self.cleanup_started + 60)
        value = max(0., end - self.clock())
        require(allow_zero or value > 0, "the shared profile metadata " + phase + " budget expired")
        return value

    def timeout(self, maximum=None):
        value = self.remaining("work")
        return value if maximum is None else min(value, maximum)

    def begin_cleanup(self):
        if self.cleanup_started is None:
            self.cleanup_started = self.clock()
        return self.remaining("cleanup", allow_zero=True)

    def snapshot(self):
        return {"origin_sha256": digest(encoded(self.origin)), "observed_monotonic": self.clock(),
            "cleanup_started_monotonic": self.cleanup_started}

    def verify(self, value, *, cleanup=False):
        require(type(value) is dict and set(value) == {"origin_sha256", "observed_monotonic", "cleanup_started_monotonic"}
            and value["origin_sha256"] == digest(encoded(self.origin)) and finite(value["observed_monotonic"])
            and self.origin["started_monotonic"] <= value["observed_monotonic"] <= self.origin["total_end_monotonic"],
            "saved metadata phase exceeds or changes its original total budget")
        first = value["cleanup_started_monotonic"]
        if cleanup:
            require(finite(first) and self.origin["started_monotonic"] <= first <= value["observed_monotonic"]
                <= min(self.origin["total_end_monotonic"], first + 60), "saved cleanup exceeded its same reserved window")
        else:
            require(value["observed_monotonic"] <= self.origin["work_end_monotonic"], "saved metadata work exceeded240seconds")
        return value


def prepare_native_control():
    """Bind only current-process child-subreaper control after pidfd preflight, never GPU APIs."""
    global _NATIVE
    capability()
    if _NATIVE is None:
        import ctypes
        handle = ctypes.CDLL(None, use_errno=True)
        fn = handle.prctl
        fn.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        fn.restype = ctypes.c_int
        _NATIVE = (ctypes, handle, fn)
    return {"scope": "own-process Linux child-subreaper only", "target": None, "GPU_or_CUDA_qualification": False}


def child_subreaper(value=None):
    ctypes, unused, fn = _NATIVE
    request = 37 if value is None else 36
    if value is None:
        observed = ctypes.c_int()
        argument = ctypes.addressof(observed)
    else:
        require(type(value) is int and value in (0, 1), "typed subreaper flag required")
        argument = value
    ctypes.set_errno(0)
    returned = fn(request, argument, 0, 0, 0)
    error = ctypes.get_errno()
    result = observed.value if value is None else value
    key = "PR_GET_CHILD_SUBREAPER" if value is None else "PR_SET_CHILD_SUBREAPER"
    _REQUEST_COUNTS[key] += 1
    require(returned == 0 and error == 0, "actual current-process subreaper request failed")
    return {"request": key, "value": result, "returncode": returned, "errno": error}




def bounded(directory, name):
    """Read a bounded own-process proc file through an already pinned directory."""
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        raw = os.read(descriptor, MAX_BYTES + 1)
        require(len(raw) <= MAX_BYTES, "actual child proc bytes exceeded their bound")
        return raw
    finally:
        os.close(descriptor)

def parsed_identity(raw):
    """Retain birth and parent identities for both living children and unreaped zombies."""
    first, remaining = raw.rsplit(") ", 1)
    fields = remaining.split()
    pid = first.split(" (", 1)[0]
    require(pid.isdecimal() and len(fields) >= 20 and all(fields[index].isdecimal() for index in (1, 2, 3, 19)),
        "incomplete actual child process stat")
    return {"pid": int(pid), "parent_pid": int(fields[1]), "process_group": int(fields[2]),
        "session": int(fields[3]), "process_start_time": fields[19], "state": fields[0]}

class MetadataScope:
    """One closed metadata-only spawn domain; never equate process groups or scan foreign PIDs."""
    def __init__(self, plan, budget):
        capability()
        self.plan, self.deadline = plan, budget
        self.observer = process_observer(plan)
        self.owner = self.observer.ProcReader().snapshot(os.getpid(), 2618919)
        self.uid, self.pid, self.birth, self.cgroup = 2618919, self.owner["pid"], self.owner["process_start_time"], self.owner["cgroup"]
        self.nodes, self.handles, self.roots, self.waits, self.events = {}, {}, {}, {}, []
        self.interventions, self.read_errors, self.signals_sent = [], [], set()
        self.active, self.closed, self.failed, self.pending_command = False, False, False, None
        self.producer_handle = self.observer.open_process_handle(self.pid)
        self.native_bootstrap = prepare_native_control()
        self.start_counts = dict(_REQUEST_COUNTS)
        self.proof = None
        self.git_guard, self.inspection_commands = None, []

    def event(self, kind, **fields):
        require(len(self.events) < MAX_EVENTS, "metadata ownership event bound exceeded")
        self.events.append({"kind": kind, "at_seconds": time.time(), **fields})

    def __enter__(self):
        require(threading.active_count() == 1 and signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL,
            "metadata scope needs one fresh default-SIGCHLD thread")
        self.deadline.remaining()
        children, rows = self.child_lists(self.pid)
        require(not children and len(rows) == 1, "metadata scope requires an empty actual own-child baseline")
        try:
            observed = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError as error:
            require(error.errno == errno.ECHILD, "metadata initial wait was not ECHILD")
        else:
            raise ValueError("empty metadata baseline did not produce ECHILD: " + repr(observed))
        self.previous = child_subreaper()
        self.changed = child_subreaper(1);self.verified = child_subreaper()
        require(self.verified["value"] == 1, "metadata child subreaper not established")
        self.thread = threading.get_ident();self.active = True
        self.baseline = {"owner": self.owner, "empty_children": rows, "actual_echild": True,
            "previous_subreaper": self.previous, "set_subreaper": self.changed, "verified_subreaper": self.verified,
            "native_bootstrap": self.native_bootstrap, "budget": self.deadline.snapshot()}
        sys.addaudithook(self.audit)
        return self

    def audit(self, event, values):
        if self.active and event == "subprocess.Popen":
            require(threading.get_ident() == self.thread, "metadata Popen cannot come from another thread")
            command = list(values[1])
            if self.pending_command is None and command and command[0] == values[0] == "git":
                require(self.git_guard is not None, "unscoped Git needs the explicit exact-source read-only policy")
                self.git_guard(event, values)
                require(len(self.inspection_commands) < MAX_EVENTS, "metadata Git inspection bound exceeded")
                self.inspection_commands.append(command)
            else:
                require(self.pending_command is not None and command == self.pending_command
                    and values[0] == self.pending_command[0], "every metadata Popen must belong to the actual closed scope")
        if self.active and event == "import" and values[0].split(".")[0] in NUMERICAL:
            raise RuntimeError("numerical import forbidden in metadata scope")
        if self.active and event in {"socket.connect", "socket.bind", "socket.getaddrinfo", "os.system", "ctypes.dlopen", "ctypes.dlsym"}:
            raise RuntimeError("network or late native-library access forbidden in metadata scope")
        if self.active and event == "open" and isinstance(values[0], (str, bytes)) and (
                os.fsdecode(values[0]).startswith(("/dev/nvidia", "/dev/dri"))
                or os.fsdecode(values[0]) == "/tmp/sl5nw_serval02_card1.lock"):
            raise RuntimeError("GPU or standing descriptor access forbidden in metadata scope")

    def run(self, command, *, input=None, timeout=None, env=None, maximum_bytes=16 * 1024 * 1024):
        require(self.active and type(command) is list and command and all(type(x) is str for x in command),
            "one exact metadata command required")
        self.deadline.remaining()
        # Only trusted metadata callers select source-pinned Python or read-only Git.
        require(command[0] in ("/usr/bin/python3", "git"), "metadata scope cannot launch another executable")
        self.pending_command = command
        try:
            child = subprocess.Popen(command, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, close_fds=True, pass_fds=())
        finally:
            self.pending_command = None
        self.roots[child.pid] = {"process": child, "command": command, "started_at_seconds": time.time()}
        self.pin(child.pid, self.pid);self.scan()
        observed = self.observer.ProcReader().snapshot(child.pid, self.uid, self.cgroup)
        actual_child = {"pid": observed["pid"], "parent_pid": observed["parent_pid"],
            "process_start_time": observed["process_start_time"], "uids": observed["uids"],
            "argv": observed["command"], "hostname": socket.gethostname().split(".")[0], "cgroup": observed["cgroup"]}
        started = time.monotonic()
        first = True
        while True:
            self.scan();self.deadline.remaining()
            require(timeout is None or time.monotonic() - started < timeout, "metadata command reached its bounded phase limit")
            try:
                stdout, stderr = child.communicate(input=input if first else None, timeout=min(.05, self.deadline.remaining()))
                break
            except subprocess.TimeoutExpired as error:
                first = False
                require(len(error.output or b"") <= maximum_bytes and len(error.stderr or b"") <= maximum_bytes,
                    "metadata child exceeded its bounded captured output")
        self.deadline.remaining()
        require(len(stdout) <= maximum_bytes and len(stderr) <= maximum_bytes, "metadata output exceeds its bound")
        returned = self.record_wait(child)
        result = subprocess.CompletedProcess(command, returned, stdout, stderr)
        result.metadata_child = actual_child
        result.metadata_wait = self.waits[child.pid]
        return result

    def record_wait(self, child):
        require(child.pid in self.roots and self.roots[child.pid]["process"] is child, "only the exact metadata Popen root may be waited")
        returned = child.wait(timeout=0)
        if child.pid not in self.waits:
            self.waits[child.pid] = {"kind": "actual_metadata_popen_wait", "pid": child.pid,
                "birth": self.nodes[child.pid]["initial"]["process_start_time"], "command": self.roots[child.pid]["command"],
                "returncode": returned, "finished_at_seconds": time.time()}
        return returned

    def finish(self, requested=False):
        if self.closed:
            return self.proof
        if requested:self.deadline.begin_cleanup()
        started = time.monotonic()
        terminal = None
        inconclusive = False
        # Unknown closure never becomes success or releases the caller's custody.
        # After the shared total deadline, hold passively until actual kernel exit;
        # no new process or signal is permitted and any eventual proof is failed.
        while terminal is None:
            try:
                self.scan();self.reap()
                if self.deadline.remaining("work", allow_zero=True) <= 0 and not requested:
                    requested = True;self.deadline.begin_cleanup()
                if requested and self.deadline.remaining("cleanup", allow_zero=True) > 0:
                    self.signal_known(signal.SIGTERM if time.monotonic() - started < 2 else signal.SIGKILL)
                terminal = self.closure()
            except Exception as error:
                requested = True;self.deadline.begin_cleanup()
                if len(self.read_errors) < 32:
                    self.read_errors.append({"error": repr(error), "at_seconds": time.time()})
            if requested and self.deadline.remaining("cleanup", allow_zero=True) <= 0:
                inconclusive = True
            if terminal is None:
                time.sleep(.02)
        self.closed = True
        restored = child_subreaper(self.previous["value"]);verified = child_subreaper()
        self.active = False
        for descriptor in self.handles.values():self.observer.close_process_handle(descriptor)
        self.observer.close_process_handle(self.producer_handle)
        snapshot = self.deadline.snapshot()
        within = snapshot["observed_monotonic"] <= self.deadline.origin[
            "total_end_monotonic" if requested else "work_end_monotonic"] and not inconclusive
        self.proof = {"schema_version": 1, "kind": "profile_metadata_kernel_scope_closure",
            "origin_sha256": digest(encoded(self.deadline.origin)), "baseline": self.baseline,
            "positive_nodes": self.nodes, "root_waits": self.waits, "events": self.events,
            "interventions": self.interventions, "read_errors": self.read_errors,
            "source_inspection_commands": self.inspection_commands, "all_historical_PIDs_enumerated": False,
            "terminal": terminal,
            "restored": restored, "verified_restore": verified, "budget": snapshot,
            "all_metadata_descendants_exited": True, "within_shared_budget": within,
            "successful_no_intervention": not requested and not self.interventions,
            "scientific_credit": 0, "operational_admission": False}
        require(verified["value"] == restored["value"] == self.previous["value"], "metadata native flag restore is not causal")
        require(within, "metadata cleanup became inconclusive at the shared deadline; late closure cannot become success")
        return self.proof

    def __exit__(self, kind, error, traceback):
        self.finish(requested=error is not None)


    def known_parent(self, pid):
        """Require this exact previously pinned process instead of trusting a numeric same-UID PID."""
        require(type(pid) is int and pid > 0, "one exact known parent PID is required")
        if pid == self.pid:
            birth, handle = self.birth, self.producer_handle
        else:
            require(pid in self.handles and pid in self.nodes, "child lists require an already pinned owned parent")
            birth, handle = self.nodes[pid]["initial"]["process_start_time"], self.handles[pid]
        require(type(birth) is str and birth.isdecimal() and int(birth) > 0,
            "the known parent's actual pinned birth is required")
        require(not self.observer.handle_exited(handle), "the pinned known parent exited before child-list access")
        return birth, handle

    def parent_identity(self, directory, pid, birth, handle):
        """Read UID, PID, birth and cgroup through the same pinned parent directory before child access."""
        require(not self.observer.handle_exited(handle), "the known parent handle exited during child-list access")
        status = bounded(directory, "status").decode()
        require(self.observer.parsed_uids(status) == [self.uid] * 4, "the pinned known parent changed user identifiers")
        raw = bounded(directory, "stat").decode()
        identity = parsed_identity(raw)
        require(identity["pid"] == pid and identity["process_start_time"] == birth
            and identity["state"] not in {"Z", "X", "x"}, "the known parent PID or pinned birth changed before child-list access")
        cgroup = bounded(directory, "cgroup").decode()
        require(cgroup == self.cgroup, "the known parent changed its exact owned cgroup")
        require(not self.observer.handle_exited(handle), "the known parent exited before its child lists could be accepted")
        return {"pid": pid, "process_start_time": birth, "uids": [self.uid] * 4,
            "status": status, "stat": raw, "cgroup": cgroup, "pidfd_live": True}

    def child_lists(self, pid):
        """Read child lists only through a pinned known-parent directory with before-and-after birth proof."""
        birth, handle = self.known_parent(pid)
        directory = os.open(PROC_ROOT / str(pid), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        task_directory = None
        try:
            information = os.fstat(directory)
            before = self.parent_identity(directory, pid, birth, handle)
            task_directory = os.open("task", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            tids = sorted(os.listdir(task_directory), key=int)
            require(0 < len(tids) <= 4096 and all(value.isdecimal() and int(value) > 0 for value in tids),
                "owned parent thread count or identifiers exceeded their explicit bounds")
            rows, children, count = [], set(), 0
            for name in tids:
                thread = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=task_directory)
                try:
                    thread_before = bounded(thread, "stat").decode()
                    first = parsed_identity(thread_before)
                    require(first["pid"] == int(name), "the pinned own-thread directory changed its identity")
                    raw = bounded(thread, "children")
                    thread_after = bounded(thread, "stat").decode()
                    last = parsed_identity(thread_after)
                    require(last["pid"] == first["pid"] and last["process_start_time"] == first["process_start_time"],
                        "the pinned own-thread identity changed during its child-list read")
                    count += len(raw) + len(thread_before.encode()) + len(thread_after.encode())
                    require(count <= MAX_BYTES and all(value.isdecimal() and int(value) > 0
                        for value in raw.decode("ascii").split()), "actual own-thread child evidence is malformed or oversized")
                    children.update(int(value) for value in raw.decode("ascii").split())
                    rows.append({"tid": int(name), "raw": raw.decode("ascii"), "sha256": hashlib.sha256(raw).hexdigest(),
                        "thread_stat_before": thread_before, "thread_stat_after": thread_after})
                finally:
                    os.close(thread)
            after = self.parent_identity(directory, pid, birth, handle)
            actual = os.fstat(directory)
            require((actual.st_dev, actual.st_ino) == (information.st_dev, information.st_ino),
                "the pinned parent proc directory changed during child-list inspection")
            rows[0]["parent_read_identity"] = {"expected_pid": pid, "expected_birth": birth,
                "proc_directory_device": information.st_dev, "proc_directory_inode": information.st_ino,
                "before": before, "after": after, "pidfd_live_before": True, "pidfd_live_after": True}
            return children, rows
        finally:
            if task_directory is not None:
                os.close(task_directory)
            os.close(directory)

    def snapshot(self, pid):
        """Check all user identifiers before retaining a known child's command, including zombie identities."""
        descriptor = os.open(PROC_ROOT / str(pid), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            status = bounded(descriptor, "status").decode()
            require(self.observer.parsed_uids(status) == [self.uid] * 4, "a scoped descendant changed user identifiers")
            cgroup = bounded(descriptor, "cgroup").decode()
            require(cgroup == self.cgroup, "a scoped descendant changed its exact producer cgroup")
            before = bounded(descriptor, "stat").decode()
            identity = parsed_identity(before)
            require(identity["pid"] == pid, "the scoped child proc directory changed PID")
            command_raw = bounded(descriptor, "cmdline")
            command = [part.decode() for part in command_raw.rstrip(b"\0").split(b"\0")] if command_raw else None
            status_after = bounded(descriptor, "status").decode()
            require(self.observer.parsed_uids(status_after) == [self.uid] * 4, "a scoped descendant changed user identifiers during command inspection")
            after = bounded(descriptor, "stat").decode()
            actual = parsed_identity(after)
            require(all(actual[key] == identity[key] for key in ("pid", "parent_pid", "process_start_time")),
                "the scoped child birth or parent changed during observation")
            return {**actual, "uids": [self.uid] * 4, "cgroup": cgroup, "stat_before": before, "stat_after": after,
                "status": status, "status_after": status_after, "command": command, "command_raw_sha256": hashlib.sha256(command_raw).hexdigest(),
                "observed_at_seconds": time.time()}
        finally:
            os.close(descriptor)

    def pin(self, pid, parent):
        """Pin only a positively listed own child or adopted child in this verified closed scope."""
        if pid in self.handles:
            current = self.snapshot(pid)
            require(current["process_start_time"] == self.nodes[pid]["initial"]["process_start_time"], "a pinned scoped PID was reused")
            self.nodes[pid]["last"] = current
            return current
        require(len(self.handles) < MAX_NODES, "scoped process count exceeded its explicit bound")
        handle = self.observer.open_process_handle(pid)
        try:
            current = self.snapshot(pid)
            require(current["parent_pid"] == parent, "the positive parent edge changed before pinning")
        except BaseException:
            self.observer.close_process_handle(handle)
            raise
        self.handles[pid] = handle
        self.nodes[pid] = {"initial": current, "last": current, "positive_parent_pid": parent}
        self.event("pinned_positive_child", pid=pid, birth=current["process_start_time"], parent_pid=parent)
        return current

    def scan(self):
        """Discover only producer children, adopted orphans and their positive owned child edges."""
        queue, seen, root_rows, root_children = [self.pid], set(), None, None
        while queue:
            parent = queue.pop(0)
            if parent in seen:
                continue
            seen.add(parent)
            if parent != self.pid and self.observer.handle_exited(self.handles[parent]):
                continue
            try:
                children, rows = self.child_lists(parent)
                if parent == self.pid:
                    root_rows, root_children = rows, children
                for pid in children:
                    try:
                        actual = self.pin(pid, parent)
                        if actual["state"] not in {"Z", "X", "x"}:
                            queue.append(pid)
                    except (FileNotFoundError, ProcessLookupError):
                        self.event("positive_child_disappeared_before_pin", pid=pid, parent_pid=parent)
            except (FileNotFoundError, ProcessLookupError):
                if parent == self.pid:
                    raise
        require(root_rows is not None and root_children is not None, "actual producer child lists were not read")
        return root_children, root_rows

    def reap(self):
        """Reap only positively pinned adopted direct children; preserve each actual Popen root wait."""
        children, unused = self.child_lists(self.pid)
        for pid in children:
            if pid not in self.handles:
                self.pin(pid, self.pid)
            if not self.observer.handle_exited(self.handles[pid]):
                continue
            if pid in self.roots:
                self.record_wait(self.roots[pid]["process"])
            else:
                actual, status = os.waitpid(pid, os.WNOHANG)
                if actual:
                    require(actual == pid, "the actual adopted-child wait returned another PID")
                    self.event("reaped_actual_adopted_child", pid=pid,
                        birth=self.nodes[pid]["initial"]["process_start_time"], wait_status=status,
                        exit_code=os.waitstatus_to_exitcode(status))

    def signal_known(self, number):
        """Signal only positive own-UID descendants through their already pinned process handles."""
        for pid, handle in reversed(list(self.handles.items())):
            if self.observer.handle_exited(handle):
                continue
            actual = self.snapshot(pid)
            require(actual["process_start_time"] == self.nodes[pid]["initial"]["process_start_time"], "a scoped signal target changed birth")
            key = (pid, actual["process_start_time"], int(number))
            if key in self.signals_sent:
                continue
            if self.deadline is not None and self.deadline.remaining('cleanup', allow_zero=True) <= 0:
                break
            self.observer.signal_handle(handle, number)
            self.signals_sent.add(key)
            self.interventions.append({"pid": pid, "birth": actual["process_start_time"], "signal": int(number), "at_seconds": time.time()})

    def closure(self):
        """Require actual empty own-child lists, all process handles exited, all waits, and ECHILD."""
        children, rows = self.scan()
        self.reap()
        children, rows = self.child_lists(self.pid)
        if children or any(not self.observer.handle_exited(handle) for handle in self.handles.values()):
            return None
        for root in self.roots.values():
            self.record_wait(root["process"])
        try:
            result = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError as error:
            require(error.errno == errno.ECHILD, "terminal wait returned an error other than ECHILD")
            return {"fresh_own_child_lists": rows, "actual_echild": True, "errno": errno.ECHILD,
                "all_pinned_handles_exited": True, "all_actual_metadata_roots_waited": len(self.waits) == len(self.roots),
                "observed_at_seconds": time.time()}
        self.event("terminal_wait_not_echild", actual_result=repr(result))
        return None


@contextmanager
def work_alarm(budget):
    """Interrupt synchronous metadata work at the same work end, leaving one cleanup reservation."""
    require(threading.active_count() == 1 and signal.getitimer(signal.ITIMER_REAL) == (0., 0.),
        "metadata work cannot replace another timer or use concurrent threads")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(number, frame):
        raise TimeoutError("the shared profile metadata work deadline expired")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, budget.remaining())
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous)


def validate_scope(proof, budget, observer, owner):
    """Check saved terminal kernel closure without a live process query or native operation."""
    require(type(proof) is dict and type(proof.get("schema_version")) is int and proof["schema_version"] == 1
        and proof.get("kind") == "profile_metadata_kernel_scope_closure"
        and proof.get("origin_sha256") == digest(encoded(budget.origin))
        and proof.get("all_metadata_descendants_exited") is True and proof.get("within_shared_budget") is True
        and proof.get("successful_no_intervention") is True and proof.get("interventions") == []
        and proof.get("read_errors") == [] and proof.get("scientific_credit") == 0
        and proof.get("operational_admission") is False, "metadata scope lacks a successful exact shared-budget closure")
    baseline, terminal = proof["baseline"], proof["terminal"]
    require(observer.stable(baseline["owner"]) == observer.stable(owner)
        and baseline["actual_echild"] is True and terminal["actual_echild"] is True
        and type(terminal["errno"]) is int and terminal["errno"] == errno.ECHILD
        and terminal["all_pinned_handles_exited"] is True
        and terminal["all_actual_metadata_roots_waited"] is True, "metadata actual kernel ECHILD/root waits incomplete")
    for rows in (baseline["empty_children"], terminal["fresh_own_child_lists"]):
        require(type(rows) is list and len(rows) == 1 and rows[0]["tid"] == owner["pid"]
            and rows[0]["raw"].strip() == ""
            and digest(rows[0]["raw"].encode()) == rows[0]["sha256"], "fresh empty own-child bytes missing")
        parent = rows[0]["parent_read_identity"]
        require(parent["expected_pid"] == owner["pid"] and parent["expected_birth"] == owner["process_start_time"]
            and parent["pidfd_live_before"] is True and parent["pidfd_live_after"] is True,
            "metadata child lists are not bound to their pinned live parent birth")
        for name in ("before", "after"):
            row = parent[name];identity = parsed_identity(row["stat"])
            require(identity["pid"] == owner["pid"] and identity["process_start_time"] == owner["process_start_time"]
                and observer.parsed_uids(row["status"]) == row["uids"] == owner["uids"]
                and row["cgroup"] == owner["cgroup"] and row["pidfd_live"] is True,
                "metadata saved child-list parent changed raw birth, owner or cgroup")
    previous = baseline["previous_subreaper"]["value"]
    require(type(previous) is int and previous in (0, 1), "metadata prior subreaper unknown")
    for item, request, value in ((baseline["set_subreaper"], "PR_SET_CHILD_SUBREAPER", 1),
            (baseline["verified_subreaper"], "PR_GET_CHILD_SUBREAPER", 1),
            (proof["restored"], "PR_SET_CHILD_SUBREAPER", previous),
            (proof["verified_restore"], "PR_GET_CHILD_SUBREAPER", previous)):
        require(item["request"] == request and type(item["value"]) is int and item["value"] == value
            and type(item["returncode"]) is int and item["returncode"] == 0
            and type(item["errno"]) is int and item["errno"] == 0, "metadata subreaper restoration is not causal")
    for key, node in proof["positive_nodes"].items():
        require(str(int(key)) == key, "metadata scoped PID key malformed")
        for name in ("initial", "last"):
            row = node[name];before, after = parsed_identity(row["stat_before"]), parsed_identity(row["stat_after"])
            require(before["pid"] == after["pid"] == row["pid"] == int(key)
                and before["process_start_time"] == after["process_start_time"] == row["process_start_time"]
                and before["parent_pid"] == after["parent_pid"] == row["parent_pid"]
                and observer.parsed_uids(row["status"]) == observer.parsed_uids(row["status_after"])
                == row["uids"] == owner["uids"] and row["cgroup"] == owner["cgroup"],
                "metadata positive-owned child raw identity changed")
        require(node["initial"]["process_start_time"] == node["last"]["process_start_time"]
            and node["positive_parent_pid"] == node["initial"]["parent_pid"], "metadata scoped parent edge changed")
    for key, row in proof["root_waits"].items():
        require(str(row["pid"]) == key and type(row["returncode"]) is int and row["returncode"] == 0
            and proof["positive_nodes"][key]["initial"]["process_start_time"] == row["birth"],
            "metadata exact root did not return normally")
    budget.verify(baseline["budget"]);budget.verify(proof["budget"],
        cleanup=proof["budget"]["cleanup_started_monotonic"] is not None)
    return proof
