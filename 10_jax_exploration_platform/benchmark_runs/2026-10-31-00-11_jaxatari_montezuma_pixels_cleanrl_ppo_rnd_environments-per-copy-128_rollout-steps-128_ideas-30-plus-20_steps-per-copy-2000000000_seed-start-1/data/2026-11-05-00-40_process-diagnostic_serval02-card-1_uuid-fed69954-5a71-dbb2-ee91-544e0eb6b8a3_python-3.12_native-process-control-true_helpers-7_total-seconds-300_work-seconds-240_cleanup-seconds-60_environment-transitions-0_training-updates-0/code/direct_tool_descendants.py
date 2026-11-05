"""Keep one producer alive until its closed Linux tool-child scope has no descendants."""
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

PROC_ROOT = Path("/proc")
MAX_NODES = 256
MAX_EVENTS = 512
MAX_BYTES = 1024 * 1024
_NATIVE = None
_REQUEST_COUNTS = {"PR_GET_CHILD_SUBREAPER": 0, "PR_SET_CHILD_SUBREAPER": 0}


def require(value, message):
    """Reject unknown native process ownership before allowing the producer to finish."""
    if not value:
        raise ValueError(message)


def prepare_native_control():
    """Bind only own-process libc prctl; perform no kernel state change or GPU operation."""
    global _NATIVE
    if _NATIVE is None:
        require(sys.platform == "linux" and ctypes.sizeof(ctypes.c_ulong) == ctypes.sizeof(ctypes.c_void_p),
            "Linux native long/pointer process-control ABI is required")
        handle = ctypes.CDLL(None, use_errno=True)
        function = handle.prctl
        function.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        function.restype = ctypes.c_int
        _NATIVE = (handle, function)
    return {"target": None, "symbol": "prctl", "scope": "own-process Linux child-subreaper flag only",
        "external_shared_library_loaded": False, "gpu_or_cuda_qualification": False}


def child_subreaper(value=None):
    """Get or set only this process's child-subreaper attribute with exact kernel returns."""
    require(_NATIVE is not None, "prepare the narrow native process-control pointer before installing the metadata guard")
    function = _NATIVE[1]
    ctypes.set_errno(0)
    if value is None:
        _REQUEST_COUNTS["PR_GET_CHILD_SUBREAPER"] += 1
        result = ctypes.c_int(-1)
        code = function(37, ctypes.cast(ctypes.byref(result), ctypes.c_void_p).value, 0, 0, 0)
        error = ctypes.get_errno()
        require(code == 0 and result.value in (0, 1), "PR_GET_CHILD_SUBREAPER failed: " + str(error))
        return {"request": "PR_GET_CHILD_SUBREAPER", "returncode": code, "errno": error, "value": result.value}
    require(type(value) is int and value in (0, 1), "one typed child-subreaper flag is required")
    _REQUEST_COUNTS["PR_SET_CHILD_SUBREAPER"] += 1
    code = function(36, value, 0, 0, 0)
    error = ctypes.get_errno()
    require(code == 0, "PR_SET_CHILD_SUBREAPER failed: " + str(error))
    return {"request": "PR_SET_CHILD_SUBREAPER", "returncode": code, "errno": error, "value": value}


def parsed_identity(raw):
    """Retain birth and parent identities for both living children and unreaped zombies."""
    first, remaining = raw.rsplit(") ", 1)
    fields = remaining.split()
    pid = first.split(" (", 1)[0]
    require(pid.isdecimal() and len(fields) >= 20 and all(fields[index].isdecimal() for index in (1, 2, 3, 19)),
        "incomplete actual child process stat")
    return {"pid": int(pid), "parent_pid": int(fields[1]), "process_group": int(fields[2]),
        "session": int(fields[3]), "process_start_time": fields[19], "state": fields[0]}


def bounded(directory, name):
    """Read a bounded own-process proc file through an already pinned directory."""
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        raw = os.read(descriptor, MAX_BYTES + 1)
        require(len(raw) <= MAX_BYTES, "actual child proc bytes exceeded their bound")
        return raw
    finally:
        os.close(descriptor)


class OwnedToolScope:
    """Adopt orphans from a single closed tool-spawn scope and require kernel terminal proof."""
    def __init__(self, directory, plan, observer, owner, check_descriptor, *, grace_seconds=120, deadline=None):
        """Use a fresh single-threaded producer, exact tool path and already verified standing descriptor."""
        self.directory, self.plan, self.observer = Path(directory), plan, observer
        self.owner, self.check_descriptor = owner, check_descriptor
        self.uid, self.pid, self.birth, self.cgroup = owner["uids"][0], owner["pid"], owner["process_start_time"], owner["cgroup"]
        require(('diagnostic_time_budget' in plan) == (deadline is not None),
            'a declared diagnostic scope requires its actual shared deadline')
        self.grace_seconds, self.deadline = grace_seconds, deadline
        self.budget_inconclusive_published = False
        if deadline is not None:deadline.remaining('work')
        self.nodes, self.handles, self.roots, self.waits, self.events = {}, {}, {}, {}, []
        self.previous = None
        self.active, self.closed, self.failed, self.pending_command = False, False, False, None
        self.interventions, self.read_errors = [], []
        self.signals_sent = set()
        self.recording_failed = False
        self.last_progress, self.sequence = 0., 0
        require(self.pid == os.getpid(), "only the actual producer may create its own descendant scope")
        self.producer_handle = self.observer.open_process_handle(self.pid)
        self.native_bootstrap = prepare_native_control()
        self.request_counts_before = dict(_REQUEST_COUNTS)

    def event(self, kind, **fields):
        """Keep bounded process-control evidence without inventing absent children."""
        record = {"kind": kind, "at_seconds": time.time(), **fields}
        if len(self.events) < MAX_EVENTS:
            self.events.append(record)
        else:
            self.failed = True
        return record

    def progress(self, phase, *, force=False):
        """Replace only this attempt's bounded cleanup progress while retaining terminal records separately."""
        if not force and time.monotonic() - self.last_progress < 1:
            return
        if self.recording_failed:
            return
        self.last_progress = time.monotonic()
        value = {"schema_version": 1, "kind": "direct_tool_descendant_cleanup_progress", "phase": phase,
            "producer_pid": self.pid, "producer_birth": self.birth, "subreaper_active": self.active,
            "known_nodes": len(self.nodes), "known_roots": len(self.roots), "events": self.events,
            "interventions": self.interventions, "read_errors": self.read_errors,
            "standing_descriptor_must_remain_open": not self.closed, "scientific_credit": 0,
            "native_process_control_only": True, "gpu_or_cuda_qualification": False}
        path = self.directory / "tool_descendant_cleanup_progress.json"
        self.sequence += 1
        temporary = path.with_name("." + path.name + ".pending-" + str(os.getpid()) + "-" + str(self.sequence))
        try:
            with temporary.open("xb") as stream:
                stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
                stream.flush(); os.fsync(stream.fileno())
            temporary.replace(path)
        except OSError as error:
            self.failed = True
            self.recording_failed = True
            if len(self.read_errors) < 32:
                self.read_errors.append({"kind": "progress_write_failed", "error": repr(error)})

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

    def audit(self, event, values):
        """Permit only declared tool Popen calls during this single-threaded closed scope."""
        if self.active and event == "subprocess.Popen":
            require(threading.get_ident() == self.thread and self.pending_command is not None
                and list(values[1]) == self.pending_command and values[0] == self.plan["profile_tool"]["path"],
                "an unrelated Popen is forbidden while the producer owns its tool-child scope")

    def __enter__(self):
        """Verify the kernel child-subreaper and actual empty baseline before every tool spawn."""
        require(self.pid == os.getpid() and threading.active_count() == 1
            and signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL, "a fresh single-threaded default-SIGCHLD producer is required")
        if self.deadline is not None:self.deadline.remaining('work')
        descriptor_before = self.check_descriptor()
        children, rows = self.child_lists(self.pid)
        require(not children and len(rows) == 1, "the tool scope requires an actual empty producer-child baseline")
        try:
            empty = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError as error:
            require(error.errno == errno.ECHILD, "initial child terminal check returned another error")
            baseline_echild = True
        else:
            require(False, "empty child lists require an actual initial ECHILD proof, not a WNOHANG result: " + repr(empty))
        self.previous = child_subreaper()
        changed = child_subreaper(1)
        verified = child_subreaper()
        require(verified["value"] == 1, "the producer did not become an actual child subreaper")
        self.thread = threading.get_ident()
        self.active = True
        sys.addaudithook(self.audit)
        self.baseline = {"producer": self.owner, "own_child_lists": rows, "actual_echild": baseline_echild,
            "standing_descriptor_before": descriptor_before,
            "native_bootstrap": self.native_bootstrap, "previous_subreaper": self.previous,
            "set_subreaper": changed, "verified_subreaper": verified, "single_threaded_closed_scope": True,
            "established_at_seconds": time.time(), "native_request_counts_before": self.request_counts_before}
        if self.deadline is not None:self.baseline['diagnostic_deadline'] = self.deadline.snapshot()
        self.observer.publish(self.directory / "tool_descendant_scope_begin.json", self.baseline)
        self.progress("scope_established", force=True)
        return self

    def spawn(self, command, **keywords):
        """Create exactly one recorded tool root with the standing descriptor closed in its children."""
        require(self.active and list(command)[0] == self.plan["profile_tool"]["path"], "only the recorded actual tool may spawn in this scope")
        require(not keywords.get("shell", False) and keywords.get("preexec_fn") is None
            and keywords.get("executable") is None and keywords.get("pass_fds", ()) == (), "tool spawning cannot add shell, executable or descriptor overrides")
        keywords["close_fds"], keywords["pass_fds"] = True, ()
        if self.deadline is not None:self.deadline.remaining('work')
        self.pending_command = list(command)
        try:
            child = subprocess.Popen(command, **keywords)
        finally:
            self.pending_command = None
        self.roots[child.pid] = {"process": child, "command": list(command), "started_at_seconds": time.time()}
        self.pin(child.pid, self.pid)
        self.scan()
        return child

    def record_wait(self, child):
        """Save the actual Popen.wait result for this exact previously spawned root."""
        require(child.pid in self.roots and self.roots[child.pid]["process"] is child, "only an actual scoped Popen object may be waited")
        returned = child.wait(timeout=0)
        if child.pid not in self.waits:
            root = self.roots[child.pid]
            value = {"kind": "actual_scoped_tool_popen_wait", "pid": child.pid,
                "birth": self.nodes[child.pid]["initial"]["process_start_time"], "command": root["command"],
                "returncode": returned, "started_at_seconds": root["started_at_seconds"], "finished_at_seconds": time.time()}
            self.waits[child.pid] = value
            self.observer.publish(self.directory / ("tool_root_wait_" + str(child.pid) + ".json"), value)
        return returned

    def wait(self, child, timeout):
        """Observe startup and adopted descendants while waiting on the exact actual tool root."""
        deadline = time.monotonic() + (timeout if self.deadline is None else self.deadline.timeout(timeout))
        while child.poll() is None:
            self.scan()
            require(time.monotonic() < deadline, "the scoped tool root exceeded its actual active deadline")
            time.sleep(.05)
        self.scan()
        if self.deadline is not None:self.deadline.remaining('work')
        return self.record_wait(child)

    def run(self, command, *, capture_output=False, timeout=None, check=False, **keywords):
        """Run a recorded metadata tool command while continuously retaining descendant ownership."""
        if capture_output:
            require("stdout" not in keywords and "stderr" not in keywords, "capture_output cannot override explicit streams")
            keywords.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        child = self.spawn(command, **keywords)
        effective = timeout if self.deadline is None else self.deadline.timeout(timeout)
        deadline = time.monotonic() + effective if effective is not None else None
        while True:
            self.scan()
            remaining = .05 if deadline is None else min(.05, max(.001, deadline - time.monotonic()))
            try:
                output, errors = child.communicate(timeout=remaining)
                break
            except subprocess.TimeoutExpired as error:
                if deadline is not None and time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, timeout, error.output, error.stderr) from error
        if self.deadline is not None:self.deadline.remaining('work')
        returned = self.record_wait(child)
        result = subprocess.CompletedProcess(command, returned, output, errors)
        if check and returned:
            raise subprocess.CalledProcessError(returned, command, output, errors)
        return result

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
                "all_pinned_handles_exited": True, "all_actual_tool_roots_waited": len(self.waits) == len(self.roots),
                "observed_at_seconds": time.time()}
        self.event("terminal_wait_not_echild", actual_result=repr(result))
        return None

    def diagnostic_budget_status(self, *, kernel_closure_verified, snapshot=None):
        """Record expiry from the current retained snapshot, including slow closure and custody checks."""
        snapshot = self.deadline.snapshot() if snapshot is None else snapshot
        end = snapshot['cleanup_end_monotonic']
        within = end is not None and snapshot['observed_monotonic'] < min(end, snapshot['total_end_monotonic'])
        if not within:
            self.failed = True
            if not self.budget_inconclusive_published:
                self.observer.publish(self.directory / 'tool_descendant_budget_inconclusive.json', {
                    'producer_pid': self.pid, 'producer_birth': self.birth, 'completed': False,
                    'deadline': snapshot, 'descriptor_release_attempted': False,
                    'kernel_closure_verified': kernel_closure_verified,
                    'no_new_tool_or_signal_requests_after_expiry': True,
                    'reason': 'diagnostic completion checks exceeded the reserved cleanup window',
                    'scientific_credit': 0, 'admission_granted': False})
                self.budget_inconclusive_published = True
        return within and not self.budget_inconclusive_published

    def finish(self, failed):
        """Never return toward descriptor closure while any tool descendant may still exist."""
        if not hasattr(self, "cleanup_started"):
            self.cleanup_started = time.monotonic()
        started = self.cleanup_started
        if self.deadline is not None:self.deadline.begin_cleanup()
        terminal = None
        requested = failed
        while terminal is None:
            try:
                self.check_descriptor()
            except BaseException as error:
                self.failed = True
                if len(self.read_errors) < 32:
                    self.read_errors.append({"kind": "standing_descriptor_check_failed", "error": repr(error)})
            try:
                verified = child_subreaper()
                require(verified["value"] == 1, "the live producer lost its child-subreaper flag")
                self.scan()
                if self.deadline is None:
                    if requested or time.monotonic() - started >= self.grace_seconds:
                        self.signal_known(signal.SIGTERM if time.monotonic() - started < self.grace_seconds else signal.SIGKILL)
                else:
                    now = time.monotonic()
                    within_budget = self.diagnostic_budget_status(kernel_closure_verified=False)
                    if within_budget and (requested or now >= started + 15):
                        self.signal_known(signal.SIGTERM if now < started + 30 else signal.SIGKILL)
                terminal = self.closure()
                if self.deadline is not None:
                    self.diagnostic_budget_status(kernel_closure_verified=terminal is not None)
            except BaseException as error:
                self.failed = True
                requested = True
                if len(self.read_errors) < 32:
                    self.read_errors.append({"kind": "cleanup_unverified", "error": repr(error)})
            self.progress("cleanup_verified" if terminal is not None else "retaining_descriptor_until_kernel_closure")
            if terminal is None:
                time.sleep(.05)
        descriptor_at_terminal = None
        try:
            descriptor_at_terminal = self.check_descriptor()
        except BaseException as error:
            self.failed = True
            self.read_errors.append({"kind": "standing_descriptor_terminal_check_failed", "error": repr(error)})
        if self.deadline is not None:self.diagnostic_budget_status(kernel_closure_verified=True)
        self.closed = True
        self.terminal = {"schema_version": 1, "kind": "direct_tool_descendant_kernel_terminal_closure",
            "begin": self.observer.receipt(self.directory / "tool_descendant_scope_begin.json"),
            "producer_pid": self.pid, "producer_birth": self.birth, "terminal": terminal,
            "actual_root_waits": self.waits, "positive_nodes": self.nodes, "events": self.events,
            "interventions": self.interventions, "read_errors": self.read_errors,
            "standing_descriptor_at_terminal": descriptor_at_terminal,
            "all_tool_descendants_exited_before_descriptor_release": descriptor_at_terminal is not None
                and not any(row["kind"].startswith("standing_descriptor") for row in self.read_errors),
            "all_historical_descendants_identified": False, "scientific_credit": 0, "gpu_or_cuda_qualification": False,
            "native_request_counts_before_restore": dict(_REQUEST_COUNTS)}
        if self.deadline is not None:
            final_snapshot = self.deadline.snapshot()
            within = self.diagnostic_budget_status(kernel_closure_verified=True, snapshot=final_snapshot)
            self.terminal['diagnostic_deadline'] = final_snapshot
            self.terminal['within_diagnostic_budget'] = within and descriptor_at_terminal is not None
        self.observer.publish(self.directory / "tool_descendant_scope_terminal.json", self.terminal)
        restored = child_subreaper(self.previous["value"])
        actual = child_subreaper()
        require(actual["value"] == self.previous["value"], "the producer child-subreaper flag was not restored after closure")
        self.observer.publish(self.directory / "tool_descendant_scope_restored.json", {"terminal": self.observer.receipt(
            self.directory / "tool_descendant_scope_terminal.json"), "restore": restored, "verified": actual,
            "native_prctl_request_counts": {key: _REQUEST_COUNTS[key] - self.request_counts_before[key] for key in _REQUEST_COUNTS},
            "native_process_control_only": True, "gpu_or_cuda_qualification": False})
        self.active = False
        for handle in self.handles.values():
            self.observer.close_process_handle(handle)
        self.observer.close_process_handle(self.producer_handle)
        self.producer_handle = None
        self.progress("kernel_closure_verified_and_native_flag_restored", force=True)

    def __exit__(self, kind, error, traceback):
        """Restore process control only after actual closure, keeping the GPU holder alive on unknown cleanup."""
        while not self.closed:
            try:
                self.finish(error is not None or self.failed)
            except BaseException as cleanup_error:
                self.failed = True
                if self.closed:
                    raise
                if len(self.read_errors) < 32:
                    self.read_errors.append({"kind": "cleanup_interrupted_before_terminal_proof", "error": repr(cleanup_error)})
        if error is None and (self.failed or self.interventions):
            raise RuntimeError("the tool scope required cleanup intervention or lost evidence; retain zero-credit raw records")
        return False


def validate_saved_scope(directory, plan, observer, producer, root_pid, root_birth, check_descriptor, *, diagnostic_deadline=None):
    """Read kernel-scope closure evidence without querying any historical process or changing native state."""
    require(('diagnostic_time_budget' in plan) == (diagnostic_deadline is not None),
        'saved budgeted scope evidence requires its exact diagnostic origin')
    directory = Path(directory)
    begin_ref = observer.receipt(directory / "tool_descendant_scope_begin.json")
    begin = observer.read_json(begin_ref["path"])
    terminal_ref = observer.receipt(directory / "tool_descendant_scope_terminal.json")
    terminal = observer.read_json(terminal_ref["path"])
    restored_ref = observer.receipt(directory / "tool_descendant_scope_restored.json")
    restored = observer.read_json(restored_ref["path"])
    if diagnostic_deadline is not None:
        require(not (directory / 'tool_descendant_budget_inconclusive.json').exists()
            and terminal.get('within_diagnostic_budget') is True,
            'expired or unverified diagnostic cleanup cannot become successful scope evidence')
        diagnostic_deadline.verify_snapshot(begin['diagnostic_deadline'], phase='work')
        diagnostic_deadline.verify_snapshot(terminal['diagnostic_deadline'], phase='total', cleanup_required=True)
    observer.validate_snapshot(begin["producer"], producer["uids"][0], producer["cgroup"])
    require(observer.stable(begin["producer"]) == observer.stable(producer)
        and begin.get("actual_echild") is True and begin.get("single_threaded_closed_scope") is True,
        "the actual producer did not establish its own empty closed tool scope")
    native = begin["native_bootstrap"]
    require(native.get("target") is None and native.get("symbol") == "prctl"
        and native.get("external_shared_library_loaded") is False and native.get("gpu_or_cuda_qualification") is False,
        "the native bootstrap must be limited to own-process prctl, never GPU qualification")
    for item, request, value in ((begin["set_subreaper"], "PR_SET_CHILD_SUBREAPER", 1),
        (begin["verified_subreaper"], "PR_GET_CHILD_SUBREAPER", 1)):
        require(item.get("request") == request and type(item.get("returncode")) is int and item["returncode"] == 0
            and type(item.get("errno")) is int and item["errno"] == 0
            and type(item.get("value")) is int and item["value"] == value, "actual child-subreaper setup was not verified")
    previous = begin["previous_subreaper"]
    require(previous["request"] == "PR_GET_CHILD_SUBREAPER" and type(previous["value"]) is int
        and previous["value"] in (0, 1) and type(previous["returncode"]) is int and previous["returncode"] == 0
        and type(previous.get("errno")) is int and previous["errno"] == 0,
        "the previous actual producer subreaper flag is unknown")
    require(type(terminal.get("schema_version")) is int and terminal["schema_version"] == 1
        and terminal.get("kind") == "direct_tool_descendant_kernel_terminal_closure" and terminal["begin"] == begin_ref
        and terminal["producer_pid"] == producer["pid"] and terminal["producer_birth"] == producer["process_start_time"]
        and terminal.get("all_tool_descendants_exited_before_descriptor_release") is True
        and terminal.get("interventions") == [] and terminal.get("read_errors") == [],
        "ordinary saved evidence requires actual complete descendant exit and uninterrupted producer custody")
    final = terminal["terminal"]
    require(final.get("actual_echild") is True and type(final.get("errno")) is int and final["errno"] == errno.ECHILD
        and final.get("all_pinned_handles_exited") is True and final.get("all_actual_tool_roots_waited") is True,
        "empty child lists cannot replace the actual terminal ECHILD and process-handle proof")
    for rows in (begin["own_child_lists"], final["fresh_own_child_lists"]):
        require(type(rows) is list and len(rows) == 1 and rows[0]["tid"] == producer["pid"],
            "the closed producer must retain one complete actual own-thread child-list observation")
        raw = rows[0]["raw"].encode("ascii")
        require(raw.strip() == b"" and hashlib.sha256(raw).hexdigest() == rows[0]["sha256"],
            "actual empty producer child-list bytes are missing or changed")
        row, parent = rows[0], rows[0]["parent_read_identity"]
        require(type(parent["expected_pid"]) is int and parent["expected_pid"] == producer["pid"]
            and parent["expected_birth"] == producer["process_start_time"]
            and all(type(parent[key]) is int and parent[key] > 0 for key in ("proc_directory_device", "proc_directory_inode"))
            and parent["pidfd_live_before"] is True and parent["pidfd_live_after"] is True,
            "the actual child-list read was not bound to a pinned live producer directory")
        for identity in (parent["before"], parent["after"]):
            actual = parsed_identity(identity["stat"])
            require(type(identity["pid"]) is int and actual["pid"] == identity["pid"] == producer["pid"]
                and actual["process_start_time"] == identity["process_start_time"] == producer["process_start_time"]
                and actual["state"] not in {"Z", "X", "x"} and identity["pidfd_live"] is True
                and observer.parsed_uids(identity["status"]) == identity["uids"] == producer["uids"]
                and identity["cgroup"] == producer["cgroup"],
                "the saved child-list parent changed PID, birth, UID, cgroup or liveness during its read")
        first, last = parsed_identity(row["thread_stat_before"]), parsed_identity(row["thread_stat_after"])
        require(first["pid"] == last["pid"] == producer["pid"]
            and first["process_start_time"] == last["process_start_time"] == producer["process_start_time"],
            "the saved own-thread child list changed thread identity during its read")
    waits = terminal["actual_root_waits"]
    require(type(waits) is dict and str(root_pid) in waits and len(waits) >= 1,
        "the actual capture Popen root is absent from terminal scope waits")
    for key, row in waits.items():
        require(type(key) is str and key.isdecimal() and str(int(key)) == key
            and type(row["pid"]) is int and row["pid"] == int(key)
            and row["kind"] == "actual_scoped_tool_popen_wait" and type(row["returncode"]) is int and row["returncode"] == 0
            and row["command"][0] == plan["profile_tool"]["path"], "the saved tool scope contains a failed or unrelated Popen root")
        expected = observer.receipt(directory / ("tool_root_wait_" + key + ".json"))
        require(observer.read_json(expected["path"]) == row, "an actual scoped Popen wait record changed")
    require(waits[str(root_pid)]["birth"] == root_birth, "the actual scoped capture root changed birth")
    nodes = terminal["positive_nodes"]
    require(type(nodes) is dict and str(root_pid) in nodes and 0 < len(nodes) <= MAX_NODES,
        "actual positive root/process records are missing from the saved tool scope")
    for key, node in nodes.items():
        require(type(key) is str and key.isdecimal() and str(int(key)) == key,
            "saved scoped PID keys must use canonical actual digits")
        for name in ("initial", "last"):
            raw = node[name]
            before, after = parsed_identity(raw["stat_before"]), parsed_identity(raw["stat_after"])
            require(before["pid"] == after["pid"] == raw["pid"] == int(key)
                and before["process_start_time"] == after["process_start_time"] == raw["process_start_time"]
                and before["parent_pid"] == after["parent_pid"] == raw["parent_pid"]
                and observer.parsed_uids(raw["status"]) == observer.parsed_uids(raw["status_after"])
                == raw["uids"] == [producer["uids"][0]] * 4 and raw["cgroup"] == producer["cgroup"],
                "saved positive scoped identity changed raw UID, cgroup, PID or birth")
        require(node["initial"]["process_start_time"] == node["last"]["process_start_time"]
            and node["positive_parent_pid"] == node["initial"]["parent_pid"],
            "the saved positive scoped edge or pinned birth changed")
    check_descriptor(begin["standing_descriptor_before"])
    check_descriptor(terminal["standing_descriptor_at_terminal"])
    require(restored["terminal"] == terminal_ref and restored.get("native_process_control_only") is True
        and restored.get("gpu_or_cuda_qualification") is False,
        "restoration must follow the exact verified native terminal closure")
    for item, request in ((restored["restore"], "PR_SET_CHILD_SUBREAPER"),
        (restored["verified"], "PR_GET_CHILD_SUBREAPER")):
        require(item.get("request") == request and type(item.get("returncode")) is int and item["returncode"] == 0
            and type(item.get("errno")) is int and item["errno"] == 0
            and type(item.get("value")) is int and item["value"] == previous["value"],
            "the recorded restore request and actual verified flag must equal the previous typed native state")
    before_counts = begin["native_request_counts_before"]
    terminal_counts = terminal["native_request_counts_before_restore"]
    counts = restored["native_prctl_request_counts"]
    keys = {"PR_GET_CHILD_SUBREAPER", "PR_SET_CHILD_SUBREAPER"}
    for value in (before_counts, terminal_counts, counts):
        require(type(value) is dict and set(value) == keys
            and all(type(item) is int and item >= 0 for item in value.values()),
            "native control counts require exact nonnegative integer request counters")
    measured = {key: terminal_counts[key] - before_counts[key] for key in keys}
    require(measured["PR_SET_CHILD_SUBREAPER"] == 1 and measured["PR_GET_CHILD_SUBREAPER"] >= 3
        and counts["PR_SET_CHILD_SUBREAPER"] == measured["PR_SET_CHILD_SUBREAPER"] + 1
        and counts["PR_GET_CHILD_SUBREAPER"] == measured["PR_GET_CHILD_SUBREAPER"] + 1,
        "the explicit setup, live checks and final restore/get requests must join their actual counter snapshots")
    return {"tool_descendant_scope_begin": begin_ref, "tool_descendant_scope_terminal": terminal_ref,
        "tool_descendant_scope_restored": restored_ref, "actual_kernel_descendant_closure_verified": True}
