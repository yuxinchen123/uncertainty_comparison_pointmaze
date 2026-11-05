"""Provide explicitly selected Linux pidfd calls when Python did not build its wrappers."""
import ctypes
import hashlib
import os
from pathlib import Path
import re
import select
import signal
import stat
import sys
import time

KIND = "linux-x86-64-libc-syscall-pidfd-version-1"
HEADERS = ("/usr/include/unistd.h", "/usr/include/x86_64-linux-gnu/asm/unistd_64.h")
NUMBERS = {"pidfd_open": 434, "pidfd_send_signal": 424}
MAX_CALLS = 4096
MAX_DESCRIPTOR_EVENTS = 4 * MAX_CALLS


def require(value, message):
    """Reject an unknown process-control ABI or image without a numeric-PID signal fallback."""
    if not value:
        raise ValueError(message)


def strict_equal(left, right):
    """Require exact JSON field types as well as values, including integer-zero metadata."""
    if type(left) is not type(right):return False
    if type(right) is dict:
        return left.keys() == right.keys() and all(strict_equal(left[key], right[key]) for key in right)
    if type(right) is list:
        return len(left) == len(right) and all(strict_equal(a, b) for a, b in zip(left, right))
    return left == right


def checked_file(path, maximum, *, root_owned):
    """Hash one actual fixed metadata/image path through a stable regular-file descriptor."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path, "the process-control file must be canonical")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= maximum
            and (not root_owned or before.st_uid == 0 and before.st_mode & 0o022 == 0),
            "the public process-control file has unsupported ownership, mode or size")
        pieces, length = [], 0
        while True:
            part = os.read(descriptor, 1024 * 1024)
            if not part:break
            length += len(part)
            require(length <= maximum, "the process-control file grew beyond its explicit bound")
            pieces.append(part)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "the process-control file changed during its actual read")
        raw = b"".join(pieces)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw),
            "uid": before.st_uid, "mode": stat.S_IMODE(before.st_mode),
            "device": before.st_dev, "inode": before.st_ino}, raw
    finally:
        os.close(descriptor)


def parse_pidfd_fdinfo(raw, expected_inode, expected_pid=None):
    """Join duplicate-safe kernel fdinfo inode, process and octal CLOEXEC fields to one descriptor."""
    require(type(raw) is str and raw.isascii() and raw.endswith('\n') and '\r' not in raw and '\0' not in raw
        and len(raw.encode()) <= 16384, 'pidfd fdinfo must be one complete bounded ASCII record')
    require(type(expected_inode) is int and 0 < expected_inode <= 18446744073709551615
        and (expected_pid is None or type(expected_pid) is int and 0 < expected_pid <= 2147483647),
        'pidfd expected inode and process must use exact positive integer types')
    # Before: raw key/value lines; after: one value per kernel field, with no ignored duplicate spelling.
    fields = {}
    for line in raw.splitlines():
        key, separator, value = line.partition(':')
        require(separator == ':' and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) and key not in fields,
            'pidfd fdinfo contains a malformed or duplicate kernel field')
        fields[key] = value.strip(' \t')
    require({'ino', 'Pid', 'flags'} <= set(fields)
        and re.fullmatch(r'[1-9][0-9]*', fields['ino'])
        and re.fullmatch(r'(?:[1-9][0-9]*|-1)', fields['Pid'])
        and re.fullmatch(r'[0-7]{8,11}', fields['flags']), 'pidfd raw inode, process or octal flags are invalid')
    inode, pid, flags = int(fields['ino']), int(fields['Pid']), int(fields['flags'], 8)
    require(inode == expected_inode and (expected_pid is None or pid in (expected_pid, -1))
        and -1 <= pid <= 2147483647 and flags <= 4294967295 and flags & os.O_CLOEXEC == os.O_CLOEXEC,
        'pidfd raw inode, process or CLOEXEC flags differ from its retained descriptor')
    return {'inode': inode, 'pid': pid, 'flags': flags, 'close_on_exec': True}


def validate_abi():
    """Support only the observed Linux x86_64 LP64 ABI with explicit C argument widths."""
    actual = {"platform": sys.platform, "machine": os.uname().machine, "byteorder": sys.byteorder,
        "long_bytes": ctypes.sizeof(ctypes.c_long), "int_bytes": ctypes.sizeof(ctypes.c_int),
        "pointer_bytes": ctypes.sizeof(ctypes.c_void_p)}
    require(actual == {"platform": "linux", "machine": "x86_64", "byteorder": "little",
        "long_bytes": 8, "int_bytes": 4, "pointer_bytes": 8}, "the native pidfd provider requires exact Linux x86_64 LP64")
    return actual


def mapped_symbol(function):
    """Join an actual function pointer to its own executable libc mapping and captured file bytes."""
    pointer = ctypes.cast(function, ctypes.c_void_p).value
    require(type(pointer) is int and pointer > 0, "an actual non-null native symbol pointer is required")
    matches = []
    for raw in Path("/proc/self/maps").read_text().splitlines():
        fields = raw.split(maxsplit=5)
        if len(fields) != 6:continue
        first, last = (int(value, 16) for value in fields[0].split("-"))
        if first <= pointer < last:
            require("x" in fields[1] and Path(fields[5]).name == "libc.so.6" and not fields[5].endswith(" (deleted)"),
                "the actual process-control symbol is not in an executable libc image")
            path = Path(fields[5]).resolve(strict=True)
            record, unused = checked_file(path, 16 * 1024 * 1024, root_owned=True)
            info = path.stat()
            require(int(fields[4]) == info.st_ino and tuple(int(value, 16) for value in fields[3].split(":"))
                == (os.major(info.st_dev), os.minor(info.st_dev)), "the mapped libc file identity changed")
            matches.append({"image": record, "symbol_file_offset": pointer - first + int(fields[2], 16),
                "actual_pointer": pointer, "actual_mapping": raw})
    require(len(matches) == 1, "the actual process-control symbol has no unique mapped libc image")
    return matches[0]


def running_python_image():
    """Join declared interpreter bytes to the executable inode retained by the running kernel process."""
    # Refuse a replaced or deleted executable rather than trusting an unrelated metadata pathname.
    require(type(sys.executable) is str and Path(sys.executable).is_absolute(),
        "the declared Python executable must have an absolute pathname")
    declared = Path(sys.executable).resolve(strict=True)
    before_link = os.readlink('/proc/self/exe')
    before = Path('/proc/self/exe').stat()
    require(before_link == str(declared) and stat.S_ISREG(before.st_mode),
        "the declared Python pathname is not the actual running executable image")
    record, unused = checked_file(declared, 128 * 1024 * 1024, root_owned=False)
    after = Path('/proc/self/exe').stat()
    require(record['mode'] & 0o111 != 0 and (record['device'], record['inode']) == (before.st_dev, before.st_ino)
        == (after.st_dev, after.st_ino) and os.readlink('/proc/self/exe') == before_link,
        "the selected Python bytes do not match the stable actual running executable inode")
    record['actual_running_image_path'] = before_link
    return record


def current_contract():
    """Capture metadata only; this does not execute a pidfd syscall or qualify a target."""
    abi = validate_abi()
    headers = {}
    for name in HEADERS:
        path = Path(name).resolve(strict=True)
        record, raw = checked_file(path, 128 * 1024, root_owned=True)
        headers[name] = record
        if name.endswith("unistd_64.h"):
            for key, number in NUMBERS.items():
                values = re.findall(rb"^#define\s+__NR_" + key.encode() + rb"\s+(\d+)\s*$", raw, re.MULTILINE)
                require(values == [str(number).encode()], "the actual local x86_64 pidfd syscall declaration differs")
        else:
            require(re.search(rb"extern\s+long\s+int\s+syscall\s*\(long\s+int\s+__sysno,\s*\.\.\.\)", raw),
                "the actual libc variadic syscall declaration was not found")
    library = ctypes.CDLL(None, use_errno=True)
    symbols = {}
    actual = {}
    for name in ("syscall", "close"):
        item = mapped_symbol(getattr(library, name))
        symbols[name] = {key: item[key] for key in ("image", "symbol_file_offset")}
        actual[name] = item
    require(symbols["syscall"]["image"] == symbols["close"]["image"], "native process calls must use one actual pinned libc image")
    executable = running_python_image()
    return {"kind": KIND, "abi": abi, "syscall_numbers": dict(NUMBERS), "headers": headers,
        "symbols": symbols, "selected_python": executable}, {"symbols": actual, "sys_executable": sys.executable,
        "sys_version": sys.version, "pid": os.getpid(), "gpu_or_cuda_qualification": False}


class NativePidfdProvider:
    """Use retained kernel process descriptors through an explicit verified native backend."""
    def __init__(self, contract):
        """Validate the exact declared metadata and finish a real self-handle preflight before use."""
        require(type(contract) is dict and contract.get("kind") == KIND, "an explicit native pidfd contract is required")
        self.owner_pid, self.calls, self.handles = os.getpid(), [], {}
        self.descriptor_events = []
        self.normal_call_count, self.cleanup_call_count = 0, 0
        self.failed, self.preflight_passed, self.preflight_call_count = False, False, 0
        actual, self.bootstrap = current_contract()
        require(strict_equal(contract, actual), "the actual process-control ABI, headers, libc symbol image or selected Python changed")
        self.contract = contract
        self.library = ctypes.CDLL(None, use_errno=True)
        self.syscall = self.library.syscall
        self.syscall.argtypes, self.syscall.restype = [ctypes.c_long], ctypes.c_long
        self.close = self.library.close
        self.close.argtypes, self.close.restype = [ctypes.c_int], ctypes.c_int
        require(all(ctypes.cast(function, ctypes.c_void_p).value == self.bootstrap["symbols"][name]["actual_pointer"]
            for name, function in (("syscall", self.syscall), ("close", self.close))),
            "the actual native call pointer changed after mapped-image verification")
        handle = None
        try:
            try:
                handle = self._open(self.owner_pid, "self_preflight")
                self._signal(handle, 0, "self_preflight")
            finally:
                if handle is not None:self._close(handle, "self_preflight")
        except BaseException as error:
            self.failed = True
            error.native_process_control_evidence = self.evidence()
            raise
        self.preflight_passed = True
        self.preflight_call_count = len(self.calls)
        require(self.preflight_call_count == 3 and not self.handles, "self pidfd preflight did not close its actual descriptor")

    def check_owner(self, *, ready=True):
        """Disallow inherited provider state and any use before the actual self preflight."""
        require(os.getpid() == self.owner_pid, "native process-control state cannot be reused by another process")
        require(not ready or self.preflight_passed, "actual self pidfd preflight is required before managing children")

    def invoke(self, name, arguments, function, *, phase, cleanup_descriptor=None, descriptor_identity=None):
        """Bound ordinary calls while reserving one TERM, KILL and close for each opened descriptor."""
        self.check_owner(ready=False)
        # Each ordinary open reserves at most three additional records, never a general unbounded bypass.
        if cleanup_descriptor is None:
            require(type(MAX_CALLS) is int and MAX_CALLS > 0 and self.normal_call_count < MAX_CALLS,
                "native process-control ordinary call evidence exceeded its explicit bound")
            self.normal_call_count += 1
            budget = "ordinary"
        else:
            require(type(cleanup_descriptor) is int and cleanup_descriptor in self.handles,
                "cleanup evidence requires a descriptor opened by this provider")
            operation = "close" if name == "close" else arguments.get("signal")
            expected = {"fd": cleanup_descriptor} if operation == "close" else {
                "syscall": NUMBERS["pidfd_send_signal"], "fd": cleanup_descriptor,
                "signal": operation, "siginfo": None, "flags": 0}
            require((name == "close" and operation == "close" or name == "pidfd_send_signal"
                and type(operation) is int and operation in (int(signal.SIGTERM), int(signal.SIGKILL)))
                and strict_equal(arguments, expected)
                and operation in self.handles[cleanup_descriptor]["cleanup_remaining"],
                "each retained descriptor permits only one reserved TERM, KILL and close")
            self.handles[cleanup_descriptor]["cleanup_remaining"].remove(operation)
            self.cleanup_call_count += 1
            budget = "reserved_descriptor_cleanup"
        if descriptor_identity is not None:
            self.record_descriptor(name, arguments['fd'], len(self.calls), descriptor_identity)
        ctypes.set_errno(0)
        started = time.time()
        result = function()
        error = ctypes.get_errno()
        require(type(result) is int and type(error) is int, "the native process call did not return typed integers")
        self.calls.append({"sequence": len(self.calls), "name": name, "phase": phase, "arguments": arguments,
            "result": result, "errno": error, "started_at_seconds": started, "finished_at_seconds": time.time(),
            "producer_pid": self.owner_pid, "call_budget": budget})
        if result == -1:
            self.failed = True
            raise OSError(error, os.strerror(error))
        require(error == 0, "a successful native process call retained an unexplained errno")
        return result

    def descriptor_identity(self, descriptor, pid=None):
        """Inspect only this provider's own descriptor, without reopening a target process by PID."""
        require(type(descriptor) is int and descriptor >= 0, "one typed retained pidfd is required")
        inheritable = os.get_inheritable(descriptor)
        require(os.readlink('/proc/self/fd/' + str(descriptor)) == "anon_inode:[pidfd]"
            and inheritable is False, "the actual native handle is not a close-on-exec pidfd")
        info = os.fstat(descriptor)
        with Path('/proc/self/fdinfo/' + str(descriptor)).open() as stream:
            raw = stream.read(16385)
        after = os.fstat(descriptor)
        require(len(raw.encode()) <= 16384 and (info.st_dev, info.st_ino) == (after.st_dev, after.st_ino)
            and os.readlink('/proc/self/fd/' + str(descriptor)) == "anon_inode:[pidfd]"
            and os.get_inheritable(descriptor) is False,
            "the retained pidfd identity changed or exceeded its metadata bound")
        parsed = parse_pidfd_fdinfo(raw, info.st_ino, pid)
        return {"device": info.st_dev, "inode": info.st_ino, "pid": parsed['pid'],
            "flags": parsed['flags'], "inheritable": inheritable,
            "raw_fdinfo": raw, "fdinfo_sha256": hashlib.sha256(raw.encode()).hexdigest()}

    def retained(self, descriptor):
        """Reject a closed, substituted or foreign descriptor before any native signal or close."""
        require(type(descriptor) is int and descriptor in self.handles, "the descriptor was not opened by this provider")
        expected = self.handles[descriptor]
        actual = self.descriptor_identity(descriptor, expected["pid"])
        require(all(actual[key] == expected[key] for key in ("device", "inode")), "the retained native pidfd descriptor changed")
        return actual

    def record_descriptor(self, operation, descriptor, call_sequence, identity):
        """Retain one bounded descriptor observation joined to its actual open-call generation."""
        require(len(self.descriptor_events) < MAX_DESCRIPTOR_EVENTS and descriptor in self.handles,
            "native descriptor identity evidence exceeded its explicit bound")
        handle = self.handles[descriptor]
        self.descriptor_events.append({'event_sequence': len(self.descriptor_events), 'operation': operation,
            'descriptor': descriptor, 'call_sequence': call_sequence, 'open_generation': handle['generation'],
            'opened_pid': handle['pid'], 'identity': identity, 'observed_at_seconds': time.time()})

    def _open(self, pid, phase):
        """Open an actual kernel pidfd; callers retain their existing positive ownership checks."""
        require(type(pid) is int and 0 < pid <= 2147483647, "one positive typed Linux process identifier is required")
        descriptor = self.invoke("pidfd_open", {"syscall": NUMBERS["pidfd_open"], "pid": pid, "flags": 0},
            lambda: self.syscall(ctypes.c_long(NUMBERS["pidfd_open"]), ctypes.c_long(pid), ctypes.c_ulong(0)), phase=phase)
        require(descriptor >= 0 and descriptor not in self.handles, "the kernel returned an invalid or duplicated pidfd")
        self.handles[descriptor] = {"pid": pid, "generation": len(self.calls) - 1,
            "cleanup_remaining": [int(signal.SIGTERM), int(signal.SIGKILL), "close"]}
        try:
            identity = self.descriptor_identity(descriptor, pid)
            require(identity["pid"] == pid, "the opened process exited before its native handle was verified")
            self.handles[descriptor].update({key: identity[key] for key in ("device", "inode")})
            self.record_descriptor('pidfd_open', descriptor, len(self.calls) - 1, identity)
        except BaseException:
            self.invoke("close", {"fd": descriptor}, lambda: self.close(descriptor), phase="failed_open_cleanup",
                cleanup_descriptor=descriptor)
            self.handles.pop(descriptor, None)
            raise
        return descriptor

    def _signal(self, descriptor, number, phase):
        """Send only signal zero or cleanup TERM/KILL through the actual retained pidfd."""
        require(type(number) in (int, signal.Signals) and number in (0, signal.SIGTERM, signal.SIGKILL),
            "unsupported native process-control signal")
        number = int(number)
        identity = self.retained(descriptor)
        result = self.invoke("pidfd_send_signal", {"syscall": NUMBERS["pidfd_send_signal"], "fd": descriptor,
            "signal": number, "siginfo": None, "flags": 0}, lambda: self.syscall(
                ctypes.c_long(NUMBERS["pidfd_send_signal"]), ctypes.c_long(descriptor), ctypes.c_long(number),
                ctypes.c_void_p(None), ctypes.c_ulong(0)), phase=phase,
                cleanup_descriptor=descriptor if number in (int(signal.SIGTERM), int(signal.SIGKILL)) else None,
                descriptor_identity=identity)
        require(result == 0, "the kernel signal call returned an unexpected nonzero result")

    def _close(self, descriptor, phase):
        """Close only a verified descriptor with its actual libc return recorded."""
        identity = self.retained(descriptor)
        result = self.invoke("close", {"fd": descriptor}, lambda: self.close(descriptor), phase=phase,
            cleanup_descriptor=descriptor, descriptor_identity=identity)
        require(result == 0, "the native close returned an unexpected nonzero result")
        del self.handles[descriptor]

    def open_process_handle(self, pid):
        """Keep the existing descriptor API under an explicitly supplied provider object."""
        self.check_owner()
        return self._open(pid, "owned_process")

    def signal_handle(self, descriptor, number):
        """Never fall back to os.kill or another numeric-PID operation."""
        self.check_owner()
        self._signal(descriptor, number, "owned_process")

    def handle_exited(self, descriptor, milliseconds=0):
        """Observe actual kernel descriptor readiness without inferring an exit returncode."""
        self.check_owner()
        require(type(milliseconds) is int and milliseconds >= 0, "typed nonnegative poll time is required")
        self.retained(descriptor)
        waiter = select.poll(); waiter.register(descriptor, select.POLLIN)
        rows = waiter.poll(milliseconds)
        require(all(key == descriptor and events & ~(select.POLLIN | select.POLLHUP) == 0 for key, events in rows),
            "the retained process descriptor reported an unknown poll error")
        return bool(rows)

    def close_process_handle(self, descriptor):
        """Close a retained handle without selecting a process by a later PID lookup."""
        self.check_owner()
        self._close(descriptor, "owned_process")

    def evidence(self):
        """Expose native process-control evidence separately from all GPU or numerical qualification."""
        return {"kind": KIND, "contract": self.contract, "bootstrap": self.bootstrap,
            "preflight_passed": self.preflight_passed, "preflight_call_count": self.preflight_call_count,
            "calls": self.calls, "open_handle_count": len(self.handles), "native_call_failure": self.failed,
            "descriptor_events": self.descriptor_events,
            "ordinary_call_count": self.normal_call_count, "ordinary_call_limit": MAX_CALLS,
            "reserved_cleanup_call_count": self.cleanup_call_count,
            "maximum_cleanup_records_per_open": 3,
            "numeric_pid_signal_fallback": False, "python_exports_changed": False,
            "gpu_or_cuda_qualification": False, "scientific_credit": 0}


class NativeProcessObserver:
    """Pass an explicit provider to existing process identity checks without changing module globals."""
    def __init__(self, metadata_observer, provider):
        """Require the real native preflight; this adapter grants no source, plan or target admission."""
        require(isinstance(provider, NativePidfdProvider) and provider.preflight_passed,
            "the explicit native provider must pass its actual self-handle preflight")
        self.metadata_observer, self.provider = metadata_observer, provider

    def __getattr__(self, name):
        """Keep process-control calls on the provider and all identity readers on their unchanged source."""
        if name in ("open_process_handle", "handle_exited", "signal_handle", "close_process_handle"):
            return getattr(self.provider, name)
        require(name != "require_process_control_apis", "native backend evidence cannot fabricate missing Python exports")
        return getattr(self.metadata_observer, name)

    def OwnedProcesses(self, root_pid, root_birth, parent, reader=None):
        """Retain the same positive ancestry and UID/cgroup checks with one explicit provider argument."""
        return self.metadata_observer.OwnedProcesses(root_pid, root_birth, parent, reader,
            process_control=self.provider)

    def parent_handle(self, intent):
        """Pin the known parent through this process's provider without changing default module globals."""
        parent = intent['parent']
        descriptor = self.provider.open_process_handle(parent['pid'])
        try:
            actual = self.metadata_observer.ProcReader().snapshot(parent['pid'], parent['uids'][0], parent['cgroup'])
            require(self.metadata_observer.stable(actual) == self.metadata_observer.stable(parent)
                and not self.provider.handle_exited(descriptor), 'the actual native supervisor disappeared or changed')
        except BaseException:
            self.provider.close_process_handle(descriptor)
            raise
        return descriptor
