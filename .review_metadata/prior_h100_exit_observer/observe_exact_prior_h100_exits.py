"""Observe only the two recorded prior H100 process identities without sending signals."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

HOST = 'serval02'
OWNER_UID = 2618919
PROC_ROOT = Path('/proc')
IDENTITIES = {'controller': (386202, '84810007'), 'numerical_child': (386669, '84815419')}
MAX_BYTES = 65536


def require(condition, message):
    """Refuse an unknown observation without inspecting commands or acting on a process."""
    if not condition:raise ValueError(message)


def utc_now():
    """Return the actual current UTC time with an explicit timezone."""
    return datetime.now(timezone.utc).isoformat()


def bounded(directory, name):
    """Read one bounded ordinary proc file through the already opened process directory."""
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), 'the known process metadata is not an ordinary proc file')
        value = os.read(descriptor, MAX_BYTES + 1)
        require(len(value) <= MAX_BYTES, 'the known process metadata exceeded its explicit bound')
        return value.decode('ascii', errors='strict')
    finally:os.close(descriptor)


def user_identifiers(raw):
    """Require exactly one complete four-user-identifier kernel field before reading process stat."""
    values = [line.split()[1:] for line in raw.splitlines() if line.startswith('Uid:')]
    require(len(values) == 1 and len(values[0]) == 4
        and all(re.fullmatch(r'[0-9]+', item) for item in values[0]), 'the known process user identifiers are incomplete')
    return [int(item) for item in values[0]]


def process_identity(raw):
    """Extract only PID and birth, without treating changing CPU counters as identity."""
    first, separator, rest = raw.rpartition(') ')
    fields = rest.split();pid = first.split(' (', 1)[0]
    require(separator and re.fullmatch(r'[1-9][0-9]*', pid) and len(fields) >= 20
        and re.fullmatch(r'[1-9][0-9]*', fields[19]), 'the known process PID or birth fields are invalid')
    return int(pid), fields[19]


def observe_process(role, *, proc_root=PROC_ROOT):
    """Check one literal prior identity; present or unknown identities never become absent rows."""
    require(role in IDENTITIES, 'only the two exact recorded process roles may be observed')
    pid, birth = IDENTITIES[role]
    row = {'role': role, 'expected_pid': pid, 'expected_birth': birth, 'observed_at_utc': utc_now()}
    if not proc_root.is_dir():
        return {**row, 'observation': 'unverified', 'original_process_exited': None, 'error_type': 'UnavailableProcRoot'}
    path = proc_root / str(pid)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        return {**row, 'observation': 'absent', 'original_process_exited': True}
    except OSError as error:
        return {**row, 'observation': 'unverified', 'original_process_exited': None, 'error_type': type(error).__name__}
    try:
        opened = os.fstat(descriptor)
        # Ownership precedes reading any status/stat bytes; no command, environment or device file is opened.
        require(opened.st_uid == OWNER_UID, 'the present process directory is not owned by the campaign owner')
        before_status = bounded(descriptor, 'status')
        require(user_identifiers(before_status) == [OWNER_UID] * 4, 'the present process does not retain all four owner identifiers')
        before = bounded(descriptor, 'stat');identity = process_identity(before)
        after = bounded(descriptor, 'stat');after_status = bounded(descriptor, 'status')
        current = os.stat(path, follow_symlinks=False)
        require(identity == process_identity(after) and identity[0] == pid
            and user_identifiers(after_status) == [OWNER_UID] * 4
            and (opened.st_dev, opened.st_ino, opened.st_uid) == (current.st_dev, current.st_ino, current.st_uid),
            'the present process changed identity or disappeared during observation')
        return {**row, 'observation': 'present_same_birth' if identity[1] == birth else 'different_owned_birth',
            'observed_birth': identity[1], 'original_process_exited': identity[1] != birth,
            'all_four_user_identifiers_match': True,
            'stat_before_sha256': hashlib.sha256(before.encode()).hexdigest(),
            'stat_after_sha256': hashlib.sha256(after.encode()).hexdigest()}
    except (OSError, ValueError, UnicodeError) as error:
        return {**row, 'observation': 'unverified', 'original_process_exited': None, 'error_type': type(error).__name__}
    finally:os.close(descriptor)


def observation_record(rows):
    """Emit the required prior-exit schema with actual observer identity and actual current time."""
    require(len(rows) == 2 and {row['role'] for row in rows} == set(IDENTITIES), 'both literal recorded roles are required')
    return {'kind': 'owned_prior_direct_process_exit_observation', 'hostname': os.uname().nodename.split('.')[0],
        'observer_uid': os.getuid(), 'observer_pid': os.getpid(), 'at_utc': utc_now(),
        'all_original_processes_exited': all(row['original_process_exited'] is True for row in rows),
        'gpu_or_standing_lock_operations': 0, 'observations': rows,
        'all_expected_processes_absent': all(row['observation'] == 'absent' for row in rows),
        'signals_sent': 0, 'global_process_scans': 0, 'read_only': True}


def verify_own_proc_namespace():
    """Refuse an unavailable proc namespace by validating only this observer's actual own process first."""
    root = os.open(PROC_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    directory = None
    try:
        require(os.fstat(root).st_uid == 0, 'the proc root must retain its system owner')
        directory = os.open(str(os.getpid()), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root)
        require(os.fstat(directory).st_uid == OWNER_UID
            and user_identifiers(bounded(directory, 'status')) == [OWNER_UID] * 4
            and process_identity(bounded(directory, 'stat'))[0] == os.getpid(),
            'the proc namespace cannot verify this actual observer process')
    finally:
        if directory is not None:os.close(directory)
        os.close(root)


def main():
    """Print one actual observation only on the approved host and owner, with no PID override."""
    require(len(sys.argv) == 1, 'this exact observer accepts no runtime arguments or process overrides')
    require(os.uname().nodename.split('.')[0] == HOST and os.getuid() == os.geteuid() == OWNER_UID,
        'the exact owner must run this read-only observer on serval02')
    verify_own_proc_namespace()
    record = observation_record([observe_process(role) for role in IDENTITIES])
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    return 0 if record['all_expected_processes_absent'] else 1


if __name__ == '__main__':raise SystemExit(main())
