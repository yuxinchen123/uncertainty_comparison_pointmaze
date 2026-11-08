"""Shared durable records and atomic claims for the initial-bonus study."""

from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from zoneinfo import ZoneInfo

PROJECT = Path('/p/rlprojects/RND')
PLATFORM = Path(__file__).resolve().parents[2]
PYTHON = PROJECT / '.venvs/platform_jax/bin/python'
ACCOUNTS = Path('/p/rlprojects/.claude/skills/submit-as-account/scripts')
os.umask(0o002)


def fixed_study(run_dir):
    """True for the fixed CleanRL PPO-RND settings study, whose source record names it."""
    path = Path(run_dir)/'source.json'
    return path.exists() and read_json(path).get('campaign') == 'fixed-settings'


def ledger_prefix(run_dir):
    """Each study keeps its own per-account job-id ledgers."""
    return 'fixed_settings' if fixed_study(run_dir) else 'initial_bonus'


def now():
    """Return an offset-bearing timestamp rather than an ambiguous local time."""
    return datetime.now().astimezone().isoformat()


def pacific(value=None):
    """Display a timestamp in the user's requested Pacific time zone."""
    instant = datetime.fromisoformat(value) if value else datetime.now().astimezone()
    return instant.astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M PT')


def read_json(path):
    """Read one required structured record, failing on malformed data."""
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Replace one record atomically on the shared filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    with os.fdopen(descriptor, 'w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o664)
    os.replace(temporary, path)


def append_json(path, value):
    """Append and flush one record under a writer lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(value, allow_nan=False) + '\n')
        handle.flush()


@contextmanager
def queue_lock(run_dir):
    """Serialize queue transitions across both submission accounts."""
    with (Path(run_dir) / 'queue/queue.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def enqueue(run_dir, task):
    """Add one fully specified task once, using its readable name as identity."""
    run_dir = Path(run_dir)
    with queue_lock(run_dir):
        existing = [run_dir / 'queue' / state / (task['name'] + '.json')
                    for state in ('pending', 'running', 'done', 'failed')]
        if any(path.exists() for path in existing):
            return False
        task = dict(task, created_at=now())
        write_json(existing[0], task)
    return True


def claim(run_dir, stage, device_memory_mb):
    """Claim the first eligible task only after a GPU process starts."""
    run_dir = Path(run_dir)
    with queue_lock(run_dir):
        for path in sorted((run_dir / 'queue/pending').glob('*.json')):
            task = read_json(path)
            if task['stage'] != stage or task.get('minimum_device_memory_mb', 0) > device_memory_mb:
                continue
            task.update(job_id=os.environ.get('SLURM_JOB_ID', 'local'),
                        account=os.environ.get('USER'), started_at=now(),
                        hostname=os.uname().nodename)
            target = run_dir / 'queue/running' / path.name
            write_json(target, task)
            path.unlink()
            return task
    return None


def finish(run_dir, task, result, successful=True):
    """Publish completion only after its result exists, then release the claim."""
    run_dir = Path(run_dir)
    with queue_lock(run_dir):
        destination = 'done' if successful else 'failed'
        write_json(run_dir / 'queue' / destination / (task['name'] + '.json'),
                   dict(task, finished_at=now(), result=result))
        (run_dir / 'queue/running' / (task['name'] + '.json')).unlink()


def queue_counts(run_dir):
    """Count actual units rather than inferring progress from Slurm job counts."""
    return {state: len(list((Path(run_dir) / 'queue' / state).glob('*.json')))
            for state in ('pending', 'running', 'done', 'failed')}


def seed_for(*parts):
    """Give each named quantity a stable seed independent of execution order."""
    return int.from_bytes(hashlib.sha256('::'.join(map(str, parts)).encode()).digest()[:4], 'big')


def own_live_jobs(run_dir):
    """Read liveness by this study's exact job identifiers across both accounts."""
    identifiers = set()
    for path in (Path(run_dir) / 'slurm').glob('submitted_jobids_*_*.txt'):
        identifiers.update(value for value in path.read_text().split() if value.isdecimal())
    if not identifiers:
        return {}
    result = subprocess.run(['squeue', '-h', '-j', ','.join(sorted(identifiers)),
                             '-o', '%i|%u|%P|%T|%j|%R|%b'], capture_output=True, text=True)
    if result.returncode and 'Invalid job id' not in result.stderr:
        raise RuntimeError(result.stderr)
    return {parts[0]: parts for row in result.stdout.splitlines()
            if len(parts := row.split('|')) == 7}
