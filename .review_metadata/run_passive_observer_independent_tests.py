"""Review frozen passive metadata behavior using only synthetic ordinary files."""

import builtins
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
EXPECTED = {
    'passive_installed_slurm_observation.py': '75b49228026a3623b85e77011cff272b341a2ba83ea391221d2d6405f1621058',
    'tests/test_passive_installed_slurm_observation.py': '96dad56427f45a5515af4e344d5a1fd7df182af9c2a5b0921c8152e19c335198',
    'uva_active_configuration_observation.py': '8cc5a3544e426e3bd3e78707f6c77dc249a1d19165a593e157d2097daa07a648',
}
NUMERICAL = {'jax', 'jaxlib', 'numpy', 'torch', 'pynvml', 'jaxtari', 'exploration_platform',
    'cupy', 'numba', 'matplotlib', 'scipy', 'flax', 'optax'}
BLOCKED = []
ORIGINAL_IMPORT = builtins.__import__


def guarded_import(name, *args, **kwargs):
    """Refuse numerical code before any imported body can run."""
    if name.split('.')[0] in NUMERICAL:
        BLOCKED.append({'kind': 'numerical_import', 'name': name})
        raise AssertionError('numerical import is forbidden in this metadata review')
    return ORIGINAL_IMPORT(name, *args, **kwargs)


def audit(event, args):
    """Forbid actual commands, network, library loading and device access."""
    if event in {'subprocess.Popen', 'os.system', 'os.posix_spawn', 'socket.connect',
            'socket.getaddrinfo', 'ctypes.dlopen'}:
        BLOCKED.append({'kind': 'operation', 'event': event})
        raise AssertionError('live operation forbidden: ' + event)
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if path.startswith(('/dev/nvidia', '/dev/dri')) or re.match(r'^/proc/[0-9]+(?:/|$)', path):
            BLOCKED.append({'kind': 'path', 'path': path})
            raise AssertionError('device or numeric process path forbidden')


assert os.environ.get('SLURM_JOB_ID') == '6781231'
assert os.environ.get('SLURM_STEP_ID', '').isdigit()
assert os.environ.get('SLURM_CPUS_PER_TASK') == '1'
before = {name: hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest() for name in EXPECTED}
assert before == EXPECTED, before
process_stat = Path('/proc/self/stat').read_text()
process_birth = process_stat[process_stat.rfind(')') + 2:].split()[19]
process_cgroup = Path('/proc/self/cgroup').read_text()
builtins.__import__ = guarded_import
sys.addaudithook(audit)
sys.path.insert(0, str(SCRIPTS / 'tests'))
sys.path.insert(0, str(SCRIPTS))
import passive_installed_slurm_observation as passive

stream = io.StringIO()
start = time.monotonic()
suite = unittest.defaultTestLoader.loadTestsFromName('test_passive_installed_slurm_observation')
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)

# These are reproductions of frozen-source findings, not acceptance tests for corrected code.
reproductions = []
secret = 'synthetic_private_token_812'
rows = passive.setting_rows(('AuthInfo="prefix File=' + secret + '"\n').encode(), passive.FILE_SETTINGS)
reproductions.append({'finding': 'quoted_unallowlisted_value_is_copied_as_an_allowed_field',
    'reproduced': any(secret in row['value'] for row in rows), 'filtered_rows': rows})

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    first = root / 'slurm.conf'
    key = root / 'jwt.key'
    first.write_text('Include ' + str(key) + '\n')
    key.write_text('synthetic_private_key_bytes')
    reader = passive.PassiveReader()
    real_read = reader.read
    opened = []

    def record_read(path, *args, **kwargs):
        """Record only synthetic read requests without touching installed credentials."""
        opened.append(str(path))
        return real_read(path, *args, **kwargs)

    with patch.object(reader, 'read', side_effect=record_read):
        tree = passive.configuration_tree(reader, first, (str(root),))
    reproductions.append({'finding': 'private_key_include_is_read',
        'reproduced': str(key) in opened, 'requested_read_names': [Path(path).name for path in opened],
        'tree_complete': tree['complete']})

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    (root / 'first').mkdir()
    (root / 'second').mkdir()
    (root / 'first/old.conf').write_bytes(b'')
    (root / 'second/new.conf').write_bytes(b'')
    alias = root / 'alias'
    alias.symlink_to('first', target_is_directory=True)
    original = passive.resolve_links

    def retarget(path, roots):
        """Change the original alias after resolving its ordinary old target."""
        value = original(path, roots)
        alias.unlink()
        alias.symlink_to('second', target_is_directory=True)
        return value

    with patch.object(passive, 'resolve_links', side_effect=retarget):
        listing = passive.directory_entries(alias, (str(root),), r'.*\.conf', 2)
    reproductions.append({'finding': 'directory_alias_retarget_is_not_recorded_as_changed',
        'reproduced': listing.get('status') == 'listed' and listing['entries'] == ['old.conf']
            and os.readlink(alias) == 'second' and 'symlink_chain_after' not in listing,
        'status': listing.get('status'), 'entries': listing.get('entries'),
        'initial_alias_target': listing.get('symlink_chain', [{}])[0].get('target'),
        'actual_alias_target_after': os.readlink(alias)})

after = {name: hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest() for name in EXPECTED}
record = {'schema_version': 1, 'kind': 'passive_installed_slurm_independent_frozen_review',
    'at_utc': datetime.now(timezone.utc).isoformat(), 'worktree': str(ROOT),
    'job_id': os.environ['SLURM_JOB_ID'], 'step_id': os.environ['SLURM_STEP_ID'],
    'hostname': os.uname().nodename, 'pid': os.getpid(), 'process_start_time': process_birth,
    'cgroup': process_cgroup, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
    'hashes_before': before, 'hashes_after': after, 'source_unchanged': before == after == EXPECTED,
    'test_count': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
    'test_stdout': stream.getvalue(), 'seconds': time.monotonic() - start,
    'blocked_attempts': BLOCKED, 'numerical_modules_loaded': sorted(name for name in sys.modules
        if name.split('.')[0] in NUMERICAL), 'reproductions': reproductions,
    'suite_passed': result.wasSuccessful() and result.testsRun == 29 and before == after == EXPECTED and not BLOCKED,
    'source_approved': False, 'actual_collection_performed': False,
    'additional_source_findings': [
        'controller_query buffers complete subprocess stdout/stderr before checking its byte limit',
        'gres.conf and cgroup.conf whitespace Include directives have no explicit unresolved record',
    ],
    'case_credit': 0, 'idea_credit': 0, 'final_training_credit': 0,
    'physical_gres_mapping_verified': False, 'kernel_device_enforcement_verified': False,
    'qualification_verified': False, 'admission_granted': False}
destination = ROOT / '.review_metadata' / ('passive_installed_slurm_independent_first_'
    + record['job_id'] + '_' + record['step_id'] + '.json')
with destination.open('x') as output:
    json.dump(record, output, indent=2, allow_nan=False)
    output.write('\n')
    output.flush()
    os.fsync(output.fileno())
print(json.dumps({'suite_passed': record['suite_passed'], 'test_count': result.testsRun,
    'reproduced_findings': [row['finding'] for row in reproductions if row['reproduced']],
    'receipt': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}, indent=2))
assert record['suite_passed'], stream.getvalue()
assert all(row['reproduced'] for row in reproductions)
