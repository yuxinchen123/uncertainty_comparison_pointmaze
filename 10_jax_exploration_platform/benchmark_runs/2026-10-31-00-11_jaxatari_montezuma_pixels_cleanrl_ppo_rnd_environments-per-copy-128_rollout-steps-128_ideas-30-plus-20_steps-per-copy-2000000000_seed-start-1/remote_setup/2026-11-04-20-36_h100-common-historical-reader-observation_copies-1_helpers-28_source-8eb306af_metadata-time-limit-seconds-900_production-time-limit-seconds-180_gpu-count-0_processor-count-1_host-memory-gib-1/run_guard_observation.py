"""Snapshot explicit immutable input references around one reviewed metadata-only observer."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time

PACKET = Path(__file__).resolve(strict=True).parent
OUTPUTS = {'input_snapshot_before.json', 'input_snapshot_after.json', 'input_snapshot_comparison.json',
    'wrapper_intent.json', 'wrapper_return.json'}


def require(condition, message):
    """Refuse changed inputs or an output path outside this new diagnostic packet."""
    if not condition:
        raise ValueError(message)


def digest(raw):
    """Bind exact source and metadata bytes without interpreting scientific results."""
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    """Write only a new fixed packet-local output and preserve every prior attempt."""
    require(name in OUTPUTS and Path(name).name == name, 'unlisted diagnostic output')
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with (PACKET / name).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {'path': str(PACKET / name), 'sha256': digest(raw)}


def snapshot(references):
    """Hash only the explicitly listed ordinary metadata/reference files with stable full identities."""
    result = {}
    for role, reference in references.items():
        path = Path(reference['path'])
        require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
            'consumed input path changed: ' + role)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            before = os.fstat(descriptor)
            require(stat.S_ISREG(before.st_mode) and before.st_uid == 2618919 and before.st_size <= 16 * 1024 * 1024,
                'consumed metadata input has unverified ownership or size: ' + role)
            hasher = hashlib.sha256()
            with os.fdopen(os.dup(descriptor), 'rb') as stream:
                for raw in iter(lambda: stream.read(1024 * 1024), b''):
                    hasher.update(raw)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        fields = ('st_dev', 'st_ino', 'st_uid', 'st_gid', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        identity = {name: getattr(before, name) for name in fields}
        require(identity == {name: getattr(after, name) for name in fields}
            and identity == {name: getattr(path.stat(), name) for name in fields}
            and hasher.hexdigest() == reference['sha256'], 'consumed input bytes or identity changed: ' + role)
        result[role] = {'reference': reference, 'identity': identity}
    return result


def main():
    """Run the exact observer once inside the caller's single external large-read lock."""
    require(not sys.argv[1:], 'the diagnostic wrapper accepts no operational overrides')
    require(os.environ.get('SLURM_JOB_ID') == '6783003' and os.environ.get('SLURM_STEP_ID', '').isdigit()
        and os.getuid() == os.geteuid() == 2618919, 'use only the exact owned numeric CPU step')
    manifest = json.loads((PACKET / 'manifest.yaml').read_bytes())
    references = json.loads((PACKET / 'consumed_input_references.json').read_bytes())['references_by_role']
    observer_path = PACKET / 'observe_guard_phases.py'
    require(digest(observer_path.read_bytes()) == manifest['observer_source']['sha256']
        and manifest['observer_source']['sha256'] == '4b3918181151625d589755c8fa7e1fb365397db14d80d5f68fa14fce11831a63'
        and digest(Path(__file__).read_bytes()) == manifest['wrapper_source']['sha256']
        and manifest['diagnostic_timeout_seconds'] == 900 and manifest['production_timeout_seconds_unchanged'] == 180
        and manifest['registered'] is False, 'the exact reviewed diagnostic source or role changed')
    # Before: declared genuine input references. After: the same list plus exact new local source/plan records.
    for name in ('manifest.yaml', 'consumed_input_references.json', 'unregistered_guard_input_plan.json',
            'diagnostic_unit_identity.json', 'source_publication_binding.json', 'observe_guard_phases.py', 'run_guard_observation.py'):
        path = PACKET / name
        references['new_packet_' + name] = {'path': str(path), 'sha256': digest(path.read_bytes())}
    specification = importlib.util.spec_from_file_location('reviewed_metadata_observer_only', observer_path)
    observer = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(observer)
    before = snapshot(references)
    before_ref = save('input_snapshot_before.json', before)
    parent = observer.process(os.getpid())
    save('wrapper_intent.json', {'reviewed_entry': str(observer_path) + ':parent', 'parent': parent,
        'job_id': os.environ['SLURM_JOB_ID'], 'step_id': os.environ['SLURM_STEP_ID'],
        'input_snapshot': before_ref, 'diagnostic_unit_id': PACKET.name,
        'automatic_retry_permitted': False, 'scientific_credit': 0, 'admission_granted': False})
    started = time.time()
    failure = None
    after_error = None
    after = None
    try:
        # The unchanged observer owns its actual child, finite 900-second wait and exact-child cleanup.
        observer.parent()
    except BaseException as error:
        failure = error
    try:
        after = snapshot(references)
        after_ref = save('input_snapshot_after.json', after)
    except BaseException as error:
        after_error = error
        after_ref = save('input_snapshot_after.json', {'snapshot_failed': True,
            'error': {'type': type(error).__name__, 'message': str(error)}})
    returned = save('wrapper_return.json', {'reviewed_entry': str(observer_path) + ':parent',
        'observer_parent_completed_normally': failure is None,
        'observer_failure': None if failure is None else {'type': type(failure).__name__, 'message': str(failure)},
        'after_snapshot_failure': None if after_error is None else {'type': type(after_error).__name__, 'message': str(after_error)},
        'parent_before': parent, 'parent_after': observer.process(os.getpid()),
        'started_at_seconds': started, 'finished_at_seconds': time.time()})
    save('input_snapshot_comparison.json', {'before': before_ref, 'after': after_ref, 'all_inputs_unchanged': before == after,
        'observer_return': returned, 'diagnostic_unit_id': PACKET.name, 'borrowed_identity_is_production_unit_proof': False,
        'raw_payloads_verified_by_this_snapshot': False, 'production_guard_qualified': False,
        'formal_timing_preparation_permitted': False, 'scientific_credit': 0, 'admission_granted': False})
    if failure is not None:
        raise failure
    if after_error is not None:
        raise after_error
    require(before == after, 'consumed inputs changed; preserve this diagnostic without retry')


if __name__ == '__main__':
    main()
