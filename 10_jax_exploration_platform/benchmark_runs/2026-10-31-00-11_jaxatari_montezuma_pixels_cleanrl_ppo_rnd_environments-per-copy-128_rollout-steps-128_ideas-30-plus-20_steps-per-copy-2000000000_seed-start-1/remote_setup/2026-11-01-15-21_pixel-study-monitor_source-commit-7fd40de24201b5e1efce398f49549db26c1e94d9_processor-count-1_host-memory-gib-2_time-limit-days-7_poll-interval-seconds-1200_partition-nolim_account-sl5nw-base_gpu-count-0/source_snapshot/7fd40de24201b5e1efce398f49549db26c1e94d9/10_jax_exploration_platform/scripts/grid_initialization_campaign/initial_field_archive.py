"""Archive exact initial field arrays for paired confirmation across GPU types."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile

from common import PLATFORM, now


ARCHIVE_SCHEMA_VERSION = 1
FIELD_ALGORITHM_VERSION = 'coordinate_and_path_initial_bonus_fields_v1'


def uses_canonical_archive(specification):
    """Keep development untouched and require archived arrays for both confirmation waves."""
    return (specification.get('wave') in {'confirmation', 'long-confirmation'} or
            specification.get('name', '').startswith(('confirmation_', 'long-confirmation_')))


def array_digest(values):
    """Hash an array's exact values, shape and dtype without recording GPU hardware."""
    import numpy as np

    # Before: an array view; after: a shape/dtype header followed by contiguous little-endian bytes.
    array = np.asarray(values)
    header = json.dumps({'shape': list(array.shape), 'dtype': array.dtype.name}, sort_keys=True)
    encoded = np.ascontiguousarray(array.astype(array.dtype.newbyteorder('<'), copy=False))
    return hashlib.sha256(header.encode() + b'\n' + encoded.tobytes()).hexdigest()


def generation_identity(geometry, kind, severity_exponent, seed, source_version=None):
    """Identify the base generation inputs before any field transform or policy is chosen."""
    import numpy as np

    # Coordinates belong to the field geometry, which may differ from the training geometry.
    coordinates = np.asarray(geometry.coordinates)
    if (coordinates.dtype != np.float32 or coordinates.shape != (geometry.state_count, 2)
            or not np.isfinite(coordinates).all()):
        raise ValueError('Canonical field geometry requires finite float32 [states,2] coordinates.')
    source_path = PLATFORM/'src/exploration_platform/grid_initialization/fields.py'
    if source_version is None:
        source_version = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if not isinstance(source_version, str) or not source_version:
        raise ValueError('Canonical field generation requires an explicit source version.')
    return {'archive_schema_version': ARCHIVE_SCHEMA_VERSION,
            'field_algorithm_version': FIELD_ALGORITHM_VERSION,
            'field_source_version': source_version, 'field_kind': kind,
            'severity_exponent': int(severity_exponent), 'field_seed': int(seed),
            'field_geometry': {'kind': geometry.kind, 'width': int(geometry.width),
                               'turns': int(geometry.turns), 'state_count': geometry.state_count},
            'coordinates_sha256': array_digest(coordinates),
            'generation_defaults': {'width_exponent': 2, 'period_exponent': 3,
                                    'step_count_exponent': 2, 'constant_coordinate': False}}


def identity_digest(identity):
    """Give the complete scientific generation record one content identifier."""
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def archive_path(run_dir, identity):
    """Name fields by their generation rule and seed within a confirmation-only archive."""
    return (Path(run_dir)/'data/confirmation_initial_fields'/identity['field_kind'] /
            f"severity-exponent-{identity['severity_exponent']}" /
            f"field-seed-{identity['field_seed']}_identity-sha256-{identity_digest(identity)}.npz")


def validate_arrays(raw, normalized, coordinates, identity, metadata):
    """Reject wrong shapes, nonfinite values, inconsistent normalization and mismatched metadata."""
    import numpy as np

    count = identity['field_geometry']['state_count']
    if (raw.dtype != np.float32 or normalized.dtype != np.float32 or
            raw.shape != (count,) or normalized.shape != (count,) or
            coordinates.dtype != np.float32 or coordinates.shape != (count, 2)):
        raise ValueError('Malformed canonical field archive: array dtype or shape does not match generation.')
    if not all(np.isfinite(values).all() for values in (raw, normalized, coordinates)):
        raise ValueError('Malformed canonical field archive: nonfinite array value.')
    if array_digest(coordinates) != identity['coordinates_sha256']:
        raise ValueError('Malformed canonical field archive: field coordinates disagree with generation.')
    if not isinstance(metadata, dict) or metadata.get('kind') != identity['field_kind'] or metadata.get('seed') != identity['field_seed']:
        raise ValueError('Malformed canonical field archive: field metadata disagrees with generation.')
    # Reproduce only the existing mean normalization; never repair or clip stored values.
    if identity['field_kind'] == 'none':
        if np.any(raw != 0) or np.any(normalized != 0):
            raise ValueError('Malformed canonical no-bonus field: arrays must be zero.')
    else:
        if np.any(raw <= 0) or np.any(normalized <= 0):
            raise ValueError('Malformed canonical field archive: initial bonuses must be positive.')
        expected = (raw.astype(np.float64)/raw.astype(np.float64).mean()).astype(np.float32)
        if not np.array_equal(normalized, expected):
            raise ValueError('Malformed canonical field archive: raw and normalized values disagree.')


def load_archive(path, identity, expected_coordinates):
    """Read a completed archive and verify it without regenerating missing or corrupt contents."""
    import numpy as np
    from exploration_platform.grid_initialization.fields import BonusField

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'raw', 'normalized', 'coordinates', 'manifest_json'}:
            raise ValueError(f'Malformed canonical field archive: wrong contents in {path}.')
        raw, normalized, coordinates = (archive[name].copy() for name in ('raw', 'normalized', 'coordinates'))
        manifest_array = archive['manifest_json']
        if manifest_array.shape != () or manifest_array.dtype.kind != 'U':
            raise ValueError(f'Malformed canonical field archive: invalid manifest in {path}.')
        manifest = json.loads(str(manifest_array.item()))
    if (manifest.get('generation') != identity or
            manifest.get('generation_identity_sha256') != identity_digest(identity)):
        raise ValueError(f'Malformed canonical field archive: generation identity mismatch in {path}.')
    metadata = manifest.get('field_metadata')
    validate_arrays(raw, normalized, coordinates, identity, metadata)
    if not np.array_equal(coordinates, np.asarray(expected_coordinates)):
        raise ValueError(f'Malformed canonical field archive: unexpected coordinates in {path}.')
    digests = {name: array_digest(values) for name, values in
               (('raw', raw), ('normalized', normalized), ('coordinates', coordinates))}
    if manifest.get('array_sha256') != digests:
        raise ValueError(f'Malformed canonical field archive: array checksum mismatch in {path}.')
    # Preserve the original field metadata and add stable references to the exact canonical arrays.
    metadata = dict(metadata, canonical_archive={'path': str(path),
        'generation_identity_sha256': identity_digest(identity), 'array_sha256': digests,
        'created_at': manifest['created_at'], 'creator': manifest['creator']})
    return BonusField(raw, normalized, metadata)


def load_or_create_initial_field(run_dir, geometry, kind, severity_exponent, seed, *,
                                 generate=None, source_version=None):
    """Create a canonical field once under a file lock, then reuse its exact archived arrays."""
    import numpy as np
    from exploration_platform.grid_initialization.fields import make_initial_bonus_field

    identity = generation_identity(geometry, kind, severity_exponent, seed, source_version)
    path = archive_path(run_dir, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return load_archive(path, identity, geometry.coordinates)
        # Only the first process generates; every later GPU receives these exact bytes.
        field = (generate() if generate is not None else make_initial_bonus_field(
            kind, geometry, seed=int(seed), severity_exponent=int(severity_exponent)))
        raw, normalized = np.asarray(field.raw), np.asarray(field.normalized)
        coordinates = np.asarray(geometry.coordinates)
        validate_arrays(raw, normalized, coordinates, identity, field.metadata)
        manifest = {'generation': identity, 'generation_identity_sha256': identity_digest(identity),
            'field_metadata': field.metadata,
            'array_sha256': {name: array_digest(values) for name, values in
                (('raw', raw), ('normalized', normalized), ('coordinates', coordinates))},
            'created_at': now(), 'creator': {'hostname': socket.gethostname(),
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'gpu_class': os.environ.get('GRID_GPU_CLASS')}}
        descriptor, temporary = tempfile.mkstemp(prefix=path.name+'.temporary-', dir=path.parent)
        try:
            with os.fdopen(descriptor, 'wb') as handle:
                np.savez_compressed(handle, raw=raw, normalized=normalized, coordinates=coordinates,
                    manifest_json=np.asarray(json.dumps(manifest, sort_keys=True, allow_nan=False)))
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o664)
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return load_archive(path, identity, geometry.coordinates)
