"""Compare two saved training states array by array: identical, or floats within a small relative difference.

A worker job on a remote server runs this once, before it takes its first unit. The reference file was written on UVA
by `check_online_path_unchanged.py --dump` on the processor backend; the candidate is the same dump made on the
server. This checks that the copied code and environment compute the same thing. It measures no speed.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile

import numpy as np

# Different processor families round differently (measured 2026-11-09: AMD EPYC 7543 against UVA's Ice Lake gave
# float differences up to 1.0e-2 after the four training iterations, with every integer array, such as visit counts
# and positions, equal; the orthogonal weight draw itself already differs by 2e-4 on its smallest entries). So
# integers must match exactly and floats may differ by this much; a wrong code path or environment gives differences
# of order one or different visit counts.
RELATIVE_TOLERANCE = 5e-2
# A value smaller than this in size is compared on this scale, so 0 against 1e-12 passes (difference below 1e-8).
SCALE_FLOOR = 1e-4


def compare_arrays(reference, candidate):
    """Return (passed, reason, largest relative difference) for one pair of arrays."""
    # Arrays of different shape or number kind cannot be the same state.
    if reference.shape != candidate.shape or reference.dtype.kind != candidate.dtype.kind:
        return False, 'shape or number kind differs', None
    # Integers, booleans and random-number keys must match exactly.
    if reference.dtype.kind not in 'fc':
        equal = bool(np.array_equal(reference, candidate))
        return equal, 'identical' if equal else 'integer values differ', None
    if np.array_equal(reference, candidate, equal_nan=True):
        return True, 'identical', 0.0
    # Infinite and not-a-number entries must sit at the same places with the same values.
    finite = np.isfinite(reference) & np.isfinite(candidate)
    if not np.array_equal(reference[~finite], candidate[~finite], equal_nan=True):
        return False, 'infinite or not-a-number entries differ', None
    # Before: reference [1.0, 0.0], candidate [1.00001, 1e-12]; after: relative differences [1e-5, 1e-8].
    first, second = reference[finite].astype(np.float64), candidate[finite].astype(np.float64)
    scale = np.maximum(np.maximum(np.abs(first), np.abs(second)), SCALE_FLOOR)
    largest = float((np.abs(first - second) / scale).max()) if first.size else 0.0
    passed = largest <= RELATIVE_TOLERANCE
    return passed, 'within the relative tolerance' if passed else 'floats differ beyond the tolerance', largest


def compare_files(reference_path, candidate_path):
    """Compare every array of two .npz files and return the verdict record."""
    reference, candidate = np.load(reference_path), np.load(candidate_path)
    failed, identical, largest, largest_name = [], 0, 0.0, None
    # A missing or extra array is a different state structure.
    for name in sorted(set(reference.files) ^ set(candidate.files)):
        failed.append({'array': name, 'reason': 'present in only one file', 'relative_difference': None})
    for name in sorted(set(reference.files) & set(candidate.files)):
        passed, reason, difference = compare_arrays(reference[name], candidate[name])
        identical += reason == 'identical'
        if difference is not None and difference > largest:
            largest, largest_name = difference, name
        if not passed:
            failed.append({'array': name, 'reason': reason, 'relative_difference': difference})
    # Before: every failed array listed; after: the first ten and their count, so the record stays short.
    failed_count, failed = len(failed), failed[:10]
    return {'passed': not failed, 'failed_array_count': failed_count,
            'arrays_compared': len(set(reference.files) | set(candidate.files)),
            'identical_arrays': identical, 'largest_relative_difference': largest,
            'largest_difference_array': largest_name, 'failed_arrays': failed,
            'relative_tolerance': RELATIVE_TOLERANCE, 'scale_floor': SCALE_FLOOR}


def processor_model():
    """Read the processor's name, which explains a rounding difference between servers."""
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):
            return line.split(':', 1)[1].strip()
    return 'unknown'


def main():
    """Write the verdict where every later worker job of the server reads it, and exit 1 on a failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server', required=True)
    args = parser.parse_args()
    record = compare_files(args.reference, args.candidate)
    record.update(server=args.server, at=datetime.now().astimezone().isoformat(), hostname=os.uname().nodename,
                  job_id=os.environ.get('SLURM_JOB_ID'), processor=processor_model(),
                  reference=str(args.reference), candidate=str(args.candidate))
    # Several jobs may finish the check at the same moment: each replaces the file in one step.
    descriptor, temporary = tempfile.mkstemp(prefix=args.output.name + '.', dir=args.output.parent)
    with os.fdopen(descriptor, 'w') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write('\n')
    os.chmod(temporary, 0o664)
    os.replace(temporary, args.output)
    print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(0 if record['passed'] else 1)


if __name__ == '__main__':
    main()
