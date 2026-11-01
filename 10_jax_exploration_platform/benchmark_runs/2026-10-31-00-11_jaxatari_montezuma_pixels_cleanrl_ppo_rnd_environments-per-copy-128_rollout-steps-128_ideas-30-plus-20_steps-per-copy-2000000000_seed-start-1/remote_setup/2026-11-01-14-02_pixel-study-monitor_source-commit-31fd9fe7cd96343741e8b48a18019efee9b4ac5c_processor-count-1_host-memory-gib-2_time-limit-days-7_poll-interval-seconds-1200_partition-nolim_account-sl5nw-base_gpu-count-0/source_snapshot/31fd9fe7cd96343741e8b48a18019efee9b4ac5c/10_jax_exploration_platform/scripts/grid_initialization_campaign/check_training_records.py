"""Exercise the production training entry point and idempotent completed-copy records."""

import argparse
from pathlib import Path
import subprocess

from common import PYTHON, read_json, write_json


def main():
    """Keep short verification outputs separate from all scientific result selection."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    folder = args.run_dir/'data/integration_verification'
    for policy in ('tabular', 'neural'):
        specification = {'name': f'verification_policy-{policy}_steps-256_seeds-0-3',
            'wave': 'verification', 'geometry': {'kind': 'snake', 'width': 4, 'turns': 2},
            'policy_kind': policy, 'field_kind': 'random_network', 'severity_exponent': 0,
            'settings': [{'learning_rate': 2**-8, 'intrinsic_coefficient': 1.,
                         'entropy_coefficient': 2**-10, 'rollout_steps': 128, 'gamma_int': .99}],
            'total_timesteps': 256, 'seed_start': 0, 'seed_count': 4}
        path = folder/'specifications'/(specification['name']+'.json')
        write_json(path, specification)
        command = [str(PYTHON), str(Path(__file__).parent/'train_unit.py'),
                   '--run-dir', str(folder), '--specification', str(path)]
        subprocess.run(command, check=True)
        result = folder/'data/research'/specification['name']/'summary.json'
        previous = result.read_bytes()
        subprocess.run(command, check=True)
        if result.read_bytes() != previous:
            raise AssertionError('Re-running completed work changed its result.')
        summary = read_json(result)
        if not summary['completed'] or len(summary['copy_records']) != 4:
            raise AssertionError('Production output omitted independent policy results.')
    write_json(args.run_dir/'training_record_checks.json', {'passed': True,
        'policy_kinds': ['tabular', 'neural'], 'idempotent': True})


if __name__ == '__main__':
    main()
