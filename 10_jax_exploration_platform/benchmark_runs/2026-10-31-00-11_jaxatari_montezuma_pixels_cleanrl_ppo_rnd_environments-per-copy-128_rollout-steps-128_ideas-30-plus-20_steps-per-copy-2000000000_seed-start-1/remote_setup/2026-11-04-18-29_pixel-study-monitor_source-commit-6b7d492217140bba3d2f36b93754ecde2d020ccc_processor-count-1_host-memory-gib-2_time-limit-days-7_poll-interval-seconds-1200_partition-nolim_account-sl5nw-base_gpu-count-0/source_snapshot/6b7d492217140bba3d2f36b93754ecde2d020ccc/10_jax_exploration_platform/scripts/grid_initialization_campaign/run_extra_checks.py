"""Run additional multi-update and orchestration checks on allocated CPU resources."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from common import PLATFORM, PROJECT, now, write_json


def main():
    """Validate uneven-field updates without occupying the profiling GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Additional numerical tests require allocated compute resources.')
    environment = dict(os.environ, PYTHONPATH=str(PLATFORM/'src'), JAX_PLATFORMS='cpu',
                       JAX_ENABLE_X64='true', PYTHONNOUSERSITE='1')
    python = PROJECT/'.venvs/atari_pixels_jax/bin/python'
    commands = [
        [str(python), '-m', 'pytest', '-q', str(PLATFORM/'tests/grid_initialization/test_campaign_decisions.py'),
         str(PLATFORM/'scripts/grid_initialization_campaign/tests/test_performance_report.py')],
        [str(python), '-m', 'pytest', '-q', str(PLATFORM/'tests/grid_initialization/test_training.py'),
         '-k', 'uneven_fields_multistep_equivalence']]
    results = []
    for command in commands:
        result = subprocess.run(command, env=environment)
        results.append({'command': command, 'exit_code': result.returncode})
        write_json(args.run_dir/'extra_checks.json', {'passed': result.returncode == 0,
            'completed': len(results) == len(commands), 'results': results, 'at': now()})
        if result.returncode:
            sys.exit(result.returncode)


if __name__ == '__main__':
    main()
