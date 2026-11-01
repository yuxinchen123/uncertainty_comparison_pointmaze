"""Run required numerical verification only inside an allocated GPU job."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from common import PLATFORM, PROJECT, PYTHON, now, write_json


def main():
    """Require the grid tests and record their actual process status."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical tests require a Slurm allocation.')
    environment = dict(os.environ, PYTHONPATH=str(PLATFORM/'src'), JAX_ENABLE_X64='true')
    commands = [[str(PYTHON), '-m', 'pytest', '-q',
        str(PLATFORM/'tests/test_grid_initialization_environment.py'),
        str(PLATFORM/'tests/grid_initialization'),
        str(PLATFORM/'scripts/grid_initialization_campaign/tests')]]
    results = []
    for command in commands:
        result = subprocess.run(command, env=environment)
        results.append({'command': command, 'exit_code': result.returncode})
        if result.returncode:
            write_json(args.run_dir/'checks.json', {'passed': False, 'at': now(), 'results': results})
            sys.exit(result.returncode)
    write_json(args.run_dir/'checks.json', {'passed': True, 'at': now(), 'results': results})


if __name__ == '__main__':
    main()
