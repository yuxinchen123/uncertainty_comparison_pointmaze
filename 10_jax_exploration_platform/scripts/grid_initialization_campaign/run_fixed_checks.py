"""Run the fixed-settings study's checks inside an allocated GPU job and record each result."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from common import PLATFORM, PROJECT, PYTHON, now, write_json


def main():
    """Tests, the unchanged-count-path check and the tiny real units, in that order."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical checks require a Slurm allocation.')
    environment = dict(os.environ, PYTHONPATH=str(PLATFORM/'src'), JAX_ENABLE_X64='true')
    here = Path(__file__).parent
    commands = [[str(PROJECT/'.venvs/atari_pixels_jax/bin/python'), '-m', 'pytest', '-q',
                 str(PLATFORM/'tests/grid_initialization')],
                [str(PYTHON), str(here/'check_count_path_unchanged.py'), '--run-dir', str(args.run_dir)],
                [str(PYTHON), str(here/'check_fixed_unit.py'), '--run-dir', str(args.run_dir)]]
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
