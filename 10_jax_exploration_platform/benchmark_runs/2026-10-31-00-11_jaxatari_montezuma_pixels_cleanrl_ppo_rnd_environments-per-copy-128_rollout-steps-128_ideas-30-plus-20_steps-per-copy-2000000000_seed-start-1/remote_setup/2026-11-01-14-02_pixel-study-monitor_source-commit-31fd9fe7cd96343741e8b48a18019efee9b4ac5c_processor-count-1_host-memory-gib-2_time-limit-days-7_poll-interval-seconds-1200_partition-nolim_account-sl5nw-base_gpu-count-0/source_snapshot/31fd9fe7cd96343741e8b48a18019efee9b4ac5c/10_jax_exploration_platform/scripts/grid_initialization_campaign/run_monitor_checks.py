"""Verify report scheduling and publication without using training GPUs."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from common import PLATFORM, PROJECT, now, write_json


def main():
    """Exercise real reporting on small fixtures and mock only scheduler side effects."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Checks require an allocated compute node.')
    command = [str(PROJECT/'.venvs/atari_pixels_jax/bin/python'), '-m', 'pytest', '-q',
        str(PLATFORM/'scripts/grid_initialization_campaign/tests'),
        str(PLATFORM/'tests/grid_initialization/test_campaign_decisions.py'),
        str(PLATFORM/'tests/test_grid_initialization_report.py')]
    environment = dict(os.environ, PYTHONPATH=str(PLATFORM/'src'), JAX_PLATFORMS='cpu',
                       JAX_ENABLE_X64='true', PYTHONNOUSERSITE='1')
    result = subprocess.run(command, env=environment)
    write_json(args.run_dir/'monitor_checks.json', {'passed': result.returncode == 0,
                'command': command, 'at': now()})
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
