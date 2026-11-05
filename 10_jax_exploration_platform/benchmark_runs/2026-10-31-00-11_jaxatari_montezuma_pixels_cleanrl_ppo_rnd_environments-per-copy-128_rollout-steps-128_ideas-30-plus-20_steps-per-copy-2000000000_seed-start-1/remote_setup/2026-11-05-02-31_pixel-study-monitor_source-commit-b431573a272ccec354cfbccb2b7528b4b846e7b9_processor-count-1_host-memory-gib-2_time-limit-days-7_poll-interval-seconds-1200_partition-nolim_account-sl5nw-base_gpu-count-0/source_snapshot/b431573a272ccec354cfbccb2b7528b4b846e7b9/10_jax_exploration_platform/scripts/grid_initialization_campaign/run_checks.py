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
    parser.add_argument('--document-only', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical tests require a Slurm allocation.')
    environment = dict(os.environ, PYTHONPATH=str(PLATFORM/'src'), JAX_ENABLE_X64='true')
    # The reference environment supplies pytest; its JAX versions match production.
    reference_python = PROJECT/'.venvs/atari_pixels_jax/bin/python'
    commands = [[str(reference_python), '-m', 'pytest', '-q',
        str(PLATFORM/'tests/test_grid_initialization_environment.py'),
        str(PLATFORM/'tests/test_grid_initialization_report.py'),
        str(PLATFORM/'tests/grid_initialization'),
        str(PLATFORM/'scripts/grid_initialization_campaign/tests')],
        [str(PYTHON), '-c',
         'import jax,numpy as np; '
         'from exploration_platform.grid_initialization.config import GridPPOConfig; '
         'from exploration_platform.grid_initialization.environments import make_geometry; '
         'from exploration_platform.grid_initialization.training import GridTrainer; '
         'assert jax.default_backend()=="gpu"; '
         'w=make_geometry("straight",4); '
         'c=GridPPOConfig(n_copies=2,num_steps=8,num_minibatches=2,update_epochs=1,total_timesteps=8); '
         't=GridTrainer(c,w.transition_table,w.coordinates,np.ones((2,4),np.float32)); '
         's,m=t.train_iteration(t.initialize([1,2]));jax.block_until_ready(s); '
         'assert np.asarray(s.counts).sum()==16; print("Production environment GPU smoke passed.")'],
        [str(PYTHON), str(Path(__file__).parent/'check_training_records.py'), '--run-dir', str(args.run_dir)],
        [str(PYTHON), str(Path(__file__).parent/'build_report.py'), '--run-dir', str(args.run_dir)]]
    results = []
    if args.document_only:
        from common import read_json
        previous = read_json(args.run_dir/'checks.json')['results']
        if len(previous) != 4 or any(item['exit_code'] for item in previous[:3]):
            raise RuntimeError('Document-only verification requires completed numerical and production checks.')
        results = previous[:3]
        commands = commands[-1:]
    for command in commands:
        result = subprocess.run(command, env=environment)
        results.append({'command': command, 'exit_code': result.returncode})
        if result.returncode:
            write_json(args.run_dir/'checks.json', {'passed': False, 'at': now(), 'results': results})
            sys.exit(result.returncode)
    write_json(args.run_dir/'checks.json', {'passed': True, 'at': now(), 'results': results})


if __name__ == '__main__':
    main()
