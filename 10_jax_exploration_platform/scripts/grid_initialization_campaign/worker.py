"""Run resource-compatible study tasks from one shared queue."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from common import PYTHON, claim, finish, now, write_json


def hardware():
    """Read the allocated card before importing a numerical framework."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    rows = [row.strip().split(',') for row in result.stdout.splitlines()]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    index = int(visible) if visible.isdecimal() and int(visible) < len(rows) else 0
    name, memory, driver = rows[index]
    return {'gpu': name.strip(), 'memory_mb': int(memory.strip()), 'driver': driver.strip(),
            'hostname': os.uname().nodename, 'recorded_at': now(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def main():
    """Process queue units sequentially with bounded measurement subprocesses."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['checks', 'infrastructure', 'research'], required=True)
    parser.add_argument('--max-seconds', type=int, default=14000)
    parser.add_argument('--canary', action='store_true')
    args = parser.parse_args()
    os.umask(0o002)
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    folder = args.run_dir / 'slurm/jobs' / f'{job_id}_{os.uname().nodename}'
    card = hardware()
    write_json(folder / 'hardware.json', card)
    if args.canary:
        # A new GPU class first completes a bounded compatibility and memory measurement.
        gpu_class = os.environ.get('GRID_GPU_CLASS', card['gpu'].replace(' ', '_'))
        output = args.run_dir / 'infrastructure/compatibility' / (gpu_class + '.json')
        command = [str(PYTHON), str(Path(__file__).resolve().parent/'benchmark.py'),
            '--run-dir', str(args.run_dir), '--idea', 'baseline', '--copies', '32',
            '--policy-kind', 'all', '--geometry', 'all', '--output', str(output)]
        try:
            result = subprocess.run(command, timeout=590)
        except subprocess.TimeoutExpired:
            write_json(output, {'completed': False, 'failure': 'Compatibility measurement exceeded 590 seconds.',
                               'hardware': card})
            raise SystemExit(124)
        if output.exists():
            from common import read_json
            record = read_json(output)
            record.update(gpu_type=gpu_class, hardware=card)
            write_json(output, record)
        raise SystemExit(result.returncode)
    deadline = time.monotonic() + args.max_seconds
    while time.monotonic() < deadline - 650:
        if (args.run_dir / 'STOP').exists():
            break
        task = claim(args.run_dir, args.stage, card['memory_mb'])
        if task is None:
            break
        write_json(folder / 'assignment.json', task)
        # Execute the immutable sibling entry point, even when the task was prepared live.
        entrypoint = Path(__file__).resolve().parent / Path(task['command'][0]).name
        command = [str(PYTHON), str(entrypoint), *task['command'][1:]]
        environment = dict(os.environ, PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           XLA_PYTHON_CLIENT_PREALLOCATE='false',
                           XLA_PYTHON_CLIENT_MEM_FRACTION='0.75')
        started = time.monotonic()
        with (folder / (task['name'] + '.log')).open('a') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
            limit = task.get('limit_seconds', min(7200, int(deadline-time.monotonic()-60)))
            timed_out = False
            try:
                return_code = process.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                import signal
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return_code = 124
        result = {'exit_code': return_code, 'timed_out': timed_out,
                  'wall_seconds': time.monotonic()-started, 'hardware': card,
                  'log': str(folder / (task['name'] + '.log'))}
        # Every failed measurement remains a visible, completed measurement attempt.
        finish(args.run_dir, task, result, successful=return_code == 0)
        print(task['name'], result, flush=True)


if __name__ == '__main__':
    main()
