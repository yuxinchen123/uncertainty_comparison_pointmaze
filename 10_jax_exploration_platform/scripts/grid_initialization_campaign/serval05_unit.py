"""Run one count-bonus unit of the fixed-settings study on serval05, then queue the next serval05 unit.

serval05 has one H100 NVL and no Slurm; each call is one unit of its shared queue, run under the
card's lock, so other sessions get the card between units. Only count-bonus units go here: they
take about 5 to 10 minutes on this card, while online-RND units take about 2 hours.
"""

import argparse
import os
from pathlib import Path
import subprocess

from common import PYTHON, finish, now, queue_lock, read_json, write_json

ENQUEUE = Path('/p/rlprojects/.claude/skills/serval05-queue/scripts/enqueue.sh')


def claim_count_unit(run_dir):
    """Claim the first pending count-bonus research unit, marking the claim as serval05's."""
    with queue_lock(run_dir):
        for path in sorted((run_dir/'queue/pending').glob('fixed-*.json')):
            task = read_json(path)
            if task['stage'] == 'research' and task.get('intrinsic_source') == 'count':
                task.update(job_id='serval05-queue', account=os.environ.get('USER'), started_at=now(),
                            hostname=os.uname().nodename)
                write_json(run_dir/'queue/running'/path.name, task)
                path.unlink()
                return task
    return None


def main():
    """Claim, train, record, and queue the next serval05 unit while count units remain."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--unit-script', type=Path, required=True)
    args = parser.parse_args()
    if (args.run_dir/'STOP').exists() or (args.run_dir/'SWEEP_COMPLETE').exists():
        return
    task = claim_count_unit(args.run_dir)
    if task is None:
        print('no pending count-bonus unit; serval05 chain ends', flush=True)
        return
    entrypoint = Path(__file__).resolve().parent/Path(task['command'][0]).name
    log = args.run_dir/'logs'/f"serval05_{task['name'][:150]}.log"
    with log.open('a') as handle:
        result = subprocess.run([str(PYTHON), str(entrypoint), *task['command'][1:]], stdout=handle,
                                stderr=subprocess.STDOUT, env=dict(os.environ, GRID_GPU_CLASS='nvidia_h100_nvl'))
    finish(args.run_dir, task, {'exit_code': result.returncode, 'hardware': {'gpu': 'NVIDIA H100 NVL',
           'hostname': os.uname().nodename}, 'log': str(log)}, successful=result.returncode == 0)
    if result.returncode == 0:
        subprocess.run(['bash', str(ENQUEUE), 'grid-fixed-count-unit', str(args.unit_script)], check=True)


if __name__ == '__main__':
    main()
