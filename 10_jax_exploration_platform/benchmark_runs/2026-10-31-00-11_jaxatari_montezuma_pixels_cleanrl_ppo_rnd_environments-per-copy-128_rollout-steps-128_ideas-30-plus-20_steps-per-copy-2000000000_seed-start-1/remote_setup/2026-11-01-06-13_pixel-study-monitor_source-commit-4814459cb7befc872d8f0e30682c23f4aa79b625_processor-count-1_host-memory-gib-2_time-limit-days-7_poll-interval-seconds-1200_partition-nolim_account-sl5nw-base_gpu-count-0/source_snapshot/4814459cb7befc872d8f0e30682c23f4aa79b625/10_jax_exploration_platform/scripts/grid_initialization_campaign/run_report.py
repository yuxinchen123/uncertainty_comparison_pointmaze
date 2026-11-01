"""Run heavy report generation on an allocated CPU job and record successful coverage."""

import argparse
import os
from pathlib import Path
import socket
import subprocess

from common import PYTHON, now, read_json, write_json


def completed_result_count(run_dir):
    """Count only final successful result summaries using small JSON reads."""
    return sum(bool(read_json(path).get('completed'))
               for path in (Path(run_dir)/'data/research').glob('*/summary.json'))


def render_report(run_dir, final=False):
    """Advance the rendered-result marker only after the complete build succeeds."""
    # Capture the input count before plotting; later arrivals require a subsequent report.
    run_dir = Path(run_dir)
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('portal', 'login')):
        raise RuntimeError('Heavy report generation requires an allocated compute node.')
    count = completed_result_count(run_dir)
    attempt = {'job_id': os.environ['SLURM_JOB_ID'], 'started_at': now(),
               'completed_result_count_at_start': count, 'final': final, 'completed': False}
    path = run_dir/'slurm/report_attempts'/(attempt['job_id']+'.json')
    write_json(path, attempt)
    subprocess.run([str(PYTHON), str(Path(__file__).parent/'build_report.py'),
                    '--run-dir', str(run_dir)], check=True)
    # A failed subprocess leaves the prior marker unchanged and can be retried next tick.
    write_json(run_dir/'last_rendered_result_count.json',
               {'count': count, 'at': now(), 'job_id': attempt['job_id'], 'final': final})
    write_json(path, dict(attempt, completed=True, finished_at=now()))
    return count


def main():
    """Execute the heavy report wrapper inside its separately recorded Slurm job."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    render_report(args.run_dir, final=args.final)


if __name__ == '__main__':
    main()
