"""Run at the start of a fast-card job: move one held development unit into pending, under the queue lock.

The job's worker starts about a second later and claims it. File moves only (standard library).
"""
import fcntl
import os
import sys
from datetime import datetime
from pathlib import Path

run = Path(sys.argv[1])
queue = run / 'queue'
with (queue / 'queue.lock').open('a') as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    units = sorted((queue / 'held_for_fast_cards').glob('*.json'))
    text = 'no held unit left'
    if units:
        units[0].rename(queue / 'pending' / units[0].name)
        text = f'released {units[0].name}'
with (run / 'infrastructure/fast_card_release.log').open('a') as log:
    log.write(f"{datetime.now().astimezone().isoformat()} job {os.environ.get('SLURM_JOB_ID')} on {os.uname().nodename}: {text}\n")
