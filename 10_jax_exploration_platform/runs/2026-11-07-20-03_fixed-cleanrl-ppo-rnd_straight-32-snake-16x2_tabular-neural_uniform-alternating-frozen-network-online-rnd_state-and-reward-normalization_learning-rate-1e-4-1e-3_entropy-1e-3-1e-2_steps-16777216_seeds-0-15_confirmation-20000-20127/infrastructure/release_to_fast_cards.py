"""Hold development units aside and release one to the pending queue when a listed fast-card worker starts.

File moves only (standard library), under the campaign's queue lock. Slow-card workers that finish
meanwhile find no pending unit and exit, so the held units go to the fast cards.
"""
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

run = Path(sys.argv[1])
queue = run / 'queue'
held = queue / 'held_for_fast_cards'
fast_list = run / 'infrastructure/fast_card_workers.txt'
log = run / 'infrastructure/fast_card_release.log'
served = set()


def note(text):
    with log.open('a') as handle:
        handle.write(f'{datetime.now().astimezone().isoformat()} {text}\n')


def running_ids(ids):
    if not ids:
        return set()
    output = subprocess.run(['squeue', '-h', '-t', 'R', '-j', ','.join(sorted(ids)), '-o', '%i'],
                            capture_output=True, text=True).stdout
    return set(output.split())


while True:
    ids = set(fast_list.read_text().split()) if fast_list.exists() else set()
    started = running_ids(ids - served)
    with (queue / 'queue.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        units = sorted(held.glob('*.json'))
        if not units:
            note('no held units left; stopping')
            break
        phase_running = sum(1 for path in (queue / 'running').glob('fixed-*.json'))
        if phase_running < 4:
            # Never let the development phase look finished while units are held aside.
            for path in units:
                path.rename(queue / 'pending' / path.name)
            note(f'only {phase_running} study units running; released all {len(units)} held units')
            break
        for job in sorted(started):
            if not units:
                break
            path = units.pop(0)
            path.rename(queue / 'pending' / path.name)
            served.add(job)
            note(f'job {job} started; released {path.name}')
    time.sleep(5)
