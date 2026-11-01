"""Persistently inspect progress, advance research, and refill eligible GPU capacity."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import time

from campaign import advance
from common import PYTHON, append_json, now, own_live_jobs, pacific, queue_counts, read_json, write_json
from submit import top_up


def recover_claims(run_dir, live):
    """Release claims only after their exact owning job is absent from Slurm."""
    from common import queue_lock
    with queue_lock(run_dir):
        for path in (run_dir/'queue/running').glob('*.json'):
            task = read_json(path)
            if task['job_id'] not in live:
                task['recovery_count'] = task.get('recovery_count', 0)+1
                target = 'pending' if task['recovery_count'] <= 2 else 'failed'
                write_json(run_dir/'queue'/target/path.name, task)
                path.unlink()
                append_json(run_dir/'infrastructure_events.jsonl', {'at': now(),
                    'event': 'own_job_ended_with_unfinished_unit', 'task': task['name'],
                    'job_id': task['job_id'], 'destination': target})


def production_profiles(run_dir):
    """Retain only execution choices with actual passing correctness measurements."""
    profiles = {}
    for path in (run_dir/'data/benchmarks').rglob('*.json'):
        document = read_json(path)
        for case in document.get('cases', []):
            if not case.get('completed') or not case.get('correctness', {}).get('passed'):
                continue
            policy = case['policy_kind']
            rate = case.get('steps_per_second_total', 0)
            # The first admission remains usable by the smallest modern cards.
            if case.get('peak_device_memory_mb', 0) > 5500:
                continue
            if rate > profiles.get(policy, {}).get('steps_per_second_total', 0):
                profiles[policy] = {'copies': case['copies'], 'changes': document.get('changes', {}),
                    'steps_per_second_total': rate, 'source': str(path)}
    if profiles:
        write_json(run_dir/'infrastructure/production.json', profiles)


def tick(run_dir):
    """Perform an actual research/controller tick and publish one durable status."""
    live = own_live_jobs(run_dir)
    recover_claims(run_dir, live)
    state = advance(run_dir)
    counts = queue_counts(run_dir)
    production_profiles(run_dir)
    deadline = state.get('deadline')
    expired = deadline and datetime.now().astimezone() >= datetime.fromisoformat(deadline)
    if expired:
        (run_dir/'STOP').write_text(now()+'\n')
        state['deadline_reached'] = True
    if not expired and state['phase'] != 'complete':
        stage = ('checks' if state['phase'] == 'checks' else 'infrastructure'
                 if 'measurements' in state['phase'] else 'research')
        top_up(run_dir, stage)
    summary = {'at': now(), 'phase': state['phase'], 'counts': counts,
               'deadline': deadline, 'deadline_reached': bool(expired),
               'jobs': list(live.values())}
    write_json(run_dir/'status.json', summary)
    append_json(run_dir/'metrics.jsonl', summary)
    lines = ['run or job                 pool                  gpu           state      progress',
             f'initial bonus study        UVA both accounts     see jobs      {state["phase"][:12]:12} '
             f'{counts["done"]} done, {counts["running"]} running, {counts["pending"]} pending',
             'Updated '+pacific()]
    (run_dir/'status.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)
    return state, live


def main():
    """Keep the monitor alive across queued or running phases on a compute node."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args()
    while True:
        state, live = tick(args.run_dir)
        if state['phase'] == 'complete' or state.get('deadline_reached'):
            # Numerical aggregation and plotting run here, never on the login node.
            if (Path(__file__).parent/'report.py').exists():
                subprocess.run([str(PYTHON), str(Path(__file__).parent/'report.py'),
                                '--run-dir', str(args.run_dir)], check=True)
            write_json(args.run_dir/'summary.json', {'state': state, 'at': now(),
                       'counts': queue_counts(args.run_dir)})
            break
        if not args.loop:
            break
        time.sleep(1200)


if __name__ == '__main__':
    main()
