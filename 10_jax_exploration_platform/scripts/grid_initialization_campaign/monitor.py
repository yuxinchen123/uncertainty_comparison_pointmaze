"""Inspect progress, advance research, and refill measured GPU capacity."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import time

from campaign import advance
from common import (ACCOUNTS, PYTHON, append_json, now, own_live_jobs, pacific,
                    queue_counts, read_json, write_json)
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
    """Choose from complete, correct measurements using both geometries' actual field names."""
    # Before: four benchmark cases. After: one conservative profile per policy family.
    profiles = {}
    for path in (run_dir/'data/benchmarks').rglob('*.json'):
        document = read_json(path)
        if not document.get('completed'):
            continue
        for policy in ('tabular', 'neural'):
            cases = [case for case in document.get('cases', []) if case['policy_kind'] == policy]
            if ({case['geometry'] for case in cases} != {'straight', 'snake'} or not all(
                    case.get('completed') and case.get('correctness', {}).get('passed') for case in cases)):
                continue
            peak = max(case['peak_device_memory_mb'] for case in cases)
            if 1.25*peak > 7000:
                continue
            rate = min(case['steps_per_second_total'] for case in cases)
            if rate > profiles.get(policy, {}).get('steps_per_second_total', 0):
                profiles[policy] = {'copies': min(case['copies'] for case in cases),
                    'changes': document.get('changes', {}), 'steps_per_second_total': rate,
                    'steps_per_second_per_copy': min(case['steps_per_second_per_copy'] for case in cases),
                    'peak_device_memory_mb': peak, 'device_kind': document.get('device_kind'),
                    'source': str(path), 'both_geometries_checked': True}
    if profiles:
        write_json(run_dir/'infrastructure/production.json', profiles)
    return profiles


def stop_workers(run_dir, live):
    """Cancel only this study's recorded live training ids after writing STOP."""
    # Never derive actionable identifiers from an account-wide query.
    (run_dir/'STOP').write_text(now()+'\n')
    cancelled = []
    for account in ('sl5nw', 'yuxinchen'):
        ledger = run_dir/'slurm'/f'submitted_jobids_initial_bonus_{account}.txt'
        if not ledger.exists():
            continue
        permitted = set(ledger.read_text().split())
        identifiers = []
        for path in (run_dir/'slurm/submissions').glob('*.json'):
            record = read_json(path)
            identifier = record['job_id']
            if (record['account'] == account and record['stage'] != 'monitor'
                    and identifier in permitted and identifier in live
                    and identifier != os.environ.get('SLURM_JOB_ID')):
                identifiers.append(identifier)
        if identifiers:
            subprocess.run([str(ACCOUNTS/'cancel_as.sh'), 'uva-'+account,
                '--idfile', str(ledger), *sorted(identifiers)], check=True, stdin=subprocess.DEVNULL)
            cancelled.extend(identifiers)
    append_json(run_dir/'infrastructure_events.jsonl', {'at': now(),
                'event': 'stopped_recorded_study_jobs', 'job_ids': cancelled})
    return cancelled


def status_lines(run_dir, state, live, counts):
    """Display actual pools and card requests separately from completed unit counts."""
    lines = ['run or job           pool              gpu                state      progress']
    submissions = {record['job_id']: record for path in (run_dir/'slurm/submissions').glob('*.json')
                   if (record := read_json(path))['job_id'] in live}
    for group, running in (('TRAINING', True), ('WAITING', False)):
        rows = [(identifier, row) for identifier, row in live.items()
                if (row[3] == 'RUNNING') == running]
        if not rows:
            continue
        lines.append(group)
        for identifier, row in rows:
            record = submissions.get(identifier, {})
            name = 'study '+record.get('stage', 'job')
            pool = f'{row[2]} {row[1]}'
            card = f'{record.get("cards", 0)} {record.get("gpu_type") or "none"}'
            progress = f'{counts["done"]} done; {counts["running"]} active'
            lines.append(f'{name[:20]:20} {pool[:17]:17} {card[:18]:18} {row[3][:10]:10} {progress}')
    lines += ['FINISHED', f'completed study units: {counts["done"]}; failed units: {counts["failed"]}',
              'PLANNED', f'{counts["pending"]} queued units; phase {state["phase"]}', 'Updated '+pacific()]
    return lines


def tick(run_dir):
    """Enforce the deadline before advancing or submitting further work."""
    run_dir = Path(run_dir)
    live = own_live_jobs(run_dir)
    previous = read_json(run_dir/'campaign_state.json')
    deadline = previous.get('deadline')
    expired = bool(deadline and datetime.now().astimezone() >= datetime.fromisoformat(deadline))
    if expired or (run_dir/'STOP').exists():
        state = dict(previous, deadline_reached=expired, stopped_at=now())
        stop_workers(run_dir, live)
        if not expired:
            state['phase'] = 'stopped'
        write_json(run_dir/'campaign_state.json', state)
    else:
        recover_claims(run_dir, live)
        production_profiles(run_dir)
        state = advance(run_dir)
        if state['phase'] != 'complete':
            stage = ('checks' if state['phase'] == 'checks' else 'infrastructure'
                     if 'measurements' in state['phase'] else 'research')
            top_up(run_dir, stage)
        else:
            stop_workers(run_dir, live)
    # Re-read after submissions so newly queued jobs appear in this tick's report.
    live = own_live_jobs(run_dir)
    counts = queue_counts(run_dir)
    summary = {'at': now(), 'phase': state['phase'], 'counts': counts,
               'deadline': state.get('deadline'), 'deadline_reached': expired,
               'cadence_seconds': 1200, 'jobs': list(live.values())}
    write_json(run_dir/'status.json', summary)
    append_json(run_dir/'metrics.jsonl', summary)
    lines = status_lines(run_dir, state, live, counts)
    (run_dir/'status.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)
    return state, live


def main():
    """Run the monitor through research completion or the explicit execution deadline."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args()
    while True:
        state, live = tick(args.run_dir)
        if state['phase'] in ('complete', 'stopped') or state.get('deadline_reached'):
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
