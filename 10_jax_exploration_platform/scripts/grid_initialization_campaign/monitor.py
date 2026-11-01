"""Inspect progress, advance research, and refill measured GPU capacity."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import textwrap
import time

from campaign import advance
from common import (ACCOUNTS, PYTHON, append_json, now, own_live_jobs, pacific,
                    queue_counts, read_json, write_json)
from submit import RESOURCE_FILE, top_up, submit_report_job
from run_report import completed_result_count


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
            if (record['account'] == account and record['stage'] not in ('monitor', 'report')
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


def ensure_report_job(run_dir, final=False):
    """Coalesce plotting into one live owner job and wait for successful final coverage."""
    from common import queue_lock
    # Serialize the decision with other control ticks; re-read ids after acquiring the lock.
    with queue_lock(run_dir):
        live = own_live_jobs(run_dir)
        active = [record for path in (run_dir/'slurm/submissions').glob('*.json')
                  if (record := read_json(path))['stage'] == 'report' and record['job_id'] in live]
        count = completed_result_count(run_dir)
        marker = run_dir/'last_rendered_result_count.json'
        rendered = read_json(marker) if marker.exists() else {'count': 0, 'final': False}
        ready = rendered['count'] >= count and (not final or rendered.get('final', False))
        if active or ready:
            return {'completed': ready, 'job_id': active[0]['job_id'] if active else None,
                    'result_count': count, 'rendered_result_count': rendered['count']}
        identifier = submit_report_job(run_dir, final=final)
        return {'completed': False, 'job_id': identifier, 'result_count': count,
                'rendered_result_count': rendered['count']}


def last_progress_record(path):
    """Read at most eight KiB from a progress file and ignore an unfinished final line."""
    if not path.exists():
        return None
    with path.open('rb') as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size-8192))
        content = handle.read(8192)
    lines = content.splitlines()
    if size > 8192:
        lines = lines[1:]
    if content and not content.endswith(b'\n'):
        lines = lines[:-1]
    return __import__('json').loads(lines[-1]) if lines else None


def inspect_progress(run_dir):
    """Record per-task progress, unchanged forty-minute observations, and collaborator reports."""
    observation_path = run_dir/'monitor_progress_observations.json'
    prior = read_json(observation_path) if observation_path.exists() else {}
    current, warnings = {}, []
    for path in (run_dir/'queue/running').glob('*.json'):
        task = read_json(path)
        progress_path = run_dir/'data/research'/task['name']/'progress.jsonl'
        size = progress_path.stat().st_size if progress_path.exists() else 0
        previous = prior.get(task['name'])
        unchanged_since = previous['unchanged_since'] if previous and previous['size'] == size else time.time()
        progress = last_progress_record(progress_path)
        stale = time.time()-unchanged_since >= 2400
        current[task['name']] = {'size': size, 'unchanged_since': unchanged_since,
                                  'progress': progress, 'stale': stale, 'job_id': task['job_id']}
        if stale:
            warnings.append('Progress unchanged for forty minutes: '+task['name'])
    warnings += ['Collaborator report: '+path.name
                 for path in (run_dir/'for_collaborator/problems/open').glob('*') if path.is_file()]
    write_json(observation_path, current)
    write_json(run_dir/'monitor_warnings.json', {'at': now(), 'warnings': warnings})
    return current


def status_lines(run_dir, state, live, counts):
    """Display actual pools and card requests separately from completed unit counts."""
    lines = ['run or job           pool              gpu                state      progress']
    submissions = {record['job_id']: record for path in (run_dir/'slurm/submissions').glob('*.json')
                   if (record := read_json(path))['job_id'] in live}
    progress_path = run_dir/'monitor_progress_observations.json'
    observations = read_json(progress_path) if progress_path.exists() else {}
    # The resource catalog supplies readable names, rather than truncated scheduler identifiers.
    gpu_names = {entry['gres_type']: entry['display_name']
                 for entry in read_json(RESOURCE_FILE)['classes']}
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
            gpu_type = record.get('gpu_type')
            card = f'{record.get("cards", 0)} {gpu_names.get(gpu_type, gpu_type or "none")}'
            tasks = [(task_name, observation) for task_name, observation in observations.items()
                     if observation['job_id'] == identifier and observation.get('progress')]
            if not tasks:
                tasks = [(name, {'progress': None, 'stale': False})]
            for task_name, observation in tasks:
                values = observation['progress']
                progress = (f'copies {values["completed_copies"]}/{values["total_copies"]}; '
                    f'iterations {values["iteration"]}/{values["total_iterations"]}' if values
                    else f'{counts["done"]} done; {counts["running"]} active')
                if observation['stale']:
                    progress += ' unchanged >40min'
                prefix = (f'{task_name.replace("_", " ")[:18]:18} {pool[:14]:14} '
                          f'{card:20} {row[3][:8]:8} ')
                wrapped = textwrap.wrap(progress, width=max(1, 100-len(prefix)))
                lines.extend((prefix if index == 0 else ' '*len(prefix))+text
                             for index, text in enumerate(wrapped))
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
    # A separate CPU job performs plotting; this tick only submits or observes it.
    final = state['phase'] in ('complete', 'stopped') or state.get('deadline_reached', False)
    report = ensure_report_job(run_dir, final=final)
    state['final_report_ready'] = final and report['completed']
    write_json(run_dir/'report_job_status.json', report)
    # Re-read after submissions so newly queued jobs appear in this tick's report.
    live = own_live_jobs(run_dir)
    counts = queue_counts(run_dir)
    inspect_progress(run_dir)
    from performance_report import write_report
    write_report(run_dir)
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
        tick_started = time.monotonic()
        state, live = tick(args.run_dir)
        if state.get('final_report_ready'):
            write_json(args.run_dir/'summary.json', {'state': state, 'at': now(),
                       'counts': queue_counts(args.run_dir)})
            break
        if not args.loop:
            break
        time.sleep(max(0, 1200-(time.monotonic()-tick_started)))


if __name__ == '__main__':
    main()
