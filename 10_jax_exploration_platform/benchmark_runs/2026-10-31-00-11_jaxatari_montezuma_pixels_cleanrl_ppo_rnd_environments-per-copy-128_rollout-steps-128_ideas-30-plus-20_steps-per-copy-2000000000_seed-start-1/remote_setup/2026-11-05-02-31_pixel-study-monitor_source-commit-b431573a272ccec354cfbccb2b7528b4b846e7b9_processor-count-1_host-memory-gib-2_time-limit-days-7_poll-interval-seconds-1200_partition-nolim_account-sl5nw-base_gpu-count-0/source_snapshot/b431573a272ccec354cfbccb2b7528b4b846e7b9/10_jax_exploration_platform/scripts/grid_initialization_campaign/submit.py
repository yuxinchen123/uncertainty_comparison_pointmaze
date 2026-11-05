"""Submit GPU jobs within live account, partition, node, and work limits."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from common import ACCOUNTS, PROJECT, PYTHON, now, own_live_jobs, read_json, write_json

RESOURCE_FILE = Path('/p/rlprojects/.claude/skills/uva-submit-gpu-sweep/school_compute_resource/server_introduction.json')
CPU_PER_CARD = 4
# Confirmation jobs measured at most 1.11 GiB of host memory per allocated card.
# This request applies only to newly submitted training jobs.
MEMORY_GIB_PER_CARD = 8


def slurm_fields(row):
    """Read the key/value fields of one scheduler record."""
    return dict(part.split('=', 1) for part in row.split() if '=' in part)


def duration_seconds(value):
    """Convert Slurm's day and clock duration to seconds."""
    days, clock = value.split('-', 1) if '-' in value else ('0', value)
    hours, minutes, seconds = map(int, clock.split(':'))
    return int(days)*86400 + hours*3600 + minutes*60 + seconds


def allowed_seconds(partition, reservation=None, canary=False):
    """Respect partition, maintenance, and reservation limits before every submission."""
    # All ordinary jobs request the actual maximum; compatibility measurements are bounded.
    fields = slurm_fields(subprocess.check_output(
        ['scontrol', 'show', 'partition', partition, '-o'], text=True))
    seconds = duration_seconds(fields['MaxTime'])
    current = datetime.now().astimezone()
    rows = subprocess.check_output(['scontrol', 'show', 'reservation', '-o'], text=True)
    for row in rows.splitlines():
        record = slurm_fields(row)
        if 'MAINT' in record.get('Flags', ''):
            start = datetime.fromisoformat(record['StartTime']).astimezone()
            if start > current:
                seconds = min(seconds, int((start-current).total_seconds())-1800)
        if record.get('ReservationName') == reservation:
            end = datetime.fromisoformat(record['EndTime']).astimezone()
            seconds = min(seconds, int((end-current).total_seconds())-60)
    seconds = min(seconds, 2400) if canary else seconds
    if seconds < 900:
        raise RuntimeError('No submission window of at least fifteen minutes remains.')
    return seconds


def submit_job(run_dir, identity, stage, partition='gpu', node=None, gpu_type=None,
               cards=1, reservation=None, monitor=False, canary=False, exclude_nodes=(),
               report=False, final_report=False):
    """Submit one recorded source snapshot through the account's exact-id wrapper."""
    # Build scripts from the frozen snapshot, never from the mutable project source.
    run_dir = Path(run_dir)
    account = identity.removeprefix('uva-')
    scripts = run_dir/'slurm/scripts'
    scripts.mkdir(parents=True, exist_ok=True)
    (run_dir/'logs').mkdir(exist_ok=True)
    name = f'grid-initial-{"monitor" if monitor else stage}-{time.time_ns()}'
    script = scripts/(name+'.sh')
    source = read_json(run_dir/'source.json')
    directory = Path(source['snapshot'])/'10_jax_exploration_platform/scripts/grid_initialization_campaign'
    seconds = allowed_seconds(partition, reservation, canary)
    if monitor:
        body = f'exec {shlex.quote(str(PYTHON))} {shlex.quote(str(directory/"monitor.py"))} --run-dir {shlex.quote(str(run_dir))} --loop\n'
    elif report:
        final = ' --final' if final_report else ''
        body = (f'exec {shlex.quote(str(PYTHON))} {shlex.quote(str(directory/"run_report.py"))} '
                f'--run-dir {shlex.quote(str(run_dir))}{final}\n')
    else:
        extra = ' --canary' if canary else ''
        body = ('IFS=, read -ra devices <<< "$CUDA_VISIBLE_DEVICES"\n'
            f'test "${{#devices[@]}}" -eq {cards}\n'
            'pids=()\nfor device in "${devices[@]}"; do\n'
            f'  CUDA_VISIBLE_DEVICES="$device" {shlex.quote(str(PYTHON))} '
            f'{shlex.quote(str(directory/"worker.py"))} --run-dir {shlex.quote(str(run_dir))} '
            f'--stage {stage} --max-seconds {seconds-120}{extra} &\n'
            '  pids+=("$!")\ndone\n'
            'status=0\nfor pid in "${pids[@]}"; do wait "$pid" || status=1; done\nexit "$status"\n')
    script.write_text('#!/bin/bash\nset -euo pipefail\numask 0002\n'
        'export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n'
        'export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75\n'
        f'export GRID_GPU_CLASS={shlex.quote(gpu_type or "none")}\n'
        f'export PYTHONPATH={shlex.quote(str(Path(source["snapshot"])/"10_jax_exploration_platform/src"))}\n'
        f'cd {shlex.quote(str(PROJECT))}\n'+body)
    # Each subprocess owns one allocated card; processes never share a preallocated pool.
    memory = 16 if monitor or report else MEMORY_GIB_PER_CARD*cards
    cpus = 1 if monitor else 2 if report else CPU_PER_CARD*cards
    command = [str(ACCOUNTS/'submit_as.sh'), identity, '--idfile',
        str(run_dir/'slurm'/f'submitted_jobids_initial_bonus_{account}.txt'),
        '--dir', str(PROJECT), '--', '--job-name', name, '--partition', partition,
        '--nodes', '1', '--ntasks', '1', '--cpus-per-task', str(cpus),
        '--mem', f'{memory}G', '--time', str(seconds),
        '--output', str(run_dir/'logs'/'%x_%j.log'), '--open-mode', 'append']
    # Slurm's bare integer time is minutes, so spell the exact duration as hours:minutes:seconds.
    command[command.index('--time')+1] = f'{seconds//3600}:{seconds%3600//60:02d}:{seconds%60:02d}'
    command += ['--gpus-per-node', '0'] if monitor or report else ['--gres', f'gpu:{gpu_type}:{cards}']
    if node:
        command += ['--nodelist', node]
    if exclude_nodes:
        command += ['--exclude', ','.join(sorted(exclude_nodes))]
    if reservation:
        command += ['--reservation', reservation, '--qos', 'csresnolim']
    command.append(str(script))
    output = subprocess.check_output(command, text=True, stdin=subprocess.DEVNULL).strip()
    job_id = output.splitlines()[-1]
    if not job_id.isdecimal():
        raise ValueError(f'Submission did not return a job identifier: {output}')
    write_json(run_dir/'slurm/submissions'/f'{job_id}.json', {'job_id': job_id, 'account': account,
        'stage': 'monitor' if monitor else stage, 'node': node, 'gpu_type': gpu_type,
        'cards': 0 if monitor or report else cards, 'cpus': cpus, 'memory_gib': memory,
        'final_report': final_report,
        'partition': partition, 'canary': canary, 'source_commit': source.get('commit'),
        'reservation': reservation, 'qos': 'csresnolim' if reservation else None,
        'excluded_nodes': sorted(exclude_nodes),
        'submitted_at': now(), 'script': str(script), 'wall_seconds': seconds})
    return job_id


def submit_report_job(run_dir, final=False):
    """Submit plotting only to an owner CPU pool with actual processor and memory room."""
    # Reports never claim graphics cards or the collaborator account's resources.
    capacity = json.loads(subprocess.check_output(
        [str(PYTHON), str(ACCOUNTS/'capacity.py'), '--no-mit', '--json'], text=True))
    owner = capacity.get('uva-sl5nw', {})
    if not owner.get('reachable'):
        raise RuntimeError('The report owner account is unavailable.')
    pools = {pool['partition']: pool for pool in owner['pools']}
    for partition in ('cpu', 'nolim'):
        pool = pools.get(partition)
        if pool and pool['cpu_free'] >= 18 and pool['memory_gib_free'] >= 16:
            return submit_job(run_dir, 'uva-sl5nw', 'report', partition=partition,
                              report=True, final_report=final)
    return None


def owner_reservation():
    """Rediscover the owner's active jaguar03 reservation."""
    output = subprocess.check_output(['scontrol', 'show', 'reservation', '-o'], text=True)
    for row in output.splitlines():
        fields = slurm_fields(row)
        if (fields.get('State') == 'ACTIVE' and 'sl5nw' in fields.get('Users', '').split(',')
                and 'jaguar03' in fields.get('Nodes', '')):
            return fields['ReservationName']
    return None


def classify_submission_reservations(submissions, live):
    """Recover legacy live reservation identities without rewriting historical records."""
    records = []
    for original in submissions:
        record = dict(original)
        if record['job_id'] in live and 'reservation' not in record:
            fields = slurm_fields(subprocess.check_output(
                ['scontrol', 'show', 'job', record['job_id'], '-o'], text=True))
            reservation = fields.get('Reservation')
            record['reservation'] = None if reservation in (None, '(null)') else reservation
            record['qos'] = fields.get('QOS')
        records.append(record)
    return records


def capacity_budgets(capacity, submissions, live):
    """Reserve pending study requests as well as every account's actual running usage."""
    # Before: account pool free figures; after: independent per-account/per-pool budgets.
    budgets = {}
    for identity in ('uva-sl5nw', 'uva-yuxinchen'):
        account = capacity.get(identity, {})
        if not account.get('reachable'):
            continue
        for pool in account['pools']:
            if pool['partition'] not in ('gpu', 'gnolim'):
                continue
            key = (identity, pool['partition'])
            # The live partition total intentionally includes running reservations.
            # Reserved admission overrides these ceilings separately; it does not add
            # their usage back into the ordinary pool's reported free capacity.
            budgets[key] = {
                'cards': max(0, pool['gpu_free']),
                'cpus': max(0, pool['cpu_free']-(16 if identity == 'uva-sl5nw' else 0)),
                'memory_gib': max(0, pool['memory_gib_free'])}
    for record in submissions:
        status = live.get(record['job_id'])
        key = ('uva-'+record['account'], record['partition'])
        if status and status[3] != 'RUNNING' and key in budgets and not record.get('reservation'):
            for resource, amount in [('cards', record['cards']),
                    ('cpus', record.get('cpus', CPU_PER_CARD*record['cards'])),
                    ('memory_gib', record['memory_gib'])]:
                budgets[key][resource] = max(0, budgets[key][resource]-amount)
    return budgets


def reservation_budget(node, physical, submissions, live, reservation):
    """Admit reserved work from its own physical resources, retaining sixteen owner CPUs."""
    budget = {'cards': physical['cards'], 'cpus': max(0, physical['cpus']-16),
              'memory_gib': physical['memory_gib']}
    for record in submissions:
        status = live.get(record['job_id'])
        if (record.get('reservation') == reservation and record.get('node') in (None, node)
                and status and status[3] != 'RUNNING'):
            for resource, amount in [('cards', record['cards']), ('cpus', record['cpus']),
                                     ('memory_gib', record['memory_gib'])]:
                budget[resource] = max(0, budget[resource]-amount)
    return budget


def node_resources(output):
    """Read physical node availability without treating down or drained nodes as usable."""
    nodes = {}
    for row in output.splitlines():
        record = slurm_fields(row)
        if not record or any(word in record.get('State', '') for word in ('DOWN', 'DRAIN', 'FAIL', 'INVAL', 'MAINT')):
            continue
        gpu = re.search(r'gpu:(?:[^,:()]+:)?(\d+)', record.get('Gres', ''))
        if not gpu:
            continue
        allocated = re.search(r'gres/gpu=(\d+)', record.get('AllocTRES', ''))
        total = int(gpu.group(1))
        nodes[record['NodeName']] = {'cards': max(0, total-int(allocated.group(1) if allocated else 0)),
            'total_cards': total, 'cpus': max(0, int(record.get('CPUEfctv', record['CPUTot']))-int(record['CPUAlloc'])),
            'memory_gib': max(0, int(record['RealMemory'])-int(record['AllocMem'])
                             -int(record.get('MemSpecLimit', 0)))/1024,
            'reservation': record.get('ReservationName')}
    return nodes


def compatibility_classes(run_dir):
    """Admit a GPU class only after every case of its real short measurement passed."""
    ready, failed = set(), set()
    for path in (run_dir/'infrastructure/compatibility').glob('*.json'):
        record = read_json(path)
        gpu_type = record.get('gpu_type')
        if not gpu_type:
            continue
        cases = record.get('cases', [])
        if record.get('completed') and len(cases) == 4 and all(
                case.get('completed') and case.get('correctness', {}).get('passed') for case in cases):
            ready.add(gpu_type)
        else:
            failed.add(gpu_type)
    return ready, failed-ready


def top_up(run_dir, stage):
    """Fill both eligible GPU partitions under measured live resource and queue ceilings."""
    run_dir = Path(run_dir)
    if (run_dir/'STOP').exists() or (run_dir/'SWEEP_COMPLETE').exists():
        return []
    live = own_live_jobs(run_dir)
    submissions = [read_json(path) for path in (run_dir/'slurm/submissions').glob('*.json')]
    submissions = classify_submission_reservations(submissions, live)
    active = [record for record in submissions if record['job_id'] in live]
    tasks = [read_json(path) for state in ('pending', 'running')
             for path in (run_dir/'queue'/state).glob('*.json')]
    outstanding = sum(task['stage'] == stage for task in tasks)
    remaining = outstanding-sum(record['cards'] for record in active if record['stage'] == stage)
    if remaining <= 0:
        return []
    # Account figures include unrelated sessions; pending requests reserve future room too.
    capacity = json.loads(subprocess.check_output(
        [str(PYTHON), str(ACCOUNTS/'capacity.py'), '--no-mit', '--json'], text=True))
    write_json(run_dir/'slurm/capacity_latest.json', capacity)
    budgets = capacity_budgets(capacity, submissions, live)
    nodes = node_resources(subprocess.check_output(['scontrol', 'show', 'nodes', '-o'], text=True))
    classes = read_json(RESOURCE_FILE)['classes']
    held_nodes = {record['node'] for record in active if record.get('node')}
    held_nodes.update(row[5] for row in live.values() if row[3] == 'RUNNING'
                      and row[5] in nodes)
    ready, failed = compatibility_classes(run_dir)
    type_counts = {}
    for record in classes:
        type_counts[record['gres_type']] = type_counts.get(record['gres_type'], 0)+record['gpus_per_node']*len(record['nodes'])
    for record in active:
        if record.get('gpu_type') in type_counts:
            type_counts[record['gpu_type']] -= record['cards']
    reservation = owner_reservation()
    jobs = []
    # Profiling stays on the requested machine; research cards are chosen when jobs start.
    ordered = sorted(classes, key=lambda row: (row['partition'] != 'gpu', row.get('capability_rank', 999)))
    # Open partitions come first: actual running reservation usage still counts
    # against their partition totals, while reserved admission overrides that ceiling.
    for reserved in (False, True):
        for record in ordered:
            if record.get('gpu_mem_mb', 0) < 7000 or record['gres_type'] in failed:
                continue
            for node in record['nodes']:
                if remaining <= 0:
                    return jobs
                if node not in nodes or node in held_nodes or (node == 'jaguar03') != reserved:
                    continue
                if stage in ('checks', 'infrastructure') and node != 'jaguar03':
                    continue
                physical = nodes[node]
                if physical['reservation'] not in (None, '(null)', reservation):
                    continue
                if reserved and not reservation:
                    continue
                identities = ('uva-sl5nw',) if reserved else ('uva-sl5nw', 'uva-yuxinchen')
                identities = sorted(identities, key=lambda identity: -budgets.get((identity, record['partition']), {}).get('cards', 0))
                for identity in identities:
                    budget = (reservation_budget(node, physical, submissions, live, reservation)
                              if reserved else budgets.get((identity, record['partition'])))
                    if not budget:
                        continue
                    available = min(physical['cards'], physical['cpus']//CPU_PER_CARD,
                        int(physical['memory_gib']//MEMORY_GIB_PER_CARD), budget['cards'],
                        budget['cpus']//CPU_PER_CARD, int(budget['memory_gib']//MEMORY_GIB_PER_CARD),
                        remaining, type_counts[record['gres_type']])
                    if available < 1:
                        continue
                    canary = stage == 'research' and record['gres_type'] not in ready
                    cards = 1 if canary or stage == 'checks' else int(available)
                    identifier = submit_job(run_dir, identity, stage, partition=record['partition'],
                        node=node, gpu_type=record['gres_type'], cards=cards,
                        reservation=reservation if reserved else None, canary=canary)
                    jobs.append(identifier)
                    held_nodes.add(node)
                    remaining -= cards
                    type_counts[record['gres_type']] -= cards
                    budget['cards'] -= cards
                    budget['cpus'] -= CPU_PER_CARD*cards
                    budget['memory_gib'] -= MEMORY_GIB_PER_CARD*cards
                    if len(jobs) % 10 == 0:
                        time.sleep(30)
                    break
    # Let large-card requests age while cards are occupied, without pinning a waiting job.
    # One compatibility request per type precedes full-node requests of that measured type.
    if stage == 'research' and remaining > 0:
        queued_types = set()
        for record in ordered:
            gpu_type = record['gres_type']
            if (gpu_type in queued_types or gpu_type in failed
                    or record.get('capability_rank', 999) > 2):
                continue
            queued_types.add(gpu_type)
            canary = gpu_type not in ready
            if canary and any(entry.get('gpu_type') == gpu_type for entry in active):
                continue
            if canary and any(entry.get('gpu_type') == gpu_type for entry in (
                    read_json(path) for path in (run_dir/'slurm/submissions').glob('*.json'))
                    if entry['job_id'] in jobs):
                continue
            cards = 1 if canary else record['gpus_per_node']
            for identity in ('uva-sl5nw', 'uva-yuxinchen'):
                budget = budgets.get((identity, record['partition']))
                if not budget:
                    continue
                submitted_here = False
                while (remaining >= cards and type_counts[gpu_type] >= cards
                        and budget['cards'] >= cards and budget['cpus'] >= CPU_PER_CARD*cards
                        and budget['memory_gib'] >= MEMORY_GIB_PER_CARD*cards):
                    identifier = submit_job(run_dir, identity, stage,
                        partition=record['partition'], gpu_type=gpu_type, cards=cards,
                        canary=canary, exclude_nodes=held_nodes)
                    jobs.append(identifier)
                    submitted_here = True
                    remaining -= cards
                    type_counts[gpu_type] -= cards
                    budget['cards'] -= cards
                    budget['cpus'] -= CPU_PER_CARD*cards
                    budget['memory_gib'] -= MEMORY_GIB_PER_CARD*cards
                    if len(jobs) % 10 == 0:
                        time.sleep(30)
                    if canary:
                        break
                if canary and submitted_here:
                    break
    return jobs


def main():
    """Submit the requested training phase or its persistent long-partition monitor."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['checks', 'infrastructure', 'research'])
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()
    if args.monitor:
        print(submit_job(args.run_dir, 'uva-sl5nw', 'monitor', partition='nolim', monitor=True))
    else:
        print(top_up(args.run_dir, args.stage))


if __name__ == '__main__':
    main()
