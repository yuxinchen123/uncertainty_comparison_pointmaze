"""Submit measured GPU job shapes through the approved account ledgers."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess

from common import ACCOUNTS, PROJECT, PYTHON, now, own_live_jobs, read_json, write_json

HERE = Path(__file__).resolve().parent
RESOURCE_FILE = Path('/p/rlprojects/.claude/skills/uva-submit-gpu-sweep/school_compute_resource/server_introduction.json')


def submit_job(run_dir, identity, stage, partition='gpu', node=None, gpu_type=None,
               cards=1, reservation=None, monitor=False):
    """Submit one exact script and let the shared wrapper immediately record its id."""
    account = identity.removeprefix('uva-')
    scripts = run_dir / 'slurm/scripts'
    scripts.mkdir(parents=True, exist_ok=True)
    number = len(list(scripts.glob('*.sh')))
    name = f'grid-initial-{"monitor" if monitor else stage}-{number}'
    script = scripts / (name + '.sh')
    source = read_json(run_dir/'source.json')
    directory = Path(source['snapshot'])/'10_jax_exploration_platform/scripts/grid_initialization_campaign'
    if monitor:
        body = f'exec {shlex.quote(str(PYTHON))} {shlex.quote(str(directory/"monitor.py"))} --run-dir {shlex.quote(str(run_dir))} --loop\n'
    else:
        body = ('IFS=, read -ra devices <<< "$CUDA_VISIBLE_DEVICES"\n'
            'for device in "${devices[@]}"; do\n'
            f'  CUDA_VISIBLE_DEVICES="$device" {shlex.quote(str(PYTHON))} '
            f'{shlex.quote(str(directory/"worker.py"))} --run-dir {shlex.quote(str(run_dir))} '
            f'--stage {stage} --max-seconds 14000 &\n'
            'done\nwait\n')
    script.write_text('#!/bin/bash\nset -euo pipefail\numask 0002\n'
        'export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n'
        'export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75\n'
        f'export PYTHONPATH={shlex.quote(str(Path(source["snapshot"])/"10_jax_exploration_platform/src"))}\n'
        f'cd {shlex.quote(str(PROJECT))}\n'+body)
    command = [str(ACCOUNTS/'submit_as.sh'), identity, '--idfile',
               str(run_dir/'slurm'/f'submitted_jobids_initial_bonus_{account}.txt'),
               '--dir', str(PROJECT), '--', '--job-name', name, '--partition', partition,
               '--nodes', '1', '--ntasks', '1', '--cpus-per-task', str(1 if monitor else 4*cards),
               '--mem', f'{4 if monitor else 16*cards}G', '--time', '4-00:00:00' if monitor else '04:00:00',
               '--output', str(run_dir/'logs'/'%x_%j.log'), '--open-mode', 'append']
    if monitor:
        command += ['--gpus-per-node', '0']
    else:
        command += ['--gres', f'gpu:{gpu_type}:{cards}']
    if node:
        command += ['--nodelist', node]
    if reservation:
        command += ['--reservation', reservation]
    command.append(str(script))
    output = subprocess.check_output(command, text=True, stdin=subprocess.DEVNULL).strip()
    job_id = output.splitlines()[-1]
    if not job_id.isdecimal():
        raise ValueError(f'Submission did not return a job identifier: {output}')
    write_json(run_dir/'slurm/submissions'/f'{job_id}.json', {'job_id': job_id, 'account': account,
        'stage': 'monitor' if monitor else stage, 'node': node, 'gpu_type': gpu_type, 'cards': cards,
        'partition': partition, 'submitted_at': now(), 'script': str(script)})
    return job_id


def owner_reservation():
    """Rediscover the owner's active jaguar03 reservation before each submission."""
    output = subprocess.check_output(['scontrol', 'show', 'reservation', '-o'], text=True)
    for row in output.splitlines():
        fields = dict(part.split('=', 1) for part in row.split() if '=' in part)
        if fields.get('State') == 'ACTIVE' and 'sl5nw' in fields.get('Users', '').split(',') and 'jaguar03' in fields.get('Nodes', ''):
            return fields['ReservationName']
    return None


def top_up(run_dir, stage):
    """Submit generic jobs to both accounts without assigning work in advance."""
    live = own_live_jobs(run_dir)
    submissions = [read_json(path) for path in (run_dir/'slurm/submissions').glob('*.json')]
    active = [record for record in submissions if record['job_id'] in live and record['stage'] == stage]
    pending = sum(read_json(path)['stage'] == stage for path in (run_dir/'queue/pending').glob('*.json'))
    if pending <= len(active):
        return []
    if stage in ('checks', 'infrastructure'):
        reservation = owner_reservation()
        if not reservation:
            raise RuntimeError('Jaguar03 profiling requires an active accessible reservation.')
        if active:
            return []
        cards = 1 if stage == 'checks' else min(8, pending)
        return [submit_job(run_dir, 'uva-sl5nw', stage, node='jaguar03',
                gpu_type='nvidia_rtx_a4500', cards=cards, reservation=reservation)]
    # Capacity is read from every job each account holds, including unrelated sessions.
    capacity = subprocess.check_output([str(PYTHON), str(ACCOUNTS/'capacity.py'), '--no-mit', '--json'], text=True)
    write_json(run_dir/'slurm/capacity_latest.json', __import__('json').loads(capacity))
    node_output = subprocess.check_output(['sinfo', '-h', '-p', 'gpu', '-N', '-o', '%N|%t'], text=True)
    available_nodes = {row.split('|')[0] for row in node_output.splitlines()
                       if row.split('|')[1].split('+')[0] in ('idle', 'mix', 'alloc', 'resv')}
    classes = read_json(RESOURCE_FILE)['classes']
    classes = sorted(classes, key=lambda record: record.get('capability_rank', 999))
    held_nodes = {record['node'] for record in submissions if record['job_id'] in live and record.get('node')}
    jobs = []
    # Node-local card sets give one job per node; all work remains dynamically claimed.
    for record in classes:
        if record.get('compute_capability', 0) < 7.5:
            continue
        for node in record['nodes']:
            if node not in available_nodes or node in held_nodes or pending <= len(active)+len(jobs):
                continue
            if node == 'jaguar03':
                reservation = owner_reservation()
                if not reservation:
                    continue
                identity = 'uva-sl5nw'
            else:
                reservation = None
                identity = ('uva-sl5nw', 'uva-yuxinchen')[len(jobs) % 2]
            # Start with one card per node until that class has recorded real compatibility.
            jobs.append(submit_job(run_dir, identity, stage, node=node,
                gpu_type=record['gres_type'], cards=1, reservation=reservation))
            held_nodes.add(node)
            if len(jobs) % 10 == 0:
                import time
                time.sleep(30)
    return jobs


def main():
    """Submit the requested phase or the required persistent monitor."""
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
