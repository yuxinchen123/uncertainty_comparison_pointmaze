"""Check the remote fleet tool on fabricated records and a pretend server made of local folders.

No test imports jax or opens an ssh connection: the "server" is a folder, reached through a tiny script that takes
the place of ssh, with pretend sbatch, squeue, scancel, nvidia-smi and python programs on PATH.
"""

from argparse import Namespace
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pytest

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[2] / 'scripts/grid_initialization_campaign'
sys.path.insert(0, str(SCRIPT_DIRECTORY))
import remote_fleet as fleet  # noqa: E402

PRETEND_PYTHON = '''#!{python}
"""Stand in for the server's python: the card check, the first-unit check and train_unit.py."""
import json, os, sys
from pathlib import Path
arguments = sys.argv[1:]
if arguments[0] == '-c':
    print('jax devices: [pretend card]')
    sys.exit(1 if os.environ.get('PRETEND_NO_CARD') else 0)
script = Path(arguments[0]).name
if script == 'check_online_path_unchanged.py':
    Path(arguments[arguments.index('--dump') + 1]).write_text('dump')
elif script == 'remote_first_unit_compare.py':
    passed = not os.environ.get('PRETEND_CHECK_FAILS')
    Path(arguments[arguments.index('--output') + 1]).write_text(
        json.dumps({{'passed': passed, 'largest_relative_difference': 0.0 if passed else 0.5}}, indent=2))
    sys.exit(0 if passed else 1)
elif script == 'train_unit.py':
    run = Path(arguments[arguments.index('--run-dir') + 1])
    specification = json.loads(Path(arguments[arguments.index('--specification') + 1]).read_text())
    assert os.environ['GRID_GPU_CLASS'] == 'pretend_class' and 'LD_LIBRARY_PATH' not in os.environ
    if 'bad' in specification['name']:
        print('RuntimeError: pretend failure')
        sys.exit(1)
    folder = run / 'data/research' / specification['name']
    folder.mkdir(parents=True, exist_ok=True)
    copies = len(specification['settings']) * specification['seed_count']
    (folder / 'behavior.npz').write_text('arrays')
    (folder / 'summary.json').write_text(json.dumps(
        {{'completed': True, 'copy_records': [{{}}] * copies, 'wall_seconds': 12.5}}))
'''
PRETEND_PROGRAMS = {
    'pretend_ssh': '#!/bin/bash\n# Take the place of ssh: drop the host name and run the command here.\nshift\nexec "$@"\n',
    'nvidia-smi': '#!/bin/bash\necho "NVIDIA H100 80GB HBM3, 81559"\n',
    'sbatch': ('#!/bin/bash\n# Give each submission the next id and keep its arguments.\n'
               '[ -e "$PRETEND_STATE/sbatch_refuses" ] && { echo "sbatch: error: 14 CPUs per GPU" >&2; exit 1; }\n'
               'n=$(( $(cat "$PRETEND_STATE/next_id" 2> /dev/null || echo 500) + 1 ))\necho "$n" > "$PRETEND_STATE/next_id"\n'
               'echo "$@" >> "$PRETEND_STATE/sbatch_calls"\necho "$n;pretend"\n'),
    'squeue': ('#!/bin/bash\n# List the pretend jobs: every one for -u, one state for -j.\n'
               '[ -e "$PRETEND_STATE/squeue_fails" ] && exit 1\n'
               'f="$PRETEND_STATE/jobs"; [ -e "$f" ] || exit 0\n'
               'if [ "$2" = "-j" ]; then grep "^$3|" "$f" | cut -d"|" -f2; else cat "$f"; fi\n'),
    'scancel': '#!/bin/bash\necho "$1" >> "$PRETEND_STATE/cancelled"\n',
    'sacct': '#!/bin/bash\ncat "$PRETEND_STATE/cards" 2> /dev/null\n',
}


def unit_record(name, settings=2, seeds=3):
    """Build one unit's queue file and specification."""
    task = {'name': name, 'stage': 'research', 'command': ['train_unit.py']}
    return task, {'name': name, 'settings': [{}] * settings, 'seed_count': seeds}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Make a UVA run folder with four pending units, and a pretend server with one pool."""
    run, server, programs, state = (tmp_path / part for part in ('uva_run', 'server_run', 'bin', 'state'))
    for folder in ('queue/pending', 'queue/deferred', 'queue/done', 'queue/failed', 'specifications', 'slurm',
                   'logs', 'data/research', 'infrastructure/remote_fleet'):
        (run / folder).mkdir(parents=True)
    programs.mkdir()
    state.mkdir()
    for name, text in {**PRETEND_PROGRAMS, 'python': PRETEND_PYTHON.format(python=sys.executable)}.items():
        (programs / name).write_text(text)
        (programs / name).chmod(0o755)
    monkeypatch.setenv('PATH', f'{programs}:{os.environ["PATH"]}')
    monkeypatch.setenv('PRETEND_STATE', str(state))
    monkeypatch.delenv('SLURM_JOB_END_TIME', raising=False)
    for name in ('unit_a', 'unit_b', 'unit_c', 'unit_d'):
        task, specification = unit_record(name)
        (run / 'queue/pending' / f'{name}.json').write_text(json.dumps(task))
        (run / 'specifications' / f'{name}.json').write_text(json.dumps(specification))
    (run / 'infrastructure/remote_fleet/first_unit_reference_uva.npz').write_text('reference')
    table = {'pretend': {'ssh': [str(programs / 'pretend_ssh'), 'pretendhost'], 'check': [],
                         'remote_run': str(server), 'python': str(programs / 'python'), 'code': str(tmp_path / 'code'),
                         'unit_seconds': 100, 'budget_command': 'echo "|acct|2027|Y|100.00|40.00|60.00|PI|"',
                         'budget_field': 6, 'budget_unit': 'SUs', 'ready': True, 'failed_calls_in_a_row': 0,
                         'pools': {'main': {'gpu_class': 'pretend_class', 'time_limit': '01:00:00', 'max_jobs': 2,
                                            'sbatch_options': ['-p', 'gpu', '--gres=gpu:1']}}}}
    fleet.write_json(fleet.table_path(run), table)
    return Namespace(run=run, server=server, state=state, table=table, programs=programs)


def stage_arguments(**changes):
    """Build the arguments of a stage call."""
    return Namespace(**{'server': 'pretend', 'units': 2, 'from_end': True, 'include_deferred': False, 'match': None,
                        'dry_run': False, **changes})


def run_worker(world, job, **environment):
    """Run the pool's job script here as the pretend job with this id."""
    return subprocess.run(['bash', str(world.server / 'worker_job_main.sh')], capture_output=True, text=True,
                          env={**os.environ, 'SLURM_JOB_ID': str(job), 'LD_LIBRARY_PATH': '/module/cuda', **environment})


def folders(world):
    """Return where every unit's queue file is, on UVA and on the server."""
    uva = {path.stem: path.parent.name for path in (world.run / 'queue').glob('*/*.json')}
    server = {path.stem: path.parent.name for path in (world.server / 'queue').glob('*/*.json')}
    return uva, server


def test_choose_units_takes_the_end_of_the_sorted_names():
    """Units come from the end of the names of both folders, each with the folder it leaves."""
    assert fleet.choose_units(['a', 'c'], ['b', 'd'], 2, True, None) == [('c', 'pending'), ('d', 'deferred')]
    assert fleet.choose_units(['a', 'c'], ['b', 'd'], 3, False, None) == [('a', 'pending'), ('b', 'deferred'), ('c', 'pending')]
    assert fleet.choose_units(['tabular_1', 'neural_1', 'tabular_2'], [], 5, True, 'tabular') == [
        ('tabular_1', 'pending'), ('tabular_2', 'pending')]
    assert fleet.choose_units(['a'], ['b'], 0, True, None) == []


def test_allowed_jobs_respects_cap_and_unclaimed_work():
    """A submission never exceeds the pool's room or the units no waiting job will take."""
    assert fleet.allowed_jobs(8, 12, 2, 5, 1) == 4
    assert fleet.allowed_jobs(8, 2, 1, 16, 0) == 1
    assert fleet.allowed_jobs(3, 4, 4, 9, 0) == 0
    assert fleet.allowed_jobs(3, 4, 0, 1, 2) == 0


def test_small_readers():
    """Time limits, pool names, packing and budgets are read as documented."""
    assert fleet.limit_seconds('12:00:00') == 43200 and fleet.limit_seconds('2-00:00:00') == 172800
    assert fleet.pool_of('mit', 'rndfleet-mit-preemptable-20261109-210501-1') == 'preemptable'
    assert fleet.pool_of('mit', 'cmpt-b-s10') is None
    record = fleet.packing_record()
    assert all(record[key] == {'copies': 512} for key in fleet.PACKING_KEYS) and 'serval05' in record['source']
    aces = {'budget_command': 'myproject', 'budget_field': 6, 'budget_unit': 'SUs'}
    line = '|158590754315|  2027|        Y|   145000.00|          55667.82|    89332.18|Liu, Shuze Daniel   |'
    assert fleet.budget_text(aces, [line]) == '89332.18 SUs'
    assert fleet.budget_text(aces, []) == 'no answer'
    assert fleet.budget_text({'budget_note': 'standby is not charged'}, []) == 'standby is not charged'


def test_card_mismatches_reads_requested_and_allocated_counts():
    """A running fleet job holding a different number of cards than it asked for is reported."""
    lines = ['7|rndfleet-mit-advanced-20261109-210501-1|billing=4,cpu=4,gres/gpu=1,mem=16G|cpu=4,gres/gpu=2,mem=16G',
             '8|rndfleet-mit-advanced-20261109-210501-2|cpu=4,gres/gpu=1,gres/gpu:h200=1|cpu=4,gres/gpu=1',
             '9|another-study|cpu=4,gres/gpu=1|cpu=4,gres/gpu=4']
    assert fleet.card_mismatches('mit', lines) == [('7', '1', '2')]


def test_parse_status_refuses_a_cut_off_answer():
    """An answer without its end marker is never read as an empty queue."""
    with pytest.raises(fleet.RemoteCommandFailed):
        fleet.parse_status('@@fleet:queue\npending unit_a\n')
    assert fleet.parse_status('@@fleet:norun\n@@fleet:end\n')['prepared'] is False


def status_of(queue, claims=None, jobs=None, starts=(), check=None, squeue_ok=True):
    """Build a parsed status with the given queue folders and jobs."""
    return {'prepared': True, 'queue': {state: queue.get(state, []) for state in fleet.REMOTE_STATES},
            'claims': claims or {}, 'jobs': jobs or {}, 'squeue_ok': squeue_ok, 'starts': list(starts), 'ledger': [],
            'check': check, 'budget': [], 'cards': [], 'tails': {}}


def job(state, name='rndfleet-s-main-20261109-210501-1'):
    """Build one job row."""
    return {'state': state, 'reason': 'None', 'name': name, 'elapsed': '1:00', 'partition': 'gpu'}


def test_plan_tick_sorts_units_by_what_happened():
    """Done units are pulled, failed ones reported, orphaned ones recovered, and missing ones named."""
    moment = time.time()
    assignment = {unit: {'staged_at': fleet.now(), 'recoveries': count} for unit, count in
                  (('done_1', 0), ('failed_1', 0), ('dead_1', 0), ('dead_2', 2), ('waiting_1', 0), ('alive_1', 0),
                   ('fresh_1', 0), ('lost_1', 0), ('incoming_1', 0))}
    status = status_of({'done': ['done_1', 'old_done'], 'failed': ['failed_1'], 'incoming': ['incoming_1'],
                        'running': ['dead_1', 'dead_2', 'waiting_1', 'alive_1', 'fresh_1']},
                       claims={('running', 'dead_1'): {'job': '10'}, ('running', 'dead_2'): {'job': '11'},
                               ('running', 'waiting_1'): {'job': '12'}, ('running', 'alive_1'): {'job': '13'}},
                       jobs={'12': job('PENDING'), '13': job('RUNNING')})
    plan = fleet.plan_tick('s', sorted(assignment), status, assignment, {'12', '13'}, moment)
    assert plan['pull'] == ['done_1'] and plan['failed'] == ['failed_1'] and plan['place'] == ['incoming_1']
    # A dead job and a job waiting again after a preemption both orphan their unit; two recoveries are the limit.
    assert plan['recover'] == ['dead_1', 'waiting_1'] and plan['pull_back'] == ['dead_2']
    assert plan['missing'] == ['lost_1'] and plan['withdraw'] == []
    # Without a job listing no unit may be called orphaned.
    blind = fleet.plan_tick('s', sorted(assignment), {**status, 'squeue_ok': False}, assignment, set(), moment)
    assert blind['recover'] == [] and blind['pull_back'] == [] and blind['pull'] == ['done_1']


def test_plan_tick_withdraws_only_long_waits_without_any_start():
    """Unclaimed units go back after two hours only when no job runs and none started since they were staged."""
    moment = time.time()
    old = fleet.datetime.fromtimestamp(moment - 3 * 3600).astimezone().isoformat()
    assignment = {'old_1': {'staged_at': old, 'recoveries': 0}, 'new_1': {'staged_at': fleet.now(), 'recoveries': 0}}
    queue = {'pending': ['old_1', 'new_1']}
    assert fleet.plan_tick('s', ['new_1', 'old_1'], status_of(queue), assignment, set(), moment)['withdraw'] == ['old_1']
    running = status_of(queue, jobs={'5': job('RUNNING')})
    assert fleet.plan_tick('s', ['new_1', 'old_1'], running, assignment, {'5'}, moment)['withdraw'] == []
    waiting = status_of(queue, jobs={'5': job('PENDING')})
    assert fleet.plan_tick('s', ['new_1', 'old_1'], waiting, assignment, {'5'}, moment)['withdraw'] == ['old_1']
    started = status_of(queue, starts=[('4', moment - 600)])
    assert fleet.plan_tick('s', ['new_1', 'old_1'], started, assignment, set(), moment)['withdraw'] == []
    # A failed first-unit check sends every unclaimed unit back at once.
    failed = status_of(queue, check={'passed': False})
    assert fleet.plan_tick('s', ['new_1', 'old_1'], failed, assignment, set(), moment)['withdraw'] == ['new_1', 'old_1']


def test_verify_unit_requires_complete_records(tmp_path):
    """A unit counts as verified only with a completed summary of the right size and its arrays."""
    _, specification = unit_record('unit_a')
    (tmp_path / 'specifications').mkdir()
    (tmp_path / 'specifications/unit_a.json').write_text(json.dumps(specification))
    folder = tmp_path / 'data/research/unit_a'
    folder.mkdir(parents=True)
    assert fleet.verify_unit(tmp_path, 'unit_a') == 'summary.json is missing'
    (folder / 'summary.json').write_text(json.dumps({'completed': True, 'copy_records': [{}] * 5}))
    assert fleet.verify_unit(tmp_path, 'unit_a') == '5 copy records, expected 6'
    (folder / 'summary.json').write_text(json.dumps({'completed': False, 'copy_records': [{}] * 6}))
    assert fleet.verify_unit(tmp_path, 'unit_a') == 'summary.json does not say completed'
    (folder / 'summary.json').write_text(json.dumps({'completed': True, 'copy_records': [{}] * 6}))
    assert fleet.verify_unit(tmp_path, 'unit_a') == 'behavior.npz is missing or empty'
    (folder / 'behavior.npz').write_text('arrays')
    assert fleet.verify_unit(tmp_path, 'unit_a') is None


def test_move_unit_keeps_a_unit_in_one_folder(tmp_path):
    """A move adds its fields, leaves one file, and refuses a folder that already holds the unit."""
    for folder in ('remote_s', 'done'):
        (tmp_path / 'queue' / folder).mkdir(parents=True)
    (tmp_path / 'queue/remote_s/unit_a.json').write_text(json.dumps({'name': 'unit_a'}))
    fleet.move_unit(tmp_path, 'unit_a', 'remote_s', 'done', {'result': {'server': 's'}})
    assert [path.parent.name for path in (tmp_path / 'queue').glob('*/unit_a.json')] == ['done']
    assert json.loads((tmp_path / 'queue/done/unit_a.json').read_text()) == {'name': 'unit_a', 'result': {'server': 's'}}
    (tmp_path / 'queue/remote_s/unit_a.json').write_text('{}')
    with pytest.raises(RuntimeError):
        fleet.move_unit(tmp_path, 'unit_a', 'remote_s', 'done')


def test_result_block_reads_the_claim():
    """The result names the server and job and carries the unit's exit code and time."""
    block = fleet.result_block('aces', {'job': '12', 'host': 'n1', 'gpu': 'NVIDIA_H100,_81559', 'exit': '0',
                                        'wall_seconds': '4000'}, {'wall_seconds': 3990.5})
    assert (block['server'], block['job_id'], block['exit_code'], block['wall_seconds']) == ('aces', '12', 0, 4000.0)
    assert block['training_wall_seconds'] == 3990.5 and block['pulled_at']
    assert fleet.result_block('aces', {})['exit_code'] is None


def test_worker_script_text(world):
    """The job script is valid bash, sets the card class, clears module libraries and claims by moving files."""
    entry = world.table['pretend']
    text = fleet.worker_script('pretend', entry, 'main', entry['pools']['main'])
    assert subprocess.run(['bash', '-n'], input=text, text=True).returncode == 0
    for expected in ('unset LD_LIBRARY_PATH', 'export GRID_GPU_CLASS=pretend_class', 'UNIT_SECONDS=100',
                     'LIMIT_SECONDS=3600', 'mv "$file" "queue/running/$name.json"', 'XLA_PYTHON_CLIENT_PREALLOCATE=false',
                     'JAX_PLATFORMS=cpu', '(no card guard on this server)'):
        assert expected in text
    assert re.search(r'@[A-Z_]+@', text) is None


def test_worker_script_card_guard_counts_cards_with_jax(world):
    """With the card guard the script holds the shared guard, counting cards by jax and requeueing by default."""
    if not fleet.CARD_GUARD.exists():
        pytest.skip('the shared card guard file is not on this machine')
    entry = world.table['pretend']
    text = fleet.worker_script('pretend', entry, 'main', {**entry['pools']['main'], 'card_guard': True})
    assert subprocess.run(['bash', '-n'], input=text, text=True).returncode == 0
    assert 'NEED_CARDS=1' in text and 'GUARD_ACTION=${GUARD_ACTION:-requeue}' in text
    assert fleet.CARD_GUARD_JAX in text and 'import torch' not in text
    assert text.index('[card guard]') < text.index('slurm/job_starts.txt')


def test_stage_dry_run_changes_nothing(world):
    """A dry run names the units and moves no file, on UVA or on the server."""
    before = folders(world)
    fleet.stage(world.run, world.table, stage_arguments(dry_run=True))
    assert folders(world) == before and not world.server.exists() and not (world.run / 'slurm/fleet_log.txt').exists()


def test_units_travel_to_the_server_and_come_back_verified(world):
    """Stage, submit, run and tick: a unit is done on UVA only after its records arrived and were verified."""
    fleet.stage(world.run, world.table, stage_arguments())
    uva, server = folders(world)
    assert uva == {'unit_a': 'pending', 'unit_b': 'pending', 'unit_c': 'remote_pretend', 'unit_d': 'remote_pretend'}
    assert server == {'unit_c': 'pending', 'unit_d': 'pending'}
    assert (world.server / 'specifications/unit_d.json').exists()
    assert json.loads((world.server / 'infrastructure/packing/pretend_class.json').read_text())['neural_count'] == {'copies': 512}
    assignment = fleet.load_assignment(world.run)
    assert assignment['unit_c']['server'] == 'pretend' and assignment['unit_c']['origin'] == 'pending'
    # Five jobs are asked for; the pool's cap is two.
    fleet.submit(world.run, world.table, Namespace(server='pretend', jobs=5, pool=None, dry_run=True))
    assert not (world.state / 'sbatch_calls').exists()
    fleet.submit(world.run, world.table, Namespace(server='pretend', jobs=5, pool=None, dry_run=False))
    assert fleet.ledger_ids(world.run, 'pretend') == ['501', '502']
    assert (world.server / 'slurm/submitted_jobids_pretend.txt').read_text().split() == ['501', '502']
    calls = (world.state / 'sbatch_calls').read_text().splitlines()
    assert len(calls) == 2 and '-t 01:00:00 -p gpu --gres=gpu:1 -J rndfleet-pretend-main-' in calls[0]
    # The job name stands before the script (sbatch would hand a later one to the script), and differs per job.
    assert calls[0].endswith('/worker_job_main.sh') and calls[0].split('-J ')[1] != calls[1].split('-J ')[1]
    # The pretend job runs both units; the first-unit check runs once and passes.
    worker = run_worker(world, 501)
    assert worker.returncode == 0, worker.stdout + worker.stderr
    assert json.loads((world.server / 'first_unit_check.json').read_text())['passed'] is True
    assert folders(world)[1] == {'unit_c': 'done', 'unit_d': 'done'}
    assert 'job=501' in (world.server / 'queue/done/unit_c.job').read_text()
    assert folders(world)[0]['unit_c'] == 'remote_pretend'
    fleet.tick(world.run, world.table, Namespace(dry_run=True))
    assert folders(world)[0]['unit_c'] == 'remote_pretend'
    fleet.tick(world.run, world.table, Namespace(dry_run=False))
    uva, server = folders(world)
    assert uva['unit_c'] == 'done' and uva['unit_d'] == 'done' and server['unit_c'] == 'done'
    done = json.loads((world.run / 'queue/done/unit_c.json').read_text())
    assert done['result']['server'] == 'pretend' and done['result']['job_id'] == '501' and done['result']['exit_code'] == 0
    assert done['result']['training_wall_seconds'] == 12.5 and done['name'] == 'unit_c'
    assert (world.run / 'data/research/unit_c/behavior.npz').read_text() == 'arrays'
    assert (world.server / 'data/research/unit_c/summary.json').exists()
    assert fleet.load_assignment(world.run)['unit_c']['state'] == 'done'
    log = (world.run / 'slurm/fleet_log.txt').read_text()
    assert 'balance: 60.00 SUs' in log and 'unit_c verified and moved to queue/done' in log and ' PT | ' in log


def test_unverified_records_leave_the_unit_staged(world):
    """A unit the server calls done stays staged when its records on UVA are incomplete."""
    fleet.stage(world.run, world.table, stage_arguments(units=1))
    assert run_worker(world, 501).returncode == 0
    summary = world.server / 'data/research/unit_d/summary.json'
    summary.write_text(json.dumps({'completed': True, 'copy_records': [{}]}))
    fleet.tick(world.run, world.table, Namespace(dry_run=False))
    assert folders(world)[0]['unit_d'] == 'remote_pretend'
    assert 'NOT verified on UVA (1 copy records, expected 6)' in (world.run / 'slurm/fleet_log.txt').read_text()


def test_failed_unit_stops_its_job_and_reaches_the_failed_folder(world):
    """After a failed unit the job takes no more; the tick keeps the log's end and marks the unit failed."""
    for name in ('bad_unit', 'good_unit'):
        task, specification = unit_record(name)
        (world.run / 'queue/pending' / f'{name}.json').write_text(json.dumps(task))
        (world.run / 'specifications' / f'{name}.json').write_text(json.dumps(specification))
    fleet.stage(world.run, world.table, stage_arguments(match='_unit'))
    worker = run_worker(world, 501)
    assert worker.returncode == 1 and folders(world)[1] == {'bad_unit': 'failed', 'good_unit': 'pending'}
    fleet.tick(world.run, world.table, Namespace(dry_run=False))
    uva = folders(world)[0]
    assert uva['bad_unit'] == 'failed' and uva['good_unit'] == 'remote_pretend'
    assert 'pretend failure' in (world.run / 'logs/remote_pretend_bad_unit.log').read_text()
    assert json.loads((world.run / 'queue/failed/bad_unit.json').read_text())['result']['exit_code'] == 1


def test_job_guards_leave_units_unclaimed(world):
    """A job with too little time, no card backend or a failed first-unit check takes no unit."""
    fleet.stage(world.run, world.table, stage_arguments())
    pending = {'unit_c': 'pending', 'unit_d': 'pending'}
    assert run_worker(world, 501, PRETEND_NO_CARD='1').returncode == 3 and folders(world)[1] == pending
    assert run_worker(world, 502, PRETEND_CHECK_FAILS='1').returncode == 4 and folders(world)[1] == pending
    assert json.loads((world.server / 'first_unit_check.json').read_text())['passed'] is False
    # The tick sends the units of a server that failed its check back to UVA and stops using the server.
    fleet.tick(world.run, world.table, Namespace(dry_run=False))
    uva, server = folders(world)
    assert uva['unit_c'] == 'pending' and uva['unit_d'] == 'pending' and server == {'unit_c': 'withdrawn', 'unit_d': 'withdrawn'}
    assert fleet.read_json(fleet.table_path(world.run))['pretend']['ready'] is False
    # A job that finds nothing pending ends at once. With a unit pending again, the next job repeats the check,
    # passes it, and close to its time limit still takes nothing.
    assert run_worker(world, 503).returncode == 0
    (world.server / 'queue/withdrawn/unit_c.json').rename(world.server / 'queue/pending/unit_c.json')
    short = run_worker(world, 504, SLURM_JOB_END_TIME=str(int(time.time()) + 50))
    assert short.returncode == 0 and 'is below 100 s' in short.stdout
    assert (world.server / 'queue/pending/unit_c.json').exists()
    assert len((world.server / 'slurm/job_starts.txt').read_text().splitlines()) == 4


def test_orphaned_unit_is_recovered_twice_then_returns_to_uva(world):
    """A unit whose job died goes back to the server's queue twice; the third time it returns to UVA with its records."""
    fleet.stage(world.run, world.table, stage_arguments(units=1))
    for attempt in (1, 2, 3):
        (world.server / 'queue/pending/unit_d.json').rename(world.server / 'queue/running/unit_d.json')
        (world.server / 'queue/running/unit_d.job').write_text(f'job=90{attempt} host=n1 gpu=card started=now\n')
        (world.server / 'data/research/unit_d').mkdir(parents=True, exist_ok=True)
        (world.server / 'data/research/unit_d/completed_copies_0-0_to_0-1.json').write_text('{"completed": true}')
        fleet.tick(world.run, world.table, Namespace(dry_run=False))
        uva, server = folders(world)
        if attempt < 3:
            assert server == {'unit_d': 'pending'} and uva['unit_d'] == 'remote_pretend'
            assert fleet.load_assignment(world.run)['unit_d']['recoveries'] == attempt
    assert server == {'unit_d': 'withdrawn'} and uva['unit_d'] == 'pending'
    assert (world.run / 'data/research/unit_d/completed_copies_0-0_to_0-1.json').exists()
    assert fleet.load_assignment(world.run)['unit_d']['state'] == 'returned'


def test_refused_submission_records_no_job_and_shows_the_reason(world):
    """A job the scheduler refuses gets no ledger line, and the scheduler's words reach the log."""
    fleet.stage(world.run, world.table, stage_arguments())
    (world.state / 'sbatch_refuses').write_text('')
    fleet.submit(world.run, world.table, Namespace(server='pretend', jobs=1, pool=None, dry_run=False))
    assert fleet.ledger_ids(world.run, 'pretend') == []
    assert not (world.server / 'slurm/submitted_jobids_pretend.txt').exists()
    assert 'REFUSED by the scheduler: rndfleet-pretend-main-' in (world.run / 'slurm/fleet_log.txt').read_text()
    assert '14 CPUs per GPU' in (world.run / 'slurm/fleet_log.txt').read_text()


def test_failed_job_listing_recovers_nothing(world):
    """When the scheduler's job listing fails, a unit whose job cannot be seen is left where it is."""
    fleet.stage(world.run, world.table, stage_arguments(units=1))
    (world.server / 'queue/pending/unit_d.json').rename(world.server / 'queue/running/unit_d.json')
    (world.server / 'queue/running/unit_d.job').write_text('job=901 host=n1 gpu=card started=now')
    (world.state / 'squeue_fails').write_text('')
    assert fleet.read_status(world.run, world.table, 'pretend')[1]['squeue_ok'] is False
    fleet.tick(world.run, world.table, Namespace(dry_run=False))
    assert folders(world)[1] == {'unit_d': 'running'} and fleet.load_assignment(world.run)['unit_d']['recoveries'] == 0
    assert 'the job listing failed' in (world.run / 'slurm/fleet_log.txt').read_text()


def test_requeued_job_releases_its_own_earlier_claim(world):
    """A job that starts again under the same id puts the unit of its earlier start back and runs it."""
    fleet.stage(world.run, world.table, stage_arguments(units=1))
    (world.server / 'first_unit_check.json').write_text('{\n  "passed": true\n}')
    (world.server / 'queue/pending/unit_d.json').rename(world.server / 'queue/running/unit_d.json')
    (world.server / 'queue/running/unit_d.job').write_text('job=501 host=n1 gpu=card started=now\n')
    # While the job waits again, the tick may recover the unit too; here the job itself does it.
    worker = run_worker(world, 501)
    assert worker.returncode == 0 and 'released unit_d' in worker.stdout
    assert folders(world)[1] == {'unit_d': 'done'}
    assert len(list((world.server / 'logs').glob('unit_d_501_released_*.job'))) == 1


def test_long_wait_without_a_start_sends_units_back(world):
    """Units unclaimed for more than two hours with no job start go back to the UVA pending folder."""
    fleet.stage(world.run, world.table, stage_arguments())
    assignment = fleet.load_assignment(world.run)
    assignment['unit_d']['staged_at'] = fleet.datetime.fromtimestamp(time.time() - 3 * 3600).astimezone().isoformat()
    fleet.write_json(fleet.assignment_path(world.run), assignment)
    (world.state / 'jobs').write_text('501|PENDING|Priority|rndfleet-pretend-main-20261109-210501-1|0:00|gpu\n')
    (world.run / 'slurm/submitted_jobids_pretend.txt').write_text('501\n')
    fleet.tick(world.run, world.table, Namespace(dry_run=False))
    uva, server = folders(world)
    assert uva['unit_d'] == 'pending' and server['unit_d'] == 'withdrawn'
    assert uva['unit_c'] == 'remote_pretend' and server['unit_c'] == 'pending'
    # The waiting job is left queued; cancel-pending then cancels exactly the ledger's waiting ids.
    assert not (world.state / 'cancelled').exists()
    (world.state / 'jobs').write_text('501|PENDING|Priority|a|0:00|gpu\n502|RUNNING|None|b|5:00|gpu\n')
    (world.run / 'slurm/submitted_jobids_pretend.txt').write_text('501\n502\n777\n')
    fleet.cancel_pending(world.run, world.table, Namespace(server='pretend'))
    assert (world.state / 'cancelled').read_text().split() == ['501']


def test_take_back_returns_waiting_units_at_once_and_leaves_claimed_ones(world):
    """A waiting unit comes back to UVA's pending folder on request; a unit a job has claimed stays on the server."""
    fleet.stage(world.run, world.table, stage_arguments(units=3))
    # A worker has claimed unit_c meanwhile: it is no longer in the server's pending folder.
    (world.server / 'queue/running').mkdir(exist_ok=True)
    os.rename(world.server / 'queue/pending/unit_c.json', world.server / 'queue/running/unit_c.json')
    fleet.take_back(world.run, world.table, Namespace(server='pretend', units=1, from_end=True, match=None, hold=False))
    fleet.take_back(world.run, world.table, Namespace(server='pretend', units=1, from_end=True, match=None, hold=True))
    uva, server = folders(world)
    # Of the waiting units b and d, both are taken back; c stays where its worker has it.
    # unit_d came back to pending; unit_b was held: it sits in deferred, where no UVA worker takes it.
    assert uva['unit_b'] == 'deferred' and server['unit_b'] == 'withdrawn'
    assert uva['unit_d'] == 'pending' and server['unit_d'] == 'withdrawn'
    assert uva['unit_c'] == 'remote_pretend' and server['unit_c'] == 'running'
    assignment = fleet.load_assignment(world.run)
    assert assignment['unit_d']['state'] == 'returned' and assignment['unit_c']['state'] == 'staged'
    # Nothing matches: nothing moves.
    fleet.take_back(world.run, world.table, Namespace(server='pretend', units=2, from_end=False, match='unit_z', hold=False))
    assert folders(world) == (uva, server)


def test_copy_failure_moves_the_units_back(world, monkeypatch):
    """When the copy to the server fails, the units return to the UVA folder they came from."""
    (world.run / 'queue/pending/unit_d.json').rename(world.run / 'queue/deferred/unit_d.json')
    real = fleet.rsync

    def failing(run_dir, table, name, arguments, timeout=900):
        """Let the set-up copy pass and fail the copy of the specifications."""
        if arguments[-1].endswith('specifications/'):
            raise fleet.RemoteCommandFailed('pretend copy failure')
        return real(run_dir, table, name, arguments, timeout)

    monkeypatch.setattr(fleet, 'rsync', failing)
    before = folders(world)[0]
    with pytest.raises(fleet.RemoteCommandFailed):
        fleet.stage(world.run, world.table, stage_arguments(include_deferred=True))
    assert folders(world)[0] == before and before['unit_d'] == 'deferred'
    assert fleet.load_assignment(world.run) == {} and folders(world)[1] == {}


def test_unreachable_server_is_marked_not_ready_at_its_limit(world):
    """Failed calls in a row are counted; at the limit the server is marked not ready in the table."""
    world.table['pretend']['ssh'] = ['bash', '-c', 'exit 255', 'pretendhost']
    world.table['pretend']['failed_call_limit'] = 2
    for expected_ready in (True, False):
        with pytest.raises(fleet.ServerUnreachable):
            fleet.remote(world.run, world.table, 'pretend', 'true\n')
        assert fleet.read_json(fleet.table_path(world.run))['pretend']['ready'] is expected_ready
    # A closed shared connection sends nothing to the server.
    world.table['pretend'].update(check=['false'], ready=True, failed_calls_in_a_row=0)
    with pytest.raises(fleet.ServerUnreachable, match='shared connection is closed'):
        fleet.remote(world.run, world.table, 'pretend', 'true\n')


def test_status_table_lists_pools_jobs_and_balance(world):
    """The status table has one row per pool with job states, unit counts, the balance and the check verdict."""
    fleet.stage(world.run, world.table, stage_arguments())
    (world.state / 'jobs').write_text('501|PENDING|Priority|rndfleet-pretend-main-20261109-210501-1|0:00|gpu\n'
                                      '9|RUNNING|None|another-study|9:00|gpu\n')
    (world.run / 'slurm/submitted_jobids_pretend.txt').write_text('501\n')
    assigned, status = fleet.read_status(world.run, world.table, 'pretend')
    rows = fleet.status_rows(world.run, world.table, {'pretend': (status, assigned)})
    assert '1 PENDING' in rows[2] and '2/0/0/0/0' in rows[2] and '60.00 SUs' in rows[2] and 'not run yet' in rows[2]
    assert len(rows) == 4 and 'job 501' in rows[3] and '(Priority)' in rows[3]


def test_compare_arrays_accepts_rounding_and_refuses_real_differences(tmp_path):
    """The first-unit comparison passes identical and nearly equal floats and fails integers that differ."""
    numpy = pytest.importorskip('numpy')
    import remote_first_unit_compare as compare
    ones = numpy.ones(4, numpy.float32)
    assert compare.compare_arrays(ones, ones.copy()) == (True, 'identical', 0.0)
    passed, reason, difference = compare.compare_arrays(ones, ones * numpy.float32(1.00001))
    assert passed and reason == 'within the relative tolerance' and 5e-6 < difference < 2e-5
    assert compare.compare_arrays(ones, ones * numpy.float32(1.01))[0] is True
    assert compare.compare_arrays(ones, ones * numpy.float32(1.2))[0] is False
    # Values near zero are compared on an absolute scale; integers and shapes must match exactly.
    assert compare.compare_arrays(numpy.zeros(2), numpy.asarray([1e-12, 0.]))[0] is True
    assert compare.compare_arrays(numpy.arange(3), numpy.asarray([0, 1, 3])) == (False, 'integer values differ', None)
    assert compare.compare_arrays(ones, numpy.ones(5, numpy.float32))[1] == 'shape or number kind differs'
    assert compare.compare_arrays(numpy.asarray([numpy.nan, 1.]), numpy.asarray([numpy.nan, 1.]))[0] is True
    assert compare.compare_arrays(numpy.asarray([numpy.nan, 1.]), numpy.asarray([2., 1.]))[0] is False
    numpy.savez(tmp_path / 'reference.npz', floats=ones, integers=numpy.arange(3), only_here=ones)
    numpy.savez(tmp_path / 'candidate.npz', floats=ones * numpy.float32(1.00001), integers=numpy.arange(3))
    record = compare.compare_files(tmp_path / 'reference.npz', tmp_path / 'candidate.npz')
    assert record['passed'] is False and [row['array'] for row in record['failed_arrays']] == ['only_here']
    assert record['arrays_compared'] == 3 and record['identical_arrays'] == 1
    assert record['largest_difference_array'] == 'floats'


def test_a_command_reads_the_table_only_once_it_holds_the_lock(tmp_path, monkeypatch):
    """A writer that holds the lock adds a server; the command that waited for the lock works on the new table."""
    (tmp_path / 'slurm').mkdir()
    path = fleet.table_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fleet.write_json(path, {'gilbreth': {'ready': True}})
    real_lock, seen = fleet.fleet_lock, []

    def lock_held_by_another_writer(where):
        """Before the lock is handed over, the other writer adds a server."""
        fleet.write_json(path, {'gilbreth': {'ready': True}, 'mit': {'ready': False}})
        return real_lock(where)

    monkeypatch.setattr(fleet, 'fleet_lock', lock_held_by_another_writer)
    monkeypatch.setattr(fleet, 'tick', lambda run, table, args, act=True: seen.append(sorted(table)))
    monkeypatch.setattr(sys, 'argv', ['remote_fleet.py', '--run-dir', str(tmp_path), 'tick'])
    fleet.main()
    assert seen == [['gilbreth', 'mit']]
