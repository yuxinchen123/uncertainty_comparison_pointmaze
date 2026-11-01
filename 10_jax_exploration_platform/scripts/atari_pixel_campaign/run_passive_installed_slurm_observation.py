"""Prepare and run one strictly bound zero-GPU passive Slurm observation.

Preparation writes immutable standard records with no job identifier. A separate
binding action may record the actual held identifier only after the root has
fsynced its canonical owned ledger. This module never submits or releases a job.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import subprocess
import time

import passive_installed_slurm_observation as passive
import uva_active_configuration_observation as ownership


FILENAME = 'run_passive_installed_slurm_observation.py'
KIND = 'owned_zero_gpu_passive_installed_slurm_observation'
PREFIX = ownership.PREFIX
FILES = (FILENAME, 'passive_installed_slurm_observation.py', 'uva_active_configuration_observation.py')
LIBRARY_SHA256 = '97ce634c07cd7a2b045d4253938db21c844abd93f3bb7b9d4dbebfd5f33bf842'
OWNERSHIP_SHA256 = '8cc5a3544e426e3bd3e78707f6c77dc249a1d19165a593e157d2097daa07a648'
PYTHON = '/usr/bin/python3'
ZERO_CREDIT = dict(passive.ZERO_CREDIT)


def require(condition, message):
    """Reject a changed source, identity or preparation before collecting installed metadata."""
    if not condition:
        raise ValueError(message)


def sha256(raw):
    """Hash exact staged or retained bytes."""
    return hashlib.sha256(raw).hexdigest()


def source_identity(root, commit):
    """Bind all three files to one clean detached source with exact approved dependency bytes."""
    require(root.is_absolute() and root.resolve(strict=True) == root and re.fullmatch(r'[0-9a-f]{40}', commit),
            'an exact ordinary detached source root and full commit are required')
    command = ['git', '-C', str(root)]
    require(subprocess.check_output(command + ['rev-parse', '--show-toplevel'], text=True).strip() == str(root)
        and subprocess.check_output(command + ['rev-parse', 'HEAD'], text=True).strip() == commit
        and subprocess.run(command + ['symbolic-ref', '--quiet', 'HEAD'], capture_output=True).returncode == 1
        and not subprocess.check_output(command + ['status', '--porcelain=v1', '-uall'], text=True).strip(),
        'the passive source must remain clean and detached at its recorded commit')
    hashes = {}
    for name in FILES:
        relative = PREFIX + name
        raw = subprocess.check_output(command + ['show', commit + ':' + relative])
        require(ownership.ordinary(root / relative) == raw, 'passive source differs from its committed file: ' + name)
        hashes[relative] = sha256(raw)
    require(hashes[PREFIX + FILES[1]] == LIBRARY_SHA256 and hashes[PREFIX + FILES[2]] == OWNERSHIP_SHA256,
            'only the independently reviewed passive library and ownership dependency are supported')
    return {'source_root': str(root), 'source_commit': commit, 'files': hashes}


def preparation_artifacts(unit, plan, source_root):
    """Construct standard metadata records while retaining the original strict ownership plan separately."""
    source = plan['source']
    owner_source = {'source_root': source['source_root'], 'source_commit': source['source_commit'],
        'files': {PREFIX + FILES[2]: source['files'][PREFIX + FILES[2]]}}
    owner_plan = {'schema_version': 1, 'kind': ownership.KIND, 'campaign_run_dir': plan['campaign_run_dir'],
        'unit_id': plan['unit_id'], 'source': owner_source, 'owner': ownership.OWNER,
        'shape': ownership.SHAPE, 'python': PYTHON, 'job_id': None, 'reservation': None, 'gpu_uuid': None,
        'owned_job_ledger': plan['owned_job_ledger'], 'settings_allowlist': list(ownership.SETTINGS),
        **ownership.ZERO_CREDIT}
    command = [PYTHON, '-B', str(unit / 'code' / FILENAME), 'run', str(unit)]
    background = ('## Purpose\n\nRead bounded installed Slurm and configuration metadata in one owned zero-GPU '
        'allocation. No daemon, GPU library, device, learner or external network client is used. '
        'Only the reviewed Slurm read commands query the scheduler. This grants no configuration '
        'qualification, physical mapping, device enforcement, scientific credit or training admission.\n\n'
        '## Key hyperparameters\n\nRequest one processor, two GiB of host memory and five minutes on jaguar03, '
        'with zero GPUs. Record the actual one or two allocated simultaneous threads and numeric step. '
        'Use the actual owned reservation only in a separately recorded returned-identifier binding.\n\n'
        '## Code and config changes\n\nThe wrapper binds the exact committed passive library and unchanged strict '
        'ownership observer before collection. It retains separate raw ownership and passive observations. '
        'Unknown Includes, package/build identities, startup records, configuration trust and GPU mapping '
        'remain explicit unknowns. No slurmd command or plugin loading occurs.\n\n'
        '## Git state\n\nCommit: ' + source['source_commit'] + '. The source is clean and detached; '
        'all three staged source hashes are checked before and after collection.\n').encode()
    artifacts = {'passive_plan.json': ownership.encoded(plan), 'manifest.yaml': ownership.encoded(plan),
        'config_resolved.yaml': ownership.encoded({'shape': ownership.SHAPE, 'limits': passive.LIMITS,
            'source_files': source['files'], 'python': PYTHON, **ZERO_CREDIT}),
        'experiment_background.md': background, 'command.txt': (shlex.join(command) + '\n').encode(),
        'observation_plan.json': ownership.encoded(owner_plan)}
    artifacts['prepared.json'] = ownership.encoded({'plan_sha256': sha256(artifacts['observation_plan.json'])})
    artifacts.update({'code/' + name: ownership.ordinary(source_root / PREFIX / name) for name in FILES})
    return artifacts


def prepare(run, unit_id, source_root, source_commit):
    """Write a fresh immutable unit without guessing a job, reservation or GPU identity."""
    require(run.is_absolute() and run.resolve(strict=True) == run and run.is_dir(), 'exact existing campaign required')
    require(unit_id and Path(unit_id).name == unit_id and unit_id not in {'.', '..'}, 'plain passive unit name required')
    unit = run / 'data' / unit_id
    require(not unit.exists() and not unit.is_symlink(), 'preserve the previous unit; prepare a new name')
    require(not (run / 'data').exists() or ((run / 'data').is_dir()
        and (run / 'data').resolve(strict=True) == run / 'data'), 'campaign data parent is not an ordinary directory')
    source = source_identity(source_root, source_commit)
    plan = {'schema_version': 1, 'kind': KIND, 'campaign_run_dir': str(run), 'unit_id': unit_id,
        'source': source, 'owner': ownership.OWNER, 'shape': ownership.SHAPE, 'python': PYTHON,
        'library_capability': passive.CAPABILITY, 'job_id': None, 'reservation': None, 'gpu_uuid': None,
        'owned_job_ledger': str(run / 'slurm/submitted_jobids_uva.txt'), **ZERO_CREDIT}
    artifacts = preparation_artifacts(unit, plan, source_root)
    for name, raw in artifacts.items():
        ownership.write_once(unit / name, raw)
    ownership.write_once(unit / 'passive_prepared.json', ownership.encoded({'schema_version': 1, 'kind': KIND,
        'artifact_sha256': {name: sha256(raw) for name, raw in artifacts.items()}}))
    ownership.progress(unit, {'kind': KIND, 'unit_id': unit_id, 'state': 'planned',
        'phase': 'prepared_unsubmitted', 'time': datetime.now(timezone.utc).isoformat(), **ZERO_CREDIT})
    return unit


def verify_static(unit, execution=False):
    """Verify exact plans, source closure and staged import paths before any scheduler or file collection."""
    plan = json.loads(ownership.ordinary(unit / 'passive_plan.json'))
    require(type(plan.get('schema_version')) is int and plan['schema_version'] == 1 and plan['kind'] == KIND
        and plan['owner'] == ownership.OWNER and plan['shape'] == ownership.SHAPE and plan['python'] == PYTHON
        and plan['library_capability'] == passive.CAPABILITY
        and unit == Path(plan['campaign_run_dir']) / 'data' / plan['unit_id']
        and plan['owned_job_ledger'] == str(Path(plan['campaign_run_dir']) / 'slurm/submitted_jobids_uva.txt')
        and all(plan[name] is None for name in ('job_id', 'reservation', 'gpu_uuid'))
        and all(type(plan[name]) is type(value) and plan[name] == value for name, value in ZERO_CREDIT.items()),
        'unsupported passive preparation, identity or nonzero credit')
    require(source_identity(Path(plan['source']['source_root']), plan['source']['source_commit']) == plan['source'],
            'the detached passive source identity changed')
    expected = preparation_artifacts(unit, plan, Path(plan['source']['source_root']))
    prepared = json.loads(ownership.ordinary(unit / 'passive_prepared.json'))
    require(type(prepared.get('schema_version')) is int and prepared == {'schema_version': 1, 'kind': KIND,
        'artifact_sha256': {name: sha256(raw) for name, raw in expected.items()}}, 'passive preparation roster changed')
    require(all(ownership.ordinary(unit / name) == raw for name, raw in expected.items()), 'a staged passive artifact changed')
    ownership.verify_static(unit)
    if execution:
        for name, path in [(FILENAME, __file__), (FILES[1], passive.__file__), (FILES[2], ownership.__file__)]:
            require(Path(path).absolute() == unit / 'code' / name,
                    'execute only the exact staged wrapper, library and ownership modules: ' + name)
    return plan


def bind_held_identifier(unit, job_id, reservation):
    """Record only the actual returned identifier after the canonical owned ledger already contains it once."""
    plan = verify_static(unit)
    binding = ownership.bind_held_submission(unit, job_id, reservation)
    record = {'schema_version': 1, 'kind': KIND + '_held_identifier_binding', 'job_id': job_id,
        'reservation': reservation, 'passive_plan_sha256': sha256(ownership.ordinary(unit / 'passive_plan.json')),
        'ownership_binding_sha256': sha256(ownership.ordinary(unit / 'held_identifier_binding.json')),
        'source': plan['source'], **ZERO_CREDIT}
    ownership.write_once(unit / 'passive_held_identifier_binding.json', ownership.encoded(record))
    return record


def owned_query(command, job, step):
    """Capture only the exact admitted job or numeric-step query with bounded local pipes."""
    require(command in (['scontrol', 'show', 'job', '-d', job],
        ['scontrol', 'show', 'step', '-d', job + '.' + step]), 'unsupported owned scheduler command')
    query = {'command': command, 'returncode': None, 'capture_status': 'not_started'}
    buffers, process = {'stdout': bytearray(), 'stderr': bytearray()}, None
    selector = selectors.DefaultSelector()
    deadline = time.monotonic() + passive.LIMITS['query_timeout_seconds']
    try:
        # The campaign's own ledger is verified by collect_owned_allocation before this helper is called.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        for name in buffers:
            stream = getattr(process, name)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        query['capture_status'] = 'completed'
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                query['capture_status'] = 'timeout'
                break
            for event, unused in selector.select(remaining):
                stream, name = event.fileobj, event.data
                try:
                    raw = os.read(stream.fileno(), min(65536, passive.LIMITS['query_bytes'] + 1 - len(buffers[name])))
                except BlockingIOError:
                    continue
                if not raw:
                    selector.unregister(stream)
                    continue
                buffers[name].extend(raw)
                if len(buffers[name]) > passive.LIMITS['query_bytes']:
                    query.update(capture_status='response_limit_exceeded', exceeded_stream=name)
                    break
            if query['capture_status'] != 'completed':
                break
        if query['capture_status'] != 'completed' and process.poll() is None:
            process.kill()
        try:
            query['returncode'] = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            query['capture_status'] = 'timeout'
            process.kill()
            query['returncode'] = process.wait(timeout=5)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        query.update(capture_status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            for name in buffers:
                getattr(process, name).close()
        selector.close()
    for name, raw in buffers.items():
        query.update({name: bytes(raw).decode('utf-8', errors='replace'), name + '_capture_sha256': sha256(raw),
            name + '_captured_byte_count': len(raw)})
        try:
            bytes(raw).decode('utf-8', errors='strict')
        except UnicodeError:
            query['capture_status'] = 'unsupported_text_encoding'
    return query


def collect_owned_allocation(unit):
    """Reuse exact ownership/resource rules without executing the old configuration-reading stage."""
    plan = ownership.verify_static(unit, execution=True)
    record = {'schema_version': 1, 'kind': ownership.KIND,
        'plan_sha256': sha256(ownership.ordinary(unit / 'observation_plan.json')),
        'source': plan['source'], 'started_utc': datetime.now(timezone.utc).isoformat(), **ownership.ZERO_CREDIT}
    try:
        record['process'] = ownership.own_process()
        job, step = os.environ.get('SLURM_JOB_ID', ''), os.environ.get('SLURM_STEP_ID', '')
        record.update(job_id=job, step_id=step, environment={name: os.environ[name] for name in
            ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_CPUS_PER_TASK', 'SLURM_CONF', 'SLURM_GPUS_ON_NODE',
             'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES') if name in os.environ})
        require(re.fullmatch(r'[1-9][0-9]*', job) and re.fullmatch(r'[0-9]+', step), 'actual owned numeric job/step required')
        require(record['process']['uid_fields'] == [ownership.OWNER['uid']] * 4
            and record['process']['uid'] == ownership.OWNER['uid'], 'refuse another owner before any scheduler query')
        binding = json.loads(ownership.ordinary(unit / 'held_identifier_binding.json'))
        frozen = ownership.ordinary(unit / 'held_owned_job_ledger_snapshot.txt')
        require(type(binding.get('schema_version')) is int and binding['schema_version'] == 1
            and binding['kind'] == ownership.KIND + '_returned_held_identifier_binding'
            and binding['cluster'] == 'uva' and binding['owner'] == ownership.OWNER
            and binding['job_id'] == job and binding['plan_sha256'] == record['plan_sha256']
            and binding['owned_ledger_origin'] == plan['owned_job_ledger']
            and binding['owned_ledger_sha256'] == sha256(frozen) and frozen.decode().splitlines().count(job) == 1
            and re.fullmatch(r'sl5nw_[1-9][0-9]*', binding['reservation'])
            and binding['job_released_by_producer'] is False
            and all(type(binding[name]) is type(value) and binding[name] == value for name, value in ownership.ZERO_CREDIT.items()),
            'actual job differs from the immutable strict ownership binding')
        ledger = ownership.ordinary(Path(plan['owned_job_ledger']))
        require(ledger.decode().splitlines().count(job) == 1, 'actual job must occur once in the canonical owned ledger')
        ownership.write_once(unit / 'owned_job_ledger_snapshot.txt', ledger)
        record.update(held_identifier_binding=binding,
            owned_ledger={'origin': plan['owned_job_ledger'], 'sha256': sha256(ledger)})
        for name, command in [('job_query', ['scontrol', 'show', 'job', '-d', job]),
                ('step_query', ['scontrol', 'show', 'step', '-d', job + '.' + step])]:
            query = record[name] = owned_query(command, job, step)
            require(query['capture_status'] == 'completed' and query['returncode'] == 0 and not query['stderr'],
                    'owned scheduler query failed or exceeded its byte/time limit')
        ownership.validate_allocation(record)
        node = record['cpu_topology_query'] = passive.controller_query(
            ['scontrol', 'show', 'node', 'jaguar03', '-o'], ('NodeName', 'ThreadsPerCore'))
        threads = passive.unique_setting(node['fields'], 'ThreadsPerCore')
        require(node['capture_status'] == 'completed' and node['returncode'] == 0
            and node['stderr_byte_count'] == 0 and node['filter_status'] == 'parsed'
            and passive.unique_setting(node['fields'], 'NodeName') == 'jaguar03' and threads in {'1', '2'}
            and (ownership.scheduler_field(record['job_query']['stdout'], 'NumCPUs') != '2' or threads == '2'),
            'a two-logical-processor charge requires actual two-thread-per-core evidence on the owned node')
        record.update(allocation_binding_passed=True, completed=True)
        ownership.write_once(unit / 'hardware.json', ownership.encoded({'kind': KIND,
            'hostname': record['process']['hostname'], 'pid': record['process']['pid'], 'slurm_job_id': job,
            'gpu_query': '', 'actual_cpu_affinity': record['process']['cpu_affinity'], 'allocated_gpu_count': 0, **ZERO_CREDIT}))
    except Exception as error:
        record.update(completed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        ownership.write_once(unit / 'observation.json', ownership.encoded(record))
    return record


def run(unit):
    """Retain both actual observations and fail without retrying or granting any scientific credit."""
    plan = verify_static(unit, execution=True)
    require(not any((unit / name).exists() for name in ('passive_observation.json', 'result.json', 'failure.json', 'observation.json')),
            'preserve the prior attempt and prepare a fresh passive unit')
    record = {'schema_version': 1, 'kind': KIND, 'source': plan['source'],
        'passive_plan_sha256': sha256(ownership.ordinary(unit / 'passive_plan.json')),
        'started_utc': datetime.now(timezone.utc).isoformat(), **ZERO_CREDIT}
    try:
        record.update(process=ownership.own_process(), observed_job_id=os.environ.get('SLURM_JOB_ID'),
            observed_step_id=os.environ.get('SLURM_STEP_ID'))
        binding = json.loads(ownership.ordinary(unit / 'passive_held_identifier_binding.json'))
        require(type(binding.get('schema_version')) is int and binding['schema_version'] == 1
            and binding['kind'] == KIND + '_held_identifier_binding' and binding['source'] == plan['source']
            and binding['job_id'] == os.environ.get('SLURM_JOB_ID')
            and re.fullmatch(r'sl5nw_[1-9][0-9]*', binding['reservation'])
            and binding['passive_plan_sha256'] == record['passive_plan_sha256']
            and binding['ownership_binding_sha256'] == sha256(ownership.ordinary(unit / 'held_identifier_binding.json'))
            and all(type(binding[name]) is type(value) and binding[name] == value for name, value in ZERO_CREDIT.items()),
            'actual owned job differs from the frozen passive held binding')
        owned_binding = json.loads(ownership.ordinary(unit / 'held_identifier_binding.json'))
        require(binding['job_id'] == owned_binding['job_id'] and binding['reservation'] == owned_binding['reservation'],
                'passive and strict ownership bindings differ')
        owned = collect_owned_allocation(unit)
        require(owned['completed'] is True and owned['allocation_binding_passed'] is True,
                'the strict actual zero-GPU allocation observation did not complete')
        record.update(job_id=owned['job_id'], step_id=owned['step_id'],
            ownership_observation_sha256=sha256(ownership.ordinary(unit / 'observation.json')))
        ownership.progress(unit, {'kind': KIND, 'unit_id': plan['unit_id'], 'state': 'running',
            'phase': 'passive_installed_metadata', 'slurm_job_id': owned['job_id'],
            'time': datetime.now(timezone.utc).isoformat(), **ZERO_CREDIT})
        observed = passive.collect_owned_metadata(owned)
        ownership.write_once(unit / 'passive_observation.json', ownership.encoded(observed))
        record['passive_observation_sha256'] = sha256(ownership.ordinary(unit / 'passive_observation.json'))
        require(observed['kind'] == passive.KIND and observed['collection_completed'] is True
            and all(type(observed[name]) is type(value) and observed[name] == value for name, value in ZERO_CREDIT.items()),
            'passive collection failed or claimed nonzero credit; retain its raw evidence')
        verify_static(unit, execution=True)
        record['completed'] = True
    except Exception as error:
        record.update(completed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        ownership.write_once(unit / ('result.json' if record.get('completed') else 'failure.json'), ownership.encoded(record))
        ownership.progress(unit, {'kind': KIND, 'unit_id': plan['unit_id'],
            'state': 'completed' if record.get('completed') else 'failed', 'phase': 'passive_installed_metadata',
            'slurm_job_id': record.get('job_id', os.environ.get('SLURM_JOB_ID')),
            'time': record['finished_utc'], **ZERO_CREDIT})
    return record


def main():
    """Expose preparation, actual held-ID binding and one staged run without submission or release commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    preparation = actions.add_parser('prepare')
    preparation.add_argument('--run-dir', type=Path, required=True)
    preparation.add_argument('--unit-id', required=True)
    preparation.add_argument('--source-root', type=Path, required=True)
    preparation.add_argument('--source-commit', required=True)
    execution = actions.add_parser('run')
    execution.add_argument('unit', type=Path)
    binding = actions.add_parser('bind-held-identifier')
    binding.add_argument('unit', type=Path)
    binding.add_argument('--job-id', required=True)
    binding.add_argument('--reservation', required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps({'unit': str(prepare(args.run_dir, args.unit_id, args.source_root, args.source_commit)), 'submitted': False}))
    elif args.action == 'bind-held-identifier':
        print(json.dumps(bind_held_identifier(args.unit, args.job_id, args.reservation)))
    else:
        print(json.dumps(run(args.unit)))


if __name__ == '__main__':
    main()
