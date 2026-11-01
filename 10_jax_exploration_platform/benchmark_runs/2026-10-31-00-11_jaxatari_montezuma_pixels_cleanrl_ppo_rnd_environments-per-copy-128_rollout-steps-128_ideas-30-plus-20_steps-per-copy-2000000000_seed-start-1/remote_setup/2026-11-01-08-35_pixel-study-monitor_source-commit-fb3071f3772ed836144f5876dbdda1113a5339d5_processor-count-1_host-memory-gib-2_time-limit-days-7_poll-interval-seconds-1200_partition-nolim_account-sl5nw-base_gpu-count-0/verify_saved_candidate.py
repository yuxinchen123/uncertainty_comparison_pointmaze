"""Verify the completed candidate's saved report, preserving its extra-newline wrapper failure without a second tick."""

import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys


PACKET = Path(__file__).resolve().parent
RUN = PACKET.parent.parent


def save(name, value):
    """Fsync an immutable verification or roster without replacing the original candidate evidence."""
    raw = (json.dumps(value, indent=2) + '\n').encode()
    with (PACKET / name).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(PACKET, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)
    return {'path': str(PACKET / name), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    """Check actual source/report bytes and inherited SMT guards using saved metadata only."""
    assert os.environ.get('SLURM_JOB_ID') == '6781087'
    plan = json.loads((PACKET / 'monitor_replacement_plan.json').read_bytes())
    source_manifest = json.loads((PACKET / 'source_snapshot.json').read_bytes())
    source = Path(source_manifest['source_root'])
    assert plan['helper_commit'] == source_manifest['source_commit'] == 'fb3071f3772ed836144f5876dbdda1113a5339d5'
    assert plan['job_id'] is plan['gpu_uuid'] is plan['reservation'] is None
    for name, expected in source_manifest['files'].items():
        path = source / name
        assert path.is_file() and not path.is_symlink()
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected['sha256']
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == expected['git_blob']
        assert stat.S_IMODE(path.stat().st_mode) & 0o111 == (0o111 if expected['git_mode'] == '100755' else 0)
    assert {str(path.relative_to(source)) for path in source.rglob('*') if path.is_file()} == set(source_manifest['files'])
    stdout = (PACKET / 'candidate_full_tick_stdout.raw').read_bytes()
    report = (PACKET / 'candidate_full_tick_report.md').read_bytes()
    status = json.loads((PACKET / 'candidate_full_tick_status.json').read_bytes())
    table = report.split(b'```text\n', 1)[1].removesuffix(b'```\n')
    assert stdout == table + b'\n'
    assert all(heading.encode() in table for heading in ('TRAINING\n', 'WAITING\n', 'HELD\n', 'FINISHED\n', 'PLANNED\n'))
    assert max(map(len, table.decode().splitlines())) <= 100
    failed = json.loads((PACKET / 'candidate_full_tick_failure.json').read_bytes())
    assert failed['type'] == 'AssertionError' and failed['step_id'] == '63' and not failed['blocked_action_attempts']
    assert 'line 147' in failed['traceback'] and 'raw_stdout in report' in failed['traceback']
    wrapper_lines = (PACKET / 'candidate_full_tick.py').read_text().splitlines()
    assert wrapper_lines[144].strip() == 'assert before == after and not BLOCKED and complete is False'
    assert not (RUN / '20_mins_monitoring/controller_error.json').exists()
    allocation = json.loads((PACKET / 'candidate_allocation.json').read_bytes())
    assert allocation['job_id'] == '6781087' and allocation['step_id'] == '63' and allocation['new_monitor_job_id'] is None
    specification = importlib.util.spec_from_file_location('saved_monitor_guard', PACKET / 'allocation_guard.py')
    guard = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(guard)
    fields = dict(token.split('=', 1) for token in allocation['job']['stdout'].split() if '=' in token)
    node = dict(token.split('=', 1) for token in allocation['node']['stdout'].split() if '=' in token)
    guard.validate(fields, '6781087', allocation['hostname'], allocation['cgroup'], node)
    single = copy.deepcopy(fields)
    single['NumCPUs'] = '1'
    single['AllocTRES'] = single['AllocTRES'].replace('cpu=2', 'cpu=1')
    guard.validate(single, '6781087', allocation['hostname'], allocation['cgroup'], node)
    negatives = []
    for name, key, value in [('requested_two_cpus', 'ReqTRES', fields['ReqTRES'].replace('cpu=1', 'cpu=2')),
                              ('allocated_three_cpus', 'NumCPUs', '3'),
                              ('nonzero_gpu', 'AllocTRES', fields['AllocTRES'] + ',gres/gpu=1'),
                              ('wrong_owner', 'UserId', 'other(1)'), ('requeue_enabled', 'Requeue', '1')]:
        changed = {**fields, key: value}
        try:
            guard.validate(changed, '6781087', allocation['hostname'], allocation['cgroup'], node)
        except ValueError:
            negatives.append(name)
        else:
            raise AssertionError('Invalid CPU monitor shape accepted: ' + name)
    expected_functions = {row.name: ast.dump(row, include_attributes=False) for row in ast.parse((PACKET / 'allocation_guard.py').read_text()).body if isinstance(row, ast.FunctionDef)}
    old_packet = Path(json.loads((PACKET / 'monitor_replacement_plan.json').read_bytes())['resource_basis']['current_monitor_plan']['path']).parent
    old_functions = {row.name: ast.dump(row, include_attributes=False) for row in ast.parse((old_packet / 'allocation_guard.py').read_text()).body if isinstance(row, ast.FunctionDef)}
    assert expected_functions == old_functions
    assert not any(name.split('.')[0] in {'jax', 'jaxlib', 'numpy', 'torch', 'jaxtari', 'flax', 'optax'} for name in sys.modules)
    result = {'schema_version': 1, 'kind': 'saved_actual_published_monitor_candidate_verification', 'passed': True,
        'at_utc': datetime.now(timezone.utc).isoformat(), 'source_commit': source_manifest['source_commit'],
        'source_file_count': len(source_manifest['files']), 'actual_candidate_job': '6781087', 'actual_candidate_step': '63',
        'metadata_verification_step': os.environ['SLURM_STEP_ID'], 'new_job_id': None,
        'candidate_full_tick_completed': True, 'full_tick_repeated': False,
        'original_failed_verifier_preserved': True, 'formatting_correction': 'Actual print(text) emits one extra newline; saved Markdown embeds text once. Exact equality stdout == fenced_table + one newline passed.',
        'protected_file_equality_passed_before_original_wrapper_failure': True,
        'protected_names': ['next_actions.json', 'campaign_provenance.json', 'SOURCE_AND_GATES_READY.json'],
        'protected_write_attempts': [], 'source_and_modes_verified': True, 'controller_error': None,
        'all_five_status_sections_present': True, 'maximum_report_line_length': max(map(len, table.decode().splitlines())),
        'reported_queue_counts': status['campaign']['queue_counts'], 'old_cpu_monitor_preserved': '6781087',
        'unchanged_allocation_guard_functions': True, 'actual_two_thread_and_synthetic_single_thread_goldens_passed': True,
        'negative_allocation_cases_rejected': negatives, 'numerical_imports': False, 'gpu_or_device_calls': False,
        'python_socket_connections': False, 'local_scheduler_read_queries_performed': True,
        'new_jobs_submitted': 0, 'job_releases_or_cancellations': 0,
        'scientific_case_credit': 0, 'scientific_hypothesis_credit': 0, 'final_training_credit': 0, 'worker_admission': False}
    verification = save('saved_candidate_full_tick_verification.json', result)
    artifacts = []
    unit = RUN / 'data' / PACKET.name
    for scope, root in (('packet', PACKET), ('data_unit', unit)):
        for path in sorted(root.rglob('*')):
            assert not path.is_symlink()
            if path.is_file():
                artifacts.append({'scope': scope, 'relative_path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                    'mode': oct(stat.S_IMODE(path.stat().st_mode)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    roster = save('preparation_and_candidate_file_roster.json', {'schema_version': 1, 'kind': 'frozen_cpu_monitor_preparation_file_roster',
        'source_commit': source_manifest['source_commit'], 'packet': str(PACKET), 'data_unit': str(unit),
        'verification': verification, 'files': artifacts, 'new_jobs_submitted': 0, 'old_monitor_to_preserve': '6781087'})
    print(json.dumps({'passed': True, 'verification': verification, 'roster': roster, 'source_files': len(source_manifest['files']),
        'exact_artifact_count': len(artifacts), 'metadata_step': os.environ['SLURM_STEP_ID']}))


if __name__ == '__main__':
    main()
