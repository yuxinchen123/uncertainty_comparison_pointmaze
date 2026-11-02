"""Freeze the reviewed published monitor source and its proven CPU-only startup records without submission."""

import ast
import importlib.util
import socket
import sys
import traceback
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import tarfile


PACKET = Path(__file__).resolve().parent
RUN = PACKET.parent.parent
UNIT = RUN / 'data' / PACKET.name
CONTEXT = json.loads((PACKET / 'preparation_context.json').read_text())
REPOSITORY = Path(CONTEXT['clean_source_root'])
COMMIT = '79a01e851ff92ddba7d0e12eaf8164cab2028878'
OLD_COMMIT = '31fd9fe7cd96343741e8b48a18019efee9b4ac5c'
OLD = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-14-02_pixel-study-monitor_source-commit-31fd9fe7cd96343741e8b48a18019efee9b4ac5c_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0')
PREFIXES = ['10_jax_exploration_platform/src', '10_jax_exploration_platform/scripts']


def write(path, raw, mode=0o644):
    """Fsync new preparation bytes without replacing any earlier packet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(mode)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)


def document(path, value):
    """Use JSON as valid YAML for exact machine-readable preparation records."""
    write(path, (json.dumps(value, indent=2) + '\n').encode())


def receipt(path):
    """Bind an ordinary prepared artifact to its exact current bytes."""
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def git(*arguments):
    """Read committed source objects without changing a repository index or reference."""
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(REPOSITORY), *arguments],
                                   env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})


def duration_seconds(value):
    """Read a canonical Slurm duration as seconds; a longer partition cap never rejects a fitting request."""
    if value in {"UNLIMITED", "INFINITE"}:
        return None
    matched = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", value)
    if matched is None:
        raise ValueError("unsupported actual partition duration: " + value)
    days, hours, minutes, seconds = matched.groups()
    days, hours, minutes, seconds = int(days or 0), int(hours), int(minutes), int(seconds)
    if minutes >= 60 or seconds >= 60 or matched.group(1) is not None and hours >= 24:
        raise ValueError("invalid actual partition duration: " + value)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def main():
    """Create a complete detached source archive, unchanged resource guard and unresolved held-command plan."""
    assert socket.gethostname().split('.')[0] == 'portal05' and os.getuid() == 2618919
    assert not any(name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax'} for name in sys.modules)
    print('portal05 standard-library static recovery preparation; no actual monitor tick', flush=True)
    assert git('rev-parse', 'HEAD').decode().strip() == COMMIT
    assert not git('status', '--porcelain', '--untracked-files=all').strip()
    assert subprocess.run(['git', '--no-optional-locks', '-C', str(REPOSITORY), 'symbolic-ref', '--quiet', 'HEAD'], capture_output=True).returncode == 1
    assert git('rev-parse', COMMIT).decode().strip() == COMMIT
    assert subprocess.run(['git', '--no-optional-locks', '-C', str(REPOSITORY), 'merge-base', '--is-ancestor',
                           COMMIT, 'origin/merge-atari-cleanrl']).returncode == 0
    archive = git('archive', '--format=tar', '--prefix=' + COMMIT + '/', COMMIT, *PREFIXES)
    write(PACKET / 'source_git_archive.tar', archive)
    entries = {}
    for entry in git('ls-tree', '-rz', COMMIT, '--', *PREFIXES).split(b'\0'):
        if entry:
            header, name = entry.split(b'\t', 1)
            mode, kind, blob = header.decode().split()
            assert mode in {'100644', '100755'} and kind == 'blob'
            entries[name.decode()] = {'git_mode': mode, 'git_blob': blob}

    # Before: an exact Git tar member; after: one fresh ordinary file in the named immutable source tree.
    source = PACKET / 'source_snapshot' / COMMIT
    extracted = set()
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as stream:
        for member in stream:
            parts = Path(member.name).parts
            assert parts and parts[0] == COMMIT and '..' not in parts and not member.name.startswith('/')
            if member.isdir():
                continue
            assert member.isfile()
            name = Path(*parts[1:]).as_posix()
            assert name in entries and name not in extracted
            raw = stream.extractfile(member).read()
            record = entries[name]
            assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == record['git_blob']
            write(source / name, raw, 0o755 if record['git_mode'] == '100755' else 0o644)
            record['sha256'] = hashlib.sha256(raw).hexdigest()
            record['bytes'] = len(raw)
            extracted.add(name)
    assert extracted == set(entries)
    document(PACKET / 'source_snapshot.json', {'schema_version': 1, 'kind': 'complete_committed_monitor_source_archive',
        'source_commit': COMMIT, 'source_root': str(source), 'source_prefixes': PREFIXES, 'files': entries,
        'archive': receipt(PACKET / 'source_git_archive.tar'), 'working_tree_bytes_used': False,
        'clean_detached_source_root': str(REPOSITORY), 'supports_reviewed_direct_selected_and_profile_readers': True, 'direct_final_capacity_scaling_portable_supported': False})

    # Keep the proven SMT validation functions byte-identical; only this new source/script identity changes.
    guard = (OLD / 'allocation_guard.py').read_text().replace(str(OLD), str(PACKET)).replace(OLD_COMMIT, COMMIT)
    write(PACKET / 'allocation_guard.py', guard.encode())
    old_functions = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse((OLD / 'allocation_guard.py').read_text()).body if isinstance(node, ast.FunctionDef)}
    new_functions = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(guard).body if isinstance(node, ast.FunctionDef)}
    assert old_functions == new_functions
    # The prior allocation failed; use its saved terminal evidence, never an old RUNNING requirement.
    rows = [line.split('\t') for line in (RUN / 'slurm/submissions.tsv').read_text().splitlines() if line.split('\t')[0] == '6782816']
    assert (RUN / 'slurm/submitted_jobids_uva.txt').read_text().splitlines().count('6782816') == 1
    assert rows == [['6782816', 'monitor', OLD_COMMIT, str(OLD / 'monitor.sbatch')]]
    terminal_return = json.loads((OLD / 'actual_failure_terminal_accounting_return.json').read_text())
    terminal_raw = (OLD / 'actual_failure_terminal_accounting_stdout.raw').read_text()
    assert terminal_return['returncode'] == 0 and '--jobs=6782816' in terminal_return['command']
    terminal_rows = [row.split('|') for row in terminal_raw.splitlines()]
    for identity in ('6782816', '6782816.batch'):
        selected = [row for row in terminal_rows if row[0] == identity]
        assert len(selected) == 1 and selected[0][2:4] == ['FAILED', '1:0']
    partition_command = ['scontrol', 'show', 'partition', 'nolim', '-o']
    partition = subprocess.run(partition_command, capture_output=True, text=True, check=True)
    partition_fields = dict(token.split('=', 1) for token in partition.stdout.split() if '=' in token)
    maximum_seconds = duration_seconds(partition_fields['MaxTime'])
    requested_seconds = duration_seconds('7-00:00:00')
    assert partition_fields['State'] == 'UP' and (maximum_seconds is None or requested_seconds <= maximum_seconds)
    duration_proof = {'actual_partition_max_time': partition_fields['MaxTime'], 'actual_partition_max_seconds': maximum_seconds,
        'requested_time_limit': '7-00:00:00', 'requested_seconds': requested_seconds, 'within_actual_cap': True,
        'command': partition_command, 'returncode': partition.returncode, 'stdout': partition.stdout, 'stderr': partition.stderr}
    document(PACKET / 'duration_cap_proof.json', duration_proof)
    document(PACKET / 'owned_failed_monitor_resource_basis.json', {'schema_version': 1,
        'kind': 'owned_failed_cpu_monitor_terminal_and_historical_resource_basis', 'job_id': '6782816',
        'metadata_host': socket.gethostname(), 'metadata_allocation': None, 'old_allocation_usable': False,
        'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'prior_monitor_source': OLD_COMMIT,
        'terminal_query': receipt(OLD / 'actual_failure_terminal_accounting_return.json'),
        'terminal_stdout': receipt(OLD / 'actual_failure_terminal_accounting_stdout.raw'),
        'terminal_stderr': receipt(OLD / 'actual_failure_terminal_accounting_stderr.raw'),
        'historical_runtime': receipt(OLD / 'runtime_allocation.json'),
        'actual_current_old_allocation_validation': False, 'new_allocation_identity': None,
        'runtime_guard_functions_unchanged': True, 'actual_new_runtime_guard_required_after_start': True})
    source_validation = {}
    for name in ('monitor_host_hostname_author_6782816_48.json', 'monitor_host_hostname_independent_6782816_49.json'):
        raw = git('show', COMMIT + ':.review_metadata/' + name)
        value = json.loads(raw)
        assert value['passed'] is True and value['tests'] == 23 and value['live_scheduler_commands'] == 0
        assert value['gpu_library_calls'] == value['network_actions'] == 0 and value['numerical_import_attempts'] == []
        assert value['source_before'] == value['source_after'] and value['protected_before'] == value['protected_after']
        write(PACKET / name, raw)
        source_validation[name] = receipt(PACKET / name)
    canonical = json.loads((RUN / 'next_actions.json').read_text())
    cron = canonical['monitoring_session_cron']
    assert cron['id'] == 'ca8af776-26dd-4ee8-8c00-45ddc8bbc0cd' and cron['cron'] == '*/20 * * * *'
    assert cron['exact_prompt_and_schedule_verified'] is True
    document(PACKET / 'native_cron_recovery_basis.json', {'schema_version': 1,
        'kind': 'frozen_root_verified_native_cron_recovery_basis', 'root_reported_current_task': cron,
        'captured_at_utc': datetime.now(timezone.utc).isoformat(), 'source_path': str(RUN / 'next_actions.json'),
        'source_sha256_at_capture': receipt(RUN / 'next_actions.json')['sha256'],
        'actual_task_tool_queried_by_this_preparation': False, 'new_monitor_job_id': None})
    document(PACKET / 'recovery_basis.json', {'schema_version': 1, 'kind': 'explicit_persistent_cpu_monitor_recovery_basis',
        'source_commit': COMMIT, 'old_failed_job_id': '6782816', 'unreleased_superseded_job_id': '6782952',
        'source_validation': source_validation, 'native_cron': receipt(PACKET / 'native_cron_recovery_basis.json'),
        'old_terminal_evidence': receipt(PACKET / 'owned_failed_monitor_resource_basis.json'),
        'prelaunch_full_candidate_available': False, 'prelaunch_full_candidate_executed': False,
        'prelaunch_actual_full_report_available': False, 'old_monitor_running_required': False,
        'actual_first_full_report_required_after_start': True, 'actual_source_and_runtime_review_required_after_start': True,
        'future_work_blocked_until_first_full_report_independent_review': True,
        'scientific_credit': 0, 'worker_admitted': False})
    monitor = source / '10_jax_exploration_platform/scripts/atari_pixel_campaign/monitor.py'
    shape = {'account': 'sl5nw_base', 'partition': 'nolim', 'qos': 'normal', 'nodes': 1, 'ntasks': 1,
        'cpus_per_task': 1, 'host_memory_gib': 2, 'gpus_per_node': 0, 'time': '7-00:00:00',
        'interval_seconds': 1200, 'requeue': False}
    loop = ['/usr/bin/python3', '-B', str(monitor), '--run-dir', str(RUN), '--loop', '--interval', '1200']
    held = ['sbatch', '--parsable', '--hold', '--no-requeue', '--account=sl5nw_base', '--partition=nolim', '--qos=normal',
        '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=2G', '--gpus-per-node=0', '--time=7-00:00:00',
        '--job-name=pixel_study_monitor', '--output=' + str(RUN / 'slurm/jobs/monitor_%j.log'), str(PACKET / 'monitor.sbatch')]
    cpu_policy = {'requested_cpus': 1, 'accepted_allocated_logical_cpus': [1, 2],
        'two_requires_actual_node_threads_per_core': 2, 'requested_and_allocated_nodes': 1,
        'requested_and_allocated_memory': '2G', 'requested_and_allocated_gpus': 0}
    config = {'campaign_run_dir': str(RUN), 'source_commit': COMMIT, 'old_monitor_job_id': '6782816',
        'job_id': None, 'gpu_uuid': None, 'reservation': None, **shape, 'python': '/usr/bin/python3',
        'monitor_command': loop, 'cpu_allocation_policy': cpu_policy, 'component_only': True, 'duration_cap_proof': receipt(PACKET / 'duration_cap_proof.json'), 'supports_corrected_a4500_readers': True, 'supports_reviewed_direct_selected_and_profile_readers': True, 'direct_final_capacity_scaling_portable_supported': False,
        'scientific_case_credit': 0, 'scientific_hypothesis_credit': 0, 'final_training_credit': 0, 'worker_admission': False}
    document(PACKET / 'config_resolved.yaml', config)
    background = (PACKET / 'experiment_background.md').read_bytes()
    write(PACKET / 'command.txt', (shlex.join(held) + '\n').encode())
    expected = {name: hashlib.sha256((PACKET / name).read_bytes()).hexdigest()
                for name in ('source_snapshot.json', 'config_resolved.yaml', 'allocation_guard.py', 'source_git_archive.tar')}
    verifier = (OLD / 'verify_source.py').read_text()
    verifier = re.sub(r'^PACKET=.*$', 'PACKET=Path(' + repr(str(PACKET)) + ')', verifier, flags=re.M)
    verifier = re.sub(r'^EXPECTED=.*$', 'EXPECTED=' + repr(expected), verifier, flags=re.M)
    write(PACKET / 'verify_source.py', verifier.encode())
    verification_hash = hashlib.sha256(verifier.encode()).hexdigest()
    shell = ('#!/bin/bash\nset -euo pipefail\nunset PYTHONPATH PYTHONHOME\n'
        'export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1\n'
        "printf '%s  %s\\n' " + shlex.quote(verification_hash) + ' ' + shlex.quote(str(PACKET / 'verify_source.py')) + ' | sha256sum --check --status\n'
        + shlex.join(['/usr/bin/python3', '-B', str(PACKET / 'verify_source.py')]) + '\n'
        + shlex.join(['/usr/bin/python3', '-B', str(PACKET / 'allocation_guard.py')]) + '\nexec ' + shlex.join(loop) + '\n')
    write(PACKET / 'monitor.sbatch', shell.encode(), 0o755)
    plan = {'schema_version': 1, 'kind': 'persistent_pixel_study_monitor_replacement',
        'state': 'static recovery from published source; no prelaunch full candidate; not submitted', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'campaign_run_dir': str(RUN), 'helper_commit': COMMIT, 'source': str(source / '10_jax_exploration_platform'),
        'source_manifest': receipt(PACKET / 'source_snapshot.json'), 'source_file_count': len(entries), 'job_shape': shape,
        'job_id': None, 'gpu_uuid': None, 'reservation': None, 'data_unit': {'name': UNIT.name, 'path': str(UNIT)},
        'job_script': receipt(PACKET / 'monitor.sbatch'), 'submission_command': held,
        'old_monitor_job_id': '6782816', 'cadence_seconds': 1200, 'cpu_allocation_policy': cpu_policy,
        'supports_corrected_a4500_readers': True, 'supports_reviewed_direct_selected_and_profile_readers': True, 'direct_final_capacity_scaling_portable_supported': False, 'duration_cap_proof': receipt(PACKET / 'duration_cap_proof.json'),
        'recovery_basis': receipt(PACKET / 'recovery_basis.json'), 'prelaunch_full_candidate_available': False, 'actual_first_full_report_review_required_after_start': True,
        'required_submission_row': ['<exact_returned_job_id>', 'monitor', COMMIT, str(PACKET / 'monitor.sbatch')],
        'resource_basis': {'current_monitor_runtime': receipt(OLD / 'runtime_allocation.json'),
            'current_monitor_plan': receipt(OLD / 'monitor_replacement_plan.json'),
            'historical_smt': json.loads((OLD / 'monitor_replacement_plan.json').read_text())['resource_basis']['historical_smt'],
            'failed_previous_monitor': receipt(PACKET / 'owned_failed_monitor_resource_basis.json')},
        'scientific_case_credit': 0, 'scientific_hypothesis_credit': 0, 'final_training_credit': 0, 'worker_admission': False}
    document(PACKET / 'monitor_replacement_plan.json', plan)
    manifest = {'schema_version': 1, 'kind': 'prepared_persistent_pixel_study_monitor_replacement',
        'source_commit': COMMIT, 'plan': receipt(PACKET / 'monitor_replacement_plan.json'), 'submission_command': held,
        'job_id': None, 'gpu_uuid': None, 'reservation': None, 'scientific_case_credit': 0,
        'scientific_hypothesis_credit': 0, 'final_training_credit': 0, 'worker_admission': False}
    document(PACKET / 'manifest.yaml', manifest)
    for name in ('experiment_background.md', 'manifest.yaml', 'config_resolved.yaml', 'command.txt'):
        write(UNIT / name, (PACKET / name).read_bytes())
    document(UNIT / 'progress.json', {'kind': 'persistent_cpu_monitor_replacement', 'state': 'planned',
        'job_id': None, 'source_commit': COMMIT, 'old_monitor_job_id': '6782816',
        'scientific_case_credit': 0, 'scientific_hypothesis_credit': 0, 'final_training_credit': 0, 'worker_admission': False})
    # The full unrelated archive may contain Python3.12 report syntax; the actual monitor closure remains Python3.10.
    campaign_source = source / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
    campaign_files = sorted(campaign_source.rglob('*.py'))
    for path in campaign_files:
        compile(path.read_bytes(), str(path), 'exec')
    full_check = "import json,sys;from pathlib import Path;p=Path(sys.argv[1]);m=json.loads(p.read_text());s=Path(m['source_root']);f=[s/n for n in m['files'] if n.endswith('.py')];[compile(q.read_bytes(),str(q),'exec') for q in f];assert not any(n.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari'} for n in sys.modules);print(json.dumps({'python':sys.version,'python_files_compiled':len(f),'numerical_imports':False}))"
    full_command = ['/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python', '-I', '-S', '-B', '-c', full_check, str(PACKET / 'source_snapshot.json')]
    checked = subprocess.run(full_command, capture_output=True, text=True)
    document(PACKET / 'full_archive_syntax.json', {'command': full_command, 'returncode': checked.returncode,
        'stdout': checked.stdout, 'stderr': checked.stderr})
    assert checked.returncode == 0
    assert subprocess.run(['bash', '-n', str(PACKET / 'monitor.sbatch')]).returncode == 0
    source_check = subprocess.run(['/usr/bin/python3', '-I', '-S', '-B', str(PACKET / 'verify_source.py')], capture_output=True, text=True)
    document(PACKET / 'owned_compute_source_visibility.json', {'metadata_host': 'portal05', 'metadata_allocation': None, 'compute_visibility_verified': False,
        'command': ['/usr/bin/python3', '-I', '-S', '-B', str(PACKET / 'verify_source.py')], 'returncode': source_check.returncode,
        'stdout': source_check.stdout, 'stderr': source_check.stderr})
    assert source_check.returncode == 0
    assert not git('status', '--porcelain', '--untracked-files=all').strip()
    document(PACKET / 'preparation_receipt.json', {'schema_version': 1,
        'kind': 'published_source_cpu_monitor_static_preparation', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'metadata_host': 'portal05', 'metadata_allocation': None, 'prelaunch_full_candidate_available': False, 'clean_detached_source_root': str(REPOSITORY),
        'source_commit': COMMIT, 'source_files': len(entries), 'archive': receipt(PACKET / 'source_git_archive.tar'),
        'source_manifest': receipt(PACKET / 'source_snapshot.json'), 'plan': receipt(PACKET / 'monitor_replacement_plan.json'),
        'manifest': receipt(PACKET / 'manifest.yaml'), 'job_script': receipt(PACKET / 'monitor.sbatch'),
        'full_archive_syntax': receipt(PACKET / 'full_archive_syntax.json'), 'actual_monitor_closure_python_3_10_syntax_files': len(campaign_files),
        'owned_compute_source_visibility': receipt(PACKET / 'owned_compute_source_visibility.json'),
        'unchanged_smt_guard_functions': True, 'old_monitor_to_preserve': '6782816',
        'source_syntax_checked': True, 'shell_syntax_checked': True, 'new_jobs_submitted': 0,
        'numerical_imports': False, 'candidate_full_tick_pending': False, 'actual_first_full_report_required_after_start': True, 'supports_reviewed_direct_selected_and_profile_readers': True, 'direct_final_capacity_scaling_portable_supported': False,
        'qualification_packet_modified': False, 'writes_outside_new_packet_and_data_unit': False, 'no_actual_report_or_monitor_tick_on_login': True})
    print(json.dumps({'packet': str(PACKET), 'unit': str(UNIT), 'files': len(entries),
                      'plan': receipt(PACKET / 'monitor_replacement_plan.json'), 'submitted': False}))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        document(PACKET / 'preparation_failure.json', {'owned_cpu_job': os.environ.get('SLURM_JOB_ID'),
            'step_id': os.environ.get('SLURM_STEP_ID'), 'error': str(error), 'traceback': traceback.format_exc(),
            'new_jobs_submitted': 0, 'qualification_packet_modified': False})
        raise
