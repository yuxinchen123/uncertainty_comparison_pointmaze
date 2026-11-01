"""Freeze the reviewed published monitor source and its proven CPU-only startup records without submission."""

import ast
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
REPOSITORY = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl')
COMMIT = 'fb3071f3772ed836144f5876dbdda1113a5339d5'
OLD_COMMIT = '4814459cb7befc872d8f0e30682c23f4aa79b625'
OLD = RUN / ('remote_setup/2026-11-01-06-13_pixel-study-monitor_source-commit-' + OLD_COMMIT +
    '_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0')
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


def main():
    """Create a complete detached source archive, unchanged resource guard and unresolved held-command plan."""
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
        'archive': receipt(PACKET / 'source_git_archive.tar'), 'working_tree_bytes_used': False})

    # Keep the proven SMT validation functions byte-identical; only this new source/script identity changes.
    guard = (OLD / 'allocation_guard.py').read_text().replace(str(OLD), str(PACKET)).replace(OLD_COMMIT, COMMIT)
    write(PACKET / 'allocation_guard.py', guard.encode())
    old_functions = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse((OLD / 'allocation_guard.py').read_text()).body if isinstance(node, ast.FunctionDef)}
    new_functions = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(guard).body if isinstance(node, ast.FunctionDef)}
    assert old_functions == new_functions
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
    config = {'campaign_run_dir': str(RUN), 'source_commit': COMMIT, 'old_monitor_job_id': '6781087',
        'job_id': None, 'gpu_uuid': None, 'reservation': None, **shape, 'python': '/usr/bin/python3',
        'monitor_command': loop, 'cpu_allocation_policy': cpu_policy, 'component_only': True,
        'scientific_case_credit': 0, 'scientific_hypothesis_credit': 0, 'final_training_credit': 0, 'worker_admission': False}
    document(PACKET / 'config_resolved.yaml', config)
    background = ('## Purpose\n\nPrepare the persistent twenty-minute monitor from the reviewed final-protocol source. '
        'Current CPU monitor6781087 remains active until a separately submitted replacement passes independent startup review.\n\n'
        '## Key hyperparameters\n\nOne requested CPU, two GiB host memory, zero GPUs, seven days, partition nolim, '
        'account sl5nw_base, normal quality of service and1200-second cadence. One allocated logical CPU is accepted; '
        'two require the actual two-thread-per-core topology and consistent resources. Submit held with no automatic requeue.\n\n'
        '## Code and config changes\n\nBefore: the current frozen monitor predates the integrated final-protocol readers. '
        'After: this packet contains every committed src/scripts file from source' + COMMIT + '. '
        'The proven CPU allocation guard functions are unchanged. Initialization removes inherited Python search-path overrides. '
        'A candidate full tick runs only on the current owned CPU allocation and preserves exact raw report evidence. '
        'Canonical next_actions and provenance are protected; this preparation submits, releases and cancels no job.\n\n'
        '## Git state\n\nCommitted source:' + COMMIT + '. Exact Git archive and all blob/SHA256/mode records are retained. '
        'No uncommitted tiny-component source is included. The direct GPU worker and numerical source remain unchanged. '
        'Scientific case, idea and final-training credit are zero; no worker admission is granted.\n').encode()
    write(PACKET / 'experiment_background.md', background)
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
        'state': 'prepared from published source; not submitted', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'campaign_run_dir': str(RUN), 'helper_commit': COMMIT, 'source': str(source / '10_jax_exploration_platform'),
        'source_manifest': receipt(PACKET / 'source_snapshot.json'), 'source_file_count': len(entries), 'job_shape': shape,
        'job_id': None, 'gpu_uuid': None, 'reservation': None, 'data_unit': {'name': UNIT.name, 'path': str(UNIT)},
        'job_script': receipt(PACKET / 'monitor.sbatch'), 'submission_command': held,
        'old_monitor_job_id': '6781087', 'cadence_seconds': 1200, 'cpu_allocation_policy': cpu_policy,
        'required_submission_row': ['<exact_returned_job_id>', 'monitor', COMMIT, str(PACKET / 'monitor.sbatch')],
        'resource_basis': {'current_monitor_runtime': receipt(OLD / 'runtime_allocation.json'),
            'current_monitor_plan': receipt(OLD / 'monitor_replacement_plan.json'),
            'historical_smt': receipt(OLD / 'smt_resource_basis.json')},
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
        'job_id': None, 'source_commit': COMMIT, 'old_monitor_job_id': '6781087',
        'scientific_case_credit': 0, 'scientific_hypothesis_credit': 0, 'final_training_credit': 0, 'worker_admission': False})
    for path in source.rglob('*.py'):
        compile(path.read_bytes(), str(path), 'exec')
    assert subprocess.run(['bash', '-n', str(PACKET / 'monitor.sbatch')]).returncode == 0
    document(PACKET / 'preparation_receipt.json', {'schema_version': 1,
        'kind': 'published_source_cpu_monitor_static_preparation', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': COMMIT, 'source_files': len(entries), 'archive': receipt(PACKET / 'source_git_archive.tar'),
        'plan': receipt(PACKET / 'monitor_replacement_plan.json'), 'manifest': receipt(PACKET / 'manifest.yaml'),
        'job_script': receipt(PACKET / 'monitor.sbatch'), 'unchanged_smt_guard_functions': True,
        'old_monitor_to_preserve': '6781087', 'source_syntax_checked': True, 'shell_syntax_checked': True,
        'new_jobs_submitted': 0, 'numerical_imports': False, 'candidate_full_tick_pending': True,
        'writes_outside_new_packet_and_data_unit': False})
    print(json.dumps({'packet': str(PACKET), 'unit': str(UNIT), 'files': len(entries),
                      'plan': receipt(PACKET / 'monitor_replacement_plan.json'), 'submitted': False}))


if __name__ == '__main__':
    main()
