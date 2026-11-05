"""Derive one concrete two-copy zero-credit profile qualification request from actual reviewed metadata.

This operational utility neither prepares the runtime unit nor observes software,
launches a process, submits/releases a job, or generates an independent review.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import re

RUNTIME = 'd1469fd063a44167ec56760729eb0beea7ecf7d3'
PREFIX = '10_jax_exploration_platform/scripts/atari_pixel_campaign/'
SOURCE = 'd28e6735e79788f618acb0ebfde8609db1f776af'


def require(value, message):
    """Refuse absent actual inputs before writing a packet."""
    if not value:raise ValueError(message)


def runtime(root):
    """Load only the actual immutable published runtime helper snapshot."""
    root = Path(root).resolve(strict=True)
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == RUNTIME, 'select a dedicated clean runtime checkout at the exact reviewed commit')
    path = root / PREFIX / 'profile_target_qualification.py'
    raw = path.read_bytes()
    committed = subprocess.check_output(['git', '-C', str(root), 'show', RUNTIME + ':' + PREFIX + path.name])
    require(raw == committed, 'the runtime qualification helper differs from its reviewed committed bytes')
    spec = importlib.util.spec_from_file_location('operational_exact_profile_qualification', path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def derive(args):
    """Freeze actual two-copy inputs after the complete fact and genuine review exist."""
    q = runtime(args.helper_root)
    run = args.run_dir.resolve(strict=True)
    require(args.hardware_class in ('h100', 'jaguar03_rtx_a4500'), 'use the exact declared hardware spelling')
    packet = args.packet.absolute()
    require(packet.parent == run / 'remote_setup' and packet.parent.resolve(strict=True) == packet.parent
        and not packet.exists() and not packet.is_symlink(), 'write a fresh exact campaign packet')
    require(packet.name == args.profile_unit_id, 'the actual packet name must equal the profile capture unit name')
    require(all(Path(value).name == value and value not in ('', '.', '..') and len(value.encode()) <= 240
        for value in (args.controller_unit_id, args.profile_unit_id))
        and args.controller_unit_id != args.profile_unit_id, 'select distinct full actual unit names')
    require(not any((run / 'data' / value).exists() for value in (args.controller_unit_id, args.profile_unit_id)),
        'neither requested runtime unit may already exist')
    proof_ref, review_ref = q.reference(args.common_proof), q.reference(args.common_review)
    proof = q.checked(proof_ref)
    entries = [entry for entry in proof.get('contexts', []) if entry.get('hardware_class') == args.hardware_class and entry.get('copies') == 2]
    require(len(entries) == 1 and type(entries[0].get('copies')) is int, 'the actual accepted two-copy context is missing')
    accepted = q.checked(entries[0]['review'])
    common = q.checked(accepted['plan'])
    require(common['campaign_run_dir'] == str(run) and common['hardware_class'] == args.hardware_class and common['copy_seeds'] == [1, 2]
        and all(type(seed) is int for seed in common['copy_seeds'])
        and type(common['configuration']['learner']['n_copies']) is int
        and common['configuration']['learner']['n_copies'] == 2, 'the selected actual common context differs')
    configured = args.hardware_class == 'jaguar03_rtx_a4500'
    profile = q.adjacent({'helpers': {'files': {PREFIX + 'profile_selected.py': hashlib.sha256(
        (args.helper_root / PREFIX / 'profile_selected.py').read_bytes()).hexdigest()}}}, 'profile_selected')
    helper = profile.helper_receipt(configured=configured, direct=not configured, owned_processes=configured,
        direct_profile=not configured, configured_native=configured, configured_prerequisites=configured, profile_target=True)
    proposed = {**common, 'helpers': helper, 'inputs': {**common['inputs'], 'accepted_confirmation': accepted['result']}}
    validated = q.require_four_contexts(proposed, proof_ref, review_ref)
    metadata = q.adjacent(proposed, 'configured_profile_prerequisites')
    files = metadata.MetadataFiles()
    historical = next(row['historical_plan'] for row in validated['contexts'] if row['hardware_class'] == 'h100' and row['copies'] == 1)
    snapshot = metadata.initial_snapshot(files, proof['initial_snapshot_inputs'], historical)
    registry = snapshot['historical_registry']
    qualification = common['target_qualification_inputs' if configured else 'direct_qualification_inputs']
    original_path = Path(qualification['result']['path']).parent / 'measurement_plan.json'
    original_ref = q.reference(original_path);original = q.checked(original_ref)
    require(original['source_commit'] == SOURCE and original['python'] == common['python']
        and type(original['configuration']['learner']['n_copies']) is int
        and original['configuration']['learner']['n_copies'] == 1
        and original['copy_seeds'] == [1] and type(original['copy_seeds'][0]) is int
        and q.same(original['configuration'], original['reference_configuration']),
        'the accepted full original qualification must retain its source, selected Python and one reference copy')
    native_result_ref, native_review_ref = q.reference(args.component_result), q.reference(args.component_review)
    component = q.checked(native_result_ref)
    component_review = q.checked(native_review_ref)
    component_plan = q.checked(component['plan'])
    require(component_plan['python'] == common['python'] and component_review.get('approved') is True
        and component_review.get('passed') is True and component_review.get('result') == native_result_ref,
        'the actual independently accepted same-target native component differs')
    expected_review = '4264288c209606dccc15c883724ffeccaf5a480c943c5e29dd9b1c409c814da5' if configured else '3f4173e5b8d52624e10f90e66095df0c2c0587ecc4beb7b07c8c2874709ae788'
    require(native_review_ref['sha256'] == expected_review, 'reuse only the actual accepted same-target process component')
    backend_ref = q.reference(args.backend_record)
    backend = q.adjacent(proposed, 'profile_qualification_backend')
    backend.validate_expected(q.checked(backend_ref), common['python'])
    profile_input = common['inputs']['configured_isolation_profile' if configured else 'direct_binding_profile']
    actual_profile = q.checked(profile_input)
    require(actual_profile == common['configured_isolation' if configured else 'direct_binding']['profile'],
        'the immutable explicit profile input differs from accepted common performance')
    baseline = {'source_commit': SOURCE, 'hardware_class': args.hardware_class,
        'runtime_environment': common['runtime_environment'], 'confirmation_record': accepted['result']['path'],
        'configured_isolation_profile' if configured else 'direct_binding_profile': actual_profile}
    argv = ['--prepare-only', '--source-root', common['source_root'], '--source-commit', SOURCE,
        '--run-dir', str(run), '--unit-id', args.profile_unit_id, '--configuration', str(packet / 'configuration.json'),
        '--runtime-environment', str(packet / 'runtime_environment.json'), '--asset-record', str(packet / 'asset_record.json'),
        '--study-registry', registry['path'], '--accepted-baseline-specification', str(packet / 'accepted_baseline.json'),
        '--profile-tool-record', str(packet / 'profile_tool_record.json'), '--hardware-class', args.hardware_class,
        '--seed-start', '1', '--minimum-updates', '1', '--maximum-active-seconds', '10200', '--python', common['python']]
    if common.get('jaxtari_source_root') is not None:argv += ['--jaxtari-source-root', common['jaxtari_source_root']]
    if configured:
        argv += ['--configured-isolation-profile', profile_input['path'],
            '--configured-native-process-qualification', native_result_ref['path'], '--configured-native-process-review', native_review_ref['path'],
            '--configured-profile-prerequisite-proof', proof_ref['path'], '--configured-profile-prerequisite-review', review_ref['path'],
            '--target-qualification-record', qualification['result']['path'], '--target-qualification-review', qualification['review']['path']]
    else:
        argv += ['--gpu-uuid', 'GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3', '--direct-binding-profile', profile_input['path'],
            '--direct-history-inputs', str(run / 'data' / args.profile_unit_id / 'direct_history/inputs.json'),
            '--direct-backend-image-record', backend_ref['path'],
            '--direct-qualification-record', qualification['result']['path'], '--direct-qualification-review', qualification['review']['path'],
            '--direct-confirmation-review', entries[0]['review']['path'], '--profile-process-component-result', native_result_ref['path'],
            '--profile-process-component-review', native_review_ref['path']]
    request = {'kind': 'profile_target_qualification_preparation_request', 'protocol': q.PROTOCOL, 'source_commit': SOURCE,
        'profile_prepare_arguments': argv, 'controller_unit_id': args.controller_unit_id,
        'four_context_proof': proof_ref, 'four_context_review': review_ref, 'original_qualification_plan': original_ref,
        'backend_image_record': backend_ref, 'accepted_confirmation_sha256': accepted['result']['sha256'],
        'controller_maximum_seconds': 10800}
    if not configured:
        require(argv.count('--direct-backend-image-record') == 1
            and argv[argv.index('--direct-backend-image-record') + 1] == request['backend_image_record']['path'],
            'the parser presence argument must name the same fresh profile backend record')
    packet.mkdir()
    for name, value in (('configuration.json', common['configuration']), ('runtime_environment.json', common['runtime_environment']),
            ('asset_record.json', common['asset_record']), ('accepted_baseline.json', baseline),
            ('profile_tool_record.json', component_plan['profile_tool']), ('preparation_request.json', request)):
        q.publish(packet / name, value)
    q.publish(packet / 'manifest.yaml', {'schema_version': 1, 'kind': 'planned_full_original_profile_target_qualification',
        'description': 'One exact two-copy current-source profile target qualification; no ordinary or worker admission.',
        'runtime_helper_commit': RUNTIME, 'source_commit': SOURCE, 'request': q.reference(packet / 'preparation_request.json'),
        'actual_runtime_unit_prepared': False, 'job_id': None, 'controller_pid': None, **q.ZERO})
    q.publish(packet / 'config_resolved.yaml', {'configuration': common['configuration'], 'copy_seeds': [1, 2],
        'qualification_budget': q.QUALIFICATION_BUDGET, 'hardware_class': args.hardware_class, 'gpu_uuid': None if configured else 'GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3'})
    command = ['/usr/bin/python3', '-I', '-S', '-B', str(args.helper_root / PREFIX / 'profile_target_qualification.py'),
        '--request', str(packet / 'preparation_request.json')]
    import shlex
    q.publish(packet / 'command.txt', (shlex.join(command) + '\n').encode())
    q.publish(packet / 'experiment_background.md', (
        '## Purpose\n\nPrepare one separate full original profile target qualification. This packet has no scientific, profile or worker credit.\n\n'
        '## Key hyperparameters\n\nEach of two independent copies uses 128 environments, 128 rollout steps, 4 epochs, 4 minibatches and 50 normalization rollouts. '
        'One separate original reference copy completes one full update. The resource limit is three hours, with 10200 seconds for work and 600 seconds for cleanup.\n\n'
        '## Code and config changes\n\nThe unchanged captured profile is followed by the unchanged original tests, renderer and reference update. '
        'Current binding, backend bytes and all tool descendants require separate actual verification. No GPU action has happened.\n\n'
        f'## Git state\n\nNumerical source: `{SOURCE}`. Runtime helper source: `{RUNTIME}`. Operational source bytes: `{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}`.\n').encode())
    return {'packet': str(packet), 'request': q.reference(packet / 'preparation_request.json'),
        'runtime_preparation_command': command, 'actual_runtime_prepared': False, 'gpu_operations': 0, **q.ZERO}


def main():
    """Write actual metadata only after the caller supplies every genuine required record."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('helper-root', 'run-dir', 'packet', 'common-proof', 'common-review', 'component-result', 'component-review', 'backend-record'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--hardware-class', choices=('h100', 'jaguar03_rtx_a4500'), required=True)
    parser.add_argument('--controller-unit-id', required=True);parser.add_argument('--profile-unit-id', required=True)
    parser.add_argument('--owned-job-id', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'action_taken': False, 'actual_runtime_prepared': False, 'gpu_operations': 0}))
        return
    job, step = args.owned_job_id, os.environ.get('SLURM_STEP_ID', '')
    cgroup = Path('/proc/self/cgroup').read_text()
    require(re.fullmatch(r'[1-9][0-9]*', job) is not None and os.environ.get('SLURM_JOB_ID') == job
        and re.fullmatch(r'[0-9]+', step) is not None and ('/job_' + job + '/step_' + step + '/') in (cgroup + '/'),
        'derive metadata only in the declared exact owned CPU job and numeric step')
    require((args.run_dir / 'slurm/submitted_jobids_uva.txt').read_text().splitlines().count(job) == 1,
        'the exact CPU job must occur once in this campaign owned ledger')
    print(json.dumps(derive(args), sort_keys=True))


if __name__ == '__main__':main()
