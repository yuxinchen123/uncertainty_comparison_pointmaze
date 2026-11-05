"""Prepare, verify and bind fresh common timing metadata; never launch a process on a remote host."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import signal

SCRIPTS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('common_timing_controller_metadata', SCRIPTS / 'direct_common_timing_controller.py')
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)
PREFIX = '10_jax_exploration_platform/scripts/atari_pixel_campaign/'


def setup(packet):
    """Bind metadata-only controller functions to the explicit fresh packet, never a guessed process."""
    packet = Path(packet).resolve(strict=True)
    request = control.document(packet / 'preparation_request.json')
    run, unit = Path(request['campaign_run_dir']), Path(request['data_unit'])
    control.require(packet == run / 'remote_setup' / request['unit_id'] and unit == run / 'data' / request['unit_id']
        and request['packet'] == str(packet) and type(request['schema_version']) is int and request['schema_version'] == 1
        and request['kind'] == 'direct_common_timing_preparation_request'
        and type(request['serial_order']) is int and request['serial_order'] in (1, 2),
        'only a fresh explicit common timing preparation request is supported')
    control.PACKET, control.RUN, control.UNIT = packet, run, unit
    control.timing_copy_order(request)
    control.no_numerical_modules()
    return request, packet, unit


def checked_reference(value):
    """Require exact existing ordinary bytes for every supplied accepted input."""
    control.require(type(value) is dict and set(value) == {'path', 'sha256'}
        and control.receipt(value['path']) == value, 'an actual accepted preparation reference changed or is absent')
    return control.document(value['path'])


def write(path, value):
    """Create one durable metadata record without replacing a previous preparation or validation."""
    control.publish(Path(path), control.encoded(value))


def preparation_command(request, packet, unit):
    """Build the exact twenty-nine-helper metadata preparer command from actual reviewed references."""
    root = Path(request['helper_source_root']).resolve(strict=True)
    control.require(request['helper_source_commit'] == subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, timeout=30).strip()
        and not subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain=v1'], text=True, timeout=30),
        'use the exact clean committed twenty-nine-helper source checkout')
    for value in (*request['accepted_evidence_snapshot_inputs'].values(), *request['direct_qualification_inputs'].values(),
            *request['common_correctness_inputs'].values(), request['qualification_plan']):
        checked_reference(value)
    args = [sys.executable, '-I', '-S', '-B', str(root / PREFIX / 'measure_selected.py'), '--prepare-only',
        '--mode', 'confirmation', '--source-root', request['numerical_source_root'], '--source-commit', control.SOURCE_COMMIT,
        '--configuration', str(packet / 'candidate_configuration.json'), '--reference-configuration', str(packet / 'reference_configuration.json'),
        '--runtime-environment', str(packet / 'runtime_environment.json'), '--asset-record', str(packet / 'asset_record.json'),
        '--run-dir', request['campaign_run_dir'], '--unit-id', request['unit_id'], '--gpu-uuid', control.GPU_UUID,
        '--hardware-class', 'h100', '--seed-start', '1', '--rounds', '7', '--minimum-updates-per-round', '10',
        '--direct-binding-profile', str(packet / 'direct_binding_profile.json'), '--direct-backend-image-record', str(control.BACKEND_PREPARATION_RECORD),
        '--direct-history-inputs', str(unit / 'direct_history/inputs.json'), '--python', request['python']]
    options = {'direct-qualification-record': request['direct_qualification_inputs']['result'],
        'direct-qualification-review': request['direct_qualification_inputs']['review'],
        'common-correctness-record': request['common_correctness_inputs']['result'],
        'common-correctness-review': request['common_correctness_inputs']['review'],
        'accepted-evidence-snapshot': request['accepted_evidence_snapshot_inputs']['proof'],
        'accepted-evidence-snapshot-review': request['accepted_evidence_snapshot_inputs']['review']}
    for name, ref in options.items():args.extend(['--' + name, ref['path']])
    return args


def prepare(request_path):
    """Run one bounded metadata preparation with actual existing snapshot, correctness and predecessor records."""
    request, packet, unit = setup(Path(request_path).parent)
    control.require(Path(request_path) == packet / 'preparation_request.json' and not unit.exists(),
        'never reuse an existing or attempted timing unit')
    for name, value in [('reference_configuration.json', request['reference_configuration']),
            ('candidate_configuration.json', request['configuration']), ('runtime_environment.json', request['runtime_environment']),
            ('asset_record.json', request['asset_record']), ('direct_binding_profile.json', request['direct_binding_profile'])]:
        write(packet / name, value)
    command = preparation_command(request, packet, unit)
    result = subprocess.run(command, capture_output=True, timeout=240, check=False)
    control.publish(packet / 'actual_preparation_stdout.raw', result.stdout)
    control.publish(packet / 'actual_preparation_stderr.raw', result.stderr)
    write(packet / 'actual_preparation_return.json', {'command': command, 'returncode': result.returncode,
        'maximum_seconds': 240, 'numerical_operations': 0, 'gpu_operations': 0})
    control.require(result.returncode == 0, 'metadata preparation failed; preserve its actual output and do not reuse this unit')
    plan = control.document(unit / 'measurement_plan.json')
    control.timing_snapshot(plan, request)
    # Canonical new timing sources use legacy staged names solely for the unchanged publication function.
    mapping = {}
    for source, destination in [(SCRIPTS / 'direct_common_timing_controller.py', packet / 'direct_qualification_controller.py'),
            (Path(__file__), packet / 'prepare_common_timing.py'),
            (SCRIPTS / 'tests/test_direct_common_timing_controller.py', packet / 'test_direct_qualification_controller.py')]:
        raw = control.ordinary_bytes(source);control.publish(destination, raw)
        mapping[str(source)] = {'staged': str(destination), 'sha256': control.digest(raw)}
    canonical = Path(__file__).resolve(strict=True)
    phases = {'prepare': [sys.executable, '-I', '-S', '-B', str(canonical), 'prepare', '--request', str(request_path)],
        **{name: [sys.executable, '-I', '-S', '-B', str(canonical), name, '--packet', str(packet)]
            for name in ('verify', 'bind')}}
    write(packet / 'canonical_controller_source_mapping.json', {'kind': 'common_timing_canonical_to_legacy_staged_source_mapping',
        'legacy_names_admit_no_correctness_operations': True, 'files': mapping,
        'copied_preparer_is_archival_only': True, 'canonical_preparer': control.receipt(canonical),
        'all_metadata_phase_commands': phases})
    write(packet / 'metadata_preparation_result.json', {'kind': 'direct_common_timing_metadata_preparation_result',
        'passed': True, 'plan': control.receipt(unit / 'measurement_plan.json'), 'helper_commit': request['helper_source_commit'],
        'helper_count': 29, 'actual_process_ids': None, 'gpu_operations': 0, 'scientific_credit': 0})
    write(packet / 'manifest.yaml', request);write(packet / 'config_resolved.yaml', {'reference': request['reference_configuration'],
        'candidate': request['configuration'], 'copy_seeds': request['copy_seeds'], 'rounds': 7, 'complete_updates_per_arm': 10})
    control.publish(packet / 'command.txt', (shlex.join(command) + '\n').encode())
    control.publish(packet / 'experiment_background.md', ("## Purpose\n\nMeasure common pixel PPO plus Random Network Distillation throughput on approved serval02 card 1.\n\n"
        "## Key hyperparameters\n\nSeven alternating pairs of ten complete updates per arm; 128 environments, 128 rollout steps, four epochs, four minibatches and fifty normalization warmups per independent copy. Original two-billion-transition annealing horizon and seeds remain unchanged.\n\n"
        "## Code and config changes\n\nOnly room background caching and final-two-frame rendering differ. Historical evidence uses an actual complete CPU snapshot and independent review. The controller holds no GPU standing descriptor.\n\n"
        "## Git state\n\nThe exact numerical and helper commits and all actual input hashes appear in the immutable measurement plan. No numerical or GPU work occurs during metadata preparation.\n").encode())
    return control.receipt(unit / 'measurement_plan.json')


def verify(packet):
    """Check the actual staged metadata child separately within a single bounded validation phase."""
    request, packet, unit = setup(packet)
    plan = control.document(unit / 'measurement_plan.json')
    selected = control.timing_snapshot(plan, request)
    support = control.timing_support(plan, unit, 'direct_common_confirmation')
    command = ['/usr/bin/python3', '-I', '-S', '-B', str(unit / 'code/direct_common_confirmation.py'), '--plan-stdin']
    raw = (unit / 'measurement_plan.json').read_bytes()
    result = subprocess.run(command, input=raw, capture_output=True, timeout=180, check=False,
        env=support.metadata_environment(dict(__import__('os').environ)))
    control.publish(packet / 'cold_metadata_validation_stdout.raw', result.stdout)
    control.publish(packet / 'cold_metadata_validation_stderr.raw', result.stderr)
    write(packet / 'cold_metadata_validation_return.json', {'command': command, 'returncode': result.returncode,
        'stdout_sha256': control.digest(result.stdout), 'stderr_sha256': control.digest(result.stderr)})
    control.require(result.returncode == 0 and not result.stderr and len(result.stdout) <= support.LIMIT,
        'the actual cold staged timing prerequisite failed; preserve the complete metadata attempt')
    proof = json.loads(result.stdout)
    control.require(proof.get('passed') is True and proof.get('plan_sha256') == control.digest(raw)
        and proof.get('accepted_evidence_snapshot_inputs') == plan['accepted_evidence_snapshot_inputs']
        and proof.get('parent_pid') == __import__('os').getpid()
        and proof.get('numerical_imports') == [] and type(proof.get('gpu_operations')) is int and proof['gpu_operations'] == 0,
        'the cold metadata child returned another plan, parent or operation scope')
    write(packet / 'cold_metadata_validation.json', proof)
    # Current source/runtime/assets/backend still receive the ordinary live checks in the numerical producer.
    write(packet / 'shared_software_assets_staged_verification.json', {
        'kind': 'actual_direct_common_timing_shared_metadata_and_staged_verification', 'passed': True,
        'plan': control.receipt(unit / 'measurement_plan.json'), 'helper_count': 29,
        'accepted_snapshot': request['accepted_evidence_snapshot_inputs'],
        'current_live_source_runtime_assets_backend_cuda_checks_deferred_to_unchanged_producer': True,
        'prior_gpu_raw_cases_reexecuted': False, 'metadata_cold_child': control.receipt(packet / 'cold_metadata_validation.json'),
        'numerical_imports': [], 'gpu_operations': 0, 'scientific_credit': 0, 'admission_granted': False})
    return proof


def bind(packet):
    """Bind only actual completed preparation, cold metadata proof and exact predecessor exits."""
    request, packet, unit = setup(packet)
    plan = control.document(unit / 'measurement_plan.json')
    control.timing_snapshot(plan, request)
    control.check_serial_predecessor_inputs(request, control.receipt(packet / 'serial_predecessor_exit_inputs.json'))
    proof = control.document(packet / 'cold_metadata_validation.json')
    control.require(proof.get('passed') is True and proof.get('kind') == 'reviewed_accepted_direct_common_correctness_validation'
        and proof.get('plan_sha256') == control.receipt(unit / 'measurement_plan.json')['sha256']
        and proof.get('accepted_evidence_snapshot_inputs') == plan['accepted_evidence_snapshot_inputs']
        and proof.get('numerical_imports') == [] and proof.get('gpu_operations') == 0,
        'binding requires the actual successful cold child for this exact fresh plan')
    freeze = {'schema_version': 1, 'kind': 'actual_direct_common_timing_packet_metadata_freeze',
        'files': {str(path): control.digest(control.ordinary_bytes(path)) for path in control.packet_metadata_paths(plan)}}
    write(packet / 'packet_metadata_freeze.json', freeze)
    names = {'plan': unit / 'measurement_plan.json', 'preparation': packet / 'metadata_preparation_result.json',
        'software': packet / 'shared_software_assets_staged_verification.json', 'packet_freeze': packet / 'packet_metadata_freeze.json',
        'request': packet / 'preparation_request.json', 'serial_predecessor_exit': packet / 'serial_predecessor_exit_inputs.json'}
    write(packet / 'controller_bound_metadata.json', {'schema_version': 1,
        'kind': 'actual_prepared_direct_common_timing_controller_inputs', **{name: control.receipt(path) for name, path in names.items()}})
    control.checked_inputs()
    return control.receipt(packet / 'controller_bound_metadata.json')


def stop_metadata_phase(signum, frame):
    """Stop an overlong CPU metadata phase while its subprocess context reaps its own child."""
    raise TimeoutError('the metadata phase exceeded its 240-second work allowance; retain failure and cleanup')


def main():
    """Expose three separate bounded metadata phases and no launch or remote transport option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'verify', 'bind'))
    parser.add_argument('--request', type=Path)
    parser.add_argument('--packet', type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, stop_metadata_phase)
    signal.setitimer(signal.ITIMER_REAL, 240)
    if args.operation == 'prepare':
        control.require(args.request is not None and args.packet is None, 'prepare requires only one exact request path')
        result = prepare(args.request)
    else:
        control.require(args.packet is not None and args.request is None, 'verification and binding require one exact packet')
        result = verify(args.packet) if args.operation == 'verify' else bind(args.packet)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({'kind': 'direct_common_timing_metadata_phase', 'operation': args.operation,
        'result': result, 'jobs_submitted': 0, 'remote_actions': 0, 'gpu_operations': 0}))


if __name__ == '__main__':
    main()
