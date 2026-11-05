"""Exercise tiny pixel learning and latency components with no formal evidence credit."""

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback


SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import device_memory_capture_regression as assignment
import device_memory_limit as ownership
import diagnose_gpu_execution as diagnostic
import measure_selected as measurement
import post_comparison_validation as validation
import short_learning
import train_optimized as training
import validate_training_deployment as deployment

KIND = short_learning.COMPONENT_KIND
SHAPE = {'hostname': 'jaguar03', 'nodes': 1, 'ntasks': 1, 'cpus_per_task': 8,
    'host_memory_gib': 64, 'time_limit_minutes': 30, 'gpu_count': 1,
    'gpu_type': 'nvidia_rtx_a4500', 'account': 'sl5nw_base', 'partition': 'gpu', 'qos': 'csresnolim'}
ZERO_CREDIT = {'formal_learning_credit': 0, 'case_credit': 0, 'idea_credit': 0,
    'packing_point_credit': 0, 'throughput_credit': 0, 'final_training_credit': 0, 'admission_granted': False}
PREFIX = '10_jax_exploration_platform/scripts/atari_pixel_campaign/'
FILENAME = 'learning_latency_component.py'
HELPERS = tuple(dict.fromkeys((*measurement.HELPERS, 'device_memory_capture_regression.py', 'short_learning.py', FILENAME)))


def write(path, value):
    """Keep original component evidence immutable, including every failed attempt."""
    measurement.write_once(path, measurement.json_bytes(value))


def read(path):
    """Read exact ordinary component artifacts with duplicate JSON keys rejected."""
    return ownership.json_document(assignment.binary(path))


def configurations(source):
    """Retain every frozen original control except the explicitly approved tiny dimensions and readout cases."""
    original = diagnostic.default_configuration(source)
    learner = {**original['learner'], **{name: value for name, value in short_learning.COMPONENT_RECIPE.items()
                                       if name != 'copy_seeds'}}
    environment = {**original['environment'], 'num_envs': 1}
    def variant(readout, group=1):
        """Build only valid ungrouped readout or grouped no-readout execution."""
        return {'learner': copy.deepcopy(learner), 'environment': copy.deepcopy(environment),
            'execution': {'compilation_mode': 'whole_iteration', 'iterations_per_call': group,
                          'report_readout': readout, 'report_every_iterations': 4}}
    values = {'reference': variant('scheduled'), 'candidate': variant('per_update'),
        'latency_cases': {'scheduled': {'reference': variant('scheduled'), 'candidate': variant('scheduled')},
            'per_update': {'reference': variant('per_update'), 'candidate': variant('per_update')},
            'grouped_no_readout': {'reference': variant('none'), 'candidate': variant('none', 3)}}}
    # Before: literal per-copy defaults such as (). After: identical JSON lists used by every prepared artifact.
    return json.loads(measurement.json_bytes(values))


def require_component_helpers(root):
    """Require the exact separate component declarations before staging or numerical imports."""
    tree = ast.parse((root / PREFIX / 'short_learning.py').read_bytes())
    declarations = {target.id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name) and target.id in
        {'COMPONENT_KIND', 'COMPONENT_CONTRACT', 'COMPONENT_RECIPE'}}
    expected = {name: getattr(short_learning, name) for name in ('COMPONENT_KIND', 'COMPONENT_CONTRACT', 'COMPONENT_RECIPE')}
    ownership.require(declarations == expected and {'run_component', 'component_protocol'} <=
        {node.name for node in tree.body if isinstance(node, ast.FunctionDef)},
        'the published helper lacks the explicit reviewed tiny component entry')


def prepare(args):
    """Freeze a distinct component plan, complete source closure and standard records without a guessed GPU/job."""
    run = args.run_dir.resolve(strict=True)
    ownership.require(args.unit_id and Path(args.unit_id).name == args.unit_id and args.unit_id not in {'.', '..'},
                      'plain component unit name required')
    source = deployment.git_root(str(args.source_root), args.source_commit, validation.Evidence())
    helpers_root = deployment.git_root(str(args.helper_root), args.helper_commit, validation.Evidence())
    require_component_helpers(helpers_root)
    values = configurations(source)
    helpers = training.source_receipt(helpers_root, args.helper_commit, [PREFIX + name for name in HELPERS])
    assets = read(args.asset_record)
    ownership.require(set(assets) == {'root', 'manifest', 'sha256'} and re.fullmatch(r'[0-9a-f]{64}', assets['sha256'])
        and all(Path(assets[name]).is_absolute() for name in ('root', 'manifest')), 'exact original asset identity required')
    plan = {'schema_version': 1, 'kind': KIND, 'component_recipe': short_learning.COMPONENT_RECIPE,
        'shape': SHAPE, 'source_root': str(source), 'source_commit': args.source_commit,
        'core': diagnostic.core_identity(source, args.source_commit), 'helpers': helpers,
        'paired_source': training.source_receipt(source, args.source_commit, [PREFIX + name for name in ('paired_benchmark.py', 'ideas.py')]),
        'campaign_run_dir': str(run), 'unit_id': args.unit_id, 'python': str(args.python.absolute()),
        'asset_record': assets, 'runtime_environment': validation.runtime_environment(read(args.runtime_environment)),
        'jaxtari_source_root': str(args.jaxtari_source_root.resolve(strict=True)), 'copy_seeds': [11],
        'reference_configuration': values['reference'], 'configuration': values['candidate'],
        'latency_cases': values['latency_cases'], 'learning_protocol': short_learning.component_protocol([values['reference'], values['candidate']]),
        'job_id': None, 'gpu_uuid': None, 'reservation': None,
        'owned_job_ledger': str(run / 'slurm/submitted_jobids_uva.txt'),
        'ownership_profile': ownership.UVA_CGROUP_V2_OWNERSHIP_PROFILE, **ZERO_CREDIT}
    unit = run / 'data' / args.unit_id
    command = [plan['python'], '-B', str(unit / 'code' / FILENAME), 'run', str(unit)]
    background = ('## Purpose\n\nExercise actual tiny JAXtari pixel learning and scheduled, per-update and grouped no-readout latency components. '
        'This uses the full frozen networks and original updates but produces zero formal learning, comparison, idea, packing, throughput or final-training credit.\n\n'
        '## Key hyperparameters\n\nOne copy, one environment, two rollout steps, one epoch and minibatch, zero normalization warmup, seed11 and state donation. '
        'The original2B annealing horizon and float32 networks/Adam, float64 statistics, highest precision and deterministic operations remain. '
        'Each learning arm performs at least128 full updates in complete reporting windows. Primary transitions and additional cloned work are reported separately.\n\n'
        '## Code and config changes\n\nUse the separate component-only learning entry. Scheduled and per-update readout are ungrouped; grouping three is tested only with no readout. '
        'No generic formal protocol override or source/profile admission is provided. Physical assignment must pass before JAX; unsupported mapping remains a saved failure.\n\n'
        '## Git state\n\nNumerical source: ' + args.source_commit + '. Component helper source: ' + args.helper_commit + '.\n').encode()
    for name, raw in {'component_plan.json': measurement.json_bytes(plan), 'manifest.yaml': measurement.json_bytes(plan),
            'config_resolved.yaml': measurement.json_bytes(values), 'command.txt': (__import__('shlex').join(command) + '\n').encode(),
            'experiment_background.md': background}.items():
        measurement.write_once(unit / name, raw)
    for name in HELPERS:
        measurement.write_once(unit / 'code' / name, (helpers_root / PREFIX / name).read_bytes())
    write(unit / 'component_prepared.json', {'plan': ownership.file_receipt(unit / 'component_plan.json')})
    return unit


def verify_static(unit, execution=False):
    """Require exact original numerical/default/helper bytes and the explicitly tiny zero-credit plan."""
    plan = read(unit / 'component_plan.json')
    ownership.require(type(plan.get('schema_version')) is int and plan['schema_version'] == 1 and plan['kind'] == KIND
        and plan['shape'] == SHAPE and unit == Path(plan['campaign_run_dir']) / 'data' / plan['unit_id']
        and all(plan[name] is None for name in ('job_id', 'gpu_uuid', 'reservation'))
        and all(type(plan[name]) is type(value) and plan[name] == value for name, value in ZERO_CREDIT.items())
        and plan['owned_job_ledger'] == str(Path(plan['campaign_run_dir']) / 'slurm/submitted_jobids_uva.txt'),
        'unsupported component plan, guessed identity or nonzero formal credit')
    short_learning.checked_component_protocol(plan)
    validation.runtime_environment(plan['runtime_environment'])
    ownership.require(set(plan['asset_record']) == {'root', 'manifest', 'sha256'}
        and re.fullmatch(r'[0-9a-f]{64}', plan['asset_record']['sha256'])
        and all(Path(plan['asset_record'][name]).is_absolute() for name in ('root', 'manifest')),
        'component asset identity changed')
    ownership.require(read(unit / 'component_prepared.json') == {'plan': ownership.file_receipt(unit / 'component_plan.json')},
                      'the original component preparation changed')
    source = deployment.git_root(plan['source_root'], plan['source_commit'], validation.Evidence())
    ownership.require(diagnostic.core_identity(source, plan['source_commit']) == plan['core'], 'the frozen numerical core changed')
    values = configurations(source)
    ownership.require(values == {'reference': plan['reference_configuration'], 'candidate': plan['configuration'],
                                  'latency_cases': plan['latency_cases']}, 'component configuration differs from the original tiny recipe')
    helpers = plan['helpers']
    root = deployment.git_root(helpers['source_root'], helpers['source_commit'], validation.Evidence())
    require_component_helpers(root)
    ownership.require(set(helpers['files']) == {PREFIX + name for name in HELPERS}
        and training.source_receipt(root, helpers['source_commit'], list(helpers['files'])) == helpers,
        'the exact component helper closure changed')
    for relative, digest in helpers['files'].items():
        ownership.require(training.file_sha256(unit / 'code' / Path(relative).name) == digest, 'a staged component helper changed')
    paired = plan['paired_source']
    ownership.require(paired == training.source_receipt(source, plan['source_commit'], [PREFIX + name for name in ('paired_benchmark.py', 'ideas.py')]),
                      'the frozen selected Executor source changed')
    if execution:
        for name, module in ((FILENAME, sys.modules[__name__]), ('short_learning.py', short_learning),
                ('device_memory_capture_regression.py', assignment), ('device_memory_limit.py', ownership),
                ('measure_selected.py', measurement), ('diagnose_gpu_execution.py', diagnostic),
                ('post_comparison_validation.py', validation), ('train_optimized.py', training),
                ('validate_training_deployment.py', deployment)):
            ownership.require(Path(module.__file__).absolute() == unit / 'code' / name, 'execute only the staged component closure: ' + name)
    return plan


def validate_job_shape(owned):
    """Bind the separately approved thirty-minute one-A4500 allocation without changing the memory diagnostic's shape."""
    raw = owned['job']['stdout']
    expected = {'Account': 'sl5nw_base', 'Partition': 'gpu', 'QOS': 'csresnolim', 'NumNodes': '1',
        'NumCPUs': '8', 'NumTasks': '1', 'CPUs/Task': '8', 'TimeLimit': '00:30:00', 'Requeue': '0',
        'TresPerNode': 'gres/gpu:nvidia_rtx_a4500:1'}
    ownership.require(all(ownership.scheduler_field(raw, name) == value for name, value in expected.items()),
                      'the tiny component allocation differs from its approved resources')
    pairs = [value.split('=', 1) for value in ownership.scheduler_field(raw, 'AllocTRES').split(',')]
    ownership.require(all(len(pair) == 2 for pair in pairs) and len({pair[0] for pair in pairs}) == len(pairs), 'duplicate component resource')
    resources = dict(pairs)
    ownership.require(resources.get('cpu') == '8' and resources.get('mem') == '64G' and resources.get('node') == '1'
        and resources.get('gres/gpu') == '1' and {key for key in resources if key.startswith('gres/gpu:')} <= {'gres/gpu:nvidia_rtx_a4500'}
        and all(value == '1' for key, value in resources.items() if key.startswith('gres/gpu:')),
        'the component does not own its exact processor, memory and A4500 allocation')
    requested = [value.split('=', 1) for value in ownership.scheduler_field(raw, 'ReqTRES').split(',')]
    ownership.require(all(len(pair) == 2 for pair in requested) and len({pair[0] for pair in requested}) == len(requested),
                      'duplicate component requested resource')
    requested = dict(requested)
    ownership.require(all(requested.get(name) == resources.get(name) for name in ('cpu', 'mem', 'node', 'gres/gpu'))
        and {key for key in requested if key.startswith('gres/gpu:')} <= {'gres/gpu:nvidia_rtx_a4500'}
        and all(value == '1' for key, value in requested.items() if key.startswith('gres/gpu:')),
        'component requested resources differ from the approved allocation')
    step_pairs = [value.split('=', 1) for value in ownership.scheduler_field(owned['step']['stdout'], 'TRES').split(',')]
    ownership.require(all(len(pair) == 2 for pair in step_pairs) and len({pair[0] for pair in step_pairs}) == len(step_pairs),
                      'duplicate component step resource')
    step = dict(step_pairs)
    ownership.require(all(step.get(name) == resources.get(name) for name in ('cpu', 'mem', 'node', 'gres/gpu')),
                      'component numeric step differs from its approved actual allocation')
    reservation = ownership.scheduler_field(raw, 'Reservation')
    ownership.require(re.fullmatch(r'sl5nw_[1-9][0-9]*', reservation), 'actual owned reservation required')
    return reservation


def admit(unit, plan):
    """Preserve actual own metadata and require the reviewed explicit physical File mapping before any JAX import."""
    process = ownership.process_identity()
    job, step = os.environ.get('SLURM_JOB_ID', ''), os.environ.get('SLURM_STEP_ID', '')
    ownership.require(process['hostname'] == 'jaguar03' and process['uid'] == 2618919
        and re.fullmatch(r'[1-9][0-9]*', job) and re.fullmatch(r'[0-9]+', step), 'an owned numeric jaguar03 allocation is required')
    ledger = assignment.binary(Path(plan['owned_job_ledger']))
    ownership.require(ledger.decode().splitlines().count(job) == 1, 'the exact returned job must already be in the canonical owned ledger')
    measurement.write_once(unit / 'owned_job_ledger_snapshot.txt', ledger)
    protocol = {**ownership.PROTOCOL, 'allocation_kind': 'slurm', 'cluster': 'uva',
                'owned_job_ledger': ownership.file_receipt(unit / 'owned_job_ledger_snapshot.txt')}
    runtime_plan = {**plan, 'job_id': job, 'device_memory_limit_protocol': protocol,
        'device_memory_limit_ledger_source': {'path': plan['owned_job_ledger'], 'sha256': ownership.file_receipt(unit / 'owned_job_ledger_snapshot.txt')['sha256']}}
    ownership.capture_v2_ownership(runtime_plan, process, job, step, unit / 'allocation_bootstrap_cgroup_v2.json')
    command = ['scontrol', 'show', 'job', '-d', job]
    query = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    environment = {name: os.environ[name] for name in ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS',
        'SLURM_GPUS_ON_NODE', 'SLURM_CPUS_PER_TASK', 'SLURM_CONF', 'CUDA_VISIBLE_DEVICES') if name in os.environ}
    write(unit / 'allocation_bootstrap.json', {'process': process, 'environment': environment,
        'query': {'command': command, 'returncode': query.returncode, 'stdout': query.stdout, 'stderr': query.stderr}})
    ownership.require(query.returncode == 0 and not query.stderr
        and ownership.scheduler_field(query.stdout, 'JobId') == job
        and ownership.scheduler_field(query.stdout, 'UserId') == 'sl5nw(2618919)'
        and ownership.scheduler_field(query.stdout, 'JobState') == 'RUNNING'
        and ownership.scheduler_field(query.stdout, 'NodeList') == 'jaguar03', 'owned running job query failed')
    resolution = {}
    try:
        # AutoDetect-only or untrusted configuration remains unsupported; no sorting, ordinal or minor fallback is added.
        gpu = assignment.resolve_uuid(query.stdout, environment, resolution)
    finally:
        write(unit / 'allocation_resolution.json', resolution)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    runtime_plan.update(gpu_uuid=gpu, reservation=ownership.scheduler_field(query.stdout, 'Reservation'), hardware_class='jaguar03_rtx_a4500')
    owned = ownership.recorded_ownership(runtime_plan, unit, 'allocation_owned_observation.json')
    validate_job_shape(owned)
    admission = {'kind': KIND + '_owned_allocation', 'process': process, 'job_id': job, 'step_id': step,
                 'gpu_uuid': gpu, 'reservation': runtime_plan['reservation'], 'runtime_plan': runtime_plan, 'ownership': owned, **ZERO_CREDIT}
    write(unit / 'allocation_admission.json', admission)
    return admission


class ComponentMemoryObservation(measurement.MemoryObservation):
    """Reuse selected-card memory queries while requiring a live sampler and normal requested completion."""

    def __init__(self, device, gpu_uuid, path):
        """Retain a unique raw observation path for every whole learning or latency stage."""
        super().__init__(device, gpu_uuid)
        self.path, self.normal_stop = path, False

    def begin(self):
        """Record normal-stop state before the inherited first sample and thread start."""
        self.normal_stop = False
        super().begin()

    def _sample(self):
        """Record ordinary sampler errors; an unexpected thread exit never becomes a normal requested stop."""
        while True:
            if self.stop.wait(.2):
                self.normal_stop = True
                return
            try:
                self.samples.append(self.sample())
            except Exception as error:
                self.errors.append(repr(error))
                return

    def finish(self):
        """Preserve all raw samples even when sampling, allocation accounting or competing-use checks fail."""
        alive = self.thread is not None and self.thread.is_alive()
        result, error = None, None
        try:
            result = super().finish()
            ownership.require(alive and self.normal_stop, 'the component lost its lifetime device sampler')
            return {**result, 'sampler_alive_at_requested_stop': alive, 'sampler_normal_requested_stop': self.normal_stop}
        except BaseException as exception:
            error = {'type': type(exception).__name__, 'error': repr(exception)}
            raise
        finally:
            write(self.path, {'kind': KIND + '_raw_memory_observation', 'gpu_uuid': self.uuid,
                'samples': self.samples, 'errors': self.errors, 'alive_at_requested_stop': alive,
                'normal_requested_stop': self.normal_stop, 'result': result, 'error': error, **ZERO_CREDIT})


def successful_work_accounting(learning, latencies, batch_size, initializations):
    """Count actual recorded complete updates separately for primary trajectories and every successful clone stage."""
    primary, clones, latency_updates = {}, {}, {}
    window = learning['learning_protocol']['complete_window_updates']
    for arm in ('reference', 'candidate'):
        result = learning['arms'][arm]
        primary[arm] = result['completed_updates']
        clones[arm] = {'diagnostic_compilation_updates': len(learning['warmup'][arm]) * window,
            'selected_executor_compilation_updates': sum(row['updates'] for row in result['observer_agreement']['selected_executor_warmup']),
            'observer_agreement_updates': 2 * result['observer_agreement']['window_updates'],
            'strict_replay_updates': result['replay']['window_updates']}
    for case, records in latencies.items():
        latency_updates[case] = {}
        for arm, result in records.items():
            latency_updates[case][arm] = {'primary_synchronized_updates': result['calls'],
                'selected_executor_comparison_updates': sum(row['to_iteration'] - row['from_iteration'] for row in result['window_agreement']),
                'synchronized_executor_compilation_updates': sum(row['updates'] for row in result['warmup']),
                'selected_executor_compilation_updates': sum(row['updates'] for row in result['selected_executor_warmup'])}
    primary_count = sum(primary.values())
    additional_count = sum(value for row in clones.values() for value in row.values())
    additional_count += sum(value for case in latency_updates.values() for row in case.values() for value in row.values())
    ownership.require(len(initializations) == 2 + 2 * len(latencies)
        and all(row['probe_complete_updates'] == 1 and row['probe_environment_transitions'] == batch_size
                and row['original_initialization_preserved'] is True and row['probe_frozen_target_unchanged'] is True
                for row in initializations),
        'the component omitted a full counted disposable initialization probe')
    probe_updates = sum(row['probe_complete_updates'] for row in initializations)
    additional_count += probe_updates
    return {'primary_learning_updates_per_arm': primary, 'primary_learning_updates': primary_count,
        'primary_learning_environment_transitions': primary_count * batch_size,
        'learning_clone_updates': clones, 'latency_and_comparison_updates': latency_updates,
        'initialization_probe_updates': probe_updates, 'initialization_probe_environment_transitions': probe_updates * batch_size,
        'additional_complete_updates': additional_count, 'additional_update_environment_transitions': additional_count * batch_size,
        'total_complete_updates': primary_count + additional_count,
        'total_update_environment_transitions': (primary_count + additional_count) * batch_size,
        'normalization_warmup_updates': 0, 'normalization_warmup_environment_transitions': 0,
        'environment_initialization_work': 'recorded separately as actual initialization counters; not relabelled as PPO update transitions',
        'environment_transition_unit': 'one original trainer environment step; raw native-frame/reset counters retained separately', **ZERO_CREDIT}


def validate_learning_rows(unit, result, plan):
    """Require every primary update, full boundary bins, strict replay, actual changes and frozen target from raw rows."""
    short_learning.checked_component_protocol(plan)
    curves = Path(result['curves']['path'])
    ownership.require(curves == unit / 'learning_curves.jsonl' and training.file_sha256(curves) == result['curves']['sha256'],
                      'component curves are not the exact original raw rows')
    rows = [json.loads(line) for line in curves.read_text().splitlines()]
    declared = plan['learning_protocol']
    ownership.require(result['training_start_iteration'] == {'reference': 0, 'candidate': 0},
                      'component learning did not start both independent original schedules at zero')
    for arm in ('reference', 'candidate'):
        selected = [row for row in rows if row['variant'] == arm]
        ownership.require([row['update_in_check'] for row in selected] == list(range(1, declared['updates'] + 1)),
                          'component primary curves omit or repeat a full update')
        for row in selected:
            ownership.require(row['iteration'] == row['update_in_check'] and row['copy_seeds'] == [11]
                and row['floating_state_nonfinite_count'] == 0 and row['boundary_action_bins'] == list(range(18))
                and row['boundary_room_bins'] == list(range(33))
                and row['boundary_action_unknown_count'] == [0] and row['boundary_room_unknown_count'] == [0]
                and sum(row['boundary_action_histogram'][0]) == sum(row['boundary_room_histogram'][0]) == 1,
                'component raw policy seed, counters, finiteness or full boundary namespace changed')
            ownership.require(short_learning.DIAGNOSTIC_FIELDS <= set(row) and short_learning.METRICS <= set(row['metrics']),
                              'component curves omitted original diagnostics or metrics')
            for value in row['metrics'].values():
                validation.flattened_numbers(value)
            for name in ('optimizer_step', 'frame_count', 'step_count', 'reset_count', 'environment_key'):
                values = validation.flattened_numbers(row[name])
                ownership.require(values and all(type(value) is int and value >= 0 for value in values),
                                  'component RNG/counter rows are missing or invalid')
            ownership.require(validation.flattened_numbers(row['optimizer_step']) == [row['update_in_check']]
                and validation.flattened_numbers(row['metrics']['steps_per_policy']) == [row['update_in_check'] * 2],
                'component optimizer or transition counters differ from actual full updates')
            for name in ('observation_variance_mean', 'reward_variance', 'observation_count', 'reward_count'):
                values = validation.flattened_numbers(row[name])
                ownership.require(len(values) == 1 and all(value >= 0 for value in values)
                    and (not name.endswith('_count') or all(value > 0 for value in values)),
                    'component running statistics are missing, nonfinite or invalid')
        record = result['arms'][arm]
        ownership.require(record['completed_updates'] == declared['updates'] and record['frozen_target_unchanged'] is True
            and validation.flattened_numbers(record['optimizer_step']) == [declared['updates']]
            and record['replay']['passed'] is True and record['replay']['agreement_policy'] == short_learning.REPLAY_AGREEMENT
            and record['observer_agreement']['passed'] is True
            and record['observer_agreement']['agreement_policy'] == short_learning.OBSERVER_AGREEMENT
            and record['observer_agreement']['report_rows'] == declared['complete_window_updates'] // 4
            and all(len(record[name]) == 1 for name in ('agent_change_squared', 'predictor_change_squared'))
            and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                    for name in ('agent_change_squared', 'predictor_change_squared') for value in record[name])
            and record['precision']['initial']['all_state_finite'] is True and record['precision']['latest']['all_state_finite'] is True
            and record['precision']['initial']['network_dtype'] == record['precision']['latest']['network_dtype'] == 'float32'
            and record['precision']['initial']['statistics_dtype'] == record['precision']['latest']['statistics_dtype'] == 'float64',
            'the component did not change both trainable networks, preserve target/precision or match its real observer/replay')
    ownership.require(result['learning_protocol'] == declared and result['kind'] == KIND,
                      'component output changed its actual zero-warmup diagnostic contract')
    reports = Path(result['reports']['path'])
    ownership.require(reports == unit / 'learning_reports.jsonl' and training.file_sha256(reports) == result['reports']['sha256'],
                      'component report windows lack their original raw artifact')
    rows = [json.loads(line) for line in reports.read_text().splitlines()]
    for arm in ('reference', 'candidate'):
        selected = [row for row in rows if row['variant'] == arm]
        ownership.require([row['iteration'] for row in selected] == list(range(4, declared['updates'] + 1, 4))
            and all(row['window_updates'] == 4 and row['metrics']['steps_per_policy'] == row['iteration'] * 2 for row in selected),
            'component learning omitted a full actual selected reporting window')


def validate_latency_records(records, configurations):
    """Require actual synchronized calls, stable compilation and every selected Executor report window."""
    for arm in ('reference', 'candidate'):
        result = records[arm]
        declared = short_learning.latency_protocol(configurations[arm])
        values = result['seconds_per_synchronized_call']
        ownership.require(result['protocol'] == declared and result['calls'] == len(values) == declared['calls']
            and result['updates_per_call'] == 1 and result['synchronization_boundary'] == 'each_original_complete_update'
            and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values)
            and result['selected_iterations_per_call'] == configurations[arm]['execution']['iterations_per_call']
            and result['median_seconds_per_synchronized_call'] == statistics.median(values)
            and result['mean_seconds_per_synchronized_call'] == sum(values) / len(values)
            and result['median_amortized_seconds_per_update'] == statistics.median(values),
            'component latency omitted a real single-update synchronization boundary')
        window = declared['complete_window_updates']
        for name in ('warmup', 'selected_executor_warmup'):
            warm = result[name]
            ownership.require(len(warm) >= 2 and warm[-1]['compiled_signatures'] == warm[-2]['compiled_signatures']
                and all(row['updates'] == window and row['compiled_signatures'] for row in warm),
                'component latency did not stabilize its complete selected window')
        checks = result['window_agreement']
        ownership.require(len(checks) == 3, 'component latency omitted a complete comparison window')
        for index, check in enumerate(checks):
            reports = check['published_reports']
            ownership.require(check['passed'] is True and check['from_iteration'] == index * window
                and check['to_iteration'] == (index + 1) * window
                and check['report_rows'] == len(reports) == declared['report_rows_per_window']
                and check['state_and_final_metrics']['leaves'] > 0
                and (not reports or check['selected_reports']['leaves'] > 0),
                'component single-update path disagrees with the actual grouped/reporting Executor')
            for report_index, report in enumerate(reports, 1):
                iteration = index * window + report_index * declared['selected_report_every_iterations']
                ownership.require(report['iteration'] == iteration and report['window_updates'] == 4
                    and report['metrics']['steps_per_policy'] == iteration * 2,
                    'component latency changed its selected report accumulator/readout')
                for value in report['metrics'].values():
                    validation.flattened_numbers(value)


def preflight(unit, plan, admission):
    """Verify metadata-only actual package/source/library/assets and assigned-card availability before JAX."""
    ownership.require('jax' not in sys.modules, 'JAX was imported before component admission')
    training.enforce_runtime(plan['runtime_environment'], os.environ)
    assets = training.verify_assets(plan['asset_record'], os.environ)
    libraries = training.installed_gpu_library_receipts()
    jaxtari_source = training.jaxtari_python_receipt(Path(plan['jaxtari_source_root']))
    installed = measurement.installed_backend_receipt(jaxtari_source, libraries)
    hardware = diagnostic.hardware_record(admission['gpu_uuid'], 'jaguar03_rtx_a4500')
    hardware.update(gpu_uuid=admission['gpu_uuid'], process_start_time=admission['process']['process_start_time'],
                    slurm_job_id=admission['job_id'])
    write(unit / 'hardware.json', hardware)
    write(unit / 'installed_backend.json', installed)
    write(unit / 'asset_receipt.json', assets)
    # Only the already proved assigned physical UUID is queried; no ordinal resolver or full XML report is used.
    free = measurement.selected_query(admission['gpu_uuid'], 'uuid,name,memory.total,driver_version,mig.mode.current,memory.used')
    ownership.require(len(free) == 1 and len(free[0]) == 6 and free[0][0].strip() == admission['gpu_uuid']
        and 'A4500' in free[0][1] and free[0][4].strip() in {'Disabled', '[N/A]'} and int(free[0][5]) < 1024
        and not measurement.selected_query(admission['gpu_uuid'], 'pid', processes=True), 'the owned component card is not free')
    return assets, libraries, installed


def initialization_probe(trainer, state, unit, index, stage, jax, arrays, paired, device, admission, libraries):
    """Prove the actual CUDA identity first, then load and verify cuDNN using one discarded independent full update."""
    record = {'kind': KIND + '_discarded_initialization_probe', 'index': index, 'stage': stage,
        'initial_iteration': int(state.iteration), 'probe_started': False, 'probe_complete_updates': 0,
        'probe_environment_transitions': 0, 'original_initialization_preserved': False, 'completed': False, **ZERO_CREDIT}
    try:
        # Same-process physical identity is mandatory before executing the discarded learner update.
        context = assignment.context_capacity(device)
        write(unit / ('initialization_context_' + str(index) + '.json'), context)
        pci = ownership.validate_mapping(admission['ownership']['physical_mapping'], admission['ownership'], admission['gpu_uuid'])
        ownership.require(context['gpu_uuid'] == admission['gpu_uuid'] and context['pci_address'] == pci,
                          'same-process CUDA/JAX UUID or PCI differs from actual owned assignment')
        if index == 0:
            write(unit / 'current_cuda_context.json', context)
            write(unit / 'captured_driver_image.json', ownership.captured_driver(context['driver_library'], unit / 'context_driver'))
        else:
            ownership.require(context['driver_library'] == read(unit / 'current_cuda_context.json')['driver_library'],
                              'the mapped CUDA driver changed between initialization probes')

        # For example, preserve update-zero state, advance a separate copy to one, and return the original zero state.
        original = paired.host_copy(state)
        record['initial_precision'] = short_learning._precision(original, paired)
        record.update(probe_started=True, probe_complete_updates=None, probe_environment_transitions=None)
        probe, metrics = trainer.train_iteration(paired.device_copy(original))
        jax.block_until_ready((probe, metrics))
        updates = int(probe.iteration) - int(original.iteration)
        record.update(probe_complete_updates=updates, probe_environment_transitions=updates * trainer.config.batch_size,
                      probe_final_iteration=int(probe.iteration))
        ownership.require(int(original.iteration) == 0 and updates == 1, 'the disposable probe changed its original one-update budget')
        paired.require_finite((probe, metrics), 'component discarded initialization probe')
        record['probe_precision'] = short_learning._precision(paired.host_copy(probe), paired)
        ownership.require(jax.tree_util.tree_structure(original.target) == jax.tree_util.tree_structure(probe.target)
            and all(arrays.array_equal(before, current) for before, current in
                    zip(jax.tree.leaves(original.target), jax.tree.leaves(paired.host_copy(probe.target)))),
            'the disposable probe changed its frozen RND target')
        record['probe_frozen_target_unchanged'] = True

        # A convolution has now actually run; strict already-mapped library proof no longer assumes eager cuDNN loading.
        loaded = training.actual_gpu_library_receipt(device.client, libraries)
        training.validate_gpu_library_receipt(loaded, {'installed_gpu_libraries': libraries})
        path = unit / ('initialization_loaded_libraries_' + str(index) + '.json')
        write(path, loaded)
        record['loaded_libraries'] = ownership.file_receipt(path)
        if index == 0:
            write(unit / 'loaded_gpu_libraries.json', loaded)
        after = paired.host_copy(state)
        ownership.require(jax.tree_util.tree_structure(original) == jax.tree_util.tree_structure(after)
            and all(arrays.array_equal(before, current) for before, current in zip(jax.tree.leaves(original), jax.tree.leaves(after))),
            'the discarded update changed the original initialization state')
        record.update(original_initialization_preserved=True, completed=True)
        return record
    except BaseException as error:
        record.update(error_type=type(error).__name__, error=repr(error))
        raise
    finally:
        write(unit / ('initialization_probe_' + str(index) + '.json'), record)


def compute(unit, plan, admission):
    """Import the frozen actual game and full networks only after all metadata and assigned-card gates."""
    assets, libraries, installed = preflight(unit, plan, admission)
    root = Path(plan['source_root']) / '10_jax_exploration_platform'
    sys.path[:0] = [str(root / 'src'), str(root / 'scripts/atari_pixel_campaign')]
    import jax
    import jaxtari
    import numpy as np
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
    from exploration_platform.atari_pixels.training import PixelPPORND
    paired = importlib.import_module('paired_benchmark')
    measurement.verify_imported_sources(root, paired)
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    devices = jax.devices()
    ownership.require(len(devices) == 1 and devices[0].platform == 'gpu' and 'A4500' in devices[0].device_kind,
                      'the component did not select its single assigned A4500')
    initialized = []
    active_stage = ['learning']
    original_trainer = PixelPPORND
    class RecordingTrainer(original_trainer):
        """Observe initialization counters and actual libraries without changing the returned learner state."""

        def initialize(self, seeds):
            """Return the unchanged initial state after one counted independent probe loads and verifies its libraries."""
            state = super().initialize(seeds)
            jax.block_until_ready(state)
            event = {'stage': active_stage[0], 'copy_seeds': list(seeds), 'iteration': int(state.iteration),
                'frame_count': jax.device_get(state.environment.frame_count).tolist(),
                'step_count': jax.device_get(state.environment.step_count).tolist(),
                'reset_count': jax.device_get(state.environment.reset_count).tolist(),
                'normalization_warmup_iterations': self.config.observation_warmup_iterations}
            write(unit / ('initialization_created_' + str(len(initialized)) + '.json'), event)
            probe = initialization_probe(self, state, unit, len(initialized), active_stage[0],
                jax, np, paired, devices[0], admission, libraries)
            event.update({
                'probe_complete_updates': probe['probe_complete_updates'],
                'probe_environment_transitions': probe['probe_environment_transitions'],
                'original_initialization_preserved': probe['original_initialization_preserved'],
                'probe_frozen_target_unchanged': probe['probe_frozen_target_unchanged'],
                'probe_record': ownership.file_receipt(unit / ('initialization_probe_' + str(len(initialized)) + '.json'))})
            initialized.append(event)
            training.append_record(unit / 'initialization_counters.jsonl', event)
            return state
    observations = []
    def memory_factory(device, gpu_uuid):
        """Give each whole stage a distinct raw sampler receipt."""
        instance = ComponentMemoryObservation(device, gpu_uuid, unit / ('memory_observation_' + str(len(observations)) + '.json'))
        observations.append(instance)
        return instance
    def progress(value):
        """Publish the actual component stage without labelling it formal training."""
        diagnostic.record_progress(unit, value['state'], component_kind=KIND, **{name: item for name, item in value.items() if name != 'state'})
    result = short_learning.run_component(unit, {**plan, 'gpu_uuid': admission['gpu_uuid']}, paired,
        PixelPPOConfig, RecordingTrainer, AtariPixelEnvironment, progress, memory_factory)
    write(unit / 'component_learning_result.json', result)
    validate_learning_rows(unit, result, plan)
    latencies = {}
    for name, configurations in plan['latency_cases'].items():
        active_stage[0] = 'latency_' + name
        progress({'state': 'component_latency', 'case': name})
        memory = memory_factory(devices[0], admission['gpu_uuid'])
        memory.begin()
        try:
            rows = short_learning.single_update_latencies(paired, configurations['candidate'], configurations['reference'],
                (11,), PixelPPOConfig, RecordingTrainer, AtariPixelEnvironment)
        finally:
            observed = memory.finish()
            write(unit / ('latency_memory_' + name + '.json'), observed)
        ownership.require(not observed['other_gpu_processes'], 'another process occupied the component card')
        validate_latency_records(rows, configurations)
        latencies[name] = rows
        write(unit / ('component_latency_' + name + '.json'), {'kind': KIND, 'case': name, 'records': rows, **ZERO_CREDIT})
    work = successful_work_accounting(result, latencies, 2, initialized)
    write(unit / 'component_work_accounting.json', work)
    write(unit / 'final_loaded_gpu_libraries.json', training.actual_gpu_library_receipt(devices[0].client, libraries))
    ownership.require(training.verify_assets(plan['asset_record'], os.environ) == assets, 'the component changed original game assets')
    return {'schema_version': 1, 'kind': KIND, 'completed': True, 'learning': ownership.file_receipt(unit / 'component_learning_result.json'),
        'work': work, 'initialization_count': len(initialized), 'memory_observation_count': len(observations),
        'actual_gpu_uuid': admission['gpu_uuid'], **ZERO_CREDIT}


def run(unit):
    """Run one separately admitted component attempt and retain every raw failure without retry or formal credit."""
    plan = verify_static(unit, execution=True)
    ownership.require(not any((unit / name).exists() for name in ('allocation_bootstrap.json', 'allocation_admission.json',
        'component_result.json', 'failure.json')), 'preserve the prior component attempt and prepare a new unit')
    try:
        admission = admit(unit, plan)
        outcome = compute(unit, plan, admission)
        write(unit / 'component_result.json', outcome)
        try:
            short_learning.checked_formal_protocol(plan)
        except ValueError as error:
            write(unit / 'formal_entry_refusal.json', {'kind': KIND + '_formal_entry_refusal',
                'plan': ownership.file_receipt(unit / 'component_plan.json'),
                'component_result': ownership.file_receipt(unit / 'component_result.json'),
                'formal_function': 'short_learning.checked_formal_protocol', 'error': str(error), **ZERO_CREDIT})
        else:
            raise ValueError('the tiny component was unexpectedly accepted by the formal entry')
        return outcome
    except BaseException as error:
        curve = unit / 'learning_curves.jsonl'
        probes = [read(path) for path in sorted(unit.glob('initialization_probe_*.json'))]
        write(unit / 'failure.json', {'kind': KIND + '_failure', 'type': type(error).__name__, 'error': repr(error),
            'traceback': traceback.format_exc(), 'recorded_primary_curve_rows': len(curve.read_text().splitlines()) if curve.exists() else 0,
            'recorded_probe_updates': sum(row['probe_complete_updates'] for row in probes if type(row['probe_complete_updates']) is int),
            'incomplete_probe_work_unknown': any(row['probe_complete_updates'] is None for row in probes),
            'additional_or_unrecorded_work_may_have_occurred': True, 'partial_work_is_not_formal_evidence': True, **ZERO_CREDIT})
        raise


def main():
    """Separate metadata preparation from the owned compute-only component entry, with no job-submission command."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    preparation = actions.add_parser('prepare')
    for name in ('run-dir', 'source-root', 'helper-root', 'asset-record', 'runtime-environment', 'jaxtari-source-root', 'python'):
        preparation.add_argument('--' + name, type=Path, required=True)
    for name in ('source-commit', 'helper-commit', 'unit-id'):
        preparation.add_argument('--' + name, required=True)
    execution = actions.add_parser('run')
    execution.add_argument('unit', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps({'unit': str(prepare(args)), 'submitted': False}))
    else:
        print(json.dumps(run(args.unit)))


if __name__ == '__main__':
    main()
