"""Test diagnostic preparation and refusal boundaries with metadata and mocked APIs only."""

import copy
from contextlib import ExitStack
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import device_memory_capture_regression as regression


class CaptureRegressionMetadataTests(unittest.TestCase):
    """Separate synthetic array diagnostics from every formal packing and training record."""

    def setUp(self):
        """Keep every output in a temporary metadata directory, never a live campaign."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gpu = 'GPU-00000000-0000-0000-0000-000000000004'
        self.runtime = {'JAX_ENABLE_X64': 'true', 'JAX_DEFAULT_MATMUL_PRECISION': 'highest',
                        'XLA_FLAGS': '--xla_gpu_exclude_nondeterministic_ops=true'}
        for name, value in [('runtime', self.runtime), ('assets', {'root': '/fixture/assets',
                'manifest': '/fixture/manifest.json', 'sha256': 'a' * 64})]:
            (self.root / (name + '.json')).write_text(json.dumps(value))
        self.arguments = SimpleNamespace(run_dir=self.root, unit_id='separate_array_capture_diagnostic',
            source_root=SCRIPTS.parents[2], source_commit=subprocess.check_output(['git', '-C', str(SCRIPTS.parents[2]),
                'rev-parse', 'HEAD'], text=True).strip(), capture_helper_root=SCRIPTS.parents[2],
            runtime_environment=self.root / 'runtime.json', asset_record=self.root / 'assets.json',
            jaxtari_source_root=self.root, python=Path(sys.executable))

    def source_receipt(self, root, commit, paths):
        """Model exact source hashes without claiming this uncommitted diagnostic is published."""
        return {'source_root': str(root), 'source_commit': commit,
                'files': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}}

    def prepare(self):
        """Mock only Git publication prerequisites; keep actual metadata preparation code."""
        with patch.object(regression.training, 'source_receipt', side_effect=self.source_receipt), \
                patch.object(regression.diagnostic, 'core_identity', return_value={'source_root': str(self.arguments.source_root),
                    'source_commit': self.arguments.source_commit, 'files': {}}), \
                patch.object(regression.deployment, 'git_root', side_effect=lambda root, commit, evidence: Path(root)), \
                patch.object(regression, 'admit') as admission, patch.object(regression.measurement, 'selected_query') as query:
            result = regression.prepare(self.arguments)
            query.assert_not_called()
            admission.assert_not_called()
        return result

    def process(self):
        """Give metadata checks one fixed owned parent process and Slurm hierarchy."""
        return {'hostname': 'jaguar03', 'uid': 2618919, 'pid': 1000, 'parent_pid': 999,
            'process_start_time': '10000', 'cpu_affinity': list(range(8)),
            'cgroup': '3:memory:/slurm/uid_2618919/job_123/step_0/task_0\n'}

    def job(self):
        """Give shape checks the exact approved one-card allocation, never live scheduler output."""
        return ('JobId=123 UserId=sl5nw(2618919) JobState=RUNNING NodeList=jaguar03 '
            'GRES=gpu:nvidia_rtx_a4500:1(IDX:4) Account=sl5nw_base Partition=gpu QOS=csresnolim Reservation=sl5nw_12345 '
            'NumNodes=1 NumCPUs=8 NumTasks=1 CPUs/Task=8 TimeLimit=00:20:00 '
            'AllocTRES=cpu=8,mem=64G,node=1,gres/gpu=1 TresPerNode=gres/gpu:nvidia_rtx_a4500:1')

    def admit(self, unit):
        """Exercise the second stage with an explicitly recorded returned ID and no real scheduler query."""
        ledger = self.root / 'slurm/submitted_jobids_uva.txt'
        ledger.parent.mkdir()
        ledger.write_text('123\n')
        process = self.process()
        def owned(plan):
            """Retain the measured selector and fixed process while mocking the already tested allocation collector."""
            return {'process': process, 'environment': dict(os.environ), 'job_id': '123', 'job': {'stdout': job}}
        job = self.job()
        def resolve(job, environment, evidence):
            """Model Slurm index four resolving to physical minor eight and an unrelated CUDA selector zero."""
            raw = 'Name=gpu Type=nvidia_rtx_a4500 File=/dev/nvidia[4-11]\n'
            proc = 'GPU UUID: ' + self.gpu + '\nDevice Minor: 8\n'
            evidence.update(job=job, environment=environment, resolved_gpu_uuid=self.gpu,
                gres_file={'path': '/etc/slurm/gres.conf', 'owner_uid': 0, 'mode': 0o644, 'regular_file': True,
                    'symlink': False, 'raw': raw, 'sha256': hashlib.sha256(raw.encode()).hexdigest()},
                proc_information=[{'path': '/proc/driver/nvidia/gpus/0000:14:00.0/information',
                    'raw': proc, 'sha256': hashlib.sha256(proc.encode()).hexdigest()}])
            return regression.validate_resolution(evidence)
        with patch.object(regression.capacity, 'process_identity', return_value=process), \
                patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'SLURM_JOB_GPUS': '4',
                    'SLURM_STEP_ID': '0', 'SLURM_STEP_GPUS': '4', 'CUDA_VISIBLE_DEVICES': '0'}, clear=True), \
                patch.object(regression.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stderr='',
                    stdout=job)) as query, \
                patch.object(regression, 'resolve_uuid', side_effect=resolve) as resolver, \
                patch.object(regression.capacity, 'ownership', side_effect=owned):
            result = regression.admit(unit, regression.read(unit / 'regression_plan.json'))
            self.assertEqual(query.call_args.args[0], ['scontrol', 'show', 'job', '-d', '123'])
            resolver.assert_called_once()
        return result

    def test_preparation_has_no_guessed_id_uuid_or_array_training_claim(self):
        """Generic records describe one card and explicit zero training work before an allocation exists."""
        unit = self.prepare()
        plan = regression.read(unit / 'regression_plan.json')
        self.assertIsNone(plan['job_id'])
        self.assertIsNone(plan['gpu_uuid'])
        self.assertIsNone(plan['reservation'])
        self.assertEqual(plan['shape']['qos'], 'csresnolim')
        self.assertEqual(plan['kind'], regression.KIND)
        self.assertEqual(plan['environment_transitions'], 0)
        self.assertEqual(plan['learner_updates'], 0)
        self.assertFalse(plan['fixture']['formal_packing_evidence'])
        self.assertEqual(plan['helpers']['source_commit'], regression.CAPTURE_HELPER_COMMIT)
        for name in ['experiment_background.md', 'manifest.yaml', 'config_resolved.yaml', 'command.txt']:
            self.assertTrue((unit / name).is_file())
        self.assertFalse((unit / 'allocation_admission.json').exists())
        self.assertNotIn('jax', sys.modules)
        self.assertNotIn('numpy', sys.modules)

    def test_actual_returned_id_is_resolved_once_without_rewriting_static_plan(self):
        """A real returned-ID snapshot and physical UUID belong in a separate immutable second-stage record."""
        unit = self.prepare()
        before = (unit / 'regression_plan.json').read_bytes()
        result = self.admit(unit)
        self.assertEqual((unit / 'regression_plan.json').read_bytes(), before)
        self.assertEqual(result['job_id'], '123')
        self.assertEqual(result['gpu_uuid'], self.gpu)
        self.assertEqual(result['runtime_plan']['job_id'], '123')
        self.assertEqual(result['reservation'], 'sl5nw_12345')
        self.assertEqual(result['runtime_plan']['reservation'], 'sl5nw_12345')
        self.assertEqual((unit / 'diagnostic_owned_job_ledger.txt').read_bytes(), b'123\n')
        resolution = regression.read(unit / 'allocation_resolution.json')
        self.assertEqual(resolution['environment']['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(resolution['environment']['SLURM_JOB_GPUS'], '4')
        self.assertIn('Device Minor: 8', resolution['proc_information'][0]['raw'])
        (self.root / 'slurm/submitted_jobids_uva.txt').write_text('123\n124\n')
        with patch.object(regression.capacity, 'validate_ownership'):
            self.assertEqual(regression.validate_admission(unit, regression.read(unit / 'regression_plan.json')), result)
        resolved = regression.child_plan(unit, 'failure', result)
        self.assertEqual(resolved['job_id'], '123')
        self.assertEqual(resolved['gpu_uuid'], self.gpu)
        self.assertEqual(resolved['reservation'], 'sl5nw_12345')
        self.assertFalse(resolved['formal_packing_evidence'])

    def test_unrecorded_id_or_foreign_owner_stops_before_queries(self):
        """Admission cannot ask the scheduler or hardware about an unrecorded job or another owner."""
        unit = self.prepare()
        ledger = self.root / 'slurm/submitted_jobids_uva.txt'
        ledger.parent.mkdir()
        ledger.write_text('124\n')
        plan = regression.read(unit / 'regression_plan.json')
        for owner in (self.process(), {**self.process(), 'uid': 777}):
            with patch.object(regression.capacity, 'process_identity', return_value=owner), \
                    patch.dict(os.environ, {'SLURM_JOB_ID': '123'}, clear=True), \
                    patch.object(regression.subprocess, 'run') as query, self.assertRaises(ValueError):
                regression.admit(unit, plan)
            query.assert_not_called()
        self.assertFalse((unit / 'allocation_admission.json').exists())

    def test_admission_cannot_change_source_or_array_fixture(self):
        """Only actual allocation fields may be added after the static source and fixture are frozen."""
        unit = self.prepare()
        self.admit(unit)
        path = unit / 'allocation_admission.json'
        value = regression.read(path)
        value['runtime_plan']['fixture']['repeat_elements'] = 1
        path.write_text(json.dumps(value))
        with patch.object(regression.capacity, 'validate_ownership'), self.assertRaisesRegex(ValueError, 'only measured allocation'):
            regression.validate_admission(unit, regression.read(unit / 'regression_plan.json'))

    def test_request_uses_actual_context_capacity_and_exact_small_alignment(self):
        """The request is the next 256-byte multiple rather than an assumed card-memory number."""
        self.assertEqual(regression.next_request(1024), 1280)
        self.assertEqual(regression.next_request(1025), 1280)
        for value in (False, 0, -1, 2 ** 64 - 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                regression.next_request(value)

    def test_diagnostic_bytes_are_rejected_by_both_formal_readers(self):
        """Renaming array-fixture bytes cannot turn them into full 128-environment capacity evidence."""
        unit = self.prepare()
        (unit / 'repeat').mkdir()
        regression.write(unit / 'repeat/fixture_result.json', {'schema_version': 1,
            'kind': regression.KIND + '_small_repeat', 'environment_transitions': 0, 'learner_updates': 0})
        errors = regression.formal_refusals(unit, regression.read(unit / 'regression_plan.json'), create=True)
        self.assertEqual(set(errors), {'capacity_reader', 'packing_reader'})
        before = {str(path.relative_to(unit)): path.read_bytes() for path in unit.rglob('*') if path.is_file()}
        self.assertEqual(regression.formal_refusals(unit, regression.read(unit / 'regression_plan.json')), errors)
        after = {str(path.relative_to(unit)): path.read_bytes() for path in unit.rglob('*') if path.is_file()}
        self.assertEqual(before, after)
        self.assertNotIn('jax', sys.modules)

    def test_absent_current_context_has_no_creation_or_allocation_fallback(self):
        """An empty current context refuses the regression without trying another CUDA context."""
        calls = []
        class Function:
            """Supply ctypes-compatible pointer writes without a real shared-library call."""
            def __init__(self, name):
                """Retain the exact mocked API name."""
                self.name = name
            def __call__(self, *arguments):
                """Return a missing current context and otherwise successful metadata queries."""
                calls.append(self.name)
                if self.name == 'cuCtxGetCurrent': arguments[0]._obj.value = None
                elif self.name == 'cuCtxGetDevice': arguments[0]._obj.value = 0
                elif self.name == 'cuMemGetInfo_v2':
                    arguments[0]._obj.value, arguments[1]._obj.value = 512, 1024
                return 0
        driver = SimpleNamespace(**{name: Function(name) for name in ('cuCtxGetCurrent', 'cuCtxGetDevice', 'cuCtxSynchronize', 'cuMemGetInfo_v2')})
        with patch.object(Path, 'read_text', return_value='0000-1000 r-xp 0000 00:00 0 /fixture/libcuda.so.590\n'), \
                patch.object(regression.ctypes, 'CDLL', return_value=driver), self.assertRaisesRegex(ValueError, 'already active'):
            regression.context_capacity(SimpleNamespace(local_hardware_id=0))
        self.assertEqual(calls, ['cuCtxGetCurrent', 'cuCtxGetDevice', 'cuCtxSynchronize', 'cuMemGetInfo_v2'])

    def failure_detail(self):
        """Use literal native protobuf/error bytes and mocked allocation mapping for diagnostic-only connection tests."""
        unit = self.root / 'diagnostic_detail'
        directory = unit / 'failure'
        directory.mkdir(parents=True)
        process = {**self.process(), 'pid': 1001, 'parent_pid': 1000, 'process_start_time': '10001'}
        driver = {'path': '/node/libcuda.so.synthetic', 'sha256': 'b' * 64}
        context = {'pid': 1001, 'context': 1234, 'device': 0, 'gpu_uuid': 'GPU-fixture', 'pci_address': '0000:14:00.0',
            'driver_library': driver, 'total_bytes': 1024, 'free_bytes': 1024, 'observed_at_seconds': 1000.,
            'return_codes': {name: 0 for name in ('cuCtxGetCurrent', 'cuCtxGetDevice', 'cuCtxSynchronize',
                'cuMemGetInfo_v2', 'cuDeviceGetUuid_v2', 'cuDeviceGetPCIBusId')}}
        exception = {'error': "XlaRuntimeError('RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.25KiB.')",
                     'exception_type': 'XlaRuntimeError', 'traceback': 'synthetic original failure'}
        raw = b'Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.25KiB (rounded to 1280)\n'
        native_map = b'\x0a\x09GPU_0_bfc\x2a\x00'
        allocator = {'bytes_limit': 850, 'bytes_in_use': 0, 'peak_bytes_in_use': 0}
        parsed = regression.capacity.native_request(exception, raw, native_map, allocator)
        mapping = {'synthetic_mapping_checked_separately': True}
        identity = {'ownership': {'physical_mapping': mapping}}
        plan = {'source_commit': 'a' * 40}
        admission = {'gpu_uuid': 'GPU-fixture', 'runtime_plan': {'device_memory_limit_protocol': {'capability': 'synthetic'}}}
        requested = {'kind': regression.KIND + '_intentional_failure', 'requested_bytes': 1280, 'context': context,
            'dtype': 'uint8', 'environment_transitions': 0, 'formal_packing_evidence': False}
        for name, value in [('requested_fixture.json', requested), ('current_context.json', context),
                ('fixture_exception.json', {**exception, 'kind': regression.KIND + '_expected_failure', 'environment_transitions': 0})]:
            regression.write(directory / name, value)
        rows = '0000-1000 r-xp 0000 00:00 0 /node/xla_cuda_plugin.so\n'
        check = {'pid': 1001, 'gpu_uuid': 'GPU-fixture', 'pci_address': '0000:14:00.0', 'requested_bytes': 1280,
            'cuMemAlloc_v2': 2, 'cuMemFree_v2': None, 'context': 1234, 'context_after': 1234,
            'context_created_or_reset': False, 'allocator_settings_changed': False, 'training_retried': False,
            'driver_library': driver, 'jax_local_hardware_id': 0, 'cuda_device_handle': 0, 'cuda_device_handle_after': 0,
            'api_return_codes': {name: 0 for name in ('cuCtxGetCurrent', 'cuCtxGetDevice', 'cuCtxSynchronize',
                'cuDeviceGetUuid_v2', 'cuDeviceGetPCIBusId')},
            'post_api_return_codes': {name: 0 for name in ('cuCtxGetCurrent', 'cuCtxGetDevice', 'cuCtxSynchronize')},
            'before': {'returncode': 0, 'total_bytes': 1024, 'free_bytes': 1024, 'time_seconds': 1003.1},
            'after': {'returncode': 0, 'total_bytes': 1024, 'free_bytes': 1024, 'time_seconds': 1004.},
            'started_at_seconds': 1003.2, 'finished_at_seconds': 1004.}
        observation = {'schema_version': 1, 'kind': 'instrumented_device_memory_limit_observation',
            **regression.capacity.ZERO_CREDIT, 'protocol': admission['runtime_plan']['device_memory_limit_protocol'],
            'source_commit': plan['source_commit'], 'original_exception': exception, 'native_allocation': parsed,
            'cuda_allocation_check': check, 'native_backend': {'pid': 1001, 'required_marker': 'TF_BFC_MEMORY_DUMP',
                'library': {'path': '/node/xla_cuda_plugin.so', 'sha256': 'c' * 64}, 'mapped_image_rows': rows,
                'mapped_image_rows_sha256': hashlib.sha256(rows.encode()).hexdigest()},
            'loaded_gpu_libraries': {'pid': 1001, 'loaded_shared_libraries': [driver['path']]},
            'sampler_errors': [], 'pre_probe_capture_time_seconds': 1003.,
            'native_map_original_name': 'capacity_native_map_GPU_0_bfc.1002000000',
            'failure_progress': {'state': 'diagnostic_array_allocation'},
            'memory_samples': [{'device': {'gpu_uuid': 'GPU-fixture', 'total_bytes': 2048, 'used_bytes': 16,
                'free_bytes': 2032, 'timestamp_seconds': when, 'processes': [{'pid': 1001, 'bytes': 16}]},
                'allocator': copy.deepcopy(allocator)} for when in (999.9, 1005.)]}
        return unit, plan, admission, identity, process, {'finished_at_seconds': 1006.}, observation, parsed

    def test_native_diagnostic_connection_accepts_exact_raw_request_without_formal_credit(self):
        """Connection checks reparse native bytes and bind actual context fields without making the array fixture packing evidence."""
        arguments = self.failure_detail()
        with patch.object(regression.capacity, 'validate_mapping', return_value='0000:14:00.0'):
            self.assertIsNone(regression.validate_failure_detail(*arguments))
        self.assertEqual(arguments[-2]['packing_point_credit'], 0)
        self.assertFalse(arguments[-2]['measurement_valid'])

    def test_native_diagnostic_connection_rejects_changed_scope_context_native_image_or_samples(self):
        """Self-consistent summaries cannot replace the original native request, child, driver or lifetime observations."""
        original = self.failure_detail()
        mutations = [lambda value: value.update(case_credit=True),
            lambda value: value['cuda_allocation_check'].update(pid=999),
            lambda value: value['cuda_allocation_check'].update(cuMemAlloc_v2=0),
            lambda value: value['cuda_allocation_check'].update(context_created_or_reset=True),
            lambda value: value['cuda_allocation_check'].update(requested_bytes=1024),
            lambda value: value['cuda_allocation_check']['driver_library'].update(sha256='d' * 64),
            lambda value: value['native_backend'].update(mapped_image_rows='0000-1000 r-xp 0000 00:00 0 /other/xla_cuda_plugin.so\n'),
            lambda value: value['memory_samples'][0]['device']['processes'][0].update(pid=999),
            lambda value: value['memory_samples'][0]['device']['processes'][0].update(bytes=2049),
            lambda value: value['memory_samples'][0]['allocator'].update(peak_bytes_in_use=851),
            lambda value: value.update(native_map_original_name='capacity_native_map_GPU_0_bfc.999000000'),
            lambda value: value['failure_progress'].update(state='ordinary_training')]
        for mutation in mutations:
            arguments = copy.deepcopy(original)
            mutation(arguments[-2])
            with self.subTest(change=mutation), patch.object(regression.capacity, 'validate_mapping', return_value='0000:14:00.0'), self.assertRaises(ValueError):
                regression.validate_failure_detail(*arguments)

    def test_known_diagnostic_artifacts_cannot_be_symlinks_to_identical_bytes(self):
        """An unrelated unit's original bytes cannot replace this diagnostic's exact artifact path."""
        original = self.root / 'outside.json'
        original.write_text('{}\n')
        link = self.root / 'fixture_identity.json'
        link.symlink_to(original)
        with self.assertRaisesRegex(ValueError, 'exact absolute paths'):
            regression.read(link)

    def test_job_shape_rejects_changed_processors_host_memory_gpu_count_or_time(self):
        """An otherwise owned job cannot silently change any approved diagnostic resource."""
        raw = self.job()
        regression.validate_shape({'job': {'stdout': raw}})
        changes = [('NumCPUs=8', 'NumCPUs=16'), ('CPUs/Task=8', 'CPUs/Task=4'), ('mem=64G', 'mem=32G'),
            ('TimeLimit=00:20:00', 'TimeLimit=01:00:00'), ('gres/gpu=1', 'gres/gpu=2'),
            ('QOS=csresnolim', 'QOS=normal'), ('mem=64G', 'mem=64G,mem=32G'),
            ('Reservation=sl5nw_12345', 'Reservation=foreign_1')]
        for before, after in changes:
            with self.subTest(before=before), self.assertRaises(ValueError):
                regression.validate_shape({'job': {'stdout': raw.replace(before, after)}})
        with self.assertRaisesRegex(ValueError, 'actual recorded owned allocation reservation'):
            regression.validate_shape({'job': {'stdout': raw}}, 'sl5nw_99999')

    def test_parent_and_child_births_are_bound_to_the_original_waited_processes(self):
        """Changing a birth counter while retaining the same PID cannot replace a fresh diagnostic child."""
        parent = self.process()
        processes = [{**parent, 'pid': 1001 + index, 'parent_pid': 1000, 'process_start_time': str(10001 + index)}
                     for index in range(2)]
        plan = {'python': sys.executable}
        terminal = []
        for index, phase in enumerate(('failure', 'repeat')):
            process = processes[index]
            raw = str(process['pid']) + ' (fixture) ' + ' '.join(['S', *['0'] * 18, process['process_start_time']])
            terminal.append({'phase': phase, 'returncode': 1 - index, 'child_pid': process['pid'],
                'child_process_start_time': process['process_start_time'], 'child_proc_stat': raw, 'parent': copy.deepcopy(parent),
                'command': [sys.executable, str(self.root / 'code/device_memory_capture_regression.py'), 'child', str(self.root), phase],
                'wait_completed': True, 'timeout_used': False, 'finished_at_seconds': 10000. + index})
        regression.validate_terminal(terminal, self.root, plan, parent, processes)
        mutations = [lambda rows: rows[0]['parent'].update(process_start_time='99999'),
            lambda rows: rows[0].update(child_process_start_time='99999'),
            lambda rows: rows[0].update(child_proc_stat=rows[0]['child_proc_stat'].replace('10001', '99999')),
            lambda rows: rows[0].update(returncode=True), lambda rows: rows[0].update(timeout_used=True)]
        for mutation in mutations:
            changed = copy.deepcopy(terminal)
            mutation(changed)
            with self.subTest(change=mutation), self.assertRaises(ValueError):
                regression.validate_terminal(changed, self.root, plan, parent, processes)
        changed = copy.deepcopy(processes)
        changed[1]['process_start_time'] = '99999'
        with self.assertRaises(ValueError):
            regression.validate_terminal(terminal, self.root, plan, parent, changed)

    def test_missing_helper_cannot_pass_pre_import_static_verification(self):
        """Rehashing a plan cannot hide a producer dependency omitted before numerical imports."""
        unit = self.prepare()
        path = unit / 'regression_plan.json'
        plan = regression.read(path)
        del plan['helpers']['files']['10_jax_exploration_platform/scripts/atari_pixel_campaign/validate_training_deployment.py']
        path.write_text(json.dumps(plan))
        (unit / 'regression_prepared.json').write_text(json.dumps({'plan': regression.capacity.file_receipt(path)}))
        with self.assertRaisesRegex(ValueError, 'exact complete published helper roster'):
            regression.verify_static(unit)
        self.assertNotIn('jax', sys.modules)

    def test_deployment_module_must_be_imported_from_the_exact_staged_unit(self):
        """One stale imported deployment reader is rejected even when the other five modules match staged paths."""
        unit = self.prepare()
        modules = [('device_memory_limit.py', regression.capacity), ('measure_selected.py', regression.measurement),
            ('diagnose_gpu_execution.py', regression.diagnostic), ('post_comparison_validation.py', regression.validation),
            ('train_optimized.py', regression.training), ('validate_training_deployment.py', regression.deployment)]
        with ExitStack() as stack:
            for name, module in modules:
                stack.enter_context(patch.object(module, '__file__', str(unit / 'code' / name)))
            regression.verify_executing_modules(unit)
            with patch.object(regression.deployment, '__file__', str(SCRIPTS / 'validate_training_deployment.py')):
                with self.assertRaisesRegex(ValueError, 'validate_training_deployment.py'):
                    regression.verify_executing_modules(unit)

    def test_successful_repeat_requires_its_first_normal_requested_sampler_stop(self):
        """A dead sampler, unexpected exit or recorded ordinary error cannot qualify the successful child's lifecycle."""
        normal = {'sampler_started': True, 'alive_at_first_shutdown': True, 'stop_requested': True,
                  'joined': True, 'exit_reason': 'requested_stop'}
        regression.validate_repeat_shutdown(normal, [])
        changes = [{'alive_at_first_shutdown': False}, {'joined': False}, {'exit_reason': 'unexpected_thread_exit'}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                regression.validate_repeat_shutdown({**normal, **change}, [])
        with self.assertRaisesRegex(ValueError, 'lost its lifetime memory sampler'):
            regression.validate_repeat_shutdown(normal, ["TypeError('injected ordinary sampler error')"])

    def test_actual_owned_uva_job_uses_generic_allocation_and_separate_typed_request(self):
        """Only the original forty-minute time limit is changed to twenty minutes in this actual-record-derived shape check."""
        path = SCRIPTS / 'tests/fixtures/uva_owned_checkpoint_allocation_6780569.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
            '3a2a63b28228da22e04946d2e4fa717aead3a392123fbf3e12cf42ef092c7b85')
        observed = json.loads(path.read_text())
        self.assertEqual(observed['job_id'], '6780569')
        self.assertEqual(observed['expected']['time_limit'], '00:40:00')
        self.assertEqual(observed['stdout'].count('TimeLimit=00:40:00'), 1)
        raw = observed['stdout'].replace('TimeLimit=00:40:00', 'TimeLimit=00:20:00')
        self.assertEqual(regression.validate_shape({'job': {'stdout': raw}}), 'sl5nw_168')
        self.assertNotIn('gres/gpu:nvidia_rtx_a4500=1', raw)
        self.assertIn('TresPerNode=gres/gpu:nvidia_rtx_a4500:1', raw)
        optional = raw.replace('AllocTRES=cpu=8,mem=64G,node=1,billing=9840,gres/gpu=1',
            'AllocTRES=cpu=8,mem=64G,node=1,billing=9840,gres/gpu=1,gres/gpu:nvidia_rtx_a4500=1')
        self.assertEqual(regression.validate_shape({'job': {'stdout': optional}}), 'sl5nw_168')
        mutations = [lambda text: text.replace('TresPerNode=gres/gpu:nvidia_rtx_a4500:1', 'TresPerNode=gres/gpu:h100:1'),
            lambda text: text.replace('TresPerNode=gres/gpu:nvidia_rtx_a4500:1', 'TresPerNode=gres/gpu:nvidia_rtx_a4500:2'),
            lambda text: text.replace('AllocTRES=cpu=8,mem=64G,node=1,billing=9840,gres/gpu=1',
                'AllocTRES=cpu=8,mem=64G,node=1,billing=9840,gres/gpu=2'),
            lambda text: text.replace('gres/gpu:nvidia_rtx_a4500=1', 'gres/gpu:nvidia_rtx_a4500=2'),
            lambda text: text.replace('gres/gpu:nvidia_rtx_a4500=1', 'gres/gpu:h100=1')]
        for mutation in mutations:
            with self.subTest(change=mutation), self.assertRaises(ValueError):
                regression.validate_shape({'job': {'stdout': mutation(optional)}})

    def test_explicit_profile_checks_capture_declaration_before_staging(self):
        """The optional profile is frozen explicitly; an actual old source without its declaration is rejected."""
        self.arguments.ownership_profile = regression.capacity.UVA_CGROUP_V2_OWNERSHIP_PROFILE['kind']
        unit = self.prepare()
        self.assertEqual(regression.read(unit / 'regression_plan.json')['ownership_profile'],
            regression.capacity.UVA_CGROUP_V2_OWNERSHIP_PROFILE)
        old = Path('/p/rlprojects/RND_worktrees/2026-11-01-05-05_memory-capture-helper_source-commit-7f5bf87a8400a73afff08098c1422642ecad69f9')
        with self.assertRaisesRegex(ValueError, 'explicit version-two ownership capability'):
            regression.require_capture_version(old, regression.capacity.UVA_CGROUP_V2_OWNERSHIP_PROFILE)

    def test_published_detached_profile_helper_is_verified_and_staged_exactly(self):
        """The actual published capture root supplies all staged bytes and the explicit ownership capability."""
        # Check the real detached source before mocking only this unpublished driver's preparation identity.
        capture = Path('/p/rlprojects/RND_worktrees/2026-11-01-06-41_memory-capture-helper_source-commit-8fd2187fbe1b4a4f0907c64d012da15438d1bd02')
        self.assertEqual(regression.CAPTURE_HELPER_COMMIT, '8fd2187fbe1b4a4f0907c64d012da15438d1bd02')
        self.assertEqual(regression.deployment.git_root(str(capture), regression.CAPTURE_HELPER_COMMIT,
            regression.validation.Evidence()), capture)
        profile = regression.capacity.UVA_CGROUP_V2_OWNERSHIP_PROFILE
        regression.require_capture_version(capture, profile)
        relative = ['10_jax_exploration_platform/scripts/atari_pixel_campaign/' + name for name in regression.HELPERS]
        self.assertEqual(len(relative), 15)
        source_receipt = regression.training.source_receipt
        expected = source_receipt(capture, regression.CAPTURE_HELPER_COMMIT, relative)

        # For example, the committed capacity helper becomes the identical code/device_memory_limit.py artifact.
        self.arguments.capture_helper_root = capture
        self.arguments.ownership_profile = profile['kind']
        unit = self.prepare()
        plan = regression.read(unit / 'regression_plan.json')
        self.assertEqual(plan['helpers'], expected)
        self.assertEqual(plan['ownership_profile'], profile)
        for name, digest in expected['files'].items():
            self.assertEqual(hashlib.sha256((unit / 'code' / Path(name).name).read_bytes()).hexdigest(), digest)

        # Only the unpublished driver/core identity is synthetic; capture-source verification remains real.
        def receipt(root, commit, paths):
            """Verify actual published capture bytes while retaining the separate synthetic driver identity."""
            return source_receipt(root, commit, paths) if Path(root) == capture else self.source_receipt(root, commit, paths)
        with patch.object(regression.training, 'source_receipt', side_effect=receipt), \
                patch.object(regression.diagnostic, 'core_identity', return_value=plan['core']):
            self.assertEqual(regression.verify_static(unit), plan)
            path = unit / 'code/device_memory_limit.py'
            path.write_bytes(path.read_bytes() + b'\n# Changed staged fixture bytes.\n')
            with self.assertRaisesRegex(ValueError, 'staged diagnostic helper changed'):
                regression.verify_static(unit)
        self.assertNotIn('jax', sys.modules)
        self.assertNotIn('numpy', sys.modules)

    def test_bootstrap_own_process_proof_is_saved_before_any_gres_resolution(self):
        """A version-two metadata rejection cannot erase own-process evidence or reach a GPU selection guess."""
        self.arguments.ownership_profile = regression.capacity.UVA_CGROUP_V2_OWNERSHIP_PROFILE['kind']
        unit = self.prepare()
        def rejected(plan, process, job, step, path):
            """Model a preserved own-process record followed by an explicit pre-GRES failure."""
            regression.write(path, {'kind': 'synthetic rejected own-process metadata', 'job': job, 'step': step,
                'uid': process['uid'], 'own_process_metadata': True})
            raise PermissionError('injected own cgroup metadata rejection')
        with patch.object(regression.capacity, 'capture_v2_ownership', side_effect=rejected) as capture, self.assertRaises(PermissionError):
            self.admit(unit)
        capture.assert_called_once()
        self.assertEqual(regression.read(unit / 'allocation_bootstrap_cgroup_v2.json')['step'], '0')
        self.assertTrue((unit / 'allocation_bootstrap.json').is_file())
        self.assertFalse((unit / 'allocation_resolution.json').exists())
        self.assertFalse((unit / 'allocation_admission.json').exists())


if __name__ == '__main__':
    unittest.main()
