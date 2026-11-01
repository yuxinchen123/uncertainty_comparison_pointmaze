"""Exercise capacity exclusions with synthetic bytes and mocked APIs, never real GPU calls."""

import copy
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import device_memory_limit as capacity
import post_comparison_validation as validation
import validate_training_deployment as deployment

SPEC = importlib.util.spec_from_file_location('capacity_deployment_fixtures', SCRIPTS / 'tests/test_validate_training_deployment.py')
FIXTURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURES)
MIB = 2 ** 20


def varint(value):
    """Encode synthetic unsigned protobuf values without importing protobuf."""
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def field(number, value):
    """Encode exact integer or length-delimited fixture fields."""
    if isinstance(value, bytes):
        return varint(number * 8 + 2) + varint(len(value)) + value
    return varint(number * 8) + varint(value)


def native_map(in_use=800 * MIB, peak=800 * MIB):
    """Represent native failure-time chunks and exact allocator counters before unwinding."""
    chunk = field(1, 0x100000000) + field(2, in_use) + field(3, in_use) + field(8, 1)
    free = field(1, 0x100000000 + in_use) + field(2, 50 * MIB)
    stats = field(1, 100) + field(2, in_use) + field(3, peak) + field(4, 50 * MIB)
    return field(1, b'GPU_0_bfc') + field(3, chunk) + field(3, free) + field(5, stats)


class DeviceMemoryLimitTests(unittest.TestCase):
    """Keep native allocation evidence, complete fitting points and independent target proof distinct."""

    def setUp(self):
        """Reuse actual validators with tiny Git, package and backend fixtures."""
        self.fixture = FIXTURES.TrainingDeploymentTests('test_memory_limit_target_validates_boundary_backend_and_preserves_selected_count')
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.directory = self.fixture.directory
        self.specification = copy.deepcopy(self.fixture.specification)

    def write(self, path, value):
        """Write only a synthetic fixture artifact and return its path."""
        return self.fixture.write(path, value)

    def build(self, *, fitting=1, cuda_status=0):
        """Represent one-copy success, actual two-copy allocator failure and an independent full repeat."""
        spec = self.specification
        points = [self.fixture.target_record('packing-point', count, 'capacity_fitting_' + str(count), 1.)
                  for count in (1, 2) if count <= fitting]
        repeat = self.fixture.target_record('packing-point', fitting, 'capacity_repeat', 1.)
        result_path = self.fixture.target_record('packing-point', fitting * 2, 'capacity_excluded', 1.)
        self.unit = result_path.parent
        self.original_result = json.loads(result_path.read_text())
        result_path.unlink()
        plan = json.loads((self.unit / 'measurement_plan.json').read_text())
        ledger = self.write(self.fixture.run / 'slurm/submitted_jobids_aces.txt', None)
        ledger.write_text('100\n')
        self.ledger_source = ledger
        frozen_ledger = self.unit / 'capacity_owned_job_ledger.txt'
        frozen_ledger.write_bytes(ledger.read_bytes())
        plan.update(python=sys.executable, hypothesis_count_contribution=0, selected_by_producer=False,
            window_protocol='complete groups/report windows; requested minimum and actual count recorded',
            jobs_submitted_by_producer=False, device_memory_limit_ledger_source=capacity.file_receipt(ledger),
            device_memory_limit_protocol={**capacity.PROTOCOL,
            'allocation_kind': 'slurm', 'cluster': 'aces', 'owned_job_ledger': capacity.file_receipt(frozen_ledger)})
        self.fixture.write_prepared_plan(self.unit, plan)
        process = {'hostname': 'ac049', 'pid': self.original_result['hardware']['pid'], 'uid': 56626,
            'process_start_time': self.original_result['hardware']['process_start_time'], 'parent_pid': 999,
            'cpu_affinity': list(range(8)), 'cgroup': '3:memory:/slurm/uid_56626/job_100/step_0/task_0\n'}
        environment = {'CUDA_VISIBLE_DEVICES': 'GPU-target', 'SLURM_CONF': '/var/spool/slurmd/conf-cache/slurm.conf',
            'SLURM_JOB_ID': '100', 'SLURM_STEP_ID': '0',
            'SLURM_JOB_GPUS': '4', 'SLURM_STEP_GPUS': '4', 'SLURM_GPUS_ON_NODE': '1', 'SLURM_CPUS_PER_TASK': '8'}
        owned = {'kind': 'slurm', 'cluster': 'aces', 'ledger': plan['device_memory_limit_protocol']['owned_job_ledger'],
            'job_id': '100', 'step_id': '0', 'process': process, 'environment': environment,
            'job': {'command': ['scontrol', 'show', 'job', '-d', '100'], 'returncode': 0, 'stderr': '',
                'stdout': 'JobId=100 JobState=RUNNING UserId=u.sl359075(56626) NodeList=ac049 NumNodes=1 NumCPUs=8 AllocTRES=cpu=8,mem=32G,node=1,gres/gpu=1,gres/gpu:h100=1\n Nodes=ac049 GRES=gpu:h100:1(IDX:4)\n'},
            'step': {'command': ['scontrol', 'show', 'step', '-d', '100.0'], 'returncode': 0, 'stderr': '',
                     'stdout': 'StepId=100.0 State=RUNNING UserId=u.sl359075(56626) NodeList=ac049 TRES=cpu=8,node=1,gres/gpu=1,gres/gpu:h100=1'},
            'node': {'command': ['scontrol', 'show', 'node', '-d', 'ac049'], 'returncode': 0, 'stderr': '',
                     'stdout': 'NodeName=ac049 Gres=gpu:h100:8'}}
        owned['physical_mapping'] = self.mapping()
        parent = {'schema_version': 1, 'kind': 'device_memory_limit_parent',
            'process': {**process, 'pid': 999, 'parent_pid': 998, 'process_start_time': '999'},
            'plan': capacity.file_receipt(self.unit / 'measurement_plan.json'), 'started_at_seconds': 1000.}
        terminal = {'schema_version': 1, 'kind': 'device_memory_limit_child_exit', 'parent': parent['process'],
            'child_pid': process['pid'], 'returncode': 1, 'wait_completed': True, 'finished_at_seconds': 1005.,
            'parent_requested_termination': False, 'timeout_used': False,
            'command': [sys.executable, str(self.unit / 'code/measure_selected.py'), '--prepared-unit', str(self.unit), '--instrumented-child']}
        exception = {'error': "JaxRuntimeError('RESOURCE_EXHAUSTED: Out of memory while trying to allocate 100.00MiB.')",
                     'exception_type': 'JaxRuntimeError', 'traceback': 'synthetic original configured learner allocation failure'}
        native = ('W bfc_allocator.cc:590] Allocator (GPU_0_bfc) ran out of memory trying to allocate 100.00MiB '
                  '(rounded to 104857600) from the lower end requested by op jit_update\n').encode()
        (self.unit / 'capacity_failure_stderr.bin').write_bytes(native)
        (self.unit / 'capacity_native_memory_map.pb').write_bytes(native_map())
        allocator = {'bytes_limit': 850 * MIB, 'peak_bytes_in_use': 800 * MIB, 'bytes_in_use': 200 * MIB}
        parsed = capacity.native_request(exception, native, native_map(), allocator)
        driver = self.directory / 'libcuda.so.synthetic-driver'
        driver.write_bytes(b'synthetic actual mapped CUDA driver')
        native_backend = self.directory / 'xla_cuda_plugin.so'
        native_backend.write_bytes(b'synthetic mapped backend TF_BFC_MEMORY_DUMP')
        shared_driver = self.unit / 'capacity_driver/captured_driver' / driver.name
        shared_backend = self.unit / 'capacity_backend/xla_cuda_plugin.so'
        for original, copied in ((driver, shared_driver), (native_backend, shared_backend)):
            copied.parent.mkdir(parents=True)
            copied.write_bytes(original.read_bytes())
        driver_capture = {'schema_version': 1, 'kind': 'captured_mapped_cuda_driver_image', 'pid': process['pid'],
            'original': capacity.file_receipt(driver), 'captured': capacity.file_receipt(shared_driver), 'bytes': driver.stat().st_size}
        backend_capture = {'schema_version': 1, 'kind': 'captured_mapped_cuda_backend_image', 'pid': process['pid'],
            'original': capacity.file_receipt(native_backend), 'captured': capacity.file_receipt(shared_backend), 'bytes': native_backend.stat().st_size}
        mapped_rows = '0000-1000 r-xp 0000 00:00 0 ' + str(native_backend) + '\n'
        device_free = 50 * MIB if cuda_status == 2 else 790 * MIB
        diagnostic = {'pid': process['pid'], 'context': 12345, 'jax_local_hardware_id': 0, 'cuda_device_handle': 0,
            'gpu_uuid': 'GPU-target', 'pci_address': '0000:14:00.0', 'driver_library': capacity.file_receipt(driver),
            'api_return_codes': {key: 0 for key in ('cuCtxGetCurrent', 'cuCtxGetDevice', 'cuCtxSynchronize', 'cuDeviceGetUuid_v2', 'cuDeviceGetPCIBusId')},
            'post_api_return_codes': {key: 0 for key in ('cuCtxGetCurrent', 'cuCtxGetDevice', 'cuCtxSynchronize')},
            'context_after': 12345, 'cuda_device_handle_after': 0,
            'requested_bytes': 100 * MIB, 'cuMemAlloc_v2': cuda_status, 'cuMemFree_v2': 0 if cuda_status == 0 else None,
            'started_at_seconds': 1002., 'finished_at_seconds': 1003.,
            'before': {'returncode': 0, 'total_bytes': 1000 * MIB, 'free_bytes': device_free, 'time_seconds': 1001.},
            'after': {'returncode': 0, 'total_bytes': 1000 * MIB, 'free_bytes': device_free, 'time_seconds': 1003.},
            'context_created_or_reset': False, 'allocator_settings_changed': False, 'training_retried': False}
        raw_counter = '0\n'
        counters = [{'path': str(path), 'raw': raw_counter, 'sha256': hashlib.sha256(raw_counter.encode()).hexdigest(),
            'counters': {'failcnt': 0}} for path in capacity.host_counter_paths(process['cgroup'])]
        memory = {'gpu_uuid': 'GPU-target', 'total_bytes': 1000 * MIB, 'used_bytes': 1000 * MIB - device_free,
            'free_bytes': device_free, 'timestamp_seconds': 1001., 'processes': [{'pid': process['pid'], 'bytes': 200 * MIB}]}
        observation = {'schema_version': 1, 'kind': 'instrumented_device_memory_limit_observation', **capacity.ZERO_CREDIT,
            'protocol': plan['device_memory_limit_protocol'], 'source_commit': self.fixture.commit,
            'original_exception': exception, 'native_allocation': parsed, 'cuda_allocation_check': diagnostic,
            'native_map_original_name': 'capacity_native_map_GPU_0_bfc.1000000000', 'failure_allocator_stats': allocator,
            'pre_probe_capture_time_seconds': 1000.5, 'ownership_before': owned, 'ownership_after': copy.deepcopy(owned),
            'native_backend': {'pid': process['pid'], 'library': capacity.file_receipt(native_backend), 'required_marker': 'TF_BFC_MEMORY_DUMP',
                'package': 'jax-cuda12-pjrt', 'version': self.fixture.installed['packages']['jax-cuda12-pjrt'],
                'mapped_image_rows': mapped_rows, 'mapped_image_rows_sha256': hashlib.sha256(mapped_rows.encode()).hexdigest(),
                'capture': backend_capture}, 'captured_driver_image': driver_capture,
            'host_memory_before': counters, 'host_memory_after': copy.deepcopy(counters),
            'memory_samples': [{'device': copy.deepcopy(memory), 'allocator': allocator} for _ in range(2)], 'sampler_errors': [],
            'loaded_gpu_libraries': self.original_result['loaded_gpu_libraries'], 'failure_progress': {'state': 'compile_warmup'}}
        observation['loaded_gpu_libraries']['loaded_shared_libraries'].append(str(driver))
        observation['memory_samples'][0]['device']['timestamp_seconds'] = 1000.1
        observation['memory_samples'][-1]['device']['timestamp_seconds'] = 1004.
        failure = {**exception, 'source_commit': self.fixture.commit, 'mode': 'packing-point', 'failure_stage': 'producer_exception'}
        for name, value in [('capacity_parent.json', parent), ('capacity_child_exit.json', terminal),
                            ('capacity_observation.json', observation), ('failure.json', failure)]:
            self.write(self.unit / name, value)
        self.seal()
        spec.update(packing_stop_rule='instrumented_device_memory_limit', packing_memory_limit_record=str(self.unit / 'device_memory_limit.json'),
            packing_records=list(map(str, points)), packing_confirmation_records=[str(repeat)], selected_copies=fitting)
        # Correctness and confirmation remain independent, complete executions at the selected fitting count.
        spec['correctness_record'] = str(self.fixture.target_record('correctness', fitting, 'capacity_correctness'))
        spec['confirmation_record'] = str(self.fixture.target_record('confirmation', fitting, 'capacity_confirmation'))
        selected = json.loads(json.dumps(self.fixture.selected))
        selected['learner']['n_copies'] = fitting
        spec['copy_seeds'] = list(range(1001, 1001 + fitting))
        manifest = {'schema_version': 1, 'kind': 'final_policy_seed_blocks', 'units': {spec['unit_id']: {
            'unit_id': spec['unit_id'], 'hardware_class': 'h100', 'copy_seeds': spec['copy_seeds'],
            'configuration_digest': validation.canonical_digest(selected)}}}
        self.write(Path(spec['seed_manifest']), manifest)
        return spec

    def mapping(self):
        """Represent an explicit root-owned ACES cache, numeric GRES roster and selected physical device join."""
        prefix = '/var/spool/slurmd/conf-cache/'
        raw = {'slurm': 'GresTypes=gpu\nTaskPlugin=task/cgroup,task/affinity\nProctrackType=proctrack/cgroup\n',
            'gres': 'Name=gpu Type=h100 File=/dev/nvidia[0-7]\n',
            'cgroup': 'ConstrainDevices=yes\nConstrainCores=yes\n'}
        running = 'SLURM_CONF=/etc/slurm/slurm.conf\nTaskPlugin=task/cgroup,task/affinity\nProctrackType=proctrack/cgroup\n'
        return {'slurm_conf': prefix + 'slurm.conf', 'configuration': [
            {'path': prefix + kind + '.conf', 'kind': kind, 'owner_uid': 0, 'mode': 0o644, 'regular_file': True,
             'symlink': False, 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'filtered_bytes': text,
             'filtered_sha256': hashlib.sha256(text.encode()).hexdigest()} for kind, text in raw.items()],
            'profile': capacity.CONFIGURATION_PROFILES['aces'],
            'directory_chain': [{'path': str(path), 'owner_uid': 0, 'mode': 0o755, 'directory': True, 'symlink': False}
                for path in reversed((Path(prefix), *Path(prefix).parents))],
            'running_configuration': {'command': ['scontrol', 'show', 'config'], 'returncode': 0,
                'filtered_bytes': running, 'filtered_sha256': hashlib.sha256(running.encode()).hexdigest()},
            'other_configuration_paths': [{'path': '/etc/slurm/' + kind + '.conf', 'exists': False} for kind in raw],
            'hardware_query': {'command': ['nvidia-smi', '-i', 'GPU-target', '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader,nounits'],
                'returncode': 0, 'stdout': 'GPU-target, 00000000:14:00.0\n', 'stderr': ''},
            'proc_information': {'path': '/proc/driver/nvidia/gpus/0000:14:00.0/information',
                'raw': 'GPU UUID: GPU-target\nBus Location: 0000:14:00.0\nDevice Minor: 4\n'},
            'device_file': {'path': '/dev/nvidia4', 'owner_uid': 0, 'character_device': True,
                'symlink': False, 'major': 195, 'minor': 4}}

    def seal(self):
        """Rehash deliberate fixture mutations so validation must inspect semantics, not just stale hashes."""
        plan = json.loads((self.unit / 'measurement_plan.json').read_text())
        value = {'schema_version': 1, 'kind': 'device_memory_limit_capacity_exclusion', **capacity.ZERO_CREDIT,
            'protocol': plan['device_memory_limit_protocol'], 'artifacts': {name: capacity.file_receipt(self.unit / name) for name in capacity.FILENAMES}}
        self.write(self.unit / 'device_memory_limit.json', value)

    def change(self, name, function):
        """Change synthetic raw evidence and preserve a valid outer byte binding."""
        path = self.unit / name
        value = json.loads(path.read_text())
        function(value)
        self.write(path, value)
        self.seal()

    def test_one_copy_fits_two_copy_failure_never_becomes_a_point(self):
        """Close the one-to-two boundary with one successful repeat and zero credit for failed work."""
        spec = self.build()
        result = validation.validate_packing(spec, validation.Evidence())
        self.assertTrue(result['complete'])
        self.assertEqual(result['selected_copies'], 1)
        self.assertEqual([point['copies'] for point in result['points']], [1])
        self.assertEqual(result['capacity_exclusion']['outcome'], 'configured_gpu_allocator_limit')
        self.assertFalse(result['capacity_exclusion']['diagnostic_allocation_proves_full_fit'])
        self.assertEqual(result['capacity_exclusion']['throughput_credit'], 0)
        self.assertFalse((self.unit / 'result.json').exists())

    def test_exact_next_doubling_after_two_fitting_counts(self):
        """Keep the earlier complete 1/2 points and select repeated 2 after failed 4."""
        spec = self.build(fitting=2, cuda_status=2)
        result = validation.validate_packing(spec, validation.Evidence())
        self.assertTrue(result['complete'])
        self.assertEqual(result['selected_copies'], 2)
        self.assertEqual(result['capacity_exclusion']['copies'], 4)

    def test_target_deployment_independently_rebuilds_capacity_exclusion(self):
        """The final target verifies the failed boundary separately and retains every completed numerical gate."""
        spec = self.build()
        result = deployment.validate_deployment(spec)
        self.assertEqual(result['configuration']['learner']['n_copies'], 1)
        excluded = [row for row in result['executions'] if row['mode'] == 'device-memory-capacity-exclusion']
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]['packing_point_credit'], 0)

    def test_portable_reader_requires_all_exact_bytes_after_original_paths_are_removed(self):
        """The original failed bytes, native protobuf, driver image and Git helpers all use exact staging entries."""
        self.build()
        self.fixture.specification = self.specification
        self.fixture.test_strict_resolver_preserves_deployment_receipt_after_original_paths_are_removed()

    def test_old_helper_receipts_do_not_require_the_new_capability(self):
        """Pinned historical callers keep their old closure; new callers must declare the new dependency."""
        name = deployment.PREFIX + 'device_memory_limit.py'
        required = {name, deployment.PREFIX + 'measure_selected.py'}
        old = {deployment.PREFIX + 'post_comparison_validation.py': b'OLD_CALLER = True\n'}
        self.assertNotIn(name, validation.helper_requirements(required, old))
        new = {deployment.PREFIX + 'post_comparison_validation.py': b"DEVICE_MEMORY_LIMIT_MODULE = 'device_memory_limit.py'\n"}
        self.assertIn(name, validation.helper_requirements(required, new))

    def test_missing_repeat_stays_incomplete_and_failure_path_cannot_be_a_point(self):
        """The separate failed size cannot substitute for any completed training window."""
        spec = self.build()
        spec['packing_confirmation_records'] = []
        result = validation.validate_packing(spec, validation.Evidence())
        self.assertFalse(result['complete'])
        self.assertEqual(result['required_confirmation_counts'], [1])
        spec['packing_records'].append(spec['packing_memory_limit_record'])
        with self.assertRaises((ValueError, KeyError)):
            validation.validate_packing(spec, validation.Evidence())

    def test_original_default_and_completed_headroom_rules_do_not_accept_exclusions(self):
        """No historical receipt or default rule is reinterpreted by the new capability."""
        spec = self.build()
        spec.pop('packing_stop_rule')
        with self.assertRaisesRegex(ValueError, 'explicit instrumented'):
            validation.validate_packing(spec, validation.Evidence())
        spec['packing_stop_rule'] = 'measured_memory_headroom'
        with self.assertRaisesRegex(ValueError, 'explicit instrumented'):
            validation.validate_packing(spec, validation.Evidence())

    def test_parent_terminal_and_host_failures_are_rejected(self):
        """Ordinary nonzero exit and unchanged real host counters are mandatory."""
        self.build()
        path = self.unit / 'capacity_child_exit.json'
        original = path.read_bytes()
        for key, value in [('returncode', -9), ('returncode', 0), ('returncode', True), ('wait_completed', False),
                           ('timeout_used', True), ('parent_requested_termination', True), ('child_pid', 99999)]:
            path.write_bytes(original)
            self.change(path.name, lambda record: record.update({key: value}))
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                capacity.validate_exclusion(self.specification['packing_memory_limit_record'], self.specification, validation.Evidence())
        path.write_bytes(original)
        self.change('capacity_observation.json', lambda value: value['host_memory_after'][0].update(
            raw='1\n', sha256=hashlib.sha256(b'1\n').hexdigest(), counters={'failcnt': 1}))
        with self.assertRaisesRegex(ValueError, 'host-memory'):
            capacity.validate_exclusion(self.specification['packing_memory_limit_record'], self.specification, validation.Evidence())

    def test_device_identity_context_contamination_and_counter_changes_fail(self):
        """Reject unsupported device errors, changed identities, lost sampling and another GPU process."""
        self.build()
        path = self.unit / 'capacity_observation.json'
        original = path.read_bytes()
        changes = [lambda value: value['cuda_allocation_check'].update(gpu_uuid='GPU-foreign'),
            lambda value: value['cuda_allocation_check']['api_return_codes'].update(cuCtxSynchronize=700),
            lambda value: value['cuda_allocation_check'].update(cuMemAlloc_v2=700),
            lambda value: value['cuda_allocation_check'].update(context_created_or_reset=True),
            lambda value: value['cuda_allocation_check'].update(allocator_settings_changed=True),
            lambda value: value.update(sampler_errors=['lost sample']),
            lambda value: value['memory_samples'][0]['device']['processes'].append({'pid': 888, 'bytes': MIB}),
            lambda value: value['memory_samples'][0]['device']['processes'][0].update(bytes=1001 * MIB),
            lambda value: value['memory_samples'][0]['allocator'].update(bytes_in_use=900 * MIB),
            lambda value: value['ownership_after']['process'].update(process_start_time='999999'),
            lambda value: value['failure_allocator_stats'].update(bytes_limit=900 * MIB)]
        for index, change in enumerate(changes):
            path.write_bytes(original)
            self.change(path.name, change)
            with self.subTest(index=index), self.assertRaises(ValueError):
                capacity.validate_exclusion(self.specification['packing_memory_limit_record'], self.specification, validation.Evidence())

    def test_native_map_covers_unwinding_and_rejects_fragmentation(self):
        """Failure-time native bytes survive buffer release; incomplete or fragmented evidence is insufficient."""
        spec = self.build()
        result = capacity.validate_exclusion(spec['packing_memory_limit_record'], spec, validation.Evidence())
        self.assertEqual(result['outcome'], 'configured_gpu_allocator_limit')
        path = self.unit / 'capacity_native_memory_map.pb'
        path.write_bytes(native_map(in_use=700 * MIB))
        self.seal()
        with self.assertRaisesRegex(ValueError, 'fragmentation'):
            capacity.validate_exclusion(spec['packing_memory_limit_record'], spec, validation.Evidence())
        for invalid in (b'', b'\x08\x80', native_map() + field(1, b'GPU_0_bfc')):
            path.write_bytes(invalid)
            self.seal()
            with self.subTest(invalid=invalid[:10]), self.assertRaises(ValueError):
                capacity.validate_exclusion(spec['packing_memory_limit_record'], spec, validation.Evidence())

    def test_native_error_cannot_be_generic_host_or_conflicting(self):
        """The diagnostic probe's own failure and a generic OOM message cannot manufacture native proof."""
        spec = self.build()
        path = self.unit / 'capacity_failure_stderr.bin'
        original = path.read_bytes()
        for invalid in (b'out of memory', original.replace(b'GPU_0_bfc', b'GPU_host_bfc'),
                        original + original, original.replace(b'104857600', b'104857856')):
            path.write_bytes(invalid)
            self.seal()
            with self.subTest(invalid=invalid[:20]), self.assertRaises(ValueError):
                capacity.validate_exclusion(spec['packing_memory_limit_record'], spec, validation.Evidence())

    def test_completed_repeat_stays_full_size_with_fifteen_percent_free(self):
        """The failed size cannot relax any complete fitting-count window or memory requirement."""
        spec = self.build()
        path = Path(spec['packing_confirmation_records'][0])
        original = path.read_bytes()
        for kind in ('partial', 'headroom', 'reuse', 'size'):
            value = json.loads(original)
            if kind == 'partial': value['rounds'].pop()
            elif kind == 'headroom':
                value['rounds'][0]['memory']['device_memory_samples'][0].update(used_bytes=900 * MIB, free_bytes=100 * MIB)
            elif kind == 'reuse': value['hardware'].update(self.original_result['hardware'])
            else: value['configuration']['learner']['n_envs'] = 64
            path.write_text(json.dumps(value))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validation.validate_packing(spec, validation.Evidence())

    def test_source_capability_ledger_and_raw_hashes_cannot_change(self):
        """Old generic failures, omitted helper bytes and changed owned ledgers never gain capacity status."""
        spec = self.build()
        path = self.unit / 'measurement_plan.json'
        original = path.read_bytes()
        for kind in ('old', 'capability', 'source'):
            value = json.loads(original)
            if kind == 'old': value.pop('device_memory_limit_protocol')
            elif kind == 'capability': value['device_memory_limit_protocol']['producer_capability'] = 'old_generic_failure'
            else: value['source_commit'] = 'f' * 40
            self.fixture.write_prepared_plan(self.unit, value)
            self.seal() if kind != 'old' else None
            with self.subTest(kind=kind), self.assertRaises((ValueError, KeyError)):
                capacity.validate_exclusion(spec['packing_memory_limit_record'], spec, validation.Evidence())
        self.fixture.write_prepared_plan(self.unit, json.loads(original))
        self.seal()
        (self.unit / 'code/device_memory_limit.py').write_bytes(b'changed producer')
        with self.assertRaisesRegex(ValueError, 'staged'):
            capacity.validate_exclusion(spec['packing_memory_limit_record'], spec, validation.Evidence())

    def test_mapping_uses_explicit_gres_files_instead_of_numeric_minor_guess(self):
        """A selector-to-file join must survive independent raw cache, proc, query and CUDA checks."""
        self.build()
        original = (self.unit / 'capacity_observation.json').read_bytes()
        changes = [lambda value: value['ownership_before']['physical_mapping']['device_file'].update(minor=0),
            lambda value: value['ownership_before']['physical_mapping']['configuration'][1].update(
                filtered_bytes='Name=gpu Type=h100 File=/dev/nvidia[1-8]\n'),
            lambda value: value['ownership_before']['physical_mapping']['proc_information'].update(raw='GPU UUID: GPU-foreign\n'),
            lambda value: value['ownership_before']['physical_mapping']['configuration'][0].update(owner_uid=56626),
            lambda value: value['ownership_before']['physical_mapping']['other_configuration_paths'][0].update(exists=True),
            lambda value: value['cuda_allocation_check'].update(pci_address='0000:15:00.0')]
        for index, change in enumerate(changes):
            (self.unit / 'capacity_observation.json').write_bytes(original)
            self.change('capacity_observation.json', change)
            with self.subTest(index=index), self.assertRaises(ValueError):
                capacity.validate_exclusion(self.specification['packing_memory_limit_record'], self.specification, validation.Evidence())

    def test_native_map_and_sampler_times_cannot_be_stale_or_reordered(self):
        """Native evidence must come from this child and precede the diagnostic allocation."""
        self.build()
        original = (self.unit / 'capacity_observation.json').read_bytes()
        changes = [lambda value: value.update(native_map_original_name='capacity_native_map_GPU_0_bfc.900000000'),
            lambda value: value.update(pre_probe_capture_time_seconds=1004.),
            lambda value: value['memory_samples'][-1]['device'].update(timestamp_seconds=1000.2),
            lambda value: value['native_backend'].update(pid=999),
            lambda value: value.update(schema_version=True)]
        for index, change in enumerate(changes):
            (self.unit / 'capacity_observation.json').write_bytes(original)
            self.change('capacity_observation.json', change)
            with self.subTest(index=index), self.assertRaises(ValueError):
                capacity.validate_exclusion(self.specification['packing_memory_limit_record'], self.specification, validation.Evidence())

    def test_cuda_confirmation_uses_current_context_and_frees_success_without_retry(self):
        """Mock every driver call: no context is created, no device is enumerated and a successful probe is freed."""
        driver_path = self.directory / 'libcuda.so.mock-driver'
        driver_path.write_bytes(b'synthetic driver')
        original_read = Path.read_text
        for status in (0, 2):
            calls = []
            def call(name, *arguments):
                """Populate only actual current-context output pointers in the synthetic API."""
                calls.append(name)
                if name == 'cuCtxGetCurrent': arguments[0]._obj.value = 123
                elif name == 'cuCtxGetDevice': arguments[0]._obj.value = 0
                elif name == 'cuDeviceGetUuid_v2': ctypes.memset(arguments[0], 1, 16)
                elif name == 'cuDeviceGetPCIBusId': arguments[0].value = b'0000:14:00.0'
                elif name == 'cuMemGetInfo_v2':
                    arguments[0]._obj.value, arguments[1]._obj.value = 700 * MIB, 1000 * MIB
                elif name == 'cuMemAlloc_v2':
                    self.assertEqual(arguments[1], 100 * MIB)
                    arguments[0]._obj.value = 1234 if status == 0 else 0
                    return status
                return 0
            class Function:
                """Provide ctypes-compatible attribute slots around a pure Python driver stub."""
                def __init__(self, name):
                    """Record the allowlisted driver API name."""
                    self.name = name
                def __call__(self, *arguments):
                    """Run the synthetic API without loading a library."""
                    return call(self.name, *arguments)
            driver = SimpleNamespace(**{name: Function(name) for name in ('cuCtxGetCurrent', 'cuCtxGetDevice',
                'cuDeviceGetUuid_v2', 'cuDeviceGetPCIBusId', 'cuMemGetInfo_v2', 'cuMemAlloc_v2', 'cuMemFree_v2', 'cuCtxSynchronize')})
            def read(path, *arguments, **keywords):
                """Replace only the mapped-library metadata file with synthetic bytes."""
                return '0000-1000 r-xp 0000 00:00 0 ' + str(driver_path) + '\n' if str(path) == '/proc/self/maps' else original_read(path, *arguments, **keywords)
            with patch.object(capacity.ctypes, 'CDLL', return_value=driver), patch.object(Path, 'read_text', read):
                result = capacity.cuda_allocation_check(SimpleNamespace(local_hardware_id=0), 100 * MIB)
            self.assertEqual(result['cuMemAlloc_v2'], status)
            self.assertEqual(calls.count('cuMemFree_v2'), 1 if status == 0 else 0)
            self.assertEqual(calls.count('cuMemAlloc_v2'), 1)
            self.assertFalse(result['training_retried'])

    def test_symlinked_raw_artifact_is_rejected_even_with_identical_bytes(self):
        """Exact local receipt names cannot refer to another directory through a symlink."""
        spec = self.build()
        path = self.unit / 'capacity_failure_stderr.bin'
        external = self.directory / 'external-identical-stderr.bin'
        external.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(external)
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            capacity.validate_exclusion(spec['packing_memory_limit_record'], spec, validation.Evidence())

    def test_later_owned_submissions_do_not_rewrite_or_invalidate_ledger_snapshot(self):
        """Keep the original ownership snapshot immutable while the live campaign ledger grows normally."""
        spec = self.build()
        original = capacity.validate_exclusion(spec['packing_memory_limit_record'], spec, validation.Evidence())
        self.ledger_source.write_text('100\n101\n')
        self.assertEqual(capacity.validate_exclusion(spec['packing_memory_limit_record'], spec, validation.Evidence()), original)
        (self.unit / 'capacity_owned_job_ledger.txt').write_text('100\n101\n')
        with self.assertRaisesRegex(ValueError, 'artifact'):
            capacity.validate_exclusion(spec['packing_memory_limit_record'], spec, validation.Evidence())

    def test_earlier_low_throughput_gain_still_prevents_a_later_memory_stop(self):
        """A failure at four copies cannot replace the required stop/repeat after a small one-to-two gain."""
        spec = self.build(fitting=2)
        path = Path(spec['packing_records'][1])
        value = json.loads(path.read_text())
        for row in value['rounds']:
            row['seconds'] = 1.95
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, 'earlier below-ten-percent'):
            validation.validate_packing(spec, validation.Evidence())

    def test_real_first_allocation_larger_than_device_capacity_can_bound_next_doubling(self):
        """The real configured request may exceed the device before any allocator chunk existed."""
        spec = self.build(cuda_status=2)
        map_bytes = field(1, b'GPU_0_bfc') + field(5, b'')
        (self.unit / 'capacity_native_memory_map.pb').write_bytes(map_bytes)
        native = (self.unit / 'capacity_failure_stderr.bin').read_bytes().replace(b'100.00MiB', b'1200.00MiB').replace(b'104857600', b'1258291200')
        (self.unit / 'capacity_failure_stderr.bin').write_bytes(native)
        observation = json.loads((self.unit / 'capacity_observation.json').read_text())
        observation['original_exception']['error'] = observation['original_exception']['error'].replace('100.00MiB', '1200.00MiB')
        allocator = {'bytes_limit': 850 * MIB, 'bytes_in_use': 0, 'peak_bytes_in_use': 0}
        observation['failure_allocator_stats'] = allocator
        observation['native_allocation'] = capacity.native_request(observation['original_exception'], native, map_bytes, allocator)
        observation['cuda_allocation_check']['requested_bytes'] = 1200 * MIB
        for sample in observation['memory_samples']:
            sample['allocator'] = allocator
        self.write(self.unit / 'capacity_observation.json', observation)
        failure = json.loads((self.unit / 'failure.json').read_text())
        failure['error'] = observation['original_exception']['error']
        self.write(self.unit / 'failure.json', failure)
        self.seal()
        result = validation.validate_packing(spec, validation.Evidence())
        self.assertEqual(result['selected_copies'], 1)
        self.assertEqual(result['capacity_exclusion']['outcome'], 'physical_device_allocation_failure')

    def test_cuda_context_capacity_and_nvml_physical_total_are_kept_separate(self):
        """CUDA reports context-available bytes; physical memory samples keep the NVML total."""
        spec = self.build()
        def reduce_context(value):
            """Change only the two actual CUDA totals, preserving their independent measurement domain."""
            value['cuda_allocation_check']['before']['total_bytes'] = 950 * MIB
            value['cuda_allocation_check']['after']['total_bytes'] = 950 * MIB
        self.change('capacity_observation.json', reduce_context)
        self.assertTrue(validation.validate_packing(spec, validation.Evidence())['complete'])

    def test_foreign_consistent_owner_or_incomplete_step_allocation_is_rejected(self):
        """Self-consistent foreign metadata cannot replace this campaign's explicit account or step resources."""
        self.build()
        original = (self.unit / 'capacity_observation.json').read_bytes()
        changes = [lambda value: value['ownership_before']['process'].update(uid=123),
            lambda value: value['ownership_before']['job'].update(stdout=value['ownership_before']['job']['stdout'].replace('u.sl359075(56626)', 'foreign(56626)')),
            lambda value: value['ownership_before']['step'].update(stdout=value['ownership_before']['step']['stdout'].replace('gres/gpu=1', 'gres/gpu=0')),
            lambda value: value['ownership_before']['step'].update(stdout=value['ownership_before']['step']['stdout'].replace('cpu=8', 'cpu=4')),
            lambda value: value['ownership_before']['step'].update(stdout=value['ownership_before']['step']['stdout'].replace(' UserId=u.sl359075(56626)', ''))]
        for index, change in enumerate(changes):
            (self.unit / 'capacity_observation.json').write_bytes(original)
            self.change('capacity_observation.json', change)
            with self.subTest(index=index), self.assertRaises(ValueError):
                capacity.validate_exclusion(self.specification['packing_memory_limit_record'], self.specification, validation.Evidence())

    def test_uva_a4500_uses_its_own_owner_cache_and_step_resources(self):
        """The optional A4500 path keeps UVA's account and explicit node configuration separate from ACES."""
        spec = self.build()
        spec['hardware_class'] = 'jaguar03_rtx_a4500'
        for path in map(Path, spec['packing_records'] + spec['packing_confirmation_records']):
            result = json.loads(path.read_text())
            result['hardware_class'] = 'jaguar03_rtx_a4500'
            result['hardware']['hostname'] = 'jaguar03'
            result['hardware']['gpu_query'] = result['hardware']['gpu_query'].replace('NVIDIA H100 PCIe', 'NVIDIA RTX A4500')
            result['execution_identity']['hostname'] = 'jaguar03'
            self.write(path, result)
            self.write(path.parent / 'hardware.json', result['hardware'])
            plan = json.loads((path.parent / 'measurement_plan.json').read_text())
            plan['hardware_class'] = 'jaguar03_rtx_a4500'
            self.fixture.write_prepared_plan(path.parent, plan)
        plan = json.loads((self.unit / 'measurement_plan.json').read_text())
        plan['hardware_class'] = 'jaguar03_rtx_a4500'
        plan['device_memory_limit_protocol']['cluster'] = 'uva'
        uva_ledger = self.write(self.fixture.run / 'slurm/submitted_jobids_uva.txt', None)
        uva_ledger.write_text('100\n')
        plan['device_memory_limit_ledger_source'] = capacity.file_receipt(uva_ledger)
        self.fixture.write_prepared_plan(self.unit, plan)
        hardware = json.loads((self.unit / 'hardware.json').read_text())
        hardware.update(hostname='jaguar03', gpu_query=hardware['gpu_query'].replace('NVIDIA H100 PCIe', 'NVIDIA RTX A4500'))
        self.write(self.unit / 'hardware.json', hardware)
        observation = json.loads((self.unit / 'capacity_observation.json').read_text())
        observation['protocol'] = plan['device_memory_limit_protocol']
        for key in ('ownership_before', 'ownership_after'):
            owned = observation[key]
            owned['cluster'] = 'uva'
            owned['process'].update(hostname='jaguar03', uid=2618919,
                cgroup=owned['process']['cgroup'].replace('uid_56626', 'uid_2618919'))
            owned['environment']['SLURM_CONF'] = '/etc/slurm/slurm.conf'
            for query in ('job', 'step', 'node'):
                owned[query]['command'] = [part.replace('ac049', 'jaguar03') for part in owned[query]['command']]
                owned[query]['stdout'] = owned[query]['stdout'].replace('u.sl359075(56626)', 'sl5nw(2618919)').replace('ac049', 'jaguar03').replace('h100', 'nvidia_rtx_a4500')
            mapping = owned['physical_mapping']
            mapping['slurm_conf'] = '/etc/slurm/slurm.conf'
            mapping['other_configuration_paths'] = []
            mapping['profile'] = capacity.CONFIGURATION_PROFILES['uva']
            mapping['directory_chain'] = [{'path': str(path), 'owner_uid': 0, 'mode': 0o755, 'directory': True, 'symlink': False}
                for path in reversed((Path('/etc/slurm'), *Path('/etc/slurm').parents))]
            for row in mapping['configuration']:
                row['path'] = '/etc/slurm/' + row['kind'] + '.conf'
                row['filtered_bytes'] = row['filtered_bytes'].replace('Type=h100', 'Type=nvidia_rtx_a4500')
                row['sha256'] = row['filtered_sha256'] = hashlib.sha256(row['filtered_bytes'].encode()).hexdigest()
        for key in ('host_memory_before', 'host_memory_after'):
            for row in observation[key]:
                row['path'] = row['path'].replace('uid_56626', 'uid_2618919')
        self.write(self.unit / 'capacity_observation.json', observation)
        parent = json.loads((self.unit / 'capacity_parent.json').read_text())
        parent['process'].update(hostname='jaguar03', uid=2618919,
            cgroup=parent['process']['cgroup'].replace('uid_56626', 'uid_2618919'))
        parent['plan'] = capacity.file_receipt(self.unit / 'measurement_plan.json')
        self.write(self.unit / 'capacity_parent.json', parent)
        terminal = json.loads((self.unit / 'capacity_child_exit.json').read_text())
        terminal['parent'] = parent['process']
        self.write(self.unit / 'capacity_child_exit.json', terminal)
        self.seal()
        self.assertTrue(validation.validate_packing(spec, validation.Evidence())['complete'])

    def test_noncanonical_devices_and_writable_cache_ancestors_are_rejected(self):
        """Aliases and a writable ancestor cannot establish an explicit root-controlled GPU mapping."""
        for name in ('/dev/nvidia00', '/dev/nvidia01', '/dev/nvidia[00-07]'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                capacity.expand_devices(name)
        self.build()
        original = (self.unit / 'capacity_observation.json').read_bytes()
        def running_path(value):
            """Change a self-consistent controller-path record to an unreviewed relationship."""
            row = value['ownership_before']['physical_mapping']['running_configuration']
            row['filtered_bytes'] = row['filtered_bytes'].replace('/etc/slurm/slurm.conf', '/elsewhere/slurm.conf')
            row['filtered_sha256'] = hashlib.sha256(row['filtered_bytes'].encode()).hexdigest()
        changes = [lambda value: value['ownership_before']['physical_mapping']['directory_chain'][1].update(mode=0o777),
                   lambda value: value['ownership_before']['physical_mapping']['directory_chain'][1].update(owner_uid=56626), running_path]
        for index, change in enumerate(changes):
            (self.unit / 'capacity_observation.json').write_bytes(original)
            self.change('capacity_observation.json', change)
            with self.subTest(index=index), self.assertRaises(ValueError):
                capacity.validate_exclusion(self.specification['packing_memory_limit_record'], self.specification, validation.Evidence())

    def test_duplicate_json_keys_cannot_replace_original_capacity_fields(self):
        """New protocol records cannot hide two versions of a field behind JSON last-key behavior."""
        spec = self.build()
        path = self.unit / 'device_memory_limit.json'
        original = path.read_text()
        path.write_text(original.replace('"schema_version": 1', '"schema_version": 0, "schema_version": 1', 1))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            capacity.validate_exclusion(spec['packing_memory_limit_record'], spec, validation.Evidence())

    def test_failed_attempt_cannot_change_prepared_rounds_or_grouped_work(self):
        """The failed size retains the original full seven-round ten-update-or-longer measurement plan."""
        spec = self.build()
        original = json.loads((self.unit / 'measurement_plan.json').read_text())
        for key, changed in [('rounds', 1), ('rounds', True), ('minimum_updates_per_round', 1),
                             ('iterations', 1), ('window_protocol', 'partial updates')]:
            plan = copy.deepcopy(original)
            plan[key] = changed
            self.fixture.write_prepared_plan(self.unit, plan)
            parent = json.loads((self.unit / 'capacity_parent.json').read_text())
            parent['plan'] = capacity.file_receipt(self.unit / 'measurement_plan.json')
            self.write(self.unit / 'capacity_parent.json', parent)
            self.seal()
            with self.subTest(key=key, changed=changed), self.assertRaisesRegex(ValueError, 'complete-group/report'):
                capacity.validate_exclusion(spec['packing_memory_limit_record'], spec, validation.Evidence())

    def test_expected_uid_cgroup_and_real_distinct_parent_birth_are_required(self):
        """A foreign UID cgroup or a fabricated self-parent chain cannot attest this owned process."""
        spec = self.build()
        observation_path = self.unit / 'capacity_observation.json'
        observation = json.loads(observation_path.read_text())
        changed = copy.deepcopy(observation['ownership_before'])
        changed['process']['cgroup'] = changed['process']['cgroup'].replace('uid_56626', 'uid_999')
        plan = json.loads((self.unit / 'measurement_plan.json').read_text())
        with self.assertRaisesRegex(ValueError, 'cgroup'):
            capacity.validate_ownership(changed, plan)
        parent_path = self.unit / 'capacity_parent.json'
        original_parent = json.loads(parent_path.read_text())
        terminal_path = self.unit / 'capacity_child_exit.json'
        original_terminal = json.loads(terminal_path.read_text())
        for birth in ('0', 'invalid', True):
            parent = copy.deepcopy(original_parent)
            parent['process']['process_start_time'] = birth
            terminal = copy.deepcopy(original_terminal)
            terminal['parent'] = parent['process']
            self.write(parent_path, parent)
            self.write(terminal_path, terminal)
            self.seal()
            with self.subTest(birth=birth), self.assertRaisesRegex(ValueError, 'parent birth'):
                capacity.validate_exclusion(spec['packing_memory_limit_record'], spec, validation.Evidence())
        parent, terminal = copy.deepcopy(original_parent), copy.deepcopy(original_terminal)
        pid = observation['ownership_before']['process']['pid']
        parent['process']['pid'] = pid
        terminal['parent'] = parent['process']
        for key in ('ownership_before', 'ownership_after'):
            observation[key]['process']['parent_pid'] = pid
        self.write(parent_path, parent)
        self.write(terminal_path, terminal)
        self.write(observation_path, observation)
        self.seal()
        with self.assertRaisesRegex(ValueError, 'distinct parent/child'):
            capacity.validate_exclusion(spec['packing_memory_limit_record'], spec, validation.Evidence())

    def test_node_local_driver_and_backend_can_disappear_after_shared_capture(self):
        """CPU readers use captured bytes and never reopen either original mapped node-local image."""
        spec = self.build()
        observation = json.loads((self.unit / 'capacity_observation.json').read_text())
        Path(observation['cuda_allocation_check']['driver_library']['path']).unlink()
        Path(observation['native_backend']['library']['path']).unlink()
        self.assertTrue(validation.validate_packing(spec, validation.Evidence())['complete'])
        shared = Path(observation['captured_driver_image']['captured']['path'])
        shared.write_bytes(b'changed captured driver')
        with self.assertRaisesRegex(ValueError, 'recorded bytes'):
            capacity.validate_exclusion(spec['packing_memory_limit_record'], spec, validation.Evidence())


if __name__ == '__main__':
    unittest.main()
