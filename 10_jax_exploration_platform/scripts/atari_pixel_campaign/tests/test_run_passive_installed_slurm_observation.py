"""Verify the passive wrapper's exact source, actual-binding composition and failure artifacts."""

import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_passive_installed_slurm_observation as wrapper
from test_passive_installed_slurm_observation import SyntheticQuery


class PassiveWrapperTests(unittest.TestCase):
    """Use a detached temporary Git source and synthetic owned allocation rows only."""

    def setUp(self):
        """Commit the actual three producer files locally without any remote or scheduler command."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        for name in wrapper.FILES:
            path = self.source / wrapper.PREFIX / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((Path(wrapper.__file__).parent / name).read_bytes())
        self.git = ['git', '-C', str(self.source)]
        subprocess.run(['git', 'init', '-q', str(self.source)], check=True)
        subprocess.run(self.git + ['add', '.'], check=True)
        subprocess.run(self.git + ['-c', 'user.name=Metadata Fixture', '-c', 'user.email=metadata@example.invalid',
            'commit', '-qm', 'Record synthetic passive wrapper source'], check=True)
        self.commit = subprocess.check_output(self.git + ['rev-parse', 'HEAD'], text=True).strip()
        subprocess.run(self.git + ['checkout', '--detach', '-q', 'HEAD'], check=True)
        self.campaign = self.root / 'campaign'
        self.campaign.mkdir()

    def prepare(self):
        """Create one actual immutable preparation against the synthetic detached source."""
        return wrapper.prepare(self.campaign, 'synthetic_passive_observation', self.source, self.commit)

    def bind(self, unit, contents=b'123\n'):
        """Model the root fsyncing its actual returned held identifier before the binding command."""
        path = self.campaign / 'slurm/submitted_jobids_uva.txt'
        path.parent.mkdir()
        with path.open('xb') as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        return wrapper.bind_held_identifier(unit, '123', 'sl5nw_168')

    def allocation(self):
        """Describe synthetic request-one/allocate-two/step-one zero-GPU actual-format evidence."""
        process = {'hostname': 'jaguar03', 'pid': 500, 'uid': 2618919, 'effective_uid': 2618919,
            'uid_fields': [2618919] * 4, 'parent_pid': 499, 'process_start_time': '10000',
            'raw_proc_stat': '500 (python) ' + ' '.join(['R', '499', *['0'] * 17, '10000']) + '\n',
            'cgroup': '0::/system.slice/slurmstepd.scope/job_123/step_0/user/task_0\n', 'cpu_affinity': [0]}
        job = ('JobId=123 UserId=sl5nw(2618919) JobState=RUNNING NodeList=jaguar03 Account=sl5nw_base '
            'Partition=gpu QOS=csresnolim NumNodes=1 NumTasks=1 CPUs/Task=1 TimeLimit=00:05:00 Requeue=0 '
            'Reservation=sl5nw_168 NumCPUs=2 ReqTRES=cpu=1,mem=2G,node=1 AllocTRES=cpu=2,mem=2G,node=1,billing=1')
        step = 'StepId=123.0 State=RUNNING NodeList=jaguar03 UserId=2618919 TRES=cpu=1,mem=2G,node=1'
        return {'process': process, 'job': job, 'step': step}

    def execution(self, unit, fixture, collector=None):
        """Patch only actual process/query boundaries while retaining all real static and resource validators."""
        stack = ExitStack()
        for module, name in [(wrapper, wrapper.FILENAME), (wrapper.passive, wrapper.FILES[1]), (wrapper.ownership, wrapper.FILES[2])]:
            stack.enter_context(patch.object(module, '__file__', str(unit / 'code' / name)))
        stack.enter_context(patch.object(wrapper.ownership, 'own_process', return_value=fixture['process']))
        stack.enter_context(patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'SLURM_STEP_ID': '0', 'SLURM_CPUS_PER_TASK': '1'}, clear=True))

        def query(command, job, step):
            """Return only the two exact admitted fixture commands without executing either."""
            self.assertEqual((job, step), ('123', '0'))
            choices = {('scontrol', 'show', 'job', '-d', '123'): fixture['job'],
                ('scontrol', 'show', 'step', '-d', '123.0'): fixture['step']}
            self.assertIn(tuple(command), choices)
            return {'command': command, 'returncode': 0, 'capture_status': 'completed',
                'stdout': choices[tuple(command)], 'stderr': ''}

        stack.enter_context(patch.object(wrapper, 'owned_query', side_effect=query))
        stack.enter_context(patch.object(wrapper.passive, 'controller_query', return_value={
            'capture_status': 'completed', 'returncode': 0, 'stderr_byte_count': 0, 'filter_status': 'parsed',
            'fields': [{'name': 'NodeName', 'value': 'jaguar03'}, {'name': 'ThreadsPerCore', 'value': '2'}]}))
        default = {'schema_version': 1, 'kind': wrapper.passive.KIND, 'collection_completed': True, **wrapper.ZERO_CREDIT}
        collector_mock = stack.enter_context(patch.object(wrapper.passive, 'collect_owned_metadata',
            side_effect=collector, return_value=default))
        stack.enter_context(patch.object(wrapper.ownership, 'run', side_effect=AssertionError('old full collector must never run')))
        stack.enter_context(patch.object(wrapper.ownership, 'configuration_query', side_effect=AssertionError('old configuration query must never run')))
        return stack, collector_mock

    def test_standard_preparation_has_exact_three_sources_and_null_identity(self):
        """One immutable standard packet binds every loaded source before an identifier exists."""
        unit = self.prepare()
        plan = wrapper.verify_static(unit)
        self.assertEqual(len(plan['source']['files']), 3)
        self.assertTrue(all(plan[name] is None for name in ('job_id', 'reservation', 'gpu_uuid')))
        self.assertFalse(plan['physical_gres_mapping_verified'])
        self.assertFalse(plan['admission_granted'])
        for name in ('experiment_background.md', 'manifest.yaml', 'config_resolved.yaml', 'command.txt',
                'passive_plan.json', 'observation_plan.json', 'passive_prepared.json'):
            self.assertTrue((unit / name).is_file())
        self.assertEqual((unit / 'experiment_background.md').read_text().count('## '), 4)
        self.assertIn('/usr/bin/python3 -B ', (unit / 'command.txt').read_text())
        with self.assertRaises(ValueError):
            self.prepare()

    def test_detached_clean_source_and_approved_dependency_bytes_are_required(self):
        """A mutable branch or changed dependency cannot become the prepared helper closure."""
        subprocess.run(self.git + ['checkout', '-qb', 'mutable_fixture'], check=True)
        with self.assertRaises(ValueError):
            self.prepare()
        subprocess.run(self.git + ['checkout', '--detach', '-q', self.commit], check=True)
        path = self.source / wrapper.PREFIX / wrapper.FILES[1]
        path.write_bytes(path.read_bytes() + b'\n# changed fixture\n')
        subprocess.run(self.git + ['add', '.'], check=True)
        subprocess.run(self.git + ['-c', 'user.name=Metadata Fixture', '-c', 'user.email=metadata@example.invalid',
            'commit', '-qm', 'Change synthetic passive library'], check=True)
        changed = subprocess.check_output(self.git + ['rev-parse', 'HEAD'], text=True).strip()
        with self.assertRaisesRegex(ValueError, 'independently reviewed'):
            wrapper.prepare(self.campaign, 'changed_source', self.source, changed)

    def test_staged_artifact_and_source_mutation_are_rejected(self):
        """A byte change in a staged file refuses execution even when the detached source stays clean."""
        unit = self.prepare()
        path = unit / 'code' / wrapper.FILES[1]
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'staged passive artifact'):
            wrapper.verify_static(unit)

    def test_boolean_preparation_schema_is_not_an_integer_version(self):
        """Equivalent Python Boolean equality cannot substitute for the literal preparation schema."""
        unit = self.prepare()
        path = unit / 'passive_prepared.json'
        value = json.loads(path.read_text())
        value['schema_version'] = True
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            wrapper.verify_static(unit)

    def test_loaded_modules_must_all_come_from_the_exact_staged_directory(self):
        """A library imported from a mutable source root refuses before any scheduler command."""
        unit = self.prepare()
        with patch.object(wrapper, '__file__', str(unit / 'code' / wrapper.FILENAME)), \
                patch.object(wrapper.ownership, '__file__', str(unit / 'code' / wrapper.FILES[2])):
            with self.assertRaisesRegex(ValueError, 'staged wrapper, library'):
                wrapper.verify_static(unit, execution=True)
            with patch.object(wrapper.passive, '__file__', str(unit / 'code' / wrapper.FILES[1])):
                wrapper.verify_static(unit, execution=True)

    def test_data_parent_alias_is_rejected_before_writing(self):
        """Preparation cannot write through a campaign data-directory symlink."""
        outside = self.root / 'outside'
        outside.mkdir()
        (self.campaign / 'data').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(list(outside.iterdir()), [])

    def test_actual_id_binding_requires_fsynced_canonical_ledger_exactly_once(self):
        """No guessed or duplicated identifier can produce the separate returned-ID binding."""
        unit = self.prepare()
        with self.assertRaises(FileNotFoundError):
            wrapper.bind_held_identifier(unit, '123', 'sl5nw_168')
        with self.assertRaises(ValueError):
            self.bind(unit, b'123\n123\n')
        self.assertFalse((unit / 'held_identifier_binding.json').exists())

    def test_binding_keeps_null_plan_and_all_zero_claims(self):
        """The actual ID is separate immutable evidence and never changes the original plan."""
        unit = self.prepare()
        before = (unit / 'passive_plan.json').read_bytes()
        binding = self.bind(unit)
        self.assertEqual(binding['job_id'], '123')
        self.assertFalse(binding['admission_granted'])
        self.assertEqual((unit / 'passive_plan.json').read_bytes(), before)

    def test_actual_validator_composition_retains_separate_observations_and_zero_credit(self):
        """The narrowed collector calls existing strict validators and never the old file collector."""
        unit = self.prepare()
        self.bind(unit)
        stack, passive_call = self.execution(unit, self.allocation())
        with stack:
            result = wrapper.run(unit)
        self.assertTrue(result['completed'])
        self.assertEqual((result['job_id'], result['step_id']), ('123', '0'))
        self.assertEqual(passive_call.call_count, 1)
        owned = json.loads((unit / 'observation.json').read_text())
        self.assertTrue(owned['allocation_binding_passed'])
        self.assertEqual(owned['process']['cpu_affinity'], [0])
        self.assertEqual(result['ownership_observation_sha256'], hashlib.sha256((unit / 'observation.json').read_bytes()).hexdigest())
        self.assertFalse(result['configured_isolation_qualified'])
        self.assertFalse(result['admission_granted'])
        self.assertFalse((unit / 'failure.json').exists())

    def test_foreign_uid_and_unowned_ledger_refuse_before_query(self):
        """Foreign identities and missing ownership evidence cannot reach a live scheduler request."""
        unit = self.prepare()
        self.bind(unit)
        fixture = self.allocation()
        fixture['process']['uid_fields'] = [999] * 4
        stack, passive_call = self.execution(unit, fixture)
        with stack, patch.object(wrapper, 'owned_query') as query, self.assertRaises(ValueError):
            wrapper.run(unit)
        query.assert_not_called()
        passive_call.assert_not_called()
        self.assertFalse(json.loads((unit / 'observation.json').read_text())['completed'])
        self.assertTrue((unit / 'failure.json').is_file())

    def test_wrong_requested_gpu_or_step_memory_preserves_failed_raw_allocation(self):
        """A contradictory actual resource row refuses before the passive file reads."""
        unit = self.prepare()
        self.bind(unit)
        fixture = self.allocation()
        fixture['job'] = fixture['job'].replace('ReqTRES=cpu=1,mem=2G,node=1', 'ReqTRES=cpu=1,mem=2G,node=1,gres/gpu=1')
        stack, passive_call = self.execution(unit, fixture)
        with stack, self.assertRaises(ValueError):
            wrapper.run(unit)
        passive_call.assert_not_called()
        raw = json.loads((unit / 'observation.json').read_text())
        self.assertIn('gres/gpu=1', raw['job_query']['stdout'])
        self.assertFalse(raw['completed'])
        self.assertTrue((unit / 'failure.json').exists())

    def test_changed_live_ledger_refuses_before_any_query(self):
        """A previously bound snapshot cannot replace the current canonical owned-ID check."""
        unit = self.prepare()
        self.bind(unit)
        (self.campaign / 'slurm/submitted_jobids_uva.txt').write_bytes(b'999\n')
        stack, passive_call = self.execution(unit, self.allocation())
        with stack, patch.object(wrapper, 'owned_query') as query, self.assertRaises(ValueError):
            wrapper.run(unit)
        query.assert_not_called()
        passive_call.assert_not_called()
        self.assertTrue((unit / 'observation.json').exists())

    def test_wrong_step_memory_is_rejected_by_the_unchanged_validator(self):
        """The current job's two-GiB allocation cannot mask a larger observed numeric step."""
        unit = self.prepare()
        self.bind(unit)
        fixture = self.allocation()
        fixture['step'] = fixture['step'].replace('mem=2G', 'mem=64G')
        stack, passive_call = self.execution(unit, fixture)
        with stack, self.assertRaises(ValueError):
            wrapper.run(unit)
        passive_call.assert_not_called()
        self.assertFalse(json.loads((unit / 'observation.json').read_text())['completed'])

    def test_two_logical_processors_require_actual_two_thread_topology(self):
        """A two-processor charge is not silently called one physical core without actual node evidence."""
        unit = self.prepare()
        self.bind(unit)
        stack, passive_call = self.execution(unit, self.allocation())
        wrong = {'capture_status': 'completed', 'returncode': 0, 'stderr_byte_count': 0, 'filter_status': 'parsed',
            'fields': [{'name': 'NodeName', 'value': 'jaguar03'}, {'name': 'ThreadsPerCore', 'value': '1'}]}
        with stack, patch.object(wrapper.passive, 'controller_query', return_value=wrong), self.assertRaises(ValueError):
            wrapper.run(unit)
        passive_call.assert_not_called()
        self.assertEqual(json.loads((unit / 'observation.json').read_text())['cpu_topology_query'], wrong)

    def test_passive_failure_is_saved_before_refusal(self):
        """A returned unsupported passive observation remains intact with no success result."""
        unit = self.prepare()
        self.bind(unit)
        failed = {'kind': wrapper.passive.KIND, 'collection_completed': False, 'error': 'synthetic missing metadata', **wrapper.ZERO_CREDIT}
        stack, passive_call = self.execution(unit, self.allocation(), collector=lambda owned: failed)
        with stack, self.assertRaises(ValueError):
            wrapper.run(unit)
        self.assertEqual(json.loads((unit / 'passive_observation.json').read_text()), failed)
        self.assertTrue((unit / 'failure.json').exists())
        self.assertFalse((unit / 'result.json').exists())

    def test_nonzero_passive_claim_is_rejected(self):
        """A wrong future library result cannot grant admission through the wrapper."""
        unit = self.prepare()
        self.bind(unit)
        wrong = {'kind': wrapper.passive.KIND, 'collection_completed': True, **wrapper.ZERO_CREDIT, 'case_credit': 1}
        stack, passive_call = self.execution(unit, self.allocation(), collector=lambda owned: wrong)
        with stack, self.assertRaises(ValueError):
            wrapper.run(unit)
        self.assertFalse((unit / 'result.json').exists())

    def test_owned_query_scope_and_pipe_limits_are_enforced(self):
        """Only exact admitted query arguments are accepted and no full over-limit output is buffered."""
        child = SyntheticQuery(stdout=b'a' * 30, hold_open=True)
        with patch.object(wrapper.subprocess, 'Popen', return_value=child) as create, \
                patch.dict(wrapper.passive.LIMITS, {'query_bytes': 16}):
            query = wrapper.owned_query(['scontrol', 'show', 'job', '-d', '123'], '123', '0')
            self.assertEqual(query['capture_status'], 'response_limit_exceeded')
            self.assertEqual(query['stdout_captured_byte_count'], 17)
            self.assertTrue(child.killed)
            with self.assertRaises(ValueError):
                wrapper.owned_query(['scontrol', 'show', 'job', '-d', '999'], '123', '0')
            self.assertEqual(create.call_count, 1)

    def test_owned_query_timeout_and_bad_encoding_are_not_accepted(self):
        """An incomplete query or invalid raw byte sequence cannot supply an accepted allocation row."""
        child = SyntheticQuery(hold_open=True)
        with patch.object(wrapper.subprocess, 'Popen', return_value=child), \
                patch.dict(wrapper.passive.LIMITS, {'query_timeout_seconds': 0.005}):
            query = wrapper.owned_query(['scontrol', 'show', 'step', '-d', '123.0'], '123', '0')
        self.assertEqual(query['capture_status'], 'timeout')
        child = SyntheticQuery(stdout=b'\xff')
        with patch.object(wrapper.subprocess, 'Popen', return_value=child):
            query = wrapper.owned_query(['scontrol', 'show', 'step', '-d', '123.0'], '123', '0')
        self.assertEqual(query['capture_status'], 'unsupported_text_encoding')


if __name__ == '__main__':
    unittest.main()
