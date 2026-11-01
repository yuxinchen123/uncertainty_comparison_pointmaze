"""Reject tiny/component contracts at formal entry points without importing a numerical package."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import post_comparison_validation as validation
import short_learning
import learning_latency_component as component


class LearningLatencyComponentMetadataTests(unittest.TestCase):
    """Check only the explicit diagnostic recipe and distinct formal/diagnostic dispatch."""

    def configuration(self, readout='scheduled', group=1):
        """Use a small metadata fixture rather than claiming this partial dictionary constructs a real trainer."""
        return {'learner': {name: value for name, value in short_learning.COMPONENT_RECIPE.items() if name != 'copy_seeds'},
            'environment': {'num_envs': 1}, 'execution': {'compilation_mode': 'whole_iteration', 'iterations_per_call': group,
                'report_readout': readout, 'report_every_iterations': 4}}

    def plan(self):
        """Keep the original full annealing horizon and a component-only, zero-normalization-warmup recipe."""
        reference, selected = self.configuration(), self.configuration('per_update')
        return {'kind': short_learning.COMPONENT_KIND, 'component_recipe': copy.deepcopy(short_learning.COMPONENT_RECIPE),
            'copy_seeds': [11], 'reference_configuration': reference, 'configuration': selected,
            'learning_protocol': short_learning.component_protocol([reference, selected])}

    def test_formal_contract_keeps_fifty_warmups_and_component_records_actual_zero(self):
        """The approved diagnostic has128 complete updates per arm without changing formal protocol defaults."""
        configurations = [self.configuration(), self.configuration('per_update')]
        formal = short_learning.protocol(configurations)
        component = short_learning.component_protocol(configurations)
        self.assertEqual(formal['normalization_warmup_iterations'], 50)
        self.assertEqual(formal['kind'], short_learning.CONTRACT)
        self.assertEqual(component['normalization_warmup_iterations'], 0)
        self.assertEqual(component['kind'], short_learning.COMPONENT_CONTRACT)
        self.assertEqual(component['updates'], 128)
        self.assertEqual(component['complete_window_updates'], 4)
        self.assertEqual(component['full_training_horizon'], 2_000_000_000)
        self.assertFalse(component['admission_granted'])
        self.assertEqual(component['formal_learning_credit'], 0)

    def test_grouping_is_separate_from_active_readout_and_windows_remain_complete(self):
        """Scheduled/per-update controls remain ungrouped; grouped no-readout rounds up to129 complete updates."""
        for readout in ('scheduled', 'per_update'):
            short_learning.component_protocol([self.configuration(readout)])
            with self.subTest(readout=readout), self.assertRaises(ValueError):
                short_learning.component_protocol([self.configuration(readout, 3)])
        grouped = self.configuration('none', 3)
        protocol = short_learning.component_protocol([grouped, self.configuration('none')])
        self.assertEqual(protocol['updates'], 129)
        self.assertEqual(protocol['complete_window_updates'], 3)
        latency = short_learning.latency_protocol(grouped)
        self.assertEqual(latency['calls'], 9)
        self.assertEqual(latency['report_rows_per_window'], 0)

    def test_formal_run_rejects_component_or_tiny_plan_before_jax_and_memory(self):
        """Renaming the diagnostic entry cannot turn its one-environment/zero-warmup plan into formal learning."""
        for relabel in (False, True):
            plan = self.plan()
            if relabel:
                plan.update(kind='selected_configuration_measurement', mode='learning-check')
                plan.pop('component_recipe')
                plan['learning_protocol'] = short_learning.protocol([plan['reference_configuration'], plan['configuration']])
            with patch.object(short_learning, '_run_with_memory') as implementation, self.assertRaisesRegex(ValueError, 'original full configuration'):
                short_learning.run(None, plan, None, None, None, None, None, None)
            implementation.assert_not_called()
        self.assertNotIn('jax', sys.modules)
        self.assertNotIn('numpy', sys.modules)

    def test_formal_full_configuration_dispatch_retains_the_original_contract(self):
        """The unchanged formal shape continues through the formal entry; this test does not execute a trainer."""
        plan = self.plan()
        plan.update(kind='selected_configuration_measurement', mode='learning-check')
        plan.pop('component_recipe')
        for configuration in (plan['reference_configuration'], plan['configuration']):
            configuration['learner'].update(n_envs=128, num_steps=128, update_epochs=4,
                num_minibatches=4, observation_warmup_iterations=50)
            configuration['environment']['num_envs'] = 128
        plan['learning_protocol'] = short_learning.protocol([plan['reference_configuration'], plan['configuration']])
        with patch.object(short_learning, '_run_with_memory', return_value={'formal_fixture': True}) as implementation:
            self.assertEqual(short_learning.run(None, plan, None, None, None, None, None, None), {'formal_fixture': True})
        self.assertEqual(implementation.call_args.args[-1]['kind'], short_learning.CONTRACT)

    def test_component_dispatch_is_explicit_and_keeps_zero_formal_credit(self):
        """Only the separate entry selects the component implementation and its distinctly labelled result."""
        plan = self.plan()
        with patch.object(short_learning, '_run_with_memory', return_value={'completed': True}) as implementation:
            result = short_learning.run_component(None, plan, None, None, None, None, None, None)
        self.assertEqual(implementation.call_args.args[-1], plan['learning_protocol'])
        self.assertEqual(result['kind'], short_learning.COMPONENT_KIND)
        self.assertEqual(result['formal_learning_credit'], 0)
        self.assertFalse(result['admission_granted'])

    def test_changed_shape_seeds_horizon_or_formal_proof_is_rejected(self):
        """The component cannot expose a generic override for another shape or a formal study proof."""
        mutations = [lambda plan: plan.update(kind='selected_configuration_measurement'),
            lambda plan: plan.update(copy_seeds=[12]),
            lambda plan: plan.update(completed_study={'not': 'accepted here'}),
            lambda plan: plan['configuration']['learner'].update(total_timesteps=256),
            lambda plan: plan['configuration']['learner'].update(n_envs=2),
            lambda plan: plan['configuration']['learner'].update(donate_state=False),
            lambda plan: plan['configuration']['learner'].update(n_copies=True)]
        for mutate in mutations:
            plan = self.plan()
            mutate(plan)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                short_learning.checked_component_protocol(plan)

    def test_existing_final_reader_rejects_renamed_component_record(self):
        """Even an ordinary result.json filename and rehashed preparation retain the forbidden component producer kind."""
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            unit = run / 'data/component'
            unit.mkdir(parents=True)
            plan = {**self.plan(), 'schema_version': 1, 'campaign_run_dir': str(run), 'unit_id': 'component'}
            raw = json.dumps(plan).encode()
            (unit / 'measurement_plan.json').write_bytes(raw)
            (unit / 'measurement_prepared.json').write_text(json.dumps({'plan_sha256': hashlib.sha256(raw).hexdigest()}))
            (unit / 'result.json').write_text(json.dumps({'kind': short_learning.COMPONENT_KIND, 'completed': True}))
            with self.assertRaisesRegex(ValueError, 'original prepared producer plan'):
                validation.measurement_binding(unit / 'result.json', 'learning-check', {}, validation.Evidence())


class ComponentProducerMetadataTests(unittest.TestCase):
    """Exercise actual component preparation and evidence arithmetic with publication/owned hardware mocked explicitly."""

    def setUp(self):
        """Keep all generated artifacts and source-publication assumptions inside a temporary metadata fixture."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = Path('/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673')
        self.commit = 'd28e6735e79788f618acb0ebfde8609db1f776af'
        for name, value in [('assets', {'root': '/fixture/assets', 'manifest': '/fixture/assets_manifest.json', 'sha256': 'a' * 64}),
                ('runtime', {'JAX_ENABLE_X64': 'true', 'JAX_DEFAULT_MATMUL_PRECISION': 'highest',
                             'XLA_FLAGS': '--xla_gpu_exclude_nondeterministic_ops=true'})]:
            (self.root / (name + '.json')).write_text(json.dumps(value))
        self.args = SimpleNamespace(run_dir=self.root, unit_id='tiny_pixel_learning_component_metadata_fixture',
            source_root=self.source, source_commit=self.commit, helper_root=SCRIPTS.parents[2], helper_commit='b' * 40,
            asset_record=self.root / 'assets.json', runtime_environment=self.root / 'runtime.json',
            jaxtari_source_root=self.root, python=Path(sys.executable))
        self.real_source_receipt = component.training.source_receipt

    def source_receipt(self, root, commit, paths):
        """Keep the real frozen numerical proof while modelling only this unpublished component helper publication."""
        if root == self.source:
            return self.real_source_receipt(root, commit, paths)
        return {'source_root': str(root), 'source_commit': commit,
                'files': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}}

    def prepare(self):
        """Stage actual helper/default bytes without claiming that this draft has already been published."""
        with patch.object(component.deployment, 'git_root', side_effect=lambda root, commit, evidence: Path(root)), \
                patch.object(component.training, 'source_receipt', side_effect=self.source_receipt), \
                patch.object(component.assignment, 'resolve_uuid') as resolve:
            unit = component.prepare(self.args)
            resolve.assert_not_called()
        return unit

    def test_component_preparation_is_complete_tiny_and_has_no_guessed_identity_or_zero_transition_claim(self):
        """The recipe uses full frozen defaults with only approved dimensions/readout changed and separate zero formal credit."""
        unit = self.prepare()
        plan = component.read(unit / 'component_plan.json')
        self.assertTrue(all(plan[name] is None for name in ('job_id', 'gpu_uuid', 'reservation')))
        self.assertEqual(plan['learning_protocol']['updates'], 128)
        self.assertEqual(plan['learning_protocol']['normalization_warmup_iterations'], 0)
        self.assertEqual(plan['configuration']['environment']['num_envs'], 1)
        self.assertEqual(plan['configuration']['learner']['total_timesteps'], 2_000_000_000)
        self.assertEqual(plan['shape']['time_limit_minutes'], 30)
        self.assertNotIn('environment_transitions', plan)
        self.assertEqual(plan['formal_learning_credit'], 0)
        self.assertEqual(set(path.name for path in (unit / 'code').iterdir()), set(component.HELPERS))
        self.assertTrue(all((unit / name).is_file() for name in ['manifest.yaml', 'config_resolved.yaml',
            'experiment_background.md', 'command.txt', 'component_prepared.json']))
        self.assertNotIn('jax', sys.modules)

    def test_component_configuration_cannot_change_an_unapproved_original_control(self):
        """Even a rehashed static plan cannot change the original learning rate or staged helper bytes."""
        unit = self.prepare()
        plan = component.read(unit / 'component_plan.json')
        plan['configuration']['learner']['learning_rate'] *= 2
        (unit / 'component_plan.json').write_text(json.dumps(plan))
        (unit / 'component_prepared.json').write_text(json.dumps({'plan': component.ownership.file_receipt(unit / 'component_plan.json')}))
        with patch.object(component.deployment, 'git_root', side_effect=lambda root, commit, evidence: Path(root)), \
                patch.object(component.training, 'source_receipt', side_effect=self.source_receipt), self.assertRaisesRegex(ValueError, 'original tiny recipe'):
            component.verify_static(unit)

    def test_runtime_precision_and_asset_schema_are_checked_before_staging(self):
        """The diagnostic cannot use a different precision or an unbound asset location."""
        (self.root / 'runtime.json').write_text(json.dumps({'JAX_ENABLE_X64': 'false'}))
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse((self.root / 'data').exists())

    def test_thirty_minute_shape_is_separate_and_rejects_changed_gpu_or_resources(self):
        """The tiny learner cannot accidentally reuse the twenty-minute memory diagnostic's shape."""
        raw = ('Account=sl5nw_base Partition=gpu QOS=csresnolim NumNodes=1 NumCPUs=8 NumTasks=1 CPUs/Task=8 '
            'TimeLimit=00:30:00 Requeue=0 TresPerNode=gres/gpu:nvidia_rtx_a4500:1 '
            'ReqTRES=cpu=8,mem=64G,node=1,gres/gpu=1 '
            'AllocTRES=cpu=8,mem=64G,node=1,gres/gpu=1 Reservation=sl5nw_168')
        step = {'stdout': 'TRES=cpu=8,mem=64G,node=1,gres/gpu=1'}
        self.assertEqual(component.validate_job_shape({'job': {'stdout': raw}, 'step': step}), 'sl5nw_168')
        for old, new in [('TimeLimit=00:30:00', 'TimeLimit=00:20:00'), ('mem=64G', 'mem=2G'),
                ('gres/gpu=1', 'gres/gpu=2'), ('nvidia_rtx_a4500:1', 'h100:1')]:
            with self.subTest(change=new), self.assertRaises(ValueError):
                component.validate_job_shape({'job': {'stdout': raw.replace(old, new)}, 'step': step})
        with self.assertRaises(ValueError):
            component.validate_job_shape({'job': {'stdout': raw}, 'step': {'stdout': step['stdout'].replace('mem=64G', 'mem=2G')}})

    def test_pre_jax_backend_records_verified_jaxtari_receipt_not_a_path(self):
        """The real metadata record contains the verified installed source proof and remains serializable before JAX."""
        unit = self.root / 'metadata_preflight'
        unit.mkdir()
        plan = {'runtime_environment': {}, 'asset_record': {}, 'jaxtari_source_root': '/fixture/jaxtari'}
        admission = {'gpu_uuid': 'GPU-fixture', 'job_id': '123', 'process': {'process_start_time': '10000'}}
        source = {'source_root': '/fixture/jaxtari', 'source_commit': 'a' * 40, 'python_files': {'file': 'hash'}}
        with patch.object(component.training, 'enforce_runtime'), patch.object(component.training, 'verify_assets', return_value={'actual_assets': True}), \
                patch.object(component.training, 'installed_gpu_library_receipts', return_value={'library': {'sha256': 'b' * 64}}), \
                patch.object(component.training, 'jaxtari_python_receipt', return_value=source) as verified, \
                patch.object(component.measurement.metadata, 'version', return_value='fixture'), \
                patch.object(component.diagnostic, 'hardware_record', return_value={'hostname': 'jaguar03', 'pid': 500}), \
                patch.object(component.measurement, 'selected_query', side_effect=[[['GPU-fixture', 'NVIDIA RTX A4500', '20000', 'fixture', '[N/A]', '0']], []]):
            unused_assets, unused_libraries, installed = component.preflight(unit, plan, admission)
        verified.assert_called_once_with(Path('/fixture/jaxtari'))
        self.assertEqual(installed['jaxtari_source'], source)
        self.assertEqual(json.loads((unit / 'installed_backend.json').read_bytes()), installed)
        self.assertNotIn('jax', sys.modules)

    def test_primary_and_every_clone_update_are_counted_separately(self):
        """A concrete synthetic record has256 primary updates plus136 additional updates, rather than zero transitions."""
        arms = {arm: {'completed_updates': 128, 'observer_agreement': {'selected_executor_warmup': [{'updates': 4}] * 2,
                'window_updates': 4}, 'replay': {'window_updates': 4}} for arm in ('reference', 'candidate')}
        learning = {'learning_protocol': {'complete_window_updates': 4}, 'arms': arms,
                    'warmup': {arm: [{}, {}] for arm in arms}}
        latency = {arm: {'calls': 12, 'window_agreement': [{'from_iteration': 0, 'to_iteration': 4}] * 3,
            'warmup': [{'updates': 4}] * 2, 'selected_executor_warmup': [{'updates': 4}] * 2} for arm in arms}
        work = component.successful_work_accounting(learning, {'scheduled': latency}, 2)
        self.assertEqual(work['primary_learning_environment_transitions'], 512)
        self.assertEqual(work['additional_complete_updates'], 136)
        self.assertEqual(work['additional_update_environment_transitions'], 272)
        self.assertEqual(work['total_update_environment_transitions'], 784)
        self.assertEqual(work['formal_learning_credit'], 0)

    def test_sampler_records_an_unexpected_ordinary_error(self):
        """A TypeError ends sampling explicitly and cannot silently establish a clean lifetime."""
        memory = component.ComponentMemoryObservation(None, 'GPU-fixture', self.root / 'memory.json')
        with patch.object(memory.stop, 'wait', return_value=False), patch.object(memory, 'sample', side_effect=TypeError('injected')):
            memory._sample()
        self.assertEqual(len(memory.errors), 1)
        self.assertFalse(memory.normal_stop)

    def test_sampler_refuses_unexpected_exit_even_if_the_parent_finish_looks_successful(self):
        """A dead or abnormally completed sampler preserves raw evidence and rejects the component stage."""
        for name, alive, normal in [('dead', False, False), ('abnormal', True, False)]:
            memory = component.ComponentMemoryObservation(None, 'GPU-fixture', self.root / (name + '.json'))
            memory.thread = SimpleNamespace(is_alive=lambda: alive)
            memory.normal_stop = normal
            with patch.object(component.measurement.MemoryObservation, 'finish', return_value={'other_gpu_processes': []}), self.assertRaises(ValueError):
                memory.finish()
            self.assertTrue((self.root / (name + '.json')).is_file())

    def test_unsupported_physical_mapping_is_saved_and_never_reaches_jax(self):
        """An AutoDetect-only or inaccessible mapping remains a pre-JAX failure without an ordinal fallback."""
        unit = self.prepare()
        plan = component.read(unit / 'component_plan.json')
        with patch.object(component, 'verify_static', return_value=plan), \
                patch.object(component, 'admit', side_effect=ValueError('unsupported explicit File mapping')) as admission, \
                patch.object(component, 'compute') as compute, self.assertRaisesRegex(ValueError, 'unsupported explicit File'):
            component.run(unit)
        compute.assert_not_called()
        self.assertEqual(component.read(unit / 'failure.json')['recorded_primary_curve_rows'], 0)
        self.assertTrue(component.read(unit / 'failure.json')['additional_or_unrecorded_work_may_have_occurred'])
        self.assertNotIn('jax', sys.modules)

    def learning_rows_fixture(self):
        """Construct explicitly synthetic128-update raw rows for checking the reader, without any training claim."""
        unit = self.root / 'synthetic_learning_rows'
        unit.mkdir()
        plan = LearningLatencyComponentMetadataTests().plan()
        curves, reports = [], []
        for arm in ('reference', 'candidate'):
            for update in range(1, 129):
                metrics = {name: 0. for name in short_learning.METRICS}
                metrics['steps_per_policy'] = update * 2
                curves.append({'variant': arm, 'update_in_check': update, 'iteration': update,
                    'metrics': metrics, 'optimizer_step': [update], 'copy_seeds': [11], 'floating_state_nonfinite_count': 0,
                    'observation_variance_mean': [1.], 'reward_variance': [1.], 'observation_count': [float(update * 2)],
                    'reward_count': [float(update * 2)], 'boundary_action_histogram': [[1] + [0] * 17],
                    'boundary_action_bins': list(range(18)), 'boundary_action_unknown_count': [0],
                    'boundary_room_histogram': [[1] + [0] * 32], 'boundary_room_bins': list(range(33)),
                    'boundary_room_unknown_count': [0], 'boundary_inventory_sum': [[0] * 4],
                    'environment_key': [[update, update + 1]], 'frame_count': [[update * 2]],
                    'step_count': [[update * 2]], 'reset_count': [[0]]})
                if update % 4 == 0:
                    reports.append({'variant': arm, 'iteration': update, 'window_updates': 4, 'metrics': metrics})
        for name, rows in [('learning_curves.jsonl', curves), ('learning_reports.jsonl', reports)]:
            (unit / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
        precision = {'all_state_finite': True, 'network_dtype': 'float32', 'statistics_dtype': 'float64'}
        arms = {arm: {'completed_updates': 128, 'optimizer_step': [128], 'frozen_target_unchanged': True,
            'replay': {'passed': True, 'agreement_policy': short_learning.REPLAY_AGREEMENT},
            'observer_agreement': {'passed': True, 'agreement_policy': short_learning.OBSERVER_AGREEMENT, 'report_rows': 1},
            'agent_change_squared': [1.], 'predictor_change_squared': [1.],
            'precision': {'initial': copy.deepcopy(precision), 'latest': copy.deepcopy(precision)}} for arm in ('reference', 'candidate')}
        result = {'kind': short_learning.COMPONENT_KIND, 'learning_protocol': plan['learning_protocol'], 'arms': arms,
            'training_start_iteration': {'reference': 0, 'candidate': 0},
            'curves': component.ownership.file_receipt(unit / 'learning_curves.jsonl'),
            'reports': component.ownership.file_receipt(unit / 'learning_reports.jsonl')}
        return unit, plan, result, curves

    def test_learning_reader_requires_actual_changes_and_all128_raw_updates(self):
        """A complete synthetic fixture passes; missing rows, unchanged predictor or wrong precision do not."""
        unit, plan, result, rows = self.learning_rows_fixture()
        component.validate_learning_rows(unit, result, plan)
        for change in ({'predictor_change_squared': [0.]}, {'agent_change_squared': []}, {'frozen_target_unchanged': False}):
            changed = copy.deepcopy(result)
            changed['arms']['candidate'].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                component.validate_learning_rows(unit, changed, plan)
        changed = copy.deepcopy(result)
        changed['arms']['reference']['precision']['latest']['statistics_dtype'] = 'float32'
        with self.assertRaises(ValueError):
            component.validate_learning_rows(unit, changed, plan)
        (unit / 'learning_curves.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows[:-1]))
        result['curves'] = component.ownership.file_receipt(unit / 'learning_curves.jsonl')
        with self.assertRaisesRegex(ValueError, 'omit or repeat'):
            component.validate_learning_rows(unit, result, plan)

    def test_learning_reader_rejects_rehashed_counter_or_boundary_changes(self):
        """Correct file hashes cannot hide a wrong optimizer step, missing action bin or invalid RNG value."""
        unit, plan, result, rows = self.learning_rows_fixture()
        changes = [lambda row: row.update(optimizer_step=[999]), lambda row: row.update(boundary_action_bins=[0]),
            lambda row: row.update(environment_key=[[-1, 0]]), lambda row: row['metrics'].pop('gradient_norm')]
        for change in changes:
            altered = copy.deepcopy(rows)
            change(altered[0])
            (unit / 'learning_curves.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in altered))
            result['curves'] = component.ownership.file_receipt(unit / 'learning_curves.jsonl')
            with self.subTest(change=change), self.assertRaises(ValueError):
                component.validate_learning_rows(unit, result, plan)

    def latency_fixture(self, configurations):
        """Construct explicitly synthetic complete timing/report windows for reader checks only."""
        records = {}
        for arm in ('reference', 'candidate'):
            protocol = short_learning.latency_protocol(configurations[arm])
            window = protocol['complete_window_updates']
            checks = []
            for index in range(3):
                reports = []
                for number in range(1, protocol['report_rows_per_window'] + 1):
                    iteration = index * window + number * protocol['selected_report_every_iterations']
                    reports.append({'iteration': iteration, 'window_updates': 4, 'metrics': {'steps_per_policy': iteration * 2}})
                checks.append({'passed': True, 'from_iteration': index * window, 'to_iteration': (index + 1) * window,
                    'report_rows': len(reports), 'published_reports': reports,
                    'state_and_final_metrics': {'leaves': 1}, 'selected_reports': {'leaves': len(reports)}})
            records[arm] = {'protocol': protocol, 'calls': protocol['calls'], 'seconds_per_synchronized_call': [.25] * protocol['calls'],
                'updates_per_call': 1, 'synchronization_boundary': 'each_original_complete_update',
                'selected_iterations_per_call': configurations[arm]['execution']['iterations_per_call'],
                'median_seconds_per_synchronized_call': .25, 'mean_seconds_per_synchronized_call': .25,
                'median_amortized_seconds_per_update': .25,
                'warmup': [{'compiled_signatures': [1], 'updates': window}] * 2,
                'selected_executor_warmup': [{'compiled_signatures': [1], 'updates': window}] * 2,
                'window_agreement': checks}
        return records

    def test_latency_reader_accepts_only_complete_selected_readout_and_grouped_windows(self):
        """Scheduled/per-update and grouped/no-readout cases keep their real full selected Executor windows."""
        cases = component.configurations(self.source)['latency_cases']
        for name, configurations in cases.items():
            with self.subTest(case=name):
                component.validate_latency_records(self.latency_fixture(configurations), configurations)

    def test_latency_reader_rejects_missing_readout_partial_window_or_changed_summary(self):
        """A timing summary cannot hide a missing actual call or selected scheduled report accumulator."""
        configurations = component.configurations(self.source)['latency_cases']['scheduled']
        changes = [lambda rows: rows['candidate']['seconds_per_synchronized_call'].pop(),
            lambda rows: rows['candidate']['window_agreement'][0]['published_reports'].clear(),
            lambda rows: rows['candidate']['window_agreement'][0].update(to_iteration=3),
            lambda rows: rows['candidate'].update(median_seconds_per_synchronized_call=0.),
            lambda rows: rows['candidate']['selected_executor_warmup'][0].update(updates=1)]
        for change in changes:
            rows = self.latency_fixture(configurations)
            change(rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                component.validate_latency_records(rows, configurations)


if __name__ == '__main__':
    unittest.main()
