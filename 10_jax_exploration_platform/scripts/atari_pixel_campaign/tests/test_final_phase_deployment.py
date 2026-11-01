"""Verify the current final protocol on a separate synthetic target and through exact portable copies."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRIPTS), str(Path(__file__).resolve().parent)]
import post_comparison_validation as validation
import portable_evidence
import validate_training_deployment as deployment
import test_validate_training_deployment as historical_fixtures
from final_phase_fixtures import FinalPhaseFixture


class CurrentTargetTests(unittest.TestCase):
    """Retain the old raw-reader fixtures and add complete current global and target evidence."""

    @classmethod
    def setUpClass(cls):
        """Create independent target processes and actual-shape library/source metadata without arrays."""
        cls.data = historical_fixtures.TrainingDeploymentTests('runTest')
        cls.data.setUp()
        cls.public_patch = patch.object(validation, 'JAXTARI_COMMIT', deployment.JAXTARI_COMMIT)
        cls.public_patch.start()
        cls.current = FinalPhaseFixture(source_root=cls.data.source, asset_record=cls.data.global_asset_record)
        current = validation.validate_final_verification(cls.current.specification)
        cls.data.write(cls.data.final_path, current)
        cls.specification = copy.deepcopy(cls.data.specification)

        # Only the final confirmation uses eleven by twenty; all successful packing points remain seven by ten.
        path = Path(cls.specification['confirmation_record'])
        result = json.loads(path.read_bytes())
        result['rounds'] = [copy.deepcopy(result['rounds'][index % 7]) for index in range(11)]
        for index, pair in enumerate(result['rounds']):
            pair['order'] = ['reference', 'candidate'] if index % 2 == 0 else ['candidate', 'reference']
            for arm in ('reference', 'candidate'):
                pair[arm].update(iterations=20, training_transitions=2 * 128 * 128 * 20)
        result['protocol'] = {'rounds': 11, 'requested_minimum_updates_per_round': 20, 'actual_updates_per_round': 20}
        global_result = json.loads(Path(cls.current.specification['contexts']['h100']['confirmation_record']).read_bytes())
        result['synchronized_per_update_latency'] = global_result['synchronized_per_update_latency']
        memory = cls.data.fixture.row(copies=2, gpu='GPU-target')['memory']
        result['synchronized_per_update_latency_memory'] = memory
        cls.data.write(path.parent / 'single_update_latency_memory.json', memory)
        plan = json.loads((path.parent / 'measurement_plan.json').read_bytes())
        plan.update(rounds=11, minimum_updates_per_round=20, iterations=20,
            final_confirmation_protocol=copy.deepcopy(validation.FINAL_PROTOCOL),
            single_update_latency_protocol={arm: global_result["synchronized_per_update_latency"][arm]["protocol"] for arm in ("reference", "candidate")},
            completed_study=cls.current.completed_study, completed_study_manifest=None)
        cls.data.write_prepared_plan(path.parent, plan)
        cls.data.write(path, result)
        learning = cls.data.target_record('learning-check', 2, 'target_short_learning')
        cls.current.complete_learning(learning)
        cls.specification.update(learning_record=str(learning), learning_analysis_record=str(learning.parent / 'independent_analysis.json'))
        cls.original = {}
        for root in (cls.data.directory, cls.data.fixture.directory, cls.current.root, cls.current.raw.directory):
            cls.original.update({path: path.read_bytes() for path in root.rglob('*') if path.is_file() and '.git' not in path.parts})

    @classmethod
    def tearDownClass(cls):
        """Delete only the synthetic sources/evidence and restore patched public-source metadata."""
        cls.current.close()
        cls.public_patch.stop()
        cls.data.doCleanups()

    def tearDown(self):
        """Restore exact temporary fixture bytes after each negative test."""
        for path, raw in self.original.items():
            if not path.exists() or path.read_bytes() != raw:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)

    def test_current_target_requires_its_own_final_confirmation_learning_and_packing(self):
        """The target preserves global implementation controls and confirms its own two-policy packing."""
        receipt = deployment.validate_deployment(self.specification)
        self.assertEqual(receipt['configuration']['learner']['n_copies'], 2)
        self.assertEqual(receipt['combination']['paired_rounds'], 11)
        self.assertEqual(receipt['short_learning']['updates_per_arm'], 128)
        self.assertIs(receipt['short_learning']['learning_equivalence_established'], False)
        self.assertFalse(any(name in sys.modules for name in ('jax', 'numpy', 'torch')))

    def test_global_learning_process_is_reserved_and_historical_absence_is_preserved(self):
        """A target cannot reuse global learning, while an older context keeps its original process set."""
        final = json.loads(self.data.final_path.read_bytes())
        context = final['specification']['contexts']['h100']
        learning = validation.execution_identity(json.loads(Path(context['learning_record']).read_bytes()))
        used = deployment.global_processes(final, deployment.DeploymentEvidence())
        self.assertIn(learning, used)
        historical = copy.deepcopy(final)
        del historical['specification']['contexts']['h100']['learning_record']
        self.assertEqual(deployment.global_processes(historical, deployment.DeploymentEvidence()), used - {learning})

    def test_current_target_rejects_legacy_rounds_and_missing_independent_learning_analysis(self):
        """Old seven-by-ten target confirmation and absent learning proof cannot admit a current worker."""
        path = Path(self.specification['confirmation_record'])
        result = json.loads(path.read_bytes())
        result['rounds'] = result['rounds'][:7]
        self.data.write(path, result)
        with self.assertRaisesRegex(ValueError, 'eleven'):
            deployment.validate_deployment(self.specification)
        path.write_bytes(self.original[path])
        specification = {name: value for name, value in self.specification.items() if name != 'learning_analysis_record'}
        with self.assertRaises(ValueError):
            deployment.validate_deployment(specification)

    def test_current_target_complete_portable_copy_and_missing_mapping_rejection(self):
        """Rebuild current global and target gates from exact copied files and relocated detached Git roots."""
        path = self.data.write(self.data.directory / 'current_target_specification.json', self.specification)
        expected = deployment.validate_deployment(self.specification, specification_path=path)
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            manifest = {'schema_version': 1, 'kind': 'portable_raw_evidence', 'files': [], 'git_roots': []}
            for root, name, commit in ((self.data.source, 'numerical', self.data.commit),
                    (self.data.public, 'jaxtari', deployment.JAXTARI_COMMIT)):
                destination = staging / name
                subprocess.run(['git', 'clone', '-q', '--no-hardlinks', str(root), str(destination)], check=True)
                subprocess.run(['git', '-C', str(destination), 'checkout', '-q', '--detach', commit], check=True)
                manifest['git_roots'].append({'original_root': str(root), 'relative_path': name, 'commit': commit})
            files = {Path(item['path']) for item in expected['evidence']}
            for root in (self.data.directory, self.data.fixture.directory, self.current.root, self.current.raw.directory):
                files |= {item for item in root.rglob('*') if item.is_file() and '.git' not in item.parts}
            for index, source in enumerate(sorted(files)):
                relative = Path('files') / (str(index) + '_' + source.name)
                destination = staging / relative
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, destination)
                manifest['files'].append({'original_path': str(source), 'relative_path': str(relative),
                    'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
            manifest_path = self.data.write(staging / 'manifest.json', manifest)
            resolver = portable_evidence.ExactPathResolver(manifest_path)
            actual = deployment.validate_deployment(self.specification, resolver=resolver, specification_path=path)
            self.assertEqual(actual, expected)
            manifest['files'] = [item for item in manifest['files'] if item['original_path'] != self.specification['learning_analysis_record']]
            self.data.write(manifest_path, manifest)
            with self.assertRaises(ValueError):
                deployment.validate_deployment(self.specification, resolver=portable_evidence.ExactPathResolver(manifest_path), specification_path=path)


if __name__ == '__main__':
    unittest.main()
