"""Check observation-only changes and fresh selected-Python children on owned CPU."""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location('focused_' + name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


diagnostic = load('diagnose_initial_child_identity')
control = load('direct_common_timing_controller')


class InitialChildIdentityTests(unittest.TestCase):
    def test_each_predicate_is_preserved_without_repair(self):
        observed = {'pid': 22, 'parent_pid': 11, 'command': ['actual'], 'affinity': [3], 'process_start_time': '123'}
        for key, wrong, predicate in [('parent_pid', 12, 'parent_pid_matches'),
                ('command', ['different'], 'command_matches'), ('affinity', [4], 'inherited_affinity_matches')]:
            value = copy.deepcopy(observed);value[key] = wrong
            proof = control.child_identity_observation(value, ['actual'], [3], 11, 22)
            self.assertEqual(proof['observed_child'], value)
            self.assertFalse(proof['predicate_results'][predicate])
            self.assertFalse(proof['all_predicates_match'])
            self.assertFalse(proof['registration_performed_by_this_observation'])

    def test_existing_controller_checks_and_functions_are_unchanged(self):
        root = SCRIPTS.parents[2]
        original = subprocess.check_output(['git', '-C', str(root), 'show',
            '765480d990cff5ae556914e3b74afba0472f64ea:10_jax_exploration_platform/scripts/atari_pixel_campaign/direct_common_timing_controller.py'])
        old = {node.name: node for node in ast.parse(original).body if isinstance(node, ast.FunctionDef)}
        new = {node.name: node for node in ast.parse((SCRIPTS / 'direct_common_timing_controller.py').read_bytes()).body
            if isinstance(node, ast.FunctionDef)}
        changed = [name for name in old if ast.dump(old[name], include_attributes=False) != ast.dump(new[name], include_attributes=False)]
        self.assertEqual(changed, ['supervise'])
        tree = copy.deepcopy(new['supervise'])
        class RemoveObservationPublications(ast.NodeTransformer):
            # The new startup observer is explicit; original checks outside it remain exact.
            def visit_Assign(self, node):
                if isinstance(node.value, ast.Call) and any(isinstance(target, ast.Name) and target.id == 'startup_started'
                        for target in node.targets):
                    return None
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) \
                        and node.value.func.id == 'wait_for_exact_child_startup':
                    return None
                return node
+
            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'publish' \
                        and any(isinstance(value, ast.Constant) and value.value in {
                            'direct_controller_initial_child_observation.json', 'direct_controller_pre_registration_child_observation.json'}
                            for value in ast.walk(node)):
                    return None
                return node
        tree = RemoveObservationPublications().visit(tree)
        self.assertEqual(ast.dump(tree, include_attributes=False), ast.dump(old['supervise'], include_attributes=False))

    def run_cpu_fixture(self, *, mismatch=False):
        temporary = tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        source = diagnostic.digest_file(diagnostic.__file__)
        controller = diagnostic.digest_file(diagnostic.CONTROLLER)
        image = diagnostic.digest_file(diagnostic.SELECTED_PYTHON)
        actual = diagnostic.load_controller(controller);reader = actual.read_process
        initial_seen = []
        def first_child_only(pid, uid):
            value = reader(pid, uid)
            if mismatch and pid != os.getpid() and not initial_seen:
                initial_seen.append(pid);value = {**value, 'command': ['explicit_synthetic_initial_mismatch']}
            return value
        with patch.object(actual, 'read_process', side_effect=first_child_only), \
                patch.object(diagnostic, 'load_controller', return_value=actual):
            return diagnostic.run_diagnostic(Path(temporary.name), source, controller, image, cpu_count=1)

    def test_actual_selected_interpreter_ack_and_wait_on_one_owned_cpu(self):
        result = self.run_cpu_fixture()
        self.assertTrue(result['passed'], result)
        self.assertTrue(result['first_observation']['all_predicates_match'])
        self.assertTrue(result['acknowledged_observation']['all_predicates_match'])
        self.assertTrue(result['actual_wait_verified'])
        self.assertEqual(result['actual_wait_returncode'], 0)
        self.assertTrue(result['CPU_affinity_restored'])
        for observation in (result['first_observation'], result['acknowledged_observation']):
            self.assertFalse(observation['controller_owns_gpu_standing_descriptor'])
            self.assertFalse(observation['producer_owns_gpu_standing_descriptor'])
            self.assertEqual(observation['snapshot_scope'], 'CPU-only owned child; no standing GPU descriptor')

    def test_later_ack_does_not_replace_a_synthetic_first_mismatch(self):
        result = self.run_cpu_fixture(mismatch=True)
        self.assertFalse(result['passed'])
        self.assertFalse(result['first_observation']['all_predicates_match'])
        self.assertTrue(result['acknowledged_observation']['all_predicates_match'])
        self.assertFalse(result['release_sent_after_both_positive_observations'])
        self.assertTrue(result['actual_wait_verified'])
        self.assertEqual(result['actual_wait_returncode'], 2)


if __name__ == '__main__':
    unittest.main()
