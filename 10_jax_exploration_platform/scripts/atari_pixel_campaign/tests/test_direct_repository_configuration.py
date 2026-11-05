"""Refuse effective Git filters and changed source configuration before source inspection."""
import copy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import direct_common_confirmation as guard


class RepositoryConfigurationTests(unittest.TestCase):
    """Use synthetic configuration-query returns without running Git or reading scientific evidence."""

    def setUp(self):
        """Give each check its own baseline and exact fixed child environment."""
        self.inputs = {'/synthetic/.git/config': {'sha256': 'a' * 64, 'exists': True}}
        self.environment = guard.metadata_environment({})
        self.state = mock.patch.object(guard, '_REPOSITORY_CONFIGURATION', {})
        self.state.start()
        self.addCleanup(self.state.stop)

    def test_only_native_empty_no_match_return_is_allowed(self):
        """Native return one without output or errors pins the exact configuration inputs."""
        outcome = subprocess.CompletedProcess([], 1, b'', b'')
        with mock.patch.object(guard, 'repository_configuration_inputs', return_value=self.inputs), \
                mock.patch.object(guard.subprocess, 'run', return_value=outcome) as run:
            value = guard.verify_repository_configuration('/synthetic', self.environment)
        self.assertEqual(value['inputs'], self.inputs)
        self.assertEqual(value['matched_key_names'], [])
        self.assertFalse(value['configured_filters_or_includes'])
        self.assertEqual(run.call_args.args[0], ['git', '-C', '/synthetic', *guard.REPOSITORY_CONFIGURATION_QUERY])
        self.assertEqual(run.call_args.kwargs['env'], self.environment)

    def test_matching_filter_include_or_any_query_error_is_refused(self):
        """Configured keys and every other return/output meaning remain unverified."""
        outcomes = [(0, b'filter.local.clean\0', b''), (0, b'filter.local.process\0', b''),
            (0, b'include.path\0', b''), (0, b'includeif.gitdir:/synthetic/.path\0', b''),
            (0, b'', b''), (1, b'', b'error'), (1, b'unexpected', b''), (128, b'', b'error')]
        for code, output, errors in outcomes:
            with self.subTest(code=code, output=output), \
                    mock.patch.object(guard, 'repository_configuration_inputs', return_value=self.inputs), \
                    mock.patch.object(guard.subprocess, 'run', return_value=subprocess.CompletedProcess([], code, output, errors)), \
                    self.assertRaisesRegex(ValueError, 'configuration query forbid source inspection'):
                guard.verify_repository_configuration('/synthetic', self.environment)
            self.assertEqual(guard._REPOSITORY_CONFIGURATION, {})

    def test_changed_pinned_configuration_is_refused_before_another_query(self):
        """A later changed file identity or digest cannot replace the first observed source configuration."""
        guard._REPOSITORY_CONFIGURATION['/synthetic'] = {'inputs': self.inputs}
        changed = copy.deepcopy(self.inputs)
        changed['/synthetic/.git/config']['sha256'] = 'b' * 64
        with mock.patch.object(guard, 'repository_configuration_inputs', return_value=changed), \
                mock.patch.object(guard.subprocess, 'run') as run, self.assertRaisesRegex(ValueError, 'pinned repository configuration changed'):
            guard.verify_repository_configuration('/synthetic', self.environment)
        run.assert_not_called()

    def test_changed_configuration_during_query_is_refused(self):
        """The actual files are rehashed after the query before any status is allowed."""
        with mock.patch.object(guard, 'repository_configuration_inputs', side_effect=[self.inputs, {}]), \
                mock.patch.object(guard.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, b'', b'')), \
                self.assertRaisesRegex(ValueError, 'changed during its read-only query'):
            guard.verify_repository_configuration('/synthetic', self.environment)
        self.assertEqual(guard._REPOSITORY_CONFIGURATION, {})

    def test_exact_configuration_query_is_allowed_without_recursion(self):
        """The configuration query itself reads keys only and grants no status permission."""
        command = ['git', '-C', '/synthetic', *guard.REPOSITORY_CONFIGURATION_QUERY]
        with mock.patch.object(guard, 'verify_repository_configuration') as verify:
            guard.metadata_event_guard('subprocess.Popen', ('git', command, None, self.environment), {'/synthetic'})
        verify.assert_not_called()
        for extra in ('--edit', '--get-all', '--file=/another/file'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                guard.source_inspection_git([*command, extra], {'/synthetic'})

    def test_every_other_git_inspection_checks_configuration_first(self):
        """Existing byte, tree and dirty-source checks are preserved after the new refusal."""
        for arguments in (['show', 'a' * 40 + ':source.py'], ['status', '--porcelain', '--untracked-files=all']):
            command = ['git', '-C', '/synthetic', *arguments]
            with self.subTest(arguments=arguments), mock.patch.object(guard, 'verify_repository_configuration') as verify:
                guard.metadata_event_guard('subprocess.Popen', ('git', command, None, self.environment), {'/synthetic'})
            verify.assert_called_once_with('/synthetic', self.environment)

    def test_final_configuration_recheck_refuses_mutation(self):
        """A changed source configuration cannot produce a successful validation proof."""
        guard._REPOSITORY_CONFIGURATION['/synthetic'] = {'inputs': self.inputs}
        with mock.patch.object(guard, 'repository_configuration_inputs', return_value={}), \
                self.assertRaisesRegex(ValueError, 'pinned repository configuration changed'):
            guard.finish_repository_configurations()

    def test_returned_configuration_proof_is_explicit_and_nonempty(self):
        """A missing or changed trust/refusal statement cannot disappear from the returned child proof."""
        value = {'kind': 'same_user_pinned_repository_configuration_and_mutation_detection',
            'repositories': {'/synthetic/source': {'synthetic': True}}, 'filters_and_includes_refused': True,
            'scope': 'Pinned same-user source/configuration trust and mutation detection; no kernel-enforced file immutability.'}
        self.assertIs(guard.validate_repository_configuration_proof(value), value)
        for changed in (None, {}, {**value, 'repositories': {}}, {**value, 'filters_and_includes_refused': False},
                {**value, 'scope': 'kernel-enforced immutability'}, {**value, 'unknown': True}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                guard.validate_repository_configuration_proof(changed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
