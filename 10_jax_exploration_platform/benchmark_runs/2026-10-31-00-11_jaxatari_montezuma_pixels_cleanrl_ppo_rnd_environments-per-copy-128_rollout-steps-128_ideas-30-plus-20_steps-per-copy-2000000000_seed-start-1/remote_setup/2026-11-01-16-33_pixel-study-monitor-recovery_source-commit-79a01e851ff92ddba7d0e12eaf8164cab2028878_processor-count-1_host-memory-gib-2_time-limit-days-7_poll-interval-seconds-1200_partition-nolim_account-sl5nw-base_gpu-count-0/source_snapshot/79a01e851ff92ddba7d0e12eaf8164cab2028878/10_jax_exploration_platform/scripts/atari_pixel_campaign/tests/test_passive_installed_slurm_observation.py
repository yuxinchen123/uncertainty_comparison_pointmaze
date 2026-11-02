"""Check passive installed-file collection with local synthetic bytes only."""

import copy
import gzip
import hashlib
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import passive_installed_slurm_observation as passive


class SyntheticQuery:
    """Supply real bounded pipe bytes without creating any child process."""

    def __init__(self, stdout=b'', stderr=b'', hold_open=False, returncode=0):
        """Prepare small fixture pipes and optionally hold their writers for a timeout check."""
        self.writers = []
        self.returncode = None if hold_open else returncode
        self.killed = False
        for name, raw in [('stdout', stdout), ('stderr', stderr)]:
            reader, writer = os.pipe()
            require_small = len(raw) <= 1024
            if not require_small:
                raise AssertionError('synthetic query fixture must fit the local pipe buffer')
            if raw:
                os.write(writer, raw)
            setattr(self, name, os.fdopen(reader, 'rb', buffering=0))
            if hold_open:
                self.writers.append(writer)
            else:
                os.close(writer)

    def poll(self):
        """Return the synthetic child's known state without querying a real process."""
        return self.returncode

    def wait(self, timeout):
        """Finish only after the synthetic state is terminal."""
        if self.returncode is None:
            raise passive.subprocess.TimeoutExpired('synthetic query', timeout)
        return self.returncode

    def kill(self):
        """Close only synthetic writer descriptors and record the requested local-child stop."""
        self.killed = True
        self.returncode = -9
        for descriptor in self.writers:
            os.close(descriptor)
        self.writers.clear()


class PassiveInstalledSlurmTests(unittest.TestCase):
    """Exercise limits and claim boundaries without an installed executable or library call."""

    def setUp(self):
        """Create an ordinary temporary namespace for every file fixture."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def file(self, name, raw):
        """Write one synthetic ordinary file under the test's own temporary root."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    def test_path_spellings_and_namespaces_reject_traversal(self):
        """No relative path, hidden traversal or external namespace becomes readable."""
        for value in ('relative', '/tmp/../etc/passwd', '/tmp//file', '/tmp/file\n'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                passive.lexical_path(value)
        row, raw = passive.PassiveReader().read(Path('/dev/null'), (str(self.root),), 100)
        self.assertIsNone(raw)
        self.assertEqual(row['read_status'], 'unavailable')

    def test_reader_keeps_pinned_regular_bytes_and_owner(self):
        """Captured hashes and ownership belong to the descriptor actually read."""
        path = self.file('configuration.conf', b'TaskPlugin=task/cgroup\n')
        row, raw = passive.PassiveReader().read(path, (str(self.root),), 100, retain=True)
        self.assertEqual(raw, path.read_bytes())
        self.assertEqual(row['read_status'], 'read')
        self.assertEqual(row['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(row['inode'], path.stat().st_ino)
        self.assertEqual(row['owner_uid'], os.getuid())
        self.assertEqual(row['read_trace']['reader_before']['inode'], row['inode'])

    def test_parent_link_is_recorded_without_expanding_namespace(self):
        """An internal parent alias is explicit and an escaping alias remains unread."""
        target = self.file('installed/slurmd', b'synthetic binary')
        (self.root / 'alias').symlink_to('installed', target_is_directory=True)
        row, raw = passive.PassiveReader().read(self.root / 'alias/slurmd', (str(self.root),), 100)
        self.assertEqual(raw, b'synthetic binary')
        self.assertEqual(row['path'], str(target))
        self.assertEqual(row['symlink_chain'][0]['target'], 'installed')
        (self.root / 'outside').symlink_to('/dev', target_is_directory=True)
        row, raw = passive.PassiveReader().read(self.root / 'outside/null', (str(self.root),), 100)
        self.assertIsNone(raw)
        self.assertIn('leaves its allowed namespace', row['error'])

    def test_symlink_cycle_and_nonregular_inode_remain_unavailable(self):
        """A cycle or a directory is never opened as a metadata byte stream."""
        (self.root / 'cycle').symlink_to('cycle')
        for path in (self.root / 'cycle', self.root):
            row, raw = passive.PassiveReader().read(path, (str(self.root),), 100)
            self.assertIsNone(raw)
            self.assertEqual(row['read_status'], 'unavailable')

    def test_size_limit_keeps_prefix_out_of_parsed_content(self):
        """An over-limit file cannot supply accepted settings or an ELF record."""
        path = self.file('large.conf', b'a' * 11)
        reader = passive.PassiveReader()
        row, raw = reader.read(path, (str(self.root),), 10)
        self.assertIsNone(raw)
        self.assertEqual(row['read_status'], 'size_limit_exceeded')
        self.assertEqual(row['captured_prefix_byte_count'], 0)
        self.assertEqual(reader.bytes_read, 0)

    def test_changed_inode_is_retained_as_changed_without_parsing(self):
        """A changed file does not become a stable installed-image claim."""
        path = self.file('changing.conf', b'old')
        original = passive.ownership.pinned_configuration_bytes

        def changed(path, trace, limit):
            """Inject a recorded replacement after the real bounded fixture read."""
            raw = original(path, trace, limit)
            trace['path_replaced_during_read'] = True
            return raw

        with patch.object(passive.ownership, 'pinned_configuration_bytes', side_effect=changed):
            row, raw = passive.PassiveReader().read(path, (str(self.root),), 100)
        self.assertIsNone(raw)
        self.assertEqual(row['read_status'], 'changed_during_read')

    def test_changed_alias_is_not_silently_bound_to_old_target(self):
        """Changing a program alias during capture is retained without stable candidate bytes."""
        self.file('first', b'first')
        self.file('second', b'second')
        alias = self.root / 'program'
        alias.symlink_to('first')
        original = passive.ownership.pinned_configuration_bytes

        def replace(path, trace, limit):
            """Change only the original alias after its pinned target has been read."""
            raw = original(path, trace, limit)
            alias.unlink()
            alias.symlink_to('second')
            return raw

        with patch.object(passive.ownership, 'pinned_configuration_bytes', side_effect=replace):
            row, raw = passive.PassiveReader().read(alias, (str(self.root),), 100)
        self.assertIsNone(raw)
        self.assertEqual(row['read_status'], 'changed_during_read')
        self.assertFalse(row['symlink_chain_after'][0]['unchanged'])

    def test_settings_filter_preserves_duplicates_without_secrets(self):
        """Only named safe keys are retained, in original order and case."""
        raw = b'AuthInfo=secret\nTaskPlugin=task/cgroup\ntaskplugin=task/affinity\nNodeName=jaguar03 Gres=gpu:8\n'
        rows = passive.setting_rows(raw, passive.FILE_SETTINGS)
        self.assertEqual([row['name'] for row in rows], ['TaskPlugin', 'taskplugin', 'NodeName', 'Gres'])
        self.assertNotIn('secret', repr(rows))
        self.assertIsNone(passive.unique_setting(rows, 'TaskPlugin'))

    def test_quoted_unallowlisted_values_cannot_expose_inner_field_names(self):
        """A quoted AuthInfo value stays skipped as one value, including allowlisted-looking text."""
        raw = (b'AuthInfo="private_test_token_719 File=key_data PluginDir=/private GresTypes=wrong" '
            b'GresTypes=gpu\nTaskPlugin="task/cgroup"\n')
        rows = passive.setting_rows(raw, passive.FILE_SETTINGS)
        self.assertEqual(rows, [{'line': 1, 'name': 'GresTypes', 'value': 'gpu'},
            {'line': 2, 'name': 'TaskPlugin', 'value': 'task/cgroup'}])
        for forbidden in ('private_test_token_719', 'key_data', '/private', 'wrong'):
            self.assertNotIn(forbidden, repr(rows))

    def test_unreviewed_escaping_cannot_supply_effective_settings(self):
        """Backslash syntax stays unresolved instead of assuming shell and Slurm parsing are identical."""
        raw = b'GresTypes=gp\\u\n'
        with self.assertRaisesRegex(ValueError, 'unreviewed configuration escape syntax'):
            passive.setting_rows(raw, passive.FILE_SETTINGS)
        main = self.file('slurm.conf', raw)
        record = passive.configuration_tree(passive.PassiveReader(), main, (str(self.root),))
        self.assertFalse(record['complete'])
        self.assertEqual(record['files'][0]['parse_status'], 'unsupported_text_or_quote_syntax')

    def test_controller_command_scope_and_filtered_output(self):
        """Only two reviewed controller reads can execute, and full output is not retained."""
        child = SyntheticQuery(stdout=b'GresTypes = gpu\nAuthInfo = secret\n')
        with patch.object(passive.subprocess, 'Popen', return_value=child) as run:
            result = passive.controller_query(['scontrol', 'show', 'config'], passive.SETTINGS)
            self.assertEqual(result['fields'], [{'line': 1, 'name': 'GresTypes', 'value': 'gpu'}])
            self.assertNotIn('secret', repr(result))
            self.assertEqual(run.call_count, 1)
            for command in (['slurmd', '-G'], ['slurmd', '-V'], ['scontrol', 'reconfigure'], ['ldd', '/bin/true']):
                with self.assertRaises(ValueError):
                    passive.controller_query(command, passive.SETTINGS)
            self.assertEqual(run.call_count, 1)

    def test_controller_stdout_is_bounded_before_full_buffering(self):
        """The first byte beyond the response limit stops only the mocked query child."""
        child = SyntheticQuery(stdout=b'GresTypes=gpu\nprivate_test_token_719', hold_open=True)
        with patch.object(passive.subprocess, 'Popen', return_value=child) as run, \
                patch.dict(passive.LIMITS, {'query_bytes': 16}):
            record = passive.controller_query(['scontrol', 'show', 'config'], passive.SETTINGS)
        self.assertEqual(record['capture_status'], 'response_limit_exceeded')
        self.assertEqual(record['stdout_captured_byte_count'], 17)
        self.assertEqual(record['exceeded_stream'], 'stdout')
        self.assertEqual(record['fields'], [])
        self.assertNotIn('private_test_token_719', repr(record))
        self.assertTrue(child.killed)
        self.assertEqual(run.call_count, 1)

    def test_controller_stderr_has_its_own_capture_limit(self):
        """Untrusted error output is bounded and cannot supply configuration fields."""
        child = SyntheticQuery(stdout=b'GresTypes=gpu\n', stderr=b'private_test_token_719', hold_open=True)
        with patch.object(passive.subprocess, 'Popen', return_value=child), \
                patch.dict(passive.LIMITS, {'query_bytes': 16}):
            record = passive.controller_query(['scontrol', 'show', 'config'], passive.SETTINGS)
        self.assertEqual(record['capture_status'], 'response_limit_exceeded')
        self.assertEqual(record['exceeded_stream'], 'stderr')
        self.assertEqual(record['stderr_captured_byte_count'], 17)
        self.assertEqual(record['fields'], [])
        self.assertTrue(child.killed)

    def test_controller_timeout_retains_failure_metadata_without_retry(self):
        """A silent query child times out once and its raw buffers never become settings."""
        child = SyntheticQuery(hold_open=True)
        with patch.object(passive.subprocess, 'Popen', return_value=child) as run, \
                patch.dict(passive.LIMITS, {'query_timeout_seconds': 0.005}):
            record = passive.controller_query(['scontrol', 'show', 'config'], passive.SETTINGS)
        self.assertEqual(record['capture_status'], 'timeout')
        self.assertEqual(record['returncode'], -9)
        self.assertEqual(record['fields'], [])
        self.assertTrue(child.killed)
        self.assertEqual(run.call_count, 1)

    def test_controller_spawn_failure_and_nonzero_exit_are_explicit(self):
        """Unavailable or failed local query commands are retained without inferred controller values."""
        with patch.object(passive.subprocess, 'Popen', side_effect=FileNotFoundError('synthetic command unavailable')) as run:
            record = passive.controller_query(['scontrol', 'show', 'config'], passive.SETTINGS)
        self.assertEqual(record['capture_status'], 'failed')
        self.assertEqual(record['error_type'], 'FileNotFoundError')
        self.assertEqual(record['fields'], [])
        self.assertEqual(run.call_count, 1)
        child = SyntheticQuery(stdout=b'GresTypes=gpu\n', returncode=1)
        with patch.object(passive.subprocess, 'Popen', return_value=child):
            record = passive.controller_query(['scontrol', 'show', 'config'], passive.SETTINGS)
        self.assertEqual(record['returncode'], 1)
        self.assertEqual(record['filter_status'], 'query_incomplete_or_failed')
        self.assertEqual(record['fields'], [])

    def test_controller_quotes_and_node_hash_text_are_handled_without_disclosure(self):
        """Bad quotes stay unknown while a node kernel hash marker does not erase later fields."""
        child = SyntheticQuery(stdout=b'AuthInfo="private_test_token_719\n')
        with patch.object(passive.subprocess, 'Popen', return_value=child):
            record = passive.controller_query(['scontrol', 'show', 'config'], passive.SETTINGS)
        self.assertEqual(record['fields'], [])
        self.assertEqual(record['filter_status'], 'unsupported_text_or_quote_syntax')
        self.assertNotIn('private_test_token_719', repr(record))
        child = SyntheticQuery(stdout=b'NodeName=jaguar03 Version=23.11.6 OS=Linux 5.15 #161-Ubuntu SMP SlurmdStartTime=2026-10-26T14:37:45\n')
        with patch.object(passive.subprocess, 'Popen', return_value=child):
            record = passive.controller_query(['scontrol', 'show', 'node', 'jaguar03', '-o'],
                ('NodeName', 'Version', 'SlurmdStartTime'))
        self.assertEqual(passive.unique_setting(record['fields'], 'SlurmdStartTime'), '2026-10-26T14:37:45')

    def test_literal_include_order_and_cycle_are_recorded(self):
        """Literal child order is kept while cycles refuse any complete-tree claim."""
        main = self.file('slurm.conf', b'GresTypes=gpu\nInclude ' + str(self.root / 'child.conf').encode() + b'\n')
        self.file('child.conf', b'SlurmUser=slurm\nInclude ' + str(main).encode() + b'\n')
        tree = passive.configuration_tree(passive.PassiveReader(), main, (str(self.root),))
        self.assertFalse(tree['complete'])
        self.assertEqual([row['token'] for row in tree['include_order']], [str(self.root / 'child.conf'), str(main)])
        self.assertIn('include_cycle', [row['reason'] for row in tree['unresolved']])
        self.assertFalse(tree['qualification_granted'])

    def test_glob_and_outside_include_never_open(self):
        """Unsupported wildcard or outside paths stay unresolved without a broad file scan."""
        main = self.file('slurm.conf', b'Include /etc/*.conf\nInclude /etc/passwd\n')
        tree = passive.configuration_tree(passive.PassiveReader(), main, (str(self.root),))
        self.assertEqual(len(tree['files']), 1)
        self.assertEqual(len(tree['unresolved']), 2)

    def test_key_material_include_is_refused_before_reader_call(self):
        """A literal private-key child is recorded but never opened as configuration."""
        key = self.file('jwt.key', b'private_test_token_719')
        main = self.file('slurm.conf', b'Include ' + str(key).encode() + b'\n')
        reader = passive.PassiveReader()
        with patch.object(reader, 'read', wraps=reader.read) as read:
            tree = passive.configuration_tree(reader, main, (str(self.root),))
        self.assertEqual(read.call_count, 1)
        self.assertFalse(tree['complete'])
        self.assertEqual(tree['include_order'][0]['status'], 'prohibited_configuration_filename')
        self.assertNotIn('private_test_token_719', repr(tree))

    def test_allowed_include_alias_cannot_read_forbidden_target(self):
        """A nodes.conf alias to jwt.key is rejected after metadata resolution and before byte reading."""
        key = self.file('jwt.key', b'private_test_token_719')
        alias = self.root / 'nodes.conf'
        alias.symlink_to(key.name)
        reader = passive.PassiveReader()
        with patch.object(passive.ownership, 'pinned_configuration_bytes') as pin:
            row, raw = reader.read(alias, (str(self.root),), 100, configuration=True)
        pin.assert_not_called()
        self.assertIsNone(raw)
        self.assertEqual(row['symlink_chain'][0]['target'], 'jwt.key')
        self.assertIn('prohibited resolved configuration filename', row['error'])

    def test_credential_named_conf_and_key_directory_are_refused(self):
        """A .conf suffix does not admit explicitly credential-named files or directories."""
        reader = passive.PassiveReader()
        with patch.object(passive.ownership, 'pinned_configuration_bytes') as pin:
            for path in (self.root / 'private.conf', self.root / 'keys/nodes.conf', self.root / 'credentials.conf'):
                row, raw = reader.read(path, (str(self.root),), 100, configuration=True)
                self.assertIsNone(raw)
                self.assertIn('prohibited configuration filename', row['error'])
        pin.assert_not_called()

    def test_gres_and_cgroup_include_semantics_remain_explicitly_incomplete(self):
        """Unsupported Include directives are preserved and none of their children are read."""
        for name in ('gres.conf', 'cgroup.conf'):
            path = self.file(name, b'Include ' + str(self.root / 'unread_child.conf').encode() + b'\nConstrainDevices=yes\n')
            reader = passive.PassiveReader()
            with patch.object(reader, 'read', wraps=reader.read) as read:
                record = passive.configuration_excerpt(reader, path, (str(self.root),))
            self.assertEqual(read.call_count, 1)
            self.assertFalse(record['configuration_complete'])
            self.assertFalse(record['include_expansion_supported'])
            self.assertEqual(record['include_status'], 'non_slurm_conf_Include_semantics_not_reviewed')
            self.assertEqual(record['include_directives'][0]['token'], str(self.root / 'unread_child.conf'))

    def test_configuration_limit_preserves_explicit_incomplete_state(self):
        """A limited include tree cannot silently appear complete."""
        main = self.file('slurm.conf', b'Include ' + str(self.root / 'child.conf').encode() + b'\n')
        self.file('child.conf', b'GresTypes=gpu\n')
        with patch.dict(passive.LIMITS, {'include_files': 1}):
            tree = passive.configuration_tree(passive.PassiveReader(), main, (str(self.root),))
        self.assertEqual(len(tree['files']), 1)
        self.assertEqual(tree['unresolved'][0]['reason'], 'configuration_tree_limit')

    def test_configuration_count_and_bytes_are_shared_across_candidate_trees(self):
        """Observing both global and cached trees cannot multiply the declared configuration limits."""
        first = self.file('global/slurm.conf', b'GresTypes=gpu\n')
        second = self.file('cache/slurm.conf', b'GresTypes=gpu\n')
        reader = passive.PassiveReader()
        with patch.dict(passive.LIMITS, {'include_files': 1}):
            initial = passive.configuration_tree(reader, first, (str(self.root),))
            later = passive.configuration_tree(reader, second, (str(self.root),))
        self.assertTrue(initial['complete'])
        self.assertFalse(later['complete'])
        self.assertIn('total configuration-file count limit', later['files'][0]['error'])
        self.assertEqual(reader.configuration_read_attempts, 1)
        reader = passive.PassiveReader()
        with patch.dict(passive.LIMITS, {'configuration_total_bytes': len(first.read_bytes())}):
            initial = passive.configuration_tree(reader, first, (str(self.root),))
            later = passive.configuration_tree(reader, second, (str(self.root),))
        self.assertTrue(initial['complete'])
        self.assertFalse(later['complete'])
        self.assertIn('total configuration byte limit', later['files'][0]['error'])
        self.assertEqual(reader.configuration_bytes_read, len(first.read_bytes()))

    def test_unparseable_configuration_retains_hash_and_partial_record(self):
        """Invalid text encoding keeps its original file metadata and an unresolved parser result."""
        main = self.file('slurm.conf', b'GresTypes=gpu\n\xff\n')
        tree = passive.configuration_tree(passive.PassiveReader(), main, (str(self.root),))
        self.assertFalse(tree['complete'])
        self.assertEqual(tree['files'][0]['sha256'], hashlib.sha256(main.read_bytes()).hexdigest())
        self.assertEqual(tree['files'][0]['parse_status'], 'unsupported_text_or_quote_syntax')

    def test_service_metadata_redacts_environment_and_unknown_arguments(self):
        """No environment-file bytes or unreviewed execution values enter saved metadata."""
        raw = (b'[Service]\nUser=slurm\nEnvironment=TOKEN=private_test_token_719\nEnvironmentFile=/etc/private\n'
            b'ExecStart=/usr/sbin/slurmd -D -f /etc/slurm/slurm.conf\n'
            b'ExecStart=/usr/sbin/slurmd --token=private_test_token_719\n')
        rows = passive.service_tokens(raw)
        self.assertNotIn('private_test_token_719', repr(rows))
        self.assertEqual(rows[2]['safe_tokens'], ['/usr/sbin/slurmd', '-D', '-f', '/etc/slurm/slurm.conf'])
        self.assertEqual(rows[1]['status'], 'not_read_environment_may_contain_secrets')
        self.assertEqual(rows[3]['status'], 'unresolved_execution_tokens')
        self.assertNotIn('safe_tokens', rows[3])

    def test_service_substitutions_cannot_select_a_program(self):
        """Systemd substitutions and shell continuations remain unresolved."""
        for value in ('$SLURMD', '/usr/sbin/slurmd %n', '/usr/sbin/slurmd \\'):
            rows = passive.service_tokens(('[Service]\nExecStart=' + value + '\n').encode())
            self.assertEqual(rows[0]['status'], 'unresolved_execution_tokens')

    def test_directory_listing_is_bounded_and_does_not_recurse(self):
        """Only matching direct entries are returned and excess entries invalidate the listing."""
        self.file('dropins/one.conf', b'')
        self.file('dropins/two.conf', b'')
        self.file('dropins/subdirectory/hidden.conf', b'')
        row = passive.directory_entries(self.root / 'dropins', (str(self.root),), r'.*\.conf', 2)
        self.assertEqual(row['entries'], ['one.conf', 'two.conf'])
        row = passive.directory_entries(self.root / 'dropins', (str(self.root),), r'.*\.conf', 1)
        self.assertEqual(row['entries'], [])
        self.assertEqual(row['status'], 'unavailable')

    def test_late_directory_symlink_replacement_is_not_followed(self):
        """Descriptor-relative non-following traversal rejects a changed parent before listing."""
        directory = self.root / 'parent/dropins'
        directory.mkdir(parents=True)
        outside = self.root / 'outside'
        outside.mkdir()
        original = passive.resolve_links

        def replace(path, roots):
            """Replace the observed parent with an alias just before descriptor traversal."""
            result = original(path, roots)
            (self.root / 'parent').rename(self.root / 'saved_parent')
            (self.root / 'parent').symlink_to(outside, target_is_directory=True)
            return result

        with patch.object(passive, 'resolve_links', side_effect=replace), patch.object(passive.os, 'scandir') as scan:
            row = passive.directory_entries(directory, (str(self.root),), r'.*', 2)
        self.assertEqual(row['status'], 'unavailable')
        scan.assert_not_called()

    def test_original_directory_alias_is_rechecked_after_listing(self):
        """Retargeting the original alias invalidates a listing of its old pinned target."""
        self.file('first/old.conf', b'')
        self.file('second/new.conf', b'')
        alias = self.root / 'current'
        alias.symlink_to('first', target_is_directory=True)
        original = passive.resolve_links

        def replace(path, roots):
            """Move the original alias after resolution while preserving the resolved directory inode."""
            result = original(path, roots)
            alias.unlink()
            alias.symlink_to('second', target_is_directory=True)
            return result

        with patch.object(passive, 'resolve_links', side_effect=replace):
            record = passive.directory_entries(alias, (str(self.root),), r'.*\.conf', 2)
        self.assertEqual(record['status'], 'unavailable')
        self.assertEqual(record['entries'], [])
        self.assertFalse(record['symlink_chain_after'][0]['unchanged'])

    def elf_fixture(self):
        """Construct a tiny synthetic ELF64 note image with no executable behavior."""
        raw = bytearray(256)
        ident = b'\x7fELF\x02\x01\x01' + b'\0' * 9
        struct.pack_into('<16sHHIQQQIHHHHHH', raw, 0, ident, 2, 62, 1, 0, 64, 0, 0, 64, 56, 1, 0, 0, 0)
        note = struct.pack('<III', 4, 4, 3) + b'GNU\0' + b'\x01\x02\x03\x04'
        struct.pack_into('<IIQQQQQQ', raw, 64, 4, 0, 128, 0, 0, len(note), len(note), 4)
        raw[128:128 + len(note)] = note
        return bytes(raw)

    def test_elf_build_identity_is_parsed_without_execution(self):
        """The passive parser reads the expected build note and declares no loading."""
        result = passive.elf_metadata(self.elf_fixture())
        self.assertEqual(result['build_ids'], ['01020304'])
        self.assertEqual(result['machine'], 62)
        self.assertFalse(result['image_executed'])
        self.assertFalse(result['dependencies_loaded'])

    def test_elf_segments_and_class_cannot_leave_captured_bytes(self):
        """Malformed ELF metadata is unsupported rather than delegated to an external utility."""
        for raw in (b'not elf', self.elf_fixture()[:140], b'\x7fELF\x01\x01' + b'\0' * 100):
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                passive.elf_metadata(raw)

    def test_elf_dependency_and_search_paths_are_only_passively_parsed(self):
        """Dynamic string metadata is resolved inside captured load bytes without ldd or dlopen."""
        raw = bytearray(1024)
        ident = b'\x7fELF\x02\x01\x01' + b'\0' * 9
        struct.pack_into('<16sHHIQQQIHHHHHH', raw, 0, ident, 2, 62, 1, 0, 64, 0, 0, 64, 56, 2, 0, 0, 0)
        struct.pack_into('<IIQQQQQQ', raw, 64, 1, 4, 0, 0x400000, 0, 1024, 1024, 4096)
        entries = [(5, 0x400000 + 512), (10, 28), (1, 1), (15, 11), (29, 19), (0, 0)]
        struct.pack_into('<IIQQQQQQ', raw, 120, 2, 4, 256, 0x400100, 0, len(entries) * 16, len(entries) * 16, 8)
        raw[512:540] = b'\0libc.so.6\0$ORIGIN\0/usr/lib\0'
        for index, (tag, value) in enumerate(entries):
            struct.pack_into('<qQ', raw, 256 + index * 16, tag, value)
        result = passive.elf_metadata(bytes(raw))
        self.assertEqual(result['needed'], ['libc.so.6'])
        self.assertEqual(result['rpath'], ['$ORIGIN'])
        self.assertEqual(result['runpath'], ['/usr/lib'])
        struct.pack_into('<qQ', raw, 256 + 2 * 16, 1, 99999)
        with self.assertRaisesRegex(ValueError, 'string offset'):
            passive.elf_metadata(bytes(raw))

    def test_program_discovery_has_no_relative_or_unbounded_path_scan(self):
        """Only exact safe program names are derived from bounded installed PATH entries."""
        services = {'units_and_dropins': [{'service_fields': [{'program': '/usr/sbin/slurmd'}]}]}
        candidates, unresolved = passive.program_candidates(services, '/usr/bin:.:/home/someone/bin')
        self.assertEqual(candidates, ['/usr/sbin/slurmd', '/usr/bin/slurmd', '/usr/bin/slurmstepd', '/usr/bin/scontrol'])
        self.assertEqual(len(unresolved), 2)

    def test_plugin_dir_ambiguity_never_selects_a_plugin(self):
        """Conflicting controller PluginDir values cannot be collapsed into a chosen image."""
        rows = [{'name': 'PluginDir', 'value': '/usr/lib/slurm'}, {'name': 'plugindir', 'value': '/usr/local/lib/slurm'}]
        reader = SimpleNamespace(read=lambda *args: self.fail('ambiguous PluginDir was read'))
        result = passive.plugin_observations(reader, rows)
        self.assertEqual(result['files'], [])
        self.assertEqual(result['unresolved'], ['missing_or_ambiguous_PluginDir'])

    def test_account_context_omits_password_gecos_and_unrelated_users(self):
        """UID64030 and observed Slurm identities retain only selected public account fields."""
        raw = b'other:x:1000:1000:private:/home/other:/bin/bash\nslurm:secret:64030:64030:private:/var/lib/slurm:/usr/sbin/nologin\nroot:x:0:0:root:/root:/bin/bash\n'
        reader = SimpleNamespace(read=lambda *args: ({'sha256': hashlib.sha256(raw).hexdigest()}, raw))
        result = passive.service_accounts(reader, [{'name': 'SlurmUser', 'value': 'slurm(64030)'}])
        self.assertEqual([row['name'] for row in result['selected_accounts']], ['slurm'])
        self.assertNotIn('secret', repr(result))
        self.assertNotIn('private', repr(result))
        self.assertNotIn('/home/other', repr(result))
        self.assertTrue(result['account_metadata_does_not_establish_config_trust'])

    def test_manual_decompression_cannot_expand_without_limit(self):
        """A compressed ordinary manual retains its image hash but excessive expansion is unsupported."""
        raw = gzip.compress(b'a' * 101)
        reader = SimpleNamespace(read=lambda *args, **kwargs: ({'sha256': hashlib.sha256(raw).hexdigest()}, raw))
        with patch.dict(passive.LIMITS, {'manual_bytes': 100}):
            result = passive.installed_manuals(reader)
        self.assertTrue(all(row.get('decompression_status') == 'unsupported' for row in result[1::2]))

    def test_package_metadata_keeps_only_selected_labels_and_matching_rosters(self):
        """Selected package labels and file membership do not become a binary equivalence claim."""
        status = (b'Package: unrelated\nVersion: 9\n\nPackage: slurm-wlm\nVersion: 23.11.6-1\n'
            b'Architecture: amd64\nStatus: install ok installed\nSource: slurm-wlm\nDescription: hidden\n')

        def read(path, roots, limit):
            """Provide only synthetic package status and exact matching file-list records."""
            raw = status if path.name == 'status' else b'/usr/sbin/slurmd\n/other/file\n'
            return {'sha256': hashlib.sha256(raw).hexdigest()}, raw

        result = passive.package_metadata(SimpleNamespace(read=read), [{'path': '/usr/sbin/slurmd'}])
        self.assertEqual([row['Package'] for row in result['packages']], ['slurm-wlm'])
        self.assertNotIn('hidden', repr(result))
        self.assertFalse(result['binary_ownership_verified'])
        self.assertTrue(result['package_label_is_not_upstream_binary_attestation'])
        self.assertEqual(result['file_rosters'][0]['matching_observed_program_paths'], ['/usr/sbin/slurmd'])

    def test_claims_are_all_zero_and_no_runtime_entry_is_exposed(self):
        """This source draft cannot qualify a profile or expose a Slurm submission command."""
        self.assertFalse(passive.ZERO_CREDIT['physical_gres_mapping_verified'])
        self.assertFalse(passive.ZERO_CREDIT['kernel_device_enforcement_verified'])
        self.assertFalse(passive.ZERO_CREDIT['configured_isolation_qualified'])
        self.assertFalse(passive.ZERO_CREDIT['admission_granted'])
        self.assertTrue(all(value == 0 for name, value in passive.ZERO_CREDIT.items() if name.endswith('_credit')))
        self.assertFalse(hasattr(passive, 'main'))

    def test_foreign_current_process_refuses_before_controller_query(self):
        """A saved foreign PID cannot authorize metadata collection in the current process."""
        process = {'hostname': 'jaguar03', 'pid': 1, 'uid': 2618919, 'effective_uid': 2618919,
            'uid_fields': [2618919] * 4, 'parent_pid': 3, 'process_start_time': '1000',
            'cgroup': 'synthetic cgroup', 'cpu_affinity': [0], 'raw_proc_stat': 'synthetic stat'}
        row = {'process': process, 'allocation_binding_passed': True}
        current = copy.deepcopy(process)
        current['pid'] = 2
        with patch.object(passive.ownership, 'validate_allocation'), \
                patch.object(passive.ownership, 'own_process', return_value=current), \
                patch.object(passive, 'controller_query') as query, self.assertRaisesRegex(ValueError, 'actual observed process'):
            passive.collect_owned_metadata(row)
        query.assert_not_called()

    def test_proc_runtime_changes_do_not_invent_a_new_birth_identity(self):
        """Changing process statistics do not prevent a metadata-only same-process continuation."""
        process = {'hostname': 'jaguar03', 'pid': 1, 'uid': 2618919, 'effective_uid': 2618919,
            'uid_fields': [2618919] * 4, 'parent_pid': 3, 'process_start_time': '1000',
            'cgroup': 'synthetic cgroup', 'cpu_affinity': [0], 'raw_proc_stat': 'earlier CPU-time fields'}
        row = {'process': process, 'allocation_binding_passed': True, 'job_id': '123', 'step_id': '0'}
        current = copy.deepcopy(process)
        current['raw_proc_stat'] = 'later CPU-time fields with the same checked PID/birth'
        with patch.object(passive.ownership, 'validate_allocation'), \
                patch.object(passive.ownership, 'own_process', return_value=current), \
                patch.object(passive, 'controller_query', return_value={'fields': [], 'returncode': 0}), \
                patch.object(passive.PassiveReader, 'read', return_value=({'read_status': 'unavailable'}, None)), \
                patch.object(passive, 'service_observations', return_value={'units_and_dropins': []}), \
                patch.object(passive, 'installed_manuals', return_value=[]), \
                patch.dict(passive.os.environ, {'PATH': ''}, clear=True):
            result = passive.collect_owned_metadata(row)
        self.assertTrue(result['collection_completed'])
        self.assertEqual(result['process']['raw_proc_stat'], current['raw_proc_stat'])
        self.assertFalse(result['physical_gres_mapping_verified'])
        self.assertFalse(result['admission_granted'])

    def test_birth_change_refuses_even_when_pid_is_unchanged(self):
        """A reused PID cannot pass the same-process continuation check."""
        process = {'hostname': 'jaguar03', 'pid': 1, 'uid': 2618919, 'effective_uid': 2618919,
            'uid_fields': [2618919] * 4, 'parent_pid': 3, 'process_start_time': '1000',
            'cgroup': 'synthetic cgroup', 'cpu_affinity': [0]}
        current = {**process, 'process_start_time': '1001'}
        with patch.object(passive.ownership, 'validate_allocation'), \
                patch.object(passive.ownership, 'own_process', return_value=current), \
                patch.object(passive, 'controller_query') as query, self.assertRaises(ValueError):
            passive.collect_owned_metadata({'process': process})
        query.assert_not_called()


if __name__ == '__main__':
    unittest.main()
