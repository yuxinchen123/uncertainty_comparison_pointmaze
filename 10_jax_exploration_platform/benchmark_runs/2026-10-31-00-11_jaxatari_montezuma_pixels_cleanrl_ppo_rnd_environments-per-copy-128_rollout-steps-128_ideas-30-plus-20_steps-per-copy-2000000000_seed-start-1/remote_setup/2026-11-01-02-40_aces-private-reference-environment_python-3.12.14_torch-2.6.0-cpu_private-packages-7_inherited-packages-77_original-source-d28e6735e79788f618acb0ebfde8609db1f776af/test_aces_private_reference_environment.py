"""Test complete metadata dependencies, private-prefix guards and offline setup without pip or SSH."""

import copy
import hashlib
import importlib.util
from importlib import metadata as package_metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parent

def load(name):
    """Load one local metadata-only reviewed module for synthetic fixtures."""
    spec = importlib.util.spec_from_file_location('tested_' + name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load('setup_aces_private_reference_environment')
meta = load('aces_private_environment_metadata')
PLAN = ROOT / 'aces_private_reference_environment_setup_plan.json'


class MetadataTests(unittest.TestCase):
    """Keep all actual wheels read-only and every installation/network action mocked or absent."""

    def setUp(self):
        """Create private temporary metadata fixtures and read the frozen setup plan."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.plan = json.loads(PLAN.read_bytes())
        self.environment = {'python_version': '3.12', 'python_full_version': '3.12.14', 'sys_platform': 'linux',
            'os_name': 'posix', 'platform_machine': 'x86_64', 'implementation_name': 'cpython',
            'implementation_version': '3.12.14', 'platform_python_implementation': 'CPython', 'platform_system': 'Linux'}

    def test_actual_seven_wheels_and_complete_requested_extras_pass_without_network(self):
        """The actual84-provider plan satisfies all128 requirements including etils epath/epy."""
        with mock.patch.object(setup.subprocess, 'Popen', side_effect=AssertionError('network forbidden')):
            payload, plan, base, assets, metadata, original = setup.local_payload(PLAN)
        baseline = json.loads(payload['full_base_inventory'])
        wheels = [meta.wheel_metadata(row) for row in plan['wheels']]
        proof = meta.proposed_closure(baseline['distributions'], wheels, self.environment)
        self.assertEqual(proof['distribution_count'], 84)
        self.assertEqual(proof['active_dependency_count'], 128)
        self.assertEqual(proof['requested_extras'], {'etils': ['epath', 'epy']})
        self.assertNotIn('numpy', sys.modules)
        self.assertNotIn('torch', sys.modules)

    def test_extra_dependencies_propagate_until_fixed_point(self):
        """A downstream extra activates another extra and fails when its actual required provider is absent."""
        rows = [{'name': 'caller', 'version': '1', 'requires_dist': ['provider[epath]>=1']},
            {'name': 'provider', 'version': '1', 'requires_dist': ['provider[epy]; extra == "epath"', 'leaf>=2; extra == "epy"']},
            {'name': 'leaf', 'version': '2', 'requires_dist': []}]
        proof = meta.dependency_check(rows, self.environment)
        self.assertEqual(proof['active_dependency_count'], 3)
        self.assertEqual(proof['requested_extras'], {'provider': ['epath', 'epy']})
        with self.assertRaisesRegex(ValueError, 'closure is invalid'):
            meta.dependency_check(rows[:-1], self.environment)

    def test_duplicate_canonical_base_names_and_active_url_requirements_reject(self):
        """No dict conversion can erase a duplicate provider and URL sources need separate binding."""
        duplicate = [{'name': 'a_b', 'version': '1', 'requires_dist': []}, {'name': 'A..B', 'version': '1', 'requires_dist': []}]
        with self.assertRaisesRegex(ValueError, 'unique'):
            meta.proposed_closure(duplicate, [], self.environment)
        with self.assertRaisesRegex(ValueError, 'URL dependency'):
            meta.dependency_check([{'name': 'caller', 'version': '1', 'requires_dist': ['leaf @ https://example.invalid/package.whl']}], self.environment)

    def synthetic_wheel(self, tag):
        """Write a small explicitly synthetic CPU reference wheel with one declared ABI tag."""
        path = self.root / 'torch-2.6.0+cpu-cp312-cp312-linux_x86_64.whl'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('torch-2.6.0+cpu.dist-info/METADATA', 'Name: torch\nVersion: 2.6.0+cpu\nRequires-Python: >=3.9\n')
            archive.writestr('torch-2.6.0+cpu.dist-info/WHEEL', 'Wheel-Version: 1.0\nTag: ' + tag + '\n')
        return {'path': str(path), 'project': 'torch', 'version': '2.6.0+cpu', 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_wrong_wheel_abi_tag_and_startup_hook_reject(self):
        """Only exact CP312/Linux or prescribed pure-Python tags and no startup code are accepted."""
        row = self.synthetic_wheel('cp313-cp313-linux_x86_64')
        with self.assertRaisesRegex(ValueError, 'tag set'):
            meta.wheel_metadata(row)
        row = self.synthetic_wheel('cp312-cp312-linux_x86_64')
        with zipfile.ZipFile(row['path'], 'a') as archive:
            archive.writestr('unsafe.pth', 'import torch\n')
        row.update(bytes=Path(row['path']).stat().st_size, sha256=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, 'startup hook'):
            meta.wheel_metadata(row)

    def test_private_prefix_and_shared_provider_checks_block_wrong_install_destinations(self):
        """Newprefix/newpurelib and unchanged base provider paths are required before installing."""
        before = {'effective': {'jax': {'name': 'jax', 'version': '0.11', 'metadata_path': '/base/jax.dist-info'}}}
        private = {'prefix': self.plan['private_prefix'], 'base_prefix': self.plan['base_prefix'],
            'purelib': self.plan['private_prefix'] + '/lib/python3.12/site-packages', 'python': self.plan['private_python'],
            'version': '3.12.14', 'numerical_imports': [], 'dont_write_bytecode': True,
            'distributions': list(before['effective'].values()), 'effective': copy.deepcopy(before['effective'])}
        setup.check_prefix(self.plan, before, private)
        for key, value in (('prefix', self.plan['base_prefix']), ('purelib', '/base/site-packages'), ('dont_write_bytecode', False)):
            changed = {**private, key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'prefix/inheritance'):
                setup.check_prefix(self.plan, before, changed)
        changed = copy.deepcopy(private)
        changed['effective']['jax']['version'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'shared base package'):
            setup.check_prefix(self.plan, before, changed)

    def test_exact_seven_private_providers_and_versions_are_required_after_install(self):
        """An extra private pip or a changed Torch version cannot reach native preparation."""
        rows = {name: {'name': name, 'version': version, 'metadata_path': self.plan['private_prefix'] + '/lib/python3.12/site-packages/' + name + '.dist-info'}
            for name, version in setup.PRIVATE_VERSIONS.items()}
        private = {'prefix': self.plan['private_prefix'], 'base_prefix': self.plan['base_prefix'],
            'purelib': self.plan['private_prefix'] + '/lib/python3.12/site-packages', 'python': self.plan['private_python'],
            'version': '3.12.14', 'numerical_imports': [], 'dont_write_bytecode': True, 'effective': rows, 'distributions': list(rows.values())}
        setup.check_prefix(self.plan, {'effective': {}}, private, installed=True)
        changed = copy.deepcopy(private)
        changed['distributions'].append({'name': 'pip', 'metadata_path': self.plan['private_prefix'] + '/pip.dist-info'})
        with self.assertRaisesRegex(ValueError, 'seven allowed'):
            setup.check_prefix(self.plan, {'effective': {}}, changed, installed=True)
        changed = copy.deepcopy(private)
        changed['effective']['torch']['version'] = '2.6.0+cu124'
        with self.assertRaisesRegex(ValueError, 'another version'):
            setup.check_prefix(self.plan, {'effective': {}}, changed, installed=True)

    def test_offline_pip_arguments_and_environment_cannot_target_the_base(self):
        """Only the newprefix and seven hash-locked binary wheels are eligible for the future root-run install."""
        with mock.patch.dict(os.environ, {'HOME': '/unchanged', 'PIP_TARGET': '/old', 'PIP_PREFIX': '/old', 'PIP_USER': '1', 'PIP_INDEX_URL': 'https://example.invalid'}, clear=True):
            environment = setup.installation_environment()
        self.assertEqual(environment['HOME'], '/unchanged')
        self.assertEqual(environment['PIP_CONFIG_FILE'], os.devnull)
        self.assertNotIn('PIP_TARGET', environment)
        self.assertNotIn('PIP_PREFIX', environment)
        self.assertNotIn('PIP_USER', environment)
        command = setup.pip_command(self.plan, '/new/wheels', '/new/requirements.lock')
        for option in ('--no-index', '--no-deps', '--require-hashes', '--ignore-installed', '--no-compile', '--only-binary=:all:', '--prefix=' + self.plan['private_prefix']):
            self.assertIn(option, command)
        self.assertEqual(command[0], self.plan['private_python'])
        self.assertIn('-B', command)

    def test_arbitrary_old_python_program_and_submission_are_blocked(self):
        """The setup executor cannot run numerical code, a different installer or scheduler command."""
        with mock.patch.object(setup.subprocess, 'run') as process:
            for command in (['sbatch', 'x'], [setup.OLD_PYTHON, '-I', '-B', '-c', 'import torch'], [setup.OLD_PYTHON, '-m', 'pip', 'install', 'torch']):
                with self.subTest(command=command), self.assertRaisesRegex(ValueError, 'only the exact reviewed'):
                    setup.execute(None, command, self.root)
            process.assert_not_called()

    def test_installed_wheel_bytes_include_the_sympy_private_data_file(self):
        """The exact Sympy manpage scheme stays under newprefix and is checked alongside source files."""
        prefix = self.root / 'private'
        purelib = prefix / 'lib/python3.12/site-packages'
        purelib.mkdir(parents=True)
        package = purelib / 'sympy/__init__.py'
        package.parent.mkdir()
        package.write_bytes(b'synthetic code bytes')
        page = prefix / 'share/man/man1/isympy.1'
        page.parent.mkdir(parents=True)
        page.write_bytes(b'synthetic manual bytes')
        wheel = self.root / 'sympy.whl'
        with zipfile.ZipFile(wheel, 'w') as archive:
            archive.writestr('sympy/__init__.py', package.read_bytes())
            archive.writestr('sympy-1.13.1.data/data/share/man/man1/isympy.1', page.read_bytes())
        row = {'path': str(wheel), 'sha256': 'a' * 64}
        self.assertEqual(len(meta.installed_wheel_files(row, purelib)['verified_files']), 2)
        page.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs from its locked bytes'):
            meta.installed_wheel_files(row, purelib)

    def test_private_prepare_selects_only_new_interpreter_and_output_paths(self):
        """All numerical source, reference configuration and asset identity fields retain their existing values."""
        payload, plan, base, assets, metadata, original = setup.local_payload(PLAN)
        effective = setup.effective_plan(base, plan, original)
        before, after = original['native_prepare_parameters'], effective['native_prepare_parameters']
        self.assertEqual({key for key in before if before[key] != after[key]}, {'python', 'output-dir', 'asset-record'})
        self.assertEqual(after['asset-record'], plan['asset_record'])
        self.assertEqual(after['python'], plan['private_python'])
        self.assertEqual(effective['native_prepare_command'][4], 'prepare')
        self.assertEqual(effective['native_prepare_command'][1:3], ['-I', '-B'])
        self.assertNotIn('-O', base.SSH_OPTIONS)

    def test_software_verification_uses_fresh_private_python_when_parent_torch_is_absent(self):
        """The parent may lack Torch metadata while the selected fresh child sees its CPUreference distribution."""
        base = setup.module_from_source('fixture_base', (ROOT / 'stage_aces_full_site_inputs.py').read_text(), setup.BASE_SHA)
        helper_root = self.root / 'helper'
        helper_path = helper_root / base.PREFIX / 'aces_slurm_site_validation.py'
        helper_path.parent.mkdir(parents=True)
        helper_path.write_text("""from pathlib import Path
from importlib import metadata
import hashlib,json,sys
def prepared(directory):
 return directory,directory,{'python':sys.executable}
def checked_software(plan):
 return {'installed_backend':{'packages':{'torch':metadata.version('torch')}}}
def receipt(path):
 return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
""")
        packet = self.root / 'packet'
        packet.mkdir()
        (packet / 'software_at_prepare.json').write_text('{"fixture":"CPUreference metadata only"}')
        plan = {**self.plan, 'private_python': sys.executable, 'native_packet_directory': str(packet)}
        original = {'native_helper_root': str(helper_root)}
        command = setup.verification_command(plan, original)
        def parent_missing(name):
            """Represent the original ACES parent interpreter's absent Torch without importing Torch."""
            raise package_metadata.PackageNotFoundError(name)
        with mock.patch.object(package_metadata, 'version', side_effect=parent_missing), \
                mock.patch.object(setup, 'CURRENT_PRIVATE_PYTHON', sys.executable), \
                mock.patch.object(setup, 'CURRENT_VERIFICATION', command), mock.patch.object(setup, 'COMMANDS', []):
            result = setup.verify_preparation(base, plan, original, self.root, dict(os.environ))
        self.assertTrue(result['passed'])
        self.assertEqual(result['python'], sys.executable)
        self.assertEqual(result['numerical_imports'], [])
        self.assertNotIn('torch', sys.modules)

    def test_old_python_software_verification_is_rejected_and_base_probe_stays_old(self):
        """A successful-looking old-parent verification cannot replace the new selected interpreter."""
        original = json.loads(Path(self.plan['original_plan']['path']).read_bytes())
        result = {'kind': 'selected_private_python_preparation_verification', 'passed': True,
            'python': setup.OLD_PYTHON, 'numerical_imports': []}
        with mock.patch.object(setup, 'execute', return_value=json.dumps(result)) as run:
            with self.assertRaisesRegex(ValueError, 'selected private interpreter'):
                setup.verify_preparation(None, self.plan, original, self.root, {})
            self.assertEqual(run.call_args.args[1][0], self.plan['private_python'])
        with mock.patch.object(setup, 'execute', return_value='{}') as run:
            setup.probe(None, self.plan, {'metadata_helper': '# synthetic metadata source'}, self.root, setup.OLD_PYTHON, hash_shared=True)
            self.assertEqual(run.call_args.args[1][0], setup.OLD_PYTHON)
            self.assertIn('snapshot(True)', run.call_args.args[1][4])

    def test_bootstrap_rejects_changed_setup_source_before_it_can_execute(self):
        """The small root-run wrapper verifies the whole setup-source hash before any action."""
        source = 'COMMANDS=[]\ndef remote_setup(payload):\n print("synthetic metadata")\n'
        payload = (json.dumps({'driver_source': source}) + '\n').encode()
        for digest, expected in ((setup.digest(source.encode()), 0), ('f' * 64, 1)):
            result = subprocess.run([sys.executable, '-I', '-B', '-c', setup.BOOTSTRAP, digest], input=payload, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, expected)
            self.assertEqual(result.stdout, b'synthetic metadata\n' if expected == 0 else b'')


if __name__ == '__main__':
    unittest.main()
