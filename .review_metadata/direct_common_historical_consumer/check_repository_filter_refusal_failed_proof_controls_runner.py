"""Check actual same-size Git filter refusal in temporary repositories on owned CPU only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
sys.path.insert(0, str(SCRIPTS))
import direct_common_confirmation as guard
import direct_historical_consumer as historical


def main():
    """Require clean/process/include refusal before status can invoke the harmless temporary marker."""
    if os.environ.get('SLURM_JOB_ID') != '6783003' or not os.environ.get('SLURM_STEP_ID', '').isdigit():
        raise RuntimeError('repository filter fixtures need the exact owned CPU step')
    paths = [Path(__file__), SCRIPTS / 'direct_common_confirmation.py', SCRIPTS / 'direct_historical_consumer.py',
        SCRIPTS / 'tests/test_direct_repository_configuration.py', SCRIPTS / 'tests/test_direct_common_confirmation.py',
        SCRIPTS / 'tests/test_direct_historical_consumer.py']
    sources = {str(path): historical.digest(path.read_bytes()) for path in paths}
    original_environment = dict(os.environ)
    environment = guard.metadata_environment(original_environment)
    active = {'enabled': False, 'root': None, 'commands': []}

    def audit(event, values):
        """Apply the unchanged actual production audit decision to the one temporary status attempt."""
        if event == 'subprocess.Popen' and active['enabled']:
            active['commands'].append(list(values[1]))
            guard.metadata_event_guard(event, values, {active['root']})
        if event == 'import' and values[0].split('.')[0] in historical.NUMERICAL:
            raise RuntimeError('numerical import forbidden in configuration fixture')
        if event in {'ctypes.dlopen', 'ctypes.dlsym', 'socket.connect', 'socket.bind', 'socket.getaddrinfo'}:
            raise RuntimeError('library or network operation forbidden in configuration fixture')
        if event == 'open' and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(('/dev/nvidia', '/dev/dri')):
            raise RuntimeError('device operation forbidden in configuration fixture')

    sys.addaudithook(audit)
    records = []
    for kind in ('ordinary_dirty_source', 'clean_filter', 'process_filter', 'included_filter', 'same_size_configuration_mutation'):
        guard._REPOSITORY_CONFIGURATION = {}
        with tempfile.TemporaryDirectory(prefix='author_git_filter_refusal_') as directory:
            root = Path(directory)
            setup = []
            for command in (['git', '-C', str(root), 'init', '--quiet'],):
                result = subprocess.run(command, env=environment, capture_output=True, check=True)
                setup.append({'command': command, 'returncode': result.returncode})
            tracked = root / 'tracked.txt'
            tracked.write_bytes(b'before\n')
            command = ['git', '-C', str(root), 'add', '--', 'tracked.txt']
            result = subprocess.run(command, env=environment, capture_output=True, check=True)
            setup.append({'command': command, 'returncode': result.returncode})
            marker = root / 'marker'
            program = root / 'harmless_filter.py'
            program.write_text("from pathlib import Path\nimport sys\nPath(__file__).with_name('marker').write_text('called')\nsys.stdout.buffer.write(sys.stdin.buffer.read())\n")
            config = root / '.git/config'
            if kind in ('clean_filter', 'process_filter', 'included_filter'):
                (root / '.gitattributes').write_text('tracked.txt filter=temporary_fixture\n')
                option = 'process' if kind == 'process_filter' else 'clean'
                values = '[filter "temporary_fixture"]\n ' + option + ' = /usr/bin/python3 -I -S -B ' + str(program) + '\n'
                if kind == 'included_filter':
                    included = root / 'included_config'
                    included.write_text(values)
                    values = '[include]\n path = ' + str(included) + '\n'
                with config.open('a') as stream:
                    stream.write('\n' + values)
            if kind == 'same_size_configuration_mutation':
                with config.open('a') as stream:
                    stream.write('\n# one\n')
                guard.verify_repository_configuration(str(root), environment)
                prior = config.stat()
                raw = config.read_bytes()
                changed = raw.replace(b'# one', b'# two')
                assert len(changed) == len(raw) and changed != raw
                config.write_bytes(changed)
                os.utime(config, ns=(prior.st_atime_ns, prior.st_mtime_ns))
            tracked.write_bytes(b'change\n')
            assert tracked.stat().st_size == len(b'before\n') and not marker.exists()
            active.update(enabled=True, root=str(root), commands=[])
            command = ['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all']
            refused = None
            result = None
            try:
                result = subprocess.run(command, env=environment, capture_output=True, check=True)
                configuration = guard.finish_repository_configurations()
            except ValueError as error:
                refused = str(error)
                configuration = None
            finally:
                active['enabled'] = False
            expected_refusal = kind != 'ordinary_dirty_source'
            assert (refused is not None) is expected_refusal, (kind, refused)
            assert not marker.exists(), 'the temporary filter ran before refusal'
            if not expected_refusal:
                assert result is not None and b'tracked.txt' in result.stdout and b'M' in result.stdout
            records.append({'kind': kind, 'passed': True, 'status_refused_before_execution': expected_refusal,
                'refusal': refused, 'marker_calls': 0, 'actual_audited_commands': active['commands'],
                'setup': setup, 'configuration_validation': configuration,
                'status_stdout_sha256': None if result is None else historical.digest(result.stdout),
                'status_returncode': None if result is None else result.returncode})
    specification = importlib.util.spec_from_file_location('repository_configuration_tests', SCRIPTS / 'tests/test_direct_repository_configuration.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    # These existing classes exercise changed parent/child proof handling without repeating unrelated broad checks.
    for index, (filename, classes) in enumerate((('test_direct_common_confirmation.py', ('DirectCommonConfirmationTests',)),
            ('test_direct_historical_consumer.py', ('ReturnedProofIdentityTests', 'ReadOnlyAuditTests')))):
        specification = importlib.util.spec_from_file_location('existing_configuration_controls_' + str(index), SCRIPTS / 'tests' / filename)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        for name in classes:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(getattr(module, name)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    assert original_environment == dict(os.environ) and not historical.NUMERICAL.intersection(sys.modules)
    assert all(historical.digest(Path(path).read_bytes()) == expected for path, expected in sources.items())
    print(json.dumps({'schema_version': 1, 'kind': 'actual_temporary_repository_filter_refusal_and_focused_metadata_checks',
        'passed': True, 'actual_step_id': os.environ['SLURM_STEP_ID'], 'actual_process': historical.process(os.getpid()),
        'source_sha256': sources, 'actual_temporary_repository_cases': records,
        'focused_tests_run': result.testsRun, 'parent_environment_unchanged': True,
        'numerical_imports': [], 'gpu_operations': 0, 'scientific_credit': 0,
        'saved_scientific_evidence_recomputed': False, 'qualification_established': False,
        'admission_granted': False, 'production_child_time_limit_seconds': historical.WAIT_SECONDS,
        'optional_digest_cache_enabled': False,
        'scope': 'Temporary fixtures only; pinned same-user configuration trust and mutation detection, with no kernel-enforced immutability claim.'},
        indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
