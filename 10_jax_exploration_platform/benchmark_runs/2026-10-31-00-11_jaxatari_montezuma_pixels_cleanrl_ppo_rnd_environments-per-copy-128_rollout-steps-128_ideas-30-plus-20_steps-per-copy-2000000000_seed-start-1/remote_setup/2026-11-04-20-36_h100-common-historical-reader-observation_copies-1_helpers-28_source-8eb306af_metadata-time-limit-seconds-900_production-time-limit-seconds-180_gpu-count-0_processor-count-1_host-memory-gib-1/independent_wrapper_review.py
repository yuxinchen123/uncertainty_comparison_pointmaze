"""Review one unregistered observation wrapper using temporary fixtures only."""
import ast
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import unittest

P = Path(__file__).resolve().parent
R = P.parent.parent
I = R.parents[2]
OUT = P / 'independent_wrapper_review.json'
checks, files = [], {}


def check(value, name):
    if not value:
        raise AssertionError(name)
    checks.append(name)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    path = Path(path)
    check(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
          and not path.is_symlink(), 'exact ordinary metadata/source: ' + str(path))
    value = digest(path.read_bytes())
    files[str(path)] = value
    return value


def read(path):
    sha(path)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            check(key not in result, 'unique JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def reference(value, path):
    check(value['path'] == str(path) and value['sha256'] == sha(path), 'exact packet reference: ' + path.name)


def git(*args):
    check(args[0] in {'cat-file', 'merge-base', 'rev-parse'}, 'read-only source Git command')
    return subprocess.run(['git', '-C', str(I), *args], capture_output=True)


def main():
    check(os.environ.get('SLURM_JOB_ID') == '6783003'
          and os.environ.get('SLURM_STEP_ID', '').isdigit()
          and os.environ.get('SLURM_CPUS_PER_TASK') == '1'
          and not os.environ.get('SLURM_STEP_GPUS') and not os.environ.get('CUDA_VISIBLE_DEVICES'),
          'exact owned CPU-only metadata step')
    check(not OUT.exists(), 'preserve all prior independent reviews')
    freeze_path = P / 'preparation_source_and_input_freeze.json'
    check(sha(freeze_path) == '496b9f1099ea4e35eea0af0922616fdd46c9eabe751a4a8ca32f12b67385f03c',
          'exact externally frozen 17-file preparation')
    freeze = read(freeze_path)
    check(len(freeze['files']) == 17 and len(freeze['external_reference_roles']) == 27,
          'exact frozen local and declared external metadata cardinalities')
    for name, value in freeze['files'].items():
        path = Path(value['path'])
        check(path == P / name and sha(path) == value['sha256'] and path.stat().st_size == value['bytes'],
              'frozen local bytes/length: ' + name)
    before = {path: sha(path) for path in files}
    manifest = read(P / 'manifest.yaml')
    plan = read(P / 'unregistered_guard_input_plan.json')
    identity = read(P / 'diagnostic_unit_identity.json')
    publication = read(P / 'source_publication_binding.json')
    refs = read(P / 'consumed_input_references.json')
    check(sha(P / 'unregistered_guard_input_plan.json') == '5283db6016e84b7aac950d2ff76b5ef38fe2bf935cab6c223bbe0ce93f6fc0e2'
          and sha(P / 'run_guard_observation.py') == '02a563ce3547fb37d2f160f9c51c0f8c2691d7ccd8be1a2fc880d7dac2eba22b'
          and sha(P / 'observe_guard_phases.py') == '4b3918181151625d589755c8fa7e1fb365397db14d80d5f68fa14fce11831a63',
          'exact reviewed new wrapper/copied observer/input plan')
    check(refs['references_by_role'] == freeze['external_reference_roles']
          and refs['full_raw_history_payload_snapshot_claimed'] is False,
          'explicit external reference declarations unchanged; no raw-payload snapshot claim')
    check(manifest['diagnostic_unit_id'] == identity['diagnostic_unit_id'] == freeze['diagnostic_unit_id'] == P.name
          and identity['registered'] is False and manifest['registered'] is False
          and identity['borrowed_input_unit_id'] == manifest['borrowed_input_unit_id'] == plan['unit_id']
          and plan['unit_id'] != P.name and identity['existing_unit_files_written'] is False,
          'fresh diagnostic identity separate from borrowed accepted one-copy identity')
    for key in ('production_guard_qualified', 'formal_timing_preparation_permitted_by_this_diagnostic',
                'admission_granted', 'source_runtime_and_tolerances_changed', 'optional_digest_cache_enabled',
                'old_reader_subprocess_internally_profiled', 'internal_large_read_lock'):
        check(manifest[key] is False, 'explicit unchanged/no-credit limit: ' + key)
    for key in ('scientific_credit', 'timing_credit', 'idea_credit', 'case_credit', 'final_training_credit',
                'production_guard_qualification_credit'):
        check(type(manifest[key]) is int and manifest[key] == 0, 'exact zero diagnostic credit: ' + key)
    check(manifest['diagnostic_timeout_seconds'] == 900 and manifest['production_timeout_seconds_unchanged'] == 180
          and manifest['current_common_helper_count'] == 28 and manifest['historical_helper_count'] == 26
          and plan['direct_historical_consumer_protocol']['digest_cache'] == 'disabled'
          and plan['direct_historical_consumer_protocol']['operational_admission'] is False,
          'separate diagnostic limit and unchanged isolated old-reader/cache protocol')
    check(plan['helpers']['source_commit'] == publication['helper_source_commit'] == freeze['helper_source_commit']
          == '8eb306af9f68aed5ab5cc0d01f75b35a1f234078'
          and len(plan['helpers']['files']) == publication['helper_count'] == 28
          and plan['helpers']['files'] == publication['helper_files'] == freeze['helper_files'],
          'exact 28-helper metadata source namespace')
    ancestor = git('merge-base', '--is-ancestor', publication['helper_source_commit'], publication['published_integration_commit'])
    check(ancestor.returncode == publication['ancestor_returncode'] == 0
          and digest(ancestor.stdout) == publication['ancestor_stdout_sha256']
          and digest(ancestor.stderr) == publication['ancestor_stderr_sha256']
          and publication['published_integration_commit'] == '2980acc18ff240ef431021c8ac7b952b7dab43e5',
          'actual published source ancestry independently recomputed')
    for relative, expected in plan['helpers']['files'].items():
        blob = git('cat-file', 'blob', publication['helper_source_commit'] + ':' + relative)
        check(blob.returncode == 0 and digest(blob.stdout) == expected
              == sha(Path(plan['helpers']['source_root']) / relative),
              'exact detached helper/Git blob: ' + relative)
    reference(manifest['input_plan'], P / 'unregistered_guard_input_plan.json')
    reference(manifest['observer_source'], P / 'observe_guard_phases.py')
    reference(manifest['wrapper_source'], P / 'run_guard_observation.py')
    reference(manifest['consumed_input_references'], P / 'consumed_input_references.json')
    reference(manifest['diagnostic_identity'], P / 'diagnostic_unit_identity.json')
    command = manifest['actual_command']
    check(command == ['srun', '--jobid=6783003', '--overlap', '--exact', '--ntasks=1', '--cpus-per-task=1',
          '--mem=1G', 'flock', '-x', str(R / 'queue/cpu_monitor_large_artifact_read.lock'),
          '/usr/bin/python3', '-I', '-S', '-B', str(P / 'run_guard_observation.py')]
          and (P / 'command.txt').read_text().splitlines()[-1] == ' '.join(command),
          'one explicit external large-read flock and bounded root-only actual command')
    wrapper_text = (P / 'run_guard_observation.py').read_text()
    wrapper_tree = ast.parse(wrapper_text)
    calls = [node for node in ast.walk(wrapper_tree) if isinstance(node, ast.Call)]
    check(sum(isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
              and node.func.value.id == 'observer' and node.func.attr == 'parent' for node in calls) == 1
          and 'subprocess' not in wrapper_text and 'runpy' not in wrapper_text,
          'wrapper calls unchanged parent once in same process, without another supervisor')
    check("os.O_RDONLY | os.O_NOFOLLOW" in wrapper_text
          and "with (PACKET / name).open('xb')" in wrapper_text
          and "'st_ino'" in wrapper_text and "'st_ctime_ns'" in wrapper_text
          and "before == after" in wrapper_text,
          'explicit read-only stable file identities and exclusive fixed local outputs')
    check(not any((P / name).exists() for name in ('wrapper_intent.json', 'diagnostic_intent.json',
          'diagnostic_observation.json', 'input_snapshot_before.json')), 'actual observation remains uninvoked')
    # Nothing below may open genuine campaign inputs, spawn a process, or use numerical/device/native operations.
    forbidden_inputs = {value['path'] for value in refs['references_by_role'].values()}
    denied_events = {'subprocess.Popen', 'socket.connect', 'socket.bind', 'socket.getaddrinfo',
                     'ctypes.dlopen', 'ctypes.dlsym', 'os.system'}
    numerical = {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'jaxtari', 'nvidia', 'cupy', 'exploration_platform'}
    def audit(event, values):
        if event in denied_events or (event == 'import' and values[0].split('.')[0] in numerical):
            raise AssertionError('forbidden live operation in independent metadata fixtures: ' + event)
        if event == 'open' and isinstance(values[0], (str, bytes)):
            path = os.fsdecode(values[0])
            if path in forbidden_inputs or path.startswith(('/dev/nvidia', '/dev/dri')):
                raise AssertionError('actual scientific/device input opened in fixture')
            if path.startswith(str(R) + '/') and not path.startswith(str(P) + '/'):
                raise AssertionError('existing campaign input opened in fixture')
    sys.addaudithook(audit)
    specification = importlib.util.spec_from_file_location('independent_snapshot_fixture_only', P / 'test_run_guard_observation.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    check(result.testsRun == 7 and result.wasSuccessful() and not result.errors and not result.failures,
          'seven independent temporary wrapper fixtures pass under actual-input/process/device refusal')
    check(not numerical.intersection(sys.modules), 'no numerical modules imported')
    check(all(sha(path) == expected for path, expected in before.items()), 'all 17 frozen input/source bytes preserved after review')
    return {'kind': 'independent_fresh_h100_historical_reader_observation_wrapper_preparation_review', 'passed': True,
            'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
            'wrapper_sha256': sha(P / 'run_guard_observation.py'),
            'observer_sha256': sha(P / 'observe_guard_phases.py'),
            'exact_preparation_freeze_sha256': sha(freeze_path),
            'source_ancestry_independently_checked': True, 'helper_files_verified': 28,
            'declared_external_metadata_reference_roles': len(forbidden_inputs),
            'external_reference_role_count': len(refs['references_by_role']),
            'actual_external_scientific_inputs_opened_by_this_review': False,
            'actual_observation_invoked': False, 'source_or_runtime_or_tolerances_changed': False,
            'borrowed_identity_is_production_unit_proof': False, 'production_guard_qualified': False,
            'formal_timing_preparation_permitted': False, 'scientific_credit': 0, 'admission_granted': False,
            'review_scope': 'Approve only this exact fresh unregistered zero-credit observation wrapper and root-only declared command; actual before/after input identities, child waits and phase trace still require independent review after one real invocation.'}


started = dt.datetime.now(dt.timezone.utc).isoformat()
try:
    payload = main()
except BaseException as error:
    payload = {'passed': False, 'error': repr(error), 'traceback': traceback.format_exc(),
               'scientific_credit': 0, 'admission_granted': False}
payload.update({'schema_version': 1, 'started_at_utc': started,
                'finished_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'job_id': os.environ.get('SLURM_JOB_ID'), 'step_id': os.environ.get('SLURM_STEP_ID'),
                'pid': os.getpid(), 'requested_memory_mib': 512, 'checks_passed': len(checks),
                'checks': checks, 'source_and_packet_sha256': files,
                'review_program': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).resolve())}})
raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
with OUT.open('xb') as stream:
    stream.write(raw); stream.flush(); os.fsync(stream.fileno())
print(json.dumps({'passed': payload['passed'], 'checks_passed': payload['checks_passed'],
                  'tests_run': payload.get('tests_run'), 'step_id': payload['step_id'],
                  'receipt': str(OUT), 'sha256': digest(raw), 'error': payload.get('error')}))
raise SystemExit(0 if payload['passed'] else 1)
