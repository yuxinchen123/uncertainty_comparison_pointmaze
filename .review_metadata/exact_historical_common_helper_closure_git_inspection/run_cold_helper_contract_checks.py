"""Check the exact metadata helper closure without replaying any saved scientific result."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
METADATA = Path(__file__).resolve().parent
SCRIPTS = ROOT / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
sys.path.insert(0, str(SCRIPTS))
import direct_common_confirmation as guard

NUMERICAL = {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'jaxtari', 'nvidia', 'cupy', 'exploration_platform'}
CHECKS = []


def require(condition, message):
    """Retain each exact source or synthetic contract check and fail at the first difference."""
    if not condition:
        raise ValueError(message)
    CHECKS.append(message)


def digest(raw):
    """Identify only ordinary source and small metadata bytes."""
    return hashlib.sha256(raw).hexdigest()


def audit(event, values):
    """Refuse live processes, numerical imports, devices, later library calls and network work."""
    if event in {'subprocess.Popen', 'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'ctypes.dlopen', 'ctypes.dlsym'}:
        raise RuntimeError('live operation forbidden in the cold metadata helper fixture')
    if event == 'import' and values[0].split('.')[0] in NUMERICAL:
        raise RuntimeError('numerical import forbidden in the cold metadata helper fixture')
    if event == 'open' and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(('/dev/nvidia', '/dev/dri')):
        raise RuntimeError('device operation forbidden in the cold metadata helper fixture')


def module_statements(raw, skip_function=None):
    """Compare every full module statement while excluding only the one declared changed function."""
    return [ast.dump(node, include_attributes=False) for node in ast.parse(raw).body
        if not (isinstance(node, ast.FunctionDef) and node.name == skip_function)]


def main():
    """Exercise only current source contracts and explicitly synthetic backend replies on owned CPU."""
    require(os.environ.get('SLURM_JOB_ID') == '6783003' and os.environ.get('SLURM_STEP_ID', '').isdigit(),
        'the actual owned CPU job and numeric step are declared')
    contract = json.loads((METADATA / 'source_change_contract.json').read_bytes())
    before = {str(ROOT / path): digest((ROOT / path).read_bytes()) for path in contract['helper_files']}
    require(before == {str(ROOT / path): value for path, value in contract['helper_files'].items()},
        'all twenty-eight exact current helper source hashes match the frozen declaration')
    bootstrap = guard.stdlib_ctypes_bootstrap()
    guard.validate_bootstrap_record(bootstrap)
    sys.addaudithook(audit)

    # Before: exact old common source. After: only the already reviewed Git argument function differs.
    relative = '10_jax_exploration_platform/scripts/atari_pixel_campaign/direct_common_confirmation.py'
    original = Path(contract['base_helper_root']) / relative
    require(module_statements(original.read_bytes(), 'source_inspection_git') ==
        module_statements((ROOT / relative).read_bytes(), 'source_inspection_git'),
        'every other original common-reader module statement is unchanged')
    for path, value in contract['helper_files'].items():
        if path != relative:
            require(digest((Path(contract['base_helper_root']) / path).read_bytes()) == value,
                'unchanged historical helper: ' + path)
    node = next(n for n in ast.parse((ROOT / relative).read_bytes()).body
        if isinstance(n, ast.FunctionDef) and n.name == 'source_inspection_git')
    require(digest(ast.dump(node, include_attributes=False).encode()) == contract['approved_function_ast_sha256'],
        'the sole changed function has the exact previously approved syntax')

    # Reading this small diagnostic plan does not read its saved result, review or binary payload references.
    input_reference = contract['read_only_input_plan']
    input_raw = Path(input_reference['path']).read_bytes()
    require(digest(input_raw) == input_reference['sha256'], 'the exact prior metadata input plan is unchanged')
    plan = json.loads(input_raw)
    plan['helpers'] = {'source_root': str(ROOT), 'source_commit': contract['base_commit'],
        'files': contract['helper_files']}
    require(guard.contract(plan) == plan['common_correctness_inputs'],
        'the common-reader metadata contract accepts the exact retained helper closure')
    import direct_measurement as support
    import direct_backend_image as backend
    names = {*support.helper_names(), 'measure_selected.py', 'direct_backend_image.py',
        'direct_common_confirmation.py', 'direct_historical_consumer.py'}
    require({support.core().PREFIX + name for name in names} == set(contract['helper_files']),
        'the cold helper API computes exactly the declared twenty-eight source files')
    historical = support.historical_support(plan)
    require(historical.__file__ == str(SCRIPTS / 'direct_historical_consumer.py'),
        'the cold exact historical-consumer import resolves only to this helper directory')

    # Before: retained backend-helper hash. After: a changed hash must still fail the same source gate.
    marker = {'synthetic_backend_metadata_only': True}
    with mock.patch.object(backend, 'checked_reference', return_value=marker), \
            mock.patch.object(backend, 'validate_expected', return_value=marker) as validate:
        require(backend.expected_for_plan(plan) == marker,
            'the unchanged backend source gate accepts this exact source with synthetic heavy-input replies')
        bad = copy.deepcopy(plan)
        bad['helpers']['files'][backend.PREFIX + backend.NAME] = 'f' * 64
        try:
            backend.expected_for_plan(bad)
        except ValueError as error:
            require('exact new backend helper' in str(error), 'a changed backend helper remains refused')
        else:
            raise ValueError('the backend helper-source gate was weakened')
        require(validate.call_count == 1, 'the changed-source negative case never reaches synthetic backend validation')
    require(not NUMERICAL.intersection(sys.modules), 'no numerical module was imported')
    require(before == {path: digest(Path(path).read_bytes()) for path in before},
        'all twenty-eight source files remained unchanged during these checks')
    print(json.dumps({'kind': 'exact_historical_common_helper_cold_metadata_contract_receipt', 'passed': True,
        'checks': CHECKS, 'checks_count': len(CHECKS), 'actual_job_id': os.environ['SLURM_JOB_ID'],
        'actual_step_id': os.environ['SLURM_STEP_ID'], 'actual_process': historical.process(os.getpid()),
        'source_sha256': before, 'stdlib_ctypes_bootstrap_before_audit': bootstrap,
        'candidate_source_base_commit': contract['base_commit'], 'candidate_source_is_uncommitted_test_freeze': True,
        'real_saved_scientific_results_replayed': False, 'historical_consumer_request_called': False,
        'backend_heavy_validation_replies': 'explicit synthetic mock in a source-gate-only fixture',
        'actual_git_commands_executed': 0, 'numerical_imports': [], 'gpu_operations': 0,
        'scientific_credit': 0, 'production_guard_qualified': False, 'admission_granted': False}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
