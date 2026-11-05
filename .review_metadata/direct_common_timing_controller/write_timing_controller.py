"""Create the narrow timing controller while retaining accepted process functions verbatim."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / '10_jax_exploration_platform/scripts/atari_pixel_campaign/direct_common_timing_controller.py'
ORIGINAL = Path(__file__).with_name('accepted_correctness_controller.py').read_text()
TREE = ast.parse(ORIGINAL)
FUNCTIONS = {node.name: ast.get_source_segment(ORIGINAL, node) for node in TREE.body if isinstance(node, ast.FunctionDef)}

NEW = {}
NEW['packet_metadata_paths'] = '''def packet_metadata_paths(plan):
    """Enumerate the fresh common-timing packet, twenty-nine helpers and reviewed history pointer."""
    fixed = ('preparation_request.json', 'metadata_preparation_result.json',
        'shared_software_assets_staged_verification.json', 'cold_metadata_validation.json',
        'reference_configuration.json', 'candidate_configuration.json', 'runtime_environment.json',
        'asset_record.json', 'direct_binding_profile.json', 'serial_predecessor_exit_inputs.json',
        'manifest.yaml', 'config_resolved.yaml', 'experiment_background.md', 'command.txt',
        'prepare_common_timing.py', 'canonical_controller_source_mapping.json')
    files = [PACKET / name for name in fixed]
    files += [UNIT / name for name in ('measurement_plan.json', 'measurement_prepared.json',
        'manifest.yaml', 'config_resolved.yaml', 'experiment_background.md', 'command.txt',
        'direct_backend_image_expected.json', 'direct_history/inputs.json')]
    names = [Path(path).name for path in plan['helpers']['files']]
    require(len(names) == len(set(names)) == 29, 'only the complete distinct twenty-nine-helper timing closure is admitted')
    files += [UNIT / 'code' / name for name in names]
    return sorted(set(files))'''
NEW['validate_packet_freeze'] = FUNCTIONS['validate_packet_freeze'].replace('actual_direct_common_correctness_packet_metadata_freeze', 'actual_direct_common_timing_packet_metadata_freeze')
NEW['check_backend_inputs'] = FUNCTIONS['check_backend_inputs'].replace('qualified backend26 package record for this new correctness plan', 'qualified package record for this new twenty-nine-helper common timing plan').replace('new direct correctness unit', 'new direct timing unit').replace('len(plan["helpers"]["files"]) == 26', 'len(plan["helpers"]["files"]) == 29').replace('plan.get("mode") == "correctness"', 'plan.get("mode") == "confirmation"').replace('"final_confirmation_protocol", "common_correctness_inputs", "device_memory_limit_protocol"', '"final_confirmation_protocol", "device_memory_limit_protocol", "native_process_control"').replace('correctness requires the exact qualified backend26 scope and actual qualification inputs', 'timing requires the reviewed snapshot namespace and actual qualification inputs')
NEW['checked_inputs'] = FUNCTIONS['checked_inputs'].replace('actual_prepared_direct_common_correctness_controller_inputs', 'actual_prepared_direct_common_timing_controller_inputs').replace('direct_common_correctness_metadata_preparation_result', 'direct_common_timing_metadata_preparation_result').replace('actual_direct_common_correctness_shared_software_asset_and_staged_metadata_verification', 'actual_direct_common_timing_shared_metadata_and_staged_verification').replace('plan["helpers"]["source_commit"] == HELPER_COMMIT', 'plan["helpers"]["source_commit"] == request["helper_source_commit"]').replace('    require(plan["helpers"] == qualified["helpers"], "the current correctness helper receipt must equal its qualified backend26 source")', '    timing_snapshot(plan, request)').replace('only the approved one/two-copy correctness units', 'only the approved one/two-copy common timing units').replace('frozen correctness command', 'frozen common timing command')
serial = FUNCTIONS['check_serial_predecessor_inputs']
serial = serial.replace('prior_plan["helpers"]["source_commit"] == HELPER_COMMIT', 'prior_plan["helpers"]["source_commit"] == ("0a880984b4a03a2ff7335337cc18576628bc9fc1" if request["serial_order"] == 1 else request["helper_source_commit"])')
serial = serial.replace('and prior_plan["helpers"]["source_root"] == request["helper_source_root"]', 'and (request["serial_order"] == 1 or prior_plan["helpers"]["source_root"] == request["helper_source_root"])')
serial = serial[:serial.index('    if request["serial_order"] == 1:')] + '''    require(type(request['serial_order']) is int and request['serial_order'] in (1, 2),
        'common timing requires one literal serial position')
    if request['serial_order'] == 1:
        accepted = document(request['accepted_evidence_snapshot_inputs']['proof']['path'])['accepted']['correctness']['2']
        require(inputs['plan'] == accepted['plan'] and inputs['result'] == accepted['result']
            and prior_plan['mode'] == 'correctness' and prior_plan['copy_seeds'] == [1, 2]
            and prior_plan['configuration']['learner']['n_copies'] == 2
            and prior_result.get('correctness_attempted') is True and prior_result.get('correctness_passed') is True
            and prior_result.get('implementation_distinct') is True and prior_result.get('measurement_valid') is False
            and prior_result.get('measurement_mode') == 'correctness'
            and prior_result.get('decision') == 'correctness_passed',
            'first-copy timing must follow the actual accepted second-copy correctness and both normal exits')
    else:
        require(prior_plan['mode'] == 'confirmation' and prior_plan['copy_seeds'] == [1]
            and prior_plan['configuration']['learner']['n_copies'] == 1
            and len(prior_plan['helpers']['files']) == 29
            and inputs['result']['path'] == str(previous / 'result.json')
            and prior_result.get('completed') is True
            and prior_plan['helpers'] == document(UNIT / 'measurement_plan.json')['helpers']
            and prior_plan['accepted_evidence_snapshot_inputs'] == request['accepted_evidence_snapshot_inputs']
            and prior_plan['direct_qualification_inputs'] == request['direct_qualification_inputs'],
            'second-copy timing must follow actual first-copy timing with the same reviewed source and snapshot')
        timing_support(prior_plan, previous, 'direct_common_confirmation').validate_pair_protocol(prior_plan, prior_result)
        reference = json.loads(json.dumps(request['reference_configuration']))
        candidate = json.loads(json.dumps(request['configuration']))
        reference['learner']['n_copies'] = candidate['learner']['n_copies'] = 1
        require(prior_plan['reference_configuration'] == reference and prior_plan['configuration'] == candidate,
            'the actual first-copy timing predecessor changed the original scientific configuration')
    return inputs'''
NEW['check_serial_predecessor_inputs'] = serial

EXTRA = '''
def timing_support(plan, unit, name):
    """Load only one exact staged metadata helper after verifying its declared source bytes."""
    path = Path(unit) / 'code' / (name + '.py')
    relative = '10_jax_exploration_platform/scripts/atari_pixel_campaign/' + path.name
    require(receipt(path)['sha256'] == plan['helpers']['files'][relative], 'the staged timing metadata helper changed')
    key = 'reviewed_timing_' + name + '_' + plan['helpers']['files'][relative]
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    no_numerical_modules()
    return sys.modules[key]


def timing_snapshot(plan, request):
    """Bind actual reviewed historical inputs to the exact current twenty-nine-helper timing namespace."""
    require(plan.get('mode') == 'confirmation' and type(plan.get('rounds')) is int and plan['rounds'] == 7
        and type(plan.get('iterations')) is int and plan['iterations'] == 10
        and type(plan.get('minimum_updates_per_round')) is int and plan['minimum_updates_per_round'] == 10
        and len(plan['helpers']['files']) == 29 and plan['helpers']['source_commit'] == request['helper_source_commit']
        and re.fullmatch('[0-9a-f]{40}', request['helper_source_commit'])
        and plan.get('accepted_evidence_snapshot_inputs') == request['accepted_evidence_snapshot_inputs']
        and plan.get('common_correctness_inputs') == request['common_correctness_inputs']
        and not any(key in plan for key in ('configured_isolation', 'direct_qualification', 'target_qualification',
            'completed_study', 'device_memory_limit_protocol', 'final_confirmation_protocol', 'native_process_control')),
        'only fresh seven-by-ten common timing with actual reviewed snapshot inputs is supported')
    deadline = time.monotonic() + 180
    root = Path(request['helper_source_root'])
    require(root.resolve(strict=True) == root, 'the actual timing helper checkout must be ordinary')
    for relative, expected in plan['helpers']['files'].items():
        require(relative.startswith('10_jax_exploration_platform/scripts/atari_pixel_campaign/')
            and Path(relative).name == relative.split('/')[-1] and len(Path(relative).parts) == 4
            and re.fullmatch('[0-9a-f]{64}', expected), 'only exact declared timing helper paths and hashes are allowed')
        remaining = deadline - time.monotonic()
        require(remaining > 0, 'timing helper source verification exceeded its metadata deadline')
        raw = subprocess.check_output(['git', '-C', str(root), 'show', request['helper_source_commit'] + ':' + relative],
            timeout=min(30, remaining))
        require(digest(raw) == expected == digest(ordinary_bytes(root / relative))
            == digest(ordinary_bytes(UNIT / 'code' / Path(relative).name)), 'timing helper Git, checkout and staged source differ')
    support = timing_support(plan, UNIT, 'accepted_evidence_snapshot')
    require(set(plan['helpers']['files']) == {'10_jax_exploration_platform/scripts/atari_pixel_campaign/' + name
        for name in support.required_helper_names()}, 'the current timing helper set is not the exact reviewed twenty-nine-file closure')
    selected = support.consume(plan)
    for count in ('1', '2'):
        accepted = selected.proof['accepted']['correctness'][count]
        require(request['accepted_correctness'][count] == {name: accepted[name] for name in ('plan', 'result', 'review')},
            'the request changed a genuinely accepted correctness plan, result or review')
    require(request['qualification_plan'] == selected.qual['plan']
        and request['direct_qualification_inputs'] == {name: selected.qual[name] for name in ('result', 'review')}
        and request['common_correctness_inputs'] == {name: selected.correctness[name] for name in ('result', 'review')},
        'the current timing request lost its exact original qualification or per-count correctness')
    return selected
'''

text = ORIGINAL
for name in NEW:
    text = text.replace(FUNCTIONS[name], NEW[name])
text = text.replace('Start only one separately reviewed common correctness unit after its actual predecessor exit.',
    'Start only one separately reviewed common timing unit after its actual predecessor exit.')
text = text.replace('HELPER_COMMIT = "0a880984b4a03a2ff7335337cc18576628bc9fc1"', 'HELPER_COMMIT = None  # The fresh request must bind the actual reviewed twenty-nine-helper commit.')
text = text.replace('\ndef main(argv=None):', EXTRA + '\n\ndef main(argv=None):')
TARGET.write_text(text)
changed = {node.name: ast.get_source_segment(text, node) for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
unchanged = [name for name in FUNCTIONS if ast.dump(ast.parse(FUNCTIONS[name]), include_attributes=False)
    == ast.dump(ast.parse(changed[name]), include_attributes=False)]
assert len(unchanged) == 25 and set(FUNCTIONS) - set(unchanged) == set(NEW)
print({'retained_process_publication_functions': len(unchanged), 'modified_admission_functions': sorted(NEW)})
