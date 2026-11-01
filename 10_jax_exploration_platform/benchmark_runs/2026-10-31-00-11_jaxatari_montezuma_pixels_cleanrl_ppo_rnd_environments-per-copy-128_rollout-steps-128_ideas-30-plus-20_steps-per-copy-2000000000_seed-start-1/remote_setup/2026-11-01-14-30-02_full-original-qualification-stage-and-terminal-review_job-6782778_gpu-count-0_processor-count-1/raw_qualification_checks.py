"""Read actual raw qualification bytes before any independent review exists.

The function body is exactly the staged reader's raw checks. Only the loading
of an already existing review and its final approval predicate are absent.
A separate source assertion checks this precise two-statement difference.
This file grants no admission and is not a producer or a future measurement.
"""
from pathlib import Path
from configured_measurement import (core, validate_original_qualification, verify_saved_sources,
    verify_qualification_source_receipts, validate_window, EXPECTED_DTYPES,
    renderer_contract, verify_qualification_backend, qualification_pytest_arguments,
    validate_test_collection, require, QUALIFICATION)

def verify_unreviewed_actual_raw_qualification(plan):
    """Refuse a missing, changed, unrelated or unreviewed actual full-original qualification."""
    identity = core()
    inputs = plan['target_qualification_inputs']
    result = identity.read_receipt(inputs['result'])
    directory = Path(inputs['result']['path']).parent
    qualified_plan = identity.read_receipt(identity.receipt(directory / 'measurement_plan.json'))
    require(Path(inputs['result']['path']).name == 'target_qualification_result.json' and identity.exact(qualified_plan.get('target_qualification'), QUALIFICATION), 'a generic successful measurement cannot substitute for target qualification')
    validate_original_qualification(qualified_plan)
    verify_saved_sources(directory, qualified_plan)
    verify_qualification_source_receipts(qualified_plan)
    metadata = identity.read_receipt(identity.receipt(directory / 'configured_isolation_metadata.json'))
    binding = identity.read_receipt(identity.receipt(directory / 'resolved_device_binding.json'))
    identity.validate_binding(qualified_plan, metadata, binding, directory)
    require(result.get('kind') == 'uva_configured_isolation_full_original_qualification' and result.get('completed') is True and identity.exact(result.get('qualification_protocol'), QUALIFICATION) and (result.get('source_commit') == plan['source_commit']) and (result.get('hardware_class') == plan['hardware_class']) and (result.get('runtime_environment') == plan['runtime_environment']) and (result.get('configured_isolation', {}).get('profile') == plan['configured_isolation']['profile']) and (result.get('asset_record') == plan['asset_record']) and (result.get('all_original_gates_passed') is True), 'the actual full-original qualification is incomplete or differs from this target')
    require(all((type(result.get(key)) is int and result[key] == 0 for key in ('scientific_credit', 'case_credit', 'idea_credit', 'packing_point_credit', 'throughput_credit', 'site_qualification_credit'))) and result.get('worker_admitted') is False and (result.get('admission_granted') is False), 'target qualification cannot grant scientific or admission credit')
    require(result.get('resolved_device_binding') == binding and result.get('helpers') == qualified_plan['helpers'] and (result.get('reference_iteration') == 1) and (result.get('executed_work', {}).get('full_reference_warmup_transitions') == 819200) and (result['executed_work'].get('full_reference_training_transitions') == 16384) and (result['executed_work'].get('full_reference_updates') == 1) and (result['executed_work'].get('full_reference_optimizer_minibatch_updates') == 16), 'qualification changed its actual physical worker or controlled full-original work')
    memory = result['qualification_memory']
    rebuilt_memory = validate_window(memory['assigned_gpu_process_observation'])
    require(all((memory.get(key) == value for (key, value) in rebuilt_memory.items())) and memory['gpu_uuid'] == binding['gpu_uuid'] and (memory['assigned_gpu_process_observation']['pid'] == metadata['process']['pid']), 'qualification memory does not belong to its actual resolved worker')
    require(identity.exact(result.get('dtypes'), EXPECTED_DTYPES) and type(result.get('finite_state_leaves')) is int and (result['finite_state_leaves'] > 0), 'full reference dtype groups or finite-state evidence are absent')
    for (key, expected) in (('reference_iteration', 1), ('reference_optimizer_step', 16)):
        require(type(result.get(key)) is int and result[key] == expected, 'a reference update counter changed type or value')
    for (key, expected) in (('full_reference_warmup_transitions', 819200), ('full_reference_training_transitions', 16384), ('full_reference_updates', 1), ('full_reference_optimizer_minibatch_updates', 16)):
        require(type(result['executed_work'].get(key)) is int and result['executed_work'][key] == expected, 'controlled full-original work counters must be exact integers')
    renderer = identity.read_receipt(identity.receipt(directory / 'original_renderer.json'))
    backend = identity.read_receipt(identity.receipt(directory / 'installed_backend.json'))
    matrix = renderer_contract(qualified_plan, backend)
    require(identity.exact(qualified_plan.get('renderer_contract'), matrix) and identity.exact(result.get('renderer_contract'), matrix) and (renderer == result['renderer']) and (renderer.get('checks') == {'cache': 'passed', 'palette': 'passed', 'fused': 'passed'}) and all((type(renderer.get(key)) is int and renderer[key] == matrix[key] for key in ('rooms', 'cases', 'synthetic_pixel_cases'))), 'the complete source-bound original renderer matrix is absent, reduced or failed')
    require(result['installed_backend'] == backend, 'actual installed backend changed between unit and result')
    verify_qualification_backend(qualified_plan, backend)
    import train_optimized as training
    loaded = result['loaded_gpu_libraries']
    training.validate_gpu_library_receipt(loaded, {'installed_gpu_libraries': backend['gpu_libraries']})
    require(type(loaded.get('pid')) is int and loaded['pid'] == metadata['process']['pid'] and (loaded.get('platform') in {'gpu', 'cuda'}) and isinstance(loaded.get('platform_version'), str) and loaded['platform_version'] and (binding['native']['driver_library'] in loaded.get('loaded_shared_libraries', [])), 'the actual loaded libraries are not this native-bound GPU process')
    for (path, value) in loaded['verified_package_libraries'].items():
        require(identity.receipt(path)['sha256'] == value['sha256'], 'an actual shared CUDA/cuDNN image changed')
    actual_assets = identity.read_receipt(identity.receipt(directory / 'asset_receipt.json'))
    require(actual_assets == identity.read_receipt(qualified_plan['qualification_software']['asset_receipt']), 'the actual original asset receipt changed')
    tests = identity.read_receipt(identity.receipt(directory / 'original_tests.json'))
    require(tests['command'] == [qualified_plan['python'], '-m', 'pytest', *qualification_pytest_arguments(qualified_plan)] and type(tests['returncode']) is int and (tests['returncode'] == 0) and (tests['reports'] == result['original_test_reports']) and tests['reports'] and all((row['outcome'] == 'passed' for row in tests['reports'])), 'full original test evidence changed')
    validate_test_collection(qualified_plan, tests, (directory / 'original_tests.stdout').read_text())
    require(tests['collected_nodeids'] == result['original_collected_test_nodeids'], 'saved collected-test identities changed')
    for (name, recorded) in result.get('gate_artifacts', {}).items():
        require(name in {'original_tests.json', 'original_tests.stdout', 'original_tests.stderr', 'original_renderer.json'} and identity.receipt(directory / name) == recorded, 'actual full-gate raw artifact changed')
    require(set(result.get('gate_artifacts', {})) == {'original_tests.json', 'original_tests.stdout', 'original_tests.stderr', 'original_renderer.json'} and result.get('reference_optimizer_step') == 16 and (result.get('frozen_target_unchanged') is True) and (result.get('policy_parameters_changed') is True), 'full reference learner state or gate outputs are incomplete')
    return result
