"""Verify exact archived field reuse; run these NumPy checks on an allocated compute node."""

from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import pytest

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIRECTORY))
from initial_field_archive import (generation_identity, identity_digest,
    load_or_create_initial_field, uses_canonical_archive)
from exploration_platform.grid_initialization.environments import make_geometry
from exploration_platform.grid_initialization.fields import BonusField


def generated_field(geometry, seed, shift=0.):
    """Produce deterministic positive test arrays without initializing a neural network."""
    raw = np.arange(1, geometry.state_count+1, dtype=np.float32) + np.float32(shift)
    normalized = (raw.astype(np.float64)/raw.astype(np.float64).mean()).astype(np.float32)
    return BonusField(raw, normalized, {'kind': 'random_network', 'seed': seed,
        'state_count': geometry.state_count, 'test_generation': True})


def race_worker(run_dir, marker, output, shift):
    """Compete for one field while making any duplicate generation visible to the parent."""
    geometry = make_geometry('straight', 8)

    def generate():
        """Record each actual invocation while the production per-field lock is held."""
        with Path(marker).open('a') as handle:
            handle.write(str(shift)+'\n')
            handle.flush()
        time.sleep(.1)
        return generated_field(geometry, 2500, shift)

    field = load_or_create_initial_field(Path(run_dir), geometry, 'random_network', 0, 2500,
        generate=generate, source_version='test-field-source')
    Path(output).write_text(json.dumps({'raw': field.raw.tolist(),
        'normalized': field.normalized.tolist(), 'archive': field.metadata['canonical_archive']}))


def test_confirmation_only_gate_preserves_development():
    """Both explicit waves and their task names require archives; development does not."""
    assert uses_canonical_archive({'wave': 'confirmation'})
    assert uses_canonical_archive({'wave': 'long-confirmation'})
    assert uses_canonical_archive({'name': 'confirmation_straight-width-32'})
    assert uses_canonical_archive({'name': 'long-confirmation_straight-width-32'})
    assert not uses_canonical_archive({'wave': 'fine-tuning', 'name': 'fine-tuning_example'})
    assert not uses_canonical_archive({'wave': 'field-search'})


def test_exact_reuse_preserves_metadata_coordinates_and_source_identity(tmp_path):
    """A second GPU's generator is not invoked after the canonical archive exists."""
    geometry = make_geometry('straight', 8)
    original = generated_field(geometry, 2500)
    first = load_or_create_initial_field(tmp_path, geometry, 'random_network', 0, 2500,
        generate=lambda: original, source_version='test-field-source')

    def forbidden_generation():
        """Fail if a supposedly frozen field is silently regenerated."""
        raise AssertionError('The canonical field must be loaded, not regenerated.')

    second = load_or_create_initial_field(tmp_path, geometry, 'random_network', 0, 2500,
        generate=forbidden_generation, source_version='test-field-source')
    np.testing.assert_array_equal(first.raw, second.raw)
    np.testing.assert_array_equal(first.normalized, second.normalized)
    assert first.metadata == second.metadata
    assert all(first.metadata[name] == value for name, value in original.metadata.items())
    path = Path(first.metadata['canonical_archive']['path'])
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['coordinates'], geometry.coordinates)
        manifest = json.loads(str(archive['manifest_json'].item()))
    assert manifest['generation']['field_source_version'] == 'test-field-source'
    assert manifest['generation']['field_seed'] == 2500


def test_identity_changes_for_each_scientific_generation_input():
    """Coordinates, source, kind, severity and seed each change the archive identifier."""
    geometry = make_geometry('straight', 8)
    baseline = generation_identity(geometry, 'random_network', 0, 2500, 'source-one')
    points = geometry.coordinates.copy()
    points[0, 0] += np.float32(.01)
    alternatives = [
        generation_identity(replace(geometry, coordinates=points), 'random_network', 0, 2500, 'source-one'),
        generation_identity(geometry, 'random_network', 0, 2500, 'source-two'),
        generation_identity(geometry, 'random_log_uniform', 0, 2500, 'source-one'),
        generation_identity(geometry, 'random_network', 1, 2500, 'source-one'),
        generation_identity(geometry, 'random_network', 0, 2501, 'source-one')]
    identifiers = {identity_digest(identity) for identity in [baseline, *alternatives]}
    assert len(identifiers) == 6
    assert 'policy_kind' not in baseline and 'field_transform' not in baseline


def test_shared_path_uses_base_geometry_and_transforms_do_not_modify_archive(tmp_path):
    """Straight and snake controls reuse one field before any later reverse or shuffle."""
    snake = make_geometry('snake', 4, 2)
    field_geometry = {'kind': 'straight', 'width': snake.state_count, 'turns': 0}
    specifications = [{'geometry': {'kind': 'straight', 'width': snake.state_count, 'turns': 0},
                       'field_geometry': field_geometry},
                      {'geometry': {'kind': 'snake', 'width': 4, 'turns': 2},
                       'field_geometry': field_geometry, 'field_transform': 'reverse'}]
    result = []
    for specification in specifications:
        base = make_geometry(**specification.get('field_geometry', specification['geometry']))
        field = load_or_create_initial_field(tmp_path, base, 'random_network', 0, 2500,
            generate=lambda: generated_field(base, 2500), source_version='test-field-source')
        result.append(field)
    np.testing.assert_array_equal(result[0].raw, result[1].raw)
    np.testing.assert_array_equal(result[0].normalized, result[1].normalized)
    assert result[0].metadata['canonical_archive'] == result[1].metadata['canonical_archive']
    reversed_values = result[1].normalized.copy()[::-1].copy()
    assert not np.array_equal(reversed_values, result[0].normalized)
    again = load_or_create_initial_field(tmp_path, base, 'random_network', 0, 2500,
        generate=lambda: generated_field(base, 2500, 99), source_version='test-field-source')
    np.testing.assert_array_equal(again.normalized, result[0].normalized)


@pytest.mark.parametrize('damage', ['inconsistent_array', 'truncated'])
def test_corruption_aborts_without_regenerating(tmp_path, damage):
    """Malformed published archives fail without calling the generator a second time."""
    geometry = make_geometry('straight', 8)
    field = load_or_create_initial_field(tmp_path, geometry, 'random_network', 0, 2500,
        generate=lambda: generated_field(geometry, 2500), source_version='test-field-source')
    path = Path(field.metadata['canonical_archive']['path'])
    if damage == 'truncated':
        path.write_bytes(b'broken archive')
    else:
        with np.load(path, allow_pickle=False) as archive:
            contents = {name: archive[name].copy() for name in archive.files}
        contents['raw'][0] += np.float32(1)
        np.savez_compressed(path, **contents)

    def forbidden_generation():
        """Treat regeneration after corruption as an explicit test failure."""
        raise AssertionError('Corruption must not regenerate a canonical field.')

    with pytest.raises(ValueError):
        load_or_create_initial_field(tmp_path, geometry, 'random_network', 0, 2500,
            generate=forbidden_generation, source_version='test-field-source')


def test_processes_create_one_archive_and_reuse_identical_arrays(tmp_path):
    """Three independent processes with different candidate fields invoke generation once."""
    context = multiprocessing.get_context('spawn')
    marker = tmp_path/'generation_calls.txt'
    processes = [context.Process(target=race_worker,
        args=(str(tmp_path), str(marker), str(tmp_path/f'process_{index}.json'), index)) for index in range(3)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            pytest.fail('The canonical archive process did not finish.')
        assert process.exitcode == 0
    assert len(marker.read_text().splitlines()) == 1
    outputs = [json.loads((tmp_path/f'process_{index}.json').read_text()) for index in range(3)]
    assert all(result == outputs[0] for result in outputs)
    assert len(list((tmp_path/'data/confirmation_initial_fields').rglob('*.npz'))) == 1
    assert not list((tmp_path/'data/confirmation_initial_fields').rglob('*.temporary-*'))


def test_real_random_network_field_roundtrip(tmp_path):
    """A real untrained coordinate network field passes exact archive round-trip validation."""
    geometry = make_geometry('straight', 8)
    first = load_or_create_initial_field(tmp_path, geometry, 'random_network', 0, 2500)

    def forbidden_generation():
        """Require the second call to load the already generated neural field."""
        raise AssertionError('The real neural field was already archived.')

    second = load_or_create_initial_field(tmp_path, geometry, 'random_network', 0, 2500,
                                          generate=forbidden_generation)
    np.testing.assert_array_equal(first.raw, second.raw)
    np.testing.assert_array_equal(first.normalized, second.normalized)
    assert first.metadata['trained'] is False
    assert first.metadata['canonical_archive'] == second.metadata['canonical_archive']


def test_confirmation_train_unit_entrypoint_archives_before_training(tmp_path, monkeypatch):
    """Exercise the real entry point on allocated CPU with its placement check explicitly mocked."""
    import jax
    import train_unit

    # The mock permits this numerical unit test on the allocated CPU; production still requires a GPU.
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    specification = {'name': 'confirmation_small_entrypoint', 'wave': 'confirmation',
        'geometry': {'kind': 'straight', 'width': 8, 'turns': 0}, 'policy_kind': 'tabular',
        'field_kind': 'random_network', 'severity_exponent': 0, 'total_timesteps': 8,
        'seed_start': 10000, 'seed_count': 1,
        'settings': [{'learning_rate': 2**-10, 'intrinsic_coefficient': 1.,
                      'entropy_coefficient': 2**-10, 'gamma_int': .99, 'rollout_steps': 8}]}
    path = tmp_path/'small_specification.json'
    path.write_text(json.dumps(specification))
    monkeypatch.setattr(sys, 'argv', ['train_unit.py', '--run-dir', str(tmp_path), '--specification', str(path)])
    train_unit.main()
    folder = tmp_path/'data/research'/specification['name']
    summary = json.loads((folder/'summary.json').read_text())
    assert summary['completed'] and len(summary['copy_records']) == 1
    with np.load(folder/'behavior.npz', allow_pickle=False) as behavior:
        assert int(behavior['visit_counts'].sum()) == 8
        np.testing.assert_array_equal(behavior['field_seed_ids'], [2500])
    shards = list(folder.glob('completed_copies_*.json'))
    metadata = json.loads(shards[0].read_text())['field_metadata'][0]
    assert Path(metadata['canonical_archive']['path']).exists()
