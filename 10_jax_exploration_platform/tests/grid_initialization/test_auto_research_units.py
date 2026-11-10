"""Tests of the auto research's unit builder: RND ideas, count-rule ideas and the frozen-field count idea."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/grid_initialization_campaign'))
import auto_research_units as units  # noqa: E402


def test_unit_specifications_of_the_three_kinds_of_idea():
    """An RND idea, a uniform count idea and a frozen-field count idea give 16 units each with the right fields."""
    ideas = {'rnd-idea': {'settings': {'rnd_readout': 'length'}},
             'count-idea': {'count': {'count_update': 'per_step', 'count_decay_exponent': 1.0}},
             'frozen-idea': {'count': {'count_decay_exponent': 1.0}, 'field': 'frozen_tabular_rnd'}}
    rows = units.unit_specifications('09', ideas, learning_rates=[1e-3, 1e-4])
    by_idea = {name: [row for row in rows if row['idea'] == name] for name in ideas}
    assert [len(by_idea[name]) for name in ideas] == [16, 16, 16]
    assert len({row['name'] for row in rows}) == 48
    rnd, count, frozen = (by_idea[name][0] for name in ideas)
    assert (rnd['intrinsic_source'], rnd['field_kind'], rnd['rnd_input']) == ('online_rnd', 'random_network', 'one_hot')
    assert (count['intrinsic_source'], count['field_kind'], count['rnd_input']) == ('count', 'uniform', 'coordinates')
    assert (frozen['intrinsic_source'], frozen['field_kind'], frozen['rnd_input']) == ('count', 'random_network', 'one_hot')
    assert frozen['method'] == 'frozen-tabular-rnd' and frozen['condition'].startswith('frozen-tabular-rnd-count_')
    # A count idea without an exponent keeps -0.5; one with an exponent carries it.
    assert count['count_decay_exponent'] == 1.0 and rnd['count_decay_exponent'] == 0.5
    assert len(rnd['settings']) == 7 and rnd['stop_after_timesteps'] == 2 ** 22


def test_unit_specifications_refuse_wrong_idea_keys():
    """Unknown keys, a field on an RND idea and an unknown field are refused."""
    for idea in ({'setting': {}}, {'settings': {}, 'field': 'frozen_tabular_rnd'},
                 {'count': {}, 'field': 'alternating'}, {'count': {'rnd_readout': 'length'}}):
        with pytest.raises(ValueError):
            units.unit_specifications('09', {'bad-idea': idea})


def test_development_stage_follows_the_fixed_settings_protocol():
    """Development units cover 2 learning rates x 2 entropies x 13 weights on seeds 0-15 at the full budget."""
    rows = units.unit_specifications('09', {'rnd-idea': {'settings': {'rnd_readout': 'length'}}}, stage='development')
    assert len(rows) == 16 and all(len(row['settings']) == 13 for row in rows)
    assert {(row['seed_start'], row['seed_count'], row['stop_after_timesteps']) for row in rows} == {(0, 16, None)}
    assert rows[0]['name'].startswith('development-09_') and rows[0]['wave'] == 'fixed-development-09'
    configurations = {(s['learning_rate'], s['entropy_coefficient'], s['intrinsic_coefficient'])
                      for row in rows if row['geometry']['kind'] == 'snake' and row['policy_kind'] == 'neural'
                      for s in row['settings']}
    assert len(configurations) == 52


def test_confirmation_units_use_the_configuration_the_development_units_chose(tmp_path):
    """Per case the smallest development score is frozen and queued on 1,024 seeds in units of 128."""
    ideas = {'rnd-idea': {'settings': {'rnd_readout': 'length'}}}
    rows = units.unit_specifications('09', ideas, stage='development')
    for index, row in enumerate(rows):
        folder = tmp_path / 'data/research' / row['name']
        folder.mkdir(parents=True)
        # Every setting scores 1000 except one per case, which is the smallest: weight 4 at the first unit of a case.
        settings = [{'setting_index': place, 'configuration': setting, 'goal_fraction': 1.0,
                     'capped_mean_first_goal_steps': 10.0 if index % 4 == 0 and setting['intrinsic_coefficient'] == 4.0 else 1000.0}
                    for place, setting in enumerate(row['settings'])]
        units.write_json(folder / 'summary.json', {'completed': True, 'specification': row, 'settings': settings})
    confirmation, frozen = units.confirmation_specifications(tmp_path, '09', ideas, 20000, 1024)
    assert len(confirmation) == 4 * 8 and len(frozen) == 4
    assert all(row['settings'] == [frozen[0]['configuration']] or row['settings'][0]['intrinsic_coefficient'] == 4.0
               for row in confirmation)
    assert {entry['configuration']['intrinsic_coefficient'] for entry in frozen} == {4.0}
    assert sorted({row['seed_start'] for row in confirmation}) == list(range(20000, 21024, 128))
    assert all(row['seed_count'] == 128 and row['stop_after_timesteps'] is None for row in confirmation)
    # An unfinished development grid is refused.
    (tmp_path / 'data/research' / rows[0]['name'] / 'summary.json').unlink()
    with pytest.raises(RuntimeError):
        units.confirmation_specifications(tmp_path, '09', ideas, 20000, 1024)


def test_screen_specification_is_unchanged_by_the_rewrite():
    """The builder gives exactly the specification queued for batch 4 before the stages were added."""
    import json
    old = json.loads(Path(__file__).with_name('data_screen_04_specification.json').read_text())
    ideas = {old['idea']: {'settings': old['rnd_settings'], 'network': old['network_options']}}
    new = {row['name']: row for row in units.unit_specifications('04', ideas, learning_rates=[1e-3, 1e-4])}
    assert new[old['name']] == old
