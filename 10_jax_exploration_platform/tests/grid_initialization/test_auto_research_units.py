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
