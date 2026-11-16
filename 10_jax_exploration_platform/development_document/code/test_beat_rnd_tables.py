"""Tests of the splicing, number formatting and fresh-seed tables of beat_rnd_tables.py (no run records are read)."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat_rnd_tables as tables  # noqa: E402
from maze_campaign.analysis import records  # noqa: E402

KEY = ('pointmaze-medium', 'arrival_plus_one')


def policy(seed, value, first_goal, weight):
    """A policy of two logging windows of 100 steps whose every episode returns `value`, first at the goal at `first_goal`."""
    settings = (('clip_coef', 0.1), ('ent_coef', 0.01), ('int_coef', weight), ('learning_rate', 0.006))
    return records.Policy(
        unit_id=f'unit-weight-{weight}', environment='PointMaze Medium', environment_identity='one',
        reward_setting='arrival_plus_one', method='m', environments_per_policy=4, planned_steps=200, settings=settings,
        seed=seed, base_seed=0, step_limit=400, complete=True, steps=np.array([100, 200]), episodes=np.array([2.0, 2.0]),
        arrivals=np.array([2.0, 2.0]), return_sum=np.array([2 * value, 2 * value]),
        environment_return_sum=np.array([2 * value, 2 * value]), length_sum=np.array([200.0, 200.0]),
        steps_to_goal_sum=np.array([200.0, 200.0]), coverage=None, first_goal_step=first_goal)


def group(values, first_goal, weight):
    """One configuration of eight seeds from 30000, seed i returning values[i % len(values)]."""
    members = [policy(30000 + index, values[index % len(values)], first_goal, weight) for index in range(8)]
    return records.Configuration('PointMaze Medium', 'arrival_plus_one', 'm', 4, 200, members[0].settings, tuple(members))


def test_inject_table_replaces_only_the_marked_block(tmp_path):
    """The text between one pair of markers is replaced; the caption and everything else stay."""
    tex = tmp_path / 'section.tex'
    tex.write_text('before\n% >>> AUTO-GENERATED TABLE START: a\nold\n% <<< AUTO-GENERATED TABLE END: a\n'
                   '\\caption{kept}\n')
    tables.inject_table(tex, 'a', 'new rows')
    assert tex.read_text() == ('before\n% >>> AUTO-GENERATED TABLE START: a\nnew rows\n'
                               '% <<< AUTO-GENERATED TABLE END: a\n\\caption{kept}\n')


def test_inject_table_refuses_a_missing_or_repeated_marker(tmp_path):
    """A marker that is absent or appears twice stops the generator."""
    tex = tmp_path / 'section.tex'
    tex.write_text('% >>> AUTO-GENERATED TABLE START: a\n% <<< AUTO-GENERATED TABLE END: a\n' * 2)
    with pytest.raises(SystemExit):
        tables.inject_table(tex, 'a', 'x')
    with pytest.raises(SystemExit):
        tables.inject_table(tex, 'b', 'x')


def test_number_cells():
    """Means, differences and step counts come out as LaTeX math with their marks."""
    assert tables.math('-1,784') == '$-1{,}784$'
    assert tables.mean_cell(0.94041, 0.00302, 4) == r'$0.9404$ \newline $\pm 0.0030$'
    found = {'mean': -0.0003, 'interval': [-0.0067, 0.0059]}
    assert tables.difference_cell(found, 4, 'best') == r'{\bfseries\boldmath $-0.0003$} \newline $[-0.0067, +0.0059]$'
    assert tables.difference_text(found) == '$-0.0003$ $[-0.0067, +0.0059]$'
    assert tables.styled('x', 'second') == r'\underline{x}'
    assert tables.round_steps(8388608) == '8.39M'
    assert tables.round_steps(1048576) == '1.05M'


def test_case_verdict_needs_every_rnd_configuration():
    """A case is ahead only when ahead of each RND configuration, behind when behind any, else level."""
    assert tables.case_verdict(['ahead', 'ahead']) == 'ahead'
    assert tables.case_verdict(['ahead', 'level']) == 'level'
    assert tables.case_verdict(['ahead', 'behind']) == 'behind'


def test_rnd_role_names_where_a_configuration_was_chosen():
    """An RND row says which choices it is; a configuration that is neither is an earlier choice of the sweeps."""
    chosen, other = group([0.5], 100, 4.0), group([0.5], 100, 1.0)
    assert tables.rnd_role(chosen, chosen, chosen, 'whole') == (
        'RND, configuration for $W$ from its development sweeps and its validation blocks')
    assert tables.rnd_role(chosen, other, chosen, 'discovery') == 'RND, configuration for $M$ from its validation blocks'
    assert tables.rnd_role(other, chosen, chosen, 'whole') == 'RND, an earlier choice for $W$ of its development sweeps'


def test_fresh_tables_golden_path():
    """The arm against two RND configurations: ahead of one and level with the other, so the case is level on W;
    ahead of both on M. Rows come out sorted, with the verdicts and the case's verdict in its banner."""
    method, arm = group([0.9, 0.92], 50, 4.0), group([0.1], 50, 16.0)
    weaker, equal = group([0.5, 0.52], 150, 2.0), group([0.9, 0.92], 150, 4.5)
    whole = tables.compared_rows(method, [('RND, weaker', weaker), ('RND, equal', equal)], 200, 'whole', 'test')
    discovery = tables.compared_rows(arm, [('RND, weaker', weaker), ('RND, equal', equal)], 200, 'discovery', 'test')
    # W: RND rows from the highest W on; M: both RND rows arrive later than the arm.
    assert [row['words'] for row in whole] == ['RND, equal', 'RND, weaker']
    assert [row['verdict'] for row in whole] == ['level', 'ahead']
    assert [row['verdict'] for row in discovery] == ['ahead', 'ahead']
    assert whole[1]['result']['whole_curve_mean']['mean'] == pytest.approx(0.4)
    fresh = {KEY: {'case': tables.CASES[0], 'budget': 200,
                   'method': {'group': method, 'measures': tables.measures.configuration_measures(method.policies, 200)},
                   'arm': {'group': arm, 'measures': tables.measures.configuration_measures(arm.policies, 200)},
                   'whole': whole, 'discovery': discovery,
                   'verdict': {'whole': tables.case_verdict([row['verdict'] for row in whole]),
                               'discovery': tables.case_verdict([row['verdict'] for row in discovery])}}}
    table = tables.fresh_whole_table(fresh)
    assert r'\textbf{PointMaze Medium, reward 1 on arrival; verdict for the case: level}' in table
    assert table.count('N/A') == 3 and table.count(r'\cmidrule') == 1
    # Every seed differs by exactly 0.4, so every resample's mean, and both ends of the interval, are 0.4.
    assert r'$+0.4000$ \newline $[+0.4000, +0.4000]$ & $+0.4000$ \newline $[+0.4000, +0.4000]$ & ahead \\' in table
    table = tables.fresh_discovery_table(fresh)
    assert 'verdict for the case: ahead' in table
    assert r'$-0.1$ \newline $[-0.1, -0.1]$ & ahead \\' in table


def test_complete_side_names_a_missing_unit(tmp_path):
    """A unit whose records are absent stops the reading with an error naming it; a side with no unit does too."""
    (tmp_path / 'specifications').mkdir()
    (tmp_path / 'data').mkdir()
    unit = 'fresh-seeds_pointmaze-medium_reward-arrival-plus-one_held-from-first-arrival_seeds-30000-to-30255'
    (tmp_path / 'specifications' / f'{unit}.json').write_text(json.dumps({
        'unit_id': unit, 'phase': 'fresh-seeds', 'method': 'held-from-first-arrival', 'environment': KEY[0],
        'reward_setting': KEY[1], 'steps_per_policy': 1048576, 'windows_expected': 256,
        'cell_setting_names': ['learning_rate', 'ent_coef', 'int_coef', 'clip_coef'], 'cells': [[0.006, 0.01, 4.0, 0.1]]}))
    specifications = tables.case_sweeps.read_specifications(tmp_path)
    with pytest.raises(ValueError, match=f'{unit} \\(not started \\(no shard\\)\\)'):
        tables.complete_side(tmp_path, specifications, 'fresh-seeds', 'held-from-first-arrival', KEY, 'the method')
    with pytest.raises(ValueError, match='no unit'):
        tables.complete_side(tmp_path, specifications, 'fresh-seeds', 'fast-layer-step-0-point-1', KEY, 'the discovery arm')
