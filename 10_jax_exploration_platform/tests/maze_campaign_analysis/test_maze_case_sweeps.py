"""Tests of the sweeps read case by case: choosing, grid ends, pairing, and two made-up run folders end to end."""

import json

import pytest

from maze_record_examples import flat_policy, goal_windows, start_record, write_shard
from maze_campaign.analysis import batch_summary, case_sweeps, records

BATCH = '04_made-up-sweeps'
STEP_COST = {'ext_reward_scale': 1 / 400, 'ext_reward_offset': -1 / 400}
NAMES = ('learning_rate', 'int_coef')


def add_unit(run, phase, method, cells, goal, *, environment='pointmaze-medium', reward='arrival_plus_one',
             seeds=(0, 1), complete=True, started=True, tag=''):
    """One unit of 20 windows of 100 steps (2,000 steps per policy): its specification, and its shard when it started.

    `goal` names, per policy (cell by cell, seed by seed), the window from which it reaches the goal (None: never).
    An unfinished unit has written 10 of its windows.
    """
    family, map_name = environment.split('-')
    unit = f'{phase}_{environment}_{reward}_{method}{tag}'
    (run / 'specifications').mkdir(parents=True, exist_ok=True)
    (run / 'specifications' / f'{unit}.json').write_text(json.dumps({
        'unit_id': unit, 'phase': phase, 'method': method, 'environment': environment, 'reward_setting': reward,
        'cell_setting_names': ['learning_rate', 'int_coef'], 'cells': [list(cell) for cell in cells],
        'steps_per_policy': 2000, 'windows_expected': 20}))
    if started:
        start = start_record(unit, family=family, map_name=map_name, iterations=20, cells=cells, names=NAMES, seeds=seeds,
                             trainer=STEP_COST if reward == 'step_cost' else None)
        lines = [start] + goal_windows(start, goal)
        write_shard(run, lines if complete else lines[:11], complete=complete)
    return unit


def group(learning_rate, weight, values, first_steps, seeds=None):
    """One configuration in memory: one training curve and one first arrival per seed."""
    seeds = range(len(values)) if seeds is None else seeds
    return records.configurations([
        flat_policy(curve, seed=seed, first_goal_step=first, settings=(('int_coef', weight), ('learning_rate', learning_rate)))
        for seed, curve, first in zip(seeds, values, first_steps)])[0]


def side(slug, groups, planned=None):
    """A side in memory."""
    return case_sweeps.Side(slug=slug, name=slug.replace('-', ' '), units=(), groups=tuple(groups),
                            planned=planned or len(groups), budget=200)


def test_the_best_configuration_per_metric_and_the_ends_of_its_grid():
    """W takes the configuration that keeps its return; M the one that arrives first; ends are named per setting."""
    groups = [group(0.003, 0.25, [[1, 0], [1, 0]], [20, 40]), group(0.003, 0.5, [[0, 1], [0, 1]], [150, 150]),
              group(0.006, 0.25, [[1, 1], [0.5, 1]], [90, 110]), group(0.006, 0.5, [[0, 0], [0, 0]], [-1, -1]),
              group(0.01, 0.25, [[0, 0.5], [0, 0.5]], [160, 160]), group(0.01, 0.5, [[0, 0], [0, 1]], [-1, 180])]
    whole = case_sweeps.best_configuration(groups, 200, 'whole')
    discovery = case_sweeps.best_configuration(groups, 200, 'discovery')
    assert (whole.setting('learning_rate'), whole.setting('int_coef')) == (0.006, 0.25)
    assert (discovery.setting('learning_rate'), discovery.setting('int_coef')) == (0.003, 0.25)
    # The learning rate 0.006 is inside its grid of three; the weight 0.25 is the smaller of two.
    assert case_sweeps.grid_ends(groups, whole) == [('int_coef', 'smallest')]
    assert case_sweeps.grid_ends(groups, discovery) == [('int_coef', 'smallest'), ('learning_rate', 'smallest')]
    assert case_sweeps.ends_words(case_sweeps.grid_ends(groups, discovery)) == (
        'intrinsic weight at the smallest of its grid; learning rate at the smallest')
    assert case_sweeps.ends_words([]) == 'no'
    # A grid of one configuration has no end, and a setting that is not a number in every configuration is skipped.
    assert case_sweeps.grid_ends(groups[:1], groups[0]) == []
    assert case_sweeps.number_values(groups, 'int_coef') == [0.25, 0.5] and case_sweeps.number_values(groups, 'preset') == []
    assert case_sweeps.described_settings([side('a', groups[:1]), side('b', groups[1:2])]) == ['int_coef']
    assert case_sweeps.described_settings([side('a', groups), side('b', groups[:1])]) == ['int_coef', 'learning_rate']
    assert case_sweeps.best_words(side('a', groups), whole, ['int_coef']) == 'intrinsic weight 0.25 (best of 6)'
    assert case_sweeps.best_words(side('a', groups[:1]), groups[0], ['int_coef']) == 'intrinsic weight 0.25 (the only configuration read)'
    assert case_sweeps.configuration_words(whole, ['int_coef', 'learning_rate']) == 'intrinsic weight 0.25, learning rate 0.006'
    assert case_sweeps.configuration_words(whole, []) == 'its only configuration'


def test_a_method_is_compared_with_the_baseline_on_the_seeds_both_have():
    """Each side at its own best configuration; no baseline, one common seed and no method are all said."""
    method = side('idea', [group(0.003, 1.0, [[0.5, 1], [0.3, 1]], [100, 140]), group(0.003, 4.0, [[1, 0], [1, 0]], [20, 40])])
    baseline = side('cleanrl-ppo-rnd', [group(0.003, 1.0, [[0, 1], [0, 1]], [150, 250]),
                                        group(0.006, 1.0, [[0, 0.5], [0, 0.5]], [120, 130])])
    found = case_sweeps.compare_at_best(method, baseline, 200, 'test', 'batch')
    # W: the method at the weight 1 (0.75 and 0.65) against the baseline at the learning rate 0.003 (0.5 twice).
    assert found['whole']['best'].setting('int_coef') == 1.0 and found['whole']['baseline'].setting('learning_rate') == 0.003
    assert found['whole']['result']['whole_curve_mean']['mean'] == pytest.approx(0.2) and found['whole']['result']['seeds'] == 2
    # M: the method at the weight 4 (steps 20 and 40) against the baseline at the learning rate 0.006 (120 and 130).
    assert found['discovery']['best'].setting('int_coef') == 4.0 and found['discovery']['baseline'].setting('learning_rate') == 0.006
    assert found['discovery']['result']['capped_first_goal_step']['mean'] == pytest.approx(-95.0)
    assert found['discovery']['ends'] == [('int_coef', 'largest')] and found['discovery']['result']['ahead'] is True
    # No baseline configuration: the method's best is still named, with nothing to compare it with.
    alone = case_sweeps.compare_at_best(method, side('cleanrl-ppo-rnd', []), 200, 'test', 'batch')
    assert alone['whole']['result'] is None and alone['whole']['baseline'] is None
    assert case_sweeps.compare_at_best(side('idea', []), baseline, 200, 'test', 'batch') is None
    # One seed in common: a difference without an interval.
    other = side('idea', [group(0.003, 1.0, [[1, 1], [1, 1]], [10, 10], seeds=(1, 9))])
    assert case_sweeps.compare_at_best(other, baseline, 200, 'test', 'batch')['whole']['result']['seeds'] == 1
    table = case_sweeps.metric_table('whole', baseline, [(other, case_sweeps.compare_at_best(other, baseline, 200, 'test', 'batch'))],
                                     ['int_coef', 'learning_rate'], 200)
    assert '| idea | intrinsic weight 1, learning rate 0.003 (the only configuration read) | **1.000** ± 0.000 | fewer than two seeds in common |' in table
    assert table.splitlines()[2].startswith('| cleanrl ppo rnd | intrinsic weight 1, learning rate 0.003 (best of 2) | <u>0.500</u> ± 0.000 | baseline |')


def test_cases_of_a_study_description_and_the_selection_of_units():
    """Cases keep the description's order; a case with two budgets is refused; units are selected by phase patterns."""
    description = {'sweeps': [
        {'method': 'a', 'environments': ['pointmaze-medium'], 'reward_settings': ['arrival_plus_one'], 'steps_per_policy': 10},
        {'method': 'b', 'environments': ['pointmaze-medium'], 'reward_settings': ['arrival_plus_one'], 'steps_per_policy': 10},
        {'method': 'a', 'environments': ['pointmaze-large'], 'reward_settings': ['step_cost'], 'steps_per_policy': 40}]}
    assert case_sweeps.study_cases(description) == {
        ('pointmaze-medium', 'arrival_plus_one'): {'budget': 10, 'methods': ['a', 'b']},
        ('pointmaze-large', 'step_cost'): {'budget': 40, 'methods': ['a']}}
    description['sweeps'][1]['steps_per_policy'] = 20
    with pytest.raises(ValueError, match='one length per case is needed'):
        case_sweeps.study_cases(description)
    assert case_sweeps.case_words('pointmaze-large', 'step_cost') == 'PointMaze Large, step cost'
    assert case_sweeps.case_words('antmaze-umaze', 'arrival_plus_one') == 'AntMaze U-maze, reward 1 on arrival'
    specifications = [{'phase': phase, 'method': method, 'environment': 'pointmaze-medium', 'reward_setting': 'step_cost',
                       'steps_per_policy': steps, 'unit_id': f'{phase}-{method}-{steps}'}
                      for phase, method, steps in (('development-at-the-budget', 'm', 10), ('development-at-the-budget-lower', 'm', 10),
                                                   ('development-at-the-budget', 'm', 40), ('development-first-pass', 'm', 10),
                                                   ('development-at-the-budget', 'other', 10))]
    chosen = case_sweeps.select(specifications, ['development-at-the-budget*'], 'm', 'pointmaze-medium', 'step_cost', 10)
    assert [specification['unit_id'] for specification in chosen] == ['development-at-the-budget-m-10', 'development-at-the-budget-lower-m-10']
    assert len(case_sweeps.select(specifications, ['development-at-the-budget'], 'm', 'pointmaze-medium', 'step_cost')) == 2
    with pytest.raises(ValueError, match="development_sweeps: unknown keys \\['colour'\\], missing keys \\['baseline'\\]"):
        case_sweeps.check_keys({'colour': 1, 'study_description': 's', 'method_names': {}}, case_sweeps.DEVELOPMENT_KEYS,
                               case_sweeps.DEVELOPMENT_KEYS - {'validation'}, 'development_sweeps')


def made_up_sweeps(tmp_path):
    """A batch of two methods in two cases beside a baseline run folder; one unit unfinished, one never started."""
    run, baseline = tmp_path / 'sweep-run', tmp_path / 'baseline-run'
    folder = run / 'batches' / BATCH
    folder.mkdir(parents=True)
    grid = ((1e-4, 1.0), (1e-4, 4.0), (3e-4, 1.0), (3e-4, 4.0))
    # Idea a in PointMaze Medium with reward 1 on arrival. W per configuration: 0.5; 0.8 and 0.7; 0.9 twice (first
    # arrival at step 208); 1.0 and 0.0 (one policy arrives at step 8, the other never).
    add_unit(run, 'development-at-the-budget', 'idea-a', grid, [10, 10, 4, 6, 2, 2, 0, None])
    # Idea b there has written half its windows; in the second case idea a is complete and idea b never started.
    add_unit(run, 'development-at-the-budget', 'idea-b', grid, [0] * 8, complete=False)
    add_unit(run, 'development-at-the-budget', 'idea-a', ((1e-4, 1.0),), [0, 10], environment='pointmaze-large', reward='step_cost')
    add_unit(run, 'development-at-the-budget', 'idea-b', ((1e-4, 1.0),), [0, 0], environment='pointmaze-large', reward='step_cost',
             started=False)
    sweep = lambda method, environment, reward: {'method': method, 'environments': [environment], 'reward_settings': [reward],
                                                 'steps_per_policy': 2000}
    (folder / 'sweeps.json').write_text(json.dumps({'phase': 'development-at-the-budget', 'sweeps': [
        sweep(method, environment, reward) for environment, reward in (('pointmaze-medium', 'arrival_plus_one'),
                                                                       ('pointmaze-large', 'step_cost'))
        for method in ('idea-a', 'idea-b')]}))
    (folder / 'summary_settings.json').write_text(json.dumps({
        'title': 'Made-up sweeps', 'development_sweeps': {
            'study_description': 'sweeps.json', 'method_names': {'idea-a': 'idea a', 'idea-b': 'idea b'},
            'baseline': {'run_folder': 'baseline-run', 'method': 'cleanrl-ppo-rnd', 'name': 'CleanRL PPO-RND',
                         'phases': ['development-at-the-budget*'], 'confirmation_phases': ['confirmation']},
            'validation': {'phase': 'validation-block', 'control': 'control',
                           'method_names': {'control': 'control', 'idea-a': 'idea a'}}}}))
    # The baseline in the first case only: two units of two phases. W 0.8 and 0.7 (arrivals at steps 408 and 608),
    # then 0.4 twice; in the second unit 0.6 twice. Its best configuration for both metrics is the first one.
    add_unit(baseline, 'development-at-the-budget', 'cleanrl-ppo-rnd', ((1e-4, 1.0), (1e-4, 4.0)), [4, 6, 12, 12])
    add_unit(baseline, 'development-at-the-budget-higher-learning-rates', 'cleanrl-ppo-rnd', ((3e-4, 1.0),), [8, 8])
    # Picked configurations of the first case on two validation seeds: the control (W 0.8 and 0.7; first arrivals at
    # steps 408 and 608) and idea a at two configurations (W 0.9 twice with arrivals at step 208; W 1.0 twice with
    # arrivals at step 8).
    seeds = (10000, 10001)
    add_unit(run, 'validation-block', 'control', ((1e-4, 1.0),), [4, 6], seeds=seeds)
    add_unit(run, 'validation-block', 'idea-a', ((3e-4, 1.0),), [2, 2], seeds=seeds, tag='_first')
    add_unit(run, 'validation-block', 'idea-a', ((3e-4, 4.0),), [0, 0], seeds=seeds, tag='_second')
    # A unit of another method and a unit of another length are not the baseline's sweep.
    add_unit(baseline, 'development-at-the-budget', 'another-method', ((1e-4, 1.0),), [0, 0])
    # The baseline's first configuration on two confirmation seeds: W 0.9 twice, first arrival at step 208.
    add_unit(baseline, 'confirmation', 'cleanrl-ppo-rnd', ((1e-4, 1.0),), [2, 2], seeds=(30000, 30001))
    return run


def test_a_made_up_batch_of_development_sweeps_end_to_end(tmp_path):
    """A batch without a screen: measures, units, per case two tables and the grids; unfinished units are named."""
    run = made_up_sweeps(tmp_path)
    result = batch_summary.generate(run, BATCH)
    text = (run / 'batches' / BATCH / 'summary.md').read_text()
    assert text.startswith('# Made-up sweeps\n\nGenerated ') and '\n## Units\n' not in text and '## Records that need a look' not in text
    assert text.index('## Measures\n') < text.index('## Development sweeps against the baseline, case by case\n')
    section = text.split('## Development sweeps against the baseline, case by case\n')[1]
    assert 'run folder `baseline-run`, phases `development-at-the-budget*`' in section

    # Units: four of this batch, by case; the unfinished one and the one without a shard are named.
    units = section.split('### Units\n')[1].split('### PointMaze Medium')[0]
    assert '| 1 | idea a | PointMaze Medium, reward 1 on arrival | 4 | complete | 20 of 20 |' in units
    assert '| 2 | idea b | PointMaze Medium, reward 1 on arrival | 4 | unfinished | 10 of 20 | not finished | none |' in units
    assert '| 4 | idea b | PointMaze Large, step cost | 1 | not started (no shard) | 0 of 20 | not finished | none |' in units
    assert '2. `development-at-the-budget_pointmaze-medium_arrival_plus_one_idea-b`' in units

    # First case. The baseline's best for both metrics is (learning rate 0.0001, weight 1) of its 3 configurations:
    # W 0.8 and 0.7, arrivals at steps 408 and 608. Idea a's best is (0.0003, 1): W 0.9 twice, arrival at step 208.
    first = section.split('### PointMaze Medium, reward 1 on arrival\n')[1].split('### PointMaze Large, step cost\n')[0]
    assert 'Read at 2,000 environment steps per policy. CleanRL PPO-RND: 3 of 3 configurations read (2 of 2 units complete).' in first
    assert 'idea a: 4 of 4 configurations read (1 of 1 units complete). idea b: 0 of 4 configurations read (0 of 1 units complete).' in first
    assert 'Not read yet, no complete unit: idea b.' in first
    whole = first.split('#### Discovery and optimization: whole-curve mean\n')[1].split('#### ')[0]
    assert ('| CleanRL PPO-RND | intrinsic weight 1, learning rate 0.0001 (best of 3) | <u>0.750</u> ± 0.050 | baseline '
            '| 1.000 ± 0.000 | baseline | intrinsic weight at the smallest of its grid; learning rate at the smallest | 2 |') in whole
    assert ('| idea a | intrinsic weight 1, learning rate 0.0003 (best of 4) | **0.900** ± 0.000 | +0.150 [+0.100, +0.200] '
            '| 1.000 ± 0.000 | +0.000 [+0.000, +0.000] | intrinsic weight at the smallest of its grid; learning rate at the largest | 2 |') in whole
    assert 'On 2 confirmation seeds the same configuration of CleanRL PPO-RND has $W$ 0.900 ± 0.000.' in whole
    discovery = first.split('#### Discovery: mean capped first-goal time\n')[1].split('#### ')[0]
    assert '| CleanRL PPO-RND | intrinsic weight 1, learning rate 0.0001 (best of 3) | <u>0.508</u> ± 0.1 | baseline | 2 of 2 |' in discovery
    assert ('| idea a | intrinsic weight 1, learning rate 0.0003 (best of 4) | **0.208** ± 0 | -0.3 [-0.4, -0.2] | 2 of 2 '
            '| **0.900** | 1.000 |') in discovery
    assert 'On 2 confirmation seeds the same configuration of CleanRL PPO-RND has $M$ 0.208 ± 0 thousand steps.' in discovery
    assert ('Best configuration for $W$ at an end of its grid: idea a (intrinsic weight at the smallest of its grid; '
            'learning rate at the largest); CleanRL PPO-RND (') in first and 'The edge rule of `program.md`' in first
    # The grids: one row per learning rate, one column per weight; W then M in thousands of steps.
    grids = first.split('#### Whole-curve mean on every configuration of the methods\n')[1]
    assert '| method and its other settings | weight 1 | weight 4 |' in grids
    assert '| idea a, learning rate 0.0001 | 0.500 | <u>0.750</u> |' in grids and '| idea a, learning rate 0.0003 | **0.900** | 0.500 |' in grids
    assert '| idea a, learning rate 0.0003 | **0.208** | 1 |' in grids.split('#### Mean capped first-goal time on every configuration')[1]
    # The picked configurations of the case on the validation seeds: every configuration, then best against best for
    # both metrics, the control's only configuration against idea a's better one (weight 4: W 1.0, arrival at step 8).
    picked = first.split('#### On the validation seeds: picked configurations of this case\n')[1]
    assert 'on 2 seeds, from 10000 to 10001' in picked and 'picked as the leading ones of each side\'s sweep' in picked
    assert '| idea a | intrinsic weight 4, learning rate 0.0003 | **1.000** ± 0.000 | 1.000 ± 0.000 | **0.008** ± 0 | 2 of 2 |' in picked
    assert '| control | intrinsic weight 1, learning rate 0.0001 | 0.750 ± 0.050 | 1.000 ± 0.000 | 0.508 ± 0.1 | 2 of 2 |' in picked
    whole_best, discovery_best = picked.split('Best against best for $W$ (the highest is the best):')[1].split(
        'Best against best for $M$ (the smallest is the best):')
    assert ('| idea a | intrinsic weight 4, learning rate 0.0003 (best of 2) | **1.000** ± 0.000 | +0.250 [+0.200, +0.300] '
            '| 1.000 ± 0.000 | +0.000 [+0.000, +0.000] | 2 |') in whole_best
    assert '| control | intrinsic weight 1, learning rate 0.0001 (the only configuration read) | <u>0.750</u> ± 0.050 | control |' in whole_best
    assert '| idea a | intrinsic weight 4, learning rate 0.0003 (best of 2) | **0.008** ± 0 | -0.5 [-0.6, -0.4] | 2 of 2 |' in discovery_best
    assert 'at an end of its grid' not in picked and '##### Units' in picked
    assert '#### On the validation seeds' not in section.split('### PointMaze Large, step cost\n')[1]

    # Second case: the step cost, no baseline unit. A policy that arrives from the first window on has the return
    # (1 - 40) / 400 = -0.0975 throughout; the other has -1 in the first half: W -0.0975 and -0.54875.
    second = section.split('### PointMaze Large, step cost\n')[1]
    assert 'CleanRL PPO-RND: 0 of 0 configurations read (0 of 0 units complete).' in second
    assert '| idea a | its only configuration | -0.323 ± 0.226 | the baseline has no complete configuration |' in second
    assert 'No best configuration is at an end of its grid.' in second
    cases = result['screen'].cases
    assert list(cases) == [('pointmaze-medium', 'arrival_plus_one'), ('pointmaze-large', 'step_cost')]
    assert cases[('pointmaze-medium', 'arrival_plus_one')]['comparisons']['idea-b'] is None
    # A batch without a screen takes none of the keys that belong to one.
    settings = json.loads((run / 'batches' / BATCH / 'summary_settings.json').read_text())
    (run / 'batches' / BATCH / 'summary_settings.json').write_text(json.dumps({**settings, 'control': 'c'}))
    with pytest.raises(ValueError, match="unknown keys \\['control'\\], missing keys \\[\\]"):
        batch_summary.generate(run, BATCH)


def test_the_discovery_checks_on_the_validation_seeds_and_on_the_development_seeds(tmp_path):
    """Units of one configuration each on the validation seeds; sweeps on the development seeds with a repeated configuration."""
    run, baseline = tmp_path / 'discovery-run', tmp_path / 'baseline-run'
    seeds = (10000, 10001)
    one = lambda phase, method, cell, goal, tag, **more: add_unit(run, phase, method, (cell,), goal, tag=tag, **more)
    # The control: first arrivals at steps 408 and 608, and at 808 twice. The idea: at 208 twice; its second unit
    # has written half its windows.
    one('discovery-validation', 'control', (1e-4, 1.0), [4, 6], '_a', seeds=seeds)
    one('discovery-validation', 'control', (3e-4, 1.0), [8, 8], '_b', seeds=seeds)
    one('discovery-validation', 'idea', (1e-4, 4.0), [2, 2], '_a', seeds=seeds)
    one('discovery-validation', 'idea', (3e-4, 4.0), [0, 0], '_b', seeds=seeds, complete=False)
    # The idea's two sweeps on the development seeds share the configuration (0.0001, 4): it is read once.
    add_unit(run, 'development-for-discovery', 'idea', ((1e-4, 4.0), (3e-4, 4.0)), [6, 6, 1, 3])
    add_unit(run, 'development-for-discovery-grid', 'idea', ((1e-4, 4.0), (1e-4, 16.0)), [6, 6, 9, 9])
    add_unit(baseline, 'development-for-discovery', 'cleanrl-ppo-rnd', ((1e-4, 1.0), (3e-4, 1.0)), [4, 4, 5, 7])
    add_unit(baseline, 'confirmation-for-discovery', 'cleanrl-ppo-rnd', ((1e-4, 1.0),), [9, 9], seeds=(30000, 30001))
    block = {'environment': 'pointmaze-medium', 'reward_setting': 'arrival_plus_one',
             'validation': {'phase': 'discovery-validation', 'control': 'control', 'method_names': {'control': 'control', 'idea': 'the idea'}},
             'development': {'phases': ['development-for-discovery', 'development-for-discovery-grid'], 'method_names': {'idea': 'the idea'},
                             'baseline': {'run_folder': 'baseline-run', 'method': 'cleanrl-ppo-rnd', 'name': 'CleanRL PPO-RND',
                                          'phases': ['development-for-discovery'], 'confirmation_phases': ['confirmation-for-discovery']}}}
    text = case_sweeps.discovery_section(run, 'made-up batch', block)
    assert text.startswith('## Discovery checks beyond the screen\n') and 'in PointMaze Medium, reward 1 on arrival' in text

    # The validation seeds: every configuration, then best against best (208 twice against 408 and 608).
    validation = text.split('### On the validation seeds: the leading configurations for the first-goal time\n')[1].split('### On the development seeds')[0]
    assert 'on 2 seeds, from 10000 to 10001' in validation and 'Read at 2,000 environment steps per policy' in validation
    assert '| control | intrinsic weight 1, learning rate 0.0001 | <u>0.750</u> ± 0.050 | 1.000 ± 0.000 | <u>0.508</u> ± 0.1 | 2 of 2 |' in validation
    assert '| control | intrinsic weight 1, learning rate 0.0003 | 0.600 ± 0.000 | 1.000 ± 0.000 | 0.808 ± 0 | 2 of 2 |' in validation
    assert '| the idea | intrinsic weight 4, learning rate 0.0001 | **0.900** ± 0.000 | 1.000 ± 0.000 | **0.208** ± 0 | 2 of 2 |' in validation
    assert 'Best against best for $M$ (the smallest is the best):' in validation and 'Best against best for $W$' not in validation
    assert ('| the idea | intrinsic weight 4, learning rate 0.0001 (the only configuration read) | **0.208** ± 0 '
            '| -0.3 [-0.4, -0.2] | 2 of 2 | **0.900** | 1.000 | 2 |') in validation
    assert ('| control | intrinsic weight 1, learning rate 0.0001 (best of 2) | <u>0.508</u> ± 0.1 | control | 2 of 2 '
            '| <u>0.750</u> | 1.000 | 2 |') in validation
    assert 'at an end of its grid' not in validation
    assert 'difference in $M$ from the control [95% interval], thousand steps' in validation
    assert 'Not complete, left out: the idea: 1 of 2 configurations read (1 of 2 units complete).' in validation
    assert '| 4 | the idea | PointMaze Medium, reward 1 on arrival | 1 | unfinished | 10 of 20 |' in validation

    # The development seeds: the idea's three configurations (one trained twice) against the baseline's two.
    development = text.split("### On the development seeds: discovery sweeps against the baseline's\n")[1]
    assert 'CleanRL PPO-RND: 2 of 2 configurations read (1 of 1 units complete). the idea: 3 of 3 configurations read (2 of 2 units complete).' in development
    # The idea's best: (0.0003, 4), arrivals at steps 108 and 308; the baseline's best: (0.0001, 1), at step 408 twice.
    assert '| CleanRL PPO-RND | intrinsic weight 1, learning rate 0.0001 (best of 2) | <u>0.408</u> ± 0 | baseline | 2 of 2 |' in development
    assert ('| the idea | intrinsic weight 4, learning rate 0.0003 (best of 3) | **0.208** ± 0.1 | -0.2 [-0.3, -0.1] | 2 of 2 '
            '| **0.900** | 1.000 | intrinsic weight at the smallest of its grid; learning rate at the largest | 2 |') in development
    assert '**A best of many configurations on few seeds is too low.**' in development
    assert 'On 2 confirmation seeds the same configuration of CleanRL PPO-RND has $M$ 0.908 ± 0 thousand steps.' in development
    assert ('Best configuration for $M$ at an end of its grid: the idea (intrinsic weight at the smallest of its grid; '
            'learning rate at the largest); CleanRL PPO-RND (learning rate at the smallest of its grid).') in development
    assert 'Best configuration for $W$' not in development

    # A baseline whose best configuration was not confirmed: the configurations its confirmation units hold are named.
    add_unit(baseline, 'development-for-discovery', 'cleanrl-ppo-rnd', ((3e-4, 16.0),), [0, 0], tag='_more')
    later = case_sweeps.discovery_section(run, 'made-up batch', block)
    assert ("CleanRL PPO-RND's best configuration here has no complete unit on the confirmation seeds (phases "
            '`confirmation-for-discovery`): the confirmation units of this case hold intrinsic weight 1, learning rate 0.0001.') in later
    # The same units as a section of their own, for both metrics: the control's best for W is its first configuration.
    own = case_sweeps.validation_section(run, 'made-up batch', {**block['validation'], 'environment': 'pointmaze-medium',
                                                               'reward_setting': 'arrival_plus_one'})
    assert own.startswith('## Validation seeds: each side at the best of all its configurations\n') and '### Units' in own
    assert 'Best against best for $W$ (the highest is the best):' in own and 'Best against best for $M$' in own
    assert '| control | intrinsic weight 1, learning rate 0.0001 (best of 2) | <u>0.750</u> ± 0.050 | control |' in own
    with pytest.raises(ValueError, match="validation_of_configurations: unknown keys \\[\\], missing keys \\['environment', 'reward_setting'\\]"):
        case_sweeps.validation_section(run, 'made-up batch', block['validation'])
    # With no complete unit a part says so and still lists its units; an unknown key of a block is refused.
    empty = case_sweeps.discovery_section(tmp_path / 'baseline-run', 'made-up batch', block)
    assert 'The phase `discovery-validation` has no unit in this case.' in empty and 'No unit of these sweeps is complete yet.' in empty
    # Units of the phase that have not finished: the part lists them and reads nothing.
    late = tmp_path / 'late-run'
    add_unit(late, 'discovery-validation', 'control', ((1e-4, 1.0),), [4, 6], seeds=seeds, complete=False)
    assert 'No unit of the phase `discovery-validation` is complete yet.' in case_sweeps.discovery_section(
        late, 'made-up batch', {key: value for key, value in block.items() if key != 'development'})
    with pytest.raises(ValueError, match="discovery: unknown keys \\['colour'\\]"):
        case_sweeps.discovery_section(run, 'made-up batch', {**block, 'colour': 'blue'})
