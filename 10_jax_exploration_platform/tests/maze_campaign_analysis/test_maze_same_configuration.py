"""Tests of arms compared at one configuration, of validation blocks that name several phases, and of the predictions
written in addenda of a batch's ideas.md.

Every made-up unit has 20 windows of 100 steps. A policy that reaches the goal from window k on has the whole-curve
mean (20 - k) / 20 with reward 1 on arrival and its first arrival at step 100 k + 8.
"""

from importlib import import_module
from pathlib import Path

import pytest

from test_maze_case_sweeps import add_unit
from maze_campaign.analysis import batch_summary, same_configuration, validation_seeds

MEDIUM = ('pointmaze-medium', 'arrival_plus_one')
SEEDS = (10000, 10001)
CASES = [{'environment': 'pointmaze-medium', 'reward_setting': 'arrival_plus_one'}]
TITLE = 'the arm against two sides'
BLOCK = [{'title': TITLE, 'arm': {'name': 'the arm', 'method': 'arm', 'phases': ['validation-block']},
          'against': [{'name': 'other side', 'method': 'other', 'phases': ['discovery-*']},
                      {'name': 'rnd elsewhere', 'method': 'rnd', 'phases': ['check'], 'run_folder': 'other-run'}],
          'cases': CASES}]


def pairs_run(tmp_path):
    """A run folder and a second one beside it.

    The arm (phase validation-block) at weight 1: W 0.9 twice, arrivals at step 208; at weight 2: W 0.8 twice; a
    third unit at weight 8 has written half its windows. The other side (phase discovery-validation) at weight 1:
    W 0.8 and 0.7 (steps 408 and 608); at learning rate 3e-4: no partner. RND in the second run folder at weight 2:
    W 0.8 twice.
    """
    run = tmp_path / 'pairs-run'
    add_unit(run, 'validation-block', 'arm', ((1e-4, 1.0), (1e-4, 2.0)), [2, 2, 4, 4], seeds=SEEDS)
    add_unit(run, 'validation-block', 'arm', ((1e-4, 8.0),), [0, 0], seeds=SEEDS, complete=False, tag='-unfinished')
    add_unit(run, 'discovery-validation', 'other', ((1e-4, 1.0), (3e-4, 1.0)), [4, 6, 1, 1], seeds=SEEDS)
    add_unit(tmp_path / 'other-run', 'check', 'rnd', ((1e-4, 2.0),), [4, 4], seeds=SEEDS)
    return run


def test_pairs_are_matched_on_the_cells_across_phases_and_run_folders(tmp_path):
    """Each configuration of the arm meets the same configuration of each side, from another phase or run folder."""
    found = same_configuration.read(pairs_run(tmp_path), BLOCK)
    case = found[0]['cases'][MEDIUM]

    # The other side: one pair at weight 1 (W +0.15, M -300 steps); the arm's weight 2 has no partner there.
    pairs, missing = case['pairs']['other']
    assert [pair['arm'].setting('int_coef') for pair in pairs] == [1.0] and [group.setting('int_coef') for group in missing] == [2.0]
    assert pairs[0]['result']['whole_curve_mean']['mean'] == pytest.approx(0.15)
    assert pairs[0]['result']['capped_first_goal_step']['mean'] == pytest.approx(-300.0)
    # RND of the second run folder: one pair at weight 2, no difference.
    pairs, _ = case['pairs']['rnd']
    assert [pair['arm'].setting('int_coef') for pair in pairs] == [2.0]
    assert pairs[0]['result']['whole_curve_mean']['mean'] == pytest.approx(0.0)

    # The lookup the prediction modules use: a pair by its settings, or the words for why there is none.
    assert same_configuration.pair_at(found, TITLE, MEDIUM, 'rnd', int_coef=2.0)[0] is pairs[0]
    missing_pair, words = same_configuration.pair_at(found, TITLE, MEDIUM, 'rnd', int_coef=8.0)
    assert missing_pair is None and 'Waiting for the configuration' in words


def test_the_section_shows_both_tables_the_counts_and_what_was_left_out(tmp_path):
    """One W table and one M table per case, the sides of zero, each side's best, and the unfinished unit."""
    text = same_configuration.section(same_configuration.read(pairs_run(tmp_path), BLOCK))
    assert f'### {TITLE}\n\n#### PointMaze Medium, reward 1 on arrival\n' in text
    whole, discovery = text.split('Whole-curve mean $W$ and final level $L$:')[1].split('Mean capped first-goal time $M$:')
    assert ('| other side | intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.001 | 0.900 ± 0.000 | '
            '0.750 ± 0.050 | +0.150 [+0.100, +0.200] |') in whole
    assert '| rnd elsewhere | intrinsic weight 2, learning rate 0.0001, entropy coefficient 0.001 | 0.800 ± 0.000 | 0.800 ± 0.000 |' in whole
    assert '| other side | intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.001 | 0.208 ± 0 | 0.508 ± 0.1 | -0.3 [' in discovery
    assert ('- against other side: $W$ ahead at 1, level at 0, behind at 0 of 1; $M$ sooner at 1, level at 0, later at 0 '
            'of 1; not trained by it: intrinsic weight 2, learning rate 0.0001, entropy coefficient 0.001.') in text
    assert '- other side: best $W$ 0.9500 at intrinsic weight 1, learning rate 0.0003' in text
    assert 'Not complete, left out: the arm: 2 of 3 configurations read (1 of 2 units complete).' in text


def test_a_side_with_no_complete_unit_compares_nothing(tmp_path):
    """An arm without a complete unit gives no pairs and says so."""
    run = tmp_path / 'empty-run'
    add_unit(run, 'validation-block', 'arm', ((1e-4, 1.0),), [2, 2], seeds=SEEDS, complete=False)
    add_unit(tmp_path / 'other-run', 'check', 'rnd', ((1e-4, 1.0),), [4, 4], seeds=SEEDS)
    text = same_configuration.section(same_configuration.read(run, BLOCK))
    assert 'Nothing is compared yet: the arm has no complete unit in this case.' in text


def test_a_first_block_of_two_phases_gives_the_control_its_best_of_both(tmp_path):
    """With a list of phases the control stands, per metric, at its best among the units of both phases."""
    run = tmp_path / 'two-phases-run'
    add_unit(run, 'validation-block', 'control', ((1e-4, 1.0),), [4, 6], seeds=SEEDS)
    add_unit(run, 'discovery-validation', 'control', ((3e-4, 1.0),), [1, 1], seeds=SEEDS)
    add_unit(run, 'validation-block', 'idea', ((1e-4, 2.0),), [2, 2], seeds=SEEDS)
    arms = [{'name': 'control', 'first_block': 'control', 'control': True}, {'name': 'idea', 'first_block': 'idea'}]
    one = validation_seeds.read_cases(run, {'first_block': 'validation-block', 'arms': arms, 'cases': CASES})
    both = {'first_block': ['validation-block', 'discovery-validation'], 'arms': arms, 'cases': CASES}
    two = validation_seeds.read_cases(run, both)

    # One phase: the control's one configuration. Two phases: its best for M is the discovery unit's (step 108).
    assert len(one[MEDIUM]['first']['sides']['control'].groups) == 1
    entry = two[MEDIUM]['first']['comparisons']['idea']['discovery']
    assert entry['baseline'].setting('learning_rate') == 3e-4
    assert entry['result']['capped_first_goal_step']['mean'] == pytest.approx(100.0)
    text = validation_seeds.section(both, two)
    assert 'The units of the phases `validation-block` and `discovery-validation` in PointMaze Medium' in text


def test_predictions_of_addenda_follow_those_of_the_predictions_section():
    """Numbered items after an addendum's predictions line, or its paragraph that starts "Prediction:", are added."""
    ideas = ('## Predictions\n\n1. **First.** one\n   more\n2. **Second.** two\n\n## What decides the next step\n\n1. not one\n\n'
             '## Addendum 1, written later\n\n1. A finding, not a prediction.\n\n**Predictions of the addendum.**\n\n'
             '3. Third.\n4. Fourth\n   continued.\n\n## Addendum 2\n\nQueued: units.\n\nPrediction: the fifth\none.\n\nAfter.\n'
             '\n## Addendum 3\n\nNo prediction here.\n')
    assert batch_summary.written_predictions(ideas) == ['**First.** one more', '**Second.** two', 'Third.',
                                                        'Fourth continued.', 'the fifth one.']


def test_the_prediction_modules_of_batches_8_to_14_check_every_written_prediction():
    """Each module of batches 8 to 14 has one entry per prediction of its batch's ideas.md (addenda included)."""
    runs = sorted((Path(__file__).resolve().parents[2] / 'runs').glob('2026-11-12-00-22_*'))
    if not runs:
        pytest.skip('the study folder of batches 8 to 14 is not here')
    for number in range(8, 15):
        folder = next((runs[0] / 'batches').glob(f'{number:02d}_*'))
        module = import_module(f'maze_campaign.analysis.predictions_batch_{number:02d}')
        assert len(batch_summary.written_predictions((folder / 'ideas.md').read_text())) == len(module.PREDICTIONS), number
