"""Tests of the validation seeds read case by case on two blocks, of the standing page, and of the readings of batches 5 to 7.

Every made-up unit has 20 windows of 100 steps. A policy that reaches the goal from window k on has the whole-curve
mean (20 - k) / 20 with reward 1 on arrival, the final level 1 and its first arrival at step 100 k + 8.
"""

import json

import pytest

from test_maze_case_sweeps import add_unit
from maze_campaign.analysis import (batch_summary, predictions_batch_05, predictions_batch_06, predictions_batch_07, standing,
                                    validation_seeds)

BATCH = '06_made-up-blocks'
FIRST, SECOND = (10000, 10001), (20000, 20001)
MEDIUM = ('pointmaze-medium', 'arrival_plus_one')
ARMS = [{'name': 'control', 'first_block': 'control', 'second_block': 'control', 'control': True},
        {'name': 'idea a', 'first_block': 'idea-a', 'second_block': 'idea-a-short', 'family': 'family a'},
        {'name': 'idea b', 'second_block': 'idea-b', 'family': 'family b'},
        {'name': 'idea c', 'first_block': 'idea-c', 'second_block': 'idea-c', 'family': 'family c'},
        {'name': 'idea d', 'second_block': 'idea-d', 'family': 'family d'},
        {'name': 'idea e', 'first_block': 'idea-e', 'family': 'family e'}]
BLOCK = {'first_block': 'validation-block', 'second_block': 'validation-block-two', 'arms': ARMS,
         'cases': [{'environment': 'pointmaze-medium', 'reward_setting': 'arrival_plus_one'}]}


def two_blocks(tmp_path):
    """A run folder with one case on two blocks of two seeds each.

    First block. The control at (learning rate 1e-4, weight 1): W 0.8 and 0.7, arrivals at steps 408 and 608; at
    (3e-4, 1): W 0.6 twice, arrivals at step 808. Idea a at (3e-4, 1): W 0.9 twice (step 208); at (3e-4, 4): W 0.7
    twice. Idea c at (1e-4, 2): W 0.85 twice. Idea e at (1e-4, 8): the control's numbers.
    Second block. The control at (1e-4, 1): W 0.7 and 0.6 (steps 608 and 808); at (3e-4, 1): W 0.8 twice (step
    408), so its best here is not its first-block best. Idea a at (3e-4, 1): W 0.9 and 0.8 (steps 208 and 408); at
    (3e-4, 4): W 1.0 twice. Idea b, never run on the first block, at (1e-4, 4): W 0.95 twice (step 108). Idea c at
    (3e-4, 2), not its first-block configuration: W 0.75 twice (step 508). Idea d has written half its windows.
    """
    run = tmp_path / 'blocks-run'
    add_unit(run, 'validation-block', 'control', ((1e-4, 1.0), (3e-4, 1.0)), [4, 6, 8, 8], seeds=FIRST)
    add_unit(run, 'validation-block', 'idea-a', ((3e-4, 1.0), (3e-4, 4.0)), [2, 2, 6, 6], seeds=FIRST)
    add_unit(run, 'validation-block', 'idea-c', ((1e-4, 2.0),), [3, 3], seeds=FIRST)
    add_unit(run, 'validation-block', 'idea-e', ((1e-4, 8.0),), [4, 6], seeds=FIRST)
    add_unit(run, 'validation-block-two', 'control', ((1e-4, 1.0), (3e-4, 1.0)), [6, 8, 4, 4], seeds=SECOND)
    add_unit(run, 'validation-block-two', 'idea-a-short', ((3e-4, 1.0), (3e-4, 4.0)), [2, 4, 0, 0], seeds=SECOND)
    add_unit(run, 'validation-block-two', 'idea-b', ((1e-4, 4.0),), [1, 1], seeds=SECOND)
    add_unit(run, 'validation-block-two', 'idea-c', ((3e-4, 2.0),), [5, 5], seeds=SECOND)
    add_unit(run, 'validation-block-two', 'idea-d', ((1e-4, 1.0),), [0, 0], seeds=SECOND, complete=False)
    return run


def test_a_made_up_batch_on_two_blocks_end_to_end(tmp_path):
    """A batch of validation units only: the first block at each side's best, the second block with both comparisons."""
    run = two_blocks(tmp_path)
    folder = run / 'batches' / BATCH
    folder.mkdir(parents=True)
    (folder / 'summary_settings.json').write_text(json.dumps({'title': 'Made-up blocks', 'validation_seeds': BLOCK}))
    result = batch_summary.generate(run, BATCH)
    text = (folder / 'summary.md').read_text()
    assert text.index('## Measures\n') < text.index('## Validation seeds, case by case\n') and '## Predictions' not in text
    case = text.split('### PointMaze Medium, reward 1 on arrival\n')[1]
    first, second = case.split('#### Second block: seeds that chose nothing\n')

    # First block: each side at its best there. Idea b and idea d have no unit of that phase and are not listed.
    assert first.startswith('\n#### First block: each side at its best configuration there\n') or '#### First block' in first
    assert 'on 2 seeds, from 10000 to 10001' in first and 'idea b' not in first and 'idea d' not in first
    assert ('| idea a | intrinsic weight 1, learning rate 0.0003 (best of 2) | **0.900** ± 0.000 | **+0.150** [+0.100, +0.200] |'
            in first.split('Best against best for $W$ (the highest is the best):')[1])

    # Second block, W. The control twice: at its first-block best (0.65) and at its best here (0.8). Idea a stands at
    # its first-block best (0.85) although its other configuration has 1.0 here: +0.2 twice under equal rules, +0.1
    # and 0 with the control favoured. Idea b has no first-block unit; idea c's first-block best was not run here.
    assert 'on 2 seeds, from 20000 to 20001. Nothing was chosen on these seeds' in second
    whole, discovery = second.split('Second block, $W$ (the highest is the best):')[1].split('Second block, $M$ (the smallest is the best):')
    rows = [line for line in whole.splitlines() if line.startswith('| ')][1:]    # without the header
    assert [row.split(' | ')[0] for row in rows] == ['| control, at its first-block best', '| control, at its best on this block',
                                                     '| idea b', '| idea a', '| idea c']
    assert rows[0] == ('| control, at its first-block best | intrinsic weight 1, learning rate 0.0001 (its best on the first block) '
                       '| 0.650 ± 0.050 | control | -0.150 [-0.200, -0.100] | 1.000 ± 0.000 | control | 2 |')
    assert rows[1] == ('| control, at its best on this block | intrinsic weight 1, learning rate 0.0003 (best of 2 on this block) '
                       '| 0.800 ± 0.000 | +0.150 [+0.100, +0.200] | control | 1.000 ± 0.000 | +0.000 [+0.000, +0.000] | 2 |')
    assert rows[2] == ('| idea b | intrinsic weight 4, learning rate 0.0001 (its best on this block: it has no complete unit on '
                       'the first block) | **0.950** ± 0.000 | **+0.300** [+0.250, +0.350] | **+0.150** [+0.150, +0.150] '
                       '| 1.000 ± 0.000 | +0.000 [+0.000, +0.000] | 2 |')
    assert rows[3] == ('| idea a | intrinsic weight 1, learning rate 0.0003 (its best on the first block) | <u>0.850</u> ± 0.050 '
                       '| <u>+0.200</u> [+0.200, +0.200] | <u>+0.050</u> [+0.000, +0.100] | 1.000 ± 0.000 | +0.000 [+0.000, +0.000] | 2 |')
    assert rows[4] == ('| idea c | intrinsic weight 2, learning rate 0.0003 (its best on this block: its first-block best, '
                       'intrinsic weight 2, learning rate 0.0001, was not run here) | 0.750 ± 0.000 | +0.100 [+0.050, +0.150] '
                       '| -0.050 [-0.050, -0.050] | 1.000 ± 0.000 | +0.000 [+0.000, +0.000] | 2 |')
    assert '$W$ ↑ ± standard error' in whole and '↓' not in whole

    # Second block, M in thousand steps: the control 0.708 and 0.408; idea a 0.308 (0.4 earlier twice under equal
    # rules; 0.2 earlier and level with the control favoured).
    rows = [line for line in discovery.split('Not complete, left out')[0].splitlines() if line.startswith('| ')][1:]
    assert rows[0].startswith('| control, at its first-block best | intrinsic weight 1, learning rate 0.0001 (its best on the first '
                              'block) | 0.708 ± 0.1 | control | +0.3 [+0.2, +0.4] | 2 of 2 |')
    assert rows[1].startswith('| control, at its best on this block | intrinsic weight 1, learning rate 0.0003 (best of 2 on this '
                              'block) | 0.408 ± 0 | -0.3 [-0.4, -0.2] | control |')
    assert [row.split(' | ')[0] for row in rows[2:]] == ['| idea b', '| idea a', '| idea c']
    assert '| **0.108** ± 0 | **-0.6** [-0.7, -0.5] | **-0.3** [-0.3, -0.3] | 2 of 2 | 2 |' in rows[2]
    assert '| <u>0.308</u> ± 0.1 | <u>-0.4</u> [-0.4, -0.4] | <u>-0.1</u> [-0.2, +0] | 2 of 2 | 2 |' in rows[3]
    assert '| 0.508 ± 0 | -0.2 [-0.3, -0.1] | +0.1 [+0.1, +0.1] | 2 of 2 | 2 |' in rows[4]
    # The unfinished unit is named and not read.
    assert 'Not complete, left out: idea d: 0 of 1 configurations read (0 of 1 units complete).' in second
    assert '| idea d | PointMaze Medium, reward 1 on arrival | 1 | unfinished | 10 of 20 |' in second

    # What the prediction modules read: the first-block entries and the second-block rows of arms, once every
    # planned configuration of the arms and of the control is read; else the words for what is waited for.
    cases = result['screen'].validation_cases
    assert list(cases) == [MEDIUM]
    entries, waiting = validation_seeds.first_entries(cases, MEDIUM, 'idea-a', 'idea-c')
    assert waiting is None and entries['idea-a']['result']['whole_curve_mean']['mean'] == pytest.approx(0.15)
    assert validation_seeds.first_entries(cases, MEDIUM, 'idea-b') == (
        None, 'Waiting for complete units in PointMaze Medium, reward 1 on arrival, first block: idea b (no unit).')
    rows, waiting = validation_seeds.second_rows(cases, MEDIUM, 'idea-a-short', 'idea-b', 'idea-c')
    row = rows['idea-a-short']
    assert row['how'] == 'first' and row['equal']['whole_curve_mean']['side'] == 'ahead' and row['favouring']['whole_curve_mean']['side'] == 'level'
    assert rows['idea-b']['how'] == 'none' and rows['idea-c']['how'] == 'absent' and waiting is None
    assert validation_seeds.second_rows(cases, MEDIUM, 'idea-a-short', 'idea-d') == (
        None, 'Waiting for complete units in PointMaze Medium, reward 1 on arrival, second block: idea d: 0 of 1 configurations '
              'read (0 of 1 units complete).')


def one_idea(tmp_path, name, control_first, control_second, complete=True):
    """A run folder with the control at one configuration per block and idea a at (3e-4, 1) on both (W 0.9; 0.9 and 0.8)."""
    run = tmp_path / name
    if control_first is not None:
        add_unit(run, 'validation-block', 'control', (control_first,), [4, 6], seeds=FIRST)
    add_unit(run, 'validation-block', 'idea-a', ((3e-4, 1.0),), [2, 2], seeds=FIRST)
    add_unit(run, 'validation-block-two', 'control', (control_second,), [4, 4], seeds=SECOND, complete=complete)
    add_unit(run, 'validation-block-two', 'idea-a-short', ((3e-4, 1.0),), [2, 4], seeds=SECOND)
    return run


def second_block_text(run, block=None):
    """The second block's text of the one case of a made-up run folder."""
    block = {**BLOCK, 'arms': ARMS[:2]} if block is None else block
    cases = validation_seeds.read_cases(run, block)
    return validation_seeds.section(block, cases).split('#### Second block: seeds that chose nothing\n')[1], cases


def test_the_control_without_its_first_block_best_on_the_second_block(tmp_path):
    """Equal rules need the control's first-block best on the second block; without it the cell says why."""
    # The control's first-block best (learning rate 1e-4) was not run on the second block, where it has 3e-4 only.
    text, cases = second_block_text(one_idea(tmp_path, 'absent', (1e-4, 1.0), (3e-4, 1.0)))
    whole = text.split('Second block, $W$ (the highest is the best):')[1].split('Second block, $M$')[0]
    missing = "the control's first-block best (learning rate 0.0001) has no complete unit on this block"
    assert f'| control, at its best on this block | learning rate 0.0003 (the only configuration read on this block) | <u>0.800</u> ± 0.000 | {missing} | control |' in whole
    assert (f'| idea a | learning rate 0.0003 (its best on the first block) | **0.850** ± 0.050 | {missing} | +0.050 [+0.000, +0.100] '
            f'| 1.000 ± 0.000 | {missing} | 2 |') in whole
    assert 'control, at its first-block best' not in whole
    row = cases[MEDIUM]['second']['metrics']['whole']['rows']['idea-a-short']
    assert row['equal'] is None and row['favouring']['whole_curve_mean']['mean'] == pytest.approx(0.05)
    assert validation_seeds.second_rows(cases, MEDIUM, 'idea-a-short') == (
        None, "No comparison under equal rules in PointMaze Medium, reward 1 on arrival: the control's first-block best has no "
              'complete unit on the second block.')

    # The control has no unit on the first block at all: said in the cell, and idea a's first-block entry is None.
    text, cases = second_block_text(one_idea(tmp_path, 'no-first-control', None, (3e-4, 1.0)))
    assert '| the control has no complete unit on the first block | +0.050 [+0.000, +0.100] |' in text
    assert validation_seeds.first_entries(cases, MEDIUM, 'idea-a') == (
        None, 'Waiting for complete units in PointMaze Medium, reward 1 on arrival, first block: control (no unit).')

    # The same configuration on both blocks: one control row, and both comparisons are one result.
    text, cases = second_block_text(one_idea(tmp_path, 'same', (3e-4, 1.0), (3e-4, 1.0)))
    row = cases[MEDIUM]['second']['metrics']['whole']['rows']['idea-a-short']
    assert row['equal'] is row['favouring']
    whole = text.split('Second block, $W$ (the highest is the best):')[1].split('Second block, $M$')[0]
    assert '| control | its only configuration (its best on the first block and on this block) | <u>0.800</u> ± 0.000 | control | control |' in whole
    assert '| idea a | its only configuration (its best on the first block) | **0.850** ± 0.050 | +0.050 [+0.000, +0.100] | +0.050 [+0.000, +0.100] |' in whole

    # The control's unit on the second block is not complete: nothing is compared, the units are still listed.
    text, cases = second_block_text(one_idea(tmp_path, 'unfinished', (1e-4, 1.0), (3e-4, 1.0), complete=False))
    assert 'The control has no complete unit on this block, so nothing is compared yet.' in text and 'Second block, $W$' not in text
    assert 'Not complete, left out: control: 0 of 1 configurations read (0 of 1 units complete).' in text
    assert validation_seeds.second_rows(cases, MEDIUM, 'idea-a-short') == (
        None, 'Waiting for complete units in PointMaze Medium, reward 1 on arrival, second block: control: 0 of 1 configurations '
              'read (0 of 1 units complete).')

    # A first block of another phase name has no unit: the case says so, and the arm stands at its best on the second block.
    block = {**BLOCK, 'first_block': 'no-such-phase', 'arms': ARMS[:2]}
    cases = validation_seeds.read_cases(one_idea(tmp_path, 'no-first-block', (1e-4, 1.0), (3e-4, 1.0)), block)
    text = validation_seeds.section(block, cases)
    assert 'The phase `no-such-phase` has no unit of these methods in this case.' in text
    assert cases[MEDIUM]['second']['metrics']['whole']['rows']['idea-a-short']['how'] == 'none'
    # A block without a second phase reads the first block only.
    cases = validation_seeds.read_cases(tmp_path / 'absent', {key: value for key, value in BLOCK.items() if key != 'second_block'})
    assert cases[MEDIUM]['second'] is None
    assert validation_seeds.second_rows(cases, MEDIUM, 'idea-a-short') == (
        None, 'Waiting for units of the second block in PointMaze Medium, reward 1 on arrival.')


def test_the_settings_block_is_checked():
    """Unknown keys, an arm that names no method, and not exactly one control are refused."""
    with pytest.raises(ValueError, match="validation_seeds: unknown keys \\['colour'\\], missing keys \\['cases'\\]"):
        validation_seeds.check_block({'first_block': 'p', 'arms': [], 'colour': 1})
    with pytest.raises(ValueError, match='the arm idea names no method on either block'):
        validation_seeds.check_block({'first_block': 'p', 'cases': [], 'arms': [{'name': 'idea'}]})
    with pytest.raises(ValueError, match="exactly one arm is the control; found \\[\\]"):
        validation_seeds.check_block({'first_block': 'p', 'cases': [], 'arms': [{'name': 'idea', 'first_block': 'idea'}]})
    with pytest.raises(ValueError, match="exactly one arm is the control; found \\['a', 'b'\\]"):
        validation_seeds.check_block({'first_block': 'p', 'cases': [], 'arms': [
            {'name': 'a', 'first_block': 'a', 'control': True}, {'name': 'b', 'first_block': 'b', 'control': True}]})
    with pytest.raises(ValueError, match="cases: unknown keys \\[\\], missing keys \\['reward_setting'\\]"):
        validation_seeds.check_block({'first_block': 'p', 'arms': ARMS[:1], 'cases': [{'environment': 'pointmaze-medium'}]})


def test_the_standing_page_on_a_made_up_run_folder(tmp_path):
    """The table at the top: one verdict per family and case, on the second block where the family was run there."""
    run = two_blocks(tmp_path)
    # The discovery check: the control arrives at steps 408 and 608, the fast last layer at step 208 twice.
    add_unit(run, 'discovery-validation', 'control', ((1e-4, 1.0),), [4, 6], seeds=FIRST)
    add_unit(run, 'discovery-validation', 'fast', ((1e-4, 4.0),), [2, 2], seeds=FIRST)
    # A second configuration of idea e on the first block has written half its windows.
    add_unit(run, 'validation-block', 'idea-e', ((3e-4, 8.0),), [0, 0], seeds=FIRST, complete=False, tag='_late')
    (run / 'analysis').mkdir()
    settings = {'title': 'Made-up standing', 'families': ['family a', 'family b', 'family c', 'family d', 'family e', 'family f'],
                'validation_seeds': {**BLOCK, 'arms': ARMS + [{'name': 'idea f', 'first_block': 'idea-f', 'family': 'family f'}],
                                     'cases': BLOCK['cases'] + [{'environment': 'pointmaze-large', 'reward_setting': 'step_cost'}]},
                'discovery': {'family': 'fast last layer', 'phase': 'discovery-validation', 'control': 'control',
                              'method_names': {'control': 'control', 'fast': 'fast last layer'}}}
    (run / 'analysis' / 'standing_settings.json').write_text(json.dumps(settings))
    result = standing.generate(run)
    text = result['page'].read_text()
    assert text.startswith('# Made-up standing\n\nGenerated ') and 'standing_job.py' in text
    top = text.split('## Standing in one table\n')[1].split('## Measures\n')[0]
    assert '| family of arms | PointMaze Medium, reward 1 on arrival | PointMaze Large, step cost |' in top
    # Families a, b and c rest on the second block; family e on the first; family d has an unfinished unit only;
    # family f has no unit; nothing was run in the second case.
    assert '| family a (whole-curve mean) | ahead: +0.2000 [+0.2000, +0.2000], second block under equal rules | not run |' in top
    assert '| family b (whole-curve mean) | ahead: +0.3000 [+0.2500, +0.3500], second block under equal rules | not run |' in top
    assert '| family c (whole-curve mean) | ahead: +0.1000 [+0.0500, +0.1500], second block under equal rules | not run |' in top
    assert '| family d (whole-curve mean) | no complete comparison yet | not run |' in top
    assert '| family e (whole-curve mean) | level: +0.0000 [+0.0000, +0.0000], first block; not final, units still running | not run |' in top
    assert ('- **family e, PointMaze Medium, reward 1 on arrival:** idea e. Not final: idea e: 1 of 2 configurations read '
            '(1 of 2 units complete).') in top
    assert '| family f (whole-curve mean) | not run | not run |' in top
    assert ('| fast last layer (mean capped first-goal time) | ahead: -0.3 [-0.4, -0.2] thousand steps, first block of the '
            'discovery check | not run |') in top
    assert ('- **family a, PointMaze Medium, reward 1 on arrival:** idea a; with the control at its best on the second block '
            'the difference is +0.0500 [+0.0000, +0.1000] (level).') in top
    assert '- **fast last layer, PointMaze Medium, reward 1 on arrival:** fast last layer.' in top
    # One section per case with both blocks, the discovery check under the case that has it.
    first, second = text.split('## PointMaze Medium, reward 1 on arrival\n')[1].split('## PointMaze Large, step cost\n')
    assert '### First block: each side at its best configuration there' in first and '### Second block: seeds that chose nothing' in first
    assert '### Discovery check: the leading configurations for the first-goal time' in first
    assert '| fast last layer | intrinsic weight 4 (the only configuration read) | **0.208** ± 0 | -0.3 [-0.4, -0.2] | 2 of 2 |' in first
    assert 'The phase `validation-block` has no unit of these methods in this case.' in second and 'Discovery check' not in second
    assert list(result['sections'])[:2] == ['standing-in-one-table', 'measures'] or len(result['sections']) == 4

    # A family the arms name must be listed, and unknown keys are refused.
    (run / 'analysis' / 'standing_settings.json').write_text(json.dumps({**settings, 'families': ['family a']}))
    with pytest.raises(ValueError, match="which `families` does not list"):
        standing.generate(run)
    (run / 'analysis' / 'standing_settings.json').write_text(json.dumps({**settings, 'colour': 1}))
    with pytest.raises(ValueError, match="unknown keys \\['colour'\\], missing keys \\[\\]"):
        standing.generate(run)


# ---- the readings of batches 5 to 7 --------------------------------------------------------------------------------

def screen_of(run, block):
    """A batch without a screen whose validation seeds are read from a made-up run folder."""
    return batch_summary.Screen(batch='made-up', control=None, order=(), names={}, units=(), methods={}, differences={},
                                discovery_differences={}, budget=None, seeds=(),
                                validation_cases=validation_seeds.read_cases(run, block))


def readings(module, screen):
    """Per prediction of a module: the verdict of each part and the overall word."""
    rows = [batch_summary.evaluate(prediction['parts'], screen) for prediction in module.PREDICTIONS]
    return [[verdict for _, verdict, _ in found] for found in rows], [batch_summary.overall(found) for found in rows], rows


def arm(slug, second=None, control=False):
    """One arm of a settings block, named by its method."""
    return {'name': slug.replace('-', ' '), 'first_block': slug, **({'second_block': second} if second else {}),
            **({'control': True} if control else {})}


def test_batch_5_predictions_on_made_up_units(tmp_path, monkeypatch):
    """The held divisor against the control, the earlier hold over a stretch of the run, and the curve between two others."""
    module = predictions_batch_05
    # The stretch of prediction 2, scaled to the made-up run of 2,000 steps: steps 300 to 900, windows 3 to 8.
    monkeypatch.setattr(module, 'STRETCH', (300, 900))
    run = tmp_path / 'batch-5'
    one = lambda slug, goal, weight: add_unit(run, 'validation-block', slug, ((1e-4, weight),), goal, seeds=FIRST)
    one(module.CONTROL, [4, 6], 1.0)
    # Held after 1/16: the control's numbers (W level, L not ahead). Over windows 3 to 8 it has 5/6 and 1/2; the hold
    # after 1/64 has 1 twice: +1/6 and +1/2.
    one(module.HELD_SIXTEENTH, [4, 6], 2.0)
    one(module.HELD_SIXTY_FOURTH, [2, 2], 2.0)
    # The slower predictor: held after 1/16 has 1 and 0 over the stretch, held after 1/64 has 1 twice and W 0.95.
    one(module.SLOWER_SIXTEENTH, [3, 9], 2.0)
    one(module.SLOWER_SIXTY_FOURTH, [1, 1], 2.0)
    block = {'first_block': 'validation-block', 'cases': [dict(zip(('environment', 'reward_setting'), module.CASE))],
             'arms': [arm(module.CONTROL, control=True)] + [arm(slug) for slug in (
                 module.HELD_SIXTEENTH, module.HELD_SIXTY_FOURTH, module.SLOWER_SIXTEENTH, module.SLOWER_SIXTY_FOURTH, module.HALF_SPEED)]}
    # Before the half-speed arm has a unit, prediction 3 cannot be read and names the arm.
    verdicts, words, rows = readings(module, screen_of(run, block))
    assert verdicts == [[True, False], [True, True, True], [None]] and words == ['did not hold', 'held', 'cannot be read yet']
    assert rows[2][0][2] == ('Waiting for complete units in PointMaze Medium, reward 1 on arrival, first block: predictor half '
                            'speed held divisor (no unit).')
    assert rows[0][0][2] == '$W$ 0.7500, difference from the control +0.0000 [+0.0000, +0.0000].'
    assert rows[1][0][2] == ('Mean return over steps 300 to 900, held after 1/64 minus held after 1/16, on 2 seeds: '
                            '+0.3333 [+0.1667, +0.5000]; the interval stands entirely above zero.')
    assert '+0.5000 [+0.0000, +1.0000]; the interval stands across zero.' in rows[1][1][2] and '$W$ 0.9500, difference from the control +0.2000 [+0.1500, +0.2500].' == rows[1][2][2]
    # The half-speed arm arrives from windows 3 and 5; the control from 4 and 6; the finalist from 3 and 9 (W 0.85 and
    # 0.55). In window 5
    # the arm's mean return is 1 and both others' 0.5; in the 19 other windows it lies between them.
    one(module.HALF_SPEED, [3, 5], 2.0)
    verdicts, words, rows = readings(module, screen_of(run, block))
    assert verdicts[2] == [False] and words[2] == 'did not hold'
    assert rows[2][0][2].startswith('Of 20 logging windows the arm is between the two in 19, above both in 1 and below both in 0. '
                                    'Whole-curve means: the arm 0.8000, the control 0.7500, the slower predictor with held divisor 0.7000.')
    assert len(module.PREDICTIONS) == 3 and 'READING' in dir(module)


def test_batch_6_predictions_on_made_up_units(tmp_path):
    """The second block under equal rules, the half-speed arm beside the slower predictor, and a part that waits."""
    module = predictions_batch_06
    run = tmp_path / 'batch-6'
    one = lambda phase, slug, goal, weight, seeds=FIRST, **more: add_unit(run, phase, slug, ((1e-4, weight),), goal, seeds=seeds, **more)
    large, cost = {'environment': 'pointmaze-large'}, {'environment': 'pointmaze-large', 'reward': 'step_cost'}
    # PointMaze Medium. First block: the control 0.8 and 0.7, the arm 0.9 twice (+0.15). Second block: the control
    # 0.7 and 0.6, the arm 0.75 and 0.6 (+0.05 and 0: above, by less, the interval reaches zero).
    one('validation-block', 'control-cleanrl-ppo-rnd', [4, 6], 1.0)
    one('validation-block', module.HALF_SPEED, [2, 2], 2.0)
    one('validation-block-two', 'control-cleanrl-ppo-rnd', [6, 8], 1.0, SECOND)
    one('validation-block-two', module.HALF_SPEED_SECOND, [5, 8], 2.0, SECOND)
    # PointMaze Large, reward 1 on arrival: the control 0.5, the arm 0.75, the slower predictor 0.8 (0.05 apart).
    one('validation-block', 'control-cleanrl-ppo-rnd', [10, 10], 1.0, **large)
    one('validation-block', module.HALF_SPEED, [4, 6], 2.0, **large)
    one('validation-block', module.SLOWER, [4, 4], 4.0, **large)
    # PointMaze Large, step cost: the slower predictor has the control's numbers; all three parts is not complete.
    one('validation-block', 'control-cleanrl-ppo-rnd', [10, 10], 1.0, **cost)
    one('validation-block', module.SLOWER, [10, 10], 4.0, **cost)
    one('validation-block', module.THREE_PARTS, [0, 0], 4.0, complete=False, **cost)
    block = {'first_block': 'validation-block', 'second_block': 'validation-block-two',
             'cases': [dict(zip(('environment', 'reward_setting'), case))
                       for case in (module.MEDIUM_ARRIVAL, module.LARGE_ARRIVAL, module.LARGE_STEP_COST)],
             'arms': [arm('control-cleanrl-ppo-rnd', 'control-cleanrl-ppo-rnd', control=True),
                      arm(module.HALF_SPEED, module.HALF_SPEED_SECOND), arm(module.SLOWER), arm(module.THREE_PARTS)]}
    verdicts, words, rows = readings(module, screen_of(run, block))
    assert verdicts == [[True, True, True], [False, True], [None, True]]
    assert words == ['held', 'did not hold', 'cannot be read yet']
    assert rows[0][0][2] == '$W$ 0.6750 on the second block, difference under equal rules +0.0250 [+0.0000, +0.0500].'
    assert rows[0][1][2] == 'First block +0.1500; second block +0.0250.'
    assert rows[1][0][2] == ('Best $W$ on the first block: the arm 0.7500, the slower predictor with held divisor 0.8000 '
                            '(apart by 0.0500).')
    assert rows[2][0][2] == ('Waiting for complete units in PointMaze Large, step cost, first block: three parts held one '
                            'sixteenth: 0 of 1 configurations read (0 of 1 units complete).')
    # Without the control on the second block the three parts of prediction 1 that need it wait.
    block = {**block, 'arms': [arm('control-cleanrl-ppo-rnd', control=True)] + block['arms'][1:]}
    verdicts, _, rows = readings(module, screen_of(run, block))
    assert verdicts[0] == [None, None, None]
    assert rows[0][0][2] == 'Waiting for units of the second block in PointMaze Medium, reward 1 on arrival.'
    assert len(module.PREDICTIONS) == 3


def test_batch_7_predictions_on_made_up_units(tmp_path):
    """The held divisor in the two step-cost cases: ahead in one, level in the other, behind at one named weight."""
    module = predictions_batch_07
    run = tmp_path / 'batch-7'
    medium, large = {'reward': 'step_cost'}, {'environment': 'pointmaze-large', 'reward': 'step_cost'}
    block = {'first_block': 'validation-block',
             'cases': [dict(zip(('environment', 'reward_setting'), case)) for case in (module.MEDIUM_STEP_COST, module.LARGE_STEP_COST)],
             'arms': [arm('control-cleanrl-ppo-rnd', control=True), arm(module.HELD)]}
    # PointMaze Medium: the held divisor at weight 2 has the control's numbers; at weight 0.03125 one seed arrives later.
    add_unit(run, 'validation-block', 'control-cleanrl-ppo-rnd', ((0.006, 0.0078125),), [4, 6], seeds=FIRST, **medium)
    add_unit(run, 'validation-block', module.HELD, ((0.006, 2.0),), [4, 6], seeds=FIRST, tag='_first', **medium)
    # Before the named weight and the large case have units, their parts wait.
    verdicts, words, rows = readings(module, screen_of(run, block))
    assert verdicts == [[None], [True, None]] and words == ['cannot be read yet', 'cannot be read yet']
    assert rows[1][1][2] == 'Waiting for the configuration at learning rate 0.006 and intrinsic weight 0.03125.'
    add_unit(run, 'validation-block', module.HELD, ((0.006, 0.03125),), [6, 6], seeds=FIRST, tag='_second', **medium)
    # PointMaze Large: the held divisor arrives from windows 4 and 6, the control from window 10.
    add_unit(run, 'validation-block', 'control-cleanrl-ppo-rnd', ((0.006, 1.0),), [10, 10], seeds=FIRST, **large)
    add_unit(run, 'validation-block', module.HELD, ((0.006, 4.0),), [4, 6], seeds=FIRST, **large)
    screen = screen_of(run, block)
    verdicts, words, rows = readings(module, screen)
    assert verdicts == [[True], [True, True]] and words == ['held', 'held']
    assert rows[1][0][2].endswith('difference from the control +0.0000 [+0.0000, +0.0000]; the interval stands across zero.')
    assert rows[1][1][2].startswith('Difference from the control at its best configuration -0.0') and 'at its best' in rows[1][1][2]
    assert module.best_weight(screen) == "The held divisor's best intrinsic weight on the first block is 2; the control's is 0.0078125."
    assert len(module.PREDICTIONS) == 2
    # While another configuration of the held divisor is still running in the large case, its part waits: the best
    # configuration could still change. The tables read the complete unit and name the other.
    add_unit(run, 'validation-block', module.HELD, ((0.006, 8.0),), [0, 0], seeds=FIRST, complete=False, tag='_late', **large)
    verdicts, words, rows = readings(module, screen_of(run, block))
    assert verdicts == [[None], [True, True]] and words == ['cannot be read yet', 'held']
    assert rows[0][0][2] == ('Waiting for complete units in PointMaze Large, step cost, first block: divisor held after one '
                            'sixteenth: 1 of 2 configurations read (1 of 2 units complete).')
