"""Tests of a screening batch's summary: small helpers, the batch-1 predictions, and a made-up screen end to end."""

import json

import pytest

from maze_record_examples import flat_policy, goal_windows, grid_records, start_record, write_shard
from maze_campaign.analysis import batch_summary, compare, figures, predictions_batch_01, records

BATCH = '01_made-up-batch'
CONTROL, EVEN, FIXED = 'control-cleanrl-ppo-rnd', 'even-start', 'fixed-scale'
ERROR_LENGTH, FAST = 'error-length-as-the-bonus', 'fast-last-layer-inside-the-rollout-step-0-point-1'
# Every method the batch-1 predictions name, so that a made-up screen can be read by them.
METHODS = [CONTROL, EVEN, FIXED, predictions_batch_01.NO_DIVISOR, *predictions_batch_01.PREDICTOR.values(),
           ERROR_LENGTH, *predictions_batch_01.FAST.values(), *predictions_batch_01.SCHEDULED.values()]
IDEAS = ('# Made-up batch\n\n## Predictions (written before the runs)\n\n'
         '1. **Even start alone** changes the whole-curve mean by less than its interval; its start ratio is 1.00\n'
         '   against well above 2 for the control.\n'
         '2. **Fixed scale** raises the final level.\n3. **Separate predictor step size** text.\n'
         '4. **Error length** text.\n5. **Fast last layer** text.\n6. **Scheduled last layer** at $a$ = 1.\n\n'
         '## What would make an idea new\n\n1. Not a prediction.\n')


def unit_name(method):
    """The made-up unit of a method."""
    return f'batch-01-screen_pointmaze-large_{method}_cells-1-to-3-of-3_seeds-0-to-1'


def screen_unit(method, first_goal_windows, start_bonus, bonus_of, visits_of, divisor=1.0, trainer=None):
    """A unit of 3 intrinsic weights (1, 16, 64) x 2 seeds and 20 windows of 100 steps, with the maze-grid fields."""
    start = start_record(unit_name(method), iterations=20, cells=((1e-4, 1.0), (1e-4, 16.0), (1e-4, 64.0)),
                         probe_cells=[(1, 1), (1, 2)], trainer=trainer)
    return grid_records(start, goal_windows(start, first_goal_windows), start_bonus, bonus_of, visits_of, divisor)


def made_up_screen(tmp_path):
    """A run folder with a batch of 14 methods: three complete units, one unfinished, one not started, nine not built."""
    run = tmp_path / 'screen-run'
    folder = run / 'batches' / BATCH
    folder.mkdir(parents=True)
    (run / 'specifications').mkdir()
    (run / 'queue/running').mkdir(parents=True)
    for method in (CONTROL, EVEN, FIXED, ERROR_LENGTH, FAST):
        (run / 'specifications' / f'{unit_name(method)}.json').write_text(json.dumps(
            {'unit_id': unit_name(method), 'phase': 'batch-01-screen', 'method': method}))
    (run / 'queue/running' / f'{unit_name(ERROR_LENGTH)}.json').write_text('{}')
    (folder / 'ideas.md').write_text(IDEAS)
    (folder / 'screen.json').write_text(json.dumps({
        'phase': 'batch-01-screen', 'steps_per_policy': 2000, 'seeds': {'first': 0, 'count': 2},
        'sweeps': [{'method': method} for method in METHODS]}))
    (folder / 'summary_settings.json').write_text(json.dumps({
        'title': 'Made-up screen', 'study_description': 'screen.json', 'control': CONTROL,
        'method_names': {method: method.replace('-', ' ') for method in METHODS} | {CONTROL: 'control'},
        'consistency_check': {'run_folder': 'baseline-run', 'unit': 'development unit',
                              'settings': {'learning_rate': 1e-4, 'int_coef': 1.0}, 'reported': 'W 0.725'},
        'predictions': 'predictions_batch_01'}))

    # The control: the bonus at the visited cell 0 is 8 / (1 + visits), so its slope is -1; cell 1 is never visited
    # and its bonus is half its start. Copies are (weight 1: seeds 0, 1), (weight 16: ...), (weight 64: ...).
    falling = lambda index: [8.0 / (index + 2), 1.0]
    write_shard(run, screen_unit(CONTROL, [4, 6, 10, 10, None, None], [8.0, 2.0], falling, lambda index: [1, 0],
                                 divisor=2.0, trainer={'permutation_draw': 'x'}))
    # Even start: the same start at both cells, slope -0.5 at cell 0, and cell 1 keeps its start until its one
    # visit in the last window.
    write_shard(run, screen_unit(EVEN, [10, 12, 2, 2, 10, 10], [0.5, 0.5],
                                 lambda index: [0.5 * (index + 2) ** -0.5, 0.5], lambda index: [1, int(index == 19)]))
    # Fixed scale was started again from zero after one window; only its second attempt counts.
    first = screen_unit(FIXED, [None] * 6, [8.0, 2.0], falling, lambda index: [1, 0])[:3]
    second = screen_unit(FIXED, [None, None, None, 15, None, 12], [8.0, 2.0], falling, lambda index: [1, 0])
    write_shard(run, first + second)
    # Error length has written 10 of its 20 windows; the fast last layer has no shard.
    write_shard(run, screen_unit(ERROR_LENGTH, [0] * 6, [8.0, 2.0], falling, lambda index: [1, 0])[:12], complete=False)

    # The baseline run's unit for the consistency check: the control's cell at weight 1 on the same two seeds.
    start = start_record('development unit', iterations=20)
    write_shard(tmp_path / 'baseline-run', [start] + goal_windows(start, [4, 7]))
    return run


def test_a_made_up_screen_end_to_end(tmp_path, monkeypatch):
    """Complete units enter the tables, the others are listed and left out; every number is computed by hand."""
    monkeypatch.setattr(batch_summary, 'FEWEST_PAIRS', 2)
    run = made_up_screen(tmp_path)
    result = batch_summary.generate(run, BATCH)
    folder = run / 'batches' / BATCH
    text = (folder / 'summary.md').read_text()
    assert result['summary'] == folder / 'summary.md' and (folder / 'summary.html').exists()
    assert set(result['sections'].values()) == {'unread'} and (folder / 'section_history.json').exists()
    assert text.startswith('# Made-up screen\n\nGenerated ') and ' PT by the command below.' in text
    assert f'--run-dir=runs/screen-run \\\n    --batch={BATCH}\n```' in text

    # Units: three complete; the unfinished one and the ones with no shard or no unit are left out of every table.
    units = text.split('## Units\n')[1].split('\n## ')[0]
    assert '3 of 14 units are complete.' in units and 'Left out: fixed scale' not in units
    assert 'Left out: even start with no divisor; ' in units
    assert '| 8 | error length as the bonus | unfinished | 10 of 20 | not recorded | not finished | `queue/running` |' in units
    assert ('| 9 | fast last layer inside the rollout step 0 point 1 | not started (no shard) | 0 | none | not finished '
            '| in no queue folder |') in units
    assert '| 4 | even start with no divisor | no unit was built | 0 | none | not finished | no unit |' in units
    assert set(result['screen'].methods) == {CONTROL, EVEN, FIXED}
    tables = text.split('## Screen table\n')[1].split('## Properties')[0]
    assert 'error length' not in tables and 'fast last layer' not in tables

    # Screen table. Control: W 0.8 and 0.7 at weight 1. Even start: W 0.9 twice at weight 16, so its differences
    # are 0.1 and 0.2. Fixed scale: W 0 and 0.4 at weight 64, L 0 and 1, one policy of two arrives.
    rows = [line for line in tables.split('\n## ')[0].splitlines() if line.startswith('| ')][1:]
    assert rows == [
        '| control | 1 | <u>0.750</u> ± 0.050 | control | **1.000** | control | <u>508</u> | **2 of 2** |',
        '| even start | 16 | **0.900** ± 0.000 | **+0.150** [+0.100, +0.200] | **1.000** | **+0.000** [+0.000, +0.000] '
        '| **208** | **2 of 2** |',
        '| fixed scale | 64 | 0.200 ± 0.200 | <u>-0.550</u> [-0.800, -0.300] | <u>0.500</u> | <u>-0.500</u> [-1.000, +0.000] '
        '| none | <u>1 of 2</u> |']
    assert 'best intrinsic weight (1), on the 2 seeds both have' in tables and 'a screen and not a verdict' in tables

    # W at every weight, in two tables; the best weight's place in the grid is named.
    weights = tables.split('## Whole-curve mean at every intrinsic weight\n')[1].split('\n## ')[0]
    assert '| method | weight 1 | weight 16 | best intrinsic weight |' in weights
    assert '| control | **0.750** | <u>0.500</u> | 1 (smallest of the grid) |' in weights
    assert '| even start | <u>0.450</u> | **0.900** | 16 |' in weights
    assert '| fixed scale | 0.000 | 0.125 | 64 (largest of the grid) |' in weights
    assert '| method | weight 64 | best intrinsic weight |' in weights and '| even start | **0.500** | 16 |' in weights
    assert 'Best weight at an end of the grid: control (1, the smallest); fixed scale (64, the largest).' in weights

    # Properties: the control's start bonus is 8 and 2 (ratio 4) and falls as 1 / (1 + visits) at the visited cell;
    # its unvisited cell holds half its start. Even start: ratio 1, slope -0.5, the unvisited cell keeps its
    # start until the last window, where no pair is left without a visit.
    properties = text.split('## Properties of the bonus on the maze grid\n')[1].split('\n## ')[0]
    assert '5% is read at step 100, 25% is read at step 500, 100% is read at step 2,000' in properties
    assert '| control | 4.00 | -1.00 | -1.00 (1) to -1.00 (64) | 0.5 (2 pairs) | 0.5 (2 pairs) | 0.5 (2 pairs) |' in properties
    assert '| even start | 1.00 | -0.50 | -0.50 (1) to -0.50 (64) | 1 (2 pairs) | 1 (2 pairs) | no pair without a visit |' in properties
    # Weighted intrinsic reward of the control (weight 1, divisor 2): start (8 + 2) / 2 / 2 = 2.5; at window 5,
    # (8 / 6 + 1) / 2 / 2 = 0.583; at window 20, (8 / 21 + 1) / 2 / 2 = 0.345. Even start: 16 x 0.5 = 8 at the start.
    # After the first logging window the control's reward is (8 / 2 + 1) / 2 / 2 = 1.25; its divisor is 2 throughout.
    assert '| control | 1 | 2 and 2 | 2.5 | 1.25 | 0.583 | 0.345 |' in properties
    assert '| even start | 16 | 1 and 1 | 8 | ' in properties and 'first logging window (step 100)' in properties
    for name in ('training-curves_control-against-each-idea_best-intrinsic-weights',
                 'bonus-at-the-probes-against-cumulative-visits_best-intrinsic-weights'):
        assert (folder / 'plots' / f'{name}.png').exists() and (folder / 'plots' / f'{name}.pdf').exists()
        assert f'](plots/{name}.png)' in text

    # Predictions: number 1 has one part that did not hold; number 2 waits for a unit but already did not hold;
    # the others wait for units.
    predictions = text.split('## Predictions\n')[1].split('\n## ')[0]
    first = predictions.split('### Prediction 1: even start\n')[1].split('###')[0]
    assert ('> **Even start alone** changes the whole-curve mean by less than its interval; its start ratio is 1.00 '
            'against well above 2 for the control.\n\nObserved: **did not hold**.') in first
    assert ('- The interval of the difference in $W$ from the control contains zero: **did not hold**. Difference in '
            '$W$ from the control: +0.150 [+0.100, +0.200].') in first
    assert '- The start ratio of even start is 1.00: **held**. Median start ratio: 1.00.' in first
    assert '- The start ratio of the control is above 2: **held**. Median start ratio: 4.00.' in first
    second = predictions.split('### Prediction 2: fixed scale, and even start with no divisor\n')[1].split('###')[0]
    assert 'Observed: **did not hold**.' in second
    assert ('- Fixed scale: the final level at the large weights is above the control\'s: **did not hold**. Final level '
            'at weight 16: 0.500 against 1.000 for the control; at weight 64: 0.500 against 0.000 for the control.') in second
    assert '- Fixed scale: the best intrinsic weight is above the control\'s: **held**. Best intrinsic weight 64 against 1' in second
    assert '**cannot be read yet**. Waiting for: even start with no divisor.' in second
    assert predictions.count('Observed: **cannot be read yet**.') == 4

    # Sides, by the interval of the difference in W.
    sides = text.split('## Ideas against the control\n')[1].split('\n## ')[0]
    assert '- **Ahead of the control** (interval entirely above zero): even start (+0.150 [+0.100, +0.200]).' in sides
    assert '- **Within the interval** (it contains zero): none.' in sides
    assert '- **Behind the control** (interval entirely below zero): fixed scale (-0.550 [-0.800, -0.300]).' in sides
    assert '- **Not compared yet** (unit not complete): even start with no divisor; ' in sides

    # Consistency check: the baseline unit's W is 0.8 and 0.65, so the control is 0 and 0.05 above it.
    check = text.split('## Consistency check against the baseline run\n')[1].split('\n## ')[0]
    assert '| control of this batch (control) | 0.750 ± 0.050 | 1.000 ± 0.000 | 2 | not recorded |' in check
    assert "| the baseline run's unit | 0.725 ± 0.075 | 1.000 ± 0.000 | 2 | not recorded |" in check
    assert 'in $W$: +0.025 [+0.000, +0.050]; the interval contains zero. in $L$: +0.000 [+0.000, +0.000]' in check
    assert 'Settings that only the control records' in check and '`permutation_draw` = x' in check
    assert 'intrinsic weight 1, learning rate 0.0001' in check and 'Reported earlier for that cell: W 0.725.' in check

    # Records: the unit that was started again is named; no made-up bonus rises above its start.
    look = text.split('## Records that need a look\n')[1]
    assert f'- fixed scale (`{unit_name(FIXED)}`): started again from zero 1 time(s)' in look
    assert 'control' not in look.split('names it.')[1]


def test_no_complete_unit_gives_a_summary_that_only_lists_the_units(tmp_path):
    """With every unit unfinished the tables and figures are absent and nothing is estimated."""
    run = made_up_screen(tmp_path)
    for method in (CONTROL, EVEN, FIXED):
        shard = run / 'data' / f'{unit_name(method)}.jsonl'
        shard.write_text(''.join(line + '\n' for line in shard.read_text().splitlines()[:-1]))
    batch_summary.generate(run, BATCH)
    text = (run / 'batches' / BATCH / 'summary.md').read_text()
    assert '0 of 14 units are complete.' in text and '## Screen table' not in text and '![' not in text
    assert "The control's unit is not complete, so the check cannot be made yet." in text
    assert text.count('**cannot be read yet**') >= 6 and 'did not hold' not in text.split('## Predictions\n')[1].split('### ')[1]


def test_the_best_weight_is_chosen_by_w_and_a_tie_goes_to_the_higher_final_level():
    """Weights 1 and 4 both have W 0.5; weight 4 ends at 1 and weight 1 at 0, so weight 4 is the best."""
    groups = records.configurations(
        [flat_policy([1, 1, 0, 0], seed=seed, settings=(('int_coef', 1.0),)) for seed in (0, 1)]
        + [flat_policy([0, 0, 1, 1], seed=seed, settings=(('int_coef', 4.0),)) for seed in (0, 1)]
        + [flat_policy([0, 0, 0, 1], seed=seed, settings=(('int_coef', 16.0),)) for seed in (0, 1)])
    scores, best = batch_summary.score_weights(groups, 400)
    assert scores == {1.0: (0.5, 0.0), 4.0: (0.5, 1.0), 16.0: (0.25, 1.0)}
    assert batch_summary.weight_of(best) == 4.0


def test_cells_and_words():
    """Weights, step counts, small numbers, differences and the unvisited-cell readings as table cells."""
    assert [batch_summary.weight_text(weight) for weight in (2.0, 0.0625, 0.015625, 4.0 ** -6, 64.0)] == [
        '2', '0.0625', '$4^{-3}$', '$4^{-6}$', '64']
    assert batch_summary.steps_text(float('inf')) == 'none' and batch_summary.steps_text(421300.0) == '421.3K'
    assert [batch_summary.short(value) for value in (1.0234, 70.0, 0.00096012, 0.0000123, 0.0)] == [
        '1.02', '70', '0.00096', '0.0000123', '0']
    result = {'whole_curve_mean': {'mean': 0.021, 'interval': [0.004, 0.038]}}
    assert batch_summary.difference_cell(result, 'whole_curve_mean', 'best') == '**+0.021** [+0.004, +0.038]'
    assert batch_summary.slope_range({1.0: -0.4, 4.0: -0.9, 16.0: None}) == '-0.90 (4) to -0.40 (1)'
    assert batch_summary.slope_range({1.0: None}) == 'no slope'
    assert batch_summary.unvisited_cell({'pairs': 0, 'median': None}) == 'no pair without a visit'
    assert batch_summary.unvisited_cell({'pairs': 3, 'median': None}) == 'not printed: 3 pairs'
    assert batch_summary.unvisited_cell({'pairs': 412, 'median': 0.1234}) == '0.123 (412 pairs)'
    assert batch_summary.finite_median([1.0, float('nan'), 3.0]) == 2.0 and batch_summary.finite_median([float('nan')]) is None
    assert figures.wrapped('scheduled last layer on fixed features, decay 0.75, intrinsic weight 2') == (
        'scheduled last layer on fixed\nfeatures, decay 0.75, intrinsic\nweight 2')


def test_written_predictions_are_read_from_the_ideas_file():
    """The numbered items of the Predictions section, continuation lines joined; other numbered lists are left out."""
    written = batch_summary.written_predictions(IDEAS)
    assert len(written) == 6 and written[5] == '**Scheduled last layer** at $a$ = 1.'
    assert written[0].endswith('its start ratio is 1.00 against well above 2 for the control.')
    with pytest.raises(ValueError, match='no single section'):
        batch_summary.written_predictions('# No such section\n')


def test_a_predictions_word_follows_from_its_parts():
    """Held when all parts held, did not hold when one did not, else cannot be read yet."""
    assert batch_summary.overall([('a', True, ''), ('b', True, '')]) == 'held'
    assert batch_summary.overall([('a', True, ''), ('b', None, ''), ('c', False, '')]) == 'did not hold'
    assert batch_summary.overall([('a', True, ''), ('b', None, '')]) == 'cannot be read yet'


def test_units_of_a_batch_come_from_the_specifications(tmp_path):
    """A method without a specification has no unit; a method split over two units is refused."""
    (tmp_path / 'specifications').mkdir()
    for unit, phase, method in (('u1', 'batch-01-screen', 'control'), ('u2', 'batch-01-screen', 'even-start'),
                                ('u3', 'another-phase', 'fixed-scale')):
        (tmp_path / 'specifications' / f'{unit}.json').write_text(json.dumps({'unit_id': unit, 'phase': phase, 'method': method}))
    description = {'phase': 'batch-01-screen', 'sweeps': [{'method': name} for name in ('control', 'even-start', 'fixed-scale')]}
    assert batch_summary.batch_units(tmp_path, description) == {'control': 'u1', 'even-start': 'u2', 'fixed-scale': None}
    assert batch_summary.queue_place(tmp_path, 'u1') == 'in no queue folder'
    for name in ('done', 'running'):
        (tmp_path / 'queue' / name).mkdir(parents=True)
    (tmp_path / 'queue/done/u1.json').write_text('{}')
    assert batch_summary.queue_place(tmp_path, 'u1') == '`queue/done`'
    (tmp_path / 'specifications/u4.json').write_text(json.dumps({'unit_id': 'u4', 'phase': 'batch-01-screen', 'method': 'control'}))
    with pytest.raises(ValueError, match="split over the units \\['u1', 'u4'\\]"):
        batch_summary.batch_units(tmp_path, description)


def test_settings_of_a_batch_and_of_two_configurations(tmp_path):
    """An unknown or missing key of summary_settings.json is refused; two configurations are compared setting by setting."""
    (tmp_path / 'summary_settings.json').write_text(json.dumps({'title': 'T', 'control': 'c', 'colour': 'blue'}))
    with pytest.raises(ValueError, match="unknown keys \\['colour'\\], missing keys \\['method_names', 'study_description'\\]"):
        batch_summary.read_settings(tmp_path)
    ours, theirs = records.configurations(
        [flat_policy([1], seed=0, settings=(('clip_coef', 0.1), ('int_coef', 2.0), ('permutation_draw', 'x'))),
         flat_policy([1], seed=0, settings=(('clip_coef', 0.2), ('int_coef', 2.0)))])
    assert batch_summary.setting_differences(ours, theirs) == {
        'different': {'clip_coef': (0.1, 0.2)}, 'only_first': {'permutation_draw': 'x'}, 'only_second': {}}
    assert batch_summary.matching([ours, theirs], {'clip_coef': 0.2}, 'two rows') is theirs
    with pytest.raises(ValueError, match='2 configurations of two rows have the settings'):
        batch_summary.matching([ours, theirs], {'int_coef': 2.0}, 'two rows')


def fake_screen(values, differences=None):
    """A screen in memory: method -> (W, L, median first-arrival step, decay slope), each at a best weight of 2."""
    methods = {}
    for slug, (whole, level, arrival, slope) in values.items():
        best = records.Configuration('PointMaze Large', 'arrival_plus_one', slug, 4, 100, (('int_coef', 2.0),), ())
        unvisited = {share: {'pairs': 40, 'median': 0.5, 'step': 0} for share in batch_summary.UNVISITED_SHARES}
        methods[slug] = batch_summary.Method(
            slug=slug, name=slug, unit_id=slug, budget=100, groups=(best,), scores={}, best=best,
            measures={'whole_curve_mean': whole, 'final_level': level, 'median_first_goal_step': arrival},
            grid={'start_ratio': 3.0, 'decay_slope': slope, 'unvisited': unvisited})
    return batch_summary.Screen(batch='made up', control=CONTROL, order=tuple(METHODS), names={slug: slug for slug in METHODS},
                                units=(), methods=methods, differences=differences or {}, budget=100, seeds=(0, 1))


def difference(mean, low, high):
    """A paired difference as compare.compare_with_baseline gives it, the same for W and for L."""
    one = {'mean': mean, 'interval': [low, high], 'side': compare.side([low, high])}
    return {'whole_curve_mean': one, 'final_level': one, 'seeds': 16}


def verdicts(number, screen):
    """The verdict of each part of one batch-1 prediction."""
    return [verdict for _, verdict, _ in batch_summary.evaluate(predictions_batch_01.PREDICTIONS[number - 1]['parts'], screen)]


def test_batch_1_predictions_3_to_6_on_made_up_numbers():
    """Each part is read from the numbers it names; a part whose unit is missing cannot be read yet."""
    predictor, fast, scheduled = (predictions_batch_01.PREDICTOR, predictions_batch_01.FAST, predictions_batch_01.SCHEDULED)
    inf = float('inf')
    screen = fake_screen({
        CONTROL: (0.80, 0.99, 400e3, -0.6),
        # 4 times: earlier and lower than 1 times; 0.25 times: later but also lower; the best W is at 4 times.
        predictor[0.25]: (0.70, 0.90, 600e3, -0.4), predictor[1]: (0.75, 0.95, 500e3, -0.5), predictor[4]: (0.78, 0.92, 300e3, -0.7),
        ERROR_LENGTH: (0.79, 0.99, 450e3, -0.3),
        # The step size with the highest W (0.3) arrives at 320 thousand steps, 20% below the control's 400.
        fast[0.1]: (0.70, 0.99, 100e3, -0.6), fast[0.3]: (0.85, 0.99, 320e3, -0.6), fast[1]: (0.60, 0.99, inf, -0.6),
        scheduled[0.5]: (0.60, 0.97, 200e3, -0.5), scheduled[0.75]: (0.60, 0.99, 150e3, -0.75)},
        differences={ERROR_LENGTH: difference(-0.01, -0.03, 0.01), scheduled[0.5]: difference(-0.2, -0.3, -0.1),
                     scheduled[0.75]: difference(-0.2, -0.3, -0.1)})
    assert verdicts(3, screen) == [True, True, True, False, True]
    assert verdicts(4, screen) == [True, True]
    assert verdicts(5, screen) == [True]
    # Decay 1 is not complete: both parts of prediction 6 wait for it.
    assert verdicts(6, screen) == [None, None]
    rows = batch_summary.evaluate(predictions_batch_01.PREDICTIONS[5]['parts'], screen)
    assert rows[1][2] == f'Waiting for: {scheduled[1]}.'
    assert 'median decay slope -0.30 against -0.60 for the control' in predictions_batch_01.error_length_slopes(screen)
    assert f'{fast[0.3]}: $W$ 0.850, $L$ 0.990, median first arrival 320.0K' in predictions_batch_01.named(screen, fast)

    # With every method complete: decay 1 arrives first of all (90 thousand), but decay 0.75 ends higher than 0.5.
    values = {slug: (0.5, 0.9, 500e3, -0.5) for slug in METHODS}
    values.update({scheduled[1]: (0.6, 0.95, 90e3, -1.0), scheduled[0.5]: (0.6, 0.97, 200e3, -0.5),
                   scheduled[0.75]: (0.6, 0.99, 150e3, -0.75), fast[0.3]: (0.85, 0.99, 390e3, -0.6)})
    complete = fake_screen(values, {slug: difference(0.0, -0.1, 0.1) for slug in METHODS if slug != CONTROL})
    assert verdicts(6, complete) == [True, False]
    # While units are missing, decay 1 is the earliest so far and the claim stays open; one complete method that
    # arrives earlier (the control at 80 thousand steps) settles it.
    partial = {slug: values[slug] for slug in (*scheduled.values(), CONTROL)}
    rows = batch_summary.evaluate(predictions_batch_01.PREDICTIONS[5]['parts'], fake_screen(partial))
    assert rows[0][1] is None and 'Median first-arrival step 90.0K; the earliest of the other complete methods' in rows[0][2]
    assert f'Waiting for: {EVEN}; ' in rows[0][2]
    partial[CONTROL] = (0.8, 0.99, 80e3, -0.6)
    rows = batch_summary.evaluate(predictions_batch_01.PREDICTIONS[5]['parts'], fake_screen(partial))
    assert rows[0][1] is False and rows[0][2].endswith(f'is {CONTROL} at 80.0K.')
    # The fast last layer's best step size is now 2.5% below the control's 400 thousand steps: less than 15%.
    values[CONTROL] = (0.8, 0.99, 400e3, -0.6)
    assert verdicts(5, fake_screen(values)) == [False]
    assert 'decay 0.5: +0.000 [-0.100, +0.100]' in predictions_batch_01.scheduled_spill(complete)
    assert 'control: 0.5 (40 pairs) at 5%, 0.5 (40 pairs) at 25%' in predictions_batch_01.scheduled_spill(complete)


def test_a_bonus_that_rises_above_its_start_is_named_with_the_records(tmp_path):
    """The start bonus is 1 at both cells; at the visited cell it then grows to 3 x 20 = 60 in the last window."""
    run = made_up_screen(tmp_path)
    write_shard(run, screen_unit(EVEN, [10, 12, 2, 2, 10, 10], [1.0, 1.0], lambda index: [3.0 * (index + 1), 1.0],
                                 lambda index: [1, 0]))
    result = batch_summary.generate(run, BATCH)
    grid = result['screen'].methods[EVEN].grid
    assert grid['largest_bonus'] == {'start': 1.0, 'windows': 60.0} and grid['decay_slope'] > 0
    look = (run / 'batches' / BATCH / 'summary.md').read_text().split('## Records that need a look\n')[1]
    assert (f'- even start (`{unit_name(EVEN)}`): the largest bonus at a probe in a logging window is 60, above the '
            'largest at the start (1), over all intrinsic weights.') in look
