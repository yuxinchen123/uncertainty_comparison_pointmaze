"""Tests of a screening batch's summary: small helpers, the batch-1 predictions, and a made-up screen end to end."""

import json

import pytest

from maze_record_examples import flat_policy, goal_windows, grid_records, start_record, write_shard
from maze_campaign.analysis import batch_summary, compare, figures, predictions_batch_01, records, report

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
        'environments': ['pointmaze-large'], 'reward_settings': ['arrival_plus_one'],
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
    monkeypatch.setattr(batch_summary, 'MOST_WEIGHT_COLUMNS', 2)
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
    tables = text.split('## Discovery and optimization: whole-curve mean\n')[1].split('## Properties')[0]
    assert 'error length' not in tables and 'fast last layer' not in tables and '## Screen table' not in text

    # The table of the primary metric. Control: W 0.8 and 0.7 at weight 1. Even start: W 0.9 twice at weight 16, so its differences
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
    assert '| method | weight 1 | weight 16 | best intrinsic weight for $W$ |' in weights
    assert 'The 3 weights are split over two tables' in weights
    assert '| control | **0.750** | <u>0.500</u> | 1 (smallest of the grid) |' in weights
    assert '| even start | <u>0.450</u> | **0.900** | 16 |' in weights
    assert '| fixed scale | 0.000 | 0.125 | 64 (largest of the grid) |' in weights
    assert '| method | weight 64 | best intrinsic weight for $W$ |' in weights and '| even start | **0.500** | 16 |' in weights
    assert 'Best weight at an end of the grid: control (1, the smallest); fixed scale (64, the largest).' in weights

    # The table of the secondary metric, in thousands of steps. First arrivals at the best weight for M: control
    # (weight 1) at steps 408 and 608, so M = 508 with a standard error of 100; even start (weight 16) at step 208
    # twice, so its differences from the control are -200 and -400; fixed scale (weight 64) never and at step
    # 1,208, so M = (2,000 + 1,208) / 2 = 1,604 and its differences are +1,592 and +600.
    discovery = tables.split('## Discovery: mean capped first-goal time\n')[1].split('\n## ')[0]
    assert [line for line in discovery.splitlines() if line.startswith('| ')][1:] == [
        '| control | 1 (smallest of the grid) | <u>0.508</u> ± 0.1 | control | **2 of 2** | <u>0.750</u> ± 0.050 | **1.000** |',
        '| even start | 16 | **0.208** ± 0 | **-0.3** [-0.4, -0.2] | **2 of 2** | **0.900** ± 0.000 | **1.000** |',
        '| fixed scale | 64 (largest of the grid) | 1.6 ± 0.396 | <u>+1.1</u> [+0.6, +1.59] | <u>1 of 2</u> | 0.200 ± 0.200 '
        '| <u>0.500</u> |']
    assert "at the control's own best intrinsic weight for $M$ (1), on the 2 seeds both have" in discovery
    assert 'larger or smaller weights must be tried' in discovery and 'a screen and not a verdict' in discovery
    assert '| $M$ ↓ ± standard error, thousand steps |' in discovery and discovery.count('↓') == 1

    # M at every weight: the smallest per weight is bold; a policy with no arrival counts as the 2,000 steps.
    capped = tables.split('## Mean capped first-goal time at every intrinsic weight\n')[1].split('\n## ')[0]
    assert '| method | weight 1 | weight 16 | best intrinsic weight for $M$ |' in capped
    assert '| control | **0.508** | <u>1.01</u> | 1 (smallest of the grid) |' in capped
    assert '| even start | <u>1.11</u> | **0.208** | 16 |' in capped
    assert '| fixed scale | 2 | 1.75 | 64 (largest of the grid) |' in capped
    assert '| method | weight 64 | best intrinsic weight for $M$ |' in capped and '| even start | **1.01** | 16 |' in capped
    assert 'Best weight for $M$ at an end of the grid: control (1, the smallest); fixed scale (64, the largest).' in capped
    # The formula of M stands in the section Measures, where it is first introduced.
    assert '$M(B) = \\dfrac{1}{K}\\sum_{i=1}^{K} \\min(T_i, B)$' in text.split('## Measures\n')[1].split('\n## ')[0]

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
    # The same lists by the interval of the difference in M: below zero is ahead on discovery.
    by_discovery = sides.split('### Discovery: mean capped first-goal time\n')[1]
    assert ('- **Ahead of the control on discovery** (interval entirely below zero): even start '
            '(-0.3 [-0.4, -0.2]).') in by_discovery
    assert ('- **Behind the control on discovery** (interval entirely above zero): fixed scale '
            '(+1.1 [+0.6, +1.59]).') in by_discovery
    assert '- **Within the interval** (it contains zero): none.' in by_discovery
    assert '## Validation block' not in text

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
    assert '0 of 14 units are complete.' in text and '## Discovery' not in text and '![' not in text
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


def test_the_best_weight_for_discovery_is_the_smallest_m_and_can_differ_from_the_best_weight_for_w():
    """Weight 1 arrives first and its return then falls; weight 4 arrives later and stays: M takes 1, W takes 4."""
    def policies(values, first_steps, weight):
        return [flat_policy(values, seed=seed, first_goal_step=first, settings=(('int_coef', weight),))
                for seed, first in enumerate(first_steps)]
    # Weights 16 and 64 both have M = 260 (one policy of weight 16 never arrives and counts as the 400 steps of
    # the budget), so the weight at which more policies arrive stands first of the two.
    groups = records.configurations(
        policies([1, 0, 0, 0], [50, 70], 1.0) + policies([0, 1, 1, 1], [150, 170], 4.0)
        + policies([0, 0, 0, 1], [120, -1], 16.0) + policies([0, 0, 1, 1], [260, 260], 64.0))
    scores, best = batch_summary.score_weights_for_discovery(groups, 400)
    assert scores == {1.0: (60.0, 1.0), 4.0: (160.0, 1.0), 16.0: (260.0, 0.5), 64.0: (260.0, 1.0)}
    assert list(scores) == [1.0, 4.0, 64.0, 16.0] and batch_summary.weight_of(best) == 1.0
    assert batch_summary.weight_of(batch_summary.score_weights(groups, 400)[1]) == 4.0
    # At a budget of 100 steps only the weight 1 has arrived; the others count as the budget.
    assert batch_summary.score_weights_for_discovery(groups, 100)[0][4.0] == (100.0, 0.0)


def test_a_best_weight_at_an_end_of_the_weights_is_marked():
    """The smallest and the largest weight a method was trained at are marked; a weight inside is not."""
    assert batch_summary.end_of_grid([1.0, 16.0, 64.0], 64.0) == 'largest'
    assert batch_summary.end_of_grid([1.0, 16.0, 64.0], 1.0) == 'smallest'
    assert batch_summary.end_of_grid([1.0, 16.0, 64.0, 256.0], 64.0) is None
    method = fake_screen({CONTROL: (0.5, 0.5, 100.0, -1.0)}).methods[CONTROL]
    assert batch_summary.weight_cell(method, 2.0) == '2 (smallest of the grid)'


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
        measured = {'whole_curve_mean': whole, 'final_level': level, 'median_first_goal_step': arrival}
        methods[slug] = batch_summary.Method(
            slug=slug, name=slug, unit_id=slug, budget=100, groups=(best,), scores={}, best=best, measures=measured,
            discovery_scores={}, discovery_best=best, discovery_measures=measured,
            grid={'start_ratio': 3.0, 'decay_slope': slope, 'unvisited': unvisited, 'weight': 2.0})
    return batch_summary.Screen(batch='made up', control=CONTROL, order=tuple(METHODS), names={slug: slug for slug in METHODS},
                                units=(), methods=methods, differences=differences or {}, discovery_differences={},
                                budget=100, seeds=(0, 1))


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


LARGER = 'batch-01-screen-larger-weights'


def add_larger_weights(run, control_goal_windows=(0, 0, None, None), complete=True):
    """Add to the made-up screen a study description of two larger weights, and the control's unit of it."""
    folder = run / 'batches' / BATCH
    (folder / 'screen_larger.json').write_text(json.dumps({
        'phase': LARGER, 'steps_per_policy': 2000, 'seeds': {'first': 0, 'count': 2},
        'cells': {'kind': 'grid', 'values': {'intrinsic_weight': [256.0, 1024.0]}},
        'sweeps': [{'method': CONTROL}, {'method': EVEN}]}))
    settings = json.loads((folder / 'summary_settings.json').read_text())
    (folder / 'summary_settings.json').write_text(json.dumps({**settings, 'more_intrinsic_weights': ['screen_larger.json']}))
    unit = f'{LARGER}_pointmaze-large_{CONTROL}_cells-1-to-2-of-2_seeds-0-to-1'
    (run / 'specifications' / f'{unit}.json').write_text(json.dumps({'unit_id': unit, 'phase': LARGER, 'method': CONTROL}))
    # The unit has no maze-grid records, and its code has no `permutation_draw` option.
    start = start_record(unit, iterations=20, cells=((1e-4, 256.0), (1e-4, 1024.0)))
    write_shard(run, [start] + goal_windows(start, list(control_goal_windows)), complete=complete)
    return unit


def test_units_with_larger_weights_extend_a_method_and_leave_the_others_cells_empty(tmp_path, monkeypatch):
    """The control is also trained at the weights 256 and 1,024: its best weights are chosen over all five."""
    monkeypatch.setattr(batch_summary, 'FEWEST_PAIRS', 2)
    monkeypatch.setattr(batch_summary, 'MOST_WEIGHT_COLUMNS', 3)
    run = made_up_screen(tmp_path)
    unit = add_larger_weights(run)
    screen = batch_summary.generate(run, BATCH)['screen']
    text = (run / 'batches' / BATCH / 'summary.md').read_text()

    # Units: the control's further unit is complete; even start's was never built and is listed.
    units = text.split('## Units\n')[1].split('\n## ')[0]
    assert '4 of 16 units are complete.' in units
    assert '| 15 | control, at the intrinsic weights 256, 1024 | complete | 20 of 20 |' in units
    assert '| 16 | even start, at the intrinsic weights 256, 1024 | no unit was built | 0 |' in units
    assert f'15. `{unit}`' in units

    # At the weight 256 both policies of the control arrive at step 8 and W is 1: the best weight of both metrics,
    # and no longer at an end of the control's five weights. The other methods keep their three weights.
    control = screen.methods[CONTROL]
    assert control.weights == [1.0, 16.0, 64.0, 256.0, 1024.0] and screen.methods[EVEN].weights == [1.0, 16.0, 64.0]
    assert control.best_weight == control.discovery_weight == 256.0 and control.grid['weight'] == 1.0
    assert control.discovery_scores[256.0] == (8.0, 1.0) and control.discovery_scores[1024.0] == (2000.0, 0.0)
    primary = text.split('## Discovery and optimization: whole-curve mean\n')[1].split('\n## ')[0]
    assert '| control | 256 | **1.000** ± 0.000 | control |' in primary and 'best intrinsic weight (256)' in primary
    discovery = text.split('## Discovery: mean capped first-goal time\n')[1].split('\n## ')[0]
    assert '| control | 256 | **0.008** ± 0 | control | **2 of 2** | **1.000** ± 0.000 | **1.000** |' in discovery
    # Even start at its weight 16 (step 208 twice) against the control at 256 (step 8 twice): +200 steps, behind.
    assert '| even start | 16 | <u>0.208</u> ± 0 | **+0.2** [+0.2, +0.2] | **2 of 2** |' in discovery
    assert screen.discovery_differences[EVEN]['verdict'] == 'behind on discovery'

    # Both tables at every weight: five weights in two tables, and empty cells where a method has no unit.
    for heading, mark in (('## Whole-curve mean at every intrinsic weight\n', '$W$'),
                          ('## Mean capped first-goal time at every intrinsic weight\n', '$M$')):
        section = text.split(heading)[1].split('\n## ')[0]
        assert 'The 5 weights are split over two tables' in section and 'has an empty cell there' in section
        assert f'| method | weight 1 | weight 16 | weight 64 | best intrinsic weight for {mark} |' in section
        assert f'| method | weight 256 | weight 1024 | best intrinsic weight for {mark} |' in section
        assert '| even start |  |  | 16 |' in section and '| fixed scale |  |  | 64 (largest of the grid) |' in section
    assert '| control | 1.000 | 0.000 | 256 |' in text.split('## Whole-curve mean at every intrinsic weight\n')[1]
    assert '| control | 0.008 | 2 | 256 |' in text.split('## Mean capped first-goal time at every intrinsic weight\n')[1]

    # The property measures stay those of the screen's own unit, at its best weight among its own weights.
    properties = text.split('## Properties of the bonus on the maze grid\n')[1].split('\n## ')[0]
    assert 'It differs from the best weight for $W$ over all weights for: control.' in properties
    assert '| control | 1 | 2 and 2 | 2.5 | 1.25 | 0.583 | 0.345 |' in properties
    # A setting that only one of the two units records is named with the records.
    look = text.split('## Records that need a look\n')[1]
    assert (f"- control, at the intrinsic weights 256, 1024 (`{unit}`): settings that only the method's first unit "
            'records: `permutation_draw` = x.') in look


def test_a_further_unit_is_left_out_while_it_or_its_methods_own_unit_is_not_complete(tmp_path, monkeypatch):
    """An unfinished unit with larger weights changes nothing; a complete one without its screen unit is left out."""
    monkeypatch.setattr(batch_summary, 'FEWEST_PAIRS', 2)
    run = made_up_screen(tmp_path)
    add_larger_weights(run, complete=False)
    screen = batch_summary.generate(run, BATCH)['screen']
    assert screen.methods[CONTROL].weights == [1.0, 16.0, 64.0] and screen.methods[CONTROL].best_weight == 1.0
    assert '3 of 16 units are complete.' in (run / 'batches' / BATCH / 'summary.md').read_text()
    # The further unit complete, the control's own unit of the screen unfinished: nothing of the control is read.
    add_larger_weights(run)
    shard = run / 'data' / f'{unit_name(CONTROL)}.jsonl'
    shard.write_text(''.join(line + '\n' for line in shard.read_text().splitlines()[:-1]))
    screen = batch_summary.generate(run, BATCH)['screen']
    text = (run / 'batches' / BATCH / 'summary.md').read_text()
    assert CONTROL not in screen.methods and 'Left out: control; even start with no divisor; ' in text
    assert "complete, but left out: the method's own unit of the screen is not complete." in text
    # A further unit that differs from its method's unit in another setting than the weight is refused.
    with pytest.raises(ValueError, match="settings recorded with other values \\['clip_coef'\\]"):
        first = records.configurations([flat_policy([1], seed=0, settings=(('clip_coef', 0.1), ('int_coef', 1.0)))])
        other = records.configurations([flat_policy([1], seed=0, settings=(('clip_coef', 0.2), ('int_coef', 4.0)))])
        batch_summary.join_further_weights(first, other, {'unit_id': 'u', 'notes': []})


def add_validation_block(run):
    """Add to the made-up screen a validation block: the control at the weights 1 and 2, an idea at 1 and 4."""
    folder = run / 'batches' / BATCH
    settings = json.loads((folder / 'summary_settings.json').read_text())
    idea = 'predictor-own-step-size-1'
    (folder / 'summary_settings.json').write_text(json.dumps({**settings, 'validation_block': {
        'phase': 'validation-block', 'control': CONTROL,
        'method_names': {idea: 'separate predictor step size', CONTROL: 'control'}}}))
    # Goal from the given window on, per seed; the control's unit at the weight 2 has written half its windows.
    units = ((CONTROL, 1.0, [4, 6, 4, 6], True, 'pointmaze-large'), (CONTROL, 2.0, [1, 1, 1, 1], False, 'pointmaze-large'),
             (idea, 1.0, [2, 2, 2, 2], True, 'pointmaze-large'), (idea, 4.0, [10, 10, 10, 10], True, 'pointmaze-large'),
             (idea, 16.0, [0, 0, 0, 0], True, 'pointmaze-medium'),
             ('method-of-another-batch', 1.0, [0, 0, 0, 0], True, 'pointmaze-large'))
    for method, weight, goal, complete, environment in units:
        unit = f'validation-block_{environment}_{method}_intrinsic-weight-{weight}_seeds-10000-to-10003'
        (run / 'specifications' / f'{unit}.json').write_text(json.dumps({
            'unit_id': unit, 'phase': 'validation-block', 'method': method, 'environment': environment,
            'reward_setting': 'arrival_plus_one', 'cell_setting_names': ['learning_rate', 'int_coef'],
            'cells': [[1e-4, weight]]}))
        start = start_record(unit, iterations=20, cells=((1e-4, weight),), seeds=(10000, 10001, 10002, 10003))
        windows = goal_windows(start, goal)
        write_shard(run, [start] + (windows if complete else windows[:10]), complete=complete)


def test_the_validation_block_lists_each_method_and_weight_against_the_control_at_the_same_weight(tmp_path, monkeypatch):
    """Complete units of the block's phase in the screen's environment enter two tables; the others are listed."""
    monkeypatch.setattr(batch_summary, 'FEWEST_PAIRS', 2)
    run = made_up_screen(tmp_path)
    add_validation_block(run)
    batch_summary.generate(run, BATCH)
    text = (run / 'batches' / BATCH / 'summary.md').read_text()
    # The screen still reads its own three complete units only.
    assert '3 of 14 units are complete.' in text
    block = text.split('## Validation block\n')[1].split('\n## ')[0]
    assert '4 seeds, from 10000 to 10003' in block and '3 of 4 method and weight pairs have a complete unit' in block
    assert 'read at 2,000 environment steps per policy' in block and 'pointmaze-medium' not in block
    # W of the control at the weight 1 is 0.8 and 0.7 by seed, of the idea 0.9: differences +0.1 and +0.2.
    primary = block.split('### Discovery and optimization: whole-curve mean and final level\n')[1].split('###')[0]
    assert '| control | 1 | <u>0.750</u> ± 0.029 | control | 1.000 ± 0.000 | control | 4 |' in primary
    assert '| separate predictor step size | 1 | **0.900** ± 0.000 | +0.150 [+0.100, +0.200] | 1.000 ± 0.000 |' in primary
    assert ('| separate predictor step size | 4 | 0.500 ± 0.000 | no complete control unit at this weight | 1.000 ± 0.000 '
            '| no complete control unit at this weight | not compared |') in primary
    # First arrivals: the control at steps 408 and 608, the idea at step 208, so it is 200 and 400 steps earlier.
    secondary = block.split('### Discovery: mean capped first-goal time\n')[1]
    assert '| control | 1 | <u>0.508</u> ± 0.0577 | control | 4 of 4 | 4 |' in secondary
    assert '| separate predictor step size | 1 | **0.208** ± 0 | -0.3 [-0.4, -0.2] | 4 of 4 | 4 |' in secondary
    assert '| separate predictor step size | 4 | 1.01 ± 0 | no complete control unit at this weight | 4 of 4 | not compared |' in secondary
    assert 'Not complete, left out:\n\n- control, intrinsic weight 2: unfinished, 10 logging windows written.' in block
    # A unit of the phase whose method the block does not name belongs to another batch: named, not read.
    assert 'they are not read here: method of another batch.' in block and block.count('method of another batch') == 1
    # Best weight against best weight: the control has one complete weight (1); the idea's best weight is 1 for both
    # metrics (W 0.9 against 0.5 at the weight 4, first arrival at step 208 against 1,008).
    best = block.split('### Each method at its best intrinsic weight against the control at its best weight\n')[1]
    assert 'Each best weight was chosen on these same seeds' in best
    primary, secondary = best.split('**Mean capped first-goal time, at the best weights for $M$.**')
    assert '| control | 1 (its only weight) | <u>0.750</u> ± 0.029 | control | 1.000 ± 0.000 | control | control | 4 |' in primary
    assert ('| separate predictor step size | 1 (of 1, 4) | **0.900** ± 0.000 | +0.150 [+0.100, +0.200] | 1.000 ± 0.000 '
            '| +0.000 [+0.000, +0.000] | -0.3 [-0.4, -0.2] | 4 |') in primary
    assert '| control | 1 (its only weight) | <u>0.508</u> ± 0.0577 | control | 4 of 4 | <u>0.750</u> | 1.000 | 4 |' in secondary
    assert '| separate predictor step size | 1 (of 1, 4) | **0.208** ± 0 | -0.3 [-0.4, -0.2] | 4 of 4 | **0.900** | 1.000 | 4 |' in secondary
    # A block of the settings with a key it does not know is refused.
    with pytest.raises(ValueError, match="validation_block: unknown keys \\['colour'\\], missing keys \\['control'\\]"):
        batch_summary.read_validation(run, {}, {'phase': 'p', 'method_names': {}, 'colour': 'blue'}, BATCH)


def weight_group(method, weight, curves, first_steps, seeds=None):
    """One configuration in memory: a method at one intrinsic weight, one training curve and first arrival per seed."""
    seeds = range(len(curves)) if seeds is None else seeds
    return records.configurations([
        flat_policy(curve, seed=seed, first_goal_step=first, method=method, settings=(('int_coef', weight),))
        for seed, curve, first in zip(seeds, curves, first_steps)])[0]


def test_the_best_weights_of_a_validation_block_are_chosen_per_metric_and_compared_on_the_common_seeds():
    """The idea's best weight is 2 for W and 8 for M; the control's is 1 for both; an idea on other seeds is not compared."""
    block = {'phase': 'p', 'control': 'c', 'method_names': {'c': 'control', 'a': 'idea a', 'b': 'idea b', 'd': 'idea d'}}
    groups = {
        # The control: W 0.5 at the weight 1 (arrivals at steps 150 and 250) and 0.25 at the weight 4.
        ('c', 1.0): weight_group('c', 1.0, [[0, 1], [0, 1]], [150, 250]),
        ('c', 4.0): weight_group('c', 4.0, [[0, 0.5], [0, 0.5]], [180, 260]),
        # Idea a: W 0.75 and 0.65 at the weight 2 (arrivals at 100 and 140); at the weight 8 it arrives at steps 20
        # and 40 and ends at 0.
        ('a', 2.0): weight_group('a', 2.0, [[0.5, 1], [0.3, 1]], [100, 140]),
        ('a', 8.0): weight_group('a', 8.0, [[1, 0], [1, 0]], [20, 40]),
        # Idea b has one weight and one seed in common with the control (seed 1); idea d was trained on other seeds.
        ('b', 2.0): weight_group('b', 2.0, [[1, 1], [1, 1]], [10, 10], seeds=(1, 7)),
        ('d', 2.0): weight_group('d', 2.0, [[1, 1], [1, 1]], [10, 10], seeds=(8, 9))}
    tables = batch_summary.validation_best(groups, block, 200, 'made-up batch')
    whole = {row['slug']: row for row in tables['whole']}
    discovery = {row['slug']: row for row in tables['discovery']}
    # Rows: the control first, then from the best value on (ties in the settings' order).
    assert [row['slug'] for row in tables['whole']] == ['c', 'b', 'd', 'a']
    assert [row['slug'] for row in tables['discovery']] == ['c', 'b', 'd', 'a']
    assert (whole['c']['weight'], whole['a']['weight'], discovery['c']['weight'], discovery['a']['weight']) == (1.0, 2.0, 1.0, 8.0)
    assert whole['a']['weights'] == [2.0, 8.0] and whole['c']['result'] is None
    # W of idea a at the weight 2 minus the control's at the weight 1: 0.75 - 0.5 and 0.65 - 0.5.
    assert whole['a']['result']['whole_curve_mean']['mean'] == pytest.approx(0.2) and whole['a']['result']['seeds'] == 2
    # M at the best weights for M: 30 steps against the control's 175 (its second arrival counts as the budget, 200).
    assert discovery['a']['measures']['capped_first_goal_step'] == 30.0
    assert discovery['a']['result']['capped_first_goal_step']['mean'] == pytest.approx(-145.0)
    assert whole['b']['result']['seeds'] == 1 and whole['d']['result']['seeds'] == 0
    text = batch_summary.validation_best_text(tables, block)
    assert '| idea a | 2 (of 2, 8) | <u>0.700</u> ± 0.050 | +0.200 [+0.150, +0.250] |' in text
    assert '| idea b | 2 (its only weight) | **1.000** ± 0.000 | fewer than two seeds in common |' in text
    assert '| idea a | 8 (of 2, 8) | <u>0.03</u> ± 0.01 | -0.145 [-0.16, -0.13] | **2 of 2** |' in text
    # Without a complete control unit every method is listed and none is compared.
    alone = batch_summary.validation_best({key: group for key, group in groups.items() if key[0] != 'c'}, block, 200, 'made-up batch')
    assert all(row['result'] is None for row in alone['whole'])
    assert '| idea a | 2 (of 2, 8) | <u>0.700</u> ± 0.050 | no complete control unit |' in batch_summary.validation_best_text(alone, block)
    # Many seeds: four decimals.
    many = {('c', 1.0): weight_group('c', 1.0, [[0, 1]] * 100, [150] * 100),
            ('a', 2.0): weight_group('a', 2.0, [[0.5, 1]] * 100, [100] * 100)}
    text = batch_summary.validation_best_text(batch_summary.validation_best(many, block, 200, 'made-up batch'), block)
    assert '| idea a | 2 (its only weight) | **0.7500** ± 0.0000 | +0.2500 [+0.2500, +0.2500] |' in text


def test_a_batch_that_takes_its_control_from_another_batch_and_has_no_predictions_module(tmp_path, monkeypatch):
    """The second batch reads the control's unit of the first batch's study description; its predictions are quoted."""
    monkeypatch.setattr(batch_summary, 'FEWEST_PAIRS', 2)
    run = made_up_screen(tmp_path)
    other, idea = '02_made-up-batch', 'slower-predictor-with-held-divisor'
    folder = run / 'batches' / other
    folder.mkdir()
    (run / 'specifications' / f'{unit_name(idea)}.json').write_text(json.dumps(
        {'unit_id': unit_name(idea), 'phase': 'batch-02-screen', 'method': idea}))
    # Every policy of the idea reaches the goal from window 2 on, at every weight: W = 0.9 and first arrival at step 208.
    write_shard(run, screen_unit(idea, [2] * 6, [8.0, 2.0], lambda index: [8.0 / (index + 2), 1.0], lambda index: [1, 0]))
    description = {'phase': 'batch-02-screen', 'steps_per_policy': 2000, 'seeds': {'first': 0, 'count': 2},
                   'sweeps': [{'method': idea}]}
    (folder / 'screen.json').write_text(json.dumps(description))
    (folder / 'ideas.md').write_text('# Second batch\n\n## Predictions (written before the runs)\n\n'
                                     '1. **First:** text one.\n2. **Second:** text\n   two.\n')
    settings = {'title': 'Second made-up screen', 'study_description': 'screen.json', 'control': CONTROL,
                'control_study_description': f'../{BATCH}/screen.json',
                'method_names': {CONTROL: 'control (the unit of batch 1)', idea: 'slower predictor with held divisor'}}
    (folder / 'summary_settings.json').write_text(json.dumps(settings))
    screen = batch_summary.generate(run, other)['screen']
    text = (folder / 'summary.md').read_text()
    assert screen.order == (CONTROL, idea) and set(screen.methods) == {CONTROL, idea}
    assert '2 of 2 units are complete.' in text and '| 1 | control (the unit of batch 1) | complete | 20 of 20 |' in text
    # Both tables compare with the first batch's control: W 0.9 against 0.8 and 0.7, arrivals at 208 against 408 and 608.
    primary = text.split('## Discovery and optimization: whole-curve mean\n')[1].split('\n## ')[0]
    assert '| slower predictor with held divisor | 1 | **0.900** ± 0.000 | +0.150 [+0.100, +0.200] |' in primary
    discovery = text.split('## Discovery: mean capped first-goal time\n')[1].split('\n## ')[0]
    assert ('| slower predictor with held divisor | 1 (smallest of the grid) | **0.208** ± 0 | -0.3 [-0.4, -0.2] '
            '| 2 of 2 | **0.900** ± 0.000 | 1.000 |') in discovery
    # Without a predictions module the predictions of ideas.md are quoted and not read.
    predictions = text.split('## Predictions\n')[1].split('\n## ')[0]
    assert 'this batch has no predictions module' in predictions
    assert '### Prediction 1\n\n> **First:** text one.\n\nObserved: **not read yet**.' in predictions
    assert '### Prediction 2\n\n> **Second:** text two.\n\nObserved: **not read yet**.' in predictions
    assert '## Consistency check' not in text and '## Validation block' not in text
    # The other study description must have the same seeds and budget, and the batch must not train the control itself.
    (folder / 'screen.json').write_text(json.dumps({**description, 'steps_per_policy': 4000}))
    with pytest.raises(ValueError, match='cannot give this batch its control'):
        batch_summary.generate(run, other)
    (folder / 'screen.json').write_text(json.dumps({**description, 'sweeps': [{'method': CONTROL}, {'method': idea}]}))
    with pytest.raises(ValueError, match='cannot give this batch its control'):
        batch_summary.generate(run, other)
