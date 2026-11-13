"""Tests of the prediction readings of batches 2, 3 and 4 on made-up numbers."""

from maze_record_examples import flat_policy
from maze_campaign.analysis import (batch_summary, case_sweeps, compare, predictions_batch_02, predictions_batch_03,
                                    predictions_batch_04, records)

CONTROL = predictions_batch_02.CONTROL


def method(slug, *, whole=0.8, level=0.99, arrival=200e3, reached=16, capped=200e3, discovery_weight=16.0, scores=None,
           discovery_scores=None, start_ratio=88.0, slope=-0.6, unvisited=(1.0, 1.0), reward=(0.002, 0.004), divisor=0.1,
           apart=(20.0, None)):
    """One method of a screen in memory, at a best weight of 2 for W; every number can be set."""
    best = records.Configuration('PointMaze Large', 'arrival_plus_one', slug, 4, 100, (('int_coef', 2.0),), ())
    found = records.Configuration('PointMaze Large', 'arrival_plus_one', slug, 4, 100, (('int_coef', discovery_weight),), ())
    measured = {'whole_curve_mean': whole, 'final_level': level, 'median_first_goal_step': arrival, 'seeds': 16,
                'reached_goal_share': reached / 16, 'capped_first_goal_step': capped}
    shares = dict(zip(batch_summary.UNVISITED_SHARES, (*unvisited, None)))
    grid = {'start_ratio': start_ratio, 'decay_slope': slope, 'weight': 2.0,
            'unvisited': {share: {'pairs': 40 if median is not None else 3, 'median': median, 'step': 0} for share, median in shares.items()},
            'unvisited_over_visited': {share: {'ratio': ratio, 'unvisited_pairs': 200 if ratio is not None else 13, 'visited_pairs': 300}
                                       for share, ratio in zip(batch_summary.UNVISITED_SHARES, (*apart, None))},
            'intrinsic_reward': {'start': 0.01, 'first_window': 0.01, 0.25: reward[0], 1.0: reward[1]},
            'divisor': {'start': 1.0, 'first_window': divisor}}
    return batch_summary.Method(slug=slug, name=slug, unit_id=slug, budget=100, groups=(best,), scores=scores or {}, best=best,
                                measures=measured, discovery_scores=discovery_scores or {}, discovery_best=found,
                                discovery_measures={**measured, 'capped_first_goal_step': capped}, grid=grid)


def difference(mean, low, high):
    """A paired difference as compare.compare_with_baseline gives it, the same for W and for L."""
    one = {'mean': mean, 'interval': [low, high], 'side': compare.side([low, high])}
    return {'whole_curve_mean': one, 'final_level': one, 'seeds': 16}


def screen_of(methods, differences, order=None, units=(), validation=None):
    """A screen in memory from methods and their differences from the control."""
    slugs = order or list(methods)
    return batch_summary.Screen(batch='made up', control=CONTROL, order=tuple(slugs), names={slug: slug for slug in slugs},
                                units=tuple(units), methods=methods, differences=differences, discovery_differences={},
                                budget=100, seeds=tuple(range(16)), validation=validation)


def verdicts(module, number, screen):
    """The verdict of each part of one prediction."""
    return [verdict for _, verdict, _ in batch_summary.evaluate(module.PREDICTIONS[number - 1]['parts'], screen)]


def test_batch_2_predictions_on_made_up_numbers():
    """Each part is read from the numbers it names; two-outcome predictions hold when one outcome occurred."""
    module = predictions_batch_02
    clip, kept, slower, local, scheduled, held = module.CLIP, module.KEPT, module.SLOWER, module.LOCAL, module.SCHEDULED, module.HELD
    level = {4.0: (0.86, 0.975), 16.0: (0.57, 0.60)}
    methods = {slug: method(slug) for slug in (*clip.values(), *kept.values(), *slower.values(), *local.values(),
                                                *scheduled.values(), *held.values())}
    methods[CONTROL] = method(CONTROL, whole=0.885, arrival=192e3, scores=level, discovery_scores={2.0: (270e3, 1.0)})
    # Prediction 1: ratios 500 and 2,000; the clip at 20 arrives at 150 thousand steps (22% earlier); W moves by 0.03 with no clip.
    methods[clip['clip at 10']] = method(clip['clip at 10'], start_ratio=500.0)
    methods[clip['clip at 20']] = method(clip['clip at 20'], arrival=150e3)
    methods[clip['no clip']] = method(clip['no clip'], start_ratio=2000.0, arrival=float('inf'))
    # Prediction 2: unvisited cells have 20 times the bonus of visited ones at 5% (too few pairs at 25%), but only 4
    # times with the clip at 20; no clip is 0.02 ahead with earlier arrivals; the clip at 20 is 0.005 ahead, which is
    # neither outcome.
    methods[kept['no clip']] = method(kept['no clip'], arrival=100e3)
    methods[kept['clip at 20']] = method(kept['clip at 20'], apart=(4.0, None))
    # Prediction 3: both slower predictors have a smaller M at the weight 2; one interval lies above zero.
    for multiple, capped in ((0.0625, 150e3), (0.125, 170e3)):
        methods[slower[multiple]] = method(slower[multiple], discovery_scores={2.0: (capped, 1.0)})
    # Prediction 4: ratios below 3; all policies arrive with W 0.85, 10 of 16 arrive, 14 of 16 arrive (neither outcome).
    methods[local[0.5]] = method(local[0.5], start_ratio=1.5, whole=0.85)
    methods[local[1]] = method(local[1], start_ratio=2.0, whole=0.3, reached=10)
    methods[local[2]] = method(local[2], start_ratio=2.5, whole=0.6, reached=14)
    # Prediction 5: the bonus at unvisited cells stays at 0.95 or falls to 0.5; slopes -0.45, -0.7 and -1.05.
    methods[scheduled[(0.5, 0.5)]] = method(scheduled[(0.5, 0.5)], slope=-0.45, unvisited=(0.95, None),
                                           discovery_scores={1.0: (300e3, 1.0), 4.0: (900e3, 0.5)})
    methods[scheduled[(0.5, 1)]] = method(scheduled[(0.5, 1)], slope=-0.7, unvisited=(0.95, 0.5), discovery_scores={1.0: (300e3, 0.9)})
    methods[scheduled[(1, 1)]] = method(scheduled[(1, 1)], slope=-1.05, unvisited=(0.95, 0.95), discovery_scores={1.0: (300e3, 1.0)})
    # Prediction 6: the reward falls after a quarter of the budget for two hold fractions; the final levels are higher.
    high = {4.0: (0.89, 0.997), 16.0: (0.7, 0.99)}
    methods[held['one sixteenth']] = method(held['one sixteenth'], whole=0.895, reward=(0.004, 0.002), scores=high)
    methods[held['one eighth']] = method(held['one eighth'], whole=0.89, reward=(0.004, 0.002), scores=high)
    methods[held['one quarter']] = method(held['one quarter'], whole=0.88, reward=(0.004, 0.005),
                                         scores={4.0: (0.86, 0.97), 16.0: (0.7, 0.99)})
    differences = {slug: difference(0.0, -0.02, 0.02) for slug in methods if slug != CONTROL}
    differences[clip['no clip']] = difference(0.03, -0.01, 0.07)
    differences[kept['no clip']] = difference(0.02, -0.01, 0.05)
    differences[kept['clip at 20']] = difference(0.005, -0.02, 0.03)
    differences[slower[0.0625]] = difference(0.04, 0.01, 0.07)
    differences[held['one sixteenth']] = difference(0.010, -0.01, 0.03)
    screen = screen_of(methods, differences)
    assert verdicts(module, 1, screen) == [True, True, True, False]
    assert verdicts(module, 2, screen) == [False, False]
    assert verdicts(module, 3, screen) == [True, False]
    assert verdicts(module, 4, screen) == [True, False]
    assert verdicts(module, 5, screen) == [False, False, False]
    assert verdicts(module, 6, screen) == [False, False, True]
    rows = batch_summary.evaluate(module.PREDICTIONS[1]['parts'], screen)
    assert ('no clip: 20 (200 pairs without a visit, 300 with one) at 5%, too few pairs for a median (13 pairs without a visit, '
            '300 with one) at 25%; clip at 20: 4 (200 pairs') in rows[0][2]
    # With no share at which both kinds of cells have enough pairs, the part cannot be read.
    unread = dict(methods)
    for words in kept:
        unread[kept[words]] = method(kept[words], apart=(None, None))
    assert verdicts(module, 2, screen_of(unread, differences))[0] is None
    assert 'no clip: ahead with earlier arrivals' in rows[1][2] and 'clip at 20: neither written outcome' in rows[1][2]
    assert 'length 1: three quarters of the policies or fewer reach the goal (10 of 16 policies reach the goal, $W$ 0.300)' in (
        batch_summary.evaluate(module.PREDICTIONS[3]['parts'], screen)[1][2])
    reach = batch_summary.evaluate(module.PREDICTIONS[4]['parts'], screen)[2][2]
    assert 'decay 0.5, length 0.5: 1' in reach and 'decay 0.5, length 1: none' in reach
    assert 'control: 0.002 at a quarter of the budget and 0.004 at the end, 2.0 times as much.' == module.control_reward(screen)
    assert 'Held after one sixteenth (the highest $W$ of the three): difference in $W$ from the control +0.010' in (
        batch_summary.evaluate(module.PREDICTIONS[5]['parts'], screen)[2][2])
    # With the clip at 20 arriving later and every other part made true, each prediction holds or says what is missing.
    methods[clip['clip at 20']] = method(clip['clip at 20'], arrival=190e3)
    assert verdicts(module, 1, screen_of(methods, differences))[2] is False
    partial = screen_of({CONTROL: methods[CONTROL]}, {}, order=list(methods))
    assert verdicts(module, 6, partial) == [None, None, None]
    # The validation block's row of a method, in words; a method without a row gives no words.
    block = {'best': {'whole': [{'slug': 'divisor-held-after-one-sixteenth', 'weight': 2.0,
                                 'result': {'seeds': 256, 'whole_curve_mean': {'mean': 0.0095, 'interval': [0.0039, 0.0151]},
                                            'final_level': {'mean': 0.0056, 'interval': [0.0045, 0.0067]}}}]}}
    with_block = screen_of(methods, differences, validation=block)
    assert module.validation_words(with_block, 'divisor-held-after-one-sixteenth') == (
        "On the validation block (256 seeds), at its best weight 2 against the control at the control's best weight: "
        'difference in $W$ +0.0095 [+0.0039, +0.0151], in $L$ +0.0056 [+0.0045, +0.0067].')
    assert module.validation_words(with_block, 'another-method') == '' and module.validation_words(screen, 'any') == ''
    assert 'Held after one sixteenth: On the validation block (256 seeds)' in module.held_beside(with_block)


def test_batch_3_predictions_on_made_up_numbers():
    """The combinations against the control and against each other; a claim "of the batch" stays open while units are missing."""
    module = predictions_batch_03
    three, fast = module.THREE, module.FAST_HELD
    methods = {CONTROL: method(CONTROL, whole=0.885, level=0.994, capped=190e3),
               module.SLOW_HELD: method(module.SLOW_HELD, whole=0.881, level=0.998, capped=110e3),
               fast[0.1]: method(fast[0.1], whole=0.875, capped=62e3, scores={1.0: (0.875, 0.995), 2.0: (0.85, 0.98)}),
               fast[0.3]: method(fast[0.3], whole=0.889, capped=116e3, scores={1.0: (0.8, 0.99), 2.0: (0.7, 0.9)}),
               three['one sixteenth']: method(three['one sixteenth'], whole=0.898, capped=96e3),
               three['one eighth']: method(three['one eighth'], whole=0.894, capped=63e3)}
    differences = {slug: difference(methods[slug].measures['whole_curve_mean'] - 0.885, -0.02, 0.03) for slug in methods if slug != CONTROL}
    units = [{'name': slug, 'slug': slug, 'state': 'complete'} for slug in methods]
    screen = screen_of(methods, differences, order=[CONTROL, *module.IDEAS], units=units)
    # Prediction 1: -0.004 is not 0.015 ahead; the final level is above the control's.
    assert verdicts(module, 1, screen) == [False, True]
    # Prediction 2: the final level at the weight 2 is 0.98; M is 62 thousand; W is below the control's.
    assert verdicts(module, 2, screen) == [False, True, False]
    # Prediction 3: another combination has a smaller M (settled); the highest W so far, with one unit missing (open);
    # the two hold fractions differ by 0.004.
    assert verdicts(module, 3, screen) == [False, None, True]
    rows = batch_summary.evaluate(module.PREDICTIONS[2]['parts'], screen)
    assert f'the smallest of the other complete combinations is {fast[0.1]} at 62' in rows[0][2]
    assert rows[1][2].endswith(f'Waiting for: {module.FAST_SLOW}.')
    assert verdicts(module, 4, screen) == [True]
    assert module.validation_beside(module.SLOW_HELD)[1](screen) == (
        f'{module.SLOW_HELD}: the validation block has no complete unit of this method yet.')
    assert 'All 5 combinations are complete' in batch_summary.evaluate(module.PREDICTIONS[3]['parts'], screen)[0][2]
    # With the last unit in and below it, the highest W is settled; an unfinished unit leaves prediction 4 open.
    methods[module.FAST_SLOW] = method(module.FAST_SLOW, whole=0.883, capped=96e3)
    differences[module.FAST_SLOW] = difference(-0.002, -0.02, 0.03)
    units.append({'name': module.FAST_SLOW, 'slug': module.FAST_SLOW, 'state': 'unfinished'})
    complete = screen_of(methods, differences, order=[CONTROL, *module.IDEAS], units=units)
    assert verdicts(module, 3, complete) == [False, True, True] and verdicts(module, 4, complete) == [None]
    assert 'control: $W$ 0.885 at its best weight for $W$, $M$ 190 thousand steps' in module.control_beside(complete)


def group(method_name, learning_rate, weight, values, first_steps, reward_setting='arrival_plus_one'):
    """One configuration in memory: one training curve and one first arrival per seed."""
    return records.configurations([
        flat_policy(curve, seed=seed, first_goal_step=first, method=method_name, reward_setting=reward_setting,
                    settings=(('int_coef', weight), ('learning_rate', learning_rate)))
        for seed, (curve, first) in enumerate(zip(values, first_steps))])[0]


def case(words, baseline_groups, groups_by_finalist):
    """One case as case_sweeps.read_development gives it, from configurations in memory."""
    side = lambda slug, name, groups: case_sweeps.Side(slug=slug, name=name, units=(), groups=tuple(groups), planned=len(groups), budget=200)
    baseline = side('cleanrl-ppo-rnd', 'CleanRL PPO-RND', baseline_groups)
    sides = {slug: side(slug, name, groups_by_finalist.get(slug, ()))
             for slug, name in zip(predictions_batch_04.FINALISTS, ('all three parts', 'slower predictor with held divisor'))}
    return {'words': words, 'budget': 200, 'baseline': baseline, 'sides': sides,
            'comparisons': {slug: case_sweeps.compare_at_best(found, baseline, 200, 'test', words) for slug, found in sides.items()}}


def test_batch_4_predictions_are_read_from_the_cases():
    """Sooner, level, not ahead and the share of the worse group, each at the sides' own best configurations."""
    module = predictions_batch_04
    first, second = module.FINALISTS
    seeds = range(4)
    # PointMaze Medium with reward 1 on arrival. The baseline: W 0.5, first arrivals at step 150. The first finalist
    # is higher and sooner on every seed; the second is sooner at its weight 4 and level on W.
    baseline = [group('rnd', 0.006, 2.0, [[0, 1]] * 4, [150] * 4)]
    arrival = case('PointMaze Medium, reward 1 on arrival', baseline, {
        first: [group('a', 0.006, 0.5, [[0.5 + 0.02 * seed, 1] for seed in seeds], [100] * 4),
                group('a', 0.006, 1.0, [[0, 0]] * 4, [-1] * 4)],
        second: [group('b', 0.006, 0.5, [[0.1 * (seed % 2), 1] for seed in seeds], [160] * 4),
                 group('b', 0.006, 4.0, [[0, 0.2]] * 4, [40] * 4)]})
    # PointMaze Medium with the step cost: the first finalist is ahead of the baseline at its smallest weight.
    cost = lambda values, firsts, weight, name: group(name, 0.006, weight, values, firsts, reward_setting='step_cost')
    medium = case('PointMaze Medium, step cost', [cost([[-0.5, -0.4]] * 4, [100] * 4, 0.002, 'rnd')], {
        first: [cost([[-0.4 - 0.01 * seed, -0.3] for seed in seeds], [90] * 4, 0.0005, 'a'), cost([[-0.9, -0.9]] * 4, [90] * 4, 0.002, 'a')],
        second: [cost([[-0.5, -0.4]] * 4, [90] * 4, 0.002, 'b'), cost([[-0.9, -0.9]] * 4, [90] * 4, 0.0005, 'b')]})
    # PointMaze Large with the step cost: two of the baseline's four seeds are in the worse group (W -0.9), one of the
    # first finalist's and two of the second's.
    split = lambda worse: [[-0.9, -0.9] if seed < worse else [-0.7, -0.7] for seed in seeds]
    large = case('PointMaze Large, step cost', [cost(split(2), [100] * 4, 4.0, 'rnd')], {
        first: [cost(split(1), [100] * 4, 1.0, 'a')], second: [cost(split(2), [100] * 4, 4.0, 'b')]})
    cases = {module.MEDIUM_ARRIVAL: arrival, module.MEDIUM_STEP_COST: medium, module.LARGE_STEP_COST: large}
    screen = batch_summary.Screen(batch='made up', control=None, order=(), names={}, units=(), methods={}, differences={},
                                  discovery_differences={}, budget=None, seeds=(), cases=cases)
    assert verdicts(module, 1, screen) == [True, True]
    assert verdicts(module, 2, screen) == [False]
    assert verdicts(module, 3, screen) == [False]
    rows = batch_summary.evaluate(module.PREDICTIONS[0]['parts'], screen)
    assert 'all three parts -0.05 [-0.05, -0.05]; slower predictor with held divisor -0.11 [-0.11, -0.11]' in rows[0][2]
    # W of the first finalist: 0.75, 0.76, 0.77 and 0.78 by seed, against the baseline's 0.5.
    assert 'all three parts $W$ 0.765, difference from the baseline +0.265 [' in rows[1][2]
    assert 'all three parts: best intrinsic weight 0.0005, the smallest of its grid' in module.smallest_weight(screen)
    assert 'slower predictor with held divisor: best intrinsic weight 0.002, not the smallest of its grid' in module.smallest_weight(screen)
    shares = batch_summary.evaluate(module.PREDICTIONS[2]['parts'], screen)[0][2]
    assert shares.endswith('all three parts 25%; slower predictor with held divisor 50%; baseline 50%.')
    assert module.worse_share(large['baseline'].groups[0].policies, 200) == 0.5

    # A case whose unit is not complete cannot be read yet, and says which unit is waited for.
    cases[module.LARGE_STEP_COST] = case('PointMaze Large, step cost', [cost(split(2), [100] * 4, 4.0, 'rnd')],
                                         {first: [cost(split(1), [100] * 4, 1.0, 'a')]})
    waiting = batch_summary.evaluate(module.PREDICTIONS[2]['parts'], screen)
    assert waiting[0][1] is None and waiting[0][2] == (
        'Waiting for complete units in PointMaze Large, step cost: slower predictor with held divisor.')
    assert module.large_step_cost_beside(screen) == waiting[0][2]
    # Without a baseline configuration nothing can be compared either.
    cases[module.MEDIUM_STEP_COST] = case('PointMaze Medium, step cost', [], {first: [cost([[-0.4, -0.3]] * 4, [90] * 4, 0.0005, 'a')],
                                                                             second: [cost([[-0.5, -0.4]] * 4, [90] * 4, 0.002, 'b')]})
    assert verdicts(module, 2, screen) == [None]
    assert '(the baseline has no complete configuration either)' in batch_summary.evaluate(module.PREDICTIONS[1]['parts'], screen)[0][2]
