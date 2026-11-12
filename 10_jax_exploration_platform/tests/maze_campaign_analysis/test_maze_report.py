"""Tests of the generated report on made-up run folders: sweep tables, the rollout-shape check, verdicts, files."""

import json
from pathlib import Path
import shutil

import pytest

from maze_record_examples import curve_windows, goal_windows, start_record, write_shard
from maze_campaign.analysis import contents, records, report

ENVIRONMENTS = (('pointmaze', 'medium', 400), ('pointmaze', 'large', 400), ('antmaze', 'umaze', 700),
                ('antmaze', 'medium', 1000), ('antmaze', 'large', 1000))
BOTH_REWARDS = ('int_coef', 'ext_reward_scale', 'ext_reward_offset')


def reward_cells(weights, step_limit):
    """Every intrinsic weight once per reward setting: reward 1 on arrival, then the step cost."""
    # before: weights (1.0,), limit 400; after: ((1.0, 1.0, 0.0), (1.0, 0.0025, -0.0025)).
    return tuple((weight, scale, offset) for weight in weights
                 for scale, offset in ((1.0, 0.0), (1.0 / step_limit, -1.0 / step_limit)))


def write_study(run):
    """A made-up study: an RND sweep on four development seeds, then RND and a new method on eight fresh seeds.

    Every policy reaches the goal in all episodes from one window on. On the fresh seeds the new method reaches it
    two or three windows before RND in four environments, and in the same window as RND in AntMaze Large.
    """
    sweep_goal = {0.25: [None, 6, None, 6], 1.0: [1, 2, 1, 2], 4.0: [2, 2, 3, 2]}
    baseline_goal, method_goal = [3, 3, 4, 4, 3, 4, 3, 4], [1, 2, 1, 2, 1, 2, 1, 2]
    fresh = tuple(range(30000, 30008))
    for family, map_name, limit in ENVIRONMENTS:
        common = dict(family=family, map_name=map_name, step_limit=limit, iterations=8, names=BOTH_REWARDS,
                      trainer={'learning_rate': 1e-4})
        sweep = start_record(f'{family}-{map_name}_sweep', cells=reward_cells(sweep_goal, limit), seeds=(0, 1, 2, 3), **common)
        write_shard(run, [sweep] + goal_windows(sweep, [first for weight in sweep_goal for _ in range(2)
                                                        for first in sweep_goal[weight]]))
        baseline = start_record(f'{family}-{map_name}_baseline', cells=reward_cells((1.0,), limit), seeds=fresh, **common)
        write_shard(run, [baseline] + goal_windows(baseline, baseline_goal * 2))
        method = start_record(f'{family}-{map_name}_even-start', cells=reward_cells((1.0,), limit), seeds=fresh,
                              bonus='rnd_even_start', **common)
        goal = baseline_goal if (family, map_name) == ('antmaze', 'large') else method_goal
        write_shard(run, [method] + goal_windows(method, goal * 2))
    (run / 'report_settings.json').write_text(json.dumps({
        'title': 'Made-up study', 'methods': [{'name': 'even start', 'bonus': 'rnd_even_start'}]}))


def test_report_on_a_made_up_study(tmp_path, request):
    """The whole report: the sweep chooses the middle weight, the method beats RND in four of five environments."""
    kept = request.config.getoption('--example-study-dir')
    run = Path(kept).resolve() / 'made-up-study' if kept else tmp_path / 'made-up-study'
    if run.exists():
        shutil.rmtree(run)
    write_study(run)
    result = report.generate(run)
    assert list(result['sections']) == [
        'measures', 'records-read', 'development-sweeps', 'budget', 'training-curves',
        'comparison-with-the-baseline-on-the-confirmation-seeds', 'final-level-against-the-training-length',
        'decline-check']
    assert set(result['sections'].values()) == {'unread'}
    text = (run / 'analysis/analysis.md').read_text()
    assert text.startswith('# Made-up study\n\nGenerated ') and ' PT; times are Pacific' in text
    assert '15 units, 15 complete; 280 policies' in text

    # Development sweep, PointMaze Medium with reward 1 on arrival: W is 0.8125, 0.71875 and 0.125 for the weights
    # 1, 4 and 0.25, so the rows come in that order and the weight 1 is chosen; it is not at an end of the grid.
    sweep = text.split('### PointMaze Medium, reward 1 on arrival, CleanRL PPO-RND\n')[1].split('###')[0]
    rows = [line for line in sweep.splitlines() if line.startswith('| intrinsic weight')]
    assert [line.split(' | ')[0] for line in rows] == ['| intrinsic weight 1', '| intrinsic weight 4', '| intrinsic weight 0.25']
    assert rows[0].startswith('| intrinsic weight 1 | **0.812** ± 0.036 | **1.000** ± 0.000 | ')
    assert rows[1].startswith('| intrinsic weight 4 | <u>0.719</u> ± 0.031 | **1.000** ± 0.000 | ')
    assert rows[2].startswith('| intrinsic weight 0.25 | 0.125 ± 0.072 | <u>0.500</u> ± 0.289 | ') and rows[2].endswith('| 4 |')
    assert '3 configurations with 4 seeds each, compared at 800 environment steps per policy.' in sweep
    assert ('Chosen (highest $W$, ties by $L$): **intrinsic weight 1, learning rate 0.0001, entropy coefficient '
            '0.001**. Place in the sweep: intrinsic weight 1 is inside its 3 swept values.\n') in sweep
    assert 'edge rule' not in sweep and '**Unfinished runs:**' not in text

    # Budget rule, same case: the three weights reach 95% of their final level at steps 300, 400 and 700 of 800,
    # so twice the largest rounds up to 2,048 and the cap at the run's length applies.
    budget = text.split('## Budget')[1].split('\n## ')[0].split('### PointMaze Medium, reward 1 on arrival\n')[1].split('###')[0]
    assert '| intrinsic weight 1 | 0.812 ± 0.036 | 1.000 | 0.950 | 300 |' in budget
    assert '| intrinsic weight 4 | 0.719 ± 0.031 | 1.000 | 0.950 | 400 |' in budget
    assert '| intrinsic weight 0.25 | 0.125 ± 0.072 | 0.500 | 0.475 | 700 |' in budget
    assert ("Proposed budget: **800 steps per policy**, the run's own length. Twice the largest step count is 1,400, "
            'which rounds up to 2,048, above the 800 steps of the run, so the cap applies.') in budget
    assert 'the measures at the budget are those of the sweep table' in budget and 'recomputed' not in budget
    assert [proposal['budget'] for proposal in result['report'].budgets] == [800] * 10

    # Comparison, same case: W is 0.8125 for the method and 0.5625 for RND; every seed's L is 1 for both.
    comparison = text.split('## Comparison with the baseline on the confirmation seeds')[1].split('\n## ')[0]
    case = comparison.split('### PointMaze Medium, reward 1 on arrival\n')[1].split('###')[0]
    method_row, baseline_row = [line for line in case.splitlines() if line.startswith(('| even start', '| CleanRL'))][:2]
    assert method_row.startswith('| even start | **0.812** ± 0.024 | +0.250 [')
    assert method_row.endswith(', ahead | 1.000 ± 0.000 | +0.000 [+0.000, +0.000], level | beats | 8 |')
    assert baseline_row.startswith('| CleanRL PPO-RND | <u>0.562</u> ± 0.024 | baseline | 1.000 ± 0.000 | baseline | baseline | 8 |')
    # In AntMaze Large the method equals RND seed by seed: level in W, so it does not beat.
    large = comparison.split('### AntMaze Large, reward 1 on arrival\n')[1].split('###')[0]
    assert '+0.000 [+0.000, +0.000], level | 1.000 ± 0.000 | +0.000 [+0.000, +0.000], level | does not beat | 8 |' in large

    # The verdicts together, and the success rule: four of five in each reward setting is not success.
    summary = [line for line in comparison.splitlines() if line.startswith('| even start | ')][:2]
    assert summary[0].count('beats; $W$ +0.250, ahead; $L$ +0.000, level (8 seeds)') == 4
    for line in summary:
        assert line.count('| beats; $W$ +') == 4 and line.count(', ahead; $L$ +0.000, level (8 seeds)') == 4
        assert 'does not beat; $W$ +0.000, level; $L$ +0.000, level (8 seeds) | 4 of 5 |' in line
    assert '**Reward 1 on arrival.**' in comparison and '**Step cost.**' in comparison
    assert ('- **even start:** not met (reward 1 on arrival: beats in 4 of 5, not compared or unfinished in 0; '
            'step cost: beats in 4 of 5, not compared or unfinished in 0).') in comparison

    # Final level against the training length: at 25% of the steps no RND seed has arrived and half the method's
    # seeds have; at 50% half the RND seeds have.
    lengths = text.split('### PointMaze Medium, reward 1 on arrival, 800 planned steps (confirmation seeds)\n')[1].split('###')[0]
    assert '| CleanRL PPO-RND | 0.000 | 0.500 | 1.000 | 1.000 |' in lengths
    assert '| even start | 0.500; +0.500 [' in lengths
    assert '- **even start:** not behind the baseline in $L$ at any of the 4 lengths compared.' in lengths
    # No curve of this study falls: every chosen configuration has both ratios at 100% and no LaTeX decline table.
    decline = text.split('## Decline check')[1]
    assert ('both ratios of each of the 30 chosen configurations (the choice of every development sweep and every '
            'confirmation row): 0 below 90% of the smoothed peak, 0 below 90% of the highest single window only.') in decline
    assert decline.count('| 100.0% |') == 60 and decline.count('| no decline |') == 30
    assert 'tab:maze-decline-check' not in (run / 'analysis/results.tex').read_text()

    # Files: the LaTeX tables, the figures as PNG and PDF, and the page with colours.
    tex = (run / 'analysis/results.tex').read_text()
    assert tex.count('\\begin{table}') == tex.count('\\end{table}') > 20 and '%' not in tex.replace('\\%', '').replace('% ', '')
    assert '\\textbf{0.812} $\\pm$ 0.024 & +0.250 [' in tex and '\\MazeCampaignRunPath/analysis/plots/' in tex
    names = sorted(path.name for path in (run / 'analysis/plots').iterdir())
    assert names == sorted(f'training-curves_{seeds}_{reward}{method}.{kind}' for kind in ('pdf', 'png')
                           for reward in ('reward-1-on-arrival', 'step-cost')
                           for seeds, method in (('development-seeds', '_cleanrl-ppo-rnd'), ('confirmation-seeds', '')))
    assert 'class="unread"' in (run / 'analysis/analysis.html').read_text()

    # Marking as read and generating again from the same records leaves every section read.
    contents.mark_read(run / 'analysis', ['all'], 'Made-up study')
    assert set(report.generate(run)['sections'].values()) == {'read'}


def test_budget_section_recomputes_every_configuration_at_a_budget_below_the_run(tmp_path):
    """Runs of 3,200 steps whose three weights reach 95% at steps 300, 400 and 600: the budget is 2,048 steps."""
    sweep = start_record('sweep', iterations=32, cells=((1e-4, 1.0), (1e-4, 4.0), (1e-4, 16.0)))
    write_shard(tmp_path, [sweep] + goal_windows(sweep, [1, 1, 2, 2, 4, 4]))
    built = report.generate(tmp_path)['report']
    section = (tmp_path / 'analysis/analysis.md').read_text().split('## Budget')[1].split('\n## ')[0]
    # The smoothing range of 160 steps is full of the level 1 one window after the rise.
    assert '| intrinsic weight 1 | 0.969 ± 0.000 | 1.000 | 0.950 | 300 |' in section
    assert '| intrinsic weight 16 | 0.875 ± 0.000 | 1.000 | 0.950 | 600 |' in section
    assert ('Proposed budget: **2,048 steps per policy**: twice the largest step count is 1,200, rounded up to a '
            'power of two.') in section
    assert 'recomputed at 2,048 steps and sorted by $W$' in section and 'cap applies' not in section
    # At 2,048 steps the weight 1 has W = 1,948 / 2,048 and the weight 4 has 1,848 / 2,048.
    assert '| intrinsic weight 1 | **0.951** ± 0.000 | 1.000 ± 0.000 |' in section
    assert '| intrinsic weight 4 | <u>0.902</u> ± 0.000 | 1.000 ± 0.000 |' in section
    assert ('At the budget the highest $W$ belongs to **intrinsic weight 1**, the configuration that is also the '
            "best at the run's length, so the choice does not change.") in section
    assert built.budgets[0]['budget'] == 2048 and built.budgets[0]['capped'] is False
    assert built.budgets[0]['choice_changes'] is False
    assert '_measures-at-the-budget}' in (tmp_path / 'analysis/results.tex').read_text()


def test_budget_section_says_when_the_choice_changes_at_the_budget(tmp_path):
    """One weight is ahead early and falls to half; another rises later and stays higher: the best depends on the length."""
    early = [1.0] * 16 + [0.5] * 16
    late = [0.0] * 4 + [0.875] * 28
    low = [0.25] * 32
    start = start_record('sweep', iterations=32, cells=((1e-4, 1.0), (1e-4, 4.0), (3e-4, 4.0)))
    write_shard(tmp_path, [start] + curve_windows(start, [early, early, late, late, low, low], episodes=16))
    built = report.generate(tmp_path)['report']
    text = (tmp_path / 'analysis/analysis.md').read_text()
    section = text.split('## Budget')[1].split('\n## ')[0]
    # At 3,200 steps W is 0.766 for the late one and 0.75 for the early one, so the late one is chosen. Their
    # smoothed curves reach 95% of their final levels at steps 600 and 200, so the budget is 2,048 steps.
    assert '| intrinsic weight 4, learning rate 0.0001 | 0.766 ± 0.000 | 0.875 | 0.831 | 600 |' in section
    # The early one is at its level from the first window on: 200 is the first step with a full range of 160 steps.
    assert ('| intrinsic weight 1, learning rate 0.0001 | 0.750 ± 0.000 | 0.500 | 0.475 | 200 (the earliest step the '
            'rule can read) |') in section
    assert 'Every step count is the earliest' not in section and built.budgets[0]['every_step_is_the_earliest'] is False
    assert built.budgets[0]['budget'] == 2048 and built.budgets[0]['choice_changes'] is True
    # At 2,048 steps the early one has W = (1,600 + 224) / 2,048 and the late one 1,648 x 0.875 / 2,048.
    assert '| intrinsic weight 1, learning rate 0.0001 | **0.891** ± 0.000 |' in section
    assert '| intrinsic weight 4, learning rate 0.0001 | <u>0.704</u> ± 0.000 |' in section
    assert ('At the budget the highest $W$ belongs to **intrinsic weight 1, learning rate 0.0001**, while at the '
            "run's length it belongs to **intrinsic weight 4, learning rate 0.0001**: the choice changes. At the "
            'budget their $W$ are 0.891 ± 0.000 and 0.704 ± 0.000.') in section
    # The chosen settings' places in the sweep: two values each, so no edge rule for the intrinsic weight.
    assert ('Place in the sweep: intrinsic weight 4 is the largest of its 2 swept values; learning rate 0.0001 is '
            'the smallest of its 2 swept values.\n') in text


def test_budget_section_says_when_every_step_count_is_the_earliest_the_rule_can_read(tmp_path):
    """Curves that are at their level from the first window on give the rule's smallest budget, and the section says so."""
    sweep = start_record('sweep', iterations=32, cells=((1e-4, 1.0), (1e-4, 4.0), (1e-4, 16.0)))
    write_shard(tmp_path, [sweep] + goal_windows(sweep, [0, 0, 0, 0, 0, 0]))
    built = report.generate(tmp_path)['report']
    section = (tmp_path / 'analysis/analysis.md').read_text().split('## Budget')[1].split('\n## ')[0]
    assert section.count('| 200 (the earliest step the rule can read) |') == 3
    assert ('Proposed budget: **512 steps per policy**: twice the largest step count is 400, rounded up to a power '
            'of two. Every step count is the earliest the rule can read from this run (200: one full range of 5% of '
            'its steps), so the curves rise faster than this run can show, and a shorter run would give a smaller '
            'budget.') in section
    assert built.budgets[0]['every_step_is_the_earliest'] is True and built.budgets[0]['budget'] == 512


def test_budget_section_says_when_a_configuration_never_reaches_its_level(tmp_path):
    """A configuration that stays at the never-reached return counts as the whole run, and the section says so."""
    sweep = start_record('sweep', iterations=8, cells=((1e-4, 1.0), (1e-4, 4.0)))
    write_shard(tmp_path, [sweep] + goal_windows(sweep, [1, 1, None, None]))
    report.generate(tmp_path)
    section = (tmp_path / 'analysis/analysis.md').read_text().split('## Budget')[1].split('\n## ')[0]
    assert 'Only 2 configurations have trained the whole length, so the rule reads them.' in section
    assert '| intrinsic weight 4 | 0.000 ± 0.000 | 0.000 | 0.000 | never: its final level is $R_0$ |' in section
    assert ('Never at 95% of its final level: intrinsic weight 4. Such a configuration counts as the whole run, so '
            "the largest step count is the run's length.") in section
    assert "Proposed budget: **800 steps per policy**, the run's own length." in section


def test_decline_check_gives_the_ratio_with_the_smoothed_peak_and_with_the_highest_window(tmp_path, request):
    """A curve that falls is listed by the smoothed peak; one with a single high window by that window only."""
    kept = request.config.getoption('--example-study-dir')
    if kept:
        tmp_path = Path(kept).resolve() / 'decline-example'
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
    falls = [0.0, 0.0, 1.0] + [0.625] * 17
    one_high_window = [0.0, 0.0, 1.0, 0.75] + [0.875] * 16
    for map_name, curve in (('large', falls), ('medium', one_high_window)):
        start = start_record(f'pointmaze-{map_name}', map_name=map_name, envs=2, iterations=20)
        write_shard(tmp_path, [start] + curve_windows(start, [curve, curve], episodes=16))
    report.generate(tmp_path)
    text = (tmp_path / 'analysis/analysis.md').read_text()
    section = text.split('## Decline check')[1]
    name = 'CleanRL PPO-RND, intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.001'
    # Windows of 50 steps, a smoothing range of 100: the range with the windows at 1 and 0.625 has the mean 0.8125.
    assert (f'| PointMaze Large, reward 1 on arrival | development | {name} | 0.625 | 0.812 at 200 | 76.9% | '
            '1.000 at 150 | 62.5% | declines: below 90% of the smoothed peak | 1.0K |') in section
    # The range with the windows at 1 and 0.75 has the mean 0.875, the final level: no decline by the smoothed peak.
    assert (f'| PointMaze Medium, reward 1 on arrival | development | {name} | 0.875 | 0.875 at 200 | 100.0% | '
            '1.000 at 150 | 87.5% | below 90% of the highest single window only | 1.0K |') in section
    assert ('both ratios of each of the 2 chosen configurations (the choice of every development sweep and every '
            'confirmation row): 1 below 90% of the smoothed peak, 1 below 90% of the highest single window only.') in section
    # The sweep table's column is the ratio with the smoothed peak.
    assert '| 0.625 ± 0.000 | 76.9% |' in text and '\\label{tab:maze-decline-check}' in (tmp_path / 'analysis/results.tex').read_text()


def test_rows_that_have_not_reached_the_step_count_are_listed_under_the_table(tmp_path):
    """With one finished row the table is read at the planned steps; an unfinished row is listed, not ranked."""
    done = start_record('finished', cells=((1e-4, 1.0), (1e-4, 4.0)))
    write_shard(tmp_path, [done] + goal_windows(done, [1, 1, 2, 2]))
    late = start_record('unfinished', cells=((1e-4, 16.0),))
    write_shard(tmp_path, [late] + goal_windows(late, [0, 0])[:2], complete=False)
    report.generate(tmp_path)
    text = (tmp_path / 'analysis/analysis.md').read_text()
    assert '2 configurations with 2 seeds each, compared at 400 environment steps per policy.' in text
    assert 'Not at this step count yet:\n\n- intrinsic weight 16: 200 steps so far.\n' in text
    assert '**Unfinished runs:**' not in text and '2 units, 1 complete; 6 policies' in text
    # The chosen weight 1 is the smallest of the three swept values, so the edge rule is named.
    assert ('Place in the sweep: intrinsic weight 1 is the smallest of its 3 swept values. The intrinsic weight is '
            'at an end of its grid, so the edge rule of the protocol asks for three more values on that side') in text


def test_a_table_of_unfinished_rows_is_read_at_the_steps_every_policy_has(tmp_path):
    """When no row has reached its planned steps, all rows are compared at the largest common step count."""
    first = start_record('first', cells=((1e-4, 1.0),))
    write_shard(tmp_path, [first] + goal_windows(first, [0, 0])[:3], complete=False)
    second = start_record('second', cells=((1e-4, 4.0),))
    write_shard(tmp_path, [second] + goal_windows(second, [1, 1])[:2], complete=False)
    built = report.generate(tmp_path)['report']
    text = (tmp_path / 'analysis/analysis.md').read_text()
    assert 'compared at 200 environment steps per policy. **Unfinished runs:** the measures are read at 200 of the 400 planned' in text
    assert built.evaluation(built.groups['development'])[0] == 200
    # A step count given on the command line replaces that choice; the row that has not reached it is listed.
    report.generate(tmp_path, at_steps=300)
    assert '- intrinsic weight 4: 200 steps so far.' in (tmp_path / 'analysis/analysis.md').read_text()


def test_rollout_shape_check_names_the_best_number_of_environments_per_policy(tmp_path):
    """Widths 4, 8 and 16 with different logging windows are compared at equal steps; the best W decides."""
    cells = ((1e-4, 1.0), (1e-4, 4.0))
    # Every width trains 400 steps per policy; its windows are 100, 200 and 80 steps long.
    four = start_record('width-4', envs=4, rollout=5, iterations=20, window_iterations=5, cells=cells)
    write_shard(tmp_path, [four] + goal_windows(four, [None, None, 2, 3]))
    eight = start_record('width-8', envs=8, rollout=5, iterations=10, window_iterations=5, cells=cells)
    write_shard(tmp_path, [eight] + goal_windows(eight, [1, 1, 0, 1]))
    sixteen = start_record('width-16', envs=16, rollout=5, iterations=5, cells=cells)
    write_shard(tmp_path, [sixteen] + goal_windows(sixteen, [4, None, 2, 2]))
    report.generate(tmp_path)
    text = (tmp_path / 'analysis/analysis.md').read_text()
    section = text.split('## Rollout shape: environments per policy')[1].split('\n## ')[0]
    rows = [line for line in section.splitlines() if line.startswith(('| 4 |', '| 8 |', '| 16 |'))]
    # Best W per width: 0.75 (8, weight 4), 0.6 (16, weight 4), 0.375 (4, weight 4); mean W over both weights.
    # Every width's best configuration ends at L = 1 and reaches the goal on both seeds, so those columns have no mark.
    assert rows[0] == '| 8 | **0.750** ± 0.250 | intrinsic weight 4 | 1.000 ± 0.000 | **0.625** | 100.0% | 400 |'
    assert rows[1].startswith('| 16 | <u>0.600</u> ± 0.000 | intrinsic weight 4 | 1.000 ± 0.000 | <u>0.350</u> |')
    assert rows[2].startswith('| 4 | 0.375 ± 0.125 | intrinsic weight 4 |')
    assert 'Best at these steps: **8 environments per policy** ($W$ = 0.750, intrinsic weight 4).' in section
    assert 'compared at the same 400 environment steps per policy, with 2 seeds per configuration.' in section
    # Every configuration: widths in rows, intrinsic weights in columns, the best W of each column bold.
    assert rows[3:] == [
        '| 4 | $W$ 0.000 ± 0.000; $L$ 0.000 ± 0.000 | $W$ 0.375 ± 0.125; $L$ 1.000 ± 0.000 |',
        '| 8 | $W$ **0.500** ± 0.000; $L$ 1.000 ± 0.000 | $W$ **0.750** ± 0.250; $L$ 1.000 ± 0.000 |',
        '| 16 | $W$ <u>0.100</u> ± 0.100; $L$ 0.500 ± 0.500 | $W$ <u>0.600</u> ± 0.000; $L$ 1.000 ± 0.000 |']
    assert 'runs still training' not in section
    # The sweep table names each row by its width and weight, and the figure is written.
    assert '| 8 environments per policy, intrinsic weight 4 | **0.750** ± 0.250 |' in text
    assert (tmp_path / 'analysis/plots/rollout-shape_pointmaze-large_reward-1-on-arrival_cleanrl-ppo-rnd.png').exists()


def test_rollout_shape_check_with_a_width_still_training(tmp_path):
    """Finished widths are compared at the planned steps, and all widths again at the steps every policy has."""
    cells = ((1e-4, 1.0), (1e-4, 4.0))
    four = start_record('width-4', envs=4, rollout=5, iterations=20, window_iterations=5, cells=cells)
    write_shard(tmp_path, [four] + goal_windows(four, [None, None, 2, 3]))
    one = start_record('width-1', envs=1, rollout=5, iterations=80, window_iterations=8, cells=cells)
    write_shard(tmp_path, [one] + goal_windows(one, [0, 0, 1, 1])[:6], complete=False)
    report.generate(tmp_path)
    section = (tmp_path / 'analysis/analysis.md').read_text().split('## Rollout shape: environments per policy')[1].split('\n## ')[0]
    assert 'Best at these steps: **4 environments per policy** ($W$ = 0.375, intrinsic weight 4).' in section
    assert '- 1 environment per policy, intrinsic weight 1: 240 steps so far.' in section
    # At 240 steps the width 1 has W = 1 at the weight 1 (goal from its first window on) and is the best.
    assert ('All numbers of environments per policy at the 240 environment steps that every policy has reached. '
            '**Unfinished runs:** the measures are read at 240 of the 400 planned steps per policy') in section
    assert ('Best at these steps: **1 environment per policy** ($W$ = 1.000, intrinsic weight 1). The runs are '
            'unfinished, so this is not the check\'s result yet.') in section
    assert 'A line that stops early is a run still training.' in section
    tex = (tmp_path / 'analysis/results.tex').read_text()
    assert '_runs-still-training-included_every-configuration}' in tex and 'unfinished runs' in tex


def test_cells_for_latex_and_small_helpers():
    """Markdown marks, signs and symbols become LaTeX; formulas are kept; best and second marks handle ties."""
    assert report.latex_text('**0.412** ± 0.031') == '\\textbf{0.412} $\\pm$ 0.031'
    assert report.latex_text('<u>97.5%</u>') == '\\underline{97.5\\%}'
    assert report.latex_text('-0.021 [-0.040, +0.002], level') == '$-$0.021 [$-$0.040, +0.002], level'
    assert report.latex_text('whole-curve mean $W$ ↑, seeds 0-15') == 'whole-curve mean $W$ $\\uparrow$, seeds 0-15'
    assert report.column_marks([0.5, 0.7, 0.7, None], ['0.5', '0.7', '0.7', 'none'], True) == ['second', 'best', 'best', None]
    assert report.column_marks([3.0, 1.0], ['3', '1'], False) == ['second', 'best']
    assert report.column_marks([0.5, 0.5], ['0.5', '0.5'], True) == [None, None]
    # Values that are shown as the same number share a mark, whatever their further digits.
    assert report.mean_marks([0.00004, 0.0, 0.069]) == ['second', 'second', 'best']
    assert [report.compact_steps(value) for value in (950, 387100, 1480000)] == ['950', '387.1K', '1.48M']
    assert report.interval_text({'mean': 0.021, 'interval': [0.004, 0.038], 'side': 'ahead'}) == '+0.021 [+0.004, +0.038], ahead'
    table = report.latex_table(['row', '$W$ ↑'], [['a', '**0.5**'], ['b', '0.4']], 'Caption with 95%.', 'example')
    assert table.count('\\\\\n') == 2 and '\\caption{Caption with 95\\%.}\\label{tab:maze-example}' in table
    assert 'a & \\textbf{0.5} \\\\' in table and table.count('p{') == 2


def test_a_run_folder_without_a_window_is_refused(tmp_path):
    """A report needs at least one logging window; a folder without shards is refused as well."""
    with pytest.raises(ValueError, match='holds no shard'):
        report.generate(tmp_path)
    start = start_record('just-started')
    write_shard(tmp_path, [start], complete=False)
    with pytest.raises(ValueError, match='has written a logging window yet'):
        report.generate(tmp_path)
    assert records.read_report_settings(tmp_path) == {}


def write_repeated_run(run):
    """A sweep over the weights 1 and 4, and a unit that repeats the weight 4 on the same two seeds and finished first."""
    sweep = start_record('sweep', cells=((1e-4, 1.0), (1e-4, 4.0)))
    write_shard(run, [sweep] + goal_windows(sweep, [0, 0, 1, 1]), finished='2026-11-12T13:00:00-05:00')
    again = start_record('zz-repeat_weight-4', cells=((1e-4, 4.0),))
    write_shard(run, [again] + goal_windows(again, [2, 3]), finished='2026-11-12T09:30:00-08:00')


def test_a_repeated_configuration_is_an_error_unless_the_report_settings_allow_it(tmp_path):
    """Without the key the report refuses the folder and names the seed and both units."""
    write_repeated_run(tmp_path)
    with pytest.raises(ValueError, match="Seed 0 was trained more than once for one configuration, in units "
                                         "\\['sweep', 'zz-repeat_weight-4'\\]"):
        report.generate(tmp_path)
    (tmp_path / 'report_settings.json').write_text(json.dumps({'repeated_configurations': 'hide'}))
    with pytest.raises(ValueError, match="repeated_configurations is 'hide'; the only value is \"show\""):
        report.generate(tmp_path)


def test_a_repeated_configuration_uses_the_unit_that_finished_first_and_gets_a_section(tmp_path):
    """The repeat finished at 09:30 PT and the sweep at 10:00 PT, so the tables hold the repeat's weight 4."""
    write_repeated_run(tmp_path)
    (tmp_path / 'report_settings.json').write_text(json.dumps({'repeated_configurations': 'show'}))
    result = report.generate(tmp_path)
    text = (tmp_path / 'analysis/analysis.md').read_text()
    assert list(result['sections'])[:3] == ['measures', 'records-read', 'configurations-trained-more-than-once']
    assert '2 units, 2 complete; 6 policies with at least one logging window.' in text
    assert ' 2 of these policies repeat a configuration and seed that another unit finished earlier;' in text

    # The sweep table: W of the weight 4 is 0.375 (the repeat: goal from windows 2 and 3 of 4), not the sweep's 0.75.
    sweep = text.split('## Development sweeps')[1].split('\n## ')[0]
    assert '| intrinsic weight 1 | **1.000** ± 0.000 |' in sweep and '| intrinsic weight 4 | <u>0.375</u> ± 0.125 |' in sweep
    assert '0.750' not in sweep and '2 configurations with 2 seeds each' in sweep

    # The section: both runs side by side, the one that finished first in front, and the paired difference
    # 0.75 - 0.5 and 0.75 - 0.25 of the later run minus the one that is used.
    section = text.split('## Configurations trained more than once\n')[1].split('\n## ')[0]
    assert 'uses the unit that finished first' in section
    assert ('### PointMaze Large, reward 1 on arrival, CleanRL PPO-RND, intrinsic weight 4, learning rate 0.0001, '
            'entropy coefficient 0.001, trained for 400 steps\n\nRead at 400 environment steps per policy.') in section
    rows = [line for line in section.splitlines() if line.startswith('| unit')]
    assert rows == ['| unit 2: zz repeat, weight 4 | 2026-11-12 09:30 PT | yes | 0.375 ± 0.125 | 1.000 ± 0.000 | 2 |',
                    '| unit 1: sweep | 2026-11-12 10:00 PT | no | 0.750 ± 0.000 | 1.000 ± 0.000 | 2 |']
    assert ('Unit 1 minus unit 2, on the 2 seeds both have: in $W$ +0.375 [+0.250, +0.500] (the interval does not '
            'contain zero); in $L$ +0.000 [+0.000, +0.000] (the interval contains zero).') in section
    assert 'Configuration trained more than once: PointMaze Large' in (tmp_path / 'analysis/results.tex').read_text()
