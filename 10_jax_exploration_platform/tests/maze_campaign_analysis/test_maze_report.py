"""Tests of the generated report on made-up run folders: sweep tables, the rollout-shape check, verdicts, files."""

import json
from pathlib import Path
import shutil

import pytest

from maze_record_examples import goal_windows, start_record, write_shard
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
        'measures', 'records-read', 'development-sweeps', 'training-curves',
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
    assert 'Chosen (highest $W$, ties by $L$): **intrinsic weight 1, learning rate 0.0001, entropy coefficient 0.001**.\n' in sweep
    assert 'edge rule' not in sweep and '**Unfinished runs:**' not in text

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
    assert 'Checked: 30 chosen configurations (' in text and 'None is below 90%.' in text

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
    assert 'Its intrinsic weight is the smallest of the 3 swept values' in text


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
