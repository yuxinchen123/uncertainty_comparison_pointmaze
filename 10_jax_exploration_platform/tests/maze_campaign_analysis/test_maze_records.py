"""Tests of reading shards: policies by name, complete and unfinished units, a resumed unit, a restarted unit."""

import json

import pytest

from maze_record_examples import goal_windows, start_record, window_record, write_shard
from maze_campaign.analysis import measures, records


def test_a_unit_gives_one_policy_per_copy_with_its_settings_by_name(tmp_path):
    """Two cells of two seeds: four policies, each with its cell's values, its seed and its window columns."""
    start = start_record('unit-a', cells=((1e-4, 0.0625), (1e-4, 16.0)), seeds=(0, 1))
    windows = goal_windows(start, [None, 2, 1, 0])
    summary, policies = records.read_unit(write_shard(tmp_path, [start] + windows))
    assert summary['state'] == 'complete' and summary['windows'] == 4 and summary['steps_reached'] == 400
    assert summary['planned_steps'] == 400 and summary['reward_settings'] == ['arrival_plus_one']
    assert [(dict(policy.settings)['int_coef'], policy.seed) for policy in policies] == [
        (0.0625, 0), (0.0625, 1), (16.0, 0), (16.0, 1)]
    last = policies[3]
    assert (last.environment, last.reward_setting, last.method) == ('PointMaze Large', 'arrival_plus_one', 'CleanRL PPO-RND')
    assert last.environments_per_policy == 4 and last.step_limit == 400 and last.stage == 'development'
    # The cell's learning rate replaces the unit's own 3e-4; seeds, shapes and switches are not settings.
    settings = dict(last.settings)
    assert settings['learning_rate'] == 1e-4 and settings['ent_coef'] == 0.001 and settings['num_steps'] == 25
    assert settings['lr_anneal_fraction_of_run'] == 1.0
    assert not {'n_copies', 'n_envs', 'base_seed', 'episode_log', 'ext_reward_scale', 'setting_names'} & set(settings)
    # Policy 3 reaches the goal from window 0 on; policy 0 never does.
    assert last.steps.tolist() == [100, 200, 300, 400] and last.arrivals.tolist() == [10, 10, 10, 10]
    assert last.first_goal_step == 8 and policies[0].first_goal_step == -1
    assert policies[1].return_sum.tolist() == [0, 0, 10, 10] and policies[1].coverage.tolist() == [0.25, 0.25, 1.0, 1.0]
    assert measures.whole_curve_mean(policies[1], 400) == 0.5


def test_policies_group_into_configurations_in_seed_order(tmp_path):
    """Policies that differ only by the seed form one configuration; a seed trained twice is refused."""
    start = start_record('unit-a', cells=((1e-4, 0.0625), (1e-4, 16.0)), seeds=(1, 0))
    _, policies = records.read_unit(write_shard(tmp_path, [start] + goal_windows(start, [None] * 4)))
    groups = records.configurations(policies)
    assert len(groups) == 2 and [[policy.seed for policy in group.policies] for group in groups] == [[0, 1], [0, 1]]
    assert records.varying_settings(groups) == ['int_coef']
    assert records.describe_settings(groups[1], ['int_coef', 'learning_rate']) == 'intrinsic weight 16, learning rate 0.0001'
    again = start_record('unit-b', cells=((1e-4, 16.0),), seeds=(0,))
    _, more = records.read_unit(write_shard(tmp_path, [again] + goal_windows(again, [None])))
    with pytest.raises(ValueError, match='trained more than once'):
        records.configurations(policies + more)


def test_an_unfinished_unit_and_a_line_still_being_written(tmp_path):
    """A unit without its completion record is unfinished; a last line with no line end is left out."""
    start = start_record('unit-a')
    path = write_shard(tmp_path, [start] + goal_windows(start, [0, None])[:2], complete=False)
    with path.open('a') as handle:
        handle.write('{"record": "episode_window", "unit_id": "unit-a", "window_in')
    summary, policies = records.read_unit(path)
    assert summary['state'] == 'unfinished' and summary['windows'] == 2 and summary['steps_reached'] == 200
    assert not policies[0].complete and policies[0].steps.tolist() == [100, 200]
    with pytest.raises(ValueError, match='cannot be read'):
        measures.whole_curve_mean(policies[0], 400)


def test_a_unit_resumed_from_a_checkpoint_keeps_one_sequence_of_windows(tmp_path):
    """After a resume record the windows continue where the kept ones stop, and the unit reads as one run."""
    start = start_record('unit-a')
    windows = goal_windows(start, [1, None])
    resumed = [start] + windows[:2] + [{'record': 'job', 'unit_id': 'unit-a', 'resumed_from_iteration': 2},
                                       {'record': 'unit_resume', 'unit_id': 'unit-a', 'resumed_from_iteration': 2,
                                        'records_kept': 4}] + windows[2:]
    summary, policies = records.read_unit(write_shard(tmp_path, resumed))
    assert summary['state'] == 'complete' and summary['continued_from_a_checkpoint'] == 1
    assert summary['started_from_zero_again'] == 0 and summary['windows'] == 4
    _, uninterrupted = records.read_unit(write_shard(tmp_path / 'other', [start] + windows))
    for one, other in zip(policies, uninterrupted):
        assert one.steps.tolist() == other.steps.tolist() and one.return_sum.tolist() == other.return_sum.tolist()
        assert measures.whole_curve_mean(one, 400) == measures.whole_curve_mean(other, 400)
    assert measures.whole_curve_mean(policies[0], 400) == 0.75


def test_a_unit_started_again_from_zero_is_read_from_its_last_start(tmp_path):
    """Records before the last start record belong to an abandoned attempt and are not read."""
    start = start_record('unit-a')
    abandoned, latest = goal_windows(start, [0, 0]), goal_windows(start, [None, 2])
    path = write_shard(tmp_path, [start] + abandoned[:3] + [{'record': 'job', 'unit_id': 'unit-a'}, start] + latest[:3],
                       complete=False)
    summary, policies = records.read_unit(path)
    assert summary['started_from_zero_again'] == 1 and summary['windows'] == 3 and summary['state'] == 'unfinished'
    assert policies[0].arrivals.tolist() == [0, 0, 0] and policies[1].arrivals.tolist() == [0, 0, 10]


def test_windows_that_are_not_consecutive_are_refused(tmp_path):
    """A missing window would shift every step count, so the reader stops."""
    start = start_record('unit-a')
    windows = goal_windows(start, [None, None])
    with pytest.raises(ValueError, match='not consecutive'):
        records.read_unit(write_shard(tmp_path, [start, windows[0], windows[2]]))


def test_a_unit_without_the_episode_log_or_with_a_continuing_task_is_refused(tmp_path):
    """The measures need the per-window episode sums and episodes that end at the goal."""
    start = start_record('unit-a')
    bare = {key: value for key, value in window_record(start, 0).items() if key != 'arrivals_per_copy'}
    with pytest.raises(ValueError, match='no episode log'):
        records.read_unit(write_shard(tmp_path, [start, bare]))
    continuing = start_record('unit-b')
    continuing['environment_config']['continuing_task'] = True
    with pytest.raises(ValueError, match='end at the goal'):
        records.read_unit(write_shard(tmp_path, [continuing, window_record(continuing, 0)]))


def test_both_reward_settings_in_one_unit_are_told_apart_per_policy(tmp_path):
    """Cells that hold the reward scale and offset give policies of both settings; the step cost returns are negative."""
    names = ('int_coef', 'ext_reward_scale', 'ext_reward_offset')
    start = start_record('unit-a', names=names, cells=((1.0, 1.0, 0.0), (1.0, 1 / 400, -1 / 400)), seeds=(0,))
    summary, policies = records.read_unit(write_shard(tmp_path, [start] + goal_windows(start, [1, 1])))
    assert summary['reward_settings'] == ['arrival_plus_one', 'step_cost']
    arrival, cost = policies
    assert (arrival.reward_setting, cost.reward_setting) == ('arrival_plus_one', 'step_cost')
    assert arrival.never_reached_return == 0.0 and cost.never_reached_return == -1.0
    # Window 0: ten episodes of 400 steps return -1 each; later an arrival after 40 steps returns -(40 - 1) / 400.
    assert measures.training_curve(cost).tolist() == pytest.approx([-1.0] + [-39 / 400] * 3)
    assert measures.training_curve(arrival).tolist() == [0.0, 1.0, 1.0, 1.0]
    # The two settings never share a configuration, though every other setting is equal.
    assert len(records.configurations(policies)) == 2 and arrival.settings == cost.settings
    with pytest.raises(ValueError, match='neither'):
        records.reward_setting_of(0.5, 0.0, 400)


def test_a_run_folder_is_read_with_its_report_settings(tmp_path):
    """Methods are named by the settings file, another folder's shards join, and environments must agree."""
    run, other = tmp_path / 'run', tmp_path / 'baseline'
    new = start_record('new-method_seeds-30000-30001', bonus='rnd_even_start', seeds=(30000, 30001))
    write_shard(run, [new] + goal_windows(new, [0, 0]))
    base = start_record('baseline_seeds-30000-30001', seeds=(30000, 30001))
    write_shard(other, [base] + goal_windows(base, [2, 2]))
    defaults = start_record('baseline-at-cleanrl-defaults_seeds-30000-30001', seeds=(30000, 30001), cells=((1e-4, 2.0),))
    write_shard(other, [defaults] + goal_windows(defaults, [3, 3]), complete=False)
    (run / 'report_settings.json').write_text(json.dumps({
        'other_run_folders': [str(other)],
        'methods': [{'name': 'even start', 'bonus': 'rnd_even_start'},
                    {'name': 'CleanRL PPO-RND at CleanRL defaults', 'unit_id_pattern': '*cleanrl-defaults*',
                     'reference': True}]}))
    loaded = records.load_run(run)
    assert sorted({policy.method for policy in loaded.policies}) == [
        'CleanRL PPO-RND', 'CleanRL PPO-RND at CleanRL defaults', 'even start']
    assert loaded.baseline_method == 'CleanRL PPO-RND'
    assert loaded.reference_methods == {'CleanRL PPO-RND at CleanRL defaults'}
    assert [unit['state'] for unit in loaded.units] == ['complete', 'unfinished', 'complete']
    assert all(policy.stage == 'confirmation' for policy in loaded.policies)
    # A unit of another step limit under the same environment name cannot be pooled.
    odd = start_record('odd-step-limit', step_limit=600, seeds=(5,))
    write_shard(run, [odd] + goal_windows(odd, [None]))
    with pytest.raises(ValueError, match='different environment configurations'):
        records.load_run(run)
    (run / 'report_settings.json').write_text(json.dumps({'budget': 400}))
    with pytest.raises(ValueError, match='unknown keys'):
        records.load_run(run)


def test_a_unit_is_named_by_its_specification_file_and_can_be_left_out(tmp_path):
    """The fleet's specification file names the method; report settings can rename it or leave a unit out."""
    for unit, method in (('confirmation_baseline', 'cleanrl-ppo-rnd'), ('confirmation_new', 'even-start'),
                         ('confirmation_defaults', 'cleanrl-ppo-rnd-at-cleanrl-defaults')):
        start = start_record(unit, seeds=(30000, 30001), cells=((1e-4, 2.0 if 'defaults' in unit else 1.0),))
        write_shard(tmp_path, [start] + goal_windows(start, [1, 1]))
        (tmp_path / 'specifications').mkdir(exist_ok=True)
        (tmp_path / f'specifications/{unit}.json').write_text(json.dumps({'unit_id': unit, 'method': method}))
    loaded = records.load_run(tmp_path)
    assert [unit['method'] for unit in loaded.units] == [
        'CleanRL PPO-RND', 'cleanrl ppo rnd at cleanrl defaults', 'even start']
    (tmp_path / 'report_settings.json').write_text(json.dumps({
        'methods': [{'name': 'RND at CleanRL defaults', 'unit_id_pattern': '*_defaults'}],
        'reference_methods': ['RND at CleanRL defaults'], 'exclude_units': ['*_new']}))
    loaded = records.load_run(tmp_path)
    assert [unit['method'] for unit in loaded.units] == ['CleanRL PPO-RND', 'RND at CleanRL defaults']
    assert loaded.reference_methods == {'RND at CleanRL defaults'} and len(loaded.policies) == 4


def test_a_unit_that_has_not_written_a_window_has_no_policies(tmp_path):
    """A unit with only its start record is listed, with no policies; a shard without a start record too."""
    start = start_record('unit-a')
    summary, policies = records.read_unit(write_shard(tmp_path, [start], complete=False))
    assert policies == [] and summary['state'] == 'unfinished' and summary['policies'] == 2 and summary['windows'] == 0
    empty = tmp_path / 'data/unit-b.jsonl'
    empty.write_text(json.dumps({'record': 'job', 'unit_id': 'unit-b'}) + '\n')
    assert records.read_unit(empty)[0]['state'] == 'no start record yet'
    with pytest.raises(ValueError, match="not one of the study's five environments"):
        records.read_unit(write_shard(tmp_path, [start_record('unit-c', family='montezuma', map_name='room1')]))


def test_records_that_were_read_already_give_the_same_unit(tmp_path):
    """A caller that reads a shard once passes its records on; the unit is the one read_unit gives."""
    start = start_record('read once', seeds=(0, 1))
    path = write_shard(tmp_path, [start] + goal_windows(start, [1, None]))
    lines = records.read_lines(path)
    summary, policies = records.unit_from_records(path, lines)
    again, same = records.read_unit(path)
    assert summary == again and summary['state'] == 'complete' and summary['windows'] == 4
    assert [policy.first_goal_step for policy in policies] == [policy.first_goal_step for policy in same] == [108, -1]


def test_of_a_seed_trained_by_several_units_the_unit_that_finished_first_is_used(tmp_path):
    """Finishing times are compared as moments, not as text; a unit that has not finished comes last."""
    first = start_record('b-sweep', cells=((1e-4, 1.0), (1e-4, 4.0)))
    second = start_record('a-repeat_weight-4', cells=((1e-4, 4.0),))
    third = start_record('c-unfinished_weight-4', cells=((1e-4, 4.0),))
    write_shard(tmp_path, [first] + goal_windows(first, [0, 0, 1, 1]), finished='2026-11-12T10:00:00-05:00')
    write_shard(tmp_path, [second] + goal_windows(second, [2, 3]), finished='2026-11-12T09:00:00-08:00')
    write_shard(tmp_path, [third] + goal_windows(third, [0, 0])[:2], complete=False)
    run = records.load_run(tmp_path)
    finished = {unit['unit_id']: unit['finished'] for unit in run.units}
    assert finished == {'a-repeat_weight-4': '2026-11-12T09:00:00-08:00', 'b-sweep': '2026-11-12T10:00:00-05:00',
                        'c-unfinished_weight-4': None}
    # 10:00 at five hours behind UTC is 15:00 UTC; 09:00 at eight hours behind is 17:00 UTC: the sweep is first.
    assert sorted(finished, key=lambda unit: records.finish_order(unit, finished)) == [
        'b-sweep', 'a-repeat_weight-4', 'c-unfinished_weight-4']
    used, repeats = records.split_repeats(run.policies, finished)
    assert [(policy.unit_id, policy.seed) for policy in used] == [('b-sweep', 0), ('b-sweep', 1)] * 2
    assert [(policy.unit_id, policy.seed) for policy in repeats] == [
        ('a-repeat_weight-4', 0), ('c-unfinished_weight-4', 0), ('a-repeat_weight-4', 1), ('c-unfinished_weight-4', 1)]
    assert len(records.configurations(used)) == 2
    assert records.unit_words('development-at-the-budget_pointmaze-medium_seeds-0-to-15') == (
        'development at the budget, pointmaze medium, seeds 0 to 15')
    # The same seed twice inside one unit is never a planned repeat (the first policy read is of the unit a-repeat).
    with pytest.raises(ValueError, match='Seed 0 was trained more than once for one configuration inside the unit a-repeat_weight-4'):
        records.split_repeats(list(run.policies) + [run.policies[0]], finished)
