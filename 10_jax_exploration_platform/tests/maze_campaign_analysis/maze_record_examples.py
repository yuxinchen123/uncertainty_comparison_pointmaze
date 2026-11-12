"""Small hand-made records for the maze-campaign analysis tests: policies in memory and shards on disk."""

import json
from pathlib import Path

import numpy as np

from maze_campaign.analysis import records

SETTINGS = (('ent_coef', 0.001), ('int_coef', 1.0), ('learning_rate', 1e-4))


def make_policy(steps, episodes, returns, *, arrivals=None, steps_to_goal=None, coverage=None, first_goal_step=-1,
                seed=0, reward_setting='arrival_plus_one', method='CleanRL PPO-RND', environment='PointMaze Large',
                width=4, planned=None, settings=SETTINGS, step_limit=400):
    """One policy from per-window lists; arrivals default to the returns (reward 1 on arrival)."""
    as_array = lambda values: np.asarray(values, np.float64)
    arrivals = returns if arrivals is None else arrivals
    return records.Policy(
        unit_id='hand-made unit', environment=environment, environment_identity='{}', reward_setting=reward_setting,
        method=method, environments_per_policy=width, planned_steps=planned or int(steps[-1]), settings=settings,
        seed=seed, base_seed=0, step_limit=step_limit, complete=True, steps=np.asarray(steps, np.int64),
        episodes=as_array(episodes), arrivals=as_array(arrivals), return_sum=as_array(returns),
        environment_return_sum=as_array(arrivals), length_sum=as_array(episodes) * step_limit,
        steps_to_goal_sum=as_array(steps_to_goal if steps_to_goal is not None else np.zeros(len(steps))),
        coverage=None if coverage is None else as_array(coverage), first_goal_step=first_goal_step)


def flat_policy(values, *, seed, window=100, episodes=10, **more):
    """A policy whose training curve takes the given value in each window of `window` steps."""
    # before: values [0.2, 0.5]; after: windows ending at steps 100 and 200 with 10 episodes each and return sums 2, 5.
    steps = [window * (index + 1) for index in range(len(values))]
    return make_policy(steps, [episodes] * len(values), [value * episodes for value in values], seed=seed, **more)


def start_record(unit_id, *, family='pointmaze', map_name='large', step_limit=400, bonus='rnd_next_state', envs=4,
                 rollout=25, iterations=4, window_iterations=1, cells=((1e-4, 1.0),),
                 names=('learning_rate', 'int_coef'), seeds=(0, 1), trainer=None):
    """A unit_start record with the fields the reader uses: every cell holds every seed."""
    # before: 2 cells and seeds (0, 1); after: copy_cell [0, 0, 1, 1] and copy_seed_index [0, 1, 0, 1].
    copy_cell = [cell for cell in range(len(cells)) for _ in seeds]
    trainer_config = {'learning_rate': 3e-4, 'int_coef': 1.0, 'ent_coef': 0.001, 'clip_coef': 0.1, 'gamma_int': 0.99,
                      'n_copies': len(copy_cell), 'n_envs': envs, 'num_steps': rollout, 'base_seed': 0,
                      'ext_reward_scale': 1.0, 'ext_reward_offset': 0.0, 'episode_log': True,
                      'setting_names': list(names), **(trainer or {})}
    return {'record': 'unit_start', 'unit_id': unit_id, 'bonus': bonus,
            'environment': f'{family}_{map_name}_term{step_limit}_nonoise@1',
            'environment_config': {'map_name': map_name, 'max_episode_steps': step_limit, 'continuing_task': False,
                                   'reward_shift': 0.0},
            'trainer_config': trainer_config, 'copies': len(copy_cell), 'envs_per_copy': envs,
            'rollout_steps': rollout, 'iterations': iterations, 'window_iterations': window_iterations,
            'cell_settings': [list(cell) for cell in cells], 'cell_setting_names': list(names),
            'copy_cell': copy_cell, 'copy_seed_index': [seed for _ in cells for seed in seeds],
            'lr_anneal_iterations': iterations}


def window_record(start, index, **per_copy):
    """Window `index` (from 0) of a unit: zeros for every sum that is not given, one number per policy."""
    copies, length = start['copies'], start['window_iterations']
    record = {'record': 'episode_window', 'unit_id': start['unit_id'], 'window_index': index,
              'first_iteration': index * length + 1, 'last_iteration': (index + 1) * length,
              'env_steps_per_copy': (index + 1) * length * start['rollout_steps'] * start['envs_per_copy']}
    for name in records.WINDOW_SUMS:
        record[f'{name}_per_copy'] = list(per_copy.get(name, [0] * copies))
    record['first_goal_step_per_copy'] = list(per_copy.get('first_goal_step', [-1] * copies))
    record['coverage_per_copy'] = list(per_copy.get('coverage', [0.0] * copies))
    return record


def goal_windows(start, first_goal_windows, episodes=10, goal_steps=40):
    """Every window of a unit in which policy c reaches the goal in all episodes from window first_goal_windows[c] on.

    An episode that reaches the goal takes `goal_steps` steps; the others run to the step limit. The return the
    learner sees is scale x arrivals + offset x lengths, as the trainer writes it.
    before: first_goal_windows [1, None] with 4 windows; after: policy 0 arrives in windows 1 to 3, policy 1 never.
    """
    limit = start['environment_config']['max_episode_steps']
    steps_per_window = start['window_iterations'] * start['rollout_steps'] * start['envs_per_copy']
    cells = [records.cell_values(start, start['cell_settings'][cell]) for cell in start['copy_cell']]
    windows = []
    for index in range(start['iterations'] // start['window_iterations']):
        found = [first is not None and index >= first for first in first_goal_windows]
        arrivals = [episodes if hit else 0 for hit in found]
        lengths = [episodes * (goal_steps if hit else limit) for hit in found]
        windows.append(window_record(
            start, index, episodes=[episodes] * len(found), arrivals=arrivals, environment_return_sum=arrivals,
            length_sum=lengths, steps_to_goal_sum=[episodes * goal_steps if hit else 0 for hit in found],
            return_sum=[values['ext_reward_scale'] * arrived + values['ext_reward_offset'] * length
                        for values, arrived, length in zip(cells, arrivals, lengths)],
            first_goal_step=[-1 if first is None or index < first else first * steps_per_window + 8
                             for first in first_goal_windows],
            coverage=[1.0 if hit else 0.25 for hit in found]))
    return windows


def curve_windows(start, curves, episodes=20):
    """Every window of a unit whose policy c has, in window w, the share curves[c][w] of its episodes at the goal.

    before: curves [[0, 1.0, 0.6]] with 20 episodes per window; after: three windows with 0, 20 and 12 arrivals.
    Reward 1 on arrival only: the return sum of a window is its number of arrivals.
    """
    limit = start['environment_config']['max_episode_steps']
    windows = []
    for index in range(len(curves[0])):
        arrivals = [round(curve[index] * episodes) for curve in curves]
        windows.append(window_record(
            start, index, episodes=[episodes] * len(curves), arrivals=arrivals, return_sum=arrivals,
            environment_return_sum=arrivals, length_sum=[episodes * limit] * len(curves),
            steps_to_goal_sum=[40 * arrived for arrived in arrivals],
            first_goal_step=[-1] * len(curves), coverage=[1.0] * len(curves)))
    return windows


def write_shard(run_dir, unit_records, complete=True):
    """Write one unit's records as a shard of the run folder; returns the shard's path."""
    start = next(record for record in unit_records if record['record'] == 'unit_start')
    lines = [{'record': 'job', 'unit_id': start['unit_id']}] + list(unit_records)
    if complete:
        lines.append({'record': 'unit_complete', 'unit_id': start['unit_id'], 'nan_respawned_episodes': None})
    path = Path(run_dir) / 'data' / f"{start['unit_id']}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(line) + '\n' for line in lines))
    return path
