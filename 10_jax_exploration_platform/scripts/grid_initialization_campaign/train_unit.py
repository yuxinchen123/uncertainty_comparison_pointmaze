"""Train many independent grid policies and preserve exact behavioral evidence."""

import argparse
import gc
from dataclasses import asdict, fields as dataclass_fields
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import PLATFORM, append_json, now, read_json, seed_for, write_json
from initial_field_archive import load_or_create_initial_field, uses_canonical_archive


# Online RND settings a unit specification may carry under "rnd_settings" (GridPPOConfig names).
RND_SETTING_NAMES = ('rnd_readout', 'rnd_unit_length_target', 'rnd_loss', 'rnd_loss_power', 'rnd_shrink_power', 'rnd_optimizer',
                     'rnd_learning_rate', 'rnd_learning_rate_time_scale', 'rnd_trainable')


def network_options(spec):
    """Read the unit's RND network construction; the default is the coordinate networks."""
    from exploration_platform.grid_initialization.online_rnd import NetworkOptions
    # Before: {"rnd_input": "one_hot", "network_options": {"first_width": 64}}; after: NetworkOptions(one_hot, 64, ...).
    return NetworkOptions(input_kind=spec.get('rnd_input', 'coordinates'), **spec.get('network_options', {}))


def training_budget(spec):
    """Return the transitions actually trained: the full budget, or an earlier stop on the same schedule."""
    stop = spec.get('stop_after_timesteps')
    if stop is None:
        return spec['total_timesteps']
    if not 0 < stop <= spec['total_timesteps']:
        raise ValueError('stop_after_timesteps must lie within the learning-rate schedule.')
    return stop


def visitation_checkpoint_steps(total_timesteps, rollout_steps):
    """List exact powers-of-two visitation checkpoints from 1024 transitions through the budget."""
    if total_timesteps < 1 or rollout_steps < 1:
        raise ValueError('Visitation checkpoints require a positive budget and rollout length.')
    steps = [1 << exponent for exponent in range(10, total_timesteps.bit_length())]
    if not steps or steps[-1] != total_timesteps:
        steps.append(total_timesteps)
    if any(step % rollout_steps for step in steps):
        raise ValueError('Exact visitation checkpoints must align with complete rollout boundaries.')
    return steps


def memory_profile(run_dir, policy_kind):
    """Use measured production packing, with a conservative first compatibility batch."""
    path = run_dir / 'infrastructure/production.json'
    gpu_class = os.environ.get('GRID_GPU_CLASS', 'default')
    compatibility = run_dir/'infrastructure/compatibility'/(gpu_class+'.json')
    measured = read_json(compatibility) if compatibility.exists() else None
    if path.exists():
        record = read_json(path).get(policy_kind, {'copies': 32, 'changes': {}})
        # The jaguar03 packing result is not evidence for an older GPU class.
        if measured is None or record.get('device_kind') == measured.get('device_kind'):
            return record
    if measured is not None:
        cases = [case for case in measured.get('cases', []) if case['policy_kind'] == policy_kind
                 and case.get('completed') and case.get('correctness', {}).get('passed')]
        if cases:
            return {'copies': min(case['copies'] for case in cases), 'changes': {},
                    'source': str(compatibility), 'device_kind': measured.get('device_kind')}
    return {'copies': 32, 'changes': {}}


def fixed_settings_packing(run_dir, policy_kind, intrinsic_source):
    """Read the copies per process measured for this card class, policy and bonus source."""
    # Before: packing/nvidia_rtx_a4500.json {"neural_online_rnd": {"copies": 256}}; after: 256.
    path = run_dir/'infrastructure/packing'/(os.environ.get('GRID_GPU_CLASS', 'default')+'.json')
    entry = read_json(path).get(f'{policy_kind}_{intrinsic_source}') if path.exists() else None
    return {'copies': entry['copies'], 'changes': {}, 'source': str(path)} if entry else {'copies': 32, 'changes': {}}


def fixed_settings_field(spec, geometry, field_seed, episode_length):
    """Build one field of the fixed-settings study and its input statistics, if any."""
    from exploration_platform.grid_initialization.fields import make_initial_bonus_field
    from exploration_platform.grid_initialization.online_rnd import frozen_network_field, warmup_statistics
    # The frozen network field and online RND share networks and, when normalized, statistics.
    if spec['field_kind'] == 'random_network':
        statistics = (warmup_statistics(geometry, episode_length, field_seed)
                      if spec.get('state_normalization') else None)
        return frozen_network_field(geometry, field_seed, statistics, network_options(spec)), statistics
    if spec.get('state_normalization'):
        raise ValueError('State normalization applies only to fields built by RND networks.')
    return make_initial_bonus_field(spec['field_kind'], geometry, seed=field_seed,
                                    severity_exponent=spec['severity_exponent']), None


def main():
    """Execute one immutable settings/seed unit with per-copy state on the GPU."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--specification', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(PLATFORM / 'src'))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from exploration_platform.grid_initialization.config import GridPPOConfig
    from exploration_platform.grid_initialization.environments import make_geometry
    from exploration_platform.grid_initialization.fields import make_initial_bonus_field
    from exploration_platform.grid_initialization.training import GridTrainer
    jax.config.update('jax_enable_x64', True)
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Training requires an allocated compatible GPU; CPU fallback is forbidden.')
    spec = read_json(args.specification)
    fixed_study = spec.get('study') == 'fixed-settings'
    canonical_fields = uses_canonical_archive(spec) and not fixed_study
    # The fixed-settings study records count (and online RND error) histories in every unit.
    histories = canonical_fields or fixed_study
    source_kind = spec.get('intrinsic_source', 'count')
    online = source_kind == 'online_rnd'
    folder = args.run_dir / 'data/research' / spec['name']
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'summary.json').exists() and read_json(folder / 'summary.json').get('completed'):
        return
    geometry = make_geometry(**spec['geometry'])
    profile = (fixed_settings_packing(args.run_dir, spec['policy_kind'], source_kind) if fixed_study
               else memory_profile(args.run_dir, spec['policy_kind']))
    # Before: settings x seeds; after: independent copies whose seed ignores setting/packing.
    copies = [(setting_index, seed) for setting_index in range(len(spec['settings']))
              for seed in range(spec['seed_start'], spec['seed_start']+spec['seed_count'])]
    total_scientific_copies = len(copies)
    records, arrays, field_cache, statistics_cache = [], [], {}, {}
    completed_ids = set()
    for completed_path in sorted(folder.glob('completed_copies_*.json')):
        completed = read_json(completed_path)
        if not completed.get('completed'):
            continue
        behavior = folder / completed['behavior_file']
        if not behavior.exists():
            raise RuntimeError(f'Completed shard has no behavioral archive: {completed_path}')
        for row in completed['copies']:
            identity = (row['setting_index'], row['policy_seed'])
            if identity in completed_ids:
                raise RuntimeError(f'Duplicate completed scientific copy: {identity}')
            completed_ids.add(identity)
            records.append(row)
        arrays.append(behavior)
    # Scientific copy identifiers survive a change in physical GPU packing after a retry.
    copies = [copy for copy in copies if copy not in completed_ids]
    field_geometry = make_geometry(**spec.get('field_geometry', spec['geometry']))
    if field_geometry.state_count != geometry.state_count:
        raise ValueError('A shared path field requires equal state counts.')
    started = time.monotonic()
    batch_size = profile['copies']
    configuration_names = {item.name for item in dataclass_fields(GridPPOConfig)}
    changes = {name: value for name, value in profile.get('changes', {}).items()
               if name in configuration_names and name != 'n_copies'}
    # A GPU class receives smaller physical batches after a failed memory admission.
    override = args.run_dir / 'infrastructure' / ('batch_limit_' + os.environ.get('GRID_GPU_CLASS', 'default') + '.json')
    if override.exists():
        batch_size = min(batch_size, read_json(override)['copies'])
    for offset in range(0, len(copies), batch_size):
        batch_started = time.monotonic()
        selected = copies[offset:offset+batch_size]
        shard_identifier = f'{selected[0][0]}-{selected[0][1]}_to_{selected[-1][0]}-{selected[-1][1]}'
        shard = folder / f'completed_copies_{shard_identifier}.json'
        behavior_path = folder / f'behavior_copies_{shard_identifier}.npz'
        setting_indices = np.asarray([item[0] for item in selected], np.int32)
        seeds = np.asarray([item[1] for item in selected], np.int32)
        # Four distinct policy seeds share each random field; confirmation fields are disjoint.
        field_seeds = seeds // 4
        bonus_fields, raw_fields, statistics_rows = [], [], []
        episode_length = 1 << (8*geometry.goal_state-1).bit_length()
        for field_seed in field_seeds:
            if fixed_study and int(field_seed) not in field_cache:
                field_cache[int(field_seed)], statistics_cache[int(field_seed)] = fixed_settings_field(
                    spec, field_geometry, int(field_seed), episode_length)
            if int(field_seed) not in field_cache:
                if canonical_fields:
                    field_cache[int(field_seed)] = load_or_create_initial_field(
                        args.run_dir, field_geometry, spec['field_kind'],
                        spec['severity_exponent'], int(field_seed))
                else:
                    field_cache[int(field_seed)] = make_initial_bonus_field(
                        spec['field_kind'], field_geometry, seed=int(field_seed),
                        severity_exponent=spec['severity_exponent'])
            field = field_cache[int(field_seed)]
            normalized, raw = field.normalized.copy(), field.raw.copy()
            transform = spec.get('field_transform', 'none')
            if transform == 'reverse':
                normalized, raw = normalized[::-1].copy(), raw[::-1].copy()
            elif transform == 'shuffle':
                permutation = np.random.default_rng(seed_for('field-shuffle', int(field_seed))).permutation(len(raw))
                normalized, raw = normalized[permutation], raw[permutation]
            elif transform == 'flatten':
                normalized, raw = np.ones_like(normalized), np.full_like(raw, raw.mean())
            elif transform != 'none':
                raise ValueError(f'Unknown field transform: {transform}')
            bonus_fields.append(normalized)
            raw_fields.append(raw)
            statistics_rows.append(statistics_cache.get(int(field_seed)))
        # Count bonuses read the mean-one field; online RND reads its own raw error, so the
        # trainer receives the raw field only as the record of its starting values.
        initial_fields = np.stack(raw_fields if online else bonus_fields)
        rnd = None
        if online:
            from exploration_platform.grid_initialization.online_rnd import input_table, stack_networks
            options = network_options(spec)
            rnd = stack_networks(field_seeds, options, input_table(field_geometry, options).shape[1])
            if spec.get('state_normalization'):
                rnd['statistics'] = {name: np.stack([np.asarray(row[name], np.float64) for row in statistics_rows])
                                     for name in ('mean', 'variance', 'count')}
        per_copy = {name: tuple(spec['settings'][index][name] for index in setting_indices)
                    for name in ('learning_rate', 'intrinsic_coefficient', 'entropy_coefficient', 'gamma_int')}
        rollout_steps = spec['settings'][0].get('rollout_steps', 128)
        if any(setting.get('rollout_steps', 128) != rollout_steps for setting in spec['settings']):
            raise ValueError('One unit cannot mix structural rollout lengths.')
        configuration = GridPPOConfig(n_copies=len(selected), policy_kind=spec['policy_kind'],
            total_timesteps=spec['total_timesteps'], episode_length=1 << (8*geometry.goal_state-1).bit_length(),
            goal_state=geometry.goal_state, num_steps=rollout_steps,
            normalize_intrinsic_rewards=spec.get('normalize_intrinsic_rewards', False),
            intrinsic_source=source_kind,
            normalize_rnd_inputs=bool(online and spec.get('state_normalization')),
            count_decay_exponent=spec.get('count_decay_exponent', 0.5),
            rnd_input=spec.get('rnd_input', 'coordinates') if online else 'coordinates',
            **{name: value for name, value in spec.get('rnd_settings', {}).items() if name in RND_SETTING_NAMES},
            per_copy_scalar_settings=per_copy, **changes)
        if set(spec.get('rnd_settings', {})) - set(RND_SETTING_NAMES):
            raise ValueError(f"Unknown RND settings: {sorted(set(spec['rnd_settings']) - set(RND_SETTING_NAMES))}")
        budget = training_budget(spec)
        if budget % configuration.num_steps:
            raise ValueError('An early stop must fall on a rollout boundary.')
        trainer = GridTrainer(configuration, geometry.transition_table, geometry.coordinates, initial_fields, rnd)
        policy_seeds = np.asarray([seed_for('policy', int(seed)) for seed in seeds], np.uint32)
        state = trainer.initialize(policy_seeds)
        log_interval = max(1, configuration.total_iterations // 128)
        snapshots, steps, maximums = [], [], []
        visitation_steps = (visitation_checkpoint_steps(budget, configuration.num_steps) if histories else [])
        visitation_step_set = set(visitation_steps)
        visitation_counts, error_history = [], []
        initial_errors = np.asarray(jax.device_get(trainer.all_cell_errors(state))) if online else None
        late_start = None
        trained_iterations = budget // configuration.num_steps
        for iteration in range(trained_iterations):
            if iteration == int(.9 * trained_iterations):
                late_start = np.asarray(jax.device_get(state.counts)).copy()
            state, metrics = trainer.train_iteration(state)
            # Confirmation records the actual lifetime counts, before later goals and resets can hide delay.
            if histories and (iteration+1)*configuration.num_steps in visitation_step_set:
                visitation_counts.append(np.asarray(jax.device_get(state.counts)).copy())
                if online:
                    error_history.append(np.asarray(jax.device_get(trainer.all_cell_errors(state))))
            if iteration % log_interval == 0 or iteration == trained_iterations-1:
                host = jax.device_get(metrics)
                position = np.asarray(jax.device_get(state.positions))
                snapshots.append(position.copy())
                steps.append((iteration+1)*configuration.num_steps)
                maximums.append(np.asarray(host['maximum_progress']).copy())
                append_json(folder / 'progress.jsonl', {'at': now(), 'batch_offset': offset,
                    'completed_copies': len(records), 'total_copies': total_scientific_copies,
                    'iteration': iteration+1, 'total_iterations': trained_iterations,
                    'goal_fraction': float((np.asarray(host['first_goal_step']) >= 0).mean())})
        first = np.asarray(jax.device_get(state.first_goal_step))
        counts = np.asarray(jax.device_get(state.counts))
        probabilities = np.asarray(jax.device_get(jax.nn.softmax(trainer.policy_outputs(state)[..., :4], axis=-1)))
        reached = np.asarray(jax.device_get(state.reached)).mean(axis=1)
        blocked = np.asarray(jax.device_get(state.blocked_moves))
        # Final-policy evaluation without learning: 32 sampled episodes and one greedy episode.
        episodes = int(spec.get('evaluation_episodes', 0))
        sampled = np.asarray(jax.device_get(trainer.evaluate(state, episodes, False))) if episodes else None
        greedy = np.asarray(jax.device_get(trainer.evaluate(state, 1, True))) if episodes else None
        current_records = [{'setting_index': int(setting_index), 'policy_seed': int(seed),
            'field_seed': int(field_seed), 'first_goal_step': int(first[index]),
            'capped_first_goal_steps': int(first[index]) if first[index] >= 0 else budget,
            'coverage': float(reached[index]),
            'maximum_progress': int(np.asarray(jax.device_get(state.maximum_progress))[index]),
            'goal_arrivals': int(np.asarray(jax.device_get(state.successes))[index]),
            'blocked_fraction': float(blocked[index]/budget),
            **({'evaluation_goal_rate_sampled': float((sampled[index] >= 0).mean()),
                'evaluation_goal_rate_greedy': float(greedy[index, 0] >= 0),
                'evaluation_mean_goal_steps_sampled': (float(sampled[index][sampled[index] >= 0].mean())
                    if (sampled[index] >= 0).any() else None)} if episodes else {})}
            for index, ((setting_index, seed), field_seed) in enumerate(zip(selected, field_seeds))]
        # Before: cumulative counts at selected times; after: [snapshot,copies,states] without changing RNG.
        history = ({'visit_count_history': np.stack(visitation_counts),
                    'visit_count_history_steps': np.asarray(visitation_steps, np.int64)}
                   if histories else {})
        if online:
            history.update(rnd_error_initial=initial_errors, rnd_error_history=np.stack(error_history))
        if episodes:
            history.update(evaluation_sampled_goal_steps=sampled, evaluation_greedy_goal_steps=greedy)
        np.savez_compressed(behavior_path, initial_fields=initial_fields, raw_initial_fields=np.stack(raw_fields),
            visit_counts=counts, late_visit_counts=counts-late_start if late_start is not None else counts,
            policy_probabilities=probabilities, positions=geometry.positions, grid=geometry.grid,
            trajectory_states=np.stack(snapshots), trajectory_steps=np.asarray(steps),
            maximum_progress_history=np.stack(maximums), first_goal_steps=first,
            seed_ids=seeds, field_seed_ids=field_seeds, setting_indices=setting_indices, **history)
        write_json(shard, {'completed': True, 'copies': current_records,
            'behavior_file': behavior_path.name, 'wall_seconds': time.monotonic()-batch_started,
            'configuration': asdict(configuration), 'physical_packing_profile': profile,
            'field_metadata': [field_cache[int(seed)].metadata for seed in field_seeds]})
        records.extend(current_records)
        arrays.append(behavior_path)
        del trainer, state, probabilities
        # Keep executable memory bounded across many parameter-vector signatures.
        jax.clear_caches()
        gc.collect()
    expected_count = len(spec['settings']) * spec['seed_count']
    if len(records) != expected_count:
        raise RuntimeError(f'Expected {expected_count} completed copies, found {len(records)}')
    summaries = []
    for index, setting in enumerate(spec['settings']):
        rows = [row for row in records if row['setting_index'] == index]
        summaries.append({'setting_index': index, 'configuration': setting,
            'capped_mean_first_goal_steps': float(np.mean([row['capped_first_goal_steps'] for row in rows])),
            'goal_fraction': float(np.mean([row['first_goal_step'] >= 0 for row in rows])),
            'coverage_mean': float(np.mean([row['coverage'] for row in rows])),
            'policy_seeds': [row['policy_seed'] for row in rows],
            'field_seeds': [row['field_seed'] for row in rows],
            'first_goal_steps': [row['first_goal_step'] for row in rows],
            **({'evaluation_goal_rate_sampled': float(np.mean([row['evaluation_goal_rate_sampled'] for row in rows])),
                'evaluation_goal_rate_greedy': float(np.mean([row['evaluation_goal_rate_greedy'] for row in rows]))}
               if rows and 'evaluation_goal_rate_sampled' in rows[0] else {})})
    # Before: physical batches; after: one scientific unit with consistent copy and time axes.
    loaded = [np.load(path) for path in arrays]
    axis_one = {'trajectory_states', 'maximum_progress_history'}
    fixed = {'positions', 'grid', 'trajectory_steps'}
    if online:
        axis_one.add('rnd_error_history')
    if histories:
        axis_one.add('visit_count_history')
        fixed.add('visit_count_history_steps')
        expected_steps = np.asarray(visitation_checkpoint_steps(training_budget(spec),
            spec['settings'][0].get('rollout_steps', 128)), np.int64)
        for path, data in zip(arrays, loaded):
            if not {'visit_count_history', 'visit_count_history_steps'}.issubset(data.files):
                raise ValueError(f'Completed confirmation shard has no exact visitation history: {path}')
            saved_steps, saved_counts = data['visit_count_history_steps'], data['visit_count_history']
            if not np.array_equal(saved_steps, expected_steps):
                raise ValueError(f'Confirmation shards have different visitation checkpoint steps: {path}')
            if saved_counts.shape != (len(expected_steps), len(data['seed_ids']), geometry.state_count):
                raise ValueError(f'Confirmation visitation history has the wrong shape: {path}')
            if (np.any(saved_counts < 0) or np.any(np.diff(saved_counts, axis=0) < 0)
                    or not np.all(saved_counts.sum(axis=2) == expected_steps[:, None])
                    or not np.array_equal(saved_counts[-1], data['visit_counts'])):
                raise ValueError(f'Confirmation visitation history disagrees with exact lifetime counts: {path}')
    combined = {name: loaded[0][name] if name in fixed else
                np.concatenate([data[name] for data in loaded], axis=1 if name in axis_one else 0)
                for name in loaded[0].files}
    np.savez_compressed(folder / 'behavior.npz', **combined)
    write_json(folder / 'summary.json', {'completed': True, 'specification': spec, 'budget': training_budget(spec),
        'settings': summaries, 'copy_records': records,
        'wall_seconds': sum(read_json(path).get('wall_seconds', 0.) for path in folder.glob('completed_copies_*.json')),
        'wall_seconds_this_attempt': time.monotonic()-started, 'resumed_copy_count': len(completed_ids),
        'finished_at': now(), 'behavior': str(folder / 'behavior.npz')})


if __name__ == '__main__':
    main()
