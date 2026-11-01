"""Train many independent grid policies and preserve exact behavioral evidence."""

import argparse
from dataclasses import fields as dataclass_fields
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import PLATFORM, append_json, now, read_json, seed_for, write_json


def memory_profile(run_dir, policy_kind):
    """Use measured production packing, with a conservative first compatibility batch."""
    path = run_dir / 'infrastructure/production.json'
    if path.exists():
        record = read_json(path)
        return record.get(policy_kind, {'copies': 32, 'changes': {}})
    return {'copies': 32, 'changes': {}}


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
    folder = args.run_dir / 'data/research' / spec['name']
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'summary.json').exists() and read_json(folder / 'summary.json').get('completed'):
        return
    geometry = make_geometry(**spec['geometry'])
    profile = memory_profile(args.run_dir, spec['policy_kind'])
    # Before: settings x seeds; after: independent copies whose seed ignores setting/packing.
    copies = [(setting_index, seed) for setting_index in range(len(spec['settings']))
              for seed in range(spec['seed_start'], spec['seed_start']+spec['seed_count'])]
    records, arrays, field_cache = [], [], {}
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
        selected = copies[offset:offset+batch_size]
        shard = folder / f'completed_copies_{offset}-{offset+len(selected)-1}.json'
        behavior_path = folder / f'behavior_copies_{offset}-{offset+len(selected)-1}.npz'
        if shard.exists():
            records.extend(read_json(shard)['copies'])
            arrays.append(behavior_path)
            continue
        setting_indices = np.asarray([item[0] for item in selected], np.int32)
        seeds = np.asarray([item[1] for item in selected], np.int32)
        # Four distinct policy seeds share each random field; confirmation fields are disjoint.
        field_seeds = seeds // 4
        bonus_fields, raw_fields = [], []
        for field_seed in field_seeds:
            if int(field_seed) not in field_cache:
                field_cache[int(field_seed)] = make_initial_bonus_field(
                    spec['field_kind'], geometry, seed=int(field_seed),
                    severity_exponent=spec['severity_exponent'])
            field = field_cache[int(field_seed)]
            bonus_fields.append(field.normalized)
            raw_fields.append(field.raw)
        initial_fields = np.stack(bonus_fields)
        per_copy = {name: tuple(spec['settings'][index][name] for index in setting_indices)
                    for name in ('learning_rate', 'intrinsic_coefficient', 'entropy_coefficient', 'gamma_int')}
        rollout_steps = spec['settings'][0].get('rollout_steps', 128)
        if any(setting.get('rollout_steps', 128) != rollout_steps for setting in spec['settings']):
            raise ValueError('One unit cannot mix structural rollout lengths.')
        configuration = GridPPOConfig(n_copies=len(selected), policy_kind=spec['policy_kind'],
            total_timesteps=spec['total_timesteps'], episode_length=1 << (8*geometry.goal_state-1).bit_length(),
            goal_state=geometry.goal_state, num_steps=rollout_steps,
            per_copy_scalar_settings=per_copy, **changes)
        trainer = GridTrainer(configuration, geometry.transition_table, geometry.coordinates, initial_fields)
        policy_seeds = np.asarray([seed_for('policy', int(seed)) for seed in seeds], np.uint32)
        state = trainer.initialize(policy_seeds)
        log_interval = max(1, configuration.total_iterations // 128)
        snapshots, steps, maximums = [], [], []
        late_start = None
        for iteration in range(configuration.total_iterations):
            if iteration == int(.9 * configuration.total_iterations):
                late_start = np.asarray(jax.device_get(state.counts)).copy()
            state, metrics = trainer.train_iteration(state)
            if iteration % log_interval == 0 or iteration == configuration.total_iterations-1:
                host = jax.device_get(metrics)
                position = np.asarray(jax.device_get(state.positions))
                snapshots.append(position.copy())
                steps.append((iteration+1)*configuration.num_steps)
                maximums.append(np.asarray(host['maximum_progress']).copy())
                append_json(folder / 'progress.jsonl', {'at': now(), 'batch_offset': offset,
                    'completed_copies': len(records), 'total_copies': len(copies),
                    'iteration': iteration+1, 'total_iterations': configuration.total_iterations,
                    'goal_fraction': float((np.asarray(host['first_goal_step']) >= 0).mean())})
        first = np.asarray(jax.device_get(state.first_goal_step))
        counts = np.asarray(jax.device_get(state.counts))
        probabilities = np.asarray(jax.device_get(jax.nn.softmax(trainer.policy_outputs(state)[..., :4], axis=-1)))
        reached = np.asarray(jax.device_get(state.reached)).mean(axis=1)
        blocked = np.asarray(jax.device_get(state.blocked_moves))
        current_records = [{'setting_index': int(setting_index), 'policy_seed': int(seed),
            'field_seed': int(field_seed), 'first_goal_step': int(first[index]),
            'capped_first_goal_steps': int(first[index]) if first[index] >= 0 else spec['total_timesteps'],
            'coverage': float(reached[index]), 'blocked_fraction': float(blocked[index]/spec['total_timesteps'])}
            for index, ((setting_index, seed), field_seed) in enumerate(zip(selected, field_seeds))]
        np.savez_compressed(behavior_path, initial_fields=initial_fields, raw_initial_fields=np.stack(raw_fields),
            visit_counts=counts, late_visit_counts=counts-late_start if late_start is not None else counts,
            policy_probabilities=probabilities, positions=geometry.positions, grid=geometry.grid,
            trajectory_states=np.stack(snapshots), trajectory_steps=np.asarray(steps),
            maximum_progress_history=np.stack(maximums), first_goal_steps=first,
            seed_ids=seeds, field_seed_ids=field_seeds, setting_indices=setting_indices)
        write_json(shard, {'completed': True, 'copies': current_records})
        records.extend(current_records)
        arrays.append(behavior_path)
        del trainer, state
        # Structural compilation signatures repeat across many units; keep the JAX cache alive.
    summaries = []
    for index, setting in enumerate(spec['settings']):
        rows = [row for row in records if row['setting_index'] == index]
        summaries.append({'setting_index': index, 'configuration': setting,
            'capped_mean_first_goal_steps': float(np.mean([row['capped_first_goal_steps'] for row in rows])),
            'goal_fraction': float(np.mean([row['first_goal_step'] >= 0 for row in rows])),
            'coverage_mean': float(np.mean([row['coverage'] for row in rows])),
            'policy_seeds': [row['policy_seed'] for row in rows],
            'field_seeds': [row['field_seed'] for row in rows],
            'first_goal_steps': [row['first_goal_step'] for row in rows]})
    # Before: physical batches; after: one scientific unit with consistent copy and time axes.
    loaded = [np.load(path) for path in arrays]
    axis_one = {'trajectory_states', 'maximum_progress_history'}
    fixed = {'positions', 'grid', 'trajectory_steps'}
    combined = {name: loaded[0][name] if name in fixed else
                np.concatenate([data[name] for data in loaded], axis=1 if name in axis_one else 0)
                for name in loaded[0].files}
    np.savez_compressed(folder / 'behavior.npz', **combined)
    write_json(folder / 'summary.json', {'completed': True, 'specification': spec,
        'settings': summaries, 'copy_records': records, 'wall_seconds': time.monotonic()-started,
        'finished_at': now(), 'behavior': str(folder / 'behavior.npz')})


if __name__ == '__main__':
    main()
