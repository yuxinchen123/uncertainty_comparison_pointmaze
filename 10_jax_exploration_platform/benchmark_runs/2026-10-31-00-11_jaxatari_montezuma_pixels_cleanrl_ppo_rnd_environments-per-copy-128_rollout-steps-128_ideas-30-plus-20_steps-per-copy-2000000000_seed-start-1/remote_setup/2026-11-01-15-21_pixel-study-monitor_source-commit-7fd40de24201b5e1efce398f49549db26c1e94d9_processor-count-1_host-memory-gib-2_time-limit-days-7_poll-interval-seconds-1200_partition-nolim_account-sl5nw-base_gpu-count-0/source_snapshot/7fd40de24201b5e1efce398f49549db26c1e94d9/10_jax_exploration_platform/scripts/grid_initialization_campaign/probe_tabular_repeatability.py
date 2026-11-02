"""Locate the first difference between duplicate tabular policies without changing training code."""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

from common import now, read_json, seed_for, write_json


def reference_shard(run_dir):
    """Choose one recorded uniform snake batch whose duplicate seeds actually differed."""
    pattern = 'confirmation_snake-width-16-turns-2_policy-tabular_bonus-uniform*'
    for folder in sorted((run_dir/'data/research').glob(pattern)):
        if not (folder/'summary.json').exists():
            continue
        summary = read_json(folder/'summary.json')
        for path in sorted(folder.glob('completed_copies_*.json')):
            shard = read_json(path)
            pairs = {}
            for index, row in enumerate(shard['copies']):
                pairs.setdefault(row['policy_seed'], []).append(index)
            different = [(seed, indices) for seed, indices in pairs.items() if len(indices) == 2
                and shard['copies'][indices[0]]['first_goal_step'] != shard['copies'][indices[1]]['first_goal_step']]
            if different and len(shard['copies']) == 32:
                task = read_json(run_dir/'queue/done'/(summary['specification']['name']+'.json'))
                submission = read_json(run_dir/'slurm/submissions'/(task['job_id']+'.json'))
                return path, shard, pairs, different[0], task, submission
    raise ValueError('No completed 32-copy uniform snake shard contains a measured duplicate-seed difference.')


def repetition_check(evaluate, repetitions):
    """Repeat a pure calculation with unchanged inputs and measure exact output differences."""
    import jax
    import numpy as np

    reference = np.asarray(jax.device_get(evaluate())).copy()
    maximum, different = 0., []
    for repeat in range(1, repetitions):
        current = np.asarray(jax.device_get(evaluate())).copy()
        delta = float(np.max(np.abs(current-reference), initial=0))
        maximum = max(maximum, delta)
        if not np.array_equal(reference, current):
            different.append(repeat)
    return {'repetitions': repetitions, 'different_repetitions': different,
            'maximum_absolute_difference': maximum,
            'reference_sha256': hashlib.sha256(reference.tobytes()).hexdigest()}


def run_case(configuration, base_seeds, pair_indices, world, max_iterations, repetitions, deadline):
    """Compare duplicate-policy evolution and isolate fixed-data gradient accumulation."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from exploration_platform.atari_pixels.networks import quantity_keys
    from exploration_platform.grid_initialization.training import GridTrainer, minibatch_loss, network_outputs

    started = time.monotonic()
    left, right = (np.asarray(indices, np.int32) for indices in pair_indices)
    fields = np.ones((configuration.n_copies, world.state_count), np.float32)
    trainer = GridTrainer(configuration, world.transition_table, world.coordinates, fields)
    policy_seeds = np.asarray([seed_for('policy', int(seed)) for seed in base_seeds], np.uint32)
    initial = trainer.initialize(policy_seeds)
    capture = jax.jit(trainer.capture)
    prepared_state, rollout = capture(initial)
    prepared_state, batch = jax.jit(trainer.prepare_batch)(prepared_state, rollout)
    keys = quantity_keys(initial.policy_seeds, 'grid/rollout/actions', initial.iteration, 0)
    permutation_keys = quantity_keys(initial.policy_seeds, 'grid/update/permutation', initial.iteration, 0)
    permutations = jax.vmap(lambda key: jax.random.permutation(key, configuration.num_steps))(permutation_keys)
    rows = permutations[:, :configuration.minibatch_size]
    selected = jax.tree.map(lambda value: jax.vmap(lambda values, ids: values[ids])(value, rows), batch)
    gradient = jax.jit(jax.grad(lambda parameters, data:
        minibatch_loss(parameters, data, trainer.coordinates, configuration)[0]))
    # Every fixed-data calculation below has no random draws or mutable training state.
    full_gradient = repetition_check(lambda: gradient(initial.parameters, selected)['table'], repetitions)
    per_row_parameters = {'table': network_outputs(initial.parameters, selected['states'], trainer.coordinates, configuration)}
    per_row_batch = dict(selected, states=jnp.broadcast_to(jnp.arange(configuration.minibatch_size, dtype=jnp.int32),
        (configuration.n_copies, configuration.minibatch_size)))
    per_row_gradients = gradient(per_row_parameters, per_row_batch)['table']
    unique_row_gradient = repetition_check(lambda: gradient(per_row_parameters, per_row_batch)['table'], repetitions)
    copy_rows = jnp.arange(configuration.n_copies, dtype=jnp.int32)

    @jax.jit
    def simultaneous_accumulation(updates):
        """Accumulate repeated state indices concurrently, as a gathered table's backward pass does."""
        return jnp.zeros_like(initial.parameters['table']).at[copy_rows[:, None], selected['states']].add(updates)

    @jax.jit
    def ordered_accumulation(updates):
        """Accumulate one transition per policy at a time so state updates cannot collide."""
        def add_one(table, values):
            """Add one independent update for each policy before proceeding to the next transition."""
            states, increments = values
            return table.at[copy_rows, states].add(increments), None

        return jax.lax.scan(add_one, jnp.zeros_like(initial.parameters['table']),
            (selected['states'].T, updates.transpose(1, 0, 2)))[0]

    simultaneous = repetition_check(lambda: simultaneous_accumulation(per_row_gradients), repetitions)
    ordered = repetition_check(lambda: ordered_accumulation(per_row_gradients), repetitions)
    lowered = gradient.lower(initial.parameters, selected).compiler_ir(dialect='hlo').as_hlo_text()
    scatter_lines = [line.strip() for line in lowered.splitlines() if 'scatter(' in line or 'scatter-add' in line]
    values = np.asarray(jax.device_get(initial.parameters['table']))
    record = {'copies': configuration.n_copies, 'duplicate_pairs': len(left),
        'initial_parameters_equal': np.array_equal(values[left], values[right]),
        'initial_action_keys_equal': np.array_equal(np.asarray(keys)[left], np.asarray(keys)[right]),
        'initial_minibatch_permutations_equal': np.array_equal(np.asarray(permutations)[left], np.asarray(permutations)[right]),
        'initial_rollout_actions_equal': np.array_equal(np.asarray(rollout.actions)[:, left], np.asarray(rollout.actions)[:, right]),
        'fixed_minibatch_gradient_repeats': full_gradient,
        'unique_state_row_gradient_repeats': unique_row_gradient,
        'simultaneous_repeated_state_accumulation': simultaneous,
        'ordered_state_accumulation': ordered,
        'gradient_hlo_scatter_lines': scatter_lines[:8], 'configuration': configuration.__dict__,
        'pair_indices': [left.tolist(), right.tolist()], 'base_policy_seeds': list(map(int, base_seeds)),
        'first_parameter_difference': None, 'first_action_difference': None,
        'first_counter_difference': None, 'first_goal_time_difference': None,
        'first_rng_key_difference': None, 'capture_matches_training_counts': True}
    # Follow the original compiled updates until the first behavior difference, or the bounded budget.
    state = initial
    for iteration in range(max_iterations):
        if time.monotonic() >= deadline:
            record['stop_reason'] = 'internal time limit'
            break
        before, observed = capture(state)
        actions = np.asarray(jax.device_get(observed.actions))
        if not np.array_equal(actions[:, left], actions[:, right]) and record['first_action_difference'] is None:
            differing = np.argwhere(actions[:, left] != actions[:, right])[0]
            record['first_action_difference'] = {'iteration': iteration, 'rollout_step': int(differing[0]),
                'pair': int(differing[1]), 'transition': iteration*configuration.num_steps+int(differing[0])+1}
        action_keys = quantity_keys(state.policy_seeds, 'grid/rollout/actions', state.iteration, 0)
        if not np.array_equal(np.asarray(action_keys)[left], np.asarray(action_keys)[right]):
            record['first_rng_key_difference'] = {'iteration': iteration}
        state, metrics = trainer.train_iteration(state)
        host_parameters = np.asarray(jax.device_get(state.parameters['table']))
        difference = float(np.max(np.abs(host_parameters[left]-host_parameters[right]), initial=0))
        if difference > 0 and record['first_parameter_difference'] is None:
            record['first_parameter_difference'] = {'after_iteration': iteration+1,
                'maximum_absolute_difference': difference}
        counts = np.asarray(jax.device_get(state.counts))
        if not np.array_equal(counts, np.asarray(jax.device_get(before.counts))):
            record['capture_matches_training_counts'] = False
        if not np.array_equal(counts[left], counts[right]) and record['first_counter_difference'] is None:
            record['first_counter_difference'] = {'after_iteration': iteration+1}
        first_goal = np.asarray(jax.device_get(state.first_goal_step))
        if not np.array_equal(first_goal[left], first_goal[right]) and record['first_goal_time_difference'] is None:
            record['first_goal_time_difference'] = {'after_iteration': iteration+1}
        record['iterations_completed'] = iteration+1
        if record['first_action_difference'] and record['first_counter_difference']:
            record['stop_reason'] = 'first action and visit-count differences observed'
            break
    else:
        record['stop_reason'] = 'maximum iterations completed'
    record['wall_seconds'] = time.monotonic()-started
    return record


def main():
    """Run two diagnostic copy counts on one allocated GPU and preserve the unchanged source identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-iterations', type=int, default=1024)
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--wall-seconds', type=int, default=540)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('portal', 'login'))
            or args.max_iterations < 1 or args.repetitions < 2 or not 0 < args.wall_seconds < 590):
        raise RuntimeError('The bounded repeatability probe requires a compute allocation and valid limits.')
    started = time.monotonic()
    path, shard, pairs, differing, task, submission = reference_shard(args.run_dir)
    package = args.run_dir/'code'/('source_'+submission['source_commit'])/'10_jax_exploration_platform'
    sys.path.insert(0, str(package/'src'))
    import jax
    import numpy as np
    from exploration_platform.grid_initialization.config import GridPPOConfig
    from exploration_platform.grid_initialization.environments import make_geometry
    if jax.default_backend() != 'gpu':
        raise RuntimeError('This diagnostic requires an allocated GPU; it does not run on the monitor CPU.')
    world = make_geometry('snake', 16, 2)
    base = GridPPOConfig(**shard['configuration'])
    result = {'created_at': now(), 'command': sys.argv, 'source_commit': submission['source_commit'],
        'probe_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'training_module_sha256': hashlib.sha256((package/'src/exploration_platform/grid_initialization/training.py').read_bytes()).hexdigest(),
        'hostname': socket.gethostname(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'device': jax.devices()[0].device_kind, 'reference_shard': str(path),
        'reference_job_id': task['job_id'], 'reference_hardware': task['result']['hardware'],
        'reference_different_seed': differing[0], 'cases': [],
        'interpretation': 'A cause is not established merely by observing different trajectories.'}
    all_seeds = [copy['policy_seed'] for copy in shard['copies']]
    for indices in (differing[1], list(range(32))):
        scalars = {name: tuple(values[index] for index in indices) for name, values in base.per_copy_scalar_settings}
        configuration = replace(base, n_copies=len(indices), per_copy_scalar_settings=scalars)
        seeds = [all_seeds[index] for index in indices]
        duplicate = {}
        for index, seed in enumerate(seeds):
            duplicate.setdefault(seed, []).append(index)
        paired = [locations for locations in duplicate.values() if len(locations) == 2]
        record = run_case(configuration, seeds, ([pair[0] for pair in paired], [pair[1] for pair in paired]),
            world, args.max_iterations, args.repetitions, started+args.wall_seconds)
        result['cases'].append(record)
        result['wall_seconds'] = time.monotonic()-started
        write_json(args.output, result)
        jax.clear_caches()
        if time.monotonic() >= started+args.wall_seconds:
            break
    result['completed'] = len(result['cases']) == 2
    result['finished_at'] = now()
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
