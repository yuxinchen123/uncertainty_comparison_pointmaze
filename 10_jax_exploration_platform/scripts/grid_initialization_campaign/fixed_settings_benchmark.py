"""Measure seconds per iteration of the fixed-settings study's trainers, per card class and copy count.

Two uses: the packing measurement that admits a card class (copies per process for each policy and
bonus source), and the speed screen of the new normalizations (count bonus with and without reward
normalization; online RND with each state/reward normalization pair).
"""

import argparse
import os
from pathlib import Path
import socket
import statistics
import time

from common import now, read_json, write_json

PACKING_CASES = [('tabular', 'count', False, True), ('neural', 'count', False, True),
                 ('tabular', 'online_rnd', True, True), ('neural', 'online_rnd', True, True)]
SCREEN_CASES = [(policy, 'count', False, False) for policy in ('tabular', 'neural')] + [
    (policy, 'online_rnd', state, reward) for policy in ('tabular', 'neural')
    for state in (False, True) for reward in (False, True) if not (state and reward)]


def measure(policy, source, state_normalization, reward_normalization, copies, iterations):
    """Time one trainer on the 50-cell snake grid after its compile; return rates and peak memory."""
    import jax
    import numpy as np
    from exploration_platform.grid_initialization.config import GridPPOConfig
    from exploration_platform.grid_initialization.environments import make_geometry
    from exploration_platform.grid_initialization.online_rnd import stack_networks, warmup_statistics
    from exploration_platform.grid_initialization.training import GridTrainer
    world = make_geometry('snake', 16, 2)
    episode_length = 1 << (8*world.goal_state-1).bit_length()
    field_seeds = np.arange(copies) // 4
    rnd = None
    if source == 'online_rnd':
        rnd = stack_networks(field_seeds)
        if state_normalization:
            rows = [warmup_statistics(world, episode_length, int(seed)) for seed in sorted(set(field_seeds))]
            rnd['statistics'] = {name: np.stack([np.asarray(rows[int(seed)][name], np.float64) for seed in field_seeds])
                                 for name in ('mean', 'variance', 'count')}
    # Per-copy swept settings make the measured program the production program.
    settings = {'learning_rate': tuple([1e-4, 1e-3] * (copies // 2)),
                'intrinsic_coefficient': tuple(2.0 ** ((index % 13) - 6) for index in range(copies)),
                'entropy_coefficient': tuple([1e-3] * copies), 'gamma_int': tuple([0.99] * copies)}
    configuration = GridPPOConfig(n_copies=copies, policy_kind=policy, total_timesteps=2**24,
        episode_length=episode_length, goal_state=world.goal_state, intrinsic_source=source,
        normalize_intrinsic_rewards=reward_normalization,
        normalize_rnd_inputs=source == 'online_rnd' and state_normalization,
        per_copy_scalar_settings=settings)
    fields = np.ones((copies, world.state_count), np.float32)
    trainer = GridTrainer(configuration, world.transition_table, world.coordinates, fields, rnd)
    state = trainer.initialize(np.arange(copies, dtype=np.uint32) + 1)
    started = time.monotonic()
    for _ in range(2):
        state, metrics = trainer.train_iteration(state)
    jax.block_until_ready(state)
    compile_seconds = time.monotonic() - started
    blocks = []
    for _ in range(3):
        started = time.monotonic()
        for _ in range(iterations):
            state, metrics = trainer.train_iteration(state)
        jax.block_until_ready(state)
        blocks.append((time.monotonic() - started) / iterations)
    finite = all(bool(np.isfinite(np.asarray(value)).all()) for value in jax.tree.leaves(metrics))
    memory = jax.devices()[0].memory_stats() or {}
    seconds = statistics.median(blocks)
    total = copies * configuration.num_steps / seconds
    return {'policy_kind': policy, 'intrinsic_source': source, 'state_normalization': state_normalization,
            'reward_normalization': reward_normalization, 'copies': copies, 'seconds_per_iteration': seconds,
            'steps_per_second_total': total, 'steps_per_second_per_copy': total / copies,
            'hours_per_million_steps_per_copy': 1e6 / (3600 * total / copies),
            'compile_seconds': compile_seconds, 'timing_blocks': blocks,
            'peak_device_memory_mb': memory.get('peak_bytes_in_use', 0) / 2**20,
            'device_memory_limit_mb': memory.get('bytes_limit', 0) / 2**20,
            'completed': True, 'correctness': {'passed': finite, 'check': 'all metrics finite'}}


def choose_packing(cases):
    """Pick, per policy and source, the measured copy count with the highest total rate."""
    # Before: cases at 64..512 copies; after: {"neural_online_rnd": {"copies": 256, ...}}.
    chosen = {}
    for case in cases:
        if not case.get('completed') or not case['correctness']['passed']:
            continue
        # A completed case ran at that copy count; the recorded peak is the process's peak over
        # all earlier cases (including any that ran out of memory), so it cannot reject a case.
        key = f"{case['policy_kind']}_{case['intrinsic_source']}"
        if case['steps_per_second_total'] > chosen.get(key, {}).get('steps_per_second_total', 0):
            chosen[key] = {'copies': case['copies'], 'steps_per_second_total': case['steps_per_second_total'],
                           'peak_device_memory_mb': case['peak_device_memory_mb']}
    return chosen


def main():
    """Run the packing measurement (default) or the normalization speed screen (--screen)."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpu-class', required=True)
    parser.add_argument('--screen', action='store_true')
    parser.add_argument('--copies', default='64,128,256,512')
    parser.add_argument('--iterations', type=int, default=5)
    args = parser.parse_args()
    if socket.gethostname().startswith(('portal', 'login')) or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Benchmarks require an allocated compute node.')
    import jax
    jax.config.update('jax_enable_x64', True)
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Benchmarks require a graphics card.')
    # The packing measurement is the card class's compatibility record; the screen is separate.
    output = args.run_dir/'infrastructure'/('speed_screen' if args.screen else 'compatibility')/(args.gpu_class+'.json')
    document = {'started_at': now(), 'gpu_class': args.gpu_class, 'hostname': socket.gethostname(),
                'device_kind': jax.devices()[0].device_kind, 'cases': [], 'completed': False}
    for policy, source, state_normalization, reward_normalization in (SCREEN_CASES if args.screen else PACKING_CASES):
        for copies in map(int, args.copies.split(',')):
            try:
                case = measure(policy, source, state_normalization, reward_normalization, copies, args.iterations)
            except Exception as error:  # A memory refusal ends larger copy counts for this case only.
                if 'RESOURCE_EXHAUSTED' not in str(error) and 'out of memory' not in str(error).lower():
                    raise
                document['cases'].append({'policy_kind': policy, 'intrinsic_source': source,
                    'state_normalization': state_normalization, 'reward_normalization': reward_normalization,
                    'copies': copies, 'completed': False, 'failure': 'device memory exhausted'})
                jax.clear_caches()
                break
            document['cases'].append(case)
            write_json(output, document)
            jax.clear_caches()
    document.update(completed=True, finished_at=now())
    write_json(output, document)
    if not args.screen:
        write_json(args.run_dir/'infrastructure/packing'/(args.gpu_class+'.json'),
                   dict(choose_packing(document['cases']), measured_at=now(), source=str(output)))


if __name__ == '__main__':
    main()
