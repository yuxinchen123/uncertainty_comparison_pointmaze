"""Profile online tabular-RND training speed at several copy counts on one card (seconds per iteration).

Writes the five throughput columns (copies, seconds per iteration, total steps per second, steps per second per
copy, hours per million steps per copy) for each idea, so copies per process can be set for each card class.
"""

import argparse
import json
from pathlib import Path
import sys
import time

PLATFORM = Path(__file__).resolve().parents[2]


def profile(idea, copies, geometry_kind, warm, timed):
    """Seconds per training iteration of one idea at one copy count, after two warm-up compiles."""
    import jax
    import numpy as np
    from exploration_platform.grid_initialization.config import GridPPOConfig
    from exploration_platform.grid_initialization.environments import make_geometry
    from exploration_platform.grid_initialization.online_rnd import NetworkOptions, stack_networks
    from exploration_platform.grid_initialization.training import GridTrainer
    world = make_geometry('straight', 32, 0) if geometry_kind == 'straight' else make_geometry('snake', 16, 2)
    options = NetworkOptions(input_kind='one_hot', **idea.get('network', {}))
    rnd = stack_networks(np.arange(copies) // 4 % 4, options, world.state_count)
    configuration = GridPPOConfig(n_copies=copies, policy_kind='neural', total_timesteps=2 ** 24,
        episode_length=1 << (8 * world.goal_state - 1).bit_length(), goal_state=world.goal_state,
        intrinsic_source='online_rnd', rnd_input='one_hot', normalize_intrinsic_rewards=True,
        per_copy_scalar_settings={'learning_rate': (1e-3,) * copies}, **idea.get('settings', {}))
    trainer = GridTrainer(configuration, world.transition_table, world.coordinates,
                          np.ones((copies, world.state_count), np.float32), rnd)
    state = trainer.initialize(np.arange(copies, dtype=np.uint32) + 1)
    # Two warm-up rounds: the donated-buffer program can compile a second time (memory note).
    for _ in range(warm):
        state, metrics = trainer.train_iteration(state)
    jax.block_until_ready(metrics)
    started = time.perf_counter()
    for _ in range(timed):
        state, metrics = trainer.train_iteration(state)
    jax.block_until_ready(metrics)
    seconds = (time.perf_counter() - started) / timed
    rate = copies * configuration.num_steps / seconds
    return {'copies': copies, 'seconds_per_iteration': seconds, 'total_steps_per_second_millions': rate / 1e6,
            'steps_per_second_per_copy_thousands': rate / copies / 1e3,
            'hours_per_million_steps_per_copy': 1e6 / (3600 * rate / copies)}


def main():
    """Profile every idea of an idea file at every copy count; stop a copy count that does not fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ideas', type=Path, required=True)
    parser.add_argument('--only', nargs='*', default=None)
    parser.add_argument('--copies', nargs='+', type=int, default=[512, 1024, 2048])
    parser.add_argument('--geometry', default='snake')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timed', type=int, default=30)
    args = parser.parse_args()
    sys.path.insert(0, str(PLATFORM/'src'))
    import jax
    jax.config.update('jax_enable_x64', True)
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Profiling runs on an allocated graphics card.')
    ideas = json.loads(args.ideas.read_text())
    results = {'device': str(jax.devices()[0]), 'geometry': args.geometry, 'policy': 'coordinate-PPO', 'ideas': {}}
    for name, idea in ideas.items():
        if args.only and name not in args.only:
            continue
        rows = []
        for copies in args.copies:
            try:
                rows.append(profile(idea, copies, args.geometry, 2, args.timed))
            except Exception as error:  # A copy count that does not fit the card is recorded, not hidden.
                if 'RESOURCE_EXHAUSTED' not in str(error) and 'out of memory' not in str(error):
                    raise
                rows.append({'copies': copies, 'failure': 'out of memory'})
                break
            print(json.dumps({'idea': name, **rows[-1]}), flush=True)
            jax.clear_caches()
        results['ideas'][name] = rows
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
