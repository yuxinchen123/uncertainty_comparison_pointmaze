"""Check that online coordinate-RND training, every new setting at its default, is bit-identical to the 2026-11-07 code."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from common import PYTHON, now, write_json

# The fixed-settings study's frozen source: the online RND trainer before the 2026-11-09 options.
REFERENCE_SOURCE = Path('/p/rlprojects/RND/10_jax_exploration_platform/runs/'
    '2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-'
    'online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_'
    'confirmation-20000-20127/code/source_5abf6b2f35965445d4dd42c023fa1728aa058bf0/10_jax_exploration_platform/src')
CURRENT_SOURCE = Path(__file__).resolve().parents[2]/'src'
CASES = [(policy, state, reward) for policy in ('tabular', 'neural') for state in (False, True) for reward in (False, True)]


def dump(output):
    """Train every online case for four iterations and save the complete final states."""
    import jax
    import numpy as np
    from exploration_platform.grid_initialization.config import GridPPOConfig
    from exploration_platform.grid_initialization.environments import make_geometry
    from exploration_platform.grid_initialization.online_rnd import stack_networks, warmup_statistics
    from exploration_platform.grid_initialization.training import GridTrainer
    jax.config.update('jax_enable_x64', True)
    world = make_geometry('snake', 4, 2)
    field_seeds = np.asarray([0, 0, 1, 1])
    arrays = {}
    for policy, state_normalization, reward in CASES:
        rnd = stack_networks(field_seeds)
        if state_normalization:
            rows = [warmup_statistics(world, 64, int(seed)) for seed in field_seeds]
            rnd['statistics'] = {name: np.stack([np.asarray(row[name], np.float64) for row in rows])
                                 for name in ('mean', 'variance', 'count')}
        configuration = GridPPOConfig(n_copies=4, policy_kind=policy, hidden_width=16, num_steps=32,
            num_minibatches=4, update_epochs=2, total_timesteps=32*8, episode_length=64,
            normalize_intrinsic_rewards=reward, intrinsic_source='online_rnd',
            normalize_rnd_inputs=state_normalization,
            per_copy_scalar_settings={'intrinsic_coefficient': (0.25, 1., 4., 16.)})
        fields = np.ones((4, world.state_count), np.float32)
        trainer = GridTrainer(configuration, world.transition_table, world.coordinates, fields, rnd)
        state = trainer.initialize([3, 5, 7, 9])
        for _ in range(4):
            state, _ = trainer.train_iteration(state)
        for index, leaf in enumerate(jax.tree.leaves(state)):
            arrays[f'{policy}_{state_normalization}_{reward}_{index}'] = np.asarray(leaf)
    np.savez(output, **arrays)


def main():
    """Run both sources in separate processes on the processor backend and require equal states."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--dump', type=Path)
    args = parser.parse_args()
    if args.dump:
        dump(args.dump)
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('This numerical check requires an allocated compute node.')
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        outputs = {}
        for name, source in (('reference', REFERENCE_SOURCE), ('current', CURRENT_SOURCE)):
            outputs[name] = Path(folder)/f'{name}.npz'
            subprocess.run([str(PYTHON), __file__, '--dump', str(outputs[name])], check=True,
                           env=dict(os.environ, PYTHONPATH=str(source), PYTHONNOUSERSITE='1', JAX_PLATFORMS='cpu'))
        reference, current = np.load(outputs['reference']), np.load(outputs['current'])
        differing = [key for key in reference.files if key not in current.files
                     or not np.array_equal(reference[key], current[key])]
        if set(reference.files) != set(current.files):
            differing.append('different state structure')
    record = {'at': now(), 'passed': not differing, 'differing_leaves': differing,
              'reference_source': str(REFERENCE_SOURCE), 'leaves_compared': len(reference.files)}
    if args.run_dir:
        write_json(args.run_dir/'checks_online_path_unchanged.json', record)
    print(record, flush=True)
    sys.exit(1 if differing else 0)


if __name__ == '__main__':
    main()
