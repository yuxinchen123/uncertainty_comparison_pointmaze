"""Check that count-bonus training is bit-identical to the code before the online-RND changes."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from common import PYTHON, now, write_json

# The earlier study's final control snapshot holds the grid trainer before 2026-11-07.
REFERENCE_SOURCE = Path('/p/rlprojects/RND/10_jax_exploration_platform/runs/'
    '2026-10-31-21-48_initial-bonus_uniform-random-adversarial-neural_policy-tabular-neural_count-inverse-square-root_'
    'steps-262144-to-16777216_seeds-development-0-15_confirmation-10000-10511/code/'
    'source_0d14eb307f789689f976d2dd24bf4bd1e73a22df/10_jax_exploration_platform/src')
CURRENT_SOURCE = Path(__file__).resolve().parents[2]/'src'
CASES = [(policy, normalized) for policy in ('tabular', 'neural') for normalized in (False, True)]


def dump(output):
    """Train every case for four iterations and save the complete final states."""
    import jax
    import numpy as np
    from exploration_platform.grid_initialization.config import GridPPOConfig
    from exploration_platform.grid_initialization.environments import make_geometry
    from exploration_platform.grid_initialization.training import GridTrainer
    jax.config.update('jax_enable_x64', True)
    world = make_geometry('snake', 4, 2)
    arrays = {}
    for policy, normalized in CASES:
        # Settings are explicit so the changed defaults cannot enter the comparison.
        configuration = GridPPOConfig(n_copies=4, policy_kind=policy, hidden_width=16, num_steps=32,
            num_minibatches=4, update_epochs=2, total_timesteps=32*8, episode_length=64,
            learning_rate=2**-8, entropy_coefficient=2**-10, normalize_intrinsic_rewards=normalized,
            per_copy_scalar_settings={'intrinsic_coefficient': (0.25, 1., 4., 16.)})
        fields = np.random.default_rng(5).uniform(0.1, 3., (4, world.state_count)).astype(np.float32)
        trainer = GridTrainer(configuration, world.transition_table, world.coordinates, fields)
        state = trainer.initialize([3, 5, 7, 9])
        for _ in range(4):
            state, _ = trainer.train_iteration(state)
        for index, leaf in enumerate(jax.tree.leaves(state)):
            arrays[f'{policy}_{normalized}_{index}'] = np.asarray(leaf)
    np.savez(output, **arrays)


def main():
    """Run both sources in separate processes on the processor backend and require equal states.

    The graphics card's scatter-add order is not repeatable even for identical code (measured in
    the earlier study), so bit-identity is checked with the deterministic processor backend.
    """
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
        differing = [key for key in reference.files if not np.array_equal(reference[key], current[key])]
        if set(reference.files) != set(current.files):
            differing.append('different state structure')
    record = {'at': now(), 'passed': not differing, 'differing_leaves': differing,
              'reference_source': str(REFERENCE_SOURCE), 'leaves_compared': len(reference.files)}
    if args.run_dir:
        write_json(args.run_dir/'checks_count_path_unchanged.json', record)
    print(record, flush=True)
    sys.exit(1 if differing else 0)


if __name__ == '__main__':
    main()
