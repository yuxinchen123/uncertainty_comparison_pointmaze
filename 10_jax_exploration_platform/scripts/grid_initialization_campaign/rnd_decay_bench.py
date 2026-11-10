"""Distillation bench: train only the online tabular-RND predictor on a fixed visitation law, and measure its start and decay.

The update has the RL shape (rollouts of 128 states, 4 epochs x 4 minibatches, each sample kept with
probability 0.25) and uses the trainer's own predictor loss, optimizer step and bonus readout, so a
property measured here is the one the RL runs will have, without the policy.
"""

import argparse
import json
from pathlib import Path
import sys
import time

PLATFORM = Path(__file__).resolve().parents[2]
VISITATION_LAWS = ('uniform', 'start-heavy')


def visitation_probabilities(law, state_count):
    """Probability of each cell in a rollout: uniform, or falling to e^-3 of the start's along the path."""
    import numpy as np
    # Before: 'start-heavy', 4 cells; after: [0.53, 0.26, 0.13, 0.07] (proportional to e^{-3 s/(S-1)}).
    if law == 'uniform':
        return np.full(state_count, 1.0 / state_count)
    weights = np.exp(-3.0 * np.arange(state_count) / (state_count - 1))
    return weights / weights.sum()


def checkpoint_iterations(iterations):
    """Powers of two from 1 through the last iteration, plus the last one."""
    points = [1 << exponent for exponent in range(iterations.bit_length()) if 1 << exponent <= iterations]
    return points if points[-1] == iterations else points + [iterations]


def run_idea(idea, geometry_kind, law, iterations, field_seeds):
    """Train one idea's predictors (one copy per field seed) and return the bonus and visit histories."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from exploration_platform.atari_pixels.state import adam_step, initialize_adam
    from exploration_platform.grid_initialization.config import GridPPOConfig
    from exploration_platform.grid_initialization.environments import make_geometry
    from exploration_platform.grid_initialization.online_rnd import NetworkOptions, stack_networks
    from exploration_platform.grid_initialization.training import (
        all_cell_errors, initialize, predictor_sample_losses, predictor_step)
    world = make_geometry(geometry_kind, 32, 0) if geometry_kind == 'straight' else make_geometry('snake', 16, 2)
    copies, states = len(field_seeds), world.state_count
    options = NetworkOptions(input_kind='one_hot', **idea.get('network', {}))
    rnd = stack_networks(np.asarray(field_seeds), options, states)
    configuration = GridPPOConfig(n_copies=copies, intrinsic_source='online_rnd', rnd_input='one_hot',
                                  total_timesteps=128 * (1 << 17), **idea.get('settings', {}))
    state = initialize(configuration, world.transition_table, world.coordinates,
                       np.ones((copies, states), np.float32), np.arange(copies) + 1, rnd)
    shared = configuration.rnd_optimizer == 'shared_adam'
    optimizer = initialize_adam(state.parameters['predictor']) if shared else state.predictor_optimizer
    probabilities = jnp.asarray(visitation_probabilities(law, states), jnp.float32)
    coordinates = jnp.asarray(world.coordinates, jnp.float32)
    rate = float(idea.get('shared_learning_rate', 1e-3))

    def iteration(carry, index):
        """One rollout's states, then 4 epochs x 4 minibatches of masked predictor updates."""
        predictor, optimizer, counts = carry
        key = jax.random.fold_in(jax.random.PRNGKey(7), index)
        rollout = jax.random.choice(key, states, (copies, 128), p=probabilities)
        counts = counts + jax.vmap(lambda row: jnp.bincount(row, length=states))(rollout).astype(jnp.int32)
        inputs = state.rnd_table[rollout]

        def minibatch(carry, piece):
            """Update every copy's predictor on one 32-state minibatch."""
            predictor, optimizer = carry
            rows, kept = piece

            def loss(values):
                """Masked mean of the per-sample predictor losses, as in the RL update."""
                chosen = jax.vmap(lambda table, ids: table[ids])(inputs, rows)
                errors = predictor_sample_losses(values, state.target, chosen, configuration)
                mask = jax.vmap(lambda flags, ids: flags[ids])(kept, rows).astype(jnp.float32)
                return ((errors * mask).sum(axis=1) / jnp.maximum(mask.sum(axis=1), 1.0)).sum()

            gradients = jax.grad(loss)(predictor)
            if shared:
                # The RL predictor shares the policy's Adam (learning rate annealed over 2^24 steps) and clip.
                rates = jnp.full((copies,), rate * (1.0 - index.astype(jnp.float32) / (1 << 17)), jnp.float32)
                predictor, optimizer, _ = adam_step(predictor, gradients, optimizer, rates, 0.5, 1e-5)
            else:
                predictor, optimizer = predictor_step(predictor, gradients, optimizer, configuration)
            return (predictor, optimizer), None

        pieces = []
        for epoch in range(4):
            order = jax.random.permutation(jax.random.fold_in(key, 100 + epoch), 128).reshape(4, 32)
            kept = jax.random.uniform(jax.random.fold_in(key, 200 + epoch), (copies, 128)) < 0.25
            pieces += [(jnp.broadcast_to(order[block], (copies, 32)), kept) for block in range(4)]
        stacked = jax.tree.map(lambda *values: jnp.stack(values), *pieces)
        (predictor, optimizer), _ = jax.lax.scan(minibatch, (predictor, optimizer), stacked)
        return (predictor, optimizer, counts), None

    @jax.jit
    def advance(carry, start, stop):
        """Run iterations [start, stop) inside one compiled loop."""
        return jax.lax.fori_loop(start, stop, lambda index, value: iteration(value, index)[0], carry)

    def bonus(predictor):
        """Every cell's bonus under the idea's readout."""
        parameters = dict(state.parameters, predictor=predictor)
        return np.asarray(all_cell_errors(state._replace(parameters=parameters), coordinates, configuration))

    carry = (state.parameters['predictor'], optimizer, jnp.zeros((copies, states), jnp.int32))
    history, visits, done = [bonus(carry[0])], [np.zeros((copies, states), np.int64)], 0
    started = time.monotonic()
    for point in checkpoint_iterations(iterations):
        carry = advance(carry, done, point)
        done = point
        history.append(bonus(carry[0]))
        visits.append(np.asarray(carry[2]))
    return np.stack(history), np.stack(visits), time.monotonic() - started


def measures(history, visits):
    """Start spread, per-cell decay slopes, and how far each cell's bonus is from c/sqrt(own visits)."""
    import numpy as np
    initial = history[0]
    ratio = history / initial[None]
    slopes = []
    for copy in range(history.shape[1]):
        for cell in range(history.shape[2]):
            n, q = visits[:, copy, cell], ratio[:, copy, cell]
            usable = (n >= 2) & (q > 0)
            if usable.sum() >= 3 and np.ptp(np.log(n[usable])) > 0:
                slopes.append(np.polyfit(np.log(n[usable]), np.log(q[usable]), 1)[0])
    slopes = np.asarray(slopes)
    final, final_visits = history[-1], np.maximum(visits[-1], 1)
    # sd over cells of log(b_T sqrt(N_T)): zero when every cell's bonus is the same function c/sqrt(N) of its own visits.
    consistency = np.std(np.log(np.maximum(final, 1e-30) * np.sqrt(final_visits)), axis=1)
    target = np.minimum(1.0, np.maximum(visits, 1) ** -0.5)
    deviation = np.sqrt(np.mean((np.log(np.maximum(ratio[1:], 1e-30)) - np.log(target[1:])) ** 2, axis=0))
    return {'start_max_over_min': float(np.median(initial.max(axis=1) / initial.min(axis=1))),
            'start_sd_log': float(np.median(np.std(np.log(initial), axis=1))),
            'slope_median': float(np.median(slopes)) if len(slopes) else None,
            'slope_quartiles': [float(value) for value in np.quantile(slopes, [0.25, 0.75])] if len(slopes) else None,
            'late_count_consistency_sd_log': float(np.median(consistency)),
            'deviation_from_inverse_square_root_mean': float(deviation.mean()),
            'deviation_from_inverse_square_root_worst_cell': float(np.median(deviation.max(axis=1))),
            'final_visits_median': float(np.median(visits[-1])), 'finite': bool(np.isfinite(history).all())}


def main():
    """Run every idea of an idea file on both grids and both visitation laws; write one JSON per idea."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ideas', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=1 << 13)
    parser.add_argument('--field-seeds', type=int, default=8)
    parser.add_argument('--only', nargs='*', default=None)
    args = parser.parse_args()
    sys.path.insert(0, str(PLATFORM/'src'))
    import jax
    import numpy as np
    jax.config.update('jax_enable_x64', True)
    if jax.default_backend() != 'gpu':
        raise RuntimeError('The bench runs on an allocated graphics card.')
    args.output.mkdir(parents=True, exist_ok=True)
    ideas = json.loads(args.ideas.read_text())
    for name, idea in ideas.items():
        if args.only and name not in args.only:
            continue
        result = {'idea': name, 'definition': idea, 'iterations': args.iterations, 'device': str(jax.devices()[0])}
        for geometry in ('straight', 'snake'):
            for law in VISITATION_LAWS:
                history, visits, seconds = run_idea(idea, geometry, law, args.iterations, list(range(args.field_seeds)))
                result[f'{geometry}_{law}'] = {**measures(history, visits), 'seconds': seconds}
                np.savez_compressed(args.output/f'{name}_{geometry}_{law}.npz', bonus=history, visits=visits)
        (args.output/f'{name}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({'idea': name, **{key: {k: value[k] for k in ('start_max_over_min', 'slope_median',
              'late_count_consistency_sd_log', 'seconds')} for key, value in result.items() if '_' in key and
              isinstance(value, dict)}}), flush=True)


if __name__ == '__main__':
    main()
