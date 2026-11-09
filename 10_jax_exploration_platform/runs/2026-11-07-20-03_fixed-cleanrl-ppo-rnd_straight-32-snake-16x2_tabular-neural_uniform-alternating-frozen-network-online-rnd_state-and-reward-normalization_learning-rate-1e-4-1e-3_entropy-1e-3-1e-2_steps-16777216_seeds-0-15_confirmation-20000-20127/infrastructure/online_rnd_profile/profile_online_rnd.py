"""Split one online-RND iteration's time into its parts (15-minute screen on one card)."""
import json, sys, time
import jax, jax.numpy as jnp, numpy as np
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, sys.argv[1])
from exploration_platform.grid_initialization.config import GridPPOConfig
from exploration_platform.grid_initialization.environments import make_geometry
from exploration_platform.grid_initialization.online_rnd import stack_networks, jax_forward
from exploration_platform.grid_initialization.training import GridTrainer, prepare_batch, collect_rollout, all_cell_errors
from exploration_platform.atari_pixels.state import adam_step, initialize_adam

def clock(function, *arguments, repeats=5):
    """Median seconds of a compiled call after two warm-ups."""
    for _ in range(2): jax.block_until_ready(function(*arguments))
    times = []
    for _ in range(repeats):
        start = time.perf_counter(); jax.block_until_ready(function(*arguments)); times.append(time.perf_counter() - start)
    return float(np.median(times))

out = {'device': jax.devices()[0].device_kind}
world = make_geometry('snake', 16, 2)
for copies in (128, 512):
    rnd = stack_networks(np.arange(copies) // 4)
    for policy in ('tabular', 'neural'):
        config = GridPPOConfig(n_copies=copies, policy_kind=policy, total_timesteps=2**24, episode_length=512,
                               goal_state=world.goal_state, intrinsic_source='online_rnd')
        trainer = GridTrainer(config, world.transition_table, world.coordinates, np.ones((copies, 50), np.float32), rnd)
        state = trainer.initialize(np.arange(copies, dtype=np.uint32) + 1)
        step = jax.jit(trainer._iteration)
        out[f'{policy}_{copies}_full_iteration'] = clock(lambda s: trainer.train_iteration(s), state)
        out[f'{policy}_{copies}_rollout'] = clock(jax.jit(lambda s: collect_rollout(s, config, trainer.transition_table, trainer.coordinates, all_cell_errors(s, trainer.coordinates, config))), state)
    predictor = jax.tree.map(jnp.asarray, rnd['predictor']); target = jax.tree.map(jnp.asarray, rnd['target'])
    inputs = jnp.asarray(np.random.default_rng(0).normal(size=(copies, 32, 2)), jnp.float32)
    mask = jnp.asarray(np.random.default_rng(1).random((copies, 32)) < 0.25)
    def loss(p):
        e = ((jax_forward(p, inputs) - jax.lax.stop_gradient(jax_forward(target, inputs))) ** 2).mean(-1)
        return ((e * mask).sum(1) / jnp.maximum(mask.sum(1), 1)).sum()
    grad = jax.jit(jax.grad(loss))
    g = grad(predictor)
    optimizer = initialize_adam(predictor)
    rates = jnp.full((copies,), 1e-4)
    adam = jax.jit(lambda p, g, o: adam_step(p, g, o, rates, 0.5, 1e-5))
    out[f'predictor_{copies}_gradient_one_minibatch'] = clock(grad, predictor)
    out[f'predictor_{copies}_adam_one_minibatch'] = clock(adam, predictor, g, optimizer)
    out[f'predictor_{copies}_adam_fused_one_minibatch'] = clock(jax.jit(lambda p, g, o: adam_step(p, g, o, rates, 0.5, 1e-5, fuse_optimizer=True)), predictor, g, optimizer)
    print(json.dumps(out, indent=1), flush=True)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
