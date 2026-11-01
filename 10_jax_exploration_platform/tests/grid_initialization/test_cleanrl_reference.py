"""Compare the grid loss and gradients against statements read from frozen upstream CleanRL."""

from pathlib import Path
from types import SimpleNamespace
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import torch

from exploration_platform.atari_pixels.state import adam_step, initialize_adam
from exploration_platform.grid_initialization.config import GridPPOConfig
from exploration_platform.grid_initialization.training import minibatch_loss


def upstream_loss_code():
    """Read the untouched upstream advantage, policy, value and combined loss statements."""
    path = Path('/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl/cleanrl/ppo_rnd_envpool.py')
    source = path.read_text()
    begin = source.index('                mb_advantages = b_advantages[mb_inds]')
    end = source.index('                optimizer.zero_grad()', begin)
    return compile(textwrap.dedent(source[begin:end]), str(path), 'exec')


def test_tabular_loss_gradients_and_adam_against_upstream_statements():
    """Exercise clipped policy/value branches and compare every state-table derivative."""
    torch.set_num_threads(1)
    configuration = GridPPOConfig(n_copies=1, num_steps=4, num_minibatches=1,
                                  update_epochs=1, total_timesteps=4)
    # Before: three state rows with logits and two values; after: four selected transitions.
    table = np.array([[[.2, -.1, .3, .1, .2, -.3],
                       [-.2, .1, .0, .4, -.2, .4],
                       [.1, .4, -.3, .2, .3, .2]]], np.float32)
    batch = {'states': np.array([[0, 1, 1, 2]], np.int32),
             'actions': np.array([[0, 2, 1, 3]], np.int32),
             'log_probabilities': np.array([[-1.8, -1.2, -1.6, -1.3]], np.float32),
             'values_ext': np.array([[.5, -.5, .1, .0]], np.float32),
             'advantage_ext': np.array([[-1., 2., .5, -.25]], np.float32),
             'advantage_int': np.array([[.3, .1, -.2, .4]], np.float32),
             'return_ext': np.array([[.6, -.8, .7, -.2]], np.float32),
             'return_int': np.array([[.2, -.7, .3, .6]], np.float32)}
    parameters = {'table': jnp.asarray(table)}
    actual_batch = {name: jnp.asarray(value) for name, value in batch.items()}
    (actual_loss, _), gradients = jax.value_and_grad(minibatch_loss, has_aux=True)(
        parameters, actual_batch, jnp.zeros((3, 2), jnp.float32), configuration)
    # Feed table outputs into the original loss statements; no neural or bonus loss remains.
    parameter = torch.nn.Parameter(torch.from_numpy(table.copy()[0]))
    selected = parameter[torch.from_numpy(batch['states'][0]).long()]
    distribution = torch.distributions.Categorical(logits=selected[:, :4])
    log_probability = distribution.log_prob(torch.from_numpy(batch['actions'][0]).long())
    namespace = {'torch': torch, 'args': SimpleNamespace(norm_adv=True, clip_vloss=True,
        clip_coef=configuration.clip_coefficient, ent_coef=configuration.entropy_coefficient,
        vf_coef=configuration.value_coefficient),
        'b_advantages': torch.from_numpy(batch['advantage_ext'][0]*configuration.extrinsic_coefficient
                                        + batch['advantage_int'][0]*configuration.intrinsic_coefficient),
        'mb_inds': np.arange(4), 'ratio': (log_probability-torch.from_numpy(batch['log_probabilities'][0])).exp(),
        'new_ext_values': selected[:, 4:5], 'new_int_values': selected[:, 5:6],
        'b_ext_values': torch.from_numpy(batch['values_ext'][0]),
        'b_ext_returns': torch.from_numpy(batch['return_ext'][0]),
        'b_int_returns': torch.from_numpy(batch['return_int'][0]),
        'entropy': distribution.entropy(), 'forward_loss': torch.zeros((), dtype=torch.float32)}
    exec(upstream_loss_code(), namespace)
    expected_loss = namespace['loss']
    expected_loss.backward()
    np.testing.assert_allclose(np.asarray(actual_loss), expected_loss.detach().numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(gradients['table'][0]), parameter.grad.numpy(), rtol=3e-5, atol=3e-6)
    # The reference also checks clipping and one actual PyTorch Adam step on the same gradients.
    optimizer = torch.optim.Adam([parameter], lr=configuration.learning_rate, eps=configuration.adam_epsilon)
    torch.nn.utils.clip_grad_norm_([parameter], configuration.max_gradient_norm)
    optimizer.step()
    actual_parameters, _, _ = adam_step(parameters, gradients, initialize_adam(parameters),
        jnp.array([configuration.learning_rate], jnp.float32), configuration.max_gradient_norm,
        configuration.adam_epsilon)
    np.testing.assert_allclose(np.asarray(actual_parameters['table'][0]), parameter.detach().numpy(), rtol=3e-5, atol=3e-6)
