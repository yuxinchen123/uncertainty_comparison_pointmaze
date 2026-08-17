"""No bonus at all: the intrinsic reward is zero everywhere.

This is the arm that says what the extrinsic reward alone achieves, and it is also the proof that
selecting the bonus before compiling is real rather than nominal. The family has no parameters, no
statistics, no warm-up and no loss, so the composed program contains none of any bonus's
arithmetic — not a multiplication by zero, not a masked branch, nothing. The check that it does
not is `tests/bonuses/test_none_has_no_bonus_arithmetic.py`.

The agent's intrinsic value head and its advantage stream still exist; they are the agent's
architecture, not the bonus's. Fed a reward that is exactly zero at every step, the filter stays
at zero, the normalised intrinsic reward stays at zero, and the intrinsic advantage contributes
nothing to the policy gradient.
"""
import jax.numpy as jnp

from .. import F32
from .protocol import BonusFunctions


def build(cfg, env_cfg, n_copies: int, base_seed: int, copy_seed_index):
    """Bind this bonus to one run. Nothing about it actually depends on the run."""
    def init():
        """No parameters and no state."""
        return {}, {}

    def rollout_step(params, state, next_obs):
        """Zero for every one of the [C, M, 4] next observations."""
        return jnp.zeros(next_obs.shape[:2], F32)

    def post_rollout(params, state, next_obs_flat, scored):
        """Zero rewards, no state to move on, no extra fields for the update batch."""
        reward = rollout_step(params, state, next_obs_flat) if scored is None else scored
        return {}, reward, {}

    def loss(params, batch):
        """No loss term; the composer adds this to the agent's per-copy loss."""
        return 0.0

    def metrics(params, state):
        """Nothing beyond the intrinsic reward the composer already reports."""
        return {}

    return BonusFunctions(name="none", init=init, prime=None, rollout_step=rollout_step,
                          post_rollout=post_rollout, loss=loss, metrics=metrics)
