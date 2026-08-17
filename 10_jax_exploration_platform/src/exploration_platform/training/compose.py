"""Select the environment, the agent and the bonus, THEN compile.

This is the one place where the three pieces meet. Everything variable about a run — which bonus,
which update style, whether the rollout is hoisted, whether coverage is tracked — is decided here
in Python, and only the resulting concrete function is handed to `jax.jit`. So the compiled
program contains exactly one algorithm and no switch over algorithms; a bonus that is not chosen
contributes no arithmetic to it at all, which is what makes "PPO with no bonus" genuinely cheaper
rather than nominally cheaper.
"""
import jax

from ..agents.ppo.networks import init_agent_params
from ..envs.pointmaze.jax_pointmaze import JaxPointMaze
from ..envs.pointmaze.pm_common import EnvConfig
from ..evaluation.coverage import open_cell_mask
from .state import ParamLayout
from .sweep import build_sweep
from .train_step import build_iteration, build_prime_step, build_total_loss


class Composition:
    """One assembled run: the environment, the bonus, the parameter layout, and two programs."""

    def __init__(self, cfg, bonus_factory, env_cfg: EnvConfig = None):
        """Build the environment and the bonus for this configuration, then compile both programs.

        bonus_factory(cfg, n_copies, base_seed, copy_seed_index) -> BonusFunctions, so the family
        binds to this run's copy count and seeds before anything is traced.
        """
        self.cfg = cfg
        self.env_cfg = env_cfg or EnvConfig()
        self.sweep = build_sweep(cfg)

        # the environment, with the sweep's seed streams: paired groups meet the same environments
        self.env = JaxPointMaze(self.env_cfg, cfg.n_copies, cfg.n_envs,
                                base_seed=cfg.base_seed,
                                copy_seed_index=self.sweep.copy_seed_index)
        self.open_cells, self.n_cells = open_cell_mask(self.env_cfg.map_name)

        # the bonus binds to this run before anything is traced
        self.bonus = bonus_factory(cfg, cfg.n_copies, cfg.base_seed, self.sweep.copy_seed_index)
        self.init_agent_params = init_agent_params(cfg.n_copies, cfg.base_seed,
                                                   self.sweep.copy_seed_index)
        self.init_bonus_params, self.init_bonus_state = self.bonus.init()

        # the trainable tree is {"agent": ..., "bonus": ...}; its layout is fixed once, here, so
        # the one-array form always cuts the same columns out of the same order
        self.layout = ParamLayout({"agent": self.init_agent_params,
                                   "bonus": self.init_bonus_params}, cfg.flat_params)

        # the WHOLE iteration (rollout + statistics + advantages + update) is one program; the
        # state argument is donated so parameters, optimizer and environment buffers are updated
        # in place on the device
        self.iterate = jax.jit(build_iteration(cfg, self.env, self.bonus, self.sweep,
                                               self.layout), donate_argnums=(0,))
        prime_step = build_prime_step(cfg, self.env, self.bonus, self.layout)
        self.prime_step = jax.jit(prime_step) if prime_step is not None else None
        self.total_loss = build_total_loss(cfg, self.bonus, self.layout)

    def iteration_capturing_batch(self):
        """The same iteration, unjitted, also returning the update batch it built.

        Used by the check that the hoisted rollout and the in-scan rollout build the same batch.
        """
        return build_iteration(self.cfg, self.env, self.bonus, self.sweep, self.layout,
                               capture_batch=True)
