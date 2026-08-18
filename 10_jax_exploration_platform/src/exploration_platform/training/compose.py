"""Select the environment, the agent and the bonus, THEN compile.

This is the one place where the three pieces meet. Everything variable about a run — which
environment family, which bonus, which update style, whether the rollout is hoisted, whether
coverage is tracked — is decided here in Python, and only the resulting concrete function is
handed to `jax.jit`. So the compiled program contains exactly one algorithm and no switch over
algorithms; a bonus that is not chosen contributes no arithmetic to it at all, which is what
makes "PPO with no bonus" genuinely cheaper rather than nominally cheaper.

The environment is picked by the TYPE of the configuration object: a PointMaze `EnvConfig`
builds `JaxPointMaze`, an `AntMazeConfig` builds `JaxAntMaze`. The environment then reports its
own observation and action widths, its open-cell mask and its cell indexer, so nothing else in
the trainer knows which family it is running.
"""
import jax

from ..agents.ppo.networks import init_agent_params, init_agent_params_discrete
from ..bonuses.registry import make_bonus
from ..envs.antmaze.am_common import AntMazeConfig
from ..envs.antmaze.mjx_antmaze import JaxAntMaze
from ..envs.atari_montezuma.jax_montezuma import JaxMontezuma, MontezumaConfig
from ..envs.pointmaze.jax_pointmaze import JaxPointMaze
from ..envs.pointmaze.pm_common import EnvConfig
from .state import ParamLayout
from .sweep import build_sweep
from .train_step import build_iteration, build_prime_step, build_total_loss

# which environment class a configuration type selects; a new family adds one row
ENV_CLASSES = {EnvConfig: JaxPointMaze, AntMazeConfig: JaxAntMaze,
               MontezumaConfig: JaxMontezuma}


class Composition:
    """One assembled run: the environment, the bonus, the parameter layout, and two programs."""

    def __init__(self, cfg, bonus, env_cfg=None):
        """Build the environment and the bonus for this configuration, then compile both programs.

        `bonus` is a preset name from the registry, or a factory
        `(cfg, env_cfg, n_copies, base_seed, copy_seed_index, obs_dim) -> BonusFunctions` passed
        straight in, so the family binds to this run before anything is traced.
        """
        self.cfg = cfg
        self.env_cfg = env_cfg or EnvConfig()
        self.sweep = build_sweep(cfg)

        # the environment, with the sweep's seed streams: paired groups meet the same environments
        env_class = ENV_CLASSES[type(self.env_cfg)]
        self.env = env_class(self.env_cfg, cfg.n_copies, cfg.n_envs,
                             base_seed=cfg.base_seed,
                             copy_seed_index=self.sweep.copy_seed_index)
        self.open_cells, self.n_cells = self.env.open_cells, self.env.n_cells

        # the bonus binds to this run before anything is traced
        self.bonus = make_bonus(bonus)(cfg, self.env_cfg, cfg.n_copies, cfg.base_seed,
                                       self.sweep.copy_seed_index, obs_dim=self.env.obs_dim)
        discrete = getattr(self.env, "action_kind", "continuous") == "discrete"
        if discrete:
            self.init_agent_params = init_agent_params_discrete(
                cfg.n_copies, cfg.base_seed, self.sweep.copy_seed_index,
                obs_dim=self.env.obs_dim, n_actions=self.env.n_actions)
        else:
            self.init_agent_params = init_agent_params(cfg.n_copies, cfg.base_seed,
                                                       self.sweep.copy_seed_index,
                                                       obs_dim=self.env.obs_dim,
                                                       act_dim=self.env.act_dim)
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
        self.total_loss = build_total_loss(cfg, self.bonus, self.layout, discrete)

    def iteration_capturing_batch(self):
        """The same iteration, unjitted, also returning the update batch it built.

        Used by the check that the hoisted rollout and the in-scan rollout build the same batch.
        """
        return build_iteration(self.cfg, self.env, self.bonus, self.sweep, self.layout,
                               capture_batch=True)
