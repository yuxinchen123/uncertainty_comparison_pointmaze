"""The driver: build a state, warm it up, and step the compiled iteration in a loop.

Everything here runs on the host and is deliberately dull. The only device work it does outside
the compiled programs is reading the metrics of the iterations it records, so a long run does not
wait for the device on every step.
"""
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from .. import F32
from ..agents.ppo.config import PPOConfig
from ..agents.ppo.update import opt_init
from ..evaluation.coverage import coverage_fraction
from ..statistics import rms_init
from .compose import Composition
from .state import TrainState, put_params


class Runner:
    """Static configuration plus two compiled programs; all mutable state lives in a TrainState."""

    def __init__(self, cfg: PPOConfig, bonus="rnd_next_state", env_cfg=None):
        """Compose this configuration's environment, agent and bonus, and compile the programs.

        `bonus` is a preset name from `bonuses/registry.py`, or a factory passed straight in.
        """
        self.cfg = cfg
        self.composition = Composition(cfg, bonus, env_cfg)
        self.env = self.composition.env
        self.bonus = self.composition.bonus
        self.sweep = self.composition.sweep
        self.layout = self.composition.layout
        self.iterate = self.composition.iterate
        self.prime_step = self.composition.prime_step
        self.total_loss = self.composition.total_loss

    def init_trainable(self):
        """The starting trainable tree, {"agent": ..., "bonus": ...}, before any packing."""
        return {"agent": self.composition.init_agent_params,
                "bonus": self.composition.init_bonus_params}

    def init_state(self, run_seed: int = 0) -> TrainState:
        """Fresh state: keyed weights, zero Adam moments, reset environments, iteration zero."""
        cfg = self.cfg
        trainable = self.init_trainable()
        # the state stores the trainable tree in whichever form the layout says: one array, or
        # the two named halves
        stored = self.layout.pack(trainable) if self.layout.packed else trainable
        state = TrainState(
            env_state=self.env.reset(),
            obs=jnp.zeros((cfg.n_copies, cfg.n_envs, self.env.obs_dim), F32),
            agent_params={}, bonus_params={},
            opt=opt_init(stored),
            agent_state={"int_filter": jnp.zeros((cfg.n_copies, cfg.n_envs), F32),
                         "int_rms": rms_init(cfg.n_copies, 1)},
            bonus_state=self.composition.init_bonus_state,
            visited=jnp.zeros(
                (cfg.n_copies, self.composition.n_cells if cfg.track_coverage else 0), bool),
            rng=jax.random.PRNGKey(run_seed + 1_000_003 * cfg.base_seed),
            step=jnp.zeros((), jnp.int32))
        return put_params(state, stored, self.layout)

    def prime(self, state: TrainState) -> TrainState:
        """Run the warm-up rollouts, then respawn the environments at the carried generation.

        The respawn zeroes the step counts, mirroring the reset the torch twin performs, so the
        first training episode is a full-length one. A bonus with no warm-up hook skips straight
        to the respawn.
        """
        key = jax.random.fold_in(state.rng, 999999937)
        if self.prime_step is not None:
            for i in range(self.cfg.prime_iterations):
                state = self.prime_step(state, jax.random.fold_in(key, i))
        env_state, obs = self.env.respawn(state.env_state)
        return state._replace(env_state=env_state, obs=obs)

    def lr_argument(self, iteration: int, num_iterations: int):
        """The scalar passed to one iteration, given the annealing schedule.

        Without a rate sweep it is the rate itself; with one the per-copy rates are already a
        device constant, so it is the annealing multiplier that scales all of them together. A
        sweep over the intrinsic weight alone does not touch this: the rate stays a scalar.
        """
        frac = (1.0 - (iteration - 1.0) / num_iterations) if self.cfg.anneal_lr else 1.0
        sweeps_rate = self.sweep.lr_per_copy is not None
        return jnp.asarray(frac if sweeps_rate else self.cfg.learning_rate * frac, F32)

    def coverage(self, state: TrainState):
        """Fraction of the open maze cells each copy has visited, [C] — device to host."""
        return coverage_fraction(state.visited, self.composition.open_cells)

    def train(self, num_iterations, run_seed=0, log_every_seconds=1200, log_fn=print,
              history_every=0):
        """Full loop with sparse logging; returns (state, stats).

        history_every > 0 records per-copy reward, intrinsic reward and maze coverage every that
        many iterations. Those are the only iterations that wait for the device, so the recording
        does not serialise the loop.
        """
        cfg = self.cfg
        state = self.prime(self.init_state(run_seed))
        t0 = time.time()
        last_log = t0
        history = []
        for it in range(1, num_iterations + 1):
            state, metrics = self.iterate(state, self.lr_argument(it, num_iterations))
            now = time.time()
            if history_every and (it % history_every == 0 or it == num_iterations):
                row = {"iteration": it, "seconds": now - t0,
                       "global_step": it * cfg.num_steps * cfg.n_copies * cfg.n_envs,
                       "reward_ext_sum_per_copy": np.asarray(metrics["reward_ext_sum"]).tolist(),
                       "rint_mean_per_copy": np.asarray(metrics["rint_mean"]).tolist()}
                if cfg.track_coverage:
                    row["coverage_per_copy"] = self.coverage(state).tolist()
                history.append(row)
            if now - last_log >= log_every_seconds or it == num_iterations:
                jax.block_until_ready(metrics)
                last_log = now
                r = np.asarray(metrics["reward_ext_sum"])
                cov = (f" coverage mean {self.coverage(state).mean():.3f}"
                       if cfg.track_coverage else "")
                log_fn(f"iter {it}/{num_iterations} loss {float(metrics['loss']):.4f} "
                       f"ext-reward/copy mean {r.mean():.3f} min {r.min():.3f} "
                       f"max {r.max():.3f} rint mean "
                       f"{float(np.asarray(metrics['rint_mean']).mean()):.4f}{cov} "
                       f"elapsed {now - t0:.0f}s")
        jax.block_until_ready(state.agent_params)
        return state, {"iterations": num_iterations,
                       "global_step": num_iterations * cfg.num_steps * cfg.n_copies * cfg.n_envs,
                       "seconds": time.time() - t0,
                       "bonus": self.bonus.name,
                       "learning_rate_per_copy": (np.asarray(self.sweep.lr_per_copy).tolist()
                                                  if self.sweep.lr_per_copy is not None else None),
                       "beta_per_copy": (np.asarray(self.sweep.beta_per_copy).tolist()
                                         if self.sweep.beta_per_copy is not None else None),
                       "group_index": self.sweep.copy_group.tolist(),
                       "group_settings": [list(s) for s in self.sweep.group_settings],
                       "sweep_seed_mode": cfg.sweep_seed_mode if self.sweep.is_sweep else None,
                       "history": history}


def main():
    """Smoke / bench entry."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, default=4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--style", default="epoch_minibatch")
    args = ap.parse_args()
    runner = Runner(PPOConfig(n_copies=args.n_copies, update_style=args.style))
    _, stats = runner.train(args.iters, log_every_seconds=0)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
