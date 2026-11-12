"""The phase profile's synthetic batch takes its widths from the environment, and the update
stage accepts it for both families' widths.

Run on a compute node: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> -m pytest <file>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "harness"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.agents.ppo.update import build_update  # noqa: E402
from exploration_platform.envs.antmaze.mjx_antmaze import JaxAntMaze  # noqa: E402
from exploration_platform.envs.pointmaze.jax_pointmaze import JaxPointMaze  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402
from exploration_platform.training.state import stored_params  # noqa: E402
from profile_jax_phases import make_batch  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def test_batch_widths_come_from_the_environment():
    """PointMaze widths 4 and 2, AntMaze widths 29 and 8, at 16 environments per copy."""
    cfg = PPOConfig(n_copies=2, n_envs=16, num_steps=8, update_style="epoch_minibatch")
    for env_class in (JaxPointMaze, JaxAntMaze):
        batch = make_batch(cfg, jax.random.PRNGKey(0), env_class.obs_dim, env_class.act_dim)
        assert batch["obs"].shape == batch["rnd_input"].shape == (2, 128, env_class.obs_dim)
        assert batch["actions"].shape == (2, 128, env_class.act_dim)
        assert all(batch[name].shape == (2, 128) for name in
                   ("old_logprob", "adv", "ret_ext", "ret_int", "vext_old"))
    print("PASS the synthetic batch has the environment's observation and action widths")


def test_update_stage_accepts_the_batch():
    """The update the profile times runs on the synthetic batch and returns a finite loss."""
    runner = Runner(PPOConfig(n_copies=2, n_envs=16, num_steps=8, prime_iterations=1,
                              update_style="epoch_minibatch"))
    cfg = runner.cfg
    state = runner.prime(runner.init_state(run_seed=0))
    batch = make_batch(cfg, jax.random.PRNGKey(1), runner.env.obs_dim, runner.env.act_dim)
    update = jax.jit(build_update(cfg, runner.total_loss, runner.sweep.lr_per_copy))
    _, _, loss = update(stored_params(state, runner.layout), state.opt, batch,
                        jnp.asarray(cfg.learning_rate, jnp.float32), jax.random.PRNGKey(2))
    assert np.isfinite(float(loss))
    print(f"PASS the update stage runs on the synthetic batch, loss {float(loss):.4f}")


if __name__ == "__main__":
    test_batch_widths_come_from_the_environment()
    test_update_stage_accepts_the_batch()
