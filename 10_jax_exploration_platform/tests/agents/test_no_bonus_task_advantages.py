"""Verify that the corrected no-bonus policy uses only task advantages."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import jax
import jax.numpy as jnp
import numpy as np
from exploration_platform.agents.ppo.config import PPOConfig
from exploration_platform.training.runner import Runner


def main():
    """Compare zero and historical nonzero intrinsic weights using the captured PPO batch."""
    # Both controls share all seed streams, networks and rollout actions; only the weight differs.
    batches = []
    for weight in (0., 1.):
        cfg = PPOConfig(n_copies=2, n_envs=2, num_steps=16, scan_unroll=1,
                        update_style="full_batch",
                        parameter_settings=((1e-3, weight),), copies_per_group=2,
                        prime_iterations=0)
        runner = Runner(cfg, bonus="none")
        state = runner.prime(runner.init_state(0))
        state, metrics, batch = jax.jit(runner.composition.iteration_capturing_batch())(
            state, jnp.asarray(1., jnp.float32))
        task_only = cfg.ext_coef * (batch["ret_ext"] - batch["vext_old"])
        difference = float(np.max(np.abs(np.asarray(batch["adv"] - task_only))))
        if weight == 0:
            np.testing.assert_allclose(batch["adv"], task_only, rtol=2e-5, atol=2e-5)
            assert (np.asarray(metrics["rint_mean"]) == 0).all()
            print(f"PASS zero-weight control uses task advantages only; error {difference:.3e}",
                  flush=True)
        else:
            assert difference > 1e-3, "historical-weight control did not expose its intrinsic stream"
            print(f"PASS historical control shows a nonzero intrinsic contribution: {difference:.3e}",
                  flush=True)
        batches.append(batch)
    np.testing.assert_array_equal(batches[0]["actions"], batches[1]["actions"])


if __name__ == "__main__":
    main()
