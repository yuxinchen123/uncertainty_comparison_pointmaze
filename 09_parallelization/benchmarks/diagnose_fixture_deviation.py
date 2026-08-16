"""Absolute deviation, quantity by quantity, between the JAX forward pass and the torch fixture.

The gate `test_forward_fixture.py` reports one mixed-tolerance number per quantity, which folds
the absolute and relative parts together. When exactly one quantity fails and the rest pass, the
question that decides whether anything is wrong is whether its ABSOLUTE deviation is unusual, or
only its ratio to a small magnitude: the actor head is initialised with gain 0.01, so its outputs
are about a hundred times smaller than the critic's, and identical rounding reads as a far larger
relative error there.

Run (on the graphics-processor host, under the lock):
  bash locks/gpu_run.sh "PYTHONNOUSERSITE=1 <jax python> benchmarks/diagnose_fixture_deviation.py"
"""
import argparse
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "jax_ppo"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jax_ppo_rnd import PPOConfig, JaxPPORND, RMSState  # noqa: E402

FIX = Path("/tmp/rnd09_cross_impl/forward_fixture.npz")


def quantities():
    """Recompute every quantity the gate checks, paired with the fixture's stored value.

    before: the fixture's parameter arrays, laid out [copies, in, out] in both frameworks
    after:  [("act_mean", jax array, torch array), ("vext", ...), ...]
    """
    # goal: rebuild the trainer's parameters from the fixture exactly as the gate does
    d = np.load(FIX)
    net = lambda name, keys: {k: jnp.asarray(d[f"param__{name}__{k}"]) for k in keys}
    actor = net("actor", ["W0", "b0", "W1", "b1", "W2", "b2", "logstd"])
    critic = net("critic", ["W0", "b0", "W1", "b1", "Wext", "bext", "Wint", "bint"])
    trainer = JaxPPORND(PPOConfig(n_copies=3, n_envs=2, num_steps=8))
    trainer.target = net("target", ["W0", "b0", "W1", "b1"])
    predictor = net("predictor", ["W0", "b0", "W1", "b1", "W2", "b2"])

    # goal: the same forward calls the gate makes, on the same stored observations
    obs = jnp.asarray(d["obs"])
    rms = RMSState(jnp.asarray(d["rms_mean"]), jnp.asarray(d["rms_var"]),
                   jnp.full((3, 1), 1.0, jnp.float64))
    vext, vint = trainer._critic_values(critic, obs)
    rnd_in = trainer._whiten(obs, rms)
    tf, pf = trainer._rnd_features(predictor, rnd_in)
    return [("act_mean", trainer._actor_mean(actor, obs), d["act_mean"]),
            ("vext", vext, d["vext"]), ("vint", vint, d["vint"]),
            ("tf", tf, d["tf"]), ("pf", pf, d["pf"])]


def main():
    """Print absolute deviation beside the magnitude of the quantity it belongs to."""
    # goal: let the matmul precision be chosen, because that is the hypothesis under test —
    # JAX's default on this card computes float32 matmuls in reduced precision, which would put
    # every quantity about 1e-3 away from a fixture computed exactly on processor cores
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="default", choices=["default", "highest"])
    args = ap.parse_args()
    jax.config.update("jax_default_matmul_precision", args.precision)
    print(f"matmul precision: {args.precision}")
    print(f"{'quantity':10} {'max absolute':>14} {'magnitude':>12} {'abs/magnitude':>15} "
          f"{'gate metric':>13}")
    for name, computed, stored in quantities():
        a, b = np.asarray(computed), np.asarray(stored)
        scale = max(float(np.abs(b).max()), 1e-12)
        abs_dev = float(np.abs(a - b).max())
        gate = float(np.max(np.abs(a - b) / (1e-6 / 1e-5 + np.abs(b))))
        print(f"{name:10} {abs_dev:14.3e} {scale:12.3e} {abs_dev/scale:15.3e} {gate:13.3e}")
    print("\nThe gate's denominator has a floor of 0.1, so a quantity whose own magnitude is far "
          "below that is judged almost entirely on absolute deviation, while a quantity near 1 is "
          "allowed about ten times more of it.")


if __name__ == "__main__":
    main()
