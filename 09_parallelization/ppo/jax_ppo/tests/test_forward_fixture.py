"""Cross-framework Test 1 (spec section 17): load the torch forward fixture and assert the
jax forwards reproduce every output within 1e-5 relative.

Run dump_forward_fixture.py (torch env) first, then run this file with the platform's canonical
JAX environment registered in /p/rlprojects/RND/.venvs/ENVS.md (currently
/p/rlprojects/RND/.venvs/platform_jax):
PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python test_forward_fixture.py
"""
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from jax_ppo_rnd import RMSState, JaxPPORND, PPOConfig  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

FIX = Path("/tmp/rnd09_cross_impl/forward_fixture.npz")


def rel_err(a, b):
    """Mixed-tolerance error: max of |a-b| / (atol/rtol + |b|), so the 1e-5 gate equals the
    pass rule |a-b| <= atol + rtol*|b| with rtol 1e-5, atol 1e-6. The absolute floor absorbs
    float32 BLAS accumulation noise on near-zero outputs (the gain-0.01 actor head sits near
    1e-3, where a 1e-7 accumulation difference is 1e-4 in pure relative terms)."""
    return float(np.max(np.abs(a - b) / (1e-6 / 1e-5 + np.abs(b))))


def main():
    # This gate asks whether the two frameworks compute the SAME FUNCTION, so both sides must
    # compute it the same way. JAX's default on this card runs float32 matmuls in reduced
    # precision (TF32), while the fixture was produced exactly on processor cores; comparing the
    # two that way measures the card's matmul mode, not the algorithm, and puts every quantity
    # about 3e-4 out. Asking for exact float32 here brings the whole forward pass to ~1e-7.
    jax.config.update("jax_default_matmul_precision", "highest")
    d = np.load(FIX)
    # import torch parameters into the jax structure — weights are [C, in, out] in BOTH
    net = lambda name, keys: {k: jnp.asarray(d[f"param__{name}__{k}"]) for k in keys}
    actor = net("actor", ["W0", "b0", "W1", "b1", "W2", "b2", "logstd"])
    critic = net("critic", ["W0", "b0", "W1", "b1", "Wext", "bext", "Wint", "bint"])
    target = net("target", ["W0", "b0", "W1", "b1"])
    predictor = net("predictor", ["W0", "b0", "W1", "b1", "W2", "b2"])

    t = JaxPPORND(PPOConfig(n_copies=3, n_envs=2, num_steps=8))
    t.target = target
    params = {"actor": actor, "critic": critic, "predictor": predictor}

    obs = jnp.asarray(d["obs"])
    rms = RMSState(jnp.asarray(d["rms_mean"]), jnp.asarray(d["rms_var"]),
                   jnp.full((3, 1), 1.0, jnp.float64))
    act_mean = t._actor_mean(actor, obs)
    vext, vint = t._critic_values(critic, obs)
    rnd_in = t._whiten(obs, rms)
    tf, pf = t._rnd_features(predictor, rnd_in)
    bonus = 0.5 * ((pf - tf) ** 2).sum(-1)

    checks = [("act_mean", act_mean), ("logstd", actor["logstd"]), ("vext", vext),
              ("vint", vint), ("rnd_in", rnd_in), ("tf", tf), ("pf", pf), ("bonus", bonus)]
    worst = 0.0
    for name, val in checks:
        e = rel_err(np.asarray(val), d[name])
        worst = max(worst, e)
        status = "PASS" if e <= 1e-5 else "FAIL"
        print(f"{status} {name:10s} max rel err {e:.3e}")
        assert e <= 1e-5, name
    print(f"cross-framework forward agreement: PASS (worst {worst:.3e})")

    # The shipped trainer does NOT run at this precision — it takes the card's reduced-precision
    # default, as the torch trainer does with tf32 enabled. Report what that costs, so the number
    # is on the record rather than discovered again later as a mysterious gate failure.
    jax.config.update("jax_default_matmul_precision", "default")
    shipped = max(rel_err(np.asarray(t._actor_mean(actor, obs)), d["act_mean"]),
                  rel_err(np.asarray(t._critic_values(critic, obs)[0]), d["vext"]))
    print(f"same comparison at the shipped matmul precision: {shipped:.3e} "
          f"(reduced precision, expected near 1e-3; not a gate)")


if __name__ == "__main__":
    main()
