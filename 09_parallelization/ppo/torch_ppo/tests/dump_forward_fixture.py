"""Dump a shared-weight forward fixture for the cross-framework Test 1 (spec section 17).

Saves per-copy torch parameters (layout [C, in, out] for weights), a fixed observation
batch, fixed RND-input statistics, and every torch output the jax side must reproduce.
Run: PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/exploration/bin/python dump_forward_fixture.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from torch_ppo_rnd import PPOConfig, PPORND  # noqa: E402

OUT = Path("/tmp/rnd09_cross_impl")
C, M = 3, 7


def main():
    torch.manual_seed(7)
    t = PPORND(PPOConfig(n_copies=C, n_envs=2, num_steps=8), device="cpu")

    # per-copy distinct whitening statistics so a copy-axis mixup cannot pass unnoticed
    mean = torch.tensor([[0.1 * c + 0.05 * d for d in range(4)] for c in range(C)],
                        dtype=torch.float64)
    var = torch.tensor([[1.0 + 0.2 * c + 0.1 * d for d in range(4)] for c in range(C)],
                       dtype=torch.float64)
    t.obs_rms.mean, t.obs_rms.var = mean, var

    obs = torch.randn(C, M, 4)
    with torch.no_grad():
        act_mean = t.actor_mean(obs)
        vext, vint = t.critic_values(obs)
        rnd_in = t.whiten(obs)
        tf, pf = t.rnd_features(rnd_in)
        bonus = 0.5 * (pf - tf).square().sum(-1)

    OUT.mkdir(exist_ok=True)
    arrays = {"obs": obs, "rms_mean": mean, "rms_var": var, "act_mean": act_mean,
              "logstd": t.actor["logstd"], "vext": vext, "vint": vint, "rnd_in": rnd_in,
              "tf": tf, "pf": pf, "bonus": bonus}
    # parameters, prefixed by net name; weight layout [C, in, out] exactly as stored
    for net_name, net in [("actor", t.actor), ("critic", t.critic),
                          ("target", t.target), ("predictor", t.predictor)]:
        for k, v in net.items():
            arrays[f"param__{net_name}__{k}"] = v
    np.savez(OUT / "forward_fixture.npz",
             **{k: v.detach().numpy() for k, v in arrays.items()})
    print(f"wrote {OUT}/forward_fixture.npz ({len(arrays)} arrays)")


if __name__ == "__main__":
    main()
