"""Generate the bit-identity golden for the RND class.

Run this on the PRE-EDIT working tree BEFORE adding the original-RND switches
(rnd_activation / rnd_predictor_extra_layers / rnd_update_proportion / rnd_lr exposure /
env-steps warmup). It pins, for a set of representative CURRENT configs, the constructed
network parameters, the compute() bonus, and the predictor parameters after 5 update() steps.
After the edits, the golden test (test_original_rnd_golden.py) re-runs the SAME constructions
(whose new knobs default to relu / 0 extra layers / update_proportion 1.0) and asserts every
tensor is byte-identical, proving the defaults did not shift any RNG stream.

Usage:
    /p/rlprojects/RND/.venvs/exploration/bin/python tests/methods/rnd/goldens/_generate.py
"""
import os

import numpy as np
import torch

from rnd_exploration.methods.rnd import RND

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "preedit_v0.pt")

OBS_DIM = 4
OUTPUT_DIM = 16
N = 8


def fixed_batch(seed):
    """Build a deterministic samples dict (observations / next_observations / actions)."""
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((N, OBS_DIM)).astype(np.float32),
        "next_observations": rng.standard_normal((N, OBS_DIM)).astype(np.float32),
        "actions": rng.standard_normal((N, 2)).astype(np.float32),
    }


def flat_params(module):
    """Flatten a module's parameters into one 1-D tensor in state_dict order (stable capture)."""
    # concatenate every parameter tensor flattened, in sorted-key order for determinism
    sd = module.state_dict()
    return torch.cat([sd[k].reshape(-1) for k in sorted(sd)]) if sd else torch.zeros(0)


# The representative CURRENT configs. Each MUST stay bit-identical after the edits, because the
# new knobs default to the current behavior. Keys: label -> constructor kwargs (no new knobs).
CONFIGS = {
    # default benchmark path: adam, mse, relu, zero bias, no obs/reward norm
    "default": dict(optimizer="adam", bonus_readout="mse"),
    # keyed nonzero-bias stream (run 3.2.3 V-arm): pins the _apply_bias_init {0,2} keys survive
    "bias_normal": dict(bias_init="normal_0.5", bias_seed=111),
    # sgd1t + l2 readout (run 3.2.4 V1 family): pins the sgd schedule + l2 readout path
    "sgd1t_l2": dict(optimizer="sgd1t", bonus_readout="l2", sgd_eta0=0.1, sgd_t0=1e4),
    # reward normalization on (run 3.2.3 N1): pins the forward-filter + division path
    "reward_norm": dict(reward_norm=True, reward_norm_gamma=0.99),
}


def capture_one(kwargs):
    """Construct one RND under a fixed seed and capture init params, compute, and post-5-update state."""
    # same seed for every config so any RNG-order shift shows up as a changed init tensor
    torch.manual_seed(0)
    model = RND(obs_shape=(OBS_DIM,), output_dim=OUTPUT_DIM, device="cpu",
                use_obs_norm=False, action_dim=2, **kwargs)
    batch = fixed_batch(seed=3)
    rec = {
        "init_predictor": flat_params(model.predictor),
        "init_target": flat_params(model.target),
        "compute_pre": model.compute(batch).detach().reshape(-1),
    }
    # run 5 predictor updates on a fixed batch; feed observe() too so reward_norm exercises its filter
    for _ in range(5):
        if kwargs.get("reward_norm"):
            model.observe(fixed_batch(seed=7))
        model.update(batch)
    rec["post5_predictor"] = flat_params(model.predictor)
    rec["compute_post"] = model.compute(batch).detach().reshape(-1)
    return rec


def main():
    """Capture every config and torch.save the golden dict."""
    golden = {label: capture_one(kw) for label, kw in CONFIGS.items()}
    torch.save(golden, OUT)
    # print a small summary so the operator can eyeball that the tensors are non-trivial
    for label, rec in golden.items():
        print(f"{label}: compute_pre[:3]={rec['compute_pre'][:3].tolist()} "
              f"post5[:3]={rec['post5_predictor'][:3].tolist()}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
