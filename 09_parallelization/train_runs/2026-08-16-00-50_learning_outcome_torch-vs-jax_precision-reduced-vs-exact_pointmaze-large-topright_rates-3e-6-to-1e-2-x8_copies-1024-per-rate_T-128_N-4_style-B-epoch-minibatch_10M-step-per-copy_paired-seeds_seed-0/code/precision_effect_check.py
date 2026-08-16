"""Prove the precision knob reaches the trainer's real path, not just a bare multiplication.

A bare matrix multiplication is compiled and dispatched differently from the trainer's, which runs
inside a compiled function and, in PyTorch, inside a captured graph. So the probe in the campaign
driver is necessary but not sufficient: this check runs the ACTUAL trainer for a few iterations and
dumps its parameters, once per precision, each in its own process. Two things then have to hold:

- the same precision twice gives BITWISE identical parameters (so the run is deterministic and any
  difference is attributable to the thing that changed, not to noise);
- the two precisions give DIFFERENT parameters (so the knob was not silently ignored).

Usage:
  python precision_effect_check.py dump  --framework torch --precision reduced --out a.npy
  python precision_effect_check.py dump  --framework torch --precision reduced --out b.npy
  python precision_effect_check.py dump  --framework torch --precision exact   --out c.npy
  python precision_effect_check.py compare --repeat a.npy b.npy --other c.npy --out check.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[3]
RATES = (1e-4, 3e-4)     # two groups, so the sweep machinery is exercised as in the campaign


def dump_torch(precision, copies, iters, seed):
    """Train the real PyTorch trainer for `iters` iterations; return its flat parameters."""
    import torch
    sys.path.insert(0, str(BASE / "ppo" / "torch_ppo"))
    from torch_ppo_rnd import PPORND, sweep_config

    cfg = sweep_config(RATES, copies // len(RATES), base_seed=seed,
                       tf32=(precision == "reduced"))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    trainer = PPORND(cfg, device="cuda")
    trainer.prime_obs_rms()
    trainer._build_iteration_graph()
    for it in range(1, iters + 1):
        trainer._lr_scale.fill_(1.0 - (it - 1.0) / iters)
        trainer.iteration_captured()
    return (trainer._flat.detach().cpu().numpy(),
            {"declared_setting": torch.get_float32_matmul_precision(),
             "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)})


def dump_jax(precision, copies, iters, seed):
    """Train the real JAX trainer for `iters` iterations; return its packed parameters."""
    import jax
    if precision == "exact":
        jax.config.update("jax_default_matmul_precision", "highest")
    sys.path.insert(0, str(BASE / "ppo" / "jax_ppo"))
    from jax_ppo_rnd import JaxPPORND, sweep_config

    cfg = sweep_config(RATES, copies // len(RATES), base_seed=seed)
    trainer = JaxPPORND(cfg)
    key = jax.random.PRNGKey(seed)
    state = trainer.init_state()
    state = trainer.prime_obs_rms(state, jax.random.fold_in(key, 999999937))
    for it in range(1, iters + 1):
        state, _ = trainer._iterate(state, jax.random.fold_in(key, it),
                                    trainer.lr_argument(it, iters))
    return (np.asarray(trainer.pack(state.params)),
            {"declared_setting": str(jax.config.jax_default_matmul_precision)})


def compare(repeat_a, repeat_b, other):
    """Determinism within a precision, and the size of the difference between two precisions.

    The verdict is a RATIO, not an equality: two processes at the same precision are bitwise
    identical in PyTorch but not in JAX, where the compiler benchmarks matrix-multiply algorithms
    at build time and can pick different ones in different processes. So the question the check
    answers is whether changing the precision moves the parameters by much more than repeating
    the same precision does. Ten times is the bar.
    """
    a, b, c = (np.load(p) for p in (repeat_a, repeat_b, other))
    scale = float(np.abs(a).max())
    # before: two parameter vectors of the same length; after: their largest disagreement,
    # absolute and relative to the largest parameter magnitude
    same_max = float(np.abs(a - b).max())
    diff_max = float(np.abs(a - c).max())
    decisive = diff_max > max(10.0 * same_max, 0.0) and diff_max > 0
    return {
        "n_parameters": int(a.size),
        "largest_parameter_magnitude": scale,
        "repeat_bitwise_identical": bool(np.array_equal(a, b)),
        "repeat_largest_difference": same_max,
        "across_precision_largest_difference": diff_max,
        "across_precision_relative_difference": diff_max / scale,
        "ratio_across_precision_to_repeat": (diff_max / same_max if same_max else float("inf")),
        "verdict": ("the precision knob changes the trained parameters, by far more than "
                    "repeating the same precision does"
                    if decisive else
                    "INCONCLUSIVE — the two differences are the same size, so the knob's effect "
                    "cannot be separated from run-to-run variation"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "compare"])
    ap.add_argument("--framework", choices=["torch", "jax"])
    ap.add_argument("--precision", choices=["reduced", "exact"])
    ap.add_argument("--copies", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", nargs=2)
    ap.add_argument("--other")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.mode == "dump":
        fn = dump_torch if args.framework == "torch" else dump_jax
        params, info = fn(args.precision, args.copies, args.iters, args.seed)
        np.save(args.out, params)
        print(json.dumps({"out": args.out, "shape": list(params.shape), **info}), flush=True)
        return
    out = compare(args.repeat[0], args.repeat[1], args.other)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
