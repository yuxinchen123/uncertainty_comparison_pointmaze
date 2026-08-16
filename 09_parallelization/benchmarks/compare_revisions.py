"""How far apart two revisions of the trainer end up, from identical inputs.

A change that is meant to leave the computation alone has to be shown to leave it alone. Timing
harnesses cannot do that, and the equivalence tests inside `ppo/torch_ppo/tests/` compare two
builds of the SAME source, so they cannot see a difference introduced by a change to the source.
This runs the same training iterations under two git revisions, from the same seed, and reports
the worst difference in any parameter.

A difference of exactly zero means the change is bitwise neutral. A small non-zero difference is
only acceptable with an explanation — a multiplication library selecting a different kernel, for
instance, accumulates the same sum in a different order — and the number belongs in the ledger.

Usage (through the H100 lock wrapper):
  python compare_revisions.py --a 25876a8 --b "" --n-copies 128 --iters 3
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"


def run_revision(rev, n_copies, style, iters, seed):
    """Run `iters` captured iterations of one revision and return its parameters as a list."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ab_compare import load_module
    mod = load_module(rev)
    torch.manual_seed(seed)
    trainer = mod.PPORND(mod.production_config(n_copies, style=style), device="cuda")
    trainer.prime_obs_rms()
    trainer._build_iteration_graph()
    for _ in range(iters):
        trainer.iteration_captured()
    torch.cuda.synchronize()
    # the parameters are returned as separate tensors, not as the flat buffer, because two
    # revisions may lay the buffer out differently while holding the same twenty-one parameters
    out = [p.detach().clone() for p in trainer.trainable]
    del trainer
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="git revision for side A (empty = working tree)")
    ap.add_argument("--b", default="", help="git revision for side B (empty = working tree)")
    ap.add_argument("--n-copies", type=int, default=128)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    # each revision runs in its own subprocess: two revisions of the same module name cannot
    # both be compiled in one process without the compiler reusing the first one's artifacts.
    # The parameters travel through a file on local disk rather than through the pipe, because
    # twenty-one tensors at a thousand copies are tens of millions of numbers.
    scratch = Path("/localtmp/sl5nw/compare_revisions")
    scratch.mkdir(parents=True, exist_ok=True)
    outs = {}
    for side, rev in (("a", args.a), ("b", args.b)):
        dest = scratch / f"{side}.pt"
        payload = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parent)!r}); "
             "import compare_revisions as m, torch; "
             f"torch.save([p.cpu() for p in m.run_revision({rev!r}, {args.n_copies}, "
             f"{args.style!r}, {args.iters}, {args.seed})], {str(dest)!r}); print('RESULT ok')"],
            capture_output=True, text=True)
        assert "RESULT ok" in payload.stdout, \
            f"side {side} produced no result:\n{payload.stderr[-3000:]}"
        outs[side] = torch.load(dest, weights_only=True)

    worst_abs = worst_rel = 0.0
    magnitude = 0.0
    for ta, tb in zip(outs["a"], outs["b"]):
        d = (ta - tb).abs().max().item()
        scale = max(tb.abs().max().item(), 1e-12)
        worst_abs = max(worst_abs, d)
        worst_rel = max(worst_rel, d / scale)
        magnitude = max(magnitude, scale)
    row = {"a": args.a or "working tree", "b": args.b or "working tree",
           "n_copies": args.n_copies, "style": args.style, "iters": args.iters,
           "seed": args.seed, "worst_absolute": worst_abs, "worst_relative": worst_rel,
           "largest_parameter": magnitude}
    print(f"after {args.iters} iteration(s) at {args.n_copies} copies, "
          f"{row['a']} against {row['b']}: worst absolute {worst_abs:.3e}, "
          f"worst relative {worst_rel:.3e}, largest parameter {magnitude:.3e}")

    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / (f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_compare_revisions"
                   f"_C{args.n_copies}{args.tag}.json")
    p.write_text(json.dumps(row, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
