"""Count the operations the compiler actually emits for the JAX update stage.

The ceiling analysis argues this workload is limited by the NUMBER of separate operations rather
than by arithmetic. Before rewriting anything, this checks how many operations the compiler emits
for the update stage, and how many of them belong to the gradient clip and the optimizer — the two
places that walk every parameter tensor separately.

It reads the optimized program the compiler produces (the "high level optimizer" text), counts the
fusions and other top-level operations inside the scan body, and reports them. No timing here.

Usage (no GPU lock needed for compilation alone, but the device must exist):
  python count_jax_ops.py --n-copies 128
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "jax_ppo"))


def body_ops(text):
    """Operation kinds inside the scan body of an optimized program, and their count.

    before: the whole optimized program text; after: a Counter over operation kinds such as
    'fusion', 'reduce', 'copy', appearing in the computation the scan loops over.
    """
    kinds = Counter()
    for line in text.splitlines():
        m = re.match(r"\s+%?\S+ = \S+ (\w+)\(", line)
        if m:
            kinds[m.group(1)] += 1
    return kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, default=128)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--dump", help="write the optimized program text here")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from jax_ppo_rnd import PPOConfig, JaxPPORND
    sys.path.insert(0, str(BASE / "benchmarks"))
    from profile_jax_phases import make_batch

    cfg = PPOConfig(n_copies=args.n_copies, update_style=args.style)
    tr = JaxPPORND(cfg)
    state = tr.init_state()
    key = jax.random.PRNGKey(0)
    lr = jnp.asarray(cfg.learning_rate, jnp.float32)
    batch = make_batch(cfg, key)

    upd = jax.jit(lambda s, b: (tr._update_epoch_minibatch(s, b, lr, key)
                                if args.style == "epoch_minibatch"
                                else tr._update_full_batch(s, b, lr, key)))
    compiled = upd.lower(state, batch).compile()
    text = compiled.as_text()
    if args.dump:
        Path(args.dump).write_text(text)

    kinds = body_ops(text)
    total = sum(kinds.values())
    print(f"optimized program for the update stage, {args.n_copies} copies, {args.style}")
    print(f"  {total} operations in total")
    for k, v in kinds.most_common(12):
        print(f"  {v:>6}  {k}")
    n_param_tensors = len(jax.tree.leaves(state.params))
    print(f"\n  {n_param_tensors} trainable parameter tensors")
    print(f"  memory analysis: {compiled.memory_analysis()}")


if __name__ == "__main__":
    main()
