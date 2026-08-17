"""Audit the compiled iteration for float64 operations outside the running statistics.

Double precision is on because the running mean and variance need it (spec section 4.2). That
makes float64 the promotion target for any literal or operation not explicitly typed, so a single
leaked float64 in the rollout or the update would cost roughly a factor of two on this card and
would be invisible in every correctness test — float64 is more accurate, not less.

This counts f64 shapes in the optimized program and reports where they sit, so the answer is a
number rather than an assumption.

Run: PYTHONNOUSERSITE=1 <jax python> audit_float64.py
"""
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402


def main():
    """Lower one iteration, count f64 operations, and print the largest offenders."""
    runner = Runner(PPOConfig(n_copies=8, n_envs=4, num_steps=32, prime_iterations=1,
                              update_style="epoch_minibatch"))
    state = runner.init_state()
    text = runner.iterate.lower(state, 3e-4).compile().as_text()

    # every operation result shape in the optimized program, e.g. "f64[8,4]" or "f32[8,128,4]"
    shapes = re.findall(r"\b(f64|f32)\[[0-9,]*\]", text)
    counts = Counter(shapes)
    total = sum(counts.values())
    print(f"optimized program: {counts['f32']} f32 result shapes, {counts['f64']} f64 "
          f"({counts['f64'] / max(total, 1) * 100:.1f}%)")

    # the f64 lines, grouped, so a leak names itself
    f64_lines = [l.strip() for l in text.splitlines() if "f64[" in l]
    sizes = Counter()
    for line in f64_lines:
        for dims in re.findall(r"f64\[([0-9,]*)\]", line):
            n = 1
            for d in dims.split(","):
                if d:
                    n *= int(d)
            sizes[dims or "scalar"] += n
    print("f64 result shapes by total element count (largest first):")
    for dims, n in sizes.most_common(8):
        print(f"  [{dims}]: {n} elements")
    big = [(dims, n) for dims, n in sizes.items() if n > 1000]
    print(f"\nf64 arrays larger than 1000 elements: {big if big else 'none'}")
    print("expected: only the running statistics (mean/var/count, [C,4] and [C,1]) and their "
          "scalar helpers are float64")


if __name__ == "__main__":
    main()
