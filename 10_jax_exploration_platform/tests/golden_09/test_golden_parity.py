"""Golden parity: the copied trainer must compute exactly what the frozen 09 baseline computes.

Two module files are loaded into ONE process under private names —

  reference: 09_parallelization/ppo/jax_ppo/jax_ppo_rnd.py            (frozen, never edited)
  platform:  10_jax_exploration_platform/src/exploration_platform/agents/ppo/jax_ppo_rnd.py

— the same small configuration is built in both, both are started from their own fresh state, and
three iterations are run in each update style with the same random keys. Every reported number and
every entry of the final state must be BIT-identical, not merely close: the copy differs from the
original only in two import-path lines, so anything else is a change that must be caught here.

Both sides run in the same process on the same device, so there is no reason for the two programs to
be compiled differently. If the graphics-card compiler's automatic algorithm selection ever makes a
run non-reproducible, re-run with GOLDEN_PARITY_AUTOTUNE_OFF=1 in the environment, which sets
XLA_FLAGS=--xla_gpu_autotune_level=0 (a workaround this project has already used).

Run on the processor:
  PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu \
    /p/rlprojects/RND/.venvs/platform_jax/bin/python test_golden_parity.py

Run on the graphics card, through the serval05 lock:
  bash /p/rlprojects/RND/09_parallelization/locks/gpu_run.sh \
    "PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python <this file>"
"""
import importlib.util
import os
import sys
from pathlib import Path

# The automatic-algorithm-selection switch has to be set before jax loads its compiler.
if os.environ.get("GOLDEN_PARITY_AUTOTUNE_OFF") == "1":
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_gpu_autotune_level=0").strip()

import numpy as np

PLATFORM_ROOT = Path(__file__).resolve().parent.parent.parent      # 10_jax_exploration_platform
REPO_ROOT = PLATFORM_ROOT.parent                                   # /p/rlprojects/RND
REFERENCE_FILE = REPO_ROOT / "09_parallelization" / "ppo" / "jax_ppo" / "jax_ppo_rnd.py"
PLATFORM_FILE = (PLATFORM_ROOT / "src" / "exploration_platform" / "agents" / "ppo"
                 / "jax_ppo_rnd.py")

# the trainer imports these two by bare name off sys.path, so each side must get its own
ENV_MODULES = ("pm_common", "jax_pointmaze")

# small enough to run on a processor in seconds, large enough to exercise the whole iteration
CONFIG = dict(n_copies=4, n_envs=4, num_steps=32)
ITERATIONS = 3
SEED = 17


def load_module(name: str, path: Path):
    """Load one trainer file under a private module name, with its own environment modules.

    before: sys.modules may already hold `pm_common` / `jax_pointmaze` from the other side;
    after:  they are removed, this file is executed (so it re-imports them off its OWN sys.path
            entry), and the previous ones are put back, leaving the other side untouched.
    """
    saved = {m: sys.modules.pop(m) for m in ENV_MODULES if m in sys.modules}
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    # record where this side's environment code came from, then restore the other side's
    module.loaded_env_files = tuple(str(getattr(sys.modules[m], "__file__", "?"))
                                    for m in ENV_MODULES)
    for m in ENV_MODULES:
        sys.modules.pop(m, None)
    sys.modules.update(saved)
    return module


def leaf_arrays(tree):
    """Every array in a state or metrics object, in a fixed order, as host arrays."""
    import jax
    return [np.asarray(leaf) for leaf in jax.tree.leaves(tree)]


def compare(label: str, reference, platform) -> float:
    """Require bit-identical contents; return the worst absolute difference for the record.

    Bytes are compared rather than values, so a not-a-number in the same place on both sides
    counts as equal and a negative zero against a positive zero counts as different.
    """
    a, b = leaf_arrays(reference), leaf_arrays(platform)
    assert len(a) == len(b), f"{label}: {len(a)} arrays against {len(b)}"
    worst = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.shape == y.shape and x.dtype == y.dtype, (
            f"{label}: array {i} is {x.shape}/{x.dtype} against {y.shape}/{y.dtype}")
        if x.dtype.kind in "fc":
            diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
            worst = max(worst, float(diff.max()) if diff.size else 0.0)
        assert x.tobytes() == y.tobytes(), f"{label}: array {i} differs (shape {x.shape})"
    return worst


def run_style(reference_module, platform_module, style: str) -> float:
    """One update style: build both trainers, check the state at every stage, return worst diff."""
    import jax

    trainer_ref = reference_module.JaxPPORND(
        reference_module.PPOConfig(update_style=style, **CONFIG))
    trainer_new = platform_module.JaxPPORND(
        platform_module.PPOConfig(update_style=style, **CONFIG))

    # 1. the starting state: keyed weights, zero optimizer moments, reset environments
    state_ref, state_new = trainer_ref.init_state(), trainer_new.init_state()
    worst = compare(f"{style}: starting state", state_ref, state_new)
    print(f"  {style}: starting state identical")

    # 2. the priming pass that fills the observation statistics
    prime_key = jax.random.fold_in(jax.random.PRNGKey(SEED), 999999937)
    state_ref = trainer_ref.prime_obs_rms(state_ref, prime_key)
    state_new = trainer_new.prime_obs_rms(state_new, prime_key)
    worst = max(worst, compare(f"{style}: primed state", state_ref, state_new))
    print(f"  {style}: primed state identical")

    # 3. three training iterations from that state, same key and same learning rate each time
    key = jax.random.PRNGKey(SEED)
    for iteration in range(1, ITERATIONS + 1):
        iteration_key = jax.random.fold_in(key, iteration)
        lr_ref = trainer_ref.lr_argument(iteration, ITERATIONS)
        lr_new = trainer_new.lr_argument(iteration, ITERATIONS)
        assert np.asarray(lr_ref).tobytes() == np.asarray(lr_new).tobytes()
        state_ref, metrics_ref = trainer_ref._iterate(state_ref, iteration_key, lr_ref)
        state_new, metrics_new = trainer_new._iterate(state_new, iteration_key, lr_new)
        worst = max(worst, compare(f"{style}: iteration {iteration} metrics",
                                   metrics_ref, metrics_new))
        worst = max(worst, compare(f"{style}: iteration {iteration} state",
                                   state_ref, state_new))
        loss = float(np.asarray(metrics_ref["loss"]))
        reward = float(np.asarray(metrics_ref["reward_ext_sum"]).sum())
        print(f"  {style}: iteration {iteration} identical "
              f"(loss {loss:.8f}, extrinsic reward summed over copies {reward:.4f})")
    return worst


def main() -> None:
    """Load both trainers, run both update styles, and report the worst difference seen."""
    reference_module = load_module("golden_reference_jax_ppo_rnd", REFERENCE_FILE)
    platform_module = load_module("platform_jax_ppo_rnd", PLATFORM_FILE)
    assert reference_module.loaded_env_files != platform_module.loaded_env_files, (
        "both sides loaded the same environment files — the isolation in load_module failed")

    import jax
    print(f"jax {jax.__version__} on {jax.devices()}")
    print(f"reference {REFERENCE_FILE}")
    print(f"  its environment modules: {reference_module.loaded_env_files}")
    print(f"platform  {PLATFORM_FILE}")
    print(f"  its environment modules: {platform_module.loaded_env_files}")
    print(f"configuration {CONFIG}, {ITERATIONS} iterations, seed {SEED}")

    worst = 0.0
    for style in ("full_batch", "epoch_minibatch"):
        worst = max(worst, run_style(reference_module, platform_module, style))
    print(f"ok golden parity: every array bit-identical, "
          f"worst absolute difference {worst:.1e}")


if __name__ == "__main__":
    main()
