"""Golden parity: the platform must compute exactly what the frozen 09 baseline computes.

Two trainers run in ONE process —

  reference: 09_parallelization/ppo/jax_ppo/jax_ppo_rnd.py   (frozen, never edited, one module)
  platform:  src/exploration_platform/                       (environment / agent / bonus split)

— the same small configuration is built in both, both are started from their own fresh state, and
three iterations are run in each update style from the same key. Every array must be BIT-identical,
not merely close.

The two sides keep their state in different shapes: the baseline has one `params` tree and one
`obs_rms`, the platform has `agent_params` / `bonus_params`, an optimizer over both, and the
observation statistics inside the bonus's own state. So the comparison is by NAME, not by
position: each side is turned into a dictionary of arrays under one agreed set of names, the two
key sets must match exactly, and every array must match byte for byte. Nothing is skipped and no
tolerance is involved — the platform's two extra bookkeeping arrays (the run key and the iteration
counter, which the baseline kept on the host instead of in its state) are the only arrays with no
counterpart, and they are checked separately for being what the driver would have passed.

Run on the processor:
  PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu \
    /p/rlprojects/RND/.venvs/platform_jax/bin/python test_golden_parity.py

Run on the graphics card, through the serval05 lock:
  bash /p/rlprojects/RND/09_parallelization/locks/gpu_run.sh \
    "PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python <this file>"

If the graphics-card compiler's automatic algorithm selection ever makes a run non-reproducible,
re-run with GOLDEN_PARITY_AUTOTUNE_OFF=1 in the environment, which sets
XLA_FLAGS=--xla_gpu_autotune_level=0 (a workaround this project has already used).
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
sys.path.insert(0, str(PLATFORM_ROOT / "src"))

from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

# small enough to run on a processor in seconds, large enough to exercise the whole iteration
CONFIG = dict(n_copies=4, n_envs=4, num_steps=32)
ITERATIONS = 3
SEED = 17


def load_reference(path: Path):
    """Load the frozen baseline under a private module name.

    It puts its own environment folders on sys.path and imports `pm_common` / `jax_pointmaze` by
    bare name. The platform's copies of those files live inside the `exploration_platform`
    package, under different module names, so the two sides cannot collide.
    """
    spec = importlib.util.spec_from_file_location("golden_reference_jax_ppo_rnd", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.loaded_env_files = tuple(str(sys.modules[m].__file__)
                                    for m in ("pm_common", "jax_pointmaze"))
    return module


def flat_named(prefix: str, tree) -> dict:
    """Every array under a nested dictionary or named tuple, keyed by its path.

    before: ("params", {"actor": {"W0": [...], "b0": [...]}})
    after:  {"params/actor/W0": array, "params/actor/b0": array}
    """
    if isinstance(tree, dict):
        out = {}
        for key in sorted(tree):
            out.update(flat_named(f"{prefix}/{key}", tree[key]))
        return out
    if hasattr(tree, "_fields"):
        out = {}
        for key in tree._fields:
            out.update(flat_named(f"{prefix}/{key}", getattr(tree, key)))
        return out
    return {prefix: np.asarray(tree)}


def reference_arrays(state) -> dict:
    """The baseline's state under the agreed names."""
    out = {}
    out.update(flat_named("params", state.params))
    out.update(flat_named("opt_m", state.opt_m))
    out.update(flat_named("opt_v", state.opt_v))
    out.update(flat_named("opt_t", state.opt_t))
    out.update(flat_named("env_state", state.env_state))
    out.update(flat_named("obs", state.obs))
    out.update(flat_named("obs_rms", state.obs_rms))
    out.update(flat_named("int_rms", state.int_rms))
    out.update(flat_named("int_filter", state.int_filter))
    out.update(flat_named("visited", state.visited))
    return out


def platform_arrays(state) -> dict:
    """The platform's state under the same names, with its two halves joined back together.

    before: agent_params {"actor", "critic"}, bonus_params {"predictor"}
    after:  params/actor/..., params/critic/..., params/predictor/... — the one tree the
            baseline had, so a missing or renamed tensor shows up as a missing key
    """
    join = lambda agent, bonus: {"actor": agent["actor"], "critic": agent["critic"],
                                 "predictor": bonus["predictor"]}
    out = {}
    out.update(flat_named("params", join(state.agent_params, state.bonus_params)))
    out.update(flat_named("opt_m", join(state.opt.m["agent"], state.opt.m["bonus"])))
    out.update(flat_named("opt_v", join(state.opt.v["agent"], state.opt.v["bonus"])))
    out.update(flat_named("opt_t", state.opt.t))
    out.update(flat_named("env_state", state.env_state))
    out.update(flat_named("obs", state.obs))
    out.update(flat_named("obs_rms", state.bonus_state["obs_rms"]))
    out.update(flat_named("int_rms", state.agent_state["int_rms"]))
    out.update(flat_named("int_filter", state.agent_state["int_filter"]))
    out.update(flat_named("visited", state.visited))
    return out


def compare(label: str, reference: dict, platform: dict) -> float:
    """Require the same names and bit-identical contents; return the worst absolute difference.

    Bytes are compared rather than values, so a not-a-number in the same place on both sides
    counts as equal and a negative zero against a positive zero counts as different.
    """
    assert reference.keys() == platform.keys(), (
        f"{label}: names differ — only in the baseline {sorted(set(reference) - set(platform))}, "
        f"only in the platform {sorted(set(platform) - set(reference))}")
    worst = 0.0
    for name in sorted(reference):
        x, y = reference[name], platform[name]
        assert x.shape == y.shape and x.dtype == y.dtype, (
            f"{label}: {name} is {x.shape}/{x.dtype} against {y.shape}/{y.dtype}")
        if x.dtype.kind in "fc":
            diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
            worst = max(worst, float(diff.max()) if diff.size else 0.0)
        assert x.tobytes() == y.tobytes(), f"{label}: {name} differs (shape {x.shape})"
    return worst


def check_bookkeeping(state, iteration: int) -> None:
    """The platform's two arrays with no counterpart: the run key and the iteration counter."""
    import jax
    assert np.asarray(state.rng).tobytes() == np.asarray(jax.random.PRNGKey(SEED)).tobytes(), (
        "the platform's stored key is not the key the baseline's driver folded into")
    assert int(np.asarray(state.step)) == iteration, (
        f"the platform's iteration counter says {int(np.asarray(state.step))}, not {iteration}")


def run_style(reference_module, style: str) -> float:
    """One update style: build both trainers, check at every stage, return the worst difference."""
    trainer_ref = reference_module.JaxPPORND(
        reference_module.PPOConfig(update_style=style, **CONFIG))
    runner = Runner(PPOConfig(update_style=style, **CONFIG))

    # 1. the starting state: keyed weights, zero optimizer moments, reset environments
    state_ref = trainer_ref.init_state()
    state_new = runner.init_state(run_seed=SEED)
    worst = compare(f"{style}: starting state",
                    reference_arrays(state_ref), platform_arrays(state_new))
    check_bookkeeping(state_new, 0)
    print(f"  {style}: starting state identical")

    # 2. the priming pass that fills the observation statistics
    import jax
    prime_key = jax.random.fold_in(jax.random.PRNGKey(SEED), 999999937)
    state_ref = trainer_ref.prime_obs_rms(state_ref, prime_key)
    state_new = runner.prime(state_new)
    worst = max(worst, compare(f"{style}: primed state",
                               reference_arrays(state_ref), platform_arrays(state_new)))
    print(f"  {style}: primed state identical")

    # 3. three training iterations from that state, same key and same learning rate each time
    key = jax.random.PRNGKey(SEED)
    for iteration in range(1, ITERATIONS + 1):
        lr_ref = trainer_ref.lr_argument(iteration, ITERATIONS)
        lr_new = runner.lr_argument(iteration, ITERATIONS)
        assert np.asarray(lr_ref).tobytes() == np.asarray(lr_new).tobytes()
        state_ref, metrics_ref = trainer_ref._iterate(
            state_ref, jax.random.fold_in(key, iteration), lr_ref)
        state_new, metrics_new = runner.iterate(state_new, lr_new)
        worst = max(worst, compare(f"{style}: iteration {iteration} metrics",
                                   flat_named("metrics", metrics_ref),
                                   flat_named("metrics", metrics_new)))
        worst = max(worst, compare(f"{style}: iteration {iteration} state",
                                   reference_arrays(state_ref), platform_arrays(state_new)))
        check_bookkeeping(state_new, iteration)
        loss = float(np.asarray(metrics_ref["loss"]))
        reward = float(np.asarray(metrics_ref["reward_ext_sum"]).sum())
        print(f"  {style}: iteration {iteration} identical "
              f"(loss {loss:.8f}, extrinsic reward summed over copies {reward:.4f})")
    return worst


def main() -> None:
    """Load the baseline, run both update styles, and report the worst difference seen."""
    reference_module = load_reference(REFERENCE_FILE)

    import jax
    from exploration_platform.envs.pointmaze import jax_pointmaze
    print(f"jax {jax.__version__} on {jax.devices()}")
    print(f"reference {REFERENCE_FILE}")
    print(f"  its environment modules: {reference_module.loaded_env_files}")
    print(f"platform  {PLATFORM_ROOT / 'src' / 'exploration_platform'}")
    print(f"  its environment module: {jax_pointmaze.__file__}")
    assert jax_pointmaze.__file__ not in reference_module.loaded_env_files, (
        "both sides loaded the same environment file — the isolation failed")
    print(f"configuration {CONFIG}, {ITERATIONS} iterations, seed {SEED}")

    worst = 0.0
    for style in ("full_batch", "epoch_minibatch"):
        worst = max(worst, run_style(reference_module, style))
    print(f"ok golden parity: every array bit-identical, "
          f"worst absolute difference {worst:.1e}")


if __name__ == "__main__":
    main()
