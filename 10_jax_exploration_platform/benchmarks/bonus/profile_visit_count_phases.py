"""Where a visit-count iteration's time goes at the copy count the science run will use.

Written for the `full_batch` update style, in which one iteration takes exactly one gradient step.
The older `benchmarks/harness/profile_jax_phases.py` subtracts a fixed sixteen-step gradient probe
from the real update, which is meaningful only under `epoch_minibatch`; its clip and optimizer
rows come out negative here. This attributes time by building the pieces at the real shapes and
timing each one, so every row is a measurement rather than a subtraction of unlike things.

The rows:

  whole iteration, visit count      the shipped program
  whole iteration, no bonus         the same program with the bonus removed
  the bonus's whole share           the difference of those two
  update stage alone                one gradient step and the optimizer, on a batch of the real
                                    shapes
  rollout and post-processing       whole minus update
  index arithmetic alone            observations to flat table indices
  index + scatter                   with the counting added
  index + scatter + gather + power  the bonus's own three operations end to end

The last three run on indices taken from a real rollout of the real environment, not synthesized
ones: how far apart a rollout's indices fall decides how much of a scatter's traffic the cache
absorbs, so made-up indices would measure a different machine.

Run on the graphics card, through the serval05 lock:
  bash /p/rlprojects/RND/09_parallelization/locks/gpu_run.sh \
    "PYTHONNOUSERSITE=1 ... /p/rlprojects/RND/.venvs/platform_jax/bin/python <this file>"
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src"))
from exploration_platform import F32  # noqa: E402
from exploration_platform.agents.ppo.update import build_update  # noqa: E402
from exploration_platform.bonuses.visit_count.config import VisitCountConfig  # noqa: E402
from exploration_platform.envs.pointmaze.pm_common import MAPS  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402
from exploration_platform.training.state import stored_params  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_bonus_against_rnd import build_config  # noqa: E402

PACIFIC = ZoneInfo("America/Los_Angeles")
# the decay each preset carries, so the standalone timings compute the same power the composed
# program does. Kept beside the presets in `bonuses/registry.py`; a name missing here is refused
# rather than profiled with the wrong exponent.
PRESET_DECAY = {"gt_position_velocity_sqrt": -0.5, "gt_position_velocity_linear": -1.0}


def timed(fn, reps=20, warmup=5):
    """Median seconds per call for a function that does not consume its arguments."""
    out = None
    for _ in range(warmup):
        out = fn()
    jax.block_until_ready(out)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def timed_stateful(step, state, reps=20, warmup=5):
    """Median seconds for a function that consumes its state and returns the next one.

    The compiled iteration donates its state buffer, so the same state cannot be handed in twice;
    each call carries forward what the previous one returned.
    """
    for _ in range(warmup):
        state = step(state)
    jax.block_until_ready(state)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        state = step(state)
        jax.block_until_ready(state)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), state


def real_rollout_indices(runner, cfg, vc_cfg: VisitCountConfig):
    """Flat table indices of one real rollout, and the open-cell flags beside them.

    before: the environment at its start state, 128 steps of random actions
    after:  idx [C, T*N] int32 in [0, rows*cols*100), is_open [C, T*N] bool — the same arrays the
            bonus's scatter and gather consume inside the compiled iteration
    """
    C, N, T = cfg.n_copies, cfg.n_envs, cfg.num_steps
    env = runner.env
    wall = np.asarray(MAPS[runner.composition.env_cfg.map_name])
    rows, cols = wall.shape
    bins, clip = vc_cfg.velocity_bins, vc_cfg.velocity_clip
    open_cell = jnp.asarray(wall.reshape(-1) == 0)

    @jax.jit
    def collect(env_state, key):
        """One rollout of random actions, keeping only the observations it reached."""
        acts = jax.random.uniform(key, (T, C, N, 2), F32, -1.0, 1.0)

        def body(carry, a):
            """One environment step; only the observation the step ended on is kept."""
            env_state, _, _, _, _, final_obs = env.step(carry, a)
            return env_state, final_obs

        _, nobs = jax.lax.scan(body, env_state, acts)
        # before: nobs [T, C, N, 4]; after: [C, T*N, 4], the shape the bonus is handed
        return nobs.transpose(1, 0, 2, 3).reshape(C, T * N, 4)

    nobs = collect(env.reset(), jax.random.PRNGKey(7))
    jj = jnp.clip((nobs[..., 0] + cols / 2.0).astype(jnp.int32), 0, cols - 1)
    ii = jnp.clip((rows / 2.0 - nobs[..., 1]).astype(jnp.int32), 0, rows - 1)
    cell = ii * cols + jj
    vx = jnp.clip(nobs[..., 2], -clip, clip)
    vy = jnp.clip(nobs[..., 3], -clip, clip)
    bx = jnp.clip(((vx + clip) / (2 * clip) * bins).astype(jnp.int32), 0, bins - 1)
    by = jnp.clip(((vy + clip) / (2 * clip) * bins).astype(jnp.int32), 0, bins - 1)
    return nobs, cell * bins * bins + bx * bins + by, open_cell[cell], rows * cols * bins * bins


def make_batch(cfg, key):
    """A batch of the shapes the update stage consumes; the values only have to be finite."""
    C, B = cfg.n_copies, cfg.num_steps * cfg.n_envs
    k = jax.random.split(key, 7)
    return {"obs": jax.random.normal(k[0], (C, B, 4), F32),
            "actions": jax.random.normal(k[1], (C, B, 2), F32),
            "old_logprob": jax.random.normal(k[2], (C, B), F32),
            "adv": jax.random.normal(k[3], (C, B), F32),
            "ret_ext": jax.random.normal(k[4], (C, B), F32),
            "ret_int": jax.random.normal(k[5], (C, B), F32),
            "vext_old": jax.random.normal(k[6], (C, B), F32)}


def main():
    """Time every piece at the real shapes, print the attribution, store the record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bonus", default="gt_position_velocity_sqrt")
    parser.add_argument("--copies-per-group", type=int, default=256)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    cfg = build_config(args.copies_per_group, sweep_betas=True)
    if args.bonus not in PRESET_DECAY:
        raise ValueError(f"{args.bonus!r} is not a visit-count preset; this profiler times the "
                         f"visit-count family, and the presets are {sorted(PRESET_DECAY)}")
    vc_cfg = VisitCountConfig(decay=PRESET_DECAY[args.bonus])
    print(f"jax {jax.__version__} on {jax.devices()}; {cfg.n_copies} copies, "
          f"{cfg.update_style}, bonus {args.bonus}")

    runner = Runner(cfg, bonus=args.bonus)
    state = runner.prime(runner.init_state(run_seed=1))
    lr = runner.lr_argument(1, 1)
    t_whole, state = timed_stateful(lambda s: runner.iterate(s, lr)[0], state)

    # the same program with the bonus taken out: the difference is everything the bonus costs,
    # its index arithmetic, its two table operations and whatever it adds to the update
    plain = Runner(cfg, bonus="none")
    plain_state = plain.prime(plain.init_state(run_seed=1))
    t_none, plain_state = timed_stateful(lambda s: plain.iterate(s, lr)[0], plain_state)

    # the update stage alone, the same function the compiled iteration calls, on a real-shaped
    # batch. In full_batch this is ONE gradient step and one optimizer application.
    key = jax.random.PRNGKey(0)
    update_fn = build_update(cfg, runner.total_loss, runner.sweep.lr_per_copy)
    upd = jax.jit(lambda p, o, b: update_fn(p, o, b, lr, jax.random.fold_in(key, 4)))
    batch = make_batch(cfg, jax.random.fold_in(key, 3))
    params = stored_params(state, runner.layout)
    t_update = timed(lambda: upd(params, state.opt, batch))

    # the bonus's own three operations, on indices a real rollout produced
    nobs, idx, is_open, table_size = real_rollout_indices(runner, cfg, vc_cfg)
    counts0 = jnp.zeros((cfg.n_copies, table_size), jnp.int32)
    copy_rows = jnp.arange(cfg.n_copies)[:, None]
    wall = np.asarray(MAPS[runner.composition.env_cfg.map_name])
    rows_n, cols_n = wall.shape
    open_cell = jnp.asarray(wall.reshape(-1) == 0)

    bins, clip = vc_cfg.velocity_bins, vc_cfg.velocity_clip

    @jax.jit
    def index_only(nobs):
        """Observations to flat table indices — the arithmetic before any memory traffic."""
        jj = jnp.clip((nobs[..., 0] + cols_n / 2.0).astype(jnp.int32), 0, cols_n - 1)
        ii = jnp.clip((rows_n / 2.0 - nobs[..., 1]).astype(jnp.int32), 0, rows_n - 1)
        cell = ii * cols_n + jj
        bx = jnp.clip(((jnp.clip(nobs[..., 2], -clip, clip) + clip) / (2 * clip)
                       * bins).astype(jnp.int32), 0, bins - 1)
        by = jnp.clip(((jnp.clip(nobs[..., 3], -clip, clip) + clip) / (2 * clip)
                       * bins).astype(jnp.int32), 0, bins - 1)
        return cell * bins * bins + bx * bins + by, open_cell[cell]

    @jax.jit
    def scatter_only(counts, idx, is_open):
        """The counting scatter alone."""
        return counts.at[copy_rows, idx].add(is_open.astype(jnp.int32))

    @jax.jit
    def scatter_and_gather(counts, idx, is_open):
        """The bonus as the implementation performs it: scatter, gather, power."""
        counts = counts.at[copy_rows, idx].add(is_open.astype(jnp.int32))
        n = jnp.take_along_axis(counts, idx, axis=1)
        bonus = jnp.where(is_open,
                          jnp.minimum(1.0, jnp.maximum(n, 1).astype(F32) ** vc_cfg.decay), 1.0)
        return counts, bonus

    t_index = timed(lambda: index_only(nobs))
    t_scatter = timed(lambda: scatter_only(counts0, idx, is_open))
    t_full = timed(lambda: scatter_and_gather(counts0, idx, is_open))

    rows = {
        "whole iteration, visit count": t_whole,
        "whole iteration, no bonus": t_none,
        "the bonus's whole share (visit count minus no bonus)": t_whole - t_none,
        "update stage alone (one gradient step and the optimizer)": t_update,
        "rollout and post-processing (whole minus update)": t_whole - t_update,
        "index arithmetic alone": t_index,
        "index and scatter": t_scatter,
        "index, scatter, gather and power": t_full,
    }
    print(f"\n== visit-count iteration at {cfg.n_copies} copies, {cfg.update_style} ==")
    for k, v in rows.items():
        print(f"  {v * 1e3:8.3f} ms  {v / t_whole * 100:5.1f}% of the iteration  {k}")

    stamp = datetime.now().astimezone(PACIFIC).strftime("%Y-%m-%d-%H-%M")
    out_dir = Path(args.out_dir) if args.out_dir else BASE / "benchmark_runs" / f"{stamp}_profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stamp}_visit_count_phases_copies-{cfg.n_copies}.json"
    out.write_text(json.dumps({
        "measured_at": datetime.now().astimezone().isoformat(),
        "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "bonus": args.bonus, "copies": cfg.n_copies, "update_style": cfg.update_style,
        "rollout_steps": cfg.num_steps, "envs_per_copy": cfg.n_envs,
        "table_entries_per_copy": int(table_size),
        "rows_scattered_per_copy": int(idx.shape[1]),
        "distinct_indices_per_copy_mean": float(
            np.mean([len(np.unique(np.asarray(idx[i]))) for i in range(0, cfg.n_copies, 512)])),
        "seconds": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
