"""Where a JAX training iteration spends its time, and what the optimizer stage costs.

The ceiling analysis names one large remaining improvement for the PyTorch trainer: holding every
parameter in one flat buffer so the gradient clip and the optimizer stop walking twenty-one
separate arrays. Whether that applies to JAX is an open question, because its compiler fuses more
aggressively, so this measures the pieces before anything is changed.

Timings, all at the same size:

  whole            one complete iteration (rollout, post-rollout processing, update)
  update           the update stage alone, on a batch of the correct shapes
  gradient only    the same sixteen steps with the clip and the optimizer removed
  gradient + clip  with the clip but not the optimizer

The last two compute a different function on purpose (they do not update parameters); they exist
only to attribute time, and the differences between the three give the cost of the clip and of the
optimizer separately. Timing depends on array shapes rather than values, so the batch is
synthesized rather than captured, which keeps the trainer unmodified.

The observation and action widths come from the environment, so the same measurement runs on
PointMaze (4 and 2) and on AntMaze (29 and 8), at any number of environments per copy.

Usage (through the lock wrapper): python profile_jax_phases.py --n-copies 128
  python profile_jax_phases.py --n-copies 128 --env antmaze-umaze --envs-per-copy 16
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src"))
RESULTS = Path(__file__).resolve().parent / "results"


def timed(fn, reps=30, warmup=5):
    """Median seconds per call, waiting for the device each time. For functions that do not
    consume their arguments."""
    import jax
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
    times.sort()
    return times[len(times) // 2]


def timed_stateful(step, state, reps=30, warmup=5):
    """Median seconds for a function that CONSUMES its state and returns a new one.

    The whole iteration donates its state buffer, so the same state cannot be passed twice;
    each call must carry forward the state the previous call returned.
    """
    import jax
    for _ in range(warmup):
        state = step(state)
    jax.block_until_ready(state)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        state = step(state)
        jax.block_until_ready(state)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], state


def make_batch(cfg, key, obs_dim: int, act_dim: int):
    """A batch of the shapes the update stage consumes; values are realistic but arbitrary.

    obs_dim and act_dim are the environment's observation and action widths.
    before: 2 copies, 128 steps x 4 environments, PointMaze widths 4 and 2
    after:  obs and rnd_input [2, 512, 4], actions [2, 512, 2], every other field [2, 512]
    """
    import jax
    import jax.numpy as jnp
    C = cfg.n_copies
    B = cfg.num_steps * cfg.n_envs
    k = jax.random.split(key, 8)
    return {
        "obs": jax.random.normal(k[0], (C, B, obs_dim), jnp.float32),
        "actions": jax.random.normal(k[1], (C, B, act_dim), jnp.float32),
        "old_logprob": jax.random.normal(k[2], (C, B), jnp.float32),
        "adv": jax.random.normal(k[3], (C, B), jnp.float32),
        "ret_ext": jax.random.normal(k[4], (C, B), jnp.float32),
        "ret_int": jax.random.normal(k[5], (C, B), jnp.float32),
        "vext_old": jax.random.normal(k[6], (C, B), jnp.float32),
        "rnd_input": jax.random.normal(k[7], (C, B, obs_dim), jnp.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, default=128)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--env", default="pointmaze-large",
                    choices=["pointmaze-large", "antmaze-umaze", "antmaze-medium",
                             "antmaze-large"])
    ap.add_argument("--envs-per-copy", type=int, default=4)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from exploration_platform.agents.ppo.config import PPOConfig
    from exploration_platform.agents.ppo.update import build_update, clip_per_copy
    from exploration_platform.training.runner import Runner
    from exploration_platform.training.state import stored_params

    # the environment: PointMaze's default configuration, or one of the three ant mazes
    env_cfg = None
    if args.env != "pointmaze-large":
        from exploration_platform.envs.antmaze.am_common import preset
        env_cfg = preset(args.env.split("-")[1])
    cfg = PPOConfig(n_copies=args.n_copies, n_envs=args.envs_per_copy, update_style=args.style)
    tr = Runner(cfg, env_cfg=env_cfg)
    cfg = tr.cfg
    state = tr.prime(tr.init_state(run_seed=0))
    key = jax.random.PRNGKey(0)
    lr = jnp.asarray(cfg.learning_rate, jnp.float32)
    state, _ = tr.iterate(state, lr)   # a representative state

    batch = make_batch(cfg, jax.random.fold_in(key, 3), tr.env.obs_dim, tr.env.act_dim)
    ukey = jax.random.fold_in(key, 4)
    C = cfg.n_copies
    n_steps = cfg.update_epochs * cfg.num_minibatches
    mb_size = (cfg.num_steps * cfg.n_envs) // cfg.num_minibatches

    # the real update stage, the same function the compiled iteration calls
    update_fn = build_update(cfg, tr.total_loss, tr.sweep.lr_per_copy)
    upd = jax.jit(lambda p, o, b: update_fn(p, o, b, lr, ukey))

    # the same sixteen steps, stopping after the gradient, and after the gradient plus clip
    def steps_only(params, batch, with_clip):
        """Sixteen scanned gradient computations, optionally clipped, never applied."""
        # the same minibatch rows for every step: only the shapes affect the timing, and
        # this variant does not update parameters anyway
        mbs = {k: jnp.broadcast_to(v[None, :, :mb_size], (n_steps, C, mb_size) + v.shape[2:])
               for k, v in batch.items()}

        def body(carry, mb):
            g = jax.grad(lambda p: tr.total_loss(p, mb, style_a=False))(params)
            if with_clip:
                g = clip_per_copy(g, C, cfg.max_grad_norm)
            # consume EVERY leaf: taking only one lets the compiler delete the computation of
            # the others as dead code, which would credit their cost to whatever runs next
            return carry + sum(l.sum() for l in jax.tree.leaves(g)), None

        total, _ = jax.lax.scan(body, jnp.zeros((), jnp.float32), mbs)
        return total

    grad_only = jax.jit(lambda p, b: steps_only(p, b, False))
    grad_clip = jax.jit(lambda p, b: steps_only(p, b, True))

    # the iteration consumes its state, so it is timed by carrying the state forward; the
    # remaining timings use the state it leaves behind and do not consume anything
    t_whole, state = timed_stateful(lambda s: tr.iterate(s, lr)[0], state)
    t_update = timed(lambda: upd(stored_params(state, tr.layout), state.opt, batch))
    # the priming pass runs the same T environment steps with random actions and no networks,
    # so it isolates the environment's share of the rollout
    t_env_only, state = timed_stateful(
        lambda s: tr.prime_step(s, jax.random.fold_in(key, 6)), state)
    t_grad = timed(lambda: grad_only(stored_params(state, tr.layout), batch))
    t_gradclip = timed(lambda: grad_clip(stored_params(state, tr.layout), batch))

    rows = {
        "whole_iteration": t_whole,
        "update_stage": t_update,
        "rollout_and_post (whole minus update)": t_whole - t_update,
        "environment alone, 128 steps (the priming pass)": t_env_only,
        "policy and post-processing (rollout+post minus environment)":
            (t_whole - t_update) - t_env_only,
        "sixteen gradients only": t_grad,
        "sixteen gradients + clip": t_gradclip,
        "clip cost (clip minus gradient)": t_gradclip - t_grad,
        "optimizer cost (update minus gradient+clip)": t_update - t_gradclip,
    }
    print(f"\n== JAX iteration on {args.env} at {C} copies x {cfg.n_envs} environments, "
          f"{args.style} ==")
    for k, v in rows.items():
        share = f"{v / t_whole * 100:5.1f}% of the iteration" if t_whole else ""
        print(f"  {v*1e3:8.3f} ms  {share}  {k}")

    RESULTS.mkdir(exist_ok=True)
    out = (RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_jax_phases_{args.env}_C{C}"
                     f"_N{cfg.n_envs}_{args.style}.json")
    out.write_text(json.dumps({
        "n_copies": C, "n_envs": cfg.n_envs, "environment": args.env, "style": args.style,
        "observation_width": tr.env.obs_dim, "action_width": tr.env.act_dim, "seconds": rows,
        "jax": jax.__version__,
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
