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

Usage (through the lock wrapper): python profile_jax_phases.py --n-copies 128
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "jax_ppo"))
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


def make_batch(cfg, key):
    """A batch of the shapes the update stage consumes; values are realistic but arbitrary."""
    import jax
    import jax.numpy as jnp
    C = cfg.n_copies
    B = cfg.num_steps * cfg.n_envs
    k = jax.random.split(key, 8)
    return {
        "obs": jax.random.normal(k[0], (C, B, 4), jnp.float32),
        "actions": jax.random.normal(k[1], (C, B, 2), jnp.float32),
        "old_logprob": jax.random.normal(k[2], (C, B), jnp.float32),
        "adv": jax.random.normal(k[3], (C, B), jnp.float32),
        "ret_ext": jax.random.normal(k[4], (C, B), jnp.float32),
        "ret_int": jax.random.normal(k[5], (C, B), jnp.float32),
        "vext_old": jax.random.normal(k[6], (C, B), jnp.float32),
        "rnd_input": jax.random.normal(k[7], (C, B, 4), jnp.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, default=128)
    ap.add_argument("--style", default="epoch_minibatch")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from jax_ppo_rnd import PPOConfig, JaxPPORND

    cfg = PPOConfig(n_copies=args.n_copies, update_style=args.style)
    tr = JaxPPORND(cfg)
    state = tr.init_state()
    key = jax.random.PRNGKey(0)
    state = tr.prime_obs_rms(state, jax.random.fold_in(key, 1))
    lr = jnp.asarray(cfg.learning_rate, jnp.float32)
    state, _ = tr._iterate(state, jax.random.fold_in(key, 2), lr)   # a representative state

    batch = make_batch(cfg, jax.random.fold_in(key, 3))
    ukey = jax.random.fold_in(key, 4)
    C = cfg.n_copies
    n_steps = cfg.update_epochs * cfg.num_minibatches
    mb_size = (cfg.num_steps * cfg.n_envs) // cfg.num_minibatches

    # the real update stage
    if args.style == "epoch_minibatch":
        upd = jax.jit(lambda s, b: tr._update_epoch_minibatch(s, b, lr, ukey))
    else:
        upd = jax.jit(lambda s, b: tr._update_full_batch(s, b, lr, ukey))

    # the same sixteen steps, stopping after the gradient, and after the gradient plus clip
    def steps_only(params, batch, with_clip):
        """Sixteen scanned gradient computations, optionally clipped, never applied."""
        # the same minibatch rows for every step: only the shapes affect the timing, and
        # this variant does not update parameters anyway
        mbs = {k: jnp.broadcast_to(v[None, :, :mb_size], (n_steps, C, mb_size) + v.shape[2:])
               for k, v in batch.items()}

        def body(carry, mb):
            g = jax.grad(lambda p: tr._losses(p, mb, style_a=False))(params)
            if with_clip:
                g = tr._clip_per_copy(g)
            # consume EVERY leaf: taking only one lets the compiler delete the computation of
            # the others as dead code, which would credit their cost to whatever runs next
            return carry + sum(l.sum() for l in jax.tree.leaves(g)), None

        total, _ = jax.lax.scan(body, jnp.zeros((), jnp.float32), mbs)
        return total

    grad_only = jax.jit(lambda p, b: steps_only(p, b, False))
    grad_clip = jax.jit(lambda p, b: steps_only(p, b, True))

    # the iteration consumes its state, so it is timed by carrying the state forward; the
    # remaining timings use the state it leaves behind and do not consume anything
    t_whole, state = timed_stateful(
        lambda s: tr._iterate(s, jax.random.fold_in(key, 5), lr)[0], state)
    t_update = timed(lambda: upd(state, batch))
    # the priming pass runs the same T environment steps with random actions and no networks,
    # so it isolates the environment's share of the rollout
    t_env_only, state = timed_stateful(
        lambda s: tr._prime(s, jax.random.fold_in(key, 6)), state)
    t_grad = timed(lambda: grad_only(state.params, batch))
    t_gradclip = timed(lambda: grad_clip(state.params, batch))

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
    print(f"\n== JAX iteration at {C} copies, {args.style} ==")
    for k, v in rows.items():
        share = f"{v / t_whole * 100:5.1f}% of the iteration" if t_whole else ""
        print(f"  {v*1e3:8.3f} ms  {share}  {k}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_jax_phases_C{C}_{args.style}.json"
    out.write_text(json.dumps({
        "n_copies": C, "style": args.style, "seconds": rows,
        "jax": jax.__version__,
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
