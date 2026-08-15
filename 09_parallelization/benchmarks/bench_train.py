"""Benchmark the torch PPO+RND trainer: iterations/sec and env-steps/sec vs n_copies.

Usage (on serval05, under the H100 lock):
  python bench_train.py --style epoch_minibatch --n-copies 8 32 128
Measures full training iterations (rollout + update) after warmup; per-phase timing too.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "ppo" / "torch_ppo"))
RESULTS = Path(__file__).resolve().parent / "results"


def bench(n_copies, style, iters, warmup, rollout_mode="eager", fused_adam=False, capture_update=False, one_graph=False, tf32=False, env_backend="torch", timing="sync", rev=""):
    """Time full iterations and the rollout/update split for one copy count.

    rev selects a git revision of the trainer instead of the working tree, so a before/after
    curve over many copy counts can be measured by the same harness in the same session rather
    than by comparing two runs taken hours apart under different benchmark code.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ab_compare import load_module
    mod = load_module(rev)
    PPOConfig, PPORND, production_config = mod.PPOConfig, mod.PPORND, mod.production_config
    # start from the ONE definition of the shipped configuration, then apply this run's
    # overrides; building the config by hand here is how a benchmark silently stops measuring
    # what actually ships (it happened once: compile_post was left off)
    if one_graph and rollout_mode == "capture" and capture_update and fused_adam and tf32:
        cfg = production_config(n_copies, style=style, env_backend=env_backend)
    else:
        cfg = PPOConfig(n_copies=n_copies, update_style=style, rollout_mode=rollout_mode,
                        fused_adam=fused_adam, capture_update=capture_update,
                        one_graph=one_graph, tf32=tf32, env_backend=env_backend)
    trainer = PPORND(cfg, device="cuda")
    trainer.prime_obs_rms()
    if one_graph:
        trainer._build_iteration_graph()
    elif rollout_mode == "capture":
        trainer._build_rollout_graph()
    update = trainer.update_captured if (capture_update and not one_graph) else None
    if update is None:
        update = trainer.update_full_batch if style == "full_batch" else trainer.update_epoch_minibatch

    for _ in range(warmup):
        if one_graph:
            trainer.iteration_captured()
        else:
            update(trainer.rollout())
    torch.cuda.synchronize()

    t_roll = t_upd = 0.0
    if timing == "pipelined" and one_graph:
        # issue every iteration and wait once, so the host can run ahead of the device —
        # the arrangement the jax benchmark uses, measured here for comparison
        t0 = time.perf_counter()
        for _ in range(iters):
            trainer.iteration_captured()
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        env_steps = trainer.cfg.num_steps * n_copies * trainer.cfg.n_envs * iters
        return {"n_copies": n_copies, "style": style, "env_backend": env_backend,
                "timing": timing, "iters_timed": iters, "sec_per_iteration": total / iters,
                "rollout_sec_per_iter": None, "update_sec_per_iter": None,
                "iterations_per_sec": iters / total, "env_steps_per_sec": env_steps / total,
                "env_steps_per_sec_per_copy": env_steps / total / n_copies,
                "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20}
    t0 = time.perf_counter()
    for _ in range(iters):
        if one_graph:
            trainer.iteration_captured()
            torch.cuda.synchronize()
            continue
        a = time.perf_counter()
        batch = trainer.rollout()
        torch.cuda.synchronize()
        b = time.perf_counter()
        update(batch)
        torch.cuda.synchronize()
        c = time.perf_counter()
        t_roll += b - a
        t_upd += c - b
    total = time.perf_counter() - t0
    env_steps = trainer.cfg.num_steps * n_copies * trainer.cfg.n_envs * iters
    return {
        "n_copies": n_copies, "style": style, "env_backend": env_backend,
        "timing": timing, "iters_timed": iters,
        "compile_post": cfg.compile_post, "tf32": cfg.tf32, "one_graph": cfg.one_graph,
        "sec_per_iteration": total / iters,
        "rollout_sec_per_iter": t_roll / iters, "update_sec_per_iter": t_upd / iters,
        "iterations_per_sec": iters / total,
        "env_steps_per_sec": env_steps / total,
        "env_steps_per_sec_per_copy": env_steps / total / n_copies,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--n-copies", type=int, nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--rollout-mode", default="eager")
    ap.add_argument("--capture-update", action="store_true")
    ap.add_argument("--one-graph", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--env-backend", default="torch")
    ap.add_argument("--timing", default="sync", choices=["sync", "pipelined"])
    ap.add_argument("--fused-adam", action="store_true")
    ap.add_argument("--rev", default="", help="git revision of the trainer to measure instead "
                                              "of the working tree")
    args = ap.parse_args()

    git = subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    out = RESULTS / f"{stamp}_trainbench_torch_{args.style}{args.tag}.json"
    rows, failures = [], []
    for c in args.n_copies:
        # one copy-count failing (e.g. OOM at the memory limit) must not lose earlier rows
        try:
            r = bench(c, args.style, args.iters, args.warmup, args.rollout_mode,
                      args.fused_adam, args.capture_update, args.one_graph, args.tf32,
                      args.env_backend, args.timing, args.rev)
        except Exception as e:
            failures.append({"n_copies": c, "error": repr(e)[:400]})
            print(f"torch_ppo/{args.style} C={c:>4d}: FAILED {e!r}")
            break
        rows.append(r)
        split = (f"(rollout {r['rollout_sec_per_iter']*1e3:.1f} + "
                 f"update {r['update_sec_per_iter']*1e3:.1f}) "
                 if r.get("rollout_sec_per_iter") is not None else f"({r['timing']}) ")
        print(f"torch_ppo/{args.style} C={c:>4d}: {r['sec_per_iteration']*1e3:.1f} ms/iter "
              f"{split}= {r['env_steps_per_sec']:.3e} env-steps/s total")
        out.write_text(json.dumps({"impl": "torch_ppo", "style": args.style, "git": git,
                                   "trainer_revision": args.rev or "working tree",
                                   "gpu": torch.cuda.get_device_name(0),
                                   "torch": torch.__version__, "rows": rows,
                                   "failures": failures}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
