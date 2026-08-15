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


def bench(n_copies, style, iters, warmup, rollout_mode="eager", fused_adam=False, capture_update=False, one_graph=False, tf32=False, env_backend="torch"):
    """Time full iterations and the rollout/update split for one copy count."""
    from torch_ppo_rnd import PPOConfig, PPORND
    trainer = PPORND(PPOConfig(n_copies=n_copies, update_style=style,
                               rollout_mode=rollout_mode, fused_adam=fused_adam,
                               capture_update=capture_update, one_graph=one_graph, tf32=tf32,
                               env_backend=env_backend),
                     device="cuda")
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
        "n_copies": n_copies, "style": style, "env_backend": env_backend, "iters_timed": iters,
        "sec_per_iteration": total / iters,
        "rollout_sec_per_iter": t_roll / iters, "update_sec_per_iter": t_upd / iters,
        "iterations_per_sec": iters / total,
        "env_steps_per_sec": env_steps / total,
        "env_steps_per_sec_per_copy": env_steps / total / n_copies,
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
    ap.add_argument("--fused-adam", action="store_true")
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
                      args.env_backend)
        except Exception as e:
            failures.append({"n_copies": c, "error": repr(e)[:400]})
            print(f"torch_ppo/{args.style} C={c:>4d}: FAILED {e!r}")
            break
        rows.append(r)
        print(f"torch_ppo/{args.style} C={c:>4d}: {r['sec_per_iteration']*1e3:.1f} ms/iter "
              f"(rollout {r['rollout_sec_per_iter']*1e3:.1f} + update {r['update_sec_per_iter']*1e3:.1f}) "
              f"= {r['env_steps_per_sec']:.3e} env-steps/s total")
        out.write_text(json.dumps({"impl": "torch_ppo", "style": args.style, "git": git,
                                   "gpu": torch.cuda.get_device_name(0),
                                   "torch": torch.__version__, "rows": rows,
                                   "failures": failures}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
