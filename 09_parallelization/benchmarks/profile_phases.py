"""Phase and component breakdown of one training iteration, for the current code.

Two views, because they answer different questions:

  phases      the three phases of an iteration timed as separately captured graphs — rollout,
              post-rollout processing, update. Their sum is slightly more than the shipped
              one-graph configuration, which captures all three together; that total is
              measured too, so the difference is visible rather than assumed.
  components  individual blocks called on their real shapes with ordinary (uncompiled)
              kernels. This shows where the work is before fusion, which is what explains
              why the optimizations that were kept were the ones that were kept. It is NOT a
              decomposition of the shipped time: compiling and capturing removes most of the
              per-call overhead these numbers include.

Usage (through the H100 lock wrapper): python profile_phases.py --n-copies 128
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


def cuda_time(fn, reps=40, warmup=8):
    """Median CUDA time of fn() in microseconds, measured with events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1000.0)
    return sorted(times)[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, default=128)
    ap.add_argument("--style", default="epoch_minibatch")
    ap.add_argument("--rev", default="", help="git revision of the trainer to profile instead "
                                              "of the working tree, for before/after pairing")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    # ab_compare already knows how to load a past revision of the trainer as a module; reusing
    # it means one implementation of that trick rather than two that can drift apart
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ab_compare import load_module
    mod = load_module(args.rev)
    PPORND, production_config = mod.PPORND, mod.production_config

    C = args.n_copies
    # separate graphs per phase, so each phase can be replayed and timed on its own
    sep = PPORND(production_config(C, style=args.style, one_graph=False), device="cuda")
    sep.prime_obs_rms()
    sep._build_rollout_graph()
    batch = sep.rollout()
    sep.update_captured(batch)                       # builds the update graph
    phases = {
        "rollout (128 sequential steps: policy, environment, buffer writes)":
            cuda_time(lambda: sep._graph.replay(), reps=25),
        "post-rollout processing (values, log-probabilities, RND bonus, filter, GAE, statistics)":
            cuda_time(lambda: sep._post_body(), reps=25),
        "update (16 minibatch steps: gather, forward, backward, clip, Adam)":
            cuda_time(lambda: sep._update_graph.replay(), reps=25),
    }
    del sep
    torch.cuda.empty_cache()

    one = PPORND(production_config(C, style=args.style), device="cuda")
    one.prime_obs_rms()
    one._build_iteration_graph()
    whole = cuda_time(lambda: one._iteration_graph.replay(), reps=25)

    # component view on the same shapes, uncompiled, to show where the work sits
    env, cfg = one.env, one.cfg
    T, N = cfg.num_steps, cfg.n_envs
    obs = one._S_obs.clone()
    act = torch.rand(C, N, 2, device="cuda") * 2 - 1
    state = (env.pos.clone(), env.vel.clone(), env.goal.clone(),
             env.step_count.clone(), env.reset_count.clone())
    flat_obs = one._U["obs"]
    comp = {
        "environment: physics step (integrator and contacts)":
            cuda_time(lambda: env.dynamics_step(state[0], state[1], act)) * T,
        "environment: whole step (physics, reward, episode ends, automatic reset)":
            cuda_time(lambda: env.step_core(*state, act)) * T,
        "policy network forward (in the sequential loop)":
            cuda_time(lambda: one.actor_mean(obs)) * T,
        "value network forward (one wide pass after the loop)":
            cuda_time(lambda: one.critic_values(flat_obs)),
        "RND bonus (whitening, target and predictor networks, one wide pass)":
            cuda_time(lambda: one.rnd_features(one.whiten(flat_obs))),
        "advantage estimation and running statistics (post-rollout scans)":
            cuda_time(lambda: one._post_body()) - cuda_time(lambda: one.critic_values(flat_obs)),
        "one update step (forward, backward, clip, Adam) x 16":
            cuda_time(lambda: one._update_body_captured()),
    }

    print(f"\n== phases of one iteration, {C} copies ==")
    tot = sum(phases.values())
    for k, v in phases.items():
        print(f"  {v/1000:8.2f} ms  {v/tot*100:5.1f}%  {k}")
    print(f"  {tot/1000:8.2f} ms  100.0%  sum of the three phases (separate graphs)")
    print(f"  {whole/1000:8.2f} ms           the same iteration captured as ONE graph (shipped)")
    print(f"\n== components on the same shapes, uncompiled ==")
    for k, v in sorted(comp.items(), key=lambda kv: -kv[1]):
        print(f"  {v/1000:8.2f} ms  {k}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_profile_phases_C{C}{args.tag}.json"
    out.write_text(json.dumps({
        "n_copies": C, "style": args.style, "torch": torch.__version__,
        "trainer_revision": args.rev or "working tree",
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "phases_us": phases, "phases_sum_us": tot, "one_graph_us": whole,
        "components_us": comp}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
