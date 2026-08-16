"""Per-part timing breakdown of one training iteration (the deliverable's profiling table).

Times each logical block with CUDA events at a fixed config, in two views:
  1. component view — each block called in isolation on real shapes (eager kernels), so the
     proportions of logical work are visible: env dynamics vs policy/value forwards vs RND
     bonus vs buffer writes vs filter/GAE/statistics vs update sub-steps;
  2. production view — the captured-graph phase totals actually paid per iteration
     (rollout replay, post, update replay), for the true end-to-end split.

Writes one JSON to results/ and prints both tables.
Usage (serval05, under the lock): python profile_breakdown.py --n-copies 128
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


def cuda_time(fn, reps=50, warmup=10):
    """Median CUDA time of fn() in microseconds via events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1000)
    return sorted(times)[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-copies", type=int, default=128)
    args = ap.parse_args()
    from torch_ppo_rnd import PPOConfig, PPORND

    C = args.n_copies
    trainer = PPORND(PPOConfig(n_copies=C, rollout_mode="capture", capture_update=True,
                               fused_adam=True), device="cuda")
    torch.manual_seed(0)
    trainer.prime_obs_rms()
    trainer._build_rollout_graph()
    batch = trainer.rollout()
    trainer.update_captured(batch)      # builds the update graph
    N, T = trainer.cfg.n_envs, trainer.cfg.num_steps
    env = trainer.env
    obs = trainer._S_obs.clone()
    act = torch.rand(C, N, 2, device="cuda") * 2 - 1
    state = (env.pos.clone(), env.vel.clone(), env.goal.clone(),
             env.step_count.clone(), env.reset_count.clone())

    # ---- component view: per-step blocks (x T = per-rollout cost), eager kernels ----
    comp = {}
    comp["env dynamics_step (integrator + contacts)"] = cuda_time(
        lambda: env.dynamics_step(state[0], state[1], act)) * T
    comp["env step_core total (dynamics + reward/ends + auto-reset RNG)"] = cuda_time(
        lambda: env.step_core(*state, act)) * T
    comp["policy forward (actor mean)"] = cuda_time(lambda: trainer.actor_mean(obs)) * T
    comp["value forward (critic, both heads)"] = cuda_time(lambda: trainer.critic_values(obs)) * T
    wh = trainer.whiten(obs)
    comp["RND bonus (whiten + target + predictor)"] = cuda_time(
        lambda: trainer.rnd_features(trainer.whiten(obs))) * T
    z = torch.randn(C, N, 2, device="cuda")
    comp["action sample + logprob"] = cuda_time(
        lambda: ((trainer.actor["logstd"].exp().unsqueeze(1) * z),
                 (-0.5 * z * z - trainer.actor["logstd"].unsqueeze(1)).sum(-1))) * T
    comp["rollout buffer writes (10 copies/step)"] = cuda_time(
        lambda: [trainer._bufs[k][0].copy_(trainer._bufs[k][1]) for k in trainer._bufs]) * T

    # post blocks (once per iteration)
    b = trainer._bufs
    comp["post: bootstrap values (one big GEMM)"] = cuda_time(
        lambda: trainer.critic_values(b["nobs"].permute(1, 0, 2, 3).reshape(C, T * N, 4)))
    comp["post: intrinsic filter scan (T steps)"] = cuda_time(
        lambda: [trainer.int_filter.mul_(0.99).add_(b["rint"][t]) for t in range(T)])
    def gae():
        aext = torch.zeros(C, N, device="cuda")
        for t in range(T - 1, -1, -1):
            aext = b["rext"][t] + 0.95 * aext
    comp["post: GAE backward scan (both streams approx)"] = cuda_time(gae) * 2
    comp["post: running-statistics updates (float64)"] = cuda_time(
        lambda: trainer.obs_rms.update(b["nobs"].permute(1, 0, 2, 3).reshape(C, T * N, 4)))
    comp["post: flatten + whiten into static batch"] = cuda_time(
        lambda: trainer._U["rnd_input"].copy_(
            trainer.whiten(b["nobs"].permute(1, 0, 2, 3).reshape(C, T * N, 4))) ) * 8

    # update blocks (x 16 minibatch steps)
    idx = torch.arange(128, device="cuda").expand(C, 128)
    mb = {}
    for key in trainer._U_KEYS:
        t_ = trainer._U[key]
        ix = idx.unsqueeze(-1).expand(C, 128, t_.shape[-1]) if t_.dim() == 3 else idx
        mb[key] = t_.gather(1, ix)
    comp["update: minibatch gathers"] = cuda_time(
        lambda: [trainer._U[k].gather(1, idx.unsqueeze(-1).expand(C, 128, trainer._U[k].shape[-1])
                 if trainer._U[k].dim() == 3 else idx) for k in trainer._U_KEYS]) * 16
    comp["update: loss forward"] = cuda_time(lambda: trainer._loss_fn(mb, style_a=False)) * 16
    def fwd_bwd():
        loss = trainer._loss_fn(mb, style_a=False)
        trainer._last_grads = trainer._backward(loss)
    comp["update: forward + backward"] = cuda_time(fwd_bwd) * 16
    comp["update: per-copy clip + Adam step"] = cuda_time(
        lambda: trainer._clip_per_copy_and_step(trainer._last_grads)) * 16

    # ---- production view: captured phase totals ----
    prod = {}
    prod["rollout graph replay (T=128 steps: policy+value+sample+env+RND+writes)"] = cuda_time(
        lambda: trainer._graph.replay(), reps=30)
    prod["post-processing (bootstrap, filter, GAE, statistics, flatten)"] = cuda_time(
        lambda: trainer._post_body(), reps=30)
    prod["update graph replay (16 minibatch steps: gather+fwd+bwd+clip+Adam)"] = cuda_time(
        lambda: trainer._update_graph.replay(), reps=30)
    prod["noise + permutation refill (outside graphs)"] = cuda_time(
        lambda: (trainer._Z.normal_(),
                 trainer._perm.copy_(torch.rand(4, C, T * N, device="cuda").argsort(-1))))

    total = sum(prod.values())
    print(f"\n== production view (captured, C={C}) — one iteration ==")
    for k, v in prod.items():
        print(f"  {v/1000:8.2f} ms  {v/total*100:5.1f}%  {k}")
    print(f"  {total/1000:8.2f} ms  100.0%  TOTAL")
    ctot = sum(comp.values())
    print(f"\n== component view (eager kernels, per iteration equivalents, C={C}) ==")
    for k, v in sorted(comp.items(), key=lambda kv: -kv[1]):
        print(f"  {v/1000:8.2f} ms  {v/ctot*100:5.1f}%  {k}")

    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    out = RESULTS / f"{stamp}_profile_breakdown_C{C}.json"
    out.write_text(json.dumps({
        "n_copies": C, "torch": torch.__version__,
        "git": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "production_us": prod, "component_us": comp}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
