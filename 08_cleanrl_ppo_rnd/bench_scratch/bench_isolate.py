"""Isolated per-knob benchmark: one knob per process, so no autotune cache or allocator state carries over.

Also measures a full simulated PPO+RND iteration (128 rollout steps + 16 minibatch updates) in the
'as-written' configuration versus a 'same-numerics-only' optimized configuration, and manual CUDA
graph capture of the rollout step.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_ppo_rnd import Agent, RNDModel, NUM_ENVS, NUM_STEPS, BATCH, MINIBATCH, N_ACTIONS, timeit, wallclock

UPDATE_PROPORTION, MAX_GRAD_NORM = 0.25, 0.5


def build(dev, channels_last=False):
    """Creates agent, RND model, optimizer and the fixed minibatch tensors used by every config."""
    torch.manual_seed(0)
    agent, rnd = Agent().to(dev), RNDModel().to(dev)
    if channels_last:
        agent.to(memory_format=torch.channels_last)
        rnd.to(memory_format=torch.channels_last)
    combined = list(agent.parameters()) + list(rnd.predictor.parameters())
    d = dict(
        agent=agent, rnd=rnd, combined=combined,
        mb_obs=torch.rand(MINIBATCH, 4, 84, 84, device=dev) * 255,
        rnd_in=torch.randn(MINIBATCH, 1, 84, 84, device=dev),
        acts=torch.randint(0, N_ACTIONS, (MINIBATCH,), device=dev),
        logp=torch.randn(MINIBATCH, device=dev),
        adv=torch.randn(MINIBATCH, device=dev),
        er=torch.randn(MINIBATCH, device=dev),
        ir=torch.randn(MINIBATCH, device=dev),
        ev=torch.randn(MINIBATCH, device=dev),
    )
    if channels_last:
        d["mb_obs"] = d["mb_obs"].to(memory_format=torch.channels_last)
        d["rnd_in"] = d["rnd_in"].to(memory_format=torch.channels_last)
    return d


def make_update(d, opt, amp_dtype=None, scaler=None, dev=None):
    """Returns a closure running one PPO+RND minibatch update with the given precision settings."""
    agent, rnd, combined = d["agent"], d["rnd"], d["combined"]

    def run():
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            pf, tf = rnd(d["rnd_in"])
            fwd = F.mse_loss(pf, tf.detach(), reduction="none").mean(-1)
            mask = (torch.rand(len(fwd), device=dev) < UPDATE_PROPORTION).float()
            fwd = (fwd * mask).sum() / torch.max(mask.sum(), torch.tensor([1.0], device=dev))
            _, nlp, ent, nev, niv = agent.get_action_and_value(d["mb_obs"], d["acts"])
            ratio = (nlp - d["logp"]).exp()
            a = (d["adv"] - d["adv"].mean()) / (d["adv"].std() + 1e-8)
            pg = torch.max(-a * ratio, -a * torch.clamp(ratio, 0.9, 1.1)).mean()
            nev, niv = nev.view(-1), niv.view(-1)
            evc = (d["ev"] + torch.clamp(nev - d["ev"], -0.1, 0.1) - d["er"]) ** 2
            vl = 0.5 * torch.max((nev - d["er"]) ** 2, evc).mean() + 0.5 * ((niv - d["ir"]) ** 2).mean()
            loss = pg - 0.001 * ent.mean() + vl * 0.5 + fwd
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(combined, MAX_GRAD_NORM)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(combined, MAX_GRAD_NORM)
            opt.step()
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(0)
    cfg = a.config
    out = {"config": cfg, "gpu": p.name, "cap": f"sm_{p.major}{p.minor}", "host": os.uname().nodename}

    # ---- knob settings, one per config name ---------------------------------------------------
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cl = "channels_last" in cfg
    amp = torch.bfloat16 if "bf16" in cfg else (torch.float16 if "fp16" in cfg else None)
    if "bench" in cfg or "tf32" in cfg or "amp" in cfg or "compile" in cfg or "full" in cfg:
        torch.backends.cudnn.benchmark = True
    if "tf32" in cfg or "amp" in cfg or "compile" in cfg or cfg.startswith("full_opt"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if "cudnn_tf32_only" in cfg:
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
    if "matmul_tf32_only" in cfg:
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = False

    out["knobs"] = dict(cudnn_benchmark=torch.backends.cudnn.benchmark,
                        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                        cudnn_tf32=torch.backends.cudnn.allow_tf32,
                        channels_last=cl, amp=str(amp))

    d = build(dev, channels_last=cl)
    fused = "fused" in cfg
    opt = optim.Adam(d["combined"], lr=1e-4, eps=1e-5, **({"fused": True} if fused else {}))
    scaler = torch.amp.GradScaler("cuda") if amp is torch.float16 else None

    # ---- minibatch update -------------------------------------------------------------------
    run = make_update(d, opt, amp_dtype=amp, scaler=scaler, dev=dev)
    if "compile_update" in cfg:
        t0 = time.perf_counter()
        d["agent"] = torch.compile(d["agent"])
        d["rnd"] = torch.compile(d["rnd"])
        run = make_update(d, opt, amp_dtype=amp, scaler=scaler, dev=dev)
        run(); torch.cuda.synchronize()
        out["compile_seconds"] = round(time.perf_counter() - t0, 1)
    out["update_ms"] = round(timeit(run, 8, 30), 3)

    # ---- rollout step: eager, compiled, manual CUDA graph -------------------------------------
    agent = d["agent"] if not isinstance(d["agent"], torch._dynamo.eval_frame.OptimizedModule) else d["agent"]._orig_mod
    step_obs = torch.rand(NUM_ENVS, 4, 84, 84, device=dev) * 255
    rnd_in = torch.randn(NUM_ENVS, 1, 84, 84, device=dev)
    if cl:
        step_obs = step_obs.to(memory_format=torch.channels_last)
        rnd_in = rnd_in.to(memory_format=torch.channels_last)
    rnd = d["rnd"] if not isinstance(d["rnd"], torch._dynamo.eval_frame.OptimizedModule) else d["rnd"]._orig_mod

    def rollout_as_written():
        """Two trunk passes + RND target/predictor with grad mode on, as the script does."""
        with torch.no_grad():
            agent.get_value(step_obs)
            act, lp, _, _, _ = agent.get_action_and_value(step_obs)
        tf = rnd.target(rnd_in)
        pf = rnd.predictor(rnd_in)
        return act, ((tf - pf).pow(2).sum(1) / 2).data

    def rollout_min():
        """One trunk pass + RND under no_grad; same values."""
        with torch.no_grad():
            act, lp, ve, vi = agent.get_action_and_value_fused(step_obs)
            tf = rnd.target(rnd_in); pf = rnd.predictor(rnd_in)
            return act, (tf - pf).pow(2).sum(1) / 2

    out["rollout_as_written_ms"] = round(timeit(rollout_as_written, 15, 100), 4)
    out["rollout_min_ms"] = round(timeit(rollout_min, 15, 100), 4)

    if "cudagraph" in cfg:
        # manual CUDA graph capture of the minimal rollout step (static input buffers)
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    rollout_min()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_out = rollout_min()
            out["rollout_cudagraph_ms"] = round(timeit(g.replay, 20, 200), 4)
        except Exception as e:
            out["rollout_cudagraph_ms"] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"

    if "compile_rollout" in cfg:
        for mode in ["default", "reduce-overhead"]:
            try:
                torch._dynamo.reset()
                ca = torch.compile(agent.get_action_and_value_fused, mode=None if mode == "default" else mode)
                cr_t = torch.compile(rnd.target, mode=None if mode == "default" else mode)
                cr_p = torch.compile(rnd.predictor, mode=None if mode == "default" else mode)

                def cfn():
                    with torch.no_grad():
                        act = ca(step_obs)[0]
                        tf, pf = cr_t(rnd_in), cr_p(rnd_in)
                        return act, (tf - pf).pow(2).sum(1) / 2
                t0 = time.perf_counter(); cfn(); torch.cuda.synchronize()
                out[f"compile_rollout[{mode}]_compile_s"] = round(time.perf_counter() - t0, 1)
                out[f"rollout_compiled[{mode}]_ms"] = round(timeit(cfn, 20, 150), 4)
            except Exception as e:
                out[f"rollout_compiled[{mode}]_ms"] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"

    out["update_x16_ms"] = round(out["update_ms"] * 16, 1)
    out["rollout_x128_as_written_ms"] = round(out["rollout_as_written_ms"] * 128, 1)
    out["rollout_x128_min_ms"] = round(out["rollout_min_ms"] * 128, 1)
    out["gpu_only_iter_as_written_s"] = round((out["update_x16_ms"] + out["rollout_x128_as_written_ms"]) / 1000, 3)
    out["gpu_only_steps_per_s_ceiling"] = round(BATCH / out["gpu_only_iter_as_written_s"], 0)
    out["mem_peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    print(json.dumps(out, indent=2), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
