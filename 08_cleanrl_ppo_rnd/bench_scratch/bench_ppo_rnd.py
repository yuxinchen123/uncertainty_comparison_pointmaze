"""Microbenchmark of the CleanRL ppo_rnd_envpool.py GPU workload, one optimization at a time.

Self-contained: reproduces the exact Agent / RNDModel shapes and the exact rollout-step and
minibatch-update sequences from ppo_rnd_envpool.py, with envpool replaced by a numpy uint8
array of the right shape, so it runs anywhere a GPU is visible.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

NUM_ENVS, NUM_STEPS, N_ACTIONS = 128, 128, 18
BATCH = NUM_ENVS * NUM_STEPS          # 16384
MINIBATCH = BATCH // 4                # 4096
UPDATE_PROPORTION, MAX_GRAD_NORM = 0.25, 0.5


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal init exactly as in the CleanRL script."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Nature-CNN policy with extra layer and two value heads, copied from ppo_rnd_envpool.py."""

    def __init__(self, n_actions=N_ACTIONS):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)), nn.ReLU(),
            layer_init(nn.Linear(256, 448)), nn.ReLU(),
        )
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01), nn.ReLU(),
            layer_init(nn.Linear(448, n_actions), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action_and_value(self, x, action=None):
        """Policy forward returning action, logprob, entropy and both values."""
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        if action is None:
            action = probs.sample()
        return (action, probs.log_prob(action), probs.entropy(),
                self.critic_ext(features + hidden), self.critic_int(features + hidden))

    def get_value(self, x):
        """Value-only forward; note it recomputes the whole CNN trunk."""
        hidden = self.network(x / 255.0)
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)

    def get_action_and_value_fused(self, x):
        """Single trunk pass giving both the values and the sampled action (rollout-only merge)."""
        hidden = self.network(x / 255.0)
        features = self.extra_layer(hidden)
        value_ext, value_int = self.critic_ext(features + hidden), self.critic_int(features + hidden)
        probs = Categorical(logits=self.actor(hidden))
        action = probs.sample()
        return action, probs.log_prob(action), value_ext, value_int


class RNDModel(nn.Module):
    """RND predictor and frozen target towers on 1x84x84 input, copied from ppo_rnd_envpool.py."""

    def __init__(self):
        super().__init__()
        feature_output = 7 * 7 * 64
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 512)), nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
        )
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x):
        """Returns (predictor features, target features)."""
        return self.predictor(x), self.target(x)


def timeit(fn, n_warmup, n_iter):
    """Times fn() on the GPU with events after a warmup; returns mean milliseconds per call."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def wallclock(fn, n_warmup, n_iter):
    """Times fn() by wall clock including any host-side work and syncs; mean milliseconds."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / n_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    res = {
        "gpu": props.name,
        "capability": f"sm_{props.major}{props.minor}",
        "total_mem_GB": round(props.total_memory / 1e9, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "arch_list": torch.cuda.get_arch_list(),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "host": os.uname().nodename,
        "timings_ms": {},
        "notes": [],
    }
    print(json.dumps({k: v for k, v in res.items() if k != "timings_ms"}, indent=2), flush=True)

    torch.manual_seed(0)
    np.random.seed(0)

    # ---- storage exactly as the script allocates it -------------------------------------------
    # BEFORE: obs = torch.zeros((128,128,4,84,84)) float32 -> 462,422,016 elems -> 1.849 GB
    # AFTER (uint8 variant): same element count at 1 byte -> 0.462 GB
    obs_f32 = torch.zeros((NUM_STEPS, NUM_ENVS, 4, 84, 84), device=dev)
    res["obs_buffer_bytes_f32"] = obs_f32.numel() * 4
    b_obs = obs_f32.reshape(BATCH, 4, 84, 84)

    agent = Agent().to(dev)
    rnd = RNDModel().to(dev)
    combined = list(agent.parameters()) + list(rnd.predictor.parameters())
    res["n_params_agent"] = sum(p.numel() for p in agent.parameters())
    res["n_params_rnd_predictor"] = sum(p.numel() for p in rnd.predictor.parameters())
    res["n_params_rnd_target"] = sum(p.numel() for p in rnd.target.parameters())

    # host-side env output, as envpool returns it: uint8 (num_envs,4,84,84)
    np_obs_u8 = np.random.randint(0, 255, (NUM_ENVS, 4, 84, 84), dtype=np.uint8)
    np_obs_u8_pinned = torch.from_numpy(np_obs_u8).pin_memory()
    obs_rms_mean_f64 = np.zeros((1, 1, 84, 84), dtype=np.float64)
    obs_rms_var_f64 = np.ones((1, 1, 84, 84), dtype=np.float64)
    mean_g32 = torch.from_numpy(obs_rms_mean_f64).to(dev).float()
    var_g32 = torch.from_numpy(obs_rms_var_f64).to(dev).float()

    step_obs = obs_f32[0]                       # (128,4,84,84)
    rnd_in_128 = torch.randn(NUM_ENVS, 1, 84, 84, device=dev)
    rnd_in_mb = torch.randn(MINIBATCH, 1, 84, 84, device=dev)
    mb_obs = b_obs[:MINIBATCH]
    b_actions = torch.randint(0, N_ACTIONS, (MINIBATCH,), device=dev)
    b_logprobs = torch.randn(MINIBATCH, device=dev)
    b_adv = torch.randn(MINIBATCH, device=dev)
    b_ext_ret = torch.randn(MINIBATCH, device=dev)
    b_int_ret = torch.randn(MINIBATCH, device=dev)
    b_ext_val = torch.randn(MINIBATCH, device=dev)

    T = res["timings_ms"]

    def record(name, ms):
        T[name] = round(ms, 4)
        print(f"  {name:58s} {ms:9.3f} ms", flush=True)

    # =========================================================================================
    # SECTION 1: the per-rollout-step pieces (batch 128), 128 of these per iteration
    # =========================================================================================
    print("\n[1] rollout step components, batch=128 (128x per iteration)", flush=True)

    def policy_as_written():
        """Two full trunk passes, exactly as the script does per rollout step."""
        with torch.no_grad():
            ve, vi = agent.get_value(step_obs)
            a, lp, _, _, _ = agent.get_action_and_value(step_obs)
        return a

    def policy_merged():
        """One trunk pass giving values and action (algebraically identical outputs)."""
        with torch.no_grad():
            return agent.get_action_and_value_fused(step_obs)[0]

    record("rollout.policy_two_trunk_passes(as written)", timeit(policy_as_written, 10, 60))
    record("rollout.policy_one_trunk_pass(merged)", timeit(policy_merged, 10, 60))

    act = policy_merged()

    def action_to_host():
        """The .cpu().numpy() that hard-syncs the GPU every rollout step."""
        return act.cpu().numpy()

    record("rollout.action.cpu().numpy() [forced sync]", wallclock(action_to_host, 10, 200))

    def h2d_as_written():
        """torch.Tensor(np_uint8) converts uint8->float32 on the CPU then copies 14.5 MB H2D."""
        return torch.Tensor(np_obs_u8).to(dev)

    def h2d_uint8_then_cast():
        """Copy 3.6 MB of uint8 H2D and cast on the GPU."""
        return torch.from_numpy(np_obs_u8).to(dev, non_blocking=False).float()

    def h2d_uint8_pinned():
        """Same but from pinned host memory with a non-blocking copy."""
        return np_obs_u8_pinned.to(dev, non_blocking=True).float()

    record("rollout.h2d torch.Tensor(np).to(cuda) [f32, as written]", wallclock(h2d_as_written, 10, 100))
    record("rollout.h2d uint8 then .float() on gpu", wallclock(h2d_uint8_then_cast, 10, 100))
    record("rollout.h2d uint8 pinned non_blocking", wallclock(h2d_uint8_pinned, 10, 100))

    next_obs = torch.Tensor(np_obs_u8).to(dev)

    def norm_as_written():
        """RND input normalisation as written: obs_rms.mean/var are float64 numpy, re-uploaded each step."""
        return (((next_obs[:, 3, :, :].reshape(NUM_ENVS, 1, 84, 84)
                  - torch.from_numpy(obs_rms_mean_f64).to(dev))
                 / torch.sqrt(torch.from_numpy(obs_rms_var_f64).to(dev))).clip(-5, 5)).float()

    def norm_f32_resident():
        """Same maths with float32 statistics already resident on the GPU."""
        return (((next_obs[:, 3, :, :].reshape(NUM_ENVS, 1, 84, 84) - mean_g32)
                 / torch.sqrt(var_g32)).clip(-5, 5))

    record("rollout.rnd_norm float64 + H2D of stats (as written)", wallclock(norm_as_written, 10, 100))
    record("rollout.rnd_norm float32 gpu-resident stats", wallclock(norm_f32_resident, 10, 100))

    def rnd_rollout_as_written():
        """Target then predictor with grad mode ON -> predictor builds an autograd graph that is discarded."""
        tf = rnd.target(rnd_in_128)
        pf = rnd.predictor(rnd_in_128)
        return ((tf - pf).pow(2).sum(1) / 2).data

    def rnd_rollout_nograd():
        """Same values under no_grad (target is frozen, result is .data'd anyway)."""
        with torch.no_grad():
            tf = rnd.target(rnd_in_128)
            pf = rnd.predictor(rnd_in_128)
            return (tf - pf).pow(2).sum(1) / 2

    record("rollout.rnd_fwd grad-mode ON (as written)", timeit(rnd_rollout_as_written, 10, 60))
    record("rollout.rnd_fwd under no_grad", timeit(rnd_rollout_nograd, 10, 60))

    # =========================================================================================
    # SECTION 2: minibatch update (4096), 16 of these per iteration
    # =========================================================================================
    print("\n[2] minibatch update, size=4096 (16x per iteration)", flush=True)

    def make_opt(**kw):
        """Fresh Adam over the combined parameter list."""
        return optim.Adam(combined, lr=1e-4, eps=1e-5, **kw)

    def update_step(opt, scaler=None, amp_dtype=None, set_to_none=True, channels_last=False):
        """One minibatch of the PPO+RND update, matching the script's loss and clipping."""
        def _run():
            o = mb_obs.to(memory_format=torch.channels_last) if channels_last else mb_obs
            r = rnd_in_mb.to(memory_format=torch.channels_last) if channels_last else rnd_in_mb
            ctx = (torch.autocast("cuda", dtype=amp_dtype) if amp_dtype
                   else torch.autocast("cuda", enabled=False))
            with ctx:
                pf, tf = rnd(r)
                fwd = F.mse_loss(pf, tf.detach(), reduction="none").mean(-1)
                mask = (torch.rand(len(fwd), device=dev) < UPDATE_PROPORTION).float()
                fwd = (fwd * mask).sum() / torch.max(mask.sum(), torch.tensor([1], device=dev, dtype=torch.float32))
                _, newlogprob, entropy, nev, niv = agent.get_action_and_value(o, b_actions)
                logratio = newlogprob - b_logprobs
                ratio = logratio.exp()
                mb_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
                pg = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.9, 1.1)).mean()
                nev, niv = nev.view(-1), niv.view(-1)
                evu = (nev - b_ext_ret) ** 2
                evc = (b_ext_val + torch.clamp(nev - b_ext_val, -0.1, 0.1) - b_ext_ret) ** 2
                ext_v = 0.5 * torch.max(evu, evc).mean()
                int_v = 0.5 * ((niv - b_int_ret) ** 2).mean()
                loss = pg - 0.001 * entropy.mean() + (ext_v + int_v) * 0.5 + fwd
            opt.zero_grad(set_to_none=set_to_none)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(combined, MAX_GRAD_NORM)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(combined, MAX_GRAD_NORM)
                opt.step()
        return _run

    def bench_update(label, **kw):
        """Builds an optimizer, times one minibatch update, records it."""
        opt = make_opt(**{k: v for k, v in kw.items() if k in ("fused", "foreach")})
        fn = update_step(opt, scaler=kw.get("scaler"), amp_dtype=kw.get("amp_dtype"),
                         set_to_none=kw.get("set_to_none", True),
                         channels_last=kw.get("channels_last", False))
        try:
            record(label, timeit(fn, 5, 25))
        except Exception as e:
            T[label] = f"FAILED: {type(e).__name__}: {str(e)[:160]}"
            print(f"  {label:58s} FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)

    # baseline knobs off
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bench_update("update.baseline (tf32 off, bench off, foreach adam)")
    bench_update("update.zero_grad(set_to_none=False)", set_to_none=False)
    bench_update("update.adam foreach=False (single tensor)", foreach=False)
    bench_update("update.adam fused=True", fused=True)

    torch.backends.cudnn.benchmark = True
    bench_update("update.+cudnn.benchmark=True")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    bench_update("update.+cudnn.benchmark +TF32(matmul+cudnn)")
    bench_update("update.+TF32 +fused adam", fused=True)

    agent_cl = agent.to(memory_format=torch.channels_last)
    rnd_cl = rnd.to(memory_format=torch.channels_last)
    bench_update("update.+TF32 +channels_last +fused adam", fused=True, channels_last=True)
    agent.to(memory_format=torch.contiguous_format)
    rnd.to(memory_format=torch.contiguous_format)

    if res["bf16_supported"]:
        bench_update("update.+TF32 +AMP bf16 +fused adam", fused=True, amp_dtype=torch.bfloat16)
        agent.to(memory_format=torch.channels_last); rnd.to(memory_format=torch.channels_last)
        bench_update("update.+AMP bf16 +channels_last +fused adam",
                     fused=True, amp_dtype=torch.bfloat16, channels_last=True)
        agent.to(memory_format=torch.contiguous_format); rnd.to(memory_format=torch.contiguous_format)
    else:
        T["update.+TF32 +AMP bf16 +fused adam"] = "SKIPPED: bf16 unsupported on this GPU"

    scaler = torch.amp.GradScaler("cuda")
    bench_update("update.+AMP fp16 +GradScaler +fused adam",
                 fused=True, amp_dtype=torch.float16, scaler=scaler)
    agent.to(memory_format=torch.channels_last); rnd.to(memory_format=torch.channels_last)
    scaler2 = torch.amp.GradScaler("cuda")
    bench_update("update.+AMP fp16 +channels_last +GradScaler +fused adam",
                 fused=True, amp_dtype=torch.float16, scaler=scaler2, channels_last=True)
    agent.to(memory_format=torch.contiguous_format); rnd.to(memory_format=torch.contiguous_format)

    # =========================================================================================
    # SECTION 3: torch.compile
    # =========================================================================================
    if not args.skip_compile:
        print("\n[3] torch.compile", flush=True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        for mode in ["default", "reduce-overhead", "max-autotune"]:
            try:
                torch._dynamo.reset()
                cnet = torch.compile(agent.network, mode=(None if mode == "default" else mode))
                t0 = time.perf_counter()
                with torch.no_grad():
                    cnet(step_obs / 255.0)
                torch.cuda.synchronize()
                T[f"compile.agent_trunk[{mode}].compile_seconds"] = round(time.perf_counter() - t0, 1)
                print(f"  compile.agent_trunk[{mode}] compile time "
                      f"{T[f'compile.agent_trunk[{mode}].compile_seconds']} s", flush=True)

                def _c():
                    with torch.no_grad():
                        return cnet(step_obs / 255.0)

                def _e():
                    with torch.no_grad():
                        return agent.network(step_obs / 255.0)

                record(f"compile.agent_trunk[{mode}] batch128 fwd", timeit(_c, 15, 80))
                record("compile.agent_trunk[eager] batch128 fwd", timeit(_e, 15, 80))
            except Exception as e:
                T[f"compile.agent_trunk[{mode}]"] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"
                print(f"  compile[{mode}] FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
        torch._dynamo.reset()

    # =========================================================================================
    # SECTION 4: float64 blowup in the per-iteration RND normalisation of the whole batch
    # =========================================================================================
    print("\n[4] per-iteration whole-batch work", flush=True)
    torch.cuda.empty_cache()

    def batch_norm_as_written():
        """(16384,1,84,84) normalisation with float64 stats -> float64 temporaries."""
        r = (((b_obs[:, 3, :, :].reshape(-1, 1, 84, 84) - torch.from_numpy(obs_rms_mean_f64).to(dev))
              / torch.sqrt(torch.from_numpy(obs_rms_var_f64).to(dev))).clip(-5, 5)).float()
        return r

    def batch_norm_f32():
        """Same on float32 gpu-resident stats."""
        return (((b_obs[:, 3, :, :].reshape(-1, 1, 84, 84) - mean_g32) / torch.sqrt(var_g32)).clip(-5, 5))

    try:
        record("iter.rnd_norm whole batch float64 (as written)", wallclock(batch_norm_as_written, 2, 8))
    except Exception as e:
        T["iter.rnd_norm whole batch float64 (as written)"] = f"FAILED: {type(e).__name__}: {str(e)[:160]}"
        print(f"  float64 whole-batch norm FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
    torch.cuda.empty_cache()
    record("iter.rnd_norm whole batch float32 gpu stats", wallclock(batch_norm_f32, 2, 8))
    torch.cuda.empty_cache()

    def obs_rms_update_cpu():
        """The per-iteration obs_rms.update: 462 MB D2H then numpy mean/var over 115.6M elements."""
        arr = b_obs[:, 3, :, :].reshape(-1, 1, 84, 84).cpu().numpy()
        return np.mean(arr, axis=0), np.var(arr, axis=0)

    def obs_rms_update_gpu():
        """Same moments computed on the GPU, no transfer."""
        v = b_obs[:, 3, :, :].reshape(-1, 1, 84, 84)
        return v.mean(0), v.var(0, unbiased=False)

    record("iter.obs_rms.update D2H+numpy (as written)", wallclock(obs_rms_update_cpu, 1, 3))
    record("iter.obs_rms.update on gpu", wallclock(obs_rms_update_gpu, 2, 8))

    def gae_loop():
        """The 128-step python GAE loop over (128,) cuda tensors."""
        rewards = torch.randn(NUM_STEPS, NUM_ENVS, device=dev)
        values = torch.randn(NUM_STEPS, NUM_ENVS, device=dev)
        dones = torch.zeros(NUM_STEPS, NUM_ENVS, device=dev)
        adv = torch.zeros_like(rewards)
        last = 0
        nv = torch.randn(1, NUM_ENVS, device=dev)
        for t in reversed(range(NUM_STEPS)):
            nnt = 1.0 - (dones[t + 1] if t < NUM_STEPS - 1 else torch.zeros(NUM_ENVS, device=dev))
            nvals = values[t + 1] if t < NUM_STEPS - 1 else nv
            d = rewards[t] + 0.999 * nvals * nnt - values[t]
            adv[t] = last = d + 0.999 * 0.95 * nnt * last
        return adv

    record("iter.gae python loop (x2 for ext+int)", wallclock(gae_loop, 2, 10))

    res["mem_max_alloc_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    res["mem_max_reserved_GB"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)

    print("\n=== RESULT JSON ===", flush=True)
    print(json.dumps(res, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
