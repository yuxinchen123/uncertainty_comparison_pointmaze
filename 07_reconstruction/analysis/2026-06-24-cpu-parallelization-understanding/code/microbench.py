"""
Per-primitive CPU thread-scaling microbenchmark.

Isolates each numerical primitive used by 04_many_exploration_method.py, at the
*actual* tensor sizes the training loop uses, and measures, under a fixed thread
cap N:

  - throughput (work units per second)
  - effective_cores = process CPU-seconds / wall-seconds over the timed loop
    (how many cores the process kept busy / consumed — spin-wait counts as busy;
    1.0 == one core busy). Read it together with ops_per_sec to see wasted cores.

Run ONE thread cap per process so OpenBLAS / MKL pick up *_NUM_THREADS at import:

    OMP_NUM_THREADS=N MKL_NUM_THREADS=N OPENBLAS_NUM_THREADS=N \
        python microbench.py --n_threads N --seconds 2.5 --out results.json

Sizes (PointMaze_Large-v3, flat obs = [x, y, vx, vy] -> obs_dim=4, action_dim=2,
SAC batch_size=256, SAC net_arch=[256,256], RND output_dim=128,
EllipticalBonus feature_dim=128):

  sac_critic_mlp   : Q-net 6 -> 256 -> 256 -> 1, fwd+bwd, batch 256   (SAC's main torch work)
  rnd_mlp          : 4 -> 256 -> 128, fwd+bwd, batch 256              (RND predictor update)
  torch_inv_128    : torch.linalg.inv on 128x128 SPD                  (EllipticalBonus Lambda^-1)
  torch_phiT_phi   : (256,128)^T @ (256,128) -> 128x128               (EllipticalBonus cov build)
  np_matmul_small  : (256,128) @ (128,128)  numpy/OpenBLAS            (typical small numpy matmul)
  np_inv_128       : np.linalg.inv 128x128  numpy/OpenBLAS LAPACK
  np_runningstd    : RunningMeanStd.update on (256,4)                 (RND obs normalisation)
  python_loop_256  : pure-Python for-loop of 256 scalar dict updates  (VisitCount.compute shape)
  mujoco_step      : PointMaze_Large-v3 physics step                  (env rollout)

Contrast primitives (prove the libraries CAN scale when the problem is big, so a
flat result on the small sizes is a size effect, not a missing capability):

  torch_mlp_big    : 1024 -> 1024 -> 1024, fwd+bwd, batch 4096        (big torch matmul)
  np_matmul_big    : (2048,2048) @ (2048,2048) numpy/OpenBLAS
"""
import argparse
import json
import time

import numpy as np
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _proc_cpu_time():
    import psutil
    ct = psutil.Process().cpu_times()
    return ct.user + ct.system


def measure(fn, seconds, warmup):
    """Run fn() repeatedly; return (ops_per_sec, effective_cores, n_iters)."""
    t_end = time.perf_counter() + warmup
    while time.perf_counter() < t_end:
        fn()
    # timed window
    import psutil
    _ = psutil.Process().cpu_percent(None)
    c0 = _proc_cpu_time()
    w0 = time.perf_counter()
    n = 0
    t_stop = w0 + seconds
    while time.perf_counter() < t_stop:
        fn()
        n += 1
    w1 = time.perf_counter()
    c1 = _proc_cpu_time()
    wall = w1 - w0
    return {
        "ops_per_sec": round(n / wall, 2),
        "effective_cores": round((c1 - c0) / wall, 3),
        "n_iters": n,
        "wall_s": round(wall, 3),
    }


# ---- primitive builders: each returns a zero-arg callable doing one work unit ----

def build_sac_critic_mlp(device):
    net = torch.nn.Sequential(
        torch.nn.Linear(6, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 1),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.randn(256, 6, device=device)
    y = torch.randn(256, 1, device=device)

    def step():
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return step


def build_rnd_mlp(device):
    pred = torch.nn.Sequential(
        torch.nn.Linear(4, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128)
    ).to(device)
    tgt = torch.nn.Sequential(
        torch.nn.Linear(4, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128)
    ).to(device)
    for p in tgt.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(pred.parameters(), lr=1e-3)
    x = torch.randn(256, 4, device=device)

    def step():
        opt.zero_grad()
        with torch.no_grad():
            t = tgt(x)
        loss = ((pred(x) - t) ** 2).mean()
        loss.backward()
        opt.step()
    return step


def build_sac_actor_mlp(device):
    # SAC actor: 4 -> 256 -> 256, then mu(256->2) and log_std(256->2),
    # squashed-Gaussian sample + tanh-corrected log_prob, backward. batch 256.
    body = torch.nn.Sequential(
        torch.nn.Linear(4, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
    ).to(device)
    mu_head = torch.nn.Linear(256, 2).to(device)
    ls_head = torch.nn.Linear(256, 2).to(device)
    params = list(body.parameters()) + list(mu_head.parameters()) + list(ls_head.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    x = torch.randn(256, 4, device=device)

    def step():
        opt.zero_grad()
        h = body(x)
        mu = mu_head(h)
        log_std = ls_head(h).clamp(-5, 2)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        z = mu + std * eps
        a = torch.tanh(z)
        logp = (-0.5 * ((z - mu) / std) ** 2 - log_std).sum(1) - torch.log(1 - a.pow(2) + 1e-6).sum(1)
        loss = (-logp).mean()
        loss.backward()
        opt.step()
    return step


def build_torch_mahalanobis(device):
    # EllipticalBonus.compute hot matmul: (256,128) @ (128,128) + row reduction. no grad.
    phi = torch.randn(256, 128, device=device)
    cov_inv = torch.eye(128, device=device)

    def step():
        with torch.no_grad():
            tmp = phi @ cov_inv
            _ = (tmp * phi).sum(dim=1).clamp(min=1e-8).sqrt()
    return step


def build_adam_step(device):
    # Isolate the optimizer elementwise update over ~134k params (2 critic-sized nets).
    nets = [torch.nn.Sequential(torch.nn.Linear(6, 256), torch.nn.ReLU(),
                                torch.nn.Linear(256, 256), torch.nn.ReLU(),
                                torch.nn.Linear(256, 1)).to(device) for _ in range(2)]
    params = [p for n in nets for p in n.parameters()]
    opt = torch.optim.Adam(params, lr=1e-3)
    x = torch.randn(256, 6, device=device)
    loss = sum((n(x) ** 2).mean() for n in nets)
    loss.backward()  # populate .grad once; then time only the elementwise update

    def step():
        opt.step()  # foreach Adam update over all params, grads persist
    return step


def build_rnd_mlp_batch1(device):
    # Per-env-step intrinsic forward (logging path): batch 1, no grad.
    pred = torch.nn.Sequential(torch.nn.Linear(4, 256), torch.nn.ReLU(),
                               torch.nn.Linear(256, 128)).to(device)
    tgt = torch.nn.Sequential(torch.nn.Linear(4, 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, 128)).to(device)
    x = torch.randn(1, 4, device=device)

    def step():
        with torch.no_grad():
            _ = (pred(x) - tgt(x)).pow(2).sum()
    return step


def build_torch_inv_128(device):
    a = torch.randn(128, 128, device=device)
    spd = a @ a.t() + torch.eye(128, device=device) * 1e-3

    def step():
        torch.linalg.inv(spd)
    return step


def build_torch_phiT_phi(device):
    phi = torch.randn(256, 128, device=device)

    def step():
        _ = (phi.t() @ phi) / 256.0
    return step


def build_np_matmul_small():
    a = np.random.randn(256, 128).astype(np.float32)
    b = np.random.randn(128, 128).astype(np.float32)

    def step():
        _ = a @ b
    return step


def build_np_inv_128():
    a = np.random.randn(128, 128).astype(np.float64)
    spd = a @ a.T + np.eye(128) * 1e-3

    def step():
        _ = np.linalg.inv(spd)
    return step


def build_np_runningstd():
    from gymnasium.wrappers.utils import RunningMeanStd
    rms = RunningMeanStd(shape=(4,))
    x = np.random.randn(256, 4).astype(np.float32)

    def step():
        rms.update(x)
    return step


def build_python_loop_256():
    counts = {}
    data = [np.random.randn(4).astype(np.float32) for _ in range(256)]

    def step():
        out = np.zeros(256)
        for i in range(256):
            key = (int(data[i][0] * 4), int(data[i][1] * 4))
            counts[key] = counts.get(key, 0) + 1
            c = counts[key]
            out[i] = 1.0 if c <= 0 else min(1.0, c ** -0.5)
    return step


def build_torch_mlp_big(device):
    net = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
        torch.nn.Linear(1024, 1024), torch.nn.ReLU(),
        torch.nn.Linear(1024, 1024),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.randn(4096, 1024, device=device)
    y = torch.randn(4096, 1024, device=device)

    def step():
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return step


def build_mem_triad():
    # Pure memory-bandwidth op (STREAM-triad style): c = a + 2*b on large arrays.
    # numpy elementwise is single-threaded, so per-copy throughput dropping when
    # several copies run concurrently is direct evidence of memory-bandwidth (not
    # compute) contention. One iter moves ~3*8M*8 = 192 MB.
    n = 8_000_000
    a = np.random.rand(n)
    b = np.random.rand(n)
    c = np.empty(n)

    def step():
        np.multiply(b, 2.0, out=c)
        np.add(c, a, out=c)
    return step


def build_np_matmul_big():
    a = np.random.randn(2048, 2048).astype(np.float32)
    b = np.random.randn(2048, 2048).astype(np.float32)

    def step():
        _ = a @ b
    return step


def build_mujoco_step():
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env = gym.make("PointMaze_Large-v3", continuing_task=True, reset_target=False,
                   max_episode_steps=400)
    env.reset(seed=0)
    sp = env.action_space
    sp.seed(0)
    state = {"n": 0}

    def step():
        a = sp.sample()
        _, _, term, trunc, _ = env.step(a)
        state["n"] += 1
        if term or trunc or state["n"] % 400 == 0:
            env.reset()
    return step


PRIMS = {
    "sac_critic_mlp": ("torch (MKL/oneDNN)", lambda d: build_sac_critic_mlp(d)),
    "sac_actor_mlp": ("torch (MKL/oneDNN)", lambda d: build_sac_actor_mlp(d)),
    "rnd_mlp": ("torch (MKL/oneDNN)", lambda d: build_rnd_mlp(d)),
    "rnd_mlp_batch1": ("torch (MKL/oneDNN)", lambda d: build_rnd_mlp_batch1(d)),
    "torch_mahalanobis": ("torch (MKL)", lambda d: build_torch_mahalanobis(d)),
    "adam_step_134k": ("torch ATen foreach", lambda d: build_adam_step(d)),
    "torch_inv_128": ("torch linalg (MKL LAPACK)", lambda d: build_torch_inv_128(d)),
    "torch_phiT_phi": ("torch (MKL)", lambda d: build_torch_phiT_phi(d)),
    "np_matmul_small": ("numpy (OpenBLAS)", lambda d: build_np_matmul_small()),
    "np_inv_128": ("numpy linalg (OpenBLAS LAPACK)", lambda d: build_np_inv_128()),
    "np_runningstd": ("numpy", lambda d: build_np_runningstd()),
    "python_loop_256": ("pure Python", lambda d: build_python_loop_256()),
    "mujoco_step": ("MuJoCo (C)", lambda d: build_mujoco_step()),
    "mem_triad": ("numpy bandwidth (192MB/iter)", lambda d: build_mem_triad()),
    "torch_mlp_big": ("torch (MKL/oneDNN) BIG", lambda d: build_torch_mlp_big(d)),
    "np_matmul_big": ("numpy (OpenBLAS) BIG", lambda d: build_np_matmul_big()),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_threads", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--warmup", type=float, default=0.8)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--ops", type=str, default="all")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    torch.set_num_threads(args.n_threads)
    device = torch.device(args.device)

    ops = list(PRIMS) if args.ops == "all" else args.ops.split(",")
    results = []
    for name in ops:
        backend, builder = PRIMS[name]
        try:
            fn = builder(device)
            m = measure(fn, args.seconds, args.warmup)
            m.update({"op": name, "backend": backend, "n_threads": args.n_threads,
                      "device": args.device})
            results.append(m)
            print(f"[n={args.n_threads:2d}] {name:18s} {backend:28s} "
                  f"ops/s={m['ops_per_sec']:>10} eff_cores={m['effective_cores']:>6}",
                  flush=True)
        except Exception as e:
            print(f"[n={args.n_threads}] {name} FAILED: {e}", flush=True)
            results.append({"op": name, "n_threads": args.n_threads, "error": str(e)})

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
