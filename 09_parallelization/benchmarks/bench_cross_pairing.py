"""Measure the cross-framework env/trainer pairing cost (Module 3 pairing grid).

The native pairings (torch env + torch PPO, jax env + jax PPO) fuse inside one runtime.
A cross pairing (jax env + torch trainer, or torch env + jax trainer) must cross a
framework boundary EVERY environment step via dlpack. This script measures that boundary:
per-step cost of torch->dlpack->jax step->dlpack->torch at trainer-relevant batch sizes,
against the native compiled step — quantifying why the cross pairings are not production
paths (they also structurally break CUDA-graph capture and XLA scan fusion).

Run under the lock with the JAX env python (it has jax; torch tensors come over dlpack
from a torch subprocess? No — this script uses jax + torch in ONE env, so it must run with
a python that has BOTH. It creates a dedicated venv check first; see main().)
"""
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"


def main():
    # both frameworks in one process: prefer the torch venv if jax is importable there,
    # else instruct which env to build (torch+jax-cpu is enough for the boundary cost? No —
    # the boundary cost must be measured on GPU; jax[cuda] + torch cu13 share the GPU fine
    # for this measurement as long as only one allocates heavily.)
    try:
        import jax
        import torch
    except ImportError as e:
        print(f"needs a python with BOTH torch and jax on GPU: {e}")
        sys.exit(2)

    import jax.numpy as jnp
    import jax.dlpack as jdl
    sys.path.insert(0, str(BASE / "pointmaze" / "common"))
    sys.path.insert(0, str(BASE / "pointmaze" / "jax_env"))
    sys.path.insert(0, str(BASE / "pointmaze" / "torch_env"))
    from pm_common import EnvConfig
    from jax_pointmaze import JaxPointMaze
    from torch_pointmaze import TorchPointMaze

    rows = []
    for total in (512, 65536, 1048576):
        jenv = JaxPointMaze(EnvConfig(), 1, total)
        jstate = jenv.reset()
        jstep = jax.jit(jenv.step, donate_argnums=0)
        tenv = TorchPointMaze(EnvConfig(), 1, total, device="cuda")
        tenv.reset()
        tstep = torch.compile(tenv.step, fullgraph=True, dynamic=False)
        act_t = torch.rand(1, total, 2, device="cuda") * 2 - 1

        # native jax step (actions already jax)
        act_j = jnp.asarray(act_t.cpu().numpy())
        jstate, *_ = jstep(jstate, act_j)          # warm
        jax.block_until_ready(jstate)
        k = 200
        t0 = time.perf_counter()
        for _ in range(k):
            jstate, *out = jstep(jstate, act_j)
        jax.block_until_ready(jstate)
        native_jax = (time.perf_counter() - t0) / k * 1e6

        # cross: torch-side action -> dlpack -> jax step -> dlpack obs back to torch
        def cross():
            # modern dlpack protocol: pass the arrays themselves across the boundary
            aj = jdl.from_dlpack(act_t)
            s, obs, r, te, tr, fo = jstep_nodonate(cross.state, aj)
            cross.state = s
            return torch.utils.dlpack.from_dlpack(obs)
        jstep_nodonate = jax.jit(jenv.step)        # donation invalid when buffers cross
        cross.state = jenv.reset()
        cross(); jax.block_until_ready(cross.state)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(k):
            o = cross()
        torch.cuda.synchronize(); jax.block_until_ready(cross.state)
        cross_us = (time.perf_counter() - t0) / k * 1e6

        # native torch step
        for _ in range(10):
            tstep(act_t)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(k):
            tstep(act_t)
        torch.cuda.synchronize()
        native_torch = (time.perf_counter() - t0) / k * 1e6

        rows.append({"total_envs": total, "native_torch_us": native_torch,
                     "native_jax_us": native_jax, "cross_framework_us": cross_us,
                     "boundary_overhead_us": cross_us - native_jax})
        print(f"N={total:>8d}: native torch {native_torch:8.1f} us/step | "
              f"native jax {native_jax:8.1f} | torch<->jax dlpack boundary {cross_us:8.1f} "
              f"(overhead {cross_us - native_jax:+8.1f})")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_cross_pairing.json"
    out.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
