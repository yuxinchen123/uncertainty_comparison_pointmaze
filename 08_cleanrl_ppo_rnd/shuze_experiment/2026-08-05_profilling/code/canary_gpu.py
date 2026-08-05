"""Check that this environment can actually run on the GPU of the node it lands on.

The point is Pascal and Maxwell. `torch 2.10.0+cu128` (which sits in a user-site directory and
shadows the shared env unless PYTHONNOUSERSITE is set) compiles for sm_70 and up with no PTX
fallback, so it dies on every GTX 1080, 1080 Ti, Titan Xp, Titan X and P100 — that is the whole
gnolim partition. `torch 2.6.0+cu124`, which the cleanrl_rnd env uses, claims sm_50 upward. This
script tests the claim rather than trusting it: it runs the real convolution shapes of the RND
towers, forward and backward, and reports.
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn


def main():
    """Report the GPU, the torch build's architectures, and whether real kernels run."""
    node = os.environ.get("SLURMD_NODENAME", os.uname().nodename)
    out = {
        "node": node,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "torch_version": torch.__version__,
        "torch_arch_flags": torch._C._cuda_getArchFlags(),
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        out["verdict"] = "NO CUDA VISIBLE"
        print(json.dumps(out, indent=1), flush=True)
        return 1

    props = torch.cuda.get_device_properties(0)
    out.update({
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "gpu_memory_total_mb": props.total_memory / 1024 / 1024,
        "bf16_supported_including_emulation": torch.cuda.is_bf16_supported(),
        "bf16_supported_native": props.major >= 8,
        "triton_usable_for_torch_compile": props.major >= 7,
    })

    # The actual test: build the RND predictor's convolution stack on the 1x84x84 input it really
    # uses, run a forward and a backward, and read a number back to the host. A card the build has
    # no kernels for fails here with "no kernel image is available for execution on the device".
    net = nn.Sequential(
        nn.Conv2d(1, 32, 8, stride=4), nn.LeakyReLU(),
        nn.Conv2d(32, 64, 4, stride=2), nn.LeakyReLU(),
        nn.Conv2d(64, 64, 3, stride=1), nn.LeakyReLU(),
        nn.Flatten(), nn.Linear(7 * 7 * 64, 512),
    ).cuda()
    x = torch.randn(128, 1, 84, 84, device="cuda")
    started = time.time()
    y = net(x)
    loss = y.pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    out["forward_backward_seconds"] = time.time() - started
    out["loss_value"] = float(loss.item())
    out["verdict"] = "OK"
    print(json.dumps(out, indent=1), flush=True)

    result_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, f"canary_{node}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    os.chmod(path, 0o660)
    return 0


if __name__ == "__main__":
    sys.exit(main())
