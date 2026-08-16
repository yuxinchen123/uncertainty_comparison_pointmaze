"""Reachability probe: can the installed JAX run this node's graphics card at all?

The cluster spans compute capability 6.0 (Pascal, 2016) to 12.0 (Blackwell, 2025) and a JAX
build does not necessarily cover all of it. One short job per node class answers that before
ninety measurement jobs are submitted on the assumption that it does. The probe runs the real
trainer for one iteration at 8 copies, so a card that compiles but cannot execute the program
is caught here too.
"""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "ppo" / "jax_ppo"))

import jax                                           # noqa: E402
from jax_ppo_rnd import JaxPPORND, PPOConfig         # noqa: E402


def main():
    """Report the card this process was given and whether one trainer iteration runs on it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-class", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # what the runtime sees: a card, or nothing at all
    device = jax.local_devices()[0]
    record = {"node_class": args.node_class, "hostname": socket.gethostname(),
              "jax_version": jax.__version__, "platform": device.platform,
              "device_kind": device.device_kind,
              "compute_capability": getattr(device, "compute_capability", None),
              "device_memory_mb": (device.memory_stats() or {}).get("bytes_limit", 0) / 1e6}
    assert device.platform == "gpu", f"no graphics card visible on {record['hostname']}"

    # one real trainer iteration at the smallest useful size
    t0 = time.perf_counter()
    trainer = JaxPPORND(PPOConfig(n_copies=8, update_style="full_batch"))
    state = trainer.init_state()
    state, metrics = trainer._iterate(state, jax.random.PRNGKey(0), trainer.lr_argument(1, 100))
    jax.block_until_ready(state.params)
    record["one_iteration_including_compile_seconds"] = time.perf_counter() - t0
    record["loss"] = float(metrics["loss"])
    record["status"] = "trainer_runs"
    args.out.write_text(json.dumps(record, indent=1))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
