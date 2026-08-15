"""Replay the MuJoCo reference fixtures through a GPU-env implementation and report errors.

Usage:
  python check_against_fixtures.py --impl torch [--device cpu] [--dtype float64]

Two comparisons per fixture case:
  1. one-step (teacher-forced): every reference state (q_t, v_t) is fed in and ONE predicted
     step is compared to (q_{t+1}, v_{t+1}). This measures per-step model error with no
     chaotic compounding — the primary dynamics-fidelity metric.
  2. rollout: the action sequence is replayed closed-loop from (q0, v0).

Pass criteria per case category (physics_spec.md validation contract; the contact model is
probe-verified exact, so the gates are tight):
  every case     one-step position error <= 1e-9 (float64) / 1e-3 (float32), and
                 penetration <= 0.06 m (the one-max-speed-step physical bound; the reference
                 itself penetrates transiently up to ~8 mm on impacts)
  all except corner_graze: closed-loop rollout position error <= 1e-6 (float64) /
                 5e-2 (float32 — float32 rounding is amplified through contact events)
  corner_graze   rollout REPORTED only: its reference passes through a knife-edge
                 equilibrium (ball balanced on a corner against a diagonal push), where
                 closed-loop paths separate on rounding noise; one-step gate still applies.
Exit code 1 if any hard criterion fails.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

COMMON = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMMON))
from pm_common import MAPS, R, EnvConfig  # noqa: E402

FIXTURES = COMMON / "fixtures" / "pointmaze_large_fixtures.npz"
SETTLE_CASES = {"wall_head_on_py", "wall_head_on_mx", "wall_slide", "impact_fast"}


def wall_penetration(qs: np.ndarray) -> float:
    """Worst wall penetration over a trajectory in meters (0 = never penetrated).

    before: qs [T, 2] world positions; after: max over steps of (R - distance to the
    nearest wall-cell rectangle), floored at 0.
    """
    wall = np.array(MAPS["large"], dtype=bool)
    rows, cols = wall.shape
    wi, wj = np.nonzero(wall)
    xl = wj - cols / 2.0
    yb = rows / 2.0 - (wi + 1)
    dx = np.maximum(np.maximum(xl[None] - qs[:, :1], qs[:, :1] - (xl[None] + 1)), 0)
    dy = np.maximum(np.maximum(yb[None] - qs[:, 1:2], qs[:, 1:2] - (yb[None] + 1)), 0)
    dist = np.sqrt(dx * dx + dy * dy).min(axis=1)
    return float(np.maximum(R - dist, 0).max())


def make_torch_step(device: str, dtype: str):
    """Return step_batch(pos[B,2], vel[B,2], act[B,2]) -> (pos', vel') numpy, via the torch env."""
    import torch
    sys.path.insert(0, str(COMMON.parent / "torch_env"))
    from torch_pointmaze import TorchPointMaze
    dt = torch.float64 if dtype == "float64" else torch.float32
    env = TorchPointMaze(EnvConfig(), 1, 1, device=device, dtype=dt)

    def step_batch(pos, vel, act):
        p = torch.tensor(pos, dtype=dt, device=device).unsqueeze(0)
        v = torch.tensor(vel, dtype=dt, device=device).unsqueeze(0)
        a = torch.tensor(act, dtype=dt, device=device).unsqueeze(0)
        p2, v2 = env.dynamics_step(p, v, a)
        return p2.squeeze(0).cpu().numpy(), v2.squeeze(0).cpu().numpy()

    return step_batch


def make_jax_step(device: str, dtype: str):
    """Return the jax implementation's step_batch (same contract as make_torch_step)."""
    sys.path.insert(0, str(COMMON.parent / "jax_env"))
    from jax_pointmaze import make_step_batch
    return make_step_batch(dtype)


def make_cuda_step(device: str, dtype: str):
    """Return the fused-CUDA implementation's step_batch (same contract as make_torch_step)."""
    sys.path.insert(0, str(COMMON.parent / "cuda_env"))
    from cuda_pointmaze import make_step_batch
    return make_step_batch(dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, choices=["torch", "jax", "cuda"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    args = ap.parse_args()

    step_batch = {"torch": make_torch_step, "jax": make_jax_step, "cuda": make_cuda_step}[
        args.impl](args.device, args.dtype)
    data = np.load(FIXTURES)
    names = sorted({k.split("__")[0] for k in data.files})
    free_tol = 1e-9 if args.dtype == "float64" else 1e-4

    failed = []
    for name in names:
        q0, v0 = data[f"{name}__q0"], data[f"{name}__v0"]
        actions = data[f"{name}__actions"]
        ref_q, ref_v = data[f"{name}__qpos"], data[f"{name}__qvel"]

        # one-step teacher-forced: inputs are (q0,v0) then every reference post-step state
        in_q = np.vstack([q0[None], ref_q[:-1]])
        in_v = np.vstack([v0[None], ref_v[:-1]])
        os_q, os_v = step_batch(in_q, in_v, actions)
        os_qerr = np.abs(os_q - ref_q).max()
        os_verr = np.abs(os_v - ref_v).max()

        # closed-loop rollout
        qs, vs = [], []
        p, v = q0[None].copy(), v0[None].copy()
        for a in actions:
            p, v = step_batch(p, v, a[None])
            qs.append(p[0].copy())
            vs.append(v[0].copy())
        qs, vs = np.array(qs), np.array(vs)
        ro_qerr = np.abs(qs - ref_q).max()
        settle = np.abs(qs[-1] - ref_q[-1]).max()
        pen = wall_penetration(qs)
        diverge = np.abs(qs - ref_q).max(axis=1) > 0.05
        horizon = int(np.argmax(diverge)) if diverge.any() else len(qs)

        one_step_tol = 1e-9 if args.dtype == "float64" else 1e-3
        rollout_tol = 1e-6 if args.dtype == "float64" else 5e-2
        ok = os_qerr <= one_step_tol and pen <= 0.06
        if name != "corner_graze":
            ok = ok and ro_qerr <= rollout_tol
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed.append(name)
        print(f"{status} {name:20s} one-step|q|={os_qerr:.2e} |v|={os_verr:.2e}  "
              f"rollout|q|={ro_qerr:.2e} settle={settle:.2e} pen={pen:.2e} "
              f"diverge_horizon={horizon}/{len(qs)}")

    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
