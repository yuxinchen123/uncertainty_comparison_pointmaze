"""Host-compiled check of the kernel math: run the fixture one-step comparison through
cpu_port.so (the same pointmaze_dynamics.h the CUDA kernel uses, compiled by g++).
No GPU needed — fast regression test for the shared header.

Build first:  g++ -O2 -shared -fPIC -o cpu_port.so cpu_port.cpp
"""
import ctypes
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "common"))
from pm_common import build_geometry  # noqa: E402

LIB = ctypes.CDLL(str(Path(__file__).resolve().parent / "cpu_port.so"))
LIB.dynamics_batch_f64.argtypes = [ctypes.POINTER(ctypes.c_double)] * 5 + [
    ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int]


def step_batch(pos, vel, act, nb_mask, rows, cols):
    """One dynamics step for a [B, 2] batch through the host-compiled kernel math."""
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    vel = np.ascontiguousarray(vel, dtype=np.float64)
    act = np.ascontiguousarray(act, dtype=np.float64)
    pos_out = np.empty_like(pos)
    vel_out = np.empty_like(vel)
    p = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    LIB.dynamics_batch_f64(p(pos), p(vel), p(act), p(pos_out), p(vel_out),
                           nb_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                           len(pos), rows, cols)
    return pos_out, vel_out


def main():
    geo = build_geometry("large")
    nb_mask = geo["nb_mask"].reshape(-1).astype(np.int32)
    data = np.load(BASE / "common" / "fixtures" / "pointmaze_large_fixtures.npz")
    names = sorted({k.split("__")[0] for k in data.files})
    worst = 0.0
    for name in names:
        q0, v0 = data[f"{name}__q0"], data[f"{name}__v0"]
        actions, ref_q, ref_v = data[f"{name}__actions"], data[f"{name}__qpos"], data[f"{name}__qvel"]
        in_q = np.vstack([q0[None], ref_q[:-1]])
        in_v = np.vstack([v0[None], ref_v[:-1]])
        os_q, _ = step_batch(in_q, in_v, actions, nb_mask, geo["rows"], geo["cols"])
        err = np.abs(os_q - ref_q).max()
        worst = max(worst, err)
        assert err <= 1e-9, f"{name}: one-step error {err:.3e}"
    print(f"CPU port one-step check: ALL PASS (worst {worst:.3e})")


if __name__ == "__main__":
    main()
