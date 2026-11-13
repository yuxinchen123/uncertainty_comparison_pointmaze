"""MuJoCo Warp's full implicit integrator through MJX, which MJX 3.11.0 refuses by itself.

MuJoCo Warp implements the integrator `implicit` (implicit in velocity WITH the velocity derivatives of
the Coriolis and centrifugal forces; `mujoco_warp/_src/forward.py`, `implicit`). MJX's wrapper refuses a
model with that integrator in `_put_option` (`mujoco/mjx/_src/io.py`), because MJX's own JAX
implementation has no such integrator, although the Warp step receives the integrator as a plain number.
`allow_full_implicit()` wraps that one function for the Warp implementation: the option is put as
`implicitfast` and the integrator number is then set to `implicit`. MuJoCo Warp's own `put_model` has
already seen the model with `implicit` at that point.

With `--events` this script plays the recorded onsets again in MuJoCo Warp on processors with the full
implicit integrator, and runs two of the environment's gates with it (one step against C MuJoCo of the
same model, resting height). Run on a compute node with the platform_jax_warp interpreter.
"""
import argparse
import json
import os
import socket
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def allow_full_implicit():
    """Make `mjx.put_model(..., impl="warp")` accept a model whose integrator is `implicit`."""
    import mujoco
    from mujoco.mjx._src import io, types

    if getattr(io._put_option, "allows_full_implicit", False):
        return
    put_option = io._put_option
    full, fast = int(mujoco.mjtIntegrator.mjINT_IMPLICIT), int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)

    def put_option_allowing_full_implicit(option, impl, impl_fields=None):
        """MJX's option of a C option; for the Warp implementation `implicit` is carried as its number."""
        if impl != types.Impl.WARP or int(option.integrator) != full:
            return put_option(option, impl, impl_fields)
        option.integrator = fast
        try:
            put = put_option(option, impl, impl_fields)
        finally:
            option.integrator = full
        return put.replace(integrator=full)

    put_option_allowing_full_implicit.allows_full_implicit = True
    io._put_option = put_option_allowing_full_implicit


def main():
    """Replay the onsets with the full implicit integrator in Warp and run two gates with it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--leads", default="2,4,8,24")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import replay_events
    from benchmarks.env import antmaze_warp_checks as checks
    from exploration_platform.envs.antmaze.am_common import preset
    from exploration_platform.envs.antmaze.warp_antmaze import WarpAntMaze

    allow_full_implicit()
    cfg = replace(preset("umaze"), continuing_task=False, integrator="implicit", physics="warp")
    events = np.load(args.events)
    record = {"host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID", "none"),
              "measured_iso": datetime.now().astimezone().isoformat(), "events_file": args.events.name,
              "warp_on_processors_full_implicit_4_and_8": {}}
    for kind in ("large", "moderate"):
        obs, action, warm = events[f"{kind}/obs"], events[f"{kind}/action"], events[f"{kind}/warm"]
        last = obs.shape[1] - 1
        record["warp_on_processors_full_implicit_4_and_8"][kind] = {}
        for lead in (int(text) for text in args.leads.split(",")):
            env = WarpAntMaze(cfg, 1, len(obs))
            assert int(env.model.opt.integrator) == 2, "the Warp model does not carry the full implicit integrator"
            speeds, rows = replay_events.warp_replay(env, obs[:, last - lead:], action[:, last - lead:], warm[:, last - lead:])
            peaks = speeds.max(1)
            row = {"onsets_replayed": len(obs), "reached_100_metres_per_second": int((peaks > 100.0).sum()),
                   "reached_30_metres_per_second": int((peaks > 30.0).sum()),
                   "median_largest_speed": float(np.median(peaks)),
                   "largest_speeds_sorted_top_5": [float(min(p, 1e30)) for p in np.sort(peaks)[-5:]],
                   "largest_constraint_rows": int(rows.max())}
            record["warp_on_processors_full_implicit_4_and_8"][kind][str(lead)] = row
            print("Warp, full implicit,", kind, "onsets, lead", lead, json.dumps(row), flush=True)

    # GOAL: two gates of spec.md with the full implicit integrator (C MuJoCo runs the same model).
    record["one_step_parity"] = checks.one_step_parity(changes={"integrator": "implicit"}, with_jax=False)
    print("one-step parity:", json.dumps(record["one_step_parity"], default=str), flush=True)
    record["resting_height"] = checks.resting_height(changes={"integrator": "implicit"}, with_jax=False)
    print("resting height:", json.dumps(record["resting_height"], default=str), flush=True)
    args.out.write_text(json.dumps(record, indent=1, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
