"""The U-maze environment gates in MuJoCo Warp for one physics setting, against C MuJoCo with the same setting.

The gates are those of `diagnosis/2026-11-13_medium-and-large-readiness/scripts/map_gates.py` (reset, goal,
step limit, guard, walls, ceiling), imported from it, plus the agreement with C MuJoCo, here with the setting's
integrator on both sides:

  parity      one environment step from 50 states of a C-MuJoCo random-torque trajectory (bounds 1e-4 m
              contact-free, 5e-2 m in contact); 20 shared random-torque steps from the spawn (1e-4 m before the
              first contact, under one 4 m cell over all); the resting height after 1,200 passive steps
              (0.38248 m within 1e-4 m in both) and a C-settled state kept by Warp (moved under 1e-3 m).
              C MuJoCo runs the model of the same configuration (same integrator, time step, physics steps,
              4 solver and 8 line-search iterations), in double precision. MJX's JAX implementation is left out:
              it has no full implicit integrator.

The configuration is this folder's file for the setting (`configurations/`, written by
write_configurations.py). Run on a compute node's processors with the platform_jax_warp interpreter and
JAX_PLATFORMS=cpu:
  python integrator_gates.py --setting rk4_time-step-0.005_frame-skip-10 --out records/gates_<setting>.json
"""
import argparse
import json
import sys
import time
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLATFORM = Path("/p/rlprojects/RND/10_jax_exploration_platform")
READINESS = HERE.parents[1] / "2026-11-13_medium-and-large-readiness" / "scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(READINESS))
from integrator_options import SETTINGS, prepare  # noqa: E402
import map_gates  # noqa: E402  (imports the platform, the speed study's checks and the guard test's ants)
from map_gates import checks, jax, jnp, np  # noqa: E402
from experiment_configuration import read_experiment_configuration  # noqa: E402
from exploration_platform.envs.antmaze.am_common import AntMazeConfig  # noqa: E402
from exploration_platform.envs.antmaze.warp_antmaze import WarpAntMaze  # noqa: E402


def configuration(slug):
    """The setting's environment as this folder's configuration file gives it, and the file's path."""
    path = HERE.parent / "configurations" / f"antmaze_umaze_{slug}_guard-30-and-100.toml"
    cfg = AntMazeConfig(**read_experiment_configuration(path)["environment"])
    integrator, time_step, frame_skip = SETTINGS[slug]
    if (cfg.integrator, cfg.physics_time_step, cfg.frame_skip) != (integrator, time_step, frame_skip):
        raise ValueError(f"{path.name} gives {cfg.integrator}, {cfg.physics_time_step}, {cfg.frame_skip}")
    return cfg, path


def trajectory_divergence_warp(cfg, steps=20):
    """Shared random torques from the spawn in C MuJoCo and in Warp on the model of `cfg`: the largest position
    difference before the first contact, over all steps and at the last step (the speed study's gate 2,
    without the JAX implementation)."""
    from test_antmaze_parity_cpu_mujoco import c_trajectory
    env = WarpAntMaze(cfg, 1, 1)
    qs, _, acts, ncons, _ = c_trajectory(env, steps, jax.random.PRNGKey(5), cfg)
    first_contact = int(np.argmax(ncons > 0)) if (ncons > 0).any() else len(ncons)
    # GOAL: Warp steps the same torques; the difference to C's state after each step
    state, step, errors = env.reset(), jax.jit(env.step), []
    for k in range(steps):
        state, *_ = step(state, jnp.asarray(acts[k], jnp.float32).reshape(1, 1, 8))
        errors.append(float(np.abs(np.asarray(state.data.qpos[0], np.float64) - qs[k + 1]).max()))
    return {"steps": steps, "first_state_with_a_contact": first_contact,
            "warp": {"largest_before_the_first_contact": max(errors[:max(first_contact - 1, 1)]),
                     "largest_over_all_steps": max(errors), "at_the_last_step": errors[-1],
                     "all_finite": bool(np.isfinite(errors).all()), "per_step": errors}}


def gate_parity(cfg):
    """One step, 20 steps and the resting height against C MuJoCo with the same physics setting."""
    parity = checks.one_step_parity(cfg=replace(cfg, physics="jax"), with_jax=False)
    divergence = trajectory_divergence_warp(cfg)
    resting = checks.resting_height(changes={"integrator": cfg.integrator, "physics_time_step": cfg.physics_time_step,
                                             "frame_skip": cfg.frame_skip, "solver_iterations": cfg.solver_iterations,
                                             "ls_iterations": cfg.ls_iterations}, with_jax=False)
    free, contact = parity["contact_free"]["warp_against_c"], parity["in_contact"]["warp_against_c"]
    warp = divergence["warp"]
    result = {"one_step": parity, "trajectory": divergence, "resting": resting}
    result["passed"] = bool(
        free["count"] > 0 and contact["count"] > 0 and free["largest"] < 1e-4 and contact["largest"] < 5e-2
        and not parity["pool"]["dropped"] and parity["non_finite_resets"] == 0
        and warp["all_finite"] and warp["largest_before_the_first_contact"] < 1e-4 and warp["largest_over_all_steps"] < 4.0
        and abs(resting["warp"]["height"] - checks.REFERENCE_RESTING_HEIGHT) < 1e-4
        and abs(resting["c_mujoco"]["height"] - checks.REFERENCE_RESTING_HEIGHT) < 1e-4
        and resting["warp"]["c_settled_state_moved_by"] < 1e-3)
    return result


GATES = {"reset": map_gates.gate_reset, "goal": map_gates.gate_goal, "step_limit": map_gates.gate_step_limit,
         "guard": map_gates.gate_guard, "walls": map_gates.gate_walls, "ceiling": map_gates.gate_ceiling,
         "parity": gate_parity}


def main():
    """Run the gates for one setting; write the record after every gate, so a stopped job keeps the finished ones."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True, choices=sorted(SETTINGS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gates", default=",".join(GATES))
    arguments = parser.parse_args()
    prepare(arguments.setting)
    cfg, path = configuration(arguments.setting)
    record = {"setting": arguments.setting, "configuration_file": str(path),
              "configuration": {f.name: getattr(cfg, f.name) for f in fields(cfg)},
              "warp_model_integrator_number": int(WarpAntMaze(cfg, 1, 1).model.opt.integrator),
              "started_at": datetime.now().astimezone().isoformat(timespec="seconds"), "gates": {}}
    dump = lambda: arguments.out.write_text(json.dumps(record, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    for name in arguments.gates.split(","):
        start = time.perf_counter()
        result = GATES[name](cfg)
        result["seconds"] = round(time.perf_counter() - start, 1)
        record["gates"][name] = result
        print(f"{arguments.setting} {name}: {'passed' if result['passed'] else 'FAILED'} ({result['seconds']} s)", flush=True)
        dump()
    record["all_passed"] = all(r["passed"] for r in record["gates"].values())
    record["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    dump()
    print(f"{arguments.setting}: all gates passed: {record['all_passed']}", flush=True)


if __name__ == "__main__":
    main()
