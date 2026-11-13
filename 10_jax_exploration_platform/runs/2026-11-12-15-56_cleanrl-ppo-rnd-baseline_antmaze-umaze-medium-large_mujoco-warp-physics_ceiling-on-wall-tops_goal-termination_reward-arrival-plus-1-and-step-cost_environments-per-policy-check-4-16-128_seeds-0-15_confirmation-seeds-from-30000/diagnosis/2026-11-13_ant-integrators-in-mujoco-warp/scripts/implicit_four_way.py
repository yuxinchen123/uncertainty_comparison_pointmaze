"""Where MuJoCo Warp's full implicit step lands, against C MuJoCo's implicit and implicitfast steps.

From the 50 states of the one-step gate (a C-MuJoCo random-torque trajectory on the U-maze model of the setting),
one environment step in four steppers: C implicit, C implicitfast (both double precision), Warp implicit (forced
through `allow_full_implicit`) and Warp implicitfast. Per pair, the largest and the median of the largest
absolute position difference, over the contact-free states and over the states in contact. If Warp's implicit
step agreed with C's, the pair (Warp implicit, C implicit) would be as close as (Warp implicitfast, C
implicitfast), about 1e-6 m.

Run on a compute node's processors with the platform_jax_warp interpreter, JAX_PLATFORMS=cpu:
  python implicit_four_way.py --out records/implicit_four_way.json
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "2026-11-13_medium-and-large-readiness" / "scripts"))
from integrator_options import allow_full_implicit  # noqa: E402
import map_gates  # noqa: E402,F401  (puts the platform, the checks and the tests on the path)
from map_gates import checks, jax, jnp, mujoco, np  # noqa: E402
from exploration_platform.envs.antmaze.am_common import build_antmaze_xml, preset  # noqa: E402
from exploration_platform.envs.antmaze.warp_antmaze import WarpAntMaze  # noqa: E402
from test_antmaze_parity_cpu_mujoco import c_env_step, c_trajectory  # noqa: E402


def c_next(cfg, qs, vs, acts):
    """C MuJoCo, one environment step from each state on the model of `cfg`: next positions and contact flags."""
    model = mujoco.MjModel.from_xml_string(build_antmaze_xml(cfg))
    data = mujoco.MjData(model)
    out, contact = [], []
    for k in range(len(qs)):
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:] = qs[k], vs[k]
        mujoco.mj_forward(model, data)
        contact.append(c_env_step(model, data, acts[k], cfg.frame_skip) > 0)
        out.append(data.qpos.copy())
    return np.stack(out), np.asarray(contact)


def warp_next(cfg, qs, vs, acts):
    """Warp, one environment step from each state as one batch: next positions."""
    env = WarpAntMaze(cfg, len(qs), 1)
    state = checks.with_state(env, env.reset(), qs, vs)
    state, *_ = jax.jit(env.step)(state, jnp.asarray(acts, jnp.float32).reshape(len(qs), 1, 8))
    return np.asarray(state.data.qpos, np.float64), int(env.model.opt.integrator)


def main():
    """Four steppers, two time steps; every pair's distance on contact-free and in-contact states."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    allow_full_implicit()
    record = {}
    for time_step, frame_skip in ((0.005, 10), (0.01, 5)):
        base = replace(preset("umaze"), physics_time_step=time_step, frame_skip=frame_skip)
        implicit, fast = replace(base, integrator="implicit"), replace(base, integrator="implicitfast")
        # GOAL: the gate's 50 states, from a trajectory on the implicit model
        qs, vs, acts, _, _ = c_trajectory(WarpAntMaze(fast, 1, 1), 50, jax.random.PRNGKey(11), implicit)
        qs, vs, acts = qs[:50], vs[:50], acts[:50]
        steps = {"c_implicit": c_next(implicit, qs, vs, acts)[0], "c_implicitfast": c_next(fast, qs, vs, acts)[0]}
        contact = c_next(implicit, qs, vs, acts)[1]
        steps["warp_implicit"], number_implicit = warp_next(implicit, qs, vs, acts)
        steps["warp_implicitfast"], number_fast = warp_next(fast, qs, vs, acts)
        pairs = {}
        names = list(steps)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                err = np.abs(steps[a] - steps[b]).max(-1)
                pairs[f"{a} against {b}"] = {"contact_free": checks.spread(list(err[~contact])),
                                             "in_contact": checks.spread(list(err[contact]))}
        # GOAL: the implicit correction (implicit step minus implicitfast step) of Warp against C's, per contact-free
        # state: the cosine between the two correction vectors and the ratio of their lengths.
        # before: four [50, 15] position arrays; after: per state, cos(w, c) and |w| / |c| with
        #         w = warp_implicit - warp_implicitfast, c = c_implicit - c_implicitfast
        w = (steps["warp_implicit"] - steps["warp_implicitfast"])[~contact]
        c = (steps["c_implicit"] - steps["c_implicitfast"])[~contact]
        cosine = (w * c).sum(-1) / (np.linalg.norm(w, axis=-1) * np.linalg.norm(c, axis=-1))
        ratio = np.linalg.norm(w, axis=-1) / np.linalg.norm(c, axis=-1)
        correction = {"cosine_median": float(np.median(cosine)), "cosine_largest": float(cosine.max()),
                      "cosine_smallest": float(cosine.min()), "length_ratio_median": float(np.median(ratio))}
        key = f"time step {time_step} s x {frame_skip}"
        print(key, "implicit correction, Warp against C, contact-free states:", json.dumps(correction), flush=True)
        record[key] = {"warp_integrator_numbers": [number_implicit, number_fast], "pairs": pairs,
                       "implicit_correction_warp_against_c_contact_free": correction}
        for name, value in pairs.items():
            print(key, name, json.dumps(value), flush=True)
    arguments.out.write_text(json.dumps(record, indent=1))
    print("wrote", arguments.out)


if __name__ == "__main__":
    main()
