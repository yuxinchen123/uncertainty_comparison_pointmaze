"""Tables for the notes from the motion check's records: the Warp rollouts and the C MuJoCo closed-loop runs.

For each checkpoint (16 and 128 environments per policy) and each physics (Warp as trained; C MuJoCo with
the four settings of `c_mujoco_closed_loop.py`) it prints, per configuration:
  1. the arrival rate, the share of steps above 1.0 and 1.5 m, and how fast the ants are;
  2. how much of the arrival rate is left when episodes that leave a bound are not counted as arrivals
     (largest torso height above 1.0 or 1.5 m, largest torso speed above 5 or 10 m/s);
and over the 128 policies the rank correlation between a policy's arrival rate and how much it flies.

Run through Slurm (numpy): <platform_jax_warp python> summarize.py --records <records folder>
"""
import argparse
import json
from pathlib import Path


def rank_correlation(a, b):
    """Spearman's rank correlation of two equally long series (ties get their mean rank).

    before: a [1, 2, 3], b [10, 30, 20]   after: 0.5
    """
    import numpy as np

    def ranks(values):
        """Mean ranks of a series: [10, 30, 20, 30] -> [0, 2.5, 1, 2.5]."""
        values = np.asarray(values, float)
        order = np.argsort(values, kind="stable")
        out = np.empty(len(values))
        out[order] = np.arange(len(values))
        for value in np.unique(values):
            same = values == value
            out[same] = out[same].mean()
        return out
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def describe(name, table, columns, per_cell, configurations):
    """Print the tables of one physics from its episode table and return them as a dictionary."""
    import numpy as np
    col = {key: index for index, key in enumerate(columns)}
    out = {"per_configuration": [], "rank_correlation_over_policies": {}}
    print(f"\n=== {name}: {len(table)} episodes")
    print("configuration | episodes | arriving | arriving, counted only if the largest height stayed under 1.5 m | under 1.0 m | "
          "largest speed under 10 m/s | under 5 m/s | arriving episodes that passed 1.0 m | 1.5 m | 5 m/s | share of an arriving "
          "episode's steps above 1.0 m | mean length of an arriving episode")
    for cell, configuration in enumerate(configurations):
        rows = table[(table[:, col["policy"]] // per_cell) == cell]
        if not len(rows):
            continue
        arrived = rows[:, col["arrived"]] > 0
        height, speed = rows[:, col["largest_height"]], rows[:, col["largest_speed"]]
        rate = lambda keep: float((arrived & keep).sum() / len(rows))
        among = lambda mask: float(mask[arrived].mean()) if arrived.any() else float("nan")
        entry = {"configuration": configuration, "episodes": int(len(rows)), "arrival_rate": float(arrived.mean()),
                 "arrival_rate_height_under_1_5": rate(height <= 1.5), "arrival_rate_height_under_1_0": rate(height <= 1.0),
                 "arrival_rate_speed_under_10": rate(speed <= 10.0), "arrival_rate_speed_under_5": rate(speed <= 5.0),
                 "arriving_episodes_that_passed_1_0_m": among(height > 1.0), "arriving_episodes_that_passed_1_5_m": among(height > 1.5),
                 "arriving_episodes_that_passed_5_m_per_s": among(speed > 5.0),
                 "share_of_arriving_steps_above_1_0_m": float(rows[arrived, col["steps_above_1_0_m"]].sum() / max(rows[arrived, col["length"]].sum(), 1)),
                 "mean_length_of_arriving_episodes": float(rows[arrived, col["length"]].mean()) if arrived.any() else float("nan"),
                 "episodes_that_never_passed_1_0_m": float((height <= 1.0).mean()),
                 "largest_speed_quantiles_50_90_99": np.quantile(speed, [0.5, 0.9, 0.99]).round(1).tolist(),
                 "largest_height_quantiles_50_90_99": np.quantile(height, [0.5, 0.9, 0.99]).round(2).tolist()}
        out["per_configuration"].append(entry)
        print(f"lr {configuration[0]:g}, weight {configuration[2]:g} | {entry['episodes']} | {entry['arrival_rate']:.3f} | "
              f"{entry['arrival_rate_height_under_1_5']:.3f} | {entry['arrival_rate_height_under_1_0']:.4f} | "
              f"{entry['arrival_rate_speed_under_10']:.3f} | {entry['arrival_rate_speed_under_5']:.4f} | "
              f"{entry['arriving_episodes_that_passed_1_0_m']:.3f} | {entry['arriving_episodes_that_passed_1_5_m']:.3f} | "
              f"{entry['arriving_episodes_that_passed_5_m_per_s']:.3f} | {entry['share_of_arriving_steps_above_1_0_m']:.3f} | "
              f"{entry['mean_length_of_arriving_episodes']:.0f} | never above 1.0 m: {entry['episodes_that_never_passed_1_0_m']:.4f} | "
              f"largest speed per episode, quantiles {entry['largest_speed_quantiles_50_90_99']} | largest height {entry['largest_height_quantiles_50_90_99']}")
    # GOAL: over policies, does the arrival rate go with flying?
    policies = np.unique(table[:, col["policy"]]).astype(int)
    per_policy = {"arrival_rate": [], "share_above_1_0": [], "share_above_1_5": [], "median_largest_speed": []}
    for policy in policies:
        rows = table[table[:, col["policy"]] == policy]
        steps = rows[:, col["length"]].sum()
        per_policy["arrival_rate"].append(float((rows[:, col["arrived"]] > 0).mean()))
        per_policy["share_above_1_0"].append(float(rows[:, col["steps_above_1_0_m"]].sum() / steps))
        per_policy["share_above_1_5"].append(float(rows[:, col["steps_above_1_5_m"]].sum() / steps))
        per_policy["median_largest_speed"].append(float(np.median(rows[:, col["largest_speed"]])))
    for key in ("share_above_1_0", "share_above_1_5", "median_largest_speed"):
        out["rank_correlation_over_policies"][key] = rank_correlation(per_policy["arrival_rate"], per_policy[key])
    print(f"rank correlation over {len(policies)} policies between the arrival rate and: share of steps above 1.0 m "
          f"{out['rank_correlation_over_policies']['share_above_1_0']:+.2f}; above 1.5 m "
          f"{out['rank_correlation_over_policies']['share_above_1_5']:+.2f}; median largest speed of an episode "
          f"{out['rank_correlation_over_policies']['median_largest_speed']:+.2f}")
    return out


def main():
    """Read every record of the folder and print the tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    result = {}
    # GOAL: the ant's mass and inertia from the environment's own model, for the energy scale of the notes.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "src"))
    import mujoco
    from exploration_platform.envs.antmaze.am_common import build_antmaze_xml, preset
    model = mujoco.MjModel.from_xml_string(build_antmaze_xml(preset("umaze")))
    mass = float(model.body_mass.sum())
    result["ant"] = {"mass_kg": mass, "torso_mass_kg": float(model.body_mass[1]), "gear": float(model.actuator_gear[0, 0]),
                     "joint_armature": float(model.dof_armature[6]), "joint_damping": float(model.dof_damping[6]),
                     "kinetic_energy_of_the_whole_ant_at_5_m_per_s_joules": 0.5 * mass * 25.0,
                     "potential_energy_at_1_m_above_rest_joules": mass * 9.81 * 1.0}
    print("the ant:", json.dumps(result["ant"]))
    for count in (16, 128):
        stem = args.records / f"warp_rollout_{count}-environments-per-policy"
        if not Path(str(stem) + "_summary.json").exists():
            continue
        summary = json.loads(Path(str(stem) + "_summary.json").read_text())
        per_cell, configurations = summary["policies_per_configuration"], summary["configurations"]
        table = np.load(str(stem) + "_episodes.npz")["table"]
        result[f"warp_{count}"] = describe(f"{count} environments per policy, Warp as trained (guard on), checkpoint iteration "
                                           f"{summary['checkpoint_iteration']}", table, summary["episode_table_columns"],
                                           per_cell, configurations)
        closed = args.records / f"c_mujoco_closed_loop_{count}-environments-per-policy"
        if Path(str(closed) + ".json").exists():
            record = json.loads(Path(str(closed) + ".json").read_text())
            tables = np.load(str(closed) + "_episodes.npz")
            for setting in record["settings"]:
                result[f"c_{count}_{setting}"] = describe(f"{count} environments per policy, C MuJoCo, {setting}", tables[setting],
                                                          record["columns"], per_cell, configurations)
    (args.records / "summary_tables.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
