"""Print the tables of notes.md from this folder's records (standard library only, so it runs anywhere).

  python summarize.py            the throughput table of every profile record, and the gate numbers
"""
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CARD_NAMES = {"NVIDIA A100-PCIE-40GB": "A100 40 GB", "NVIDIA A40": "A40"}
MAP_NAMES = {"umaze": "U-maze", "medium": "Medium", "large": "Large"}


def profile_rows():
    """Every measured shape, sorted by card, environments per policy, map and copies."""
    rows = []
    for path in sorted(glob.glob(str(HERE / "records/profile_*.jsonl"))):
        for line in open(path):
            record = json.loads(line)
            if not record.get("failed"):
                rows.append(record)
    order = {"umaze": 0, "medium": 1, "large": 2}
    return sorted(rows, key=lambda r: (r["device_name"], r["environments_per_copy"], order[r["map"]], r["copies"]))


def throughput_table(rows):
    """The five throughput columns, then the card memory and the Warp pool use, one row per shape."""
    # before: one record {"device_name": "NVIDIA A40", "map": "large", "copies": 128, "seconds_per_iteration": 1.11, ...}
    # after:  "| A40 | Large | 16 | 128 | 1.110 | 0.236 | 1.85 | 0.150 | 2,607 | 4.0 | 35 |"
    lines = ["| card | map | environments per policy | copies | seconds per iteration | total steps per second "
             "(millions) | steps per second per copy (thousands) | hours per million steps per copy | card memory "
             "(MiB) | candidate pairs per environment, most | constraint rows of one environment, most |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {CARD_NAMES.get(r['device_name'], r['device_name'])} | {MAP_NAMES[r['map']]} | {r['environments_per_copy']} "
            f"| {r['copies']} | {r['seconds_per_iteration']:.3f} | {r['total_steps_per_second_millions']:.3f} "
            f"| {r['steps_per_second_per_copy_thousands']:.2f} | {r['hours_per_million_steps_per_copy']:.3f} "
            f"| {r['card_memory_used_mb_largest_reading']:,} | {r['candidate_pairs_per_environment_largest']:.1f} "
            f"| {r['pool']['nefc_peak']} |")
    return "\n".join(lines)


def gate_lines():
    """The gate verdicts and their main numbers, per map."""
    out = []
    for name in ("medium", "large"):
        # the three runs of the gates: file/reset/goal, then the physics gates with the guard on, then the
        # ceiling with the guard off (the run with the guard on is kept under its own name)
        gates = {}
        for path in (HERE / f"records/gates_{name}.json", HERE / f"records/gates_{name}_physics.json"):
            if path.exists():
                gates.update(json.loads(path.read_text())["gates"])
        guard_off = HERE / f"records/gates_{name}_ceiling-with-the-guard-off.json"
        if guard_off.exists():
            gates["ceiling_with_the_guard_on"] = gates.pop("ceiling")
            gates["ceiling"] = json.loads(guard_off.read_text())["gates"]["ceiling"]
        out.append(f"{name}: " + ", ".join(f"{gate} {'passed' if r['passed'] else 'FAILED'}" for gate, r in gates.items()))
        if "walls" in gates:
            walls = gates["walls"]
            out.append(f"  walls: {walls['cases']} cases, boxes touched in C {walls['wall_boxes_touched_in_c_mujoco']} of "
                       f"{walls['wall_boxes']}; warp {walls['warp']}; c {walls['c_mujoco']}; one step from contact "
                       f"states {walls['one_physics_step_from_c_wall_contact_states']}; pool {walls['warp_pool']}")
        if "ceiling" in gates:
            heights = [(t["cell"], t["launch_speed"], t["warp_largest_height_m"], t["c_largest_height_m"],
                        t["warp_steps_touching_the_ceiling"], t["c_steps_touching_the_ceiling"]) for t in gates["ceiling"]["throws"]]
            out.append(f"  ceiling: {heights}")
        if "parity" in gates:
            parity = gates["parity"]
            out.append(f"  parity one step: free {parity['one_step']['contact_free']['warp_against_c']}, contact "
                       f"{parity['one_step']['in_contact']['warp_against_c']}; trajectory warp {parity['trajectory']['warp']}, "
                       f"first contact {parity['trajectory']['first_state_with_a_contact']}; resting warp "
                       f"{parity['resting']['warp']['height']:.5f} c {parity['resting']['c_mujoco']['height']:.5f} moved "
                       f"{parity['resting']['warp']['c_settled_state_moved_by']:.2e}")
        if "step_limit" in gates:
            limit = gates["step_limit"]
            out.append(f"  step limit: {limit['steps_with_a_truncation']}, episodes {limit['episodes_counted']}, "
                       f"terminations {limit['terminations']}, respawns {limit['respawns_for_a_broken_state']}")
        if "guard" in gates:
            guard = gates["guard"]
            out.append(f"  guard: respawned {guard['respawned']}, volume {guard['volume_bound_inside_outside']}, "
                       f"half extent {guard['half_extent']}")
    return "\n".join(out)


if __name__ == "__main__":
    print(throughput_table(profile_rows()))
    print()
    print(gate_lines())
