"""Print the main numbers of one or more rollout records (standard library only)."""
import json
import sys

for path in sys.argv[1:]:
    r = json.load(open(path))
    e = r["environment_config"]
    print(f"== {path.split('/')[-1]}")
    print(f"   integrator {e['integrator']}, solver {e['solver_iterations']}/{e['ls_iterations']}, time step {r.get('physics_time_step', 0.01)}, "
          f"frame skip {e['frame_skip']}, guard bound {e.get('fault_torso_speed', 0.0)}")
    steps = r["environment_steps"]
    print(f"   environment steps {steps:,} in {r['chunks']} chunks; onsets above 100 m/s: {r['onsets_large']} "
          f"({r['onsets_large'] / steps * 1e6:.3f} per million steps); onsets 30 to 100 m/s: {r['onsets_moderate']}; "
          f"non-finite respawns {r['non_finite_respawns']}; guard and non-finite respawns {r.get('physics_fault_respawns')}")
    print(f"   speed histogram {r['speed_histogram_edges']}: {r['speed_histogram_counts']}")
    if "angular_speed_histogram_counts" in r:
        print(f"   angular speed {r['angular_speed_histogram_edges']}: {r['angular_speed_histogram_counts']}")
        print(f"   angular speed, torso above 1 m: {r['angular_speed_histogram_counts_torso_above_1_m']}")
        print(f"   largest joint speed {r['largest_joint_speed_histogram_edges']}: {r['largest_joint_speed_histogram_counts']}")
    print(f"   largest speed after any physics step {r['largest_speed_after_any_physics_step']:.4g}; policies above 100: "
          f"{r['largest_speed_after_any_physics_step_per_policy_over_1e2']}; steps above 1.5 m: {r['steps_above_1_5_m'] / steps:.2%}; "
          f"arrivals per 1,000 episodes: {1000 * r['arrivals'] / max(r['episodes_ended'], 1):.1f}")
    print(f"   rollout {r['rollout_environment_steps_per_second'] / 1e6:.3f} million steps per second; pool rows {r['pool']['nefc_peak']}, dropped {r['pool']['dropped']}")
    t = r.get("training_iteration")
    if t:
        print(f"   training iteration: {t['seconds_per_iteration']:.3f} s (smallest {t['seconds_per_iteration_smallest']:.3f}); total "
              f"{t['total_steps_per_second'] / 1e6:.3f} million steps/s; per copy {t['steps_per_second_per_copy'] / 1e3:.2f} thousand steps/s; "
              f"{t['hours_per_million_steps_per_copy']:.4f} hours per million steps per copy")
