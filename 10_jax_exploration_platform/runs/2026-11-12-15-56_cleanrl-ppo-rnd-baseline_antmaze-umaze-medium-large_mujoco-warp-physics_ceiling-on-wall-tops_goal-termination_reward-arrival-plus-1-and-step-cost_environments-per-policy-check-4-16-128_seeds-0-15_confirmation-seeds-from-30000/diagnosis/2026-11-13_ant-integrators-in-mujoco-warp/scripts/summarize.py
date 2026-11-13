"""Print the tables of notes.md from the records of this folder (standard library only).

  speed     records/speed_<card>*.jsonl: one row per (setting, policies), with the ratio of its seconds per
            iteration to the current setting's at the same policy count on the same card
  gates     records/gates_<setting>.json
  rollouts  records/rollout_<checkpoint>_<setting>_summary.json: totals, and the four configurations with
            learning rate 1e-2 for the 128-environment checkpoints
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from integrator_options import CURRENT, NAMES, SETTINGS  # noqa: E402

RECORDS = Path(__file__).resolve().parent.parent / "records"


def slug_of(record):
    """The setting slug of a speed record, from its configuration file's name."""
    # before: ".../antmaze_umaze_rk4_time-step-0.01_frame-skip-5_guard-30-and-100.toml"  after: "rk4_time-step-0.01_frame-skip-5"
    return Path(record["configuration_file"]).name[len("antmaze_umaze_"):-len("_guard-30-and-100.toml")]


def speed():
    """One table per card file: the five throughput columns and the ratio to the current setting."""
    for path in sorted(RECORDS.glob("speed_*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if not r.get("failed")]
        if not rows:
            continue
        # GOAL: the current setting's seconds per iteration at each policy count (median if measured twice)
        # before: rows of (slug, copies, seconds); after: {128: 1.16, 512: 2.9}
        base = {}
        for r in rows:
            if slug_of(r) == CURRENT:
                base.setdefault(r["copies"], []).append(r["seconds_per_iteration"])
        base = {k: statistics.median(v) for k, v in base.items()}
        print(f"\n{path.name} ({rows[0]['device_name']}, {rows[0]['host']})\n")
        print("| setting | policies | seconds per iteration | total steps per second (millions) | steps per second per "
              "policy (thousands) | hours per million steps per policy | time per iteration against the current setting |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        order = list(SETTINGS)
        for r in sorted(rows, key=lambda r: (r["copies"], order.index(slug_of(r)))):
            ratio = r["seconds_per_iteration"] / base[r["copies"]] if r["copies"] in base else float("nan")
            print(f"| {NAMES[slug_of(r)]} | {r['copies']} | {r['seconds_per_iteration']:.3f} | "
                  f"{r['total_steps_per_second_millions']:.3f} | {r['steps_per_second_per_copy_thousands']:.3f} | "
                  f"{r['hours_per_million_steps_per_copy']:.4f} | {ratio:.2f} |")
        print("\ncompile and memory:", "; ".join(
            f"{NAMES[slug_of(r)]} x {r['copies']}: first two calls {r['warmup_call_seconds'][0]:.0f} and "
            f"{r['warmup_call_seconds'][1]:.0f} s, card {r['card_memory_used_mb_largest_reading']} MiB, rows "
            f"{r['pool']['nefc_peak']}, dropped {r['pool']['dropped']}" for r in rows))


def gates():
    """One row per setting: each gate's verdict and the agreement numbers."""
    print("\n| setting | reset | goal | step limit | guard | walls | ceiling | one step, contact-free (m) | one step, "
          "in contact (m) | 20 steps, before the first contact (m) | 20 steps, all (m) | resting height Warp, C (m) | all |")
    print("|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|")
    for slug in SETTINGS:
        path = RECORDS / f"gates_{slug}.json"
        if not path.exists():
            continue
        g = json.loads(path.read_text())["gates"]
        verdict = lambda name: ("passed" if g[name]["passed"] else "FAILED") if name in g else "not run"
        cells = [verdict(n) for n in ("reset", "goal", "step_limit", "guard", "walls", "ceiling")]
        if "parity" in g:
            p = g["parity"]
            one, traj, rest = p["one_step"], p["trajectory"]["warp"], p["resting"]
            cells += [f"{one['contact_free']['warp_against_c']['largest']:.1e}", f"{one['in_contact']['warp_against_c']['largest']:.1e}",
                      f"{traj['largest_before_the_first_contact']:.1e}", f"{traj['largest_over_all_steps']:.2f}",
                      f"{rest['warp']['height']:.5f}, {rest['c_mujoco']['height']:.5f}", verdict("parity")]
        else:
            cells += ["not run"] * 6
        print(f"| {NAMES[slug]} | " + " | ".join(cells) + " |")


def rollouts():
    """Per checkpoint: one row per setting with totals; then the configurations with learning rate 1e-2."""
    names = sorted({p.name.split("_" + s)[0] for s in SETTINGS for p in RECORDS.glob(f"rollout_*_{s}_summary.json")})
    for name in names:
        print(f"\n{name}\n")
        print("| setting | episodes | episodes ended by a respawn | per 10,000 episodes | arrival rate | largest torso speed "
              "after any physics step (m/s) | episodes over 15 m/s in an observation | seconds per 70 steps of all environments | card |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        by_setting = {}
        for slug in SETTINGS:
            path = RECORDS / f"{name}_{slug}_summary.json"
            if not path.exists():
                continue
            s = json.loads(path.read_text())
            by_setting[slug] = s
            t = s["total"]
            over15 = sum(c["episodes_with_observed_speed_over_15"] for c in s["per_configuration"])
            print(f"| {NAMES[slug]} | {t['episodes']:,} | {t['respawned_episodes']} | {t['respawns_per_10000_episodes']:.1f} | "
                  f"{t['arrival_rate']:.4f} | {t['largest_torso_speed_after_any_physics_step']:.1f} | {over15} | "
                  f"{s['seconds_per_chunk_after_the_first_two']:.2f} | {s['device'].replace('NVIDIA ', '').replace('GeForce ', '')}, {s['host']} |")
        first = next(iter(by_setting.values()))
        print(f"\ncheckpoint iteration {first['checkpoint_iteration']}, "
              f"{first['environment_steps_per_policy_at_the_checkpoint'] / 1e6:.1f} million steps per policy, "
              f"{first['policies']} policies x {first['environments_per_policy']} environments, "
              f"{first['rollout_steps_per_environment']} steps per environment")
        print("\n| setting | " + " | ".join(f"rate {c['learning_rate']:g}, weight {c['int_coef']:g}" for c in first["per_configuration"]) + " |")
        print("|---|" + "---|" * len(first["per_configuration"]))
        for slug, s in by_setting.items():
            print(f"| {NAMES[slug]}: respawns, arrival rate | " + " | ".join(
                f"{c['respawned_episodes']}, {c['arrival_rate']:.3f}" for c in s["per_configuration"]) + " |")


if __name__ == "__main__":
    speed()
    gates()
    rollouts()
