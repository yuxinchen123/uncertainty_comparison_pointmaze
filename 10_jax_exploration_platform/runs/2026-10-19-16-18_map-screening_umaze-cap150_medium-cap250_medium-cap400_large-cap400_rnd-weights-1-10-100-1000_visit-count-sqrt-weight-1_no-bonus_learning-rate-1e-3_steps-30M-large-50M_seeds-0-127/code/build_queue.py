"""Write the screening run's work queue: one marker per (map, bonus) unit, longest units first.

Each unit is one run_training.py call. RND packs its four intrinsic weights into one 512-copy
program (128 paired copies each); the oracle 1/sqrt(n) bonus and no bonus are 128 copies each.
Usage: python3 build_queue.py   (refuses to overwrite an existing queue)
"""
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
PYTHON = "/p/rlprojects/RND/.venvs/platform_jax/bin/python"
TRAIN = "/p/rlprojects/RND/10_jax_exploration_platform/scripts/run_training.py"
STEPS_PER_ITERATION = 128 * 4

# map name, start, goal, cap, steps per copy
MAPS = [("large", "7,1", "1,10", 400, 50_000_000),
        ("medium", "6,1", "1,6", 400, 30_000_000),
        ("medium", "6,1", "1,6", 250, 30_000_000),
        ("umaze", "3,1", "1,1", 150, 30_000_000)]
# bonus name, its (learning rate, intrinsic weight) settings
BONUSES = [("rnd_next_state", [[1e-3, 1.0], [1e-3, 10.0], [1e-3, 100.0], [1e-3, 1000.0]]),
           ("gt_position_velocity_sqrt", [[1e-3, 1.0]]),
           ("none", [[1e-3, 0.0]])]


def main():
    """One marker per unit in queue/pending, ordered so the longest units are claimed first."""
    pending = RUN / "queue" / "pending"
    if any(any((RUN / "queue" / s).iterdir()) for s in ("pending", "running", "done", "failed")):
        raise SystemExit("the queue already has markers; refusing to rebuild it")
    units = []
    for bonus, settings in BONUSES:
        for map_name, start, goal, cap, steps in MAPS:
            unit = f"{map_name}-cap{cap}_{bonus}"
            params = RUN / "code" / "parameters" / f"{unit}.json"
            params.write_text(json.dumps({bonus: settings}) + "\n")
            iterations = -(-steps // STEPS_PER_ITERATION)
            # copy-steps decide the order: a 512-copy unit of 50M steps goes first
            work = len(settings) * 128 * steps
            argv = [PYTHON, "-u", TRAIN, "--run-dir", str(RUN), "--unit-id", unit,
                    "--bonus", bonus, "--parameter-settings", str(params),
                    "--copies-per-cell", "128", "--iterations", str(iterations),
                    "--window-iterations", "250", "--track-coverage", "--diagnostics",
                    "--no-aggregate", "--map", map_name, "--start-cell", start,
                    "--goal-cell", goal, "--episode-steps", str(cap),
                    "--description", "map screening: smallest map that shows the reward fall"]
            units.append((work, unit, argv))
    units.sort(key=lambda u: -u[0])
    for k, (_, unit, argv) in enumerate(units):
        marker = {"run_id": k, "run_total": len(units), "seed": 0, "seed_indices": [0, 127],
                  "group": "map-screening", "profile": unit, "argv": argv,
                  "completed_marker": None}
        (pending / f"{k:02d}_of_{len(units)}.json").write_text(json.dumps(marker) + "\n")
        print(k, unit, argv[argv.index("--iterations") + 1])


if __name__ == "__main__":
    main()
