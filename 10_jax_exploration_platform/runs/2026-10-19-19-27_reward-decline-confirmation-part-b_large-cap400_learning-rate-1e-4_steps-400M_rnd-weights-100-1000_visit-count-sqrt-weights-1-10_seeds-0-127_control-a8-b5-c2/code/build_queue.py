"""Write the confirmation run's work queue: the control and the fixes, with RND and the oracle
1/sqrt(n) bonus, at learning rate 1e-3 for 100M steps and at 1e-4 for 400M steps.

Part A (1e-3, 100M): RND with weights 1, 10, 100, 1000 (4 x 128 copies), oracle with weights 1 and 10
(2 x 128). Part B (1e-4, 400M): RND with weights 100 and 1000, oracle with 1 and 10 (2 x 128 each),
the weights that led the 1000M-step runs at 1e-4. Every idea of a part has the same copy count per
bonus, so copy k meets the same environments and policy noise across ideas.
Usage: python3 build_queue.py --part <part> --ideas control,A8,B5[,C...]   (refuses to overwrite a queue)
"""
import argparse
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
PYTHON = "/p/rlprojects/RND/.venvs/platform_jax/bin/python"
TRAIN = "/p/rlprojects/RND/10_jax_exploration_platform/scripts/run_training.py"
STEPS_PER_ITERATION = 128 * 4
ANNEAL_1000M = -(-1_000_000_000 // STEPS_PER_ITERATION)

# part: (learning rate, steps per copy, window iterations, {bonus: weights})
PARTS = {
    "lr-1e-3_steps-100M": (1e-3, 100_000_000, 500,
                           {"rnd_next_state": [1.0, 10.0, 100.0, 1000.0],
                            "gt_position_velocity_sqrt": [1.0, 10.0]}),
    "lr-1e-4_steps-400M": (1e-4, 400_000_000, 2000,
                           {"rnd_next_state": [100.0, 1000.0],
                            "gt_position_velocity_sqrt": [1.0, 10.0]}),
}

# idea id -> (name, flags given the run's iterations); "anneal" None = over 1000M (the control's)
IDEAS = {
    "control": ("control", lambda n: []),
    "A8": ("a8_learning-rate-annealed-over-the-run", lambda n: ["--lr-anneal-iterations", str(n)]),
    "B5": ("b5_terminate-at-goal_reward-shift-minus-1_gradient-clip-per-network",
           lambda n: ["--terminate-at-goal", "--reward-shift", "-1", "--clip-per-network"]),
    "C1": ("c1_terminate-at-goal_reward-shift-minus-1_anneal-in-run",
           lambda n: ["--terminate-at-goal", "--reward-shift", "-1",
                      "--lr-anneal-iterations", str(n)]),
    "C2": ("c2_entropy-coefficient-0.01_anneal-in-run",
           lambda n: ["--ent-coef", "0.01", "--lr-anneal-iterations", str(n)]),
}


def main():
    """One marker per (part, idea, bonus) in queue/pending, the longest units first."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ideas", required=True)
    parser.add_argument("--part", required=True, choices=list(PARTS))
    args = parser.parse_args()
    pending = RUN / "queue" / "pending"
    if any(any((RUN / "queue" / s).iterdir()) for s in ("pending", "running", "done", "failed")):
        raise SystemExit("the queue already has markers; refusing to rebuild it")
    units = []
    for part, (lr, steps, window, bonuses) in PARTS.items():
        if part != args.part:
            continue
        iterations = -(-steps // STEPS_PER_ITERATION)
        for idea in args.ideas.split(","):
            name, flags = IDEAS[idea]
            for bonus, weights in bonuses.items():
                unit = f"{part}_{name}_{bonus}"
                params = RUN / "code" / "parameters" / f"{part}_{bonus}.json"
                params.parent.mkdir(parents=True, exist_ok=True)
                params.write_text(json.dumps({bonus: [[lr, w] for w in weights]}) + "\n")
                argv = [PYTHON, "-u", TRAIN, "--run-dir", str(RUN), "--unit-id", unit,
                        "--bonus", bonus, "--parameter-settings", str(params),
                        "--copies-per-cell", "128", "--iterations", str(iterations),
                        "--window-iterations", str(window), "--track-coverage", "--diagnostics",
                        "--no-aggregate", "--map", "large", "--start-cell", "7,1",
                        "--goal-cell", "1,10", "--episode-steps", "400",
                        "--lr-anneal-iterations", str(ANNEAL_1000M),
                        "--description", f"reward-decline confirmation {part} {idea} {bonus}",
                        *flags(iterations)]
                units.append((len(weights) * steps, unit, argv))
    units.sort(key=lambda u: -u[0])
    for k, (_, unit, argv) in enumerate(units):
        marker = {"run_id": k, "run_total": len(units), "seed": 0, "seed_indices": [0, 127],
                  "group": "reward-decline-confirmation", "profile": unit, "argv": argv,
                  "completed_marker": None}
        (pending / f"{k:02d}_of_{len(units)}.json").write_text(json.dumps(marker) + "\n")
        print(k, unit)


if __name__ == "__main__":
    main()
