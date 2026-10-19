"""Write one idea batch's work queue: the control and every idea, each one 512-copy RND program
holding the intrinsic weights 1, 10, 100 and 1000 (128 paired copies each), longest units first.

The map and the budget are this batch's command-line arguments (the screening's choice); every
idea adds its own flags to the control's command, and nothing else differs between units.
Usage: python3 build_queue.py --map large --start-cell 7,1 --goal-cell 1,10 --episode-steps 400
       --steps 50000000 --lr-anneal-steps 1000000000
"""
import argparse
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
PYTHON = "/p/rlprojects/RND/.venvs/platform_jax/bin/python"
TRAIN = "/p/rlprojects/RND/10_jax_exploration_platform/scripts/run_training.py"
STEPS_PER_ITERATION = 128 * 4
SETTINGS = [[1e-3, 1.0], [1e-3, 10.0], [1e-3, 100.0], [1e-3, 1000.0]]

# idea id -> (unit name, the flags it adds to the control's command, relative cost per step)
IDEAS = {
    "control": ("control", [], 1.0),
    "E1": ("e1_terminate-at-goal", ["--terminate-at-goal"], 1.0),
    "E2": ("e2_terminate-at-goal_reward-shift-minus-1",
           ["--terminate-at-goal", "--reward-shift", "-1"], 1.0),
    "E3": ("e3_terminate-at-goal_goal-reward-remaining-steps",
           ["--terminate-at-goal", "--goal-reward", "remaining_steps"], 1.0),
    "A1": ("a1_intrinsic-weight-linear-decay", ["--intrinsic-weight-schedule", "linear_decay"], 1.0),
    "A2": ("a2_intrinsic-weight-goal-rate-gate-0.05", ["--success-gate-rate", "0.05"], 1.0),
    "A3": ("a3_clipped-ppo_epochs-4_minibatches-4", ["--update-style", "epoch_minibatch"], 3.0),
    "A4": ("a4_entropy-coefficient-0.01", ["--ent-coef", "0.01"], 1.0),
    "A5": ("a5_normalize-extrinsic-reward", ["--normalize-extrinsic-reward"], 1.0),
    "A6": ("a6_extrinsic-discount-0.99", ["--gamma-ext", "0.99"], 1.0),
    "A7": ("a7_gradient-clip-per-network", ["--clip-per-network"], 1.0),
}


def main():
    """One marker per idea in queue/pending; the parameter file is shared by all units."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--start-cell", required=True)
    parser.add_argument("--goal-cell", required=True)
    parser.add_argument("--episode-steps", required=True)
    parser.add_argument("--steps", type=int, required=True, help="environment steps per copy")
    parser.add_argument("--lr-anneal-steps", type=int, required=True,
                        help="environment steps per copy over which the rate anneals to zero")
    parser.add_argument("--ideas", default=",".join(IDEAS), help="comma-separated idea ids")
    args = parser.parse_args()
    pending = RUN / "queue" / "pending"
    if any(any((RUN / "queue" / s).iterdir()) for s in ("pending", "running", "done", "failed")):
        raise SystemExit("the queue already has markers; refusing to rebuild it")
    params = RUN / "code" / "parameters" / "rnd_weights-1-10-100-1000.json"
    params.parent.mkdir(parents=True, exist_ok=True)
    params.write_text(json.dumps({"rnd_next_state": SETTINGS}) + "\n")
    iterations = -(-args.steps // STEPS_PER_ITERATION)
    anneal = -(-args.lr_anneal_steps // STEPS_PER_ITERATION)
    chosen = args.ideas.split(",")
    # the expensive idea (16 gradient steps per rollout) is claimed first
    chosen.sort(key=lambda i: -IDEAS[i][2])
    for k, idea in enumerate(chosen):
        unit, flags, _ = IDEAS[idea]
        argv = [PYTHON, "-u", TRAIN, "--run-dir", str(RUN), "--unit-id", unit,
                "--bonus", "rnd_next_state", "--parameter-settings", str(params),
                "--copies-per-cell", "128", "--iterations", str(iterations),
                "--window-iterations", "250", "--track-coverage", "--diagnostics",
                "--no-aggregate", "--map", args.map, "--start-cell", args.start_cell,
                "--goal-cell", args.goal_cell, "--episode-steps", args.episode_steps,
                "--lr-anneal-iterations", str(anneal),
                "--description", f"reward-decline idea {idea}: {unit}", *flags]
        marker = {"run_id": k, "run_total": len(chosen), "seed": 0, "seed_indices": [0, 127],
                  "group": "reward-decline-batch", "profile": unit, "argv": argv,
                  "completed_marker": None}
        (pending / f"{k:02d}_of_{len(chosen)}.json").write_text(json.dumps(marker) + "\n")
        print(k, idea, unit)


if __name__ == "__main__":
    main()
