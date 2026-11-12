"""Prepare/launch an opt-in count-free neural-RND 1B Montezuma run.

This does NOT submit Slurm jobs, acquire GPU allocations, connect to clusters,
or touch the archived 12-seed baseline. Run INSIDE an already-authorized GPU
allocation, using the existing resource locks and per-node wrappers.

Examples:
 PYTHONPATH=src python scripts/atari_pixel_campaign/launch_neural_rnd_1b.py \
    --device a4500 --variant uniform --seed-start 13 --out-root /path/to/new/runs
 PYTHONPATH=src python scripts/atari_pixel_campaign/launch_neural_rnd_1b.py \
    --device h100 --variant uniform_shrink --seed-start 13 --out-root /path/to/new/runs --execute

The default only prints the command; --execute launches training in foreground.
A cluster administrator/user must supply the right GPU and Slurm allocation.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

PLATFORM = Path(__file__).resolve().parents[2]
TRAIN = PLATFORM / "scripts" / "atari_pixel_campaign" / "train_final.py"
DEFAULT_ASSETS = (
    "/p/rlprojects/RND/reference_repos/"
    "jaxtari_pixel_study_assets_montezuma_room_geometry_from_png"
)
COMMON_ENVIRONMENT = {
    "separable_rectangle_drawing": True,
    "unroll_fixed_render_loops": True,
    "masked_conditional_draws": True,
    "two_way_masked_draws": True,
    "render_final_two_together": True,
    "max_episode_steps": 27000,
}
CANDIDATES = {
    "baseline": {},
    "uniform": {"rnd_uniform_start": True},
    "power": {"rnd_predictor_loss": "power",
              "rnd_predictor_optimizer": "sgd"},
    "uniform_power": {"rnd_uniform_start": True,
                      "rnd_predictor_loss": "power",
                      "rnd_predictor_optimizer": "sgd"},
    "uniform_shrink": {"rnd_uniform_start": True,
                       "rnd_predictor_loss": "shrink",
                       "rnd_predictor_optimizer": "sgd"},
}
DEFAULT_COUNTS = {"a4500": 1, "rtx5080": 1, "h100": 4}


def build_command(args):
    if args.copies is None:
        copies = DEFAULT_COUNTS[args.device]
    else:
        copies = args.copies
    if copies < 1:
        raise ValueError("copies must be positive")
    if args.target_updates < 1 or args.target_updates > 122070:
        raise ValueError("target_updates must be between 1 and the 2B horizon")
    learner = dict(CANDIDATES[args.variant])
    if args.device == "h100":
        learner.update({
            "convolution_algorithm": "space_to_depth",
            "iteration_compiler_options": [["xla_gpu_enable_while_loop_double_buffering", "true"]],
        })
    else:
        learner["convolution_batching"] = "unrolled"
    learner.update({"cache_target_features": True, "predictor_selected_rows": 1280})
    if args.variant.endswith("power") or args.variant == "power":
        learner["rnd_power_exponent"] = args.power_exponent
    if args.variant.endswith("shrink"):
        learner["rnd_decay_exponent"] = args.decay_exponent
    if learner.get("rnd_predictor_optimizer") == "sgd":
        learner["rnd_predictor_learning_rate"] = args.predictor_lr
    # Critical: never pass total_timesteps=1B; PPO's LR still anneals over 2B.
    assert "total_timesteps" not in learner
    environment = dict(COMMON_ENVIRONMENT,
        reset_lane_budget=(16 if args.device == "h100" else 8))
    # Encode variant hyperparameters so independent experiments cannot silently
    # reuse a checkpoint directory belonging to another candidate.
    variant_id = args.variant
    if "power" in args.variant:
        variant_id += f"_p{args.power_exponent:g}_lr{args.predictor_lr:g}"
    elif "shrink" in args.variant:
        variant_id += f"_a{args.decay_exponent:g}_lr{args.predictor_lr:g}"
    directory = (args.out_root / args.device / variant_id /
                 f"seed-{args.seed_start:05d}_copies-{copies:02d}")
    command = [
        sys.executable, "-u", str(TRAIN),
        "--run-dir", str(directory),
        "--copies", str(copies),
        "--seed-start", str(args.seed_start),
        "--target-updates", str(args.target_updates),
        "--observation-warmup-iterations", "50",
        "--checkpoint-seconds", "1200",
        "--learner", json.dumps(learner, sort_keys=True),
        "--environment", json.dumps(environment, sort_keys=True),
    ]
    return command, directory, learner, environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=sorted(DEFAULT_COUNTS), required=True)
    p.add_argument("--variant", choices=sorted(CANDIDATES), required=True)
    p.add_argument("--seed-start", type=int, default=13)
    p.add_argument("--copies", type=int, default=None)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--target-updates", type=int, default=61035)
    p.add_argument("--power-exponent", type=float, choices=[2., 3.], default=2.)
    p.add_argument("--decay-exponent", type=float, choices=[0.5, 1., 2.], default=1.)
    p.add_argument("--predictor-lr", type=float, default=1e-3)
    p.add_argument("--assets", type=Path, default=Path(DEFAULT_ASSETS))
    p.add_argument("--execute", action="store_true", help="Actually run on an ALLOCATED GPU")
    args = p.parse_args()
    if args.predictor_lr <= 0:
        p.error("--predictor-lr must be positive")
    command, directory, learner, environment = build_command(args)
    print(f"GPU: {args.device}; independent copies: {args.copies or DEFAULT_COUNTS[args.device]}")
    print(f"Run: {directory}")
    print(f"Learner options: {json.dumps(learner, sort_keys=True)}")
    print(f"Game options: {json.dumps(environment, sort_keys=True)}")
    print("Command:", shlex.join(command))
    if not args.execute:
        print("DRY RUN: no program or Slurm job was started.")
        return
    # Running without an allocated GPU is a bug, not a CPU fallback.
    if not args.assets.is_dir():
        p.error(f"corrected JAXtari asset pack not found: {args.assets}")
    # Set compilation flags before importing JAX, not after device initialization.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
    flags = os.environ.get("XLA_FLAGS", "").split()
    if "--xla_gpu_exclude_nondeterministic_ops=true" not in flags:
        flags.append("--xla_gpu_exclude_nondeterministic_ops=true")
    os.environ["XLA_FLAGS"] = " ".join(flags)
    import jax
    devices = jax.devices("gpu")
    if len(devices) != 1:
        p.error("Require exactly one visible GPU under an authorized Slurm allocation")
    if args.device == "h100" and "H100" not in devices[0].device_kind.upper():
        p.error("selected H100 settings on a non-H100 card")
    if args.device == "a4500" and "A4500" not in devices[0].device_kind.upper():
        p.error("selected A4500 settings on another card")
    if args.device == "rtx5080" and "5080" not in devices[0].device_kind.upper():
        p.error("selected RTX 5080 settings on another card")
    env = dict(os.environ)
    env["XDG_DATA_HOME"] = str(args.assets)
    env["JAXTARI_SKIP_SPRITE_UPDATE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["JAX_ENABLE_X64"] = "true"
    env["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
    flags = env.get("XLA_FLAGS", "").split()
    if "--xla_gpu_exclude_nondeterministic_ops=true" not in flags:
        flags.append("--xla_gpu_exclude_nondeterministic_ops=true")
    env["XLA_FLAGS"] = " ".join(flags)
    env["PYTHONPATH"] = str(PLATFORM / "src")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "launch_specification.json").write_text(json.dumps({
        "device": args.device, "variant": args.variant, "copy_count": args.copies or DEFAULT_COUNTS[args.device],
        "seed_start": args.seed_start, "target_updates": args.target_updates,
        "anneal_horizon_agent_steps": 2_000_000_000,
        "learner": learner, "environment": environment,
        "assets": str(args.assets)}, sort_keys=True, indent=2) + "\n")
    raise SystemExit(subprocess.call(command, env=env, cwd=PLATFORM))


if __name__ == "__main__":
    main()
