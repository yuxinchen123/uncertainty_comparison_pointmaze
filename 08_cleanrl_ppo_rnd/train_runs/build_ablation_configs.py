"""Build the 150-run ablation queue: 5 arms x 30 seeds, seed outermost.

The marker format follows this project's own work-queue convention rather than the GPU skill's
default. Your runs put the STRUCTURED parameters in the marker — `arm`, `a_seed`, `config_key`,
`params`, `fixed` — so a marker is readable on its own and the command is built from it. The GPU
skill's worker instead executes a literal `argv` array, so each marker carries both: the readable
fields are the source of truth, and `argv` is what the shared worker runs.

Run-id ordering is seed OUTERMOST and arm inner, which is the property that matters for a campaign
that will not finish:

    run_id = seed_index * 5 + arm_index

Workers claim approximately in id order, so any prefix of the queue holds every arm at the same seed
count. If only 60 of the 150 runs finish, that is 12 complete seeds across all five arms — never 30
seeds of arm 1 and none of arm 5.

Run:
    PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/exploration/bin/python build_ablation_configs.py \
        --sweep_dir <run folder> --sweep_id <YYYY-MM-DD-HH-MM>_<tag>
"""

import argparse
import json
import os

PYTHON = "/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python"
TRAINER = "/p/rlprojects/RND/08_cleanrl_ppo_rnd/src/ppo_rnd_envpool_shuze.py"

# The five arms, in the fixed order that defines arm_index. Arm 1 is the reference; the others are
# arm 1 with the named knobs changed. The knob values are recorded here as well as being set by the
# trainer's own --arm flag, so a marker states what it runs without anyone having to read the code.
# The policy is clipped at max_grad_norm=0.5 in EVERY arm. What arms 2 and 5 change is that the RND
# predictor is taken out of that clip: joint_grad_clip=False clips the policy on its own norm, and
# rnd_max_grad_norm=0 leaves the predictor's gradient unscaled.
NO_RND_CLIP = {"joint_grad_clip": False, "rnd_max_grad_norm": 0.0}
ARMS = [
    ("arm1_original", {}, "CleanRL as published"),
    ("arm2_no_rnd_grad_clip", dict(NO_RND_CLIP), "the RND predictor is not clipped; the policy still is"),
    ("arm3_update_proportion_1", {"update_proportion": 1.0}, "predictor trained on the whole batch"),
    ("arm4_shallower_predictor", {"predictor_extra_blocks": 1}, "predictor one block deeper, not two"),
    ("arm5_all", {**NO_RND_CLIP, "update_proportion": 1.0, "predictor_extra_blocks": 1},
     "all three departures together"),
]
SEEDS = list(range(1, 31))

# Every value shared by all 150 runs. Repeated into each marker so a marker is self-describing and
# no run can silently disagree with the sweep's intent.
FIXED = {
    "env_id": "MontezumaRevenge-v5",
    "total_timesteps": 2_000_000_000,
    "num_envs": 128,
    "num_steps": 128,
    "num_minibatches": 4,
    "update_epochs": 4,
    "learning_rate": 1e-4,
    "gamma": 0.999,
    "int_gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.1,
    "ent_coef": 0.001,
    "vf_coef": 0.5,
    "int_coef": 1.0,
    "ext_coef": 2.0,
    "num_iterations_obs_norm_init": 50,
    "fix_envpool_autoreset": True,
    "log_every_updates": 200,
    "log_gradient_statistics": True,
    "checkpoint_every_seconds": 3600,
    "episode_history_cap": 50000,
    "episode_history_stride": 100,
}


def build_argv(run_id, run_total, seed, arm, out_dir):
    """The command the shared GPU worker executes for one run."""
    return [
        PYTHON, "-u", TRAINER,
        "--arm", arm,
        "--env_id", FIXED["env_id"],
        "--total_timesteps", str(FIXED["total_timesteps"]),
        "--seed", str(seed),
        "--run_id", str(run_id), "--run_total", str(run_total),
        "--output_dir", out_dir,
        "--log_every_updates", str(FIXED["log_every_updates"]),
        "--checkpoint_every_seconds", str(FIXED["checkpoint_every_seconds"]),
        "--episode_history_cap", str(FIXED["episode_history_cap"]),
        "--episode_history_stride", str(FIXED["episode_history_stride"]),
        "--resume",
        "--fix_envpool_autoreset", "--opt_fixed_minibatch_shape", "--log_gradient_statistics",
        # Options that leave the arithmetic unchanged.
        "--opt_gpu_obs_rms", "--opt_fused_policy_pass", "--opt_rnd_no_grad", "--opt_uint8_obs",
        "--opt_gpu_norm_stats", "--opt_no_sync_update", "--opt_fast_obs_norm_init",
        "--opt_cudnn_benchmark",
        # Options that change the arithmetic: off, because this compares algorithm choices.
        "--no-opt_amp_fp16", "--no-opt_channels_last", "--no-opt_matmul_tf32",
        "--no-opt_torch_compile",
    ]


def build_configs(sweep_dir, sweep_id):
    """One config per (seed, arm), seed outermost, as a list of marker dicts."""
    out_dir = os.path.join(sweep_dir, "data", sweep_id, "local")
    run_total = len(SEEDS) * len(ARMS)
    configs = []
    for seed_index, seed in enumerate(SEEDS):
        for arm_index, (arm, knobs, description) in enumerate(ARMS):
            # before: seed_index=2, arm_index=3  ->  run_id = 2*5 + 3 = 13
            # after:  ids 10..14 are all of seed 3, so a prefix is always arm-balanced
            run_id = seed_index * len(ARMS) + arm_index
            configs.append({
                "sweep_id": sweep_id,
                "run_id": run_id,
                "run_total": run_total,
                "seed": seed,
                "a_seed": seed,
                "seed_index": seed_index,
                "arm": arm,
                "arm_index": arm_index,
                "arm_description": description,
                "config_key": f"{FIXED['env_id']}|{arm}",
                "group": "ppo-rnd-atari",
                "profile": "ppo-rnd-atari",
                # The knobs this arm changes from arm 1. Empty for arm 1 itself.
                "params": {k: str(v) for k, v in knobs.items()},
                "fixed": FIXED,
                "argv": build_argv(run_id, run_total, seed, arm, out_dir),
                "completed_marker": {
                    "path": os.path.join(out_dir, f"{run_id}_of_{run_total}.json"),
                    "flag_field": "completed",
                },
            })
    return configs


def main():
    """Write configs.jsonl for the ablation sweep."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--sweep_id", required=True)
    args = ap.parse_args()

    configs = build_configs(args.sweep_dir, args.sweep_id)
    path = os.path.join(args.sweep_dir, "configs.jsonl")
    with open(path, "w") as f:
        f.write("\n".join(json.dumps(c) for c in configs) + "\n")
    os.chmod(path, 0o664)

    print(f"wrote {path}: {len(configs)} runs "
          f"({len(ARMS)} arms x {len(SEEDS)} seeds)")
    print("\nfirst seven, showing the seed-outermost ordering:")
    print(f"{'run_id':>7}{'seed':>6}  arm")
    for c in configs[:7]:
        print(f"{c['run_id']:>7}{c['a_seed']:>6}  {c['arm']}")
    print("      ...")
    for c in configs[-2:]:
        print(f"{c['run_id']:>7}{c['a_seed']:>6}  {c['arm']}")


if __name__ == "__main__":
    main()
