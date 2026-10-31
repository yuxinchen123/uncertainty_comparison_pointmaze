"""Prepare an immutable DeltaAI real-game check without numerical imports."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import types

REPOSITORY = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl")
COMMIT = "dd2709f928cb22bccd23ad0395a1d60b97ff3884"
CAMPAIGN = "2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1"
LOCAL_RUN = REPOSITORY / "10_jax_exploration_platform/benchmark_runs" / CAMPAIGN
PREPARATION = LOCAL_RUN / "remote_setup/deltaai_small_real_game_check"
REMOTE_BASE = "/projects/biwj/sliu51/rnd_atari_pixels"
REMOTE_CODE = f"{REMOTE_BASE}/code/{COMMIT}"
REMOTE_RUN = f"{REMOTE_BASE}/runs/{CAMPAIGN}"
UNIT = "deltaai_original_pixel_real_game_copies-1_environments-2_rollout-2_updates-2"
PYTHON = "/projects/biwj/sliu51/envs/rnd_atari_pixels_jax/bin/python"


def prepare():
    """Write the source archive, complete configuration, provenance and scheduler command."""
    # Archive only committed code; exclude existing output data and reference repositories.
    PREPARATION.mkdir(exist_ok=True)
    archive = Path("/tmp/rnd_atari_pixels_deltaai_dd2709f.tar")
    subprocess.run([
        "git", "-C", str(REPOSITORY), "archive", "--format=tar", "--output", str(archive),
        COMMIT, "10_jax_exploration_platform/src",
        "10_jax_exploration_platform/scripts/atari_pixel_campaign"], check=True)
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    (PREPARATION / "code_archive.sha256").write_text(f"{archive_hash}  {archive.name}\n")

    # Resolve dataclass defaults with Python metadata; no JAX or model module is imported.
    source = subprocess.check_output([
        "git", "-C", str(REPOSITORY), "show",
        f"{COMMIT}:10_jax_exploration_platform/src/exploration_platform/atari_pixels/config.py"], text=True)
    module = types.ModuleType("deltaai_record_configuration")
    sys.modules[module.__name__] = module
    exec(compile(source, "<committed configuration>", "exec"), module.__dict__)
    config = module.PixelPPOConfig(n_copies=1, n_envs=2, num_steps=2, update_epochs=1,
                                  num_minibatches=1, observation_warmup_iterations=1)
    resolved = {
        "configuration": asdict(config), "mode": "smoke", "measured_iterations": 2,
        "compile_warmup_iterations": 2, "frame_skip": 4, "max_episode_steps": 108000,
        "sticky_action_probability": 0.25, "noop_max": 30, "episodic_life": True,
        "source_commit": COMMIT}
    (PREPARATION / "config_resolved.yaml").write_text(json.dumps(resolved, indent=2) + "\n")
    command = [
        PYTHON, f"{REMOTE_CODE}/10_jax_exploration_platform/scripts/atari_pixel_campaign/run_unit.py",
        "--run-dir", REMOTE_RUN, "--unit-id", UNIT, "--mode", "smoke", "--copies", "1",
        "--envs-per-copy", "2", "--rollout-steps", "2", "--epochs", "1", "--minibatches", "1",
        "--warmup-rollouts", "1", "--iterations", "2"]
    command_text = shlex.join(command)
    (PREPARATION / "command.txt").write_text(command_text + "\n")

    # Record campaign identity and source hashes before any compute execution can begin.
    manifest = {
        "schema_version": 1, "run_id": CAMPAIGN,
        "description": "Small real-game check of pinned pixel PPO and RND on a DeltaAI compute node.",
        "git": {"commit": COMMIT, "dirty": False, "source_archive_sha256": archive_hash},
        "sources": {
            "cleanrl": "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb",
            "jaxtari": "04ab9aec318c72162e2e53503a594a489d7e162b",
            "alternative_sprite_archive_sha256": "7f4ef286f0763c6b36946d5a35043f542395f67d12cf1a3e91150649237ef83e"},
        "unit": UNIT, "compile_signature": {"copies": 1, "environments_per_copy": 2,
                                                "rollout_steps": 2, "epochs": 1, "minibatches": 1},
        "artifacts": {"records": f"data/{UNIT}", "submitted_job_ids": "slurm/submitted_jobids_deltaai.txt"}}
    (PREPARATION / "manifest.yaml").write_text(json.dumps(manifest, indent=2) + "\n")
    background = f"""## Purpose

Check that the committed pixel Montezuma PPO and random network distillation program runs on a DeltaAI GH200 compute node. This short check belongs to campaign {CAMPAIGN}.

## Key hyperparameters

One policy, two environments, two rollout steps, one epoch, one minibatch, and one normalization warmup rollout. Two compilation warmup updates precede two measured updates. Other settings retain the original CleanRL defaults in config_resolved.yaml. Each action repeats for four game frames; observations contain four grayscale 84 by 84 pixel frames.

## Code and config changes

The check uses an immutable git archive, the separate ARM environment, and verified alternative sprites. It saves no checkpoints and does not start full training.

## Git state

Commit: `{COMMIT}`. The archive contains only committed files and has SHA256 `{archive_hash}`. Uncommitted files are excluded.
"""
    (PREPARATION / "experiment_background.md").write_text(background)

    # Preserve the exact scheduler script and run command for this one compute allocation.
    job = f"""#!/usr/bin/env bash
#SBATCH --account=biwj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=rnd-atari-pixel-check
#SBATCH --output={REMOTE_RUN}/slurm/deltaai_pixel_check_%j.out
#SBATCH --error={REMOTE_RUN}/slurm/deltaai_pixel_check_%j.out
#SBATCH --no-requeue
set -euo pipefail
export PYTHONNOUSERSITE=1
export JAX_ENABLE_X64=true
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_PLATFORMS=cuda
export XDG_DATA_HOME={REMOTE_BASE}/assets
export JAXTARI_SKIP_SPRITE_UPDATE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
unset LD_LIBRARY_PATH
run_directory={shlex.quote(REMOTE_RUN)}
job_directory="$run_directory/slurm/jobs/${{SLURM_JOB_ID}}_${{HOSTNAME}}"
mkdir -p "$job_directory"
cp "$run_directory/slurm/deltaai_pixel_check.sbatch" "$job_directory/job.sbatch"
printf '%s\\n' '{{"unit_id":"{UNIT}","source_commit":"{COMMIT}"}}' > "$job_directory/assignment.json"
exec > >(tee "$job_directory/job.log") 2>&1
cd {shlex.quote(REMOTE_CODE)}
{command_text}
cp "$run_directory/data/{UNIT}/hardware.json" "$job_directory/hardware.json"
"""
    (PREPARATION / "deltaai_pixel_check.sbatch").write_text(job)
    print(json.dumps({"code_archive": str(archive), "preparation": str(PREPARATION),
                      "remote_code": REMOTE_CODE, "remote_run": REMOTE_RUN,
                      "unit": UNIT, "source_sha256": archive_hash}))


if __name__ == "__main__":
    prepare()
