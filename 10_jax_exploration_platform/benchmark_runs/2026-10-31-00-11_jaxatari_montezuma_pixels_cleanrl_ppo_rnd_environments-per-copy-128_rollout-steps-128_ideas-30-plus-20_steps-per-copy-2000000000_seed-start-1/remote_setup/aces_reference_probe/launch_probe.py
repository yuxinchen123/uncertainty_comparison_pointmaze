"""Prepare, stage, and submit the single authorized ACES reference resource probe."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import types


REPOSITORY = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl")
COMMIT = "d28e6735e79788f618acb0ebfde8609db1f776af"
CAMPAIGN = "2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1"
LOCAL_RUN = REPOSITORY / "10_jax_exploration_platform/benchmark_runs" / CAMPAIGN
PREPARATION = Path(__file__).resolve().parent
REMOTE_BASE = "/scratch/user/u.sl359075/rnd_atari_pixels"
REMOTE_CODE = f"{REMOTE_BASE}/code/{COMMIT}"
REMOTE_RUN = f"{REMOTE_BASE}/runs/{CAMPAIGN}"
UNIT = "aces_reference_copies_1_source_d28e6735e797"
PYTHON = "/scratch/user/u.sl359075/envs/rnd_atari_pixels_jax/bin/python"
ARCHIVE = Path("/tmp/rnd_atari_pixels_aces_d28e673.tar")
REMOTE_ARCHIVE = f"{REMOTE_BASE}/remote_setup/rnd_atari_pixels_aces_d28e673.tar"


def remote_python(source, *arguments):
    """Run standard-library metadata work through the existing SSH connection."""
    subprocess.run(["ssh", "-O", "check", "aces"], capture_output=True, text=True, check=True)
    command = " ".join(shlex.quote(value) for value in ("/usr/bin/python3", "-", *arguments))
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "aces", command], input=source,
                          capture_output=True, text=True, check=True).stdout


def prepare():
    """Write complete records and the exact immutable source archive before submission."""
    # Include only committed numerical code, campaign scripts, and the original preset.
    subprocess.run(["git", "-C", str(REPOSITORY), "archive", "--format=tar", "--output", str(ARCHIVE),
                    COMMIT, "10_jax_exploration_platform/src",
                    "10_jax_exploration_platform/scripts/atari_pixel_campaign",
                    "10_jax_exploration_platform/configs/experiments/montezuma_pixels_original_cleanrl.toml"],
                   check=True)
    archive_hash = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    (PREPARATION / "code_archive.sha256").write_text(f"{archive_hash}  {ARCHIVE.name}\n")

    # Resolve dataclass metadata without importing JAX, NumPy, or initializing a model.
    source = subprocess.check_output(["git", "-C", str(REPOSITORY), "show",
        f"{COMMIT}:10_jax_exploration_platform/src/exploration_platform/atari_pixels/config.py"], text=True)
    module = types.ModuleType("aces_probe_configuration")
    sys.modules[module.__name__] = module
    exec(compile(source, "<committed configuration>", "exec"), module.__dict__)
    learner = asdict(module.PixelPPOConfig())
    configuration = {"learner": learner, "mode": "benchmark", "rounds": 7,
                     "updates_per_round": 10, "compile_warmup_updates": 2, "copy_seeds": [1],
                     "environment": {"frame_skip": 4, "max_episode_steps": 108000,
                        "sticky_action_probability": 0.25, "noop_max": 30, "episodic_life": True},
                     "source_commit": COMMIT}
    (PREPARATION / "config_resolved.yaml").write_text(json.dumps(configuration, indent=2) + "\n")
    command = [PYTHON, f"{REMOTE_CODE}/10_jax_exploration_platform/scripts/atari_pixel_campaign/run_unit.py",
               "--run-dir", REMOTE_RUN, "--unit-id", UNIT, "--mode", "benchmark",
               "--configuration", f"{REMOTE_CODE}/10_jax_exploration_platform/configs/experiments/montezuma_pixels_original_cleanrl.toml",
               "--copies", "1", "--envs-per-copy", "128", "--rollout-steps", "128",
               "--epochs", "4", "--minibatches", "4", "--warmup-rollouts", "50",
               "--rounds", "7", "--iterations", "10", "--seed-start", "1"]
    command_text = shlex.join(command)
    (PREPARATION / "command.txt").write_text(command_text + "\n")

    # This is a resource probe in the existing study, not a paired optimization result.
    manifest = {"schema_version": 1, "run_id": CAMPAIGN,
                "description": "Full-size original reference capacity and throughput probe on ACES.",
                "git": {"commit": COMMIT, "dirty": False, "source_archive_sha256": archive_hash},
                "unit": UNIT, "classification": "resource_probe_only",
                "compile_signature": {"copies": 1, "environments_per_copy": 128,
                    "rollout_steps": 128, "epochs": 4, "minibatches": 4,
                    "network_dtype": "float32", "statistics_dtype": "float64"},
                "artifacts": {"records": f"data/{UNIT}", "job_ids": "slurm/submitted_jobids_aces.txt"}}
    (PREPARATION / "manifest.yaml").write_text(json.dumps(manifest, indent=2) + "\n")
    (PREPARATION / "experiment_background.md").write_text(f"""## Purpose

Measure actual memory, initialization time, and steady training throughput for one original reference policy on an ACES H100. This resource probe belongs to campaign {CAMPAIGN}; it is excluded from the thirty paired optimization comparisons.

## Key hyperparameters

One independent policy uses 128 environments, 128 rollout steps, four epochs, four minibatches, and fifty normalization warmup rollouts. Two compilation warmup updates precede seven rounds of ten measured updates. The seed is 1. The complete fixed CleanRL settings are in config_resolved.yaml. Neural calculations use float32 with highest matrix precision; running statistics use float64. Deterministic GPU operations are enabled.

## Code and config changes

The probe executes the exact committed source archive and original named configuration in the dedicated ACES environment with verified alternative sprites. It requests one H100, eight processors, 32 gigabytes of host memory, and one hour. It saves no checkpoints and does not start the final two-billion-step training.

## Git state

Commit: `{COMMIT}`. The archive contains committed files only, SHA256 `{archive_hash}`. Other working-tree edits are excluded.
""")
    job = f"""#!/usr/bin/env bash
#SBATCH --account=158590754315
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --job-name=rnd-atari-aces-reference-d28e673
#SBATCH --output={REMOTE_RUN}/slurm/aces_reference_%j.out
#SBATCH --error={REMOTE_RUN}/slurm/aces_reference_%j.out
#SBATCH --no-requeue
set -euo pipefail
unset LD_LIBRARY_PATH
export ATARI_PIXEL_ASSETS={REMOTE_BASE}/assets
source {REMOTE_CODE}/10_jax_exploration_platform/scripts/atari_pixel_campaign/worker_environment.sh
export JAX_PLATFORMS=cuda
run_directory={shlex.quote(REMOTE_RUN)}
job_directory="$run_directory/slurm/jobs/${{SLURM_JOB_ID}}_${{HOSTNAME}}"
mkdir -p "$job_directory"
cp "$run_directory/slurm/aces_reference_d28e673.sbatch" "$job_directory/job.sbatch"
printf '%s\\n' '{{"unit_id":"{UNIT}","source_commit":"{COMMIT}","classification":"resource_probe_only"}}' > "$job_directory/assignment.json"
exec > >(tee "$job_directory/job.log") 2>&1
cd {shlex.quote(REMOTE_CODE)}
set +e
{command_text}
probe_status=$?
set -e
if test -f "$run_directory/data/{UNIT}/hardware.json"; then
    cp "$run_directory/data/{UNIT}/hardware.json" "$job_directory/hardware.json"
fi
printf '%s\\n' "$probe_status" > "$job_directory/exit_code.txt"
exit "$probe_status"
"""
    (PREPARATION / "aces_reference_d28e673.sbatch").write_text(job)
    (PREPARATION / "remote_paths.toml").write_text(
        f'ssh_alias = "aces"\nsource_commit = "{COMMIT}"\ncode = "{REMOTE_CODE}"\n'
        f'run = "{REMOTE_RUN}"\nunit = "{UNIT}"\npython = "{PYTHON}"\n'
        f'remote_id_file = "{REMOTE_RUN}/slurm/submitted_jobids_aces.txt"\n'
        f'unit_records = "{REMOTE_RUN}/data/{UNIT}"\n'
        f'scheduler_log = "{REMOTE_RUN}/slurm/aces_reference_<jobid>.out"\n'
        'classification = "resource_probe_only"\nmonitor_cadence_minutes = 20\n'
        'parent_cron_id = "72a562a5-596b-41c0-af66-98beb4034a8e"\n')
    print(json.dumps({"prepared": str(PREPARATION), "source_sha256": archive_hash}))


def stage():
    """Copy launch records and atomically publish hash-checked immutable source on ACES."""
    # Make only metadata directories; no numerical imports execute through SSH.
    remote_python("import os,sys\nfor p in sys.argv[1:]:\n os.makedirs(p,exist_ok=True)\n",
                  f"{REMOTE_BASE}/remote_setup", f"{REMOTE_BASE}/code", f"{REMOTE_RUN}/slurm")
    subprocess.run(["scp", "-o", "BatchMode=yes", str(ARCHIVE), "aces:" + REMOTE_ARCHIVE], check=True)
    archive_hash = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    script = """import hashlib,json,os,sys,tarfile,tempfile
archive,destination,expected,commit=sys.argv[1:]
with open(archive,'rb') as stream: actual=hashlib.sha256(stream.read()).hexdigest()
if actual!=expected: raise RuntimeError('source archive hash differs')
record={'commit':commit,'archive_sha256':expected}
if os.path.exists(destination):
 with open(os.path.join(destination,'SNAPSHOT.json')) as stream: previous=json.load(stream)
 if previous!=record: raise RuntimeError('an existing immutable source differs')
else:
 staging=tempfile.mkdtemp(prefix='building_'+commit[:12]+'_',dir=os.path.dirname(destination))
 with tarfile.open(archive) as source: source.extractall(staging)
 with open(os.path.join(staging,'SNAPSHOT.json'),'w') as stream: json.dump(record,stream)
 os.rename(staging,destination)
print(json.dumps(record))
"""
    print(remote_python(script, REMOTE_ARCHIVE, REMOTE_CODE, archive_hash, COMMIT).strip())
    files = ["experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt"]
    subprocess.run(["scp", "-o", "BatchMode=yes", *[str(PREPARATION / name) for name in files],
                    "aces:" + REMOTE_RUN + "/"], check=True)
    subprocess.run(["scp", "-o", "BatchMode=yes", str(PREPARATION / "aces_reference_d28e673.sbatch"),
                    "aces:" + REMOTE_RUN + "/slurm/"], check=True)
    print(remote_python("import subprocess,sys\np=subprocess.Popen(['maxconfig','-f',sys.argv[1]],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)\no,e=p.communicate()\nprint(o)\nprint(e)\nsys.exit(p.returncode)\n",
                        f"{REMOTE_RUN}/slurm/aces_reference_d28e673.sbatch"))


def submit():
    """Submit once and persist the exact returned identifier before reporting it."""
    # The remote durable record prevents duplicate submission after a lost SSH response.
    script = """import os,subprocess,sys
run,unit,commit=sys.argv[1:]
record=os.path.join(run,'slurm',unit+'.jobid')
if os.path.exists(record):
 with open(record) as stream: identifier=stream.read().strip()
else:
 process=subprocess.Popen(['sbatch','--parsable',os.path.join(run,'slurm','aces_reference_d28e673.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 out,err=process.communicate()
 if process.returncode: raise RuntimeError(err)
 identifier=out.strip().split(';')[0]
 if not identifier.isdigit(): raise RuntimeError('unexpected sbatch output: '+out)
 with open(record,'x') as stream: stream.write(identifier+'\\n'); stream.flush(); os.fsync(stream.fileno())
 with open(os.path.join(run,'slurm','submitted_jobids_aces.txt'),'a') as stream: stream.write(identifier+'\\n'); stream.flush(); os.fsync(stream.fileno())
 with open(os.path.join(run,'slurm','submissions_aces.tsv'),'a') as stream: stream.write('\\t'.join([identifier,unit,commit])+'\\n'); stream.flush(); os.fsync(stream.fileno())
print(identifier)
"""
    identifier = remote_python(script, REMOTE_RUN, UNIT, COMMIT).strip().splitlines()[-1]
    if not identifier.isdecimal():
        raise ValueError("the remote submission did not return an exact job identifier")
    ledger = LOCAL_RUN / "slurm/submitted_jobids_aces.txt"
    known = ledger.read_text().split() if ledger.exists() else []
    if identifier not in known:
        with ledger.open("a") as stream:
            stream.write(identifier + "\n")
            stream.flush()
        with (LOCAL_RUN / "slurm/submissions_aces.tsv").open("a") as stream:
            stream.write("\t".join([identifier, UNIT, COMMIT, REMOTE_RUN]) + "\n")
            stream.flush()
    (PREPARATION / "submitted_jobid.record").write_text(identifier + "\n")
    print(json.dumps({"job_id": identifier, "unit": UNIT, "remote_run": REMOTE_RUN,
                      "source_commit": COMMIT, "classification": "resource_probe_only"}), flush=True)
    print(remote_python("import subprocess,sys\np=subprocess.Popen(['squeue','-h','-j',sys.argv[1],'-o','%i|%T|%N|%b|%R'],stdout=subprocess.PIPE,universal_newlines=True)\no,_=p.communicate()\nprint(o)\nsys.exit(p.returncode)\n", identifier))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "stage", "submit"])
    arguments = parser.parse_args()
    {"prepare": prepare, "stage": stage, "submit": submit}[arguments.action]()
