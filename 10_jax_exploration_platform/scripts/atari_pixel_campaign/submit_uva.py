"""Submit exact recorded study jobs using an immutable committed source snapshot."""

import argparse
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile


def source_snapshot(repository, run_directory):
    """Extract the current committed code without including uncommitted agent edits."""
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"],
                                     text=True).strip()
    destination = run_directory / "data" / "code_snapshots" / commit
    if not destination.exists():
        destination.mkdir(parents=True)
        with tempfile.TemporaryFile() as archive:
            subprocess.run(["git", "-C", str(repository), "archive", commit,
                            "10_jax_exploration_platform"], stdout=archive, check=True)
            archive.seek(0)
            with tarfile.open(fileobj=archive) as members:
                members.extractall(destination, filter="data")
    return destination / "10_jax_exploration_platform", commit


def submit(repository, run_directory, kind):
    """Write a complete batch script and immediately record the exact returned job ID."""
    source, commit = source_snapshot(repository, run_directory)
    slurm = run_directory / "slurm"
    jobs = slurm / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    source_quote, run_quote = shlex.quote(str(source)), shlex.quote(str(run_directory))
    interpreter = "/p/rlprojects/RND/.venvs/platform_jax/bin/python"
    pixel_python = "/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"
    environment = source / "scripts" / "atari_pixel_campaign" / "worker_environment.sh"
    header = ["#!/bin/bash", "set -euo pipefail",
              f"export PYTHONPATH={source_quote}/src",
              f"source {shlex.quote(str(environment))}"]
    if kind == "monitor":
        resources = ["-p", "nolim", "--cpus-per-task=1", "--mem=2G",
                     "--gpus-per-node=0", "--time=7-00:00:00"]
        body = [f"/usr/bin/python3 {source_quote}/scripts/atari_pixel_campaign/monitor.py "
                f"--run-dir {run_quote} --loop"]
    else:
        # The initial gate estimate is bounded by a single policy and small fake rollouts.
        reservation = subprocess.check_output(
            ["scontrol", "show", "node", "jaguar03"], text=True)
        token = next((value.split("=", 1)[1] for value in reservation.split()
                      if value.startswith("ReservationName=")), "")
        resources = ["-p", "gpu", "--nodelist=jaguar03",
                     "--gres=gpu:nvidia_rtx_a4500:1", "--cpus-per-task=8",
                     "--mem=24G", "--time=06:00:00"]
        if token and token != "(null)":
            resources += ["--reservation=" + token, "--qos=csresnolim"]
        if kind == "merge-gates":
            tests = ["agents/test_configuration_presets.py",
                     "agents/test_intrinsic_reset_return.py",
                     "agents/test_no_bonus_task_advantages.py",
                     "agents/test_reward_decline_switches.py",
                     "integration/test_selected_run_records.py",
                     "envs/test_antmaze_compose.py", "envs/test_montezuma_compose.py"]
            body = ["unset XDG_DATA_HOME"] + [
                f"{interpreter} {source_quote}/tests/{test}" for test in tests]
        elif kind == "pixel-gates":
            body = [f"{pixel_python} -m pytest -q {source_quote}/tests/atari_pixels "
                    f"{source_quote}/tests/test_atari_pixel_environment.py"]
        elif kind == "smoke":
            body = [f"{pixel_python} {source_quote}/scripts/atari_pixel_campaign/run_unit.py "
                    f"--run-dir {run_quote} --unit-id jaguar03_pixel_smoke "
                    "--mode smoke --copies 1 --envs-per-copy 2 --rollout-steps 2 "
                    "--epochs 1 --minibatches 1 --warmup-rollouts 1 --iterations 2"]
        else:
            raise ValueError(f"unsupported job kind {kind}")
    script = jobs / f"{kind}_{commit[:12]}.sh"
    script.write_text("\n".join(header + body) + "\n")
    command = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1",
               "--job-name=atari_" + kind.replace("-", "_"),
               "--output=" + str(jobs / f"{kind}_%j.log"), *resources, str(script)]
    identifier = subprocess.check_output(command, text=True).strip().split(";")[0]
    if not identifier.isdecimal():
        raise RuntimeError(f"sbatch returned an unexpected identifier {identifier!r}")
    with (slurm / "submitted_jobids_uva.txt").open("a") as ledger:
        ledger.write(identifier + "\n")
        ledger.flush()
    with (slurm / "submissions.tsv").open("a") as ledger:
        ledger.write("\t".join((identifier, kind, commit, str(script))) + "\n")
    print(f"Submitted {kind}: {identifier} from {commit}", flush=True)
    return identifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--kind", required=True,
                        choices=["monitor", "merge-gates", "pixel-gates", "smoke"])
    args = parser.parse_args()
    submit(args.repo.resolve(), args.run_dir.resolve(), args.kind)
