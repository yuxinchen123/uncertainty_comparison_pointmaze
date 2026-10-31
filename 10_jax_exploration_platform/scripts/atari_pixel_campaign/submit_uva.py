"""Submit exact recorded study jobs using an immutable committed source snapshot."""

import argparse
import fcntl
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A second submission must never observe a partly extracted source tree.
    with (destination.parent / (commit + ".lock")).open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not destination.exists():
            with tempfile.TemporaryDirectory(dir=destination.parent,
                                              prefix="building_" + commit[:12] + "_") as temporary:
                staging = Path(temporary) / "source"
                staging.mkdir()
                with tempfile.TemporaryFile() as archive:
                    subprocess.run(["git", "-C", str(repository), "archive", commit,
                                    "10_jax_exploration_platform"], stdout=archive, check=True)
                    archive.seek(0)
                    with tarfile.open(fileobj=archive) as members:
                        members.extractall(staging, filter="data")
                staging.rename(destination)
    return destination / "10_jax_exploration_platform", commit


def submit(repository, run_directory, kind, test_filter=None):
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
              # Check visibility on the compute host before opening new NFS files.
              f"for attempt in {{1..60}}; do test -r {shlex.quote(str(environment))} && break; sleep 2; done",
              f"test -r {shlex.quote(str(environment))}",
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
            selection = " -k " + shlex.quote(test_filter) if test_filter else ""
            body = [f"{pixel_python} {source_quote}/scripts/atari_pixel_campaign/run_checks.py "
                    f"--run-dir {run_quote} --unit-id jaguar03_pixel_gates_{commit[:12]} "
                    f"--source-commit {commit} -- {pixel_python} -m pytest -q{selection} "
                    f"{source_quote}/tests/atari_pixels {source_quote}/tests/test_atari_pixel_environment.py"]
        elif kind == "rendering":
            body = [f"{pixel_python} {source_quote}/scripts/atari_pixel_campaign/run_checks.py "
                    f"--run-dir {run_quote} --unit-id jaguar03_rendering_{commit[:12]} "
                    f"--source-commit {commit} -- {pixel_python} "
                    f"{source_quote}/scripts/atari_pixel_campaign/check_rendering.py --candidate all"]
        elif kind == "configuration-gates":
            body = ["unset XDG_DATA_HOME",
                    f"{pixel_python} {source_quote}/scripts/atari_pixel_campaign/run_checks.py "
                    f"--run-dir {run_quote} --unit-id configuration_files_{commit[:12]} "
                    f"--source-commit {commit} -- {pixel_python} "
                    f"{source_quote}/tests/integration/test_training_configuration_file_cli.py",
                    f"{pixel_python} {source_quote}/tests/integration/test_selected_run_records.py"]
        elif kind == "smoke":
            body = [f"{pixel_python} {source_quote}/scripts/atari_pixel_campaign/run_unit.py "
                    f"--run-dir {run_quote} --unit-id jaguar03_pixel_smoke "
                    "--mode smoke --copies 1 --envs-per-copy 2 --rollout-steps 2 "
                    "--epochs 1 --minibatches 1 --warmup-rollouts 1 --iterations 2"]
        elif kind == "baseline":
            body = [f"{pixel_python} {source_quote}/scripts/atari_pixel_campaign/run_unit.py "
                    f"--run-dir {run_quote} --unit-id jaguar03_pixel_reference_copies_1 "
                    "--mode benchmark --copies 1 --rounds 7 --iterations 10"]
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
                        choices=["monitor", "merge-gates", "pixel-gates", "smoke", "baseline",
                                 "rendering", "configuration-gates"])
    parser.add_argument("--test-filter")
    args = parser.parse_args()
    submit(args.repo.resolve(), args.run_dir.resolve(), args.kind, args.test_filter)
