"""Launch one recorded serval02 card-1 unit under its shared exclusive lock."""

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys


def launch(repository, directory, mode, copies):
    """Publish a committed snapshot and retain the remote supervisor's exact process ID."""
    module_path = Path(__file__).with_name("submit_uva.py")
    specification = importlib.util.spec_from_file_location("atari_submission", module_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    source, commit = module.source_snapshot(repository, directory)
    unit = ("serval02_pixel_" + mode + "_copies_" + str(copies)
            + "_source_" + commit[:12])
    scripts = directory / "slurm" / "direct_jobs"
    scripts.mkdir(parents=True, exist_ok=True)
    script = scripts / (unit + "_" + commit[:12] + ".sh")
    python = "/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"
    worker = source / "scripts" / "atari_pixel_campaign" / "worker_environment.sh"
    runner = source / "scripts" / "atari_pixel_campaign" / "run_unit.py"
    if mode in ("gates", "rendering"):
        checker = source / "scripts" / "atari_pixel_campaign" / "run_checks.py"
        command = ([python, "-m", "pytest", "-q", str(source / "tests" / "atari_pixels"),
                    str(source / "tests" / "test_atari_pixel_environment.py")]
                   if mode == "gates" else
                   [python, str(source / "scripts" / "atari_pixel_campaign" / "check_rendering.py"),
                    "--candidate", "all"])
        arguments = [python, str(checker), "--run-dir", str(directory), "--unit-id", unit,
                     "--source-commit", commit, "--", *command]
    else:
        arguments = [python, str(runner), "--run-dir", str(directory), "--unit-id", unit,
                     "--mode", mode, "--copies", str(copies)]
    if mode == "smoke":
        arguments += ["--envs-per-copy", "2", "--rollout-steps", "2", "--epochs", "1",
                      "--minibatches", "1", "--warmup-rollouts", "1", "--iterations", "2"]
    gpu = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
    text = [
        "#!/bin/bash", "set -euo pipefail",
        "exec 9>/tmp/sl5nw_serval02_card1.lock", "flock -w 86400 9",
        f"for attempt in {{1..90}}; do test -r {shlex.quote(str(worker))} && break; sleep 2; done",
        f"source {shlex.quote(str(worker))}",
        f"export CUDA_VISIBLE_DEVICES={gpu}",
        f"/usr/bin/python3 - {shlex.quote(gpu)} <<'PY'",
        "import subprocess,sys",
        "gpu=sys.argv[1]",
        "settings=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=mig.mode.current,memory.used','--format=csv,noheader,nounits'],text=True)",
        "mig,memory=[part.strip() for part in settings.strip().split(',')]",
        "processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)",
        "confidential=subprocess.check_output(['nvidia-smi','conf-compute','-f'],text=True)",
        "if mig!='Disabled' or int(memory)>=1024 or gpu in processes or 'CC status: OFF' not in confidential:",
        " raise RuntimeError('serval02 card 1 is not free in the required GPU mode')",
        "print('Verified free serval02 card 1; shared lock held',flush=True)", "PY",
        shlex.join(arguments)]
    script.write_text("\n".join(text) + "\n")
    log = scripts / (unit + ".log")
    command = ("nohup bash " + shlex.quote(str(script)) + " >" + shlex.quote(str(log))
               + " 2>&1 < /dev/null & echo $!")
    result = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o",
                             "ConnectTimeout=20", "serval02", command],
                            capture_output=True, text=True, check=True)
    identifier = result.stdout.strip().splitlines()[-1]
    if not identifier.isdecimal():
        raise RuntimeError(f"remote launcher returned an unexpected PID: {result.stdout!r}")
    record = {"host": "serval02", "pid": int(identifier), "unit": unit,
              "source_commit": commit, "log": str(log), "script": str(script),
              "gpu_uuid": gpu}
    ledger = directory / "processes.json"
    records = json.loads(ledger.read_text()) if ledger.exists() else []
    records.append(record)
    ledger.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "benchmark", "gates", "rendering"], required=True)
    parser.add_argument("--copies", type=int, default=1)
    args = parser.parse_args()
    launch(args.repo.resolve(), args.run_dir.resolve(), args.mode, args.copies)
