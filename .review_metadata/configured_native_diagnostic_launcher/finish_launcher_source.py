from pathlib import Path
p=Path(__file__).parent/'launch_diagnostic_once.py'
s=p.read_text()
needle='def prepared_records():\n'
insert='''def expected_job_script():
    """Return the exact reviewed one-card wrapper; it performs only the staged diagnostic."""
    lines = ["#!/bin/bash", "#SBATCH --job-name=selected-python-native-process", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1", "#SBATCH --mem=2G", "#SBATCH --time=00:05:00", "#SBATCH --partition=gpu",
        "#SBATCH --nodelist=jaguar03", "#SBATCH --account=sl5nw_base", "#SBATCH --reservation=sl5nw_168",
        "#SBATCH --qos=csresnolim", "#SBATCH --gres=gpu:nvidia_rtx_a4500:1", "#SBATCH --no-requeue",
        "#SBATCH --output=" + str(RUN / "slurm/jobs/native-process-observation-%j.out"),
        "#SBATCH --error=" + str(RUN / "slurm/jobs/native-process-observation-%j.err"), "set -euo pipefail", "umask 077",
        "unset PYTHONPATH PYTHONHOME", "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1",
        "cd " + shlex.quote(str(RUN)), "sha256sum --check --strict " + shlex.quote(str(PACKET / "source_checksums.sha256")),
        "exec " + shlex.join(["srun", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--cpu-bind=cores",
            "--gres=gpu:nvidia_rtx_a4500:1", "--kill-on-bad-exit=1", PYTHON, "-I", "-S", "-B",
            str(UNIT / "code/diagnose_configured_native_process.py"), "--prepared-unit", str(UNIT)])]
    return ("\\n".join(lines) + "\\n").encode()


'''
s=s.replace(needle,insert+needle)
s=s.replace('    plan_path = UNIT / "measurement_plan.json";plan = document(plan_path)', '    require(ordinary_bytes(PACKET / "job.sbatch") == expected_job_script(), "the exact staged diagnostic Slurm wrapper changed")\n    plan_path = UNIT / "measurement_plan.json";plan = document(plan_path)')
s=s.replace('    paths = dict(freeze["files"])', '    require(type(freeze) is dict and type(freeze.get("files")) is dict, "the exact launcher publication roster is required")\n    static = document(PACKET / "packet_static_verification.json")\n    require(static.get("passed") is True and static.get("prepared_binding") == reference(PACKET / "prepared_launch_binding.json"),\n        "the exact prepared binding has no static packet verification")\n    paths = dict(freeze["files"])')
s=s.replace('    fields = dict(re.findall(r"(?:^|\\s)([A-Za-z][A-Za-z0-9_/:]*)=(\\S*)", result.stdout.decode()))', '    pairs = re.findall(r"(?:^|\\s)([A-Za-z][A-Za-z0-9_/:]*)=(\\S*)", result.stdout.decode())\n    require(len({key for key, unused in pairs}) == len(pairs), "duplicate native held-job fields are not accepted")\n    fields = dict(pairs)')
s=s.replace('    resources = dict(item.split("=", 1) for item in fields["ReqTRES"].split(","))', '    resource_pairs = [item.split("=", 1) for item in fields["ReqTRES"].split(",")]\n    require(all(len(pair) == 2 for pair in resource_pairs) and len({pair[0] for pair in resource_pairs}) == len(resource_pairs),\n        "duplicate or malformed requested resources are not accepted")\n    resources = dict(resource_pairs)')
s=s.replace('and resources.get("mem") == "2G" and resources.get("gres/gpu") == "1"','and resources.get("mem") == "2G" and resources.get("node") == "1" and resources.get("gres/gpu") == "1"')
s=s.replace('        job = identifier(parsed.group(1)); row = submission_row(job, plan)', '        job = identifier(parsed.group(1)); row = submission_row(job, plan)\n        require(not any(line.split("\\t")[0] == job for line in ordinary_bytes(RUN / "slurm/submissions.tsv").decode().splitlines()),\n            "the returned identifier already has a campaign submission row")')
p.write_text(s)
