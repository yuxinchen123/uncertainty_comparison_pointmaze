"""Write one unit script (and, for the A4500, its Slurm job) from a short specification.

Usage: make_unit.py <gpu> <slug> <copies> <name> [--learner JSON] [--environment JSON]
       [--reference-learner JSON] [--reference-environment JSON] [--profile] [--async-allocator]
gpu is a4500-jaguar03, h100-serval02-card-1 or h100-serval05. Prints the unit path.
"""
import argparse
import shlex
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
PLATFORM = RUN.parents[1]
PYTHON = "/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"
CHECKS = {
    "h100-serval02-card-1": '''U=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3
read MIG USED <<<"$(nvidia-smi -i $U --query-gpu=mig.mode.current,memory.used --format=csv,noheader,nounits | tr -d " " | tr "," " ")"
PROCS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$U")
CC=$(nvidia-smi conf-compute -f | grep -c "CC status: OFF")
if ! { [ "$MIG" = Disabled ] && [ "$USED" -lt 1024 ] && [ "$PROCS" -eq 0 ] && [ "$CC" -eq 1 ]; }; then echo "card 1 busy: mig=$MIG used=$USED procs=$PROCS cc=$CC"; exit 3; fi
export CUDA_VISIBLE_DEVICES=$U
CACHE=/tmp/sl5nw_pixel_screen_cache_h100''',
    "h100-serval05": '''if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; then echo "serval05 card busy"; exit 3; fi
CACHE=/tmp/sl5nw_pixel_screen_cache_h100''',
    "a4500-jaguar03": "CACHE=/tmp/sl5nw_pixel_screen_cache_a4500",
}
SBATCH = """#!/bin/bash
#SBATCH --job-name=pixel-screen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --nodelist=jaguar03
#SBATCH --account=sl5nw_base
#SBATCH --qos=csresnolim
#SBATCH --reservation=sl5nw_168
#SBATCH --gres=gpu:nvidia_rtx_a4500:1
#SBATCH --no-requeue
#SBATCH --output={jobs}/%j.out
bash {unit}
"""


def main():
    parser = argparse.ArgumentParser()
    for name in ("gpu", "slug", "copies", "name"):
        parser.add_argument(name)
    for name in ("--learner", "--environment", "--reference-learner", "--reference-environment"):
        parser.add_argument(name, default="{}")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--async-allocator", action="store_true")
    args = parser.parse_args()
    unit = f"{args.gpu}_{args.slug}_copies-{args.copies}"
    output = RUN / "data" / unit
    allocator = (" XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_MEM_FRACTION=0.85"
                 if args.async_allocator else "")
    options = " ".join(f"--{key} {shlex.quote(getattr(args, key.replace('-', '_')))}"
                       for key in ("learner", "environment", "reference-learner", "reference-environment"))
    script = f"""#!/bin/bash
# Unit: {args.name}, {args.copies} copies, {args.gpu}. At most 15 minutes (user rule 2026-11-05).
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=false JAXTARI_SKIP_SPRITE_UPDATE=1 XDG_DATA_HOME=/p/rlprojects/RND/reference_repos/jaxtari_pixel_study_assets_montezuma_room_geometry_from_png{allocator}
{CHECKS[args.gpu]}
mkdir -p "$CACHE" {output}
cd {PLATFORM}
echo "start $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') host $(hostname) pid $$" >> {output}/unit.log
timeout --signal=TERM --kill-after=30 900 {PYTHON} -u scripts/atari_pixel_campaign/screen_idea.py \\
  --name {shlex.quote(args.name)} {options} --copies {args.copies}{' --profile' if args.profile else ''}{' --timing-only' if args.timing_only else ''} \\
  --budget-seconds 840 --observation-warmup-iterations 2 \\
  --output {output} --cache-directory "$CACHE" >> {output}/unit.log 2>&1
rc=$?
echo "end $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') rc $rc" >> {output}/unit.log
exit $rc
"""
    path = RUN / "units" / f"{unit}.sh"
    if path.exists():
        raise SystemExit(f"{path} exists")
    path.write_text(script)
    path.chmod(0o755)
    if args.gpu == "a4500-jaguar03":
        jobs = RUN / "slurm" / "jobs" / unit
        jobs.mkdir(parents=True, exist_ok=False)
        (jobs / "job.sbatch").write_text(SBATCH.format(jobs=jobs, unit=path))
    print(path)


if __name__ == "__main__":
    main()
