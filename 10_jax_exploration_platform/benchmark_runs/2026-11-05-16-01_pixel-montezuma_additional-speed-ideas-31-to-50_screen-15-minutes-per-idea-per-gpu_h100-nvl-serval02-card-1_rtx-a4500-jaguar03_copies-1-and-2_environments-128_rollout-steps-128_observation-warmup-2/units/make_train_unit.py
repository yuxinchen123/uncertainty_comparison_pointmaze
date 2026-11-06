"""Write one unit that runs train_final.py commands (and optional comparison commands).

Usage: make_train_unit.py <gpu> <name> <description> <body file>
The body file holds bash lines; $PY, $TRAIN, $COMPARE and $DATA are defined for it.
gpu is a4500-jaguar03, h100-serval02-card-1 or h100-serval05. Prints the unit path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_unit import CHECKS, PLATFORM, PYTHON, RUN, SBATCH

gpu, name, description, body_file = sys.argv[1:5]
unit = f"{gpu}_{name}"
output = RUN / "data" / unit
body = Path(body_file).read_text()
script = f"""#!/bin/bash
# Unit: {description}. At most 15 minutes (user rule 2026-11-05).
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=false JAXTARI_SKIP_SPRITE_UPDATE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
{CHECKS[gpu]}
export JAX_COMPILATION_CACHE_DIR="$CACHE"
PY={PYTHON}
TRAIN="$PY -u {PLATFORM}/scripts/atari_pixel_campaign/train_final.py"
COMPARE="$PY {PLATFORM}/scripts/atari_pixel_campaign/compare_resumed.py"
DATA={output}
mkdir -p "$CACHE" $DATA
cd {PLATFORM}
export PYTHONPATH={PLATFORM}/src
echo "start $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') host $(hostname) pid $$" >> $DATA/unit.log
(
{body}
) >> $DATA/unit.log 2>&1
rc=$?
echo "end $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') rc $rc" >> $DATA/unit.log
exit $rc
"""
path = RUN / "units" / f"{unit}.sh"
if path.exists():
    raise SystemExit(f"{path} exists")
path.write_text(script)
path.chmod(0o755)
if gpu == "a4500-jaguar03":
    jobs = RUN / "slurm" / "jobs" / unit
    jobs.mkdir(parents=True, exist_ok=False)
    (jobs / "job.sbatch").write_text(SBATCH.format(jobs=jobs, unit=path))
print(path)
