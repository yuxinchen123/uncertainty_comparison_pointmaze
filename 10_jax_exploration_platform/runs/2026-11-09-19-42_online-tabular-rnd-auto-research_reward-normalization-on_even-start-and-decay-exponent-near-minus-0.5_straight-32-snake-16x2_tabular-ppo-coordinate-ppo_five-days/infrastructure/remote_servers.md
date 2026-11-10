# Remote servers: environment and code for whole training units

Built 2026-11-09 between 20:05 PT and 20:52 PT. Times are Pacific, converted from servers that run on Eastern
(Purdue, MIT) and Central (ACES, DeltaAI). No speed test, throughput measurement, benchmark, canary or training was run on
any of these servers (user rule 2026-11-09: their speed is estimated from the serval05 measurements). Each build job
only installed packages and then imported the study's modules.

Nine build jobs were submitted (Gautschi 2, Gilbreth 2, ACES 3, DeltaAI 1, MIT 1) and all nine had ended by
20:52 PT; none is queued or running, and no job was cancelled. The agent that built this had no recurring-task tool, so it watched
the jobs itself until each ended and left no monitor behind.

## What was copied to every server

1. **Code**: the snapshot
   `code/source_9cd855bb4d2c2da5399974396a9183c58fba890a/` of this run folder (commit 9cd855bb), copied with
   `rsync -a --exclude __pycache__`. 171 files. The SHA-256 of the sorted per-file SHA-256 list is
   `cdef1ed94a6e4e2c6f439dcf4b69cc7aa0c028f8c4820d1d56336bf1ca463d1d` on UVA and on all four servers.
2. **Pin list** `requirements_rnd_platform_jax_pinned.txt`: the 21 packages of
   `/p/rlprojects/RND/.venvs/platform_jax/pip_freeze.txt` (jax, jaxlib, jax-cuda12-plugin and jax-cuda12-pjrt 0.11.0,
   numpy 2.5.2, ml_dtypes 0.6.0, opt_einsum 3.4.0, scipy 1.18.0, and the 13 nvidia-*-cu12 libraries at the UVA
   versions). The plotting, mujoco and Atari packages of the UVA environment were not installed; the training code
   does not import them.
3. **Build script** `build_rnd_platform_jax_env.sh` (text in the appendix). It creates Python 3.12.13 from
   conda-forge with `conda create --copy`, installs the pin list with `pip install --only-binary=:all:`, requires the
   installed set to equal the pin list exactly (pip, setuptools, wheel and packaging, which come with conda-forge's
   Python, are not compared; the UVA environment has the same four), runs `pip check`, writes
   `<environment>.pip_freeze.txt`, and runs the import check.
4. **Job script** `build_job_<server>.slurm`, which sets the server's paths and calls the build script.

Items 2 to 4 are in `<base>/rnd_grid/setup/` on each server, with the job logs in `<base>/rnd_grid/setup/logs/`.

The import check prints the library versions and `jax.devices()`, then imports
`exploration_platform.grid_initialization.{config, environments, fields, online_rnd, training}` and, from
`scripts/grid_initialization_campaign/`, `common`, `initial_field_archive` and `train_unit`.

## MIT (user shuzel)

- **Ready** (processor import check; no card was used).
- At about 20:05 PT there was no control connection to MIT and nothing was attempted. At 20:41 PT the user's
  connection was open on portal11; every MIT command since then went through it
  (`ssh portal11 'ssh -o BatchMode=yes mit ...'`, each after `ssh portal11 'ssh -O check mit'` answered
  "Master running"). No new MIT login was started.
- Environment: `/orcd/scratch/orcd/008/shuzel/envs/rnd_platform_jax` (the same folder as
  `~/orcd/scratch/envs/rnd_platform_jax`); package list beside it in `rnd_platform_jax.pip_freeze.txt`.
- Code: `/orcd/scratch/orcd/008/shuzel/rnd_grid/code/source_9cd855bb4d2c2da5399974396a9183c58fba890a/10_jax_exploration_platform/`
  (171 files, the same checksum as on UVA).
- Conda: `/orcd/software/core/001/pkg/miniforge/26.7.2-0/bin/conda` (module `miniforge/26.7.2-0`).
- Job (partition `mit_normal`, free tier, 4 cores, 16 GB; id in `../slurm/submitted_jobids_mit.txt`): 28556857, node
  node3111, submitted 20:46 PT, started 20:48 PT, COMPLETED in 3 min 44 s. The compute node reached the internet
  without a proxy.
- Import check of job 28556857 (a processor node, so the CUDA plugin finds no card and jax uses the processor; the
  plugin's `CUDA_ERROR_NO_DEVICE` traceback is printed before it):

  ```text
  installed set matches all 21 pins
  python 3.12.13 at /orcd/scratch/orcd/008/shuzel/envs/rnd_platform_jax/bin/python
  jax 0.11.0 numpy 2.5.2
  jaxlib 0.11.0
  jax-cuda12-plugin 0.11.0
  jax-cuda12-pjrt 0.11.0
  default backend: cpu
  devices: [CpuDevice(id=0)]
  study modules imported: grid_initialization.{config,environments,fields,online_rnd,training}, common, initial_field_archive, train_unit
  ```

- Scratch on MIT is purged after 6 months without use, and the scratch space was 74% full (760 of 1,024 GB) when
  the environment was built.

## Purdue Gautschi (account dsimchil)

- **Ready** (processor import check; the H100 was not used).
- Environment: `/scratch/gautschi/liu4869/envs/rnd_platform_jax`; package list beside it in
  `/scratch/gautschi/liu4869/envs/rnd_platform_jax.pip_freeze.txt`.
- Code: `/scratch/gautschi/liu4869/rnd_grid/code/source_9cd855bb4d2c2da5399974396a9183c58fba890a/10_jax_exploration_platform/`.
- Conda: `/apps/external/conda/2026.03/bin/conda` (what `module load conda` puts on the path).
- Jobs (partition `cpu`, `-q standby`, 4 cores, 7.7 GB, ids in `../slurm/submitted_jobids_gautschi.txt`):
  1. 18177307, node a065, started 20:11 PT, FAILED after 2 min 44 s: the installed set had `packaging==26.3`
     beyond the pin list (conda-forge's pip brings it). The script was changed to leave packaging out of the
     comparison, the environment was deleted, and the build was submitted again.
  2. 18177407, node a061, started 20:18 PT, COMPLETED in 2 min 54 s.
- Import check of job 18177407 (a processor node, so the CUDA plugin finds no card and jax uses the processor; the
  plugin's `CUDA_ERROR_NO_DEVICE` traceback is printed before it):

  ```text
  installed set matches all 21 pins
  python 3.12.13 at /scratch/gautschi/liu4869/envs/rnd_platform_jax/bin/python
  jax 0.11.0 numpy 2.5.2
  jaxlib 0.11.0
  jax-cuda12-plugin 0.11.0
  jax-cuda12-pjrt 0.11.0
  default backend: cpu
  devices: [CpuDevice(id=0)]
  study modules imported: grid_initialization.{config,environments,fields,online_rnd,training}, common, initial_field_archive, train_unit
  ```

- Later H100 jobs: partition `ai`, `-q normal`, `-A dsimchil`. The first one should confirm that `jax.devices()`
  lists a `CudaDevice`, since this build ran on a processor node.

## Purdue Gilbreth (account dsmichil)

- **Ready** (import check on an A10 card).
- Environment: `/scratch/gilbreth/liu4869/envs/rnd_platform_jax`; package list
  `/scratch/gilbreth/liu4869/envs/rnd_platform_jax.pip_freeze.txt`.
- Code: `/scratch/gilbreth/liu4869/rnd_grid/code/source_9cd855bb4d2c2da5399974396a9183c58fba890a/10_jax_exploration_platform/`.
- Conda: `/apps/external/conda/2025.09/bin/conda`.
- **The Slurm account on Gilbreth is `dsmichil`, not `dsimchil`.** The first `sbatch` with `-A dsimchil` was refused
  at submit ("Invalid account or account/partition combination specified"; no job id); `slist` and
  `sacctmgr show assoc user=liu4869` both show `dsmichil` (QOS normal, standby, training). The rule file
  `purdue-cluster-slurm.md` still says `dsimchil` for Gilbreth.
- Jobs (`-q standby`, partitions `a10,a30,a100-80gb`, 4 cores, 16 GB, one card, `LD_LIBRARY_PATH` cleared; ids in
  `../slurm/submitted_jobids_gilbreth.txt`):
  1. 12304152, partition a10, node gilbreth-h015, started 20:11 PT, FAILED after 2 min 27 s: the same extra
     `packaging==26.3`; environment deleted and submitted again.
  2. 12304162, partition a10, node gilbreth-h015, started 20:14 PT, COMPLETED in 2 min 37 s.
- Import check of job 12304162:

  ```text
  installed set matches all 21 pins
  E1109 23:17:12.921859 1148521 numa_hwloc.cc:121] Call to hwloc_set_cpubind() failed: Invalid argument [22]
  python 3.12.13 at /scratch/gilbreth/liu4869/envs/rnd_platform_jax/bin/python
  jax 0.11.0 numpy 2.5.2
  jaxlib 0.11.0
  jax-cuda12-plugin 0.11.0
  jax-cuda12-pjrt 0.11.0
  default backend: gpu
  devices: [CudaDevice(id=0)]
  study modules imported: grid_initialization.{config,environments,fields,online_rnd,training}, common, initial_field_archive, train_unit
  ```

  The `hwloc_set_cpubind` line is jax failing to pin its threads to the job's processor set; it did not stop the
  import or the card listing.
- At the time of the build our one `normal` A100-40GB was held by another job of the account (`slist`: 1 running,
  1 queued). Our own A100-40GB jobs use `-p a100-40gb -q normal -A dsmichil`.

## TAMU ACES (account 158590754315)

- **Ready** (processor import check; the H100 was not used).
- Environment: `/scratch/user/u.sl359075/envs/rnd_platform_jax`; package list
  `/scratch/user/u.sl359075/envs/rnd_platform_jax.pip_freeze.txt`.
- Code: `/scratch/user/u.sl359075/rnd_grid/code/source_9cd855bb4d2c2da5399974396a9183c58fba890a/10_jax_exploration_platform/`.
- Reached only through the connection open on portal05 (`ssh portal05 'ssh aces ...'`; `ssh -O check aces` on
  portal05 answered "Master running" before the work and again at 20:36 PT). The code went over with
  `rsync -a -e "ssh portal05 ssh" ...`. No certificate was renewed.
- Conda: `/sw/eb/sw/Miniforge3/26.3.2-2/bin/conda`.
- **ACES compute nodes have no direct internet.** The job script starts with `#!/bin/bash -l` and runs
  `module load WebProxy/0000`, which sets `http_proxy` and `https_proxy` to `http://10.71.8.1:8080`. The shared
  `CLAUDE.md` says every compute node on every server has outbound internet; on ACES that holds only with this module.
- Jobs (partition `cpu`, 4 cores, 16 GB; ids in `../slurm/submitted_jobids_aces.txt`):
  1. 2330985, node ac008, started 20:10 PT, FAILED after 36 s: `conda create` could not reach
     `conda.anaconda.org` ("Network is unreachable"). Nothing was installed. The web proxy was added to the job
     script.
  2. 2330988, node ac001, started 20:12 PT, FAILED after 11 min 9 s: the packages installed, then the comparison
     found `packaging==26.3` beyond the pin list (this job started before the script was changed). The environment
     was deleted and the build was submitted again.
  3. 2331041, node ac053, started 20:24 PT, COMPLETED in 12 min 4 s.
- The three jobs together ran 23 min 49 s on 4 cores, about 1.6 SUs.
- Import check of job 2331041 (a processor node, so the CUDA plugin finds no card and jax uses the processor; the
  plugin's `CUDA_ERROR_NO_DEVICE` traceback is printed before it):

  ```text
  installed set matches all 21 pins
  python 3.12.13 at /scratch/user/u.sl359075/envs/rnd_platform_jax/bin/python
  jax 0.11.0 numpy 2.5.2
  jaxlib 0.11.0
  jax-cuda12-plugin 0.11.0
  jax-cuda12-pjrt 0.11.0
  default backend: cpu
  devices: [CpuDevice(id=0)]
  study modules imported: grid_initialization.{config,environments,fields,online_rnd,training}, common, initial_field_archive, train_unit
  ```

- Later H100 jobs: partition `gpu`, `--gres=gpu:h100:1`, `--account=158590754315` (rule
  `tamu-aces-cluster-slurm.md`). The first one should confirm that `jax.devices()` lists a `CudaDevice`, since this
  build ran on a processor node.

## NCSA DeltaAI (account biwj-dtai-gh)

- **Ready** (import check on a GH200 card).
- Environment: `/projects/biwj/sliu51/envs/rnd_platform_jax` (aarch64); package list
  `/projects/biwj/sliu51/envs/rnd_platform_jax.pip_freeze.txt`.
- Code: `/projects/biwj/sliu51/rnd_grid/code/source_9cd855bb4d2c2da5399974396a9183c58fba890a/10_jax_exploration_platform/`.
- Reached only through the connection open on portal11 (`ssh portal11 'ssh deltaai ...'`; `ssh -O check deltaai`
  on portal11 answered "Master running"). The code went over with `rsync -a -e "ssh portal11 ssh" ...`.
- Conda: `/sw/user/python/miniforge3-cuquantum-26.01.0/bin/conda`. Before the build, PyPI was checked for every
  pinned version: each has a Linux aarch64 wheel for CPython 3.12 (manylinux 2.17 to 2.28; DeltaAI has glibc 2.38).
- Job (partition `ghx4`, 8 cores, 32 GB, one card, `LD_LIBRARY_PATH` cleared; id in
  `../slurm/submitted_jobids_deltaai.txt`): 3567239, node gh123, started 20:13 PT, COMPLETED in 3 min 19 s, so about
  0.06 GPU-hours were charged (balance before it: 66 of 100 hours). This job started after the packaging change, so
  it passed the first time.
- Import check of job 3567239:

  ```text
  installed set matches all 21 pins
  python 3.12.13 at /projects/biwj/sliu51/envs/rnd_platform_jax/bin/python
  jax 0.11.0 numpy 2.5.2
  jaxlib 0.11.0
  jax-cuda12-plugin 0.11.0
  jax-cuda12-pjrt 0.11.0
  default backend: gpu
  devices: [CudaDevice(id=0)]
  study modules imported: grid_initialization.{config,environments,fields,online_rnd,training}, common, initial_field_archive, train_unit
  ```

## Before placing units on these servers

1. `scripts/grid_initialization_campaign/common.py` fixes `PROJECT = /p/rlprojects/RND`,
   `PYTHON = /p/rlprojects/RND/.venvs/platform_jax/bin/python` and the UVA `ACCOUNTS` folder. `worker.py` starts
   each unit with that `PYTHON`, so on these servers a unit must be started with the server's own environment
   python (for example `<environment>/bin/python <code>/scripts/grid_initialization_campaign/train_unit.py ...`), or
   `PYTHON` must point at it. `train_unit.py` finds `src/` from its own location, so it needs no `PYTHONPATH`.
2. Every job script clears `LD_LIBRARY_PATH` (needed on Gilbreth and DeltaAI; harmless elsewhere) and sets
   `PYTHONNOUSERSITE=1`.
3. ACES compute nodes reach the internet only after `module load WebProxy/0000`. A job that downloads anything there
   needs it.
4. Gautschi, ACES and MIT were checked on processor nodes only. The fleet's worker job prints `jax.devices()` and
   ends without taking a unit unless jax uses the card. Gautschi's first worker job (18179686, node h010, 21:04 PT)
   printed `[CudaDevice(id=0)]` on an H100 80GB.
5. **Gilbreth's Slurm account is `dsmichil`** (`-A dsmichil`); `dsimchil`, the account on Gautschi and the name in
   `purdue-cluster-slurm.md`, is refused at submit on Gilbreth.
6. MIT is reached only through the user's connection on portal11; when it closes, nothing can be sent there
   until the user logs in again.
7. **Gautschi's `ai` partition refuses a job that does not ask for exactly 14 cores per card** (`-n14`; two
   submissions with `-n4` were refused at 21:02 PT and created no job).

## Worker fleet

Units are placed on these servers by `scripts/grid_initialization_campaign/remote_fleet.py`. Its server table is
`remote_fleet_servers.json` in this folder; the files it sends to each server are under `remote_fleet/`; every
decision is in `../slurm/fleet_log.txt`. Each server's run folder is `<base>/rnd_grid/runs/auto_research`. DeltaAI
is not in the table (not used now).

## Appendix: build script

`build_rnd_platform_jax_env.sh`, the same file on all four servers (the version with packaging left out of the
comparison):

```bash
#!/bin/bash
# Build the RND grid study's JAX environment on a remote server, inside a short Slurm job, then check that it imports.
# No training, timing or benchmark is run here.
#
# The job script that calls this sets:
#   SERVER    name written in the log (gautschi, gilbreth, aces, deltaai)
#   CONDA     full path of the server's conda program
#   ENV_PATH  where the new environment goes (must not exist yet)
#   CODE      the copied 10_jax_exploration_platform folder (holds src/ and scripts/grid_initialization_campaign/)
#   CACHE     a scratch folder for conda's package downloads (home folders are small)
# The pin list is read from the folder this script sits in.
set -euo pipefail
export PYTHONNOUSERSITE=1
# Module-provided CUDA libraries would be found before the pip-installed ones (Gilbreth, DeltaAI).
unset LD_LIBRARY_PATH
PINS=$(dirname "$(readlink -f "$0")")/requirements_rnd_platform_jax_pinned.txt

echo "server $SERVER, host $(hostname), job ${SLURM_JOB_ID:-none}, start $(date -Is)"
[ -f "$PINS" ] || { echo "no pin list at $PINS"; exit 1; }
[ ! -e "$ENV_PATH" ] || { echo "$ENV_PATH exists already; it is not rebuilt in place"; exit 1; }
[ -x "$CONDA" ] || { echo "no conda at $CONDA"; exit 1; }
[ -d "$CODE/src/exploration_platform" ] || { echo "no code at $CODE"; exit 1; }
export CONDA_PKGS_DIRS="$CACHE/conda_pkgs"
mkdir -p "$CONDA_PKGS_DIRS" "$(dirname "$ENV_PATH")"

# A self-contained Python 3.12.13 from conda-forge (--copy: no links into the package cache).
"$CONDA" create --copy --yes -p "$ENV_PATH" --override-channels -c conda-forge python=3.12.13
# The pinned packages, wheels only (no package is compiled here).
"$ENV_PATH/bin/python" -m pip install --no-cache-dir --only-binary=:all: -r "$PINS"

# The installed set must equal the pin list, name for name and version for version. pip, setuptools, wheel and
# packaging come with conda-forge's python and pip (the UVA environment has the same four) and are not compared.
# Before: "ml_dtypes==0.6.0" in the pins, "ml-dtypes==0.6.0" in pip's list; after: both "ml-dtypes==0.6.0".
normalize() { grep -vE '^\s*(#|$)' | sed -E 's/[[:space:]]+$//' \
              | awk -F'==' '{ n = tolower($1); gsub(/[-_.]+/, "-", n); print n "==" $2 }' \
              | grep -vE '^(pip|setuptools|wheel|packaging)==' | sort; }
want=$(normalize < "$PINS")
have=$("$ENV_PATH/bin/python" -m pip list --format=freeze | normalize)
"$ENV_PATH/bin/python" -m pip list --format=freeze > "$ENV_PATH.pip_freeze.txt"
if [ "$want" != "$have" ]; then
  echo "INSTALLED SET DIFFERS from the pins (< pinned, > installed):"
  diff <(echo "$want") <(echo "$have") || true
  exit 1
fi
"$ENV_PATH/bin/python" -m pip check
chmod -R g+rX "$ENV_PATH"
echo "installed set matches all $(echo "$want" | wc -l) pins"

# Import check: library versions, the devices jax sees, and the study's training modules.
cd "$CODE"
PYTHONPATH="$CODE/src:$CODE/scripts/grid_initialization_campaign" "$ENV_PATH/bin/python" - <<'EOF'
import importlib.metadata as md
import sys
import jax
import numpy
print("python", sys.version.split()[0], "at", sys.executable)
print("jax", jax.__version__, "numpy", numpy.__version__)
for name in ("jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt"):
    print(name, md.version(name))
print("default backend:", jax.default_backend())
print("devices:", jax.devices())
import exploration_platform.grid_initialization.config
import exploration_platform.grid_initialization.environments
import exploration_platform.grid_initialization.fields
import exploration_platform.grid_initialization.online_rnd
import exploration_platform.grid_initialization.training
import common
import initial_field_archive
import train_unit
print("study modules imported: grid_initialization.{config,environments,fields,online_rnd,training},",
      "common, initial_field_archive, train_unit")
EOF
echo "BUILD OK $SERVER $ENV_PATH $(date -Is)"
```
