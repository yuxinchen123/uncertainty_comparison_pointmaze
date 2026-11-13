# H100 and H200 servers for this folder's AntMaze units: setup notes

The user decided on 2026-11-13 that AntMaze training units may also run on the H100 and H200 servers, in the order
MIT, then Purdue Gautschi, then the ACCESS servers (TAMU ACES, then NCSA DeltaAI). A helper session prepared MIT
and Gautschi on 2026-11-13 between 11:56 and 12:30 PT. It handed over no unit, ran no `prepare` or `hand-over-tick`
for these servers, and submitted no worker job. Both new table entries say `ready: false`. Times are Pacific,
converted from the servers' Eastern clocks.

## State of the cards when the setup was done

| server | the account's cards | read at |
|---|---|---|
| MIT `mit_normal_gpu`, Advanced pool | 0 of 4 running; 8 jobs of another session (names `cs-...`, H200) waiting on `Priority` | 12:00 PT |
| MIT `mit_preemptable` | 4 of 4 H100 held by another session's `cs-...` jobs, 4 more of them waiting | 12:00 PT |
| Gautschi `ai` | 2 of 2 H100 held by another session's jobs 18517896 and 18517897 (`cs-...`), started 11:06 and 11:07 PT with a 9-hour limit, so they end by about 20:07 PT | 12:00 PT |

So, at that moment, neither server could start an AntMaze unit (rule `h100-h200-server-order.md`: a server whose
own caps are full cannot start the work, whatever cards are free). A waiting job of another session is never held,
moved or cancelled to make room.

## Purdue Gautschi `ai` (H100)

1. **Environment.** `/scratch/gautschi/liu4869/envs/rnd_platform_jax_mujoco_warp/bin/python`, built 2026-11-12 by
   the PointMaze baseline folder's `infrastructure/remote_fleet/setup/build_rnd_platform_jax_mujoco_warp_env.sh`
   and `add_jaxatari.sh`. Its package list (copy here: `remote_fleet/setup/pip_freeze_gautschi_rnd_platform_jax_mujoco_warp.txt`)
   equals UVA's `/p/rlprojects/RND/.venvs/platform_jax_warp` except `pip` 26.2.2 (UVA 26.2.1), `setuptools` 84.2.1
   (UVA 84.0.0) and `PyMuPDF` (absent; the trainer does not use it). Python 3.12.13, `jax` 0.11.0, `mujoco` and
   `mujoco-mjx` 3.11.0, `warp-lang` 1.14.0, `jaxatari` 0.1.0. It has never run on a card: the PointMaze folder's run
   folder on Gautschi has no job log.
2. **Import check, job 18521082** (`cpu`, `-q standby`, 2 cores, not charged; node a005; started 12:07:41 PT, ran
   17 s, exit 0). Script `remote_fleet/setup/import_check_rnd_platform_jax_mujoco_warp.py`, job script
   `remote_fleet/setup/import_check_processor_job_gautschi.sh`, output
   `remote_fleet/setup/import_check_gautschi_processor_job_18521082.out`. Imports only, with jax told to use the
   processor:

   ```text
   python 3.12.13 at /scratch/gautschi/liu4869/envs/rnd_platform_jax_mujoco_warp/bin/python
   jax==0.11.0 jaxlib==0.11.0 jax-cuda12-plugin==0.11.0 numpy==2.5.2 mujoco==3.11.0 mujoco-mjx==3.11.0
   warp-lang==1.14.0 flax==0.12.8 optax==0.2.8 jaxatari==0.1.0
   mujoco.mjx Warp implementation: warp installed True | mujoco_warp module loaded True
   imports took 12.9 s
   IMPORT_CHECK_PASSED processor
   ```

3. **No card check on an H100.** The rules allow an import check on a card, but the group's two H100 cards were
   both held by another session until about 20:07 PT, so a check job would have waited as long as a worker job, and
   it would have taken one of the account's four waiting places that gain age under `normal`. The first worker
   job checks the card itself before it claims a unit (the job script prints `jax devices` and stops with exit 3,
   claiming nothing, when jax does not use the card or a module the trainer imports is missing). This is the "one
   wait, not two" rule of skill `purdue-submit-gpu-jobs`. The same package list passed the card check on an MIT
   H100 (MIT, item 3); what Gautschi adds is its own driver and node environment, which the first worker job shows.
4. **Table entry `gautschi`** (added under the fleet lock at 12:06:54 PT; source
   `remote_fleet_servers_mit_gautschi_entries.json`):
   - `ssh`: `ssh -o BatchMode=yes -o ConnectTimeout=30 gautschi`; `python` as in item 1;
     `remote_run` `/scratch/gautschi/liu4869/rnd_maze/runs/2026-11-12-15-56_cleanrl-ppo-rnd-baseline-antmaze`;
     `code_root` `/scratch/gautschi/liu4869/rnd_maze/code`; `unit_seconds` 1800;
   - pool `ai-h100`: `-p ai -q normal -A dsimchil -N1 -n14 --gpus-per-node=1`, `time_limit` `1-00:00:00` (the QOS
     allows the partition's 4 days), `max_jobs` 2, `gpu_class` `h100_sxm_gautschi`; no `--mem` (14 cores bring
     128,800 MB);
   - `hand_over`: `units_at_once` 1, `match` `antmaze`, `pool` `ai-h100`, `start_within_seconds` 7200;
   - budget read with `slist` (H100-hours of the group's yearly balance; 1 H100-hour per card-hour under `normal`);
   - `ready: false`, with the reason that it waits for the session's freeze and first unit.

## MIT (H100 and H200)

MIT was reached only through the open connection on portal11 (`ssh -O check mit` there before every command; no
new login to `orcd-login.mit.edu`).

1. **Environment built:** `/orcd/scratch/orcd/008/shuzel/envs/rnd_platform_jax_mujoco_warp/bin/python`, beside the
   grid study's `rnd_platform_jax` (which has no MuJoCo and was left as it is).
   - Script `remote_fleet/setup/build_rnd_platform_jax_mujoco_warp_env_mit.sh`, with the same requirement and
     constraint files as Purdue and ACES (copied into `remote_fleet/setup/`, same checksums on MIT). Python 3.12.13
     from conda-forge with `--copy`, made by the conda that built `rnd_platform_jax`
     (`/orcd/software/core/001/pkg/miniforge/26.7.2-0/bin/conda`); the pinned packages as wheels only; JAXAtari at
     commit 035fc46b. Conda's downloads and pip's temporary files went to scratch (the home folder was 90% full).
   - **Job 28985995** on `mit_quicktest` (2 cores, 8 GB, node1601) built it from 12:10:30 to 12:13:35 PT and ended
     `COMPLETED` after 3 min 10 s; log `remote_fleet/setup/build_mit_job_28985995.out`. 79 packages, 36,678 files;
     the scratch space held 149,600 of its 1,000,000 files and 782 of 1,024 GB before.
   - The same script had first been sent to `mit_normal` as **job 28985380** (12:04 PT), which was still waiting
     unexamined among 1,390 waiting jobs of that partition 5 minutes later. The build was sent again to
     `mit_quicktest`; once that build had passed, 28985380 (still waiting, never started) was cancelled by its id
     at 12:14 PT: had it started, it would have found the environment and ended.
   - Package list `remote_fleet/setup/pip_freeze_mit_rnd_platform_jax_mujoco_warp.txt`: **identical to Gautschi's
     and ACES's**, and equal to UVA's `platform_jax_warp` except `pip` 26.2.2 (UVA 26.2.1), `setuptools` 84.2.1
     (UVA 84.0.0) and `PyMuPDF` (absent). JAXAtari's installed-file record has the checksum
     `51c1d1cc358db7f4d803f2e444ebd9bd2cd6c94973fab545d7d27e3fbcafcd4b`.
2. **Import check on the build node** (processor only, imports only), the last step of job 28985995:

   ```text
   python 3.12.13 at /orcd/scratch/orcd/008/shuzel/envs/rnd_platform_jax_mujoco_warp/bin/python
   jax==0.11.0 jaxlib==0.11.0 jax-cuda12-plugin==0.11.0 numpy==2.5.2 mujoco==3.11.0 mujoco-mjx==3.11.0
   warp-lang==1.14.0 flax==0.12.8 optax==0.2.8 jaxatari==0.1.0
   mujoco.mjx Warp implementation: warp installed True | mujoco_warp module loaded True
   imports took 3.4 s
   IMPORT_CHECK_PASSED processor
   ```

3. **Import check on one H100:** **job 28987093** (Advanced pool of `mit_normal_gpu`, `--gpus=h100:1`, 4 cores, 16 GB,
   20-minute limit; node2906, 12:23:46 to 12:23:52 PT, `COMPLETED`, exit 0; job script
   `remote_fleet/setup/import_check_card_job_mit.sh`, output `remote_fleet/setup/import_check_mit_card_job_28987093.out`).
   Imports and the device lists only, after the shared card guard with jax's count; nothing compiled or trained.
   `sacct` gave `ReqTRES` equal to `AllocTRES` (`gres/gpu:h100=1`).

   ```text
   [card guard] 2026-11-13T15:23:48-05:00 job=28987093 node=node2906 need=1 nvidia-smi=1 torch=1 restart=0 action=exit
   [card guard] ok: 1 card(s), as needed
   NVIDIA H100 80GB HBM3, 81559 MiB, 590.48.01
   mujoco.mjx Warp implementation: warp installed True | mujoco_warp module loaded True
   jax default back end: gpu | jax devices: [CudaDevice(id=0)]
   Warp 1.14.0 initialized: CUDA Toolkit 12.9, Driver 13.1; "cuda:0": "NVIDIA H100 80GB HBM3" (79 GiB, sm_90)
   warp CUDA devices: [('cuda:0', 'NVIDIA H100 80GB HBM3', 90)]
   IMPORT_CHECK_PASSED card
   ```

   An earlier run of the same check, **job 28986492** (12:18:14 to 12:18:20 PT, same node), printed the same card,
   jax and warp lines and then stopped at a mistake in the check script's last print (it read warp's driver version
   from an attribute warp 1.14 does not have). The script was corrected and run once more as 28987093. That first run
   let warp create an empty kernel folder `~/.cache/warp/1.14.0` (8 KB) in the MIT home folder; the second run and
   the fleet's trainer put the kernel folder on the node.
4. **Table entry `mit`** (added under the fleet lock at 12:06:54 PT; source
   `remote_fleet_servers_mit_gautschi_entries.json`):
   - `ssh`: `ssh -o BatchMode=yes portal11 ssh -o BatchMode=yes mit`; `check`: `ssh -o BatchMode=yes portal11 ssh -O
     check mit` (the fleet checks the shared connection before every call, so a closed connection is never followed
     by a new login); `failed_call_limit` 1 (one failed call marks the server not ready);
   - `python` as in item 1; `remote_run` `/orcd/scratch/orcd/008/shuzel/rnd_maze/runs/2026-11-12-15-56_cleanrl-ppo-rnd-baseline-antmaze`;
     `code_root` `/orcd/scratch/orcd/008/shuzel/rnd_maze/code`; `unit_seconds` 1800;
   - four pools, each with `card_guard: true` (the job script runs the shared card guard with jax's card count and
     `GUARD_ACTION` requeue, and the status call reads `ReqTRES` against `AllocTRES` of running jobs), one typed
     card `--gpus=<type>:1 --nodes=1 --ntasks=1`, 4 cores (rule: at most 4 per card), `--mem=24G` (the largest
     `MaxRSS` of the last 30 Gilbreth AntMaze jobs was 7.9 GB, and Gilbreth asks 24 GB), `--requeue`,
     `--open-mode=append`, `time_limit` `2-00:00:00` (the partition's and the Advanced pool's maximum), `max_jobs` 4:

     | pool | partition and account | card | nodes left out |
     |---|---|---|---|
     | `advanced-h100` | `mit_normal_gpu`, Advanced (`mit_amf_advanced_gpu`) | `h100` | node3401-c1 |
     | `advanced-h200` | `mit_normal_gpu`, Advanced | `h200` | node3401-c1 |
     | `preemptable-h100` | `mit_preemptable`, free tier | `h100` | the 79 nodes shared with `mit_normal_gpu` (node3401-c1 among them) |
     | `preemptable-h200` | `mit_preemptable`, free tier | `h200` | the same |

   - the account's caps hold across pools: 4 cards at once in the Advanced pool and 4 in `mit_preemptable`, shared
     with every other session of the account, so the two pools of one partition together get at most 4 cards;
   - `hand_over`: `units_at_once` 1, `match` `antmaze`, `pool` `advanced-h100`, `start_within_seconds` 7200. The
     hand-over submits to one pool only; by the card-floor rule the H100 pool comes first, and the session switches
     `hand_over.pool` to an H200 pool when no H100 is free and an H200 is;
   - `budget_note`: not charged; fair share counts processor-seconds only;
   - `ready: false`, with the reason that it waits for the session's freeze and first unit.

## TAMU ACES and NCSA DeltaAI (not built by this session)

1. **ACES is already set up from 2026-11-12** (see `infra_history.md`, 16:18 to 16:45 PT): the environment
   `/scratch/group/p.cis261532.000/envs/rnd_platform_jax_mujoco_warp` exists (checked 2026-11-13 12:07 PT through
   the open connection on portal05; package list `remote_fleet/setup/pip_freeze_aces_rnd_platform_jax_mujoco_warp.txt`),
   and its table entry is kept in `remote_fleet_servers_aces_entry.json`, not merged, to be added with
   `remote_fleet/setup/merge_aces_entry_into_server_table.py`. ACES is x86, reached only through the open connection
   on portal05 (its certificate lasts 49 hours; a new one needs the user). Nothing about ACES was changed today.
   What is still missing there: a check that jax and Warp use an ACES card, which the first worker job does. At
   12:07 PT the account had 70,489.63 SUs left (`myproject`), about 518 H100-hours at 136 SUs per hour, and 8 jobs
   of other sessions in the queue.
2. **DeltaAI would need a new environment built for ARM** (`aarch64`, NVIDIA Grace): the x86 environments of the
   other servers do not run there. What it would take:
   - a new login, which needs the NCSA password and the user's Duo push (`open_ncsa_connection_detached.py deltaai`,
     with the user present);
   - a build with NCSA's Miniforge `conda` by full path, Python 3.12.13 from conda-forge, `--copy`, under
     `/projects/biwj/sliu51/envs/`, with the same requirement and constraint files (`jax` 0.11.0 and its CUDA 12
     plugins have `aarch64` wheels; whether `mujoco` 3.11.0, `warp-lang` 1.14.0 and every pinned `nvidia-*` wheel
     exist for `aarch64` at those versions was not checked);
   - `unset LD_LIBRARY_PATH` in every job script (already in the fleet's job script);
   - the allocation is 100 GPU-hours.
   Nothing was done on DeltaAI.

## What the session does to switch a server on

1. Freeze the code if needed, then `remote_fleet.py --run-dir RUN prepare --server <mit|gautschi>` (sends the job
   scripts and the frozen code copy, and makes the folder above the copy a git repository on the server).
2. **Set `first_unit` by hand, as for Gilbreth.** The automatic format check compares a server's first unit with a
   unit trained on UVA; every finished AntMaze unit of this folder (35 at 12:05 PT) was trained on Gilbreth, so
   `uva_reference` finds none, `hand_over` picks no unit for a new server, and the server would never receive its
   first unit. Gilbreth's entry shows the by-hand form (state `passed`, one unit at a time, read by hand).
3. Set `ready: true` (under the fleet lock). The `hand_over` limit is 1 until the first unit has been read.

## What the first unit on each server must check

1. **Card count.** MIT: the card guard's line in the job log (`[card guard] ... need=1 nvidia-smi=1 torch=1`; the
   "torch" field is counted with jax here), and within a minute of the start
   `sacct -j <id> -X -n -P -o JobID,State,ReqTRES,AllocTRES,NodeList` (the two `gres/gpu=` counts must match).
   Gautschi: `jax devices: [CudaDevice(id=0)]` in the job log and the same `sacct` line.
2. **Card memory against `minimum_card_memory_mb`** (20,000 MB in this folder's descriptions). The H100 cards have
   80 GB (MIT's H100 NVL nodes 94 GB) and the H200 cards 141 GB. Largest readings so far
   (`card_memory_used_mb_largest_reading` in `device/observed/`): about 34,000 MB on an A100 80 GB with 128
   environments per policy, 4,900 MB with 16, 1,700 MB with 4.
3. **Speed against the estimate.** No AntMaze unit has run on an H100 SXM or an H200. References at 16,384
   environments (128 policies x 128 environments per policy): H100 NVL (serval02 card 1, stable physics, speed
   study `profiles/antmaze_warp_training/records_stable_physics_time-step-0-005_frame-skip-10_fault-guard/`)
   2.72 s per iteration; A100 80 GB PCIe on Gilbreth 3.9 to 4.0 s; A30 7.0 s. At 2,048 environments (16 per policy)
   the A100 80 GB took 1.16 s and at 512 (4 per policy) 0.89 s. The first unit on an H100 SXM or an H200 is
   expected at or below the H100 NVL's 2.72 s at 16,384 environments; a slower one is read before more units go.
4. The physics records the session already reads on Gilbreth's first unit: fault respawns, largest torso speed and
   turning rate, contact pool use, constraint rows.

## Found during the setup

1. **The fleet reads the server table before it takes the fleet lock** (`main` in both `remote_fleet.py` files:
   `table = read_json(...)` comes before `with fleet_lock(...)`). A tick that read the table while another process
   held the lock writes its older copy back when it changes the table, and an entry added meanwhile is lost. The two
   entries were therefore added with `flock -n` (only when no tick held the lock) and checked again after later
   ticks: both were present after the ticks of 12:09:37, 12:14:47 and 12:20:00 PT, and the Gilbreth entry was unchanged by the addition.
2. **The id files named for this setup match the pattern UVA's monitor reads.** `common.own_live_jobs` reads every
   `slurm/submitted_jobids_*_*.txt`, so `submitted_jobids_mit_setup.txt` and `submitted_jobids_gautschi_setup.txt`
   are also asked of UVA's scheduler. This is harmless while UVA's job numbers (about 6.9 million) stay far from
   MIT's (about 29 million) and Gautschi's (about 18.5 million), but a later file for a remote server should not
   have a second underscore (the fleet's own `submitted_jobids_<server>.txt` has none).
3. **MIT's preemptable exclusion list is still right.** The `-x` list copied from the tabular study's entry names
   exactly the 79 nodes of `mit_normal_gpu` (read live at 12:06 PT: 79 listed, 79 in the partition, none missing,
   none extra), so preemptable jobs avoid the nodes shared with `mit_normal_gpu`, and PI-owned H100 and H200 nodes
   remain for them.

## Left for the user

1. Nothing on MIT or Gautschi needs the user: switching them on is the session's step (section "What the session
   does to switch a server on").
2. DeltaAI, if it is wanted: a new login with the user's Duo push, then the ARM build described above.
3. ACES: its certificate lasts 49 hours; a renewal needs the user (rule `tamu-aces-cluster-slurm.md`).
4. This helper session created no recurring monitor (it has no scheduling tool): each of its jobs was watched
   directly until it ended, and none of them is left running or waiting (section "Job ids").

## Job ids

Recorded at submission in `slurm/submitted_jobids_mit_setup.txt` and `slurm/submitted_jobids_gautschi_setup.txt`.

| server | job | what | state |
|---|---|---|---|
| MIT | 28985380 | environment build on `mit_normal` | cancelled while waiting (never started), 12:14 PT |
| MIT | 28985995 | environment build on `mit_quicktest`, then the processor import check | completed, 3 min 10 s |
| MIT | 28986492 | import check on one H100 (Advanced pool) | completed, 6 s; stopped at a mistake in the check script's last print, after the device lines |
| MIT | 28987093 | the same check, corrected script | completed, 6 s, passed |
| Gautschi | 18521082 | processor import check on `cpu` under `standby` | completed, 17 s |
